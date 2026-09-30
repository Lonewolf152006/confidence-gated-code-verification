"""
signals.py -- Cheap Signal Extraction for Usage-Semantic Hallucination Detection.

Extracts near-zero-cost signals directly from generated completions and logprobs:
  1. Token-level Shannon Entropy (Decision-Point vs Whole-Sequence)
     - Mapped via AST call sites and tokenizer character offset mappings
     - EPR-style localized confidence around API consumption points
  2. Cross-Sample Semantic Usage Pattern Diversity
     - Quantifies distribution entropy across completions of the same prompt
  3. AST Structural & Code Complexity Metrics
     - Tree depth, node counts, API call frequencies

Outputs a feature matrix ready for classifier training and ablation studies.
"""

import os
import sys
import json
import ast
import re
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parse_calls import extract_api_calls, APICall
from src.label_taxonomy import check_code_for_misuse


def compute_token_entropy(topk_step: dict) -> float:
    """
    Compute Shannon entropy over top-K probabilities at a single generation step.
    H(t) = - sum(p * log2(p))
    """
    probs = topk_step.get("probs", [])
    if not probs:
        return 0.0
    p_arr = np.array(probs, dtype=float)
    total = p_arr.sum()
    if total <= 0:
        return 0.0
    p_norm = p_arr / total
    # Avoid log2(0)
    p_safe = np.clip(p_norm, 1e-12, 1.0)
    entropy = -np.sum(p_safe * np.log2(p_safe))
    return float(entropy)


def get_line_starts(text: str) -> list[int]:
    """Compute 0-indexed starting character position for each line."""
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def get_ast_complexity(code: str) -> tuple[int, int]:
    """
    Compute AST depth and total node count for a code snippet.
    Returns (max_depth, node_count).
    """
    clean_code = code.split("```")[0]
    try:
        tree = ast.parse(clean_code)
    except Exception:
        try:
            tree = ast.parse(f"def _wrap():\n{clean_code}")
        except Exception:
            return 1, 1

    node_count = 0
    max_depth = 0

    def walk_depth(node, depth):
        nonlocal node_count, max_depth
        node_count += 1
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            walk_depth(child, depth + 1)

    walk_depth(tree, 1)
    return max_depth, node_count


def compute_task_usage_diversity(samples_by_task: dict[str, list[dict]]) -> dict[str, float]:
    """
    For each task, compute the Shannon entropy of API consumption patterns
    across the generated samples.
    """
    diversity_map = {}

    for task_id, samples in samples_by_task.items():
        pattern_counts = Counter()

        for s in samples:
            raw_code = s.get("generated_code", "").split("```")[0]
            try:
                calls = extract_api_calls(f"def _wrap():\n{raw_code}")
            except Exception:
                try:
                    calls = extract_api_calls(raw_code)
                except Exception:
                    calls = []

            for c in calls:
                # Group by call and its consumption kind
                kind = c.consumed_by or "[none]"
                pattern_counts[f"{c.full_expr}->{kind}"] += 1
                for u in c.usage_points:
                    pattern_counts[f"{c.full_expr}->deferred:{u.kind}"] += 1

        total_patterns = sum(pattern_counts.values())
        if total_patterns <= 1:
            diversity_map[task_id] = 0.0
        else:
            probs = np.array(list(pattern_counts.values()), dtype=float) / total_patterns
            diversity_entropy = -np.sum(probs * np.log2(probs + 1e-12))
            diversity_map[task_id] = float(diversity_entropy)

    return diversity_map


def extract_sample_features(
    sample: dict,
    task_diversity: float,
) -> dict:
    """
    Extract full signal vector for one generated code sample.
    """
    code = sample.get("generated_code", "")
    clean_code = code.split("```")[0]
    offsets = sample.get("offset_mapping", [])
    logprobs = sample.get("topk_logprobs", [])

    # 1. Whole-sequence token entropies
    token_entropies = [compute_token_entropy(step) for step in logprobs]
    n_tokens = len(token_entropies)

    if n_tokens > 0:
        mean_seq_entropy = float(np.mean(token_entropies))
        max_seq_entropy = float(np.max(token_entropies))
        std_seq_entropy = float(np.std(token_entropies))
        pct_high_entropy = float(sum(1 for h in token_entropies if h > 2.0) / n_tokens)
    else:
        mean_seq_entropy = 0.0
        max_seq_entropy = 0.0
        std_seq_entropy = 0.0
        pct_high_entropy = 0.0

    # 2. Decision-Point Token Alignment
    line_starts = get_line_starts(clean_code)
    try:
        calls = extract_api_calls(f"def _wrap():\n{clean_code}")
    except Exception:
        try:
            calls = extract_api_calls(clean_code)
        except Exception:
            calls = []

    decision_token_indices = set()

    for c in calls:
        # Call end character offset (adjusting for _wrap line if needed)
        end_line = getattr(c, "end_lineno", c.lineno)
        # Adjust 1-indexed line number for wrapper offset if present
        target_line_idx = end_line - 2 if end_line - 1 >= len(line_starts) else end_line - 1
        if 0 <= target_line_idx < len(line_starts):
            call_end_char = line_starts[target_line_idx] + getattr(c, "end_col_offset", c.col_offset)

            # Window of tokens right after the call (next ~25 chars / immediate chaining)
            for i, (tok_s, tok_e) in enumerate(offsets):
                if 0 <= tok_s - call_end_char <= 25:
                    decision_token_indices.add(i)

        # Also mark deferred usage points
        for u in c.usage_points:
            u_line_idx = u.lineno - 2 if u.lineno - 1 >= len(line_starts) else u.lineno - 1
            if 0 <= u_line_idx < len(line_starts):
                u_start_char = line_starts[u_line_idx] + u.col_offset
                for i, (tok_s, tok_e) in enumerate(offsets):
                    if 0 <= tok_s - u_start_char <= max(20, len(u.source)):
                        decision_token_indices.add(i)

    # 3. Decision-point entropy metrics
    if decision_token_indices and n_tokens > 0:
        valid_indices = [idx for idx in decision_token_indices if idx < n_tokens]
        if valid_indices:
            decision_entropies = [token_entropies[idx] for idx in valid_indices]
            mean_decision_entropy = float(np.mean(decision_entropies))
            max_decision_entropy = float(np.max(decision_entropies))
            std_decision_entropy = float(np.std(decision_entropies))
            decision_token_ratio = float(len(valid_indices) / n_tokens)
        else:
            mean_decision_entropy = mean_seq_entropy
            max_decision_entropy = max_seq_entropy
            std_decision_entropy = std_seq_entropy
            decision_token_ratio = 0.0
    else:
        mean_decision_entropy = mean_seq_entropy
        max_decision_entropy = max_seq_entropy
        std_decision_entropy = std_seq_entropy
        decision_token_ratio = 0.0

    entropy_contrast = float(mean_decision_entropy - mean_seq_entropy)
    entropy_ratio = float((mean_decision_entropy + 1e-4) / (mean_seq_entropy + 1e-4))

    # 4. AST and complexity features
    ast_depth, ast_node_count = get_ast_complexity(clean_code)
    code_chars = len(clean_code)

    # 5. AST Def-Use Violation Indicators
    misuse_records = check_code_for_misuse(clean_code)
    ast_misuse_flag = 1.0 if len(misuse_records) > 0 else 0.0
    ast_misuse_count = float(len(misuse_records))

    # Deferred def-use usages
    total_deferred_usages = sum(len(c.usage_points) for c in calls)
    has_subscript_on_call = 1.0 if any(
        c.consumed_by == "[subscript]" or any(u.kind == "[subscript]" for u in c.usage_points)
        for c in calls
    ) or any(m.get("misuse_type") in ("requests_subscript_without_json", "string_key_on_dict_key_iteration") for m in misuse_records) else 0.0

    has_invalid_type_call = 1.0 if any(
        m.get("misuse_type") in ("hexdigest_on_bytes", "reshape_on_dict_list", "pandas_obsolete_sort", "numpy_abs_vector_norm_misuse", "dictwriter_non_dict_row")
        for m in misuse_records
    ) else 0.0

    # 6. Ground truth labels
    passed = sample.get("passed", False)
    label_info = sample.get("label", {})
    failure_category = label_info.get("failure_category", "NONE" if passed else "UNKNOWN")
    is_intent_misuse = bool(label_info.get("is_intent_misuse", False) or failure_category == "USAGE_SEMANTIC_MISUSE")

    return {
        # Identifiers
        "task_id": sample.get("task_id", ""),
        "sample_index": sample.get("sample_index", 0),
        "model": sample.get("model", ""),
        # Signal Features
        "mean_decision_entropy": round(mean_decision_entropy, 5),
        "max_decision_entropy": round(max_decision_entropy, 5),
        "std_decision_entropy": round(std_decision_entropy, 5),
        "mean_seq_entropy": round(mean_seq_entropy, 5),
        "max_seq_entropy": round(max_seq_entropy, 5),
        "std_seq_entropy": round(std_seq_entropy, 5),
        "entropy_contrast": round(entropy_contrast, 5),
        "entropy_ratio": round(entropy_ratio, 5),
        "pct_high_entropy": round(pct_high_entropy, 5),
        "decision_token_ratio": round(decision_token_ratio, 5),
        "task_usage_diversity": round(task_diversity, 5),
        "n_api_calls": len(calls),
        "ast_depth": ast_depth,
        "ast_node_count": ast_node_count,
        "code_chars": code_chars,
        "n_tokens": n_tokens,
        # AST Def-Use Violation Indicators
        "ast_misuse_flag": ast_misuse_flag,
        "ast_misuse_count": ast_misuse_count,
        "def_use_deferred_count": float(total_deferred_usages),
        "has_subscript_on_call": has_subscript_on_call,
        "has_invalid_type_call": has_invalid_type_call,
        # Ground Truth Targets
        "passed": passed,
        "target_fail": 1 if not passed else 0,
        "target_intent_misuse": 1 if is_intent_misuse else 0,
        "failure_category": failure_category,
    }


def extract_all_signals(input_path: Path, output_path: Path):
    """Process all labeled samples into a tabular feature dataset."""
    print(f"\n{'=' * 60}")
    print(f"  Cheap Signal Feature Extraction")
    print(f"{'=' * 60}")
    print(f"  Input labels: {input_path}")
    print(f"  Output path:  {output_path}")

    with open(input_path, encoding="utf-8") as f:
        samples = json.load(f)

    # Group by task for cross-sample diversity computation
    samples_by_task = {}
    for s in samples:
        t_id = s.get("task_id", "")
        samples_by_task.setdefault(t_id, []).append(s)

    print(f"  Total samples: {len(samples)} across {len(samples_by_task)} tasks")
    print(f"  Computing cross-sample usage pattern diversity...")
    task_diversity_map = compute_task_usage_diversity(samples_by_task)

    feature_records = []
    for s in tqdm(samples, desc="Extracting features", unit="sample"):
        t_id = s.get("task_id", "")
        t_div = task_diversity_map.get(t_id, 0.0)
        rec = extract_sample_features(s, t_div)
        feature_records.append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feature_records, f, indent=2)

    # Print summary table
    print(f"\n  Extracted {len(feature_records)} feature vectors.")
    print(f"  Feature columns ({len(feature_records[0]) - 5} features):")
    feature_keys = [k for k in feature_records[0].keys() if not k.startswith("target_") and k not in ("task_id", "sample_index", "model", "passed", "failure_category")]
    for k in feature_keys:
        vals = [r[k] for r in feature_records]
        print(f"    - {k:25s} mean: {np.mean(vals):.4f}  (min: {np.min(vals):.4f}, max: {np.max(vals):.4f})")

    print(f"\n  Features saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract cheap signal features from labeled completions")
    parser.add_argument(
        "--input",
        type=str,
        default="data/labels/pilot_1.5B_labels.json",
        help="Path to labeled dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/labels/pilot_1.5B_features.json",
        help="Output path for feature matrix JSON",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    extract_all_signals(input_path, output_path)


if __name__ == "__main__":
    main()
