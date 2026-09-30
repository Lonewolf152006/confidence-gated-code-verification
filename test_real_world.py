"""
test_real_world.py -- Standalone CLI Verifier for Real-World Code Snippets.

Tests arbitrary Python code against the Confidence-Gated Verification Cascade:
  1. Extracts API call sites & def-use variable chains (AST analysis)
  2. Runs static baseline tools (Pylint and Mypy)
  3. Evaluates structural signals & the trained distilled specialist model
  4. Triages code into ACCEPT, ESCALATE, or REJECT with detailed diagnosis

Usage:
  python test_real_world.py --preset requests_misuse
  python test_real_world.py --preset clean_requests
  python test_real_world.py --file path/to/script.py
  python test_real_world.py --code "import requests\nr = requests.get('url')\nx = r['data']"
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parse_calls import extract_api_calls, check_code_for_misuse
from src.baselines import run_pylint, run_mypy
from src.signals import get_ast_complexity
from src.specialist_model import SpecialistVerifier, SIGNAL_COLS

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "specialist_model.pt"

# Curated real-world test cases covering common LLM intent misuses and correct APIs
PRESETS = {
    "requests_misuse": {
        "name": "Requests: Direct Subscript on Response without .json()",
        "description": "LLM forgets that requests.get() returns a Response object, treating it directly as a dict.",
        "code": '''\
import requests

def fetch_user_profile(user_id: int):
    url = f"https://api.github.com/users/{user_id}"
    response = requests.get(url)
    
    # Intent Misuse: Response object subscripted directly without calling .json()
    user_name = response["name"]
    public_repos = response["public_repos"]
    return {"name": user_name, "repos": public_repos}
''',
        "expected": "REJECT (Intent Misuse: Response object is not subscriptable)",
    },
    "pandas_misuse": {
        "name": "Pandas: Obsolete / Invalid sort() Method Call",
        "description": "LLM calls df.sort() which was deprecated/removed in modern pandas (should be sort_values).",
        "code": '''\
import pandas as pd

def process_sales_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Intent Misuse: df.sort() does not exist in modern pandas
    sorted_df = df.sort("revenue", ascending=False)
    return sorted_df.head(10)
''',
        "expected": "REJECT (Intent Misuse / AttributeError on DataFrame)",
    },
    "numpy_misuse": {
        "name": "NumPy: Vector Magnitude using np.abs() instead of np.linalg.norm()",
        "description": "LLM computes vector magnitude by calling np.abs(), which calculates element-wise absolute value instead of norm.",
        "code": '''\
import numpy as np

def compute_velocity_magnitude(vx: float, vy: float, vz: float) -> float:
    velocity_vector = np.array([vx, vy, vz])
    # Intent Misuse: np.abs() computes element-wise abs, not 3D Euclidean magnitude
    magnitude = np.abs(velocity_vector)
    return float(np.sum(magnitude))
''',
        "expected": "ESCALATE / REJECT (Semantic logic divergence)",
    },
    "os_misuse": {
        "name": "OS: os.listdir Return Treated as Dictionary Mapping",
        "description": "LLM assumes os.listdir() returns a mapping of filenames to attributes instead of a list of strings.",
        "code": '''\
import os

def list_image_files(directory: str):
    contents = os.listdir(directory)
    # Intent Misuse: os.listdir returns list[str], does not have .items()
    images = []
    for filename, file_info in contents.items():
        if filename.endswith(".png"):
            images.append(filename)
    return images
''',
        "expected": "REJECT (Intent Misuse: list object has no attribute 'items')",
    },
    "pandas_inplace_misuse": {
        "name": "Pandas: None Assignment via Inplace Mutation (df = df.dropna(inplace=True))",
        "description": "LLM assigns the return value of an inplace pandas mutation back to the DataFrame variable, setting it to None.",
        "code": '''\
import pandas as pd

def clean_dataframe(csv_path: str):
    df = pd.read_csv(csv_path)
    # Intent Misuse: df.dropna(inplace=True) returns None, wiping out df
    df = df.dropna(subset=["email"], inplace=True)
    return df.head()
''',
        "expected": "REJECT (Intent Misuse: None assignment on inplace call)",
    },
    "requests_text_misuse": {
        "name": "Requests: response.text() Called as Method Instead of Property",
        "description": "LLM treats response.text as a callable function instead of a string property, causing TypeError.",
        "code": '''\
import requests

def get_page_summary(url: str):
    response = requests.get(url)
    # Intent Misuse: .text is a property, not a callable function
    raw_html = response.text()
    return len(raw_html)
''',
        "expected": "REJECT (Intent Misuse: 'str' object is not callable)",
    },
    "pandas_append_misuse": {
        "name": "Pandas: Obsolete df.append() Method Call (Removed in Pandas 2.0)",
        "description": "LLM calls df.append() which was completely removed in Pandas 2.0 (should use pd.concat).",
        "code": '''\
import pandas as pd

def combine_tables(df1: pd.DataFrame, df2: pd.DataFrame):
    # Intent Misuse: df.append removed in Pandas 2.0+
    combined = df1.append(df2, ignore_index=True)
    return combined
''',
        "expected": "REJECT (Intent Misuse: DataFrame object has no attribute 'append')",
    },
    "hashlib_misuse": {
        "name": "Hashlib: Calling .hexdigest() on Bytes Returned by .digest()",
        "description": "LLM calls .hexdigest() on bytes instead of calling directly on the hashlib object.",
        "code": '''\
import hashlib

def generate_checksum(data: str):
    h = hashlib.sha256(data.encode('utf-8'))
    # Intent Misuse: .digest() returns bytes, which has no .hexdigest() method
    raw_digest = h.digest()
    return raw_digest.hexdigest()
''',
        "expected": "REJECT (Intent Misuse: 'bytes' object has no attribute 'hexdigest')",
    },
    "clean_requests": {
        "name": "Requests: Correct API Usage with .json() and Status Check",
        "description": "Proper usage of requests library with .raise_for_status() and .json().",
        "code": '''\
import requests

def fetch_repository_info(repo_name: str):
    url = f"https://api.github.com/repos/{repo_name}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    return {
        "full_name": data.get("full_name"),
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0)
    }
''',
        "expected": "ACCEPT (Confidently Safe)",
    },
    "clean_pandas": {
        "name": "Pandas: Correct sort_values() and dropna() Pipeline",
        "description": "Proper modern pandas DataFrame manipulation without invalid method calls.",
        "code": '''\
import pandas as pd

def clean_and_rank_scores(csv_path: str):
    df = pd.read_csv(csv_path)
    cleaned = df.dropna(subset=["score"])
    ranked = cleaned.sort_values(by="score", ascending=False)
    return ranked[["student_id", "score"]].head(5)
''',
        "expected": "ACCEPT (Confidently Safe)",
    }
}


ENTROPY_DIVERSITY_COLS = [
    "mean_decision_entropy", "max_decision_entropy", "std_decision_entropy",
    "mean_seq_entropy", "max_seq_entropy", "std_seq_entropy",
    "entropy_contrast", "entropy_ratio", "pct_high_entropy",
    "decision_token_ratio", "task_usage_diversity"
]


def analyze_real_world_code(
    code: str,
    tau_accept: float = 0.30,
    tau_reject: float = 0.75,
    verifier: SpecialistVerifier = None,
    mode: str = "standalone",
    generation_sample: dict = None,
) -> dict:
    """
    Execute triage across static baselines, AST def-use tracking,
    and the distilled specialist model.

    Explicit Modes:
      - 'live': Full 16 features from active generation session with real logprobs.
      - 'standalone': AST + TF-IDF only; the 11 entropy/diversity features are
        explicitly zeroed and flagged (not silently degraded).
    """
    # 1. AST Call extraction & def-use tracking
    api_calls = extract_api_calls(code)
    ast_misuses = check_code_for_misuse(code)
    ast_depth, ast_nodes = get_ast_complexity(code)

    # 2. Static baseline checks (Pylint and Mypy)
    pylint_hits = run_pylint(code)
    mypy_hits = run_mypy(code)

    # 3. Feature extraction for specialist model
    tabular_vec = np.zeros((1, len(SIGNAL_COLS)), dtype=np.float32)
    feature_dict = {}

    if mode == "live" and generation_sample is not None:
        # Full live mode with real generation logprobs
        from src.signals import extract_sample_features
        task_div = generation_sample.get("task_usage_diversity", 0.0)
        feats = extract_sample_features(generation_sample, task_diversity=task_div)
        for i, col in enumerate(SIGNAL_COLS):
            val = float(feats.get(col, 0.0))
            tabular_vec[0, i] = val
            feature_dict[col] = val
    else:
        # Standalone mode: 11 entropy/diversity features are explicitly ZEROED OUT
        mode = "standalone"
        for col in ENTROPY_DIVERSITY_COLS:
            feature_dict[col] = 0.0

        # Compute structural AST & code length features
        feature_dict["n_api_calls"] = float(len(api_calls))
        feature_dict["ast_depth"] = float(ast_depth)
        feature_dict["ast_node_count"] = float(ast_nodes)
        feature_dict["code_chars"] = float(len(code))
        feature_dict["n_tokens"] = float(len(code.split()))

        # Compute AST def-use violation indicators
        feature_dict["ast_misuse_flag"] = 1.0 if len(ast_misuses) > 0 else 0.0
        feature_dict["ast_misuse_count"] = float(len(ast_misuses))
        feature_dict["def_use_deferred_count"] = float(sum(len(c.usage_points) for c in api_calls))
        feature_dict["has_subscript_on_call"] = 1.0 if any(
            c.consumed_by == "[subscript]" or any(u.kind == "[subscript]" for u in c.usage_points)
            for c in api_calls
        ) or any(m.get("misuse_type") in ("requests_subscript_without_json", "string_key_on_dict_key_iteration") for m in ast_misuses) else 0.0
        feature_dict["has_invalid_type_call"] = 1.0 if any(
            m.get("misuse_type") in ("hexdigest_on_bytes", "reshape_on_dict_list", "pandas_obsolete_sort", "numpy_abs_vector_norm_misuse", "dictwriter_non_dict_row")
            for m in ast_misuses
        ) else 0.0

        for i, col in enumerate(SIGNAL_COLS):
            tabular_vec[0, i] = feature_dict.get(col, 0.0)

    # 4. Specialist model inference
    specialist_prob = 0.5
    if verifier is not None and verifier.model is not None:
        try:
            probs = verifier.predict_proba(tabular_vec, [code])
            specialist_prob = float(probs[0])
        except Exception:
            specialist_prob = 0.5

    # 5. Confidence-Gated Verification Cascade Decision
    # Immediate AST misuse detection overrides
    if len(ast_misuses) > 0:
        verdict = "REJECT"
        confidence_level = "HIGH (AST Rule Confirmation)"
        primary_reason = f"Usage-Semantic Intent Misuse: {ast_misuses[0]['description']} (Line {ast_misuses[0]['line']})"
    elif len(pylint_hits) > 0 and any("E0001" in h or "syntax-error" in h.lower() for h in pylint_hits):
        verdict = "REJECT"
        confidence_level = "HIGH (Syntax Failure)"
        primary_reason = f"Syntax Error: {pylint_hits[0]}"
    elif specialist_prob >= tau_reject:
        verdict = "REJECT"
        confidence_level = f"HIGH ({specialist_prob:.1%} Misuse Risk)"
        primary_reason = f"Specialist verifier identified high probability of semantic misuse ({specialist_prob:.1%})"
    elif specialist_prob <= tau_accept:
        verdict = "ACCEPT"
        confidence_level = f"HIGH ({1.0 - specialist_prob:.1%} Safe Confidence)"
        primary_reason = "Code conforms to expected API usage conventions with low uncertainty"
    else:
        verdict = "ESCALATE"
        confidence_level = f"BORDERLINE ({specialist_prob:.1%} Uncertainty)"
        primary_reason = f"Uncertainty score within escalation window [{tau_accept:.2f}, {tau_reject:.2f}]. Triaged for deep verification."

    return {
        "mode": mode,
        "zeroed_features_count": len(ENTROPY_DIVERSITY_COLS) if mode == "standalone" else 0,
        "verdict": verdict,
        "confidence_level": confidence_level,
        "primary_reason": primary_reason,
        "specialist_prob": specialist_prob,
        "api_calls": [
            {
                "call": c.full_expr,
                "line": c.lineno,
                "consumed_by": c.consumed_by,
                "assigned_to": c.assigned_to,
            }
            for c in api_calls
        ],
        "ast_misuses": ast_misuses,
        "pylint_hits": pylint_hits,
        "mypy_hits": mypy_hits,
        "static_caught": len(pylint_hits) > 0 or len(mypy_hits) > 0,
        "ast_depth": ast_depth,
        "ast_nodes": ast_nodes,
        "features": feature_dict,
    }


def print_report(code: str, result: dict, name: str = "Code Snippet"):
    """Render a clean, human-readable terminal verification report."""
    verdict = result["verdict"]
    mode = result.get("mode", "standalone")
    badge = {
        "ACCEPT": "[ ACCEPTED - CONFIDENTLY SAFE ]",
        "ESCALATE": "[ ESCALATE - BORDERLINE / SPECIALIST NEEDED ]",
        "REJECT": "[ REJECTED - INTENT MISUSE DETECTED ]",
    }.get(verdict, f"[ {verdict} ]")

    print("\n" + "=" * 80)
    print(f"  VERIFICATION REPORT: {name}")
    print("=" * 80)

    if mode == "standalone":
        print("\n" + "!" * 80)
        print("  [WARNING] RUNNING IN STANDALONE MODE (No LLM generation logprobs provided)")
        print("  11 of 16 features (Shannon entropy & cross-sample diversity) are zeroed out.")
        print("  Operating on AST Def-Use rules + code TF-IDF syntax embeddings only.")
        print("  Accuracy is expected to be lower, since it is missing most of the model's signal.")
        print("!" * 80)
    else:
        print("\n  [MODE: LIVE GENERATION SESSION]")
        print("  Full 16-feature vector evaluated (Decision-point entropy & diversity active).")

    print(f"\n  >>> VERDICT: {badge}")
    print(f"  Confidence:     {result['confidence_level']}")
    print(f"  Primary Reason: {result['primary_reason']}")
    print(f"  Misuse Score:   {result['specialist_prob']:.3f} (Thresholds: Accept <= 0.30, Reject >= 0.75)")

    print("\n--- 1. AST Call Site & Def-Use Analysis ---")
    if result["api_calls"]:
        for c in result["api_calls"]:
            assign_str = f"assigned to `{c['assigned_to']}`" if c['assigned_to'] else "direct expression"
            consume_str = f"consumed by `{c['consumed_by']}`" if c['consumed_by'] else "unconsumed"
            print(f"  • Line {c['line']:2d}: `{c['call']}` ({assign_str}, {consume_str})")
    else:
        print("  • No external library API calls detected in AST.")

    if result["ast_misuses"]:
        print("\n  [!] AST Misuse Checkers Flagged:")
        for m in result["ast_misuses"]:
            print(f"      Line {m['line']}: {m['description']}")

    print("\n--- 2. Static Analysis Baseline Performance ---")
    if result["static_caught"]:
        print("  • Static tools flagged issues:")
        for h in result["pylint_hits"]:
            print(f"    [Pylint] {h.split(':')[-1].strip()[:70]}")
        for h in result["mypy_hits"]:
            print(f"    [Mypy]   {h.split(':')[-1].strip()[:70]}")
    else:
        print("  • Pylint & Mypy: 0 errors detected (Clean static pass).")
        if verdict == "REJECT":
            print("    --> CONFIRMATION: Static tools completely missed this Intent Misuse!")

    print("\n--- 3. Code Tested ---")
    for line_no, line in enumerate(code.strip().splitlines(), 1):
        print(f"  {line_no:2d} | {line}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test arbitrary real-world Python code with the Verification Cascade")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Choose a pre-configured realistic test case")
    parser.add_argument("--file", type=str, help="Path to a local .py file to inspect")
    parser.add_argument("--code", type=str, help="Direct code string to inspect")
    parser.add_argument("--mode", choices=["standalone", "live"], default="standalone",
                        help="Mode: 'live' (16 features with logprobs) or 'standalone' (AST + TF-IDF only, 11 features zeroed)")
    parser.add_argument("--gen-file", type=str, help="Path to a generation JSON file to inspect in live mode")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index in generation JSON file (default: 0)")
    parser.add_argument("--list-presets", action="store_true", help="List all available pre-configured test presets")
    parser.add_argument("--tau-accept", type=float, default=0.30, help="Confidence router acceptance threshold (default: 0.30)")
    parser.add_argument("--tau-reject", type=float, default=0.75, help="Confidence router rejection threshold (default: 0.75)")
    args = parser.parse_args()

    if args.list_presets:
        print("\nAvailable Real-World Test Presets:")
        for k, v in PRESETS.items():
            print(f"  • {k:16s}: {v['name']}")
            print(f"    Description : {v['description']}")
            print(f"    Expected    : {v['expected']}\n")
        return

    # Load specialist model if available
    verifier = None
    if MODEL_PATH.exists():
        try:
            verifier = SpecialistVerifier(MODEL_PATH)
        except Exception as e:
            print(f"[WARN] Could not load specialist checkpoint: {e}")

    gen_sample = None
    if args.gen_file:
        gf = Path(args.gen_file)
        if not gf.exists():
            print(f"Error: Generation file not found: {gf}")
            sys.exit(1)
        import json
        with open(gf, encoding="utf-8") as f:
            g_data = json.load(f)
        if isinstance(g_data, list) and len(g_data) > args.sample_idx:
            gen_sample = g_data[args.sample_idx]
            code = gen_sample.get("generated_code", "")
            name = f"Live Generation: {gf.stem} (Sample #{args.sample_idx})"
            args.mode = "live"
        else:
            print(f"Error: Invalid sample index {args.sample_idx} in {gf}")
            sys.exit(1)
    elif args.preset:
        preset_info = PRESETS[args.preset]
        code = preset_info["code"]
        name = f"Preset: {preset_info['name']}"
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        code = file_path.read_text(encoding="utf-8")
        name = f"File: {file_path.name}"
    elif args.code:
        code = args.code
        name = "Custom Code Snippet"
    else:
        print("No input provided. Defaulting to preset: 'requests_misuse'. (Use --help or --list-presets for options)")
        preset_info = PRESETS["requests_misuse"]
        code = preset_info["code"]
        name = f"Preset: {preset_info['name']}"

    result = analyze_real_world_code(
        code,
        tau_accept=args.tau_accept,
        tau_reject=args.tau_reject,
        verifier=verifier,
        mode=args.mode,
        generation_sample=gen_sample,
    )
    print_report(code, result, name=name)


if __name__ == "__main__":
    main()
