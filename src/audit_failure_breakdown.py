"""
audit_failure_breakdown.py -- Comprehensive audit of pilot execution labels.

Audits the 100 pilot samples in data/labels/pilot_1.5B_labels.json:
  - Aggregates overall pass/fail/timeout counts
  - Quantifies exact breakdown across failure taxonomy categories
  - Catalogs primary exception types and messages
  - Breaks down performance per task_id
  - Identifies environmental vs algorithmic vs syntax failures
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "data" / "labels" / "pilot_1.5B_labels.json"


def run_audit(labels_path: Path):
    if not labels_path.exists():
        print(f"Error: Labels file not found at {labels_path}")
        return

    with open(labels_path, encoding="utf-8") as f:
        samples = json.load(f)

    total_samples = len(samples)
    passed_samples = sum(1 for s in samples if s.get("passed", False))
    failed_samples = total_samples - passed_samples
    timed_out_samples = sum(1 for s in samples if s.get("timed_out", False))

    print("=" * 80)
    print("  PILOT EXECUTION & FAILURE BREAKDOWN AUDIT")
    print("  Dataset: " + str(labels_path.name))
    print(f"  Total Evaluated: {total_samples} samples across {len(set(s['task_id'] for s in samples))} unique tasks")
    print("=" * 80)

    intent_misuse_samples = sum(1 for s in samples if s.get("label", {}).get("is_intent_misuse", False))

    print("\n--- 1. High-Level Outcome Summary ---")
    print(f"  Passed:                {passed_samples:3d} / {total_samples} ({passed_samples / total_samples * 100:.1f}%)")
    print(f"  Failed:                {failed_samples:3d} / {total_samples} ({failed_samples / total_samples * 100:.1f}%)")
    print(f"  Timed Out:             {timed_out_samples:3d} / {total_samples} ({timed_out_samples / total_samples * 100:.1f}%)")
    print(f"  Intent Misuse Flagged: {intent_misuse_samples:3d} / {total_samples} ({intent_misuse_samples / total_samples * 100:.1f}%)")

    # Failure category distribution
    category_counts = Counter()
    for s in samples:
        cat = s.get("label", {}).get("failure_category", "UNKNOWN")
        category_counts[cat] += 1

    print("\n--- 2. Fine-Grained Failure Taxonomy Distribution ---")
    print(f"  {'Category':26s} | {'Count':6s} | {'Pct %':6s} | Notes")
    print(f"  {'-'*26}-+-{'-'*6}-+-{'-'*6}-+---------------------------------------------")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / total_samples * 100
        notes = ""
        if cat == "NONE":
            notes = "Passed all unit test assertions cleanly"
        elif cat == "USAGE_SEMANTIC_MISUSE":
            notes = "Target Intent Misuse: valid API name, wrong return type/usage assumption"
        elif cat == "OTHER_RUNTIME_ERROR":
            notes = "Runtime exceptions (NameError, FileNotFoundError, etc.)"
        elif cat == "ASSERTION_ERROR":
            notes = "Test assertion failed (logic / value mismatch)"
        elif cat == "SYNTAX_ERROR":
            notes = "Incomplete generation / syntax token errors"
        elif cat == "IMPORT_ERROR":
            notes = "Missing dependency or invalid import"
        elif cat == "ATTRIBUTE_ERROR":
            notes = "Method or attribute does not exist on object"
        elif cat == "KEY_INDEX_ERROR":
            notes = "KeyError or IndexError in container access"
        elif cat == "VALUE_ERROR":
            notes = "Incompatible value passed to function"
        elif cat == "TYPE_ERROR":
            notes = "Type mismatch operation"
        print(f"  {cat:26s} | {count:6d} | {pct:5.1f}% | {notes}")

    # Exception types catalog
    exception_counts = Counter()
    for s in samples:
        if not s.get("passed", False):
            exc = s.get("label", {}).get("primary_exception") or "NoneRecorded"
            exception_counts[exc] += 1

    print("\n--- 3. Primary Exception Breakdown (Failed Samples Only) ---")
    print(f"  {'Exception Type':26s} | {'Occurrences':12s} | {'% of Failures':12s}")
    print(f"  {'-'*26}-+-{'-'*12}-+-{'-'*12}")
    for exc, count in sorted(exception_counts.items(), key=lambda x: -x[1]):
        pct_fail = count / failed_samples * 100 if failed_samples else 0
        print(f"  {exc:26s} | {count:12d} | {pct_fail:10.1f}%")

    # Per-task breakdown
    task_groups = defaultdict(list)
    for s in samples:
        task_groups[s["task_id"]].append(s)

    print("\n--- 4. Per-Task Outcome Breakdown (10 completions per task) ---")
    print(f"  {'Task ID':20s} | {'Pass':5s} | {'Fail':5s} | {'Pass Rate':10s} | Top Failure Category")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+---------------------")
    for t_id in sorted(task_groups.keys()):
        t_samples = task_groups[t_id]
        t_pass = sum(1 for s in t_samples if s.get("passed", False))
        t_fail = len(t_samples) - t_pass
        t_cats = Counter(s.get("label", {}).get("failure_category") for s in t_samples if not s.get("passed", False))
        top_cat = t_cats.most_common(1)[0][0] if t_cats else "None (100% pass)"
        print(f"  {t_id:20s} | {t_pass:5d} | {t_fail:5d} | {t_pass / len(t_samples) * 100:8.1f}% | {top_cat}")

    # Execution time audit
    exec_times = [s.get("execution_time", 0.0) for s in samples]
    print("\n--- 5. Test Execution Time Profile ---")
    print(f"  Mean:   {np.mean(exec_times):.4f}s")
    print(f"  Median: {np.median(exec_times):.4f}s")
    print(f"  Min:    {np.min(exec_times):.4f}s")
    print(f"  Max:    {np.max(exec_times):.4f}s")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit execution failure breakdown")
    parser.add_argument("--input", type=str, default=str(LABELS_PATH),
                        help="Path to labeled JSON dataset")
    args = parser.parse_args()
    run_audit(Path(args.input))

