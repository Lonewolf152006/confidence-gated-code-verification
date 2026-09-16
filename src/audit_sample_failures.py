"""
audit_sample_failures.py -- Deep-dive audit of concrete failure instances.

Extracts representative failure samples across each taxonomy category:
  - Prints prompt summary & generated code snippet
  - Prints raw test execution traceback / stderr
  - Audits whether failure is:
      1. Legitimate semantic / algorithmic bug
      2. LLM syntax / truncation artifact
      3. Import / missing symbol in generation
      4. Harness / test execution issue
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "data" / "labels" / "pilot_1.5B_labels.json"


def audit_samples(labels_path: Path, max_per_cat: int = 2):
    if not labels_path.exists():
        print(f"Error: Labels file not found at {labels_path}")
        return

    with open(labels_path, encoding="utf-8") as f:
        samples = json.load(f)

    # Group failed samples by failure category
    by_category = defaultdict(list)
    for s in samples:
        if not s.get("passed", False):
            cat = s.get("label", {}).get("failure_category", "UNKNOWN")
            by_category[cat].append(s)

    print("=" * 85)
    print("  DEEP-DIVE SAMPLE FAILURE AUDIT")
    print(f"  Auditing {len(by_category)} failure categories (up to {max_per_cat} samples each)")
    print("=" * 85)

    for cat in sorted(by_category.keys()):
        cat_samples = by_category[cat]
        print(f"\n#############################################################################")
        print(f"  CATEGORY: {cat} (Total {len(cat_samples)} occurrences)")
        print(f"#############################################################################")

        for i, s in enumerate(cat_samples[:max_per_cat]):
            task_id = s.get("task_id", "")
            idx = s.get("sample_index", 0)
            code = s.get("generated_code", "").strip()
            clean_code = code.split("```")[0].strip()
            label_info = s.get("label", {})
            exc = label_info.get("primary_exception", "None")
            msg = label_info.get("exception_message", "None")
            raw_err = label_info.get("raw_error", "").strip()

            print(f"\n--- [Sample {i+1}/{min(max_per_cat, len(cat_samples))}] Task: {task_id} (Sample Index: {idx}) ---")
            print(f"  Exception: {exc}: {msg}")

            # Show first 10-15 lines of generated code
            code_lines = clean_code.splitlines()[:15]
            print(f"\n  [Generated Code Snippet - first {len(code_lines)} lines]:")
            for cl in code_lines:
                print(f"    | {cl}")
            if len(clean_code.splitlines()) > 15:
                print(f"    | ... ({len(clean_code.splitlines()) - 15} more lines)")

            # Show traceback
            err_lines = raw_err.splitlines()
            print(f"\n  [Execution Traceback - last {min(8, len(err_lines))} lines]:")
            for el in err_lines[-8:]:
                print(f"    ! {el}")

            # Failure root-cause assessment
            root_cause = "UNKNOWN"
            if cat == "SYNTAX_ERROR":
                root_cause = "LLM incomplete token generation, unclosed bracket/string, or syntax corruption"
            elif cat == "IMPORT_ERROR":
                root_cause = "Generated code relies on an unimported module or missing package"
            elif cat == "ASSERTION_ERROR":
                root_cause = "Generated function executed cleanly but returned mathematically / logically incorrect result"
            elif cat == "OTHER_RUNTIME_ERROR":
                if "NameError" in exc:
                    root_cause = "LLM used a variable/function without defining or importing it"
                elif "FileNotFoundError" in exc:
                    root_cause = "LLM failed to create necessary directory or output file before opening"
                else:
                    root_cause = f"Runtime exception: {exc}"
            elif cat in ("ATTRIBUTE_ERROR", "TYPE_ERROR", "KEY_INDEX_ERROR", "VALUE_ERROR"):
                root_cause = f"API usage semantic mismatch: invalid attribute/type/key operation on returned object"

            print(f"\n  [Audit Assessment]: {root_cause}")
            print(f"  {'-'*70}")


if __name__ == "__main__":
    audit_samples(LABELS_PATH, max_per_cat=2)
