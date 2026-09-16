"""
filter_tasks.py -- Download BigCodeBench from HuggingFace Hub and filter
to tasks using the target libraries (pandas, requests, os, json).

This script handles everything automatically:
  1. Downloads BigCodeBench-Complete from HuggingFace (cached after first run)
  2. Filters tasks by library annotations
  3. Saves the filtered subset to data/raw_tasks/filtered_tasks.json
  4. Prints statistics about the filtered dataset

No manual download needed -- just run this script.

Usage:
    python src/filter_tasks.py                  # full filtering
    python src/filter_tasks.py --pilot 10       # select 10 pilot tasks
    python src/filter_tasks.py --stats-only     # just print dataset stats
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Target libraries we want to filter for (Intent Misuse is most
# likely with these well-known, frequently-used APIs)
TARGET_LIBRARIES = {"pandas", "requests", "os", "json"}

# Additional libraries that are interesting but secondary
SECONDARY_LIBRARIES = {"numpy", "os.path", "urllib", "http"}

# Output paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_TASKS_DIR = PROJECT_ROOT / "data" / "raw_tasks"
PILOT_DIR = PROJECT_ROOT / "data" / "pilot"


def load_bigcodebench():
    """Load BigCodeBench-Complete from HuggingFace Hub."""
    from datasets import load_dataset

    print("Loading BigCodeBench from HuggingFace Hub...")
    print("  (First run downloads ~50MB; subsequent runs use cache)\n")

    # BigCodeBench has 'complete' and 'instruct' splits
    # We MUST use 'complete' -- Instruct breaks automatic testing
    # because the model can choose its own function signature
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")

    print(f"  Loaded {len(ds)} total tasks")
    print(f"  Columns: {ds.column_names}\n")

    return ds


def extract_libraries_from_code(code: str) -> set[str]:
    """
    Extract imported library names from code using regex.
    Handles: import X, from X import Y, import X as Z
    """
    libs = set()

    # Match 'import X' and 'import X as Y'
    for match in re.finditer(r'^import\s+([\w.]+)', code, re.MULTILINE):
        lib = match.group(1).split('.')[0]
        libs.add(lib)

    # Match 'from X import Y'
    for match in re.finditer(r'^from\s+([\w.]+)\s+import', code, re.MULTILINE):
        lib = match.group(1).split('.')[0]
        libs.add(lib)

    return libs


def extract_libraries_from_metadata(task: dict) -> set[str]:
    """
    Extract libraries from the task's metadata/libs field if available.
    BigCodeBench tasks have library annotations in their metadata.
    """
    libs = set()

    # Check various possible field names for library info
    for field in ['libs', 'libraries', 'packages']:
        if field in task and task[field]:
            val = task[field]
            if isinstance(val, list):
                libs.update(val)
            elif isinstance(val, str):
                libs.update(val.split(','))

    return libs


def filter_tasks(ds, target_libs: set[str], min_libs: int = 1) -> list[dict]:
    """
    Filter BigCodeBench tasks to those using at least `min_libs` of the
    target libraries.

    Returns a list of task dicts with standardized fields.
    """
    filtered = []

    for task in ds:
        # Extract libraries from multiple sources
        libs_from_code = set()
        libs_from_meta = set()

        # Try to get libs from the prompt/code
        for field in ['complete_prompt', 'instruct_prompt', 'canonical_solution',
                      'prompt', 'code', 'task_func']:
            if field in task and task[field]:
                libs_from_code.update(extract_libraries_from_code(task[field]))

        # Try to get libs from metadata
        libs_from_meta = extract_libraries_from_metadata(task)

        all_libs = libs_from_code | libs_from_meta
        matching_libs = all_libs & target_libs

        if len(matching_libs) >= min_libs:
            # Build standardized task dict
            task_dict = {
                "task_id": task.get("task_id", ""),
                "complete_prompt": task.get("complete_prompt", ""),
                "instruct_prompt": task.get("instruct_prompt", ""),
                "canonical_solution": task.get("canonical_solution", ""),
                "test": task.get("test", ""),
                "entry_point": task.get("entry_point", ""),
                "libraries_detected": sorted(all_libs),
                "target_libraries_matched": sorted(matching_libs),
                "n_target_matches": len(matching_libs),
            }

            # Include libs field if present
            if "libs" in task:
                task_dict["libs_metadata"] = task["libs"]

            filtered.append(task_dict)

    return filtered


def select_pilot_tasks(tasks: list[dict], n: int = 10, seed: int = 42) -> list[dict]:
    """
    Select a diverse pilot subset, trying to cover different libraries
    and different numbers of target-library matches.
    """
    import random
    rng = random.Random(seed)

    # Group tasks by their primary target library
    by_lib: dict[str, list[dict]] = {}
    for t in tasks:
        primary_lib = t["target_libraries_matched"][0]
        by_lib.setdefault(primary_lib, []).append(t)

    # Select roughly equal numbers from each library
    pilot = []
    per_lib = max(1, n // len(by_lib))
    remainder = n - per_lib * len(by_lib)

    for lib in sorted(by_lib.keys()):
        pool = by_lib[lib]
        rng.shuffle(pool)
        take = per_lib + (1 if remainder > 0 else 0)
        remainder -= 1
        pilot.extend(pool[:take])

    # If we still need more, pick randomly from remaining
    used_ids = {t["task_id"] for t in pilot}
    remaining = [t for t in tasks if t["task_id"] not in used_ids]
    rng.shuffle(remaining)
    while len(pilot) < n and remaining:
        pilot.append(remaining.pop())

    return pilot[:n]


def print_stats(tasks: list[dict], label: str = "Filtered"):
    """Print statistics about the task set."""
    print(f"\n{'=' * 60}")
    print(f"  {label} Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"  Total tasks: {len(tasks)}")

    if not tasks:
        print("  (no tasks found)")
        return

    # Library distribution
    lib_counts: dict[str, int] = {}
    for t in tasks:
        for lib in t["target_libraries_matched"]:
            lib_counts[lib] = lib_counts.get(lib, 0) + 1

    print(f"\n  Target library distribution:")
    for lib, count in sorted(lib_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 40)
        print(f"    {lib:12s}: {count:4d} tasks  {bar}")

    # Multi-library tasks
    multi = sum(1 for t in tasks if t["n_target_matches"] > 1)
    print(f"\n  Tasks using multiple target libraries: {multi} "
          f"({multi/len(tasks):.0%})")

    # Prompt length stats
    prompt_lens = [len(t["complete_prompt"]) for t in tasks if t["complete_prompt"]]
    if prompt_lens:
        print(f"\n  Prompt length (chars): "
              f"min={min(prompt_lens)}, median={sorted(prompt_lens)[len(prompt_lens)//2]}, "
              f"max={max(prompt_lens)}")

    # Test availability
    has_test = sum(1 for t in tasks if t["test"])
    print(f"  Tasks with test code: {has_test}/{len(tasks)} ({has_test/len(tasks):.0%})")


def main():
    parser = argparse.ArgumentParser(
        description="Filter BigCodeBench tasks by target libraries"
    )
    parser.add_argument("--pilot", type=int, default=0,
                        help="Select N pilot tasks (0 = no pilot selection)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Just print stats, don't save")
    parser.add_argument("--min-libs", type=int, default=1,
                        help="Minimum number of target libraries a task must use")
    args = parser.parse_args()

    # Load dataset
    ds = load_bigcodebench()

    # Show raw column info
    print("  Sample task fields:")
    sample = ds[0]
    for key in sorted(sample.keys()):
        val = sample[key]
        if isinstance(val, str):
            print(f"    {key}: {type(val).__name__} ({len(val)} chars)")
        elif isinstance(val, list):
            print(f"    {key}: list ({len(val)} items)")
        else:
            print(f"    {key}: {val}")

    # Filter
    print(f"\n  Filtering for libraries: {sorted(TARGET_LIBRARIES)}")
    print(f"  Minimum matches required: {args.min_libs}")

    tasks = filter_tasks(ds, TARGET_LIBRARIES, min_libs=args.min_libs)
    print_stats(tasks, "Filtered")

    if args.stats_only:
        return

    # Save filtered tasks
    RAW_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_TASKS_DIR / "filtered_tasks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(tasks)} tasks to {output_path}")

    # Pilot selection
    if args.pilot > 0:
        pilot = select_pilot_tasks(tasks, n=args.pilot)
        print_stats(pilot, "Pilot Subset")

        pilot_path = PILOT_DIR / "pilot_tasks.json"
        PILOT_DIR.mkdir(parents=True, exist_ok=True)
        with open(pilot_path, "w", encoding="utf-8") as f:
            json.dump(pilot, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(pilot)} pilot tasks to {pilot_path}")

        # Also save just the task IDs for quick reference
        pilot_ids = [t["task_id"] for t in pilot]
        ids_path = PILOT_DIR / "pilot_task_ids.json"
        with open(ids_path, "w") as f:
            json.dump(pilot_ids, f, indent=2)
        print(f"  Saved pilot task IDs to {ids_path}")

    print(f"\n  Done! Next step: run generate.py to produce LLM completions.")


if __name__ == "__main__":
    main()
