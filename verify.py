"""
verify.py -- Unified Developer CLI & Self-Healing Verifier for Python Code.

Scans Python files or code snippets for usage-semantic hallucinations,
contract violations, and logic errors using the Confidence-Gated Cascade.
Can automatically synthesize AST repairs and patch files in-place.

Usage:
  python verify.py --file path/to/script.py
  python verify.py --file path/to/script.py --fix
  python verify.py --preset requests_misuse
  python verify.py --preset requests_misuse --fix
  python verify.py --code "import requests\nr = requests.get('url')\nx = r['key']" --fix
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_real_world import PRESETS, analyze_real_world_code
from src.repair import repair_code
from src.specialist_model import SpecialistVerifier

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "specialist_model.pt"

# ANSI Terminal Colors
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"


def print_colored_diff(diff_text: str):
    """Print a colorized unified diff in the terminal."""
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            print(f"{COLOR_BOLD}{line}{COLOR_RESET}")
        elif line.startswith("+"):
            print(f"{COLOR_GREEN}{line}{COLOR_RESET}")
        elif line.startswith("-"):
            print(f"{COLOR_RED}{line}{COLOR_RESET}")
        elif line.startswith("@@"):
            print(f"{COLOR_CYAN}{line}{COLOR_RESET}")
        else:
            print(line)


def run_verification(
    code: str,
    target_name: str = "Input Code",
    file_path: Path = None,
    apply_fix: bool = False,
    mode: str = "standalone",
    verifier: SpecialistVerifier = None,
) -> int:
    """
    Run verification and optional self-healing auto-repair.
    Returns exit code (0 for pass/repaired, 1 for unfixable defect).
    """
    print(f"\n{'=' * 80}")
    print(f"  {COLOR_BOLD}CONFIDENCE-GATED VERIFIER & SELF-HEALING ENGINE{COLOR_RESET}")
    print(f"  Target: {COLOR_CYAN}{target_name}{COLOR_RESET} | Mode: {COLOR_YELLOW}{mode.upper()}{COLOR_RESET}")
    print(f"{'=' * 80}")

    # 1. Analyze code
    report = analyze_real_world_code(
        code=code,
        tau_accept=0.30,
        tau_reject=0.75,
        verifier=verifier,
        mode=mode
    )

    verdict = report.get("verdict", "UNKNOWN")
    confidence = report.get("confidence", "UNKNOWN")
    primary_reason = report.get("primary_reason", "")
    ast_misuses = report.get("ast_misuses", [])

    if "ACCEPT" in verdict:
        print(f"\n  {COLOR_GREEN}{COLOR_BOLD}>>> VERDICT: [ ACCEPTED - CONFIDENTLY SAFE ]{COLOR_RESET}")
        print(f"  Confidence:     {confidence}")
        print(f"  Primary Reason: {primary_reason}\n")
        print(f"  {COLOR_GREEN}[OK] No usage-semantic contract violations detected.{COLOR_RESET}")
        print(f"{'=' * 80}\n")
        return 0

    # 2. Defect Detected
    is_misuse = "INTENT MISUSE" in verdict or len(ast_misuses) > 0
    verdict_color = COLOR_RED if is_misuse else COLOR_YELLOW
    print(f"\n  {verdict_color}{COLOR_BOLD}>>> VERDICT: [ {verdict} ]{COLOR_RESET}")
    print(f"  Confidence:     {confidence}")
    print(f"  Primary Reason: {primary_reason}\n")

    if ast_misuses:
        print(f"  {COLOR_RED}[!] Detected {len(ast_misuses)} Semantic Contract Violation(s):{COLOR_RESET}")
        for m in ast_misuses:
            line_no = m.get("line", 1)
            desc = m.get("description", m.get("misuse_type", ""))
            print(f"      * {COLOR_BOLD}Line {line_no}:{COLOR_RESET} {desc}")

    # 3. Auto-Repair Engine
    if apply_fix:
        print(f"\n{'-' * 80}")
        print(f"  {COLOR_BOLD}AUTO-REPAIR ENGINE: Synthesizing AST Transformations...{COLOR_RESET}")
        print(f"{'-' * 80}")

        repair_result = repair_code(code)

        if repair_result.repaired and len(repair_result.remaining_issues) == 0:
            print(f"\n  {COLOR_GREEN}{COLOR_BOLD}[OK] SELF-HEALING SUCCESSFUL: All semantic contract bugs resolved!{COLOR_RESET}")
            print(f"  Actions Applied ({len(repair_result.repaired_issues)}):")
            for action in repair_result.repaired_issues:
                print(f"    - Line {action.get('line', 1)}: {action.get('action')}")

            print(f"\n  {COLOR_BOLD}Unified Patch Diff:{COLOR_RESET}")
            print_colored_diff(repair_result.diff)

            # In-place file update if requested
            if file_path and file_path.exists():
                file_path.write_text(repair_result.repaired_code, encoding="utf-8")
                print(f"\n  {COLOR_GREEN}[OK] Successfully updated file in-place: {file_path}{COLOR_RESET}")

            print(f"\n{'=' * 80}\n")
            return 0
        else:
            print(f"\n  {COLOR_YELLOW}[!] Partial repair applied, but manual review is recommended.{COLOR_RESET}")
            if repair_result.diff:
                print_colored_diff(repair_result.diff)
            print(f"\n{'=' * 80}\n")
            return 1
    else:
        print(f"\n  {COLOR_BLUE}Tip: Run with --fix to automatically synthesize and apply AST repairs.{COLOR_RESET}")
        print(f"{'=' * 80}\n")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Unified Verifier & Self-Healing Auto-Repair CLI")
    parser.add_argument("--file", type=str, help="Path to Python script to verify")
    parser.add_argument("--code", type=str, help="Inline code string to verify")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help=f"Verify a curated benchmark preset: {', '.join(PRESETS.keys())}",
    )
    parser.add_argument("--fix", action="store_true", help="Automatically synthesize and apply AST repairs")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standalone", "live"],
        default="standalone",
        help="Verification mode: standalone (zero-GPU, AST + TF-IDF) or live (with generation logprobs)",
    )
    args = parser.parse_args()

    # Load specialist model if present
    verifier = None
    if MODEL_PATH.exists():
        try:
            verifier = SpecialistVerifier(MODEL_PATH)
        except Exception:
            verifier = None

    if args.preset:
        preset_info = PRESETS[args.preset]
        target_name = f"Preset: {preset_info['name']}"
        code_to_test = preset_info["code"]
        file_path = None
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        code_to_test = file_path.read_text(encoding="utf-8")
        target_name = str(file_path)
    elif args.code:
        code_to_test = args.code
        target_name = "Inline Snippet"
        file_path = None
    else:
        print("Usage: python verify.py [--file FILE | --preset PRESET | --code CODE] [--fix]")
        sys.exit(1)

    exit_code = run_verification(
        code=code_to_test,
        target_name=target_name,
        file_path=file_path,
        apply_fix=args.fix,
        mode=args.mode,
        verifier=verifier,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
