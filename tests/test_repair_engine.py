"""
test_repair_engine.py -- Verification tests for the Self-Healing Auto-Repair Engine.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.repair import repair_code
from test_real_world import PRESETS


def test_preset_repairs():
    print("=" * 65)
    print("  VERIFYING SELF-HEALING AUTO-REPAIR ENGINE ON PRESETS")
    print("=" * 65)

    for preset_key in ("requests_misuse", "pandas_misuse", "numpy_misuse"):
        preset = PRESETS[preset_key]
        print(f"\n[*] Testing Preset: {preset['name']}")
        result = repair_code(preset["code"])

        print(f"    Repaired: {result.repaired}")
        print(f"    Actions taken: {len(result.repaired_issues)}")
        for issue in result.repaired_issues:
            print(f"      - Line {issue['line']}: {issue['action']}")

        print(f"    Remaining misuse violations: {len(result.remaining_issues)}")
        assert result.repaired, f"Expected repair on {preset_key}"
        assert len(result.remaining_issues) == 0, f"Expected 0 remaining issues on {preset_key}"

        print("    --> Unified Diff:")
        for line in result.diff.splitlines()[:12]:
            print(f"        {line}")

    # Additional Pattern 1: Requests missing parentheses: resp.json['key']
    print("\n[*] Testing Pattern: Requests missing parentheses resp.json['id']")
    code_req_parens = (
        "import requests\n"
        "resp = requests.get('https://api.github.com')\n"
        "data = resp.json['name']\n"
    )
    res_parens = repair_code(code_req_parens)
    print(f"    Repaired: {res_parens.repaired}, Remaining issues: {len(res_parens.remaining_issues)}")
    assert res_parens.repaired
    assert "resp.json()['name']" in res_parens.repaired_code
    assert len(res_parens.remaining_issues) == 0

    # Additional Pattern 2: Pandas inplace None assignment: df = df.dropna(inplace=True)
    print("\n[*] Testing Pattern: Pandas inplace None assignment: df = df.dropna(inplace=True)")
    code_inplace = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1, None, 3]})\n"
        "df = df.dropna(inplace=True)\n"
        "result = df.head()\n"
    )
    res_inplace = repair_code(code_inplace)
    print(f"    Repaired: {res_inplace.repaired}, Remaining issues: {len(res_inplace.remaining_issues)}")
    assert res_inplace.repaired
    assert "inplace=True" not in res_inplace.repaired_code
    assert len(res_inplace.remaining_issues) == 0

    print("\n" + "=" * 65)
    print("  ALL AUTO-REPAIR PRESET & EXTENDED TESTS PASSED!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    test_preset_repairs()
