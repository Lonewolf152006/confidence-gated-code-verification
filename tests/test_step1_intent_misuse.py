"""
tests/test_step1_intent_misuse.py -- Standalone validation for Step 1.

Validates that check_code_for_misuse and classify_execution return
is_intent_misuse=True for all three Intent Misuse patterns:
  a. Requests Response subscripted directly without .json()
  b. Obsolete df.sort(...) on pandas object instead of sort_values()
  c. np.abs() used on array/vector where vector magnitude/norm is intended
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.label_taxonomy import classify_execution, check_code_for_misuse

# 1. Hand-crafted Pattern a: requests response subscript
EXAMPLE_A = '''
import requests

def fetch_data(url: str):
    response = requests.get(url)
    user_id = response["user_id"]
    return user_id
'''

# 2. Hand-crafted Pattern b: pandas obsolete df.sort(...)
EXAMPLE_B = '''
import pandas as pd

def process_table(path: str):
    df = pd.read_csv(path)
    sorted_df = df.sort("revenue", ascending=False)
    return sorted_df
'''

# 3. Hand-crafted Pattern c: np.abs() for vector magnitude
EXAMPLE_C = '''
import numpy as np

def compute_velocity_magnitude(vx: float, vy: float, vz: float) -> float:
    velocity_vector = np.array([vx, vy, vz])
    magnitude = np.abs(velocity_vector)
    return float(np.sum(magnitude))
'''

def run_tests():
    print("=" * 70)
    print("  STEP 1 VALIDATION: Hand-Crafted Intent Misuse Patterns")
    print("=" * 70)

    # Test A: Requests
    misuses_a = check_code_for_misuse(EXAMPLE_A)
    res_a = classify_execution(
        EXAMPLE_A,
        passed=False,
        stderr="TypeError: 'Response' object is not subscriptable"
    )
    print(f"\n[Pattern A] Requests direct subscript:")
    print(f"  AST Misuses detected : {len(misuses_a)} -> {misuses_a[0]['misuse_type'] if misuses_a else 'None'}")
    print(f"  Failure Category      : {res_a.failure_category}")
    print(f"  is_intent_misuse      : {res_a.is_intent_misuse}")
    assert res_a.is_intent_misuse is True, "Pattern A must return is_intent_misuse=True"

    # Test B: Pandas
    misuses_b = check_code_for_misuse(EXAMPLE_B)
    res_b = classify_execution(
        EXAMPLE_B,
        passed=False,
        stderr="AttributeError: 'DataFrame' object has no attribute 'sort'"
    )
    print(f"\n[Pattern B] Pandas obsolete sort:")
    print(f"  AST Misuses detected : {len(misuses_b)} -> {misuses_b[0]['misuse_type'] if misuses_b else 'None'}")
    print(f"  Failure Category      : {res_b.failure_category}")
    print(f"  is_intent_misuse      : {res_b.is_intent_misuse}")
    assert res_b.is_intent_misuse is True, "Pattern B must return is_intent_misuse=True"

    # Test C: NumPy
    misuses_c = check_code_for_misuse(EXAMPLE_C)
    res_c = classify_execution(
        EXAMPLE_C,
        passed=False,
        stderr="AssertionError: 6.0 != 3.7416573867739413 (velocity magnitude incorrect)"
    )
    print(f"\n[Pattern C] NumPy abs vector norm misuse:")
    print(f"  AST Misuses detected : {len(misuses_c)} -> {misuses_c[0]['misuse_type'] if misuses_c else 'None'}")
    print(f"  Failure Category      : {res_c.failure_category}")
    print(f"  is_intent_misuse      : {res_c.is_intent_misuse}")
    assert res_c.is_intent_misuse is True, "Pattern C must return is_intent_misuse=True"

    print("\n" + "=" * 70)
    print("  CONFIRMED: is_intent_misuse returns True for all 3 patterns!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
