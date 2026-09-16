"""
test_parse_calls.py — Unit tests for the AST-based API call parser.

Tests both v1 (direct chain) and v2 (assign-then-misuse-later) detection,
plus correct-code false-positive checks.
"""

import sys
import os
import unittest

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from parse_calls import extract_api_calls


class TestDirectChainDetection(unittest.TestCase):
    """v1 behavior — direct chained usage is still detected."""

    def test_chained_json(self):
        """requests.get(url).json() → consumed_by should be '.json'"""
        code = "data = requests.get(url).json()"
        calls = extract_api_calls(code)
        get_call = [c for c in calls if c.full_expr == "requests.get"]
        self.assertEqual(len(get_call), 1)
        self.assertEqual(get_call[0].consumed_by, ".json")

    def test_chained_subscript(self):
        """requests.get(url)['key'] → consumed_by should be '[subscript]'"""
        code = "data = requests.get(url)['key']"
        calls = extract_api_calls(code)
        get_call = [c for c in calls if c.full_expr == "requests.get"]
        self.assertEqual(len(get_call), 1)
        self.assertEqual(get_call[0].consumed_by, "[subscript]")

    def test_discarded_result(self):
        """print(x) → consumed_by should be '[discarded]'"""
        code = "print('hello')"
        calls = extract_api_calls(code)
        print_call = [c for c in calls if c.full_expr == "print"]
        self.assertEqual(len(print_call), 1)
        self.assertEqual(print_call[0].consumed_by, "[discarded]")


class TestDeferredUsageDetection(unittest.TestCase):
    """v2 behavior — assign-then-misuse-later patterns are caught."""

    def test_assign_then_subscript(self):
        """resp = requests.get(url); resp['key'] → should detect deferred subscript usage."""
        code = """\
def f():
    resp = requests.get(url)
    data = resp['results']
"""
        calls = extract_api_calls(code)
        get_call = [c for c in calls if c.full_expr == "requests.get"]
        self.assertEqual(len(get_call), 1)

        call = get_call[0]
        self.assertEqual(call.consumed_by, "[assigned]")
        self.assertEqual(call.assigned_to, "resp")
        self.assertTrue(len(call.usage_points) > 0,
                        "Should have tracked deferred usage of 'resp'")

        # The deferred usage should be a subscript
        subscript_usages = [u for u in call.usage_points if "[" in u.kind]
        self.assertTrue(len(subscript_usages) > 0,
                        f"Should have found subscript usage, got: {call.usage_points}")

    def test_assign_then_method(self):
        """resp = requests.get(url); resp.json() → should detect deferred .json() usage."""
        code = """\
def f():
    resp = requests.get(url)
    data = resp.json()
"""
        calls = extract_api_calls(code)
        get_call = [c for c in calls if c.full_expr == "requests.get"]
        self.assertEqual(len(get_call), 1)

        call = get_call[0]
        self.assertEqual(call.assigned_to, "resp")

        json_usages = [u for u in call.usage_points if u.kind == ".json"]
        self.assertTrue(len(json_usages) > 0,
                        f"Should have found .json() usage, got: {call.usage_points}")

    def test_multiple_deferred_usages(self):
        """resp used multiple times → should detect all usages."""
        code = """\
def f():
    resp = requests.get(url)
    status = resp.status_code
    body = resp.json()
"""
        calls = extract_api_calls(code)
        get_call = [c for c in calls if c.full_expr == "requests.get"]
        self.assertEqual(len(get_call), 1)

        call = get_call[0]
        self.assertTrue(len(call.usage_points) >= 2,
                        f"Should have found ≥2 usages, got {len(call.usage_points)}: "
                        f"{call.usage_points}")

    def test_reassignment_stops_tracking(self):
        """If the variable is reassigned, stop tracking at that point."""
        code = """\
def f():
    resp = requests.get(url)
    resp = requests.get(other_url)
    data = resp.json()
"""
        calls = extract_api_calls(code)
        get_calls = [c for c in calls if c.full_expr == "requests.get"]
        # First call's usages should NOT include the .json() after reassignment
        first_call = get_calls[0]
        json_usages = [u for u in first_call.usage_points if u.kind == ".json"]
        self.assertEqual(len(json_usages), 0,
                         "First call should not track usage after variable reassignment")


class TestCorrectCodeNoFalsePositives(unittest.TestCase):
    """Correct code should parse without issues."""

    def test_correct_requests_usage(self):
        """Standard correct requests usage should parse cleanly."""
        code = """\
def f():
    resp = requests.get(url)
    data = resp.json()['results']
"""
        calls = extract_api_calls(code)
        get_call = [c for c in calls if c.full_expr == "requests.get"]
        self.assertEqual(len(get_call), 1)
        # Should find .json() as deferred usage
        json_usages = [u for u in get_call[0].usage_points if u.kind == ".json"]
        self.assertTrue(len(json_usages) > 0)

    def test_correct_pandas_usage(self):
        """Standard pandas usage should parse without errors."""
        code = """\
import pandas as pd
def f():
    df = pd.read_csv(path)
    result = df.sort_values('date')
"""
        calls = extract_api_calls(code)
        read_call = [c for c in calls if c.full_expr == "pd.read_csv"]
        self.assertEqual(len(read_call), 1)


class TestTopLevelCode(unittest.TestCase):
    """Parser should work on top-level code (not inside a function)."""

    def test_top_level_assign_then_use(self):
        """Top-level: resp = requests.get(url); resp.json()"""
        code = """\
resp = requests.get(url)
data = resp.json()
"""
        calls = extract_api_calls(code)
        get_call = [c for c in calls if c.full_expr == "requests.get"]
        self.assertEqual(len(get_call), 1)

        call = get_call[0]
        self.assertEqual(call.assigned_to, "resp")
        json_usages = [u for u in call.usage_points if u.kind == ".json"]
        self.assertTrue(len(json_usages) > 0,
                        "Should track deferred usage in top-level code too")


class TestEdgeCases(unittest.TestCase):
    """Edge cases that should not crash the parser."""

    def test_lambda_calls_skipped(self):
        """Lambda calls should be gracefully skipped (no dotted name)."""
        code = "(lambda x: x)(42)"
        calls = extract_api_calls(code)
        # Lambda calls have no dotted name → should be skipped
        self.assertEqual(len(calls), 0)

    def test_empty_code(self):
        """Empty code should return no calls."""
        calls = extract_api_calls("")
        self.assertEqual(len(calls), 0)

    def test_nested_calls(self):
        """Nested calls: json.loads(resp.text) should find both calls."""
        code = """\
import json, requests
def f():
    resp = requests.get(url)
    data = json.loads(resp.text)
"""
        calls = extract_api_calls(code)
        call_names = [c.full_expr for c in calls]
        self.assertIn("requests.get", call_names)
        self.assertIn("json.loads", call_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
