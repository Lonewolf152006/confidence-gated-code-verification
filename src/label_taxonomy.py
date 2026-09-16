"""
label_taxonomy.py -- Fine-grained error taxonomy & labeling for code generation.

Classifies test execution results into precise failure categories:
  1. PASS
  2. SYNTAX_ERROR
  3. IMPORT_ERROR
  4. TYPE_ERROR
  5. ATTRIBUTE_ERROR (Classic API hallucination: non-existent method/attr)
  6. USAGE_SEMANTIC_MISUSE (Intent Misuse: correct API name, wrong return-type / call assumption)
  7. KEY_ERROR / INDEX_ERROR (Container indexing mistakes)
  8. VALUE_ERROR
  9. ASSERTION_ERROR (Logic mismatch / incorrect return value)
 10. TIMEOUT
 11. OTHER_RUNTIME_ERROR
"""

import os
import sys
import re
from typing import Optional
from dataclasses import dataclass, asdict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.parse_calls import check_code_for_misuse


@dataclass
class LabelResult:
    status: str             # "PASS", "FAIL", "TIMEOUT", "ERROR"
    failure_category: str   # One of the taxonomy categories (or "NONE" if passed)
    primary_exception: Optional[str]  # e.g., "TypeError", "AttributeError"
    exception_message: Optional[str]
    is_intent_misuse: bool  # True if usage-semantic misuse detected by AST / execution
    ast_misuse_details: list
    raw_error: str


def extract_exception_info(stderr: str) -> tuple[Optional[str], Optional[str]]:
    """Extract exception class and message from python traceback."""
    if not stderr:
        return None, None

    # Search for standard Exception line at end of traceback: e.g. "TypeError: 'Response' object is not subscriptable"
    lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_\.]*(?:Error|Exception|Warning|Interrupt)):\s*(.*)$", line)
        if match:
            return match.group(1).split(".")[-1], match.group(2)
        # Handle simple 'AssertionError' with no message
        if line == "AssertionError" or line.startswith("AssertionError"):
            parts = line.split(":", 1)
            msg = parts[1].strip() if len(parts) > 1 else ""
            return "AssertionError", msg

    return None, None


def classify_execution(
    generated_code: str,
    passed: bool,
    stderr: str = "",
    timed_out: bool = False,
) -> LabelResult:
    """
    Classify execution outcome into the failure taxonomy.
    Combines runtime traceback analysis with static AST def-use chain inspection.
    """
    if passed:
        return LabelResult(
            status="PASS",
            failure_category="NONE",
            primary_exception=None,
            exception_message=None,
            is_intent_misuse=False,
            ast_misuse_details=[],
            raw_error="",
        )

    if timed_out:
        return LabelResult(
            status="TIMEOUT",
            failure_category="TIMEOUT",
            primary_exception="TimeoutError",
            exception_message="Execution exceeded time limit",
            is_intent_misuse=False,
            ast_misuse_details=[],
            raw_error=stderr,
        )

    # Check AST for usage-semantic misuse
    ast_misuses = check_code_for_misuse(generated_code)
    exc_type, exc_msg = extract_exception_info(stderr)

    # Categorize
    if exc_type == "SyntaxError" or "SyntaxError" in stderr:
        category = "SYNTAX_ERROR"
    elif exc_type in ("ModuleNotFoundError", "ImportError"):
        category = "IMPORT_ERROR"
    elif exc_type == "AttributeError":
        # Check if it was an intent misuse or classic hallucination
        if len(ast_misuses) > 0:
            category = "USAGE_SEMANTIC_MISUSE"
        else:
            category = "ATTRIBUTE_ERROR"
    elif exc_type == "TypeError":
        # TypeError: 'Response' object is not subscriptable is classic Intent Misuse
        if len(ast_misuses) > 0 or (exc_msg and ("not subscriptable" in exc_msg or "argument" in exc_msg)):
            category = "USAGE_SEMANTIC_MISUSE"
        else:
            category = "TYPE_ERROR"
    elif exc_type in ("KeyError", "IndexError"):
        if len(ast_misuses) > 0:
            category = "USAGE_SEMANTIC_MISUSE"
        else:
            category = "KEY_INDEX_ERROR"
    elif exc_type == "ValueError":
        category = "VALUE_ERROR"
    elif exc_type == "AssertionError" or "AssertionError" in stderr:
        category = "ASSERTION_ERROR"
    else:
        if len(ast_misuses) > 0:
            category = "USAGE_SEMANTIC_MISUSE"
        else:
            category = "OTHER_RUNTIME_ERROR"

    is_intent_misuse = (category == "USAGE_SEMANTIC_MISUSE") or (len(ast_misuses) > 0)

    return LabelResult(
        status="FAIL",
        failure_category=category,
        primary_exception=exc_type or "UnknownError",
        exception_message=exc_msg or "",
        is_intent_misuse=is_intent_misuse,
        ast_misuse_details=ast_misuses,
        raw_error=stderr[:1000] if stderr else "",
    )


if __name__ == "__main__":
    # Self-test with sample cases
    code_pass = "def task_func(x): return x * 2"
    res_pass = classify_execution(code_pass, passed=True)
    print("Pass case:", asdict(res_pass))

    code_misuse = "import requests\nresp = requests.get('url')\nx = resp['data']"
    res_misuse = classify_execution(
        code_misuse,
        passed=False,
        stderr="Traceback (most recent call last):\n  File 'test.py', line 3\nTypeError: 'Response' object is not subscriptable"
    )
    print("\nMisuse case:", asdict(res_misuse))
