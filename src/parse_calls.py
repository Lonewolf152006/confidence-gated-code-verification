"""
parse_calls.py — AST-based extraction of API call sites from generated code.

This is Phase 3 of the roadmap. It finds every function/method call in a piece
of code and records enough information to later align each call with the
tokens the LLM generated for it (needed for the entropy signal).

Tested standalone below with ast.get_source_segment — no LLM or GPU needed.

CHANGELOG:
  - v2: Added lightweight def-use chain tracking. When a call's result is
    assigned to a variable, we forward-scan to find the next usage of that
    variable and treat that as an additional "decision point". This catches
    the assign-then-misuse-later pattern (e.g. `resp = requests.get(url)` ...
    `resp['results']` two lines down) that v1 missed.
"""

import ast
from dataclasses import dataclass, field


@dataclass
class UsagePoint:
    """How a call result (or the variable it was assigned to) is consumed."""
    kind: str              # e.g. ".json()", "[subscript]", ".text", "[assigned]", "[discarded]"
    lineno: int
    col_offset: int
    source: str = ""       # source text of the consuming expression


@dataclass
class APICall:
    """One extracted call site."""
    full_expr: str          # e.g. "requests.get" or "df.dropna"
    call_source: str        # the exact source text of the full call, e.g. "requests.get(url)"
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    args_source: list = field(default_factory=list)
    # What the call's result is immediately used for (the "decision point").
    # None if the call is a standalone statement (result discarded).
    consumed_by: str | None = None
    # NEW in v2: all usage points including deferred variable usage
    usage_points: list[UsagePoint] = field(default_factory=list)
    # The variable name the result was assigned to, if any
    assigned_to: str | None = None


def _expr_to_dotted_name(node: ast.expr) -> str | None:
    """Turn an ast.Attribute/ast.Name chain like `requests.get` into a string."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _get_assignment_target(parent: ast.AST) -> str | None:
    """If the parent is an assignment, return the target variable name."""
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
        target = parent.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    elif isinstance(parent, ast.AnnAssign) and parent.target:
        if isinstance(parent.target, ast.Name):
            return parent.target.id
    return None


def _describe_usage(node: ast.AST, parent: ast.AST, code: str) -> UsagePoint | None:
    """Describe how a Name node is consumed based on its parent in the AST."""
    source = ast.get_source_segment(code, parent) or ""

    if isinstance(parent, ast.Attribute):
        # e.g. resp.json() or resp.text
        kind = f".{parent.attr}"
        return UsagePoint(kind=kind, lineno=parent.lineno,
                          col_offset=parent.col_offset, source=source)
    elif isinstance(parent, ast.Subscript):
        # e.g. resp['results'] or resp[0]
        slice_src = ast.get_source_segment(code, parent.slice) or ""
        kind = f"[{slice_src}]" if slice_src else "[subscript]"
        return UsagePoint(kind=kind, lineno=parent.lineno,
                          col_offset=parent.col_offset, source=source)
    elif isinstance(parent, ast.Call) and hasattr(parent, 'func'):
        # The variable is being passed as an argument to another function
        func_name = _expr_to_dotted_name(parent.func)
        if func_name:
            kind = f"arg_to({func_name})"
        else:
            kind = "arg_to(unknown)"
        return UsagePoint(kind=kind, lineno=parent.lineno,
                          col_offset=parent.col_offset, source=source)
    return None


def _find_variable_usages(
    func_body: list[ast.stmt],
    var_name: str,
    assign_lineno: int,
    code: str
) -> list[UsagePoint]:
    """
    Lightweight def-use chain: after `var_name = <call>(...)` on line
    `assign_lineno`, scan forward through the same function body to find
    all subsequent usages of `var_name` and describe how each is consumed.

    This is NOT full data-flow analysis — it's a simple forward scan for
    Name nodes matching `var_name` that appear after the assignment line.
    It stops tracking if `var_name` is reassigned (simple shadowing check).
    """
    usages: list[UsagePoint] = []

    # Build parent map for this function's subtree
    parent_map: dict[int, ast.AST] = {}
    for stmt in func_body:
        for node in ast.walk(stmt):
            for child in ast.iter_child_nodes(node):
                parent_map[id(child)] = node

    for stmt in func_body:
        if stmt.lineno <= assign_lineno:
            continue  # skip lines at or before the assignment

        # Check if this statement reassigns our variable (shadowing → stop tracking)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return usages  # variable reassigned, stop here
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == var_name:
                return usages

        # Walk the statement looking for Name nodes matching var_name
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == var_name:
                parent = parent_map.get(id(node))
                if parent is not None:
                    usage = _describe_usage(node, parent, code)
                    if usage is not None:
                        usages.append(usage)

    return usages


def _get_enclosing_function_body(tree: ast.AST, target_lineno: int) -> list[ast.stmt] | None:
    """Find the function body that contains the given line number."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and node.body[0].lineno <= target_lineno <= node.end_lineno:
                return node.body
    return None


def extract_api_calls(code: str) -> list[APICall]:
    """
    Walk the AST of `code` and return every Call node found, in source order.

    For each call, records:
    - The immediate consumption pattern (consumed_by)
    - All downstream usage points if the result was assigned to a variable
      (usage_points — the v2 def-use chain addition)
    """
    tree = ast.parse(code)
    calls: list[APICall] = []

    # Map each Call node to its immediate parent, so we can tell what happens
    # to its return value (e.g. is it immediately subscripted, or is
    # `.json()` called on it) — this is the "usage pattern" signal.
    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        dotted = _expr_to_dotted_name(node.func)
        if dotted is None:
            continue  # skip calls we can't name cleanly (e.g. lambda calls)

        call_src = ast.get_source_segment(code, node) or ""
        args_src = [ast.get_source_segment(code, a) or "" for a in node.args]

        # Inspect the parent to see how the return value is consumed —
        # this is exactly the "decision point" described in the project plan.
        parent = parent_map.get(id(node))
        consumed_by = None
        assigned_to = None
        usage_points: list[UsagePoint] = []

        if isinstance(parent, ast.Attribute):
            consumed_by = f".{parent.attr}"       # e.g. chained .json()
            usage_points.append(UsagePoint(
                kind=consumed_by, lineno=parent.lineno,
                col_offset=parent.col_offset,
                source=ast.get_source_segment(code, parent) or ""
            ))
        elif isinstance(parent, ast.Subscript):
            consumed_by = "[subscript]"            # e.g. response['key']
            usage_points.append(UsagePoint(
                kind=consumed_by, lineno=parent.lineno,
                col_offset=parent.col_offset,
                source=ast.get_source_segment(code, parent) or ""
            ))
        elif isinstance(parent, ast.Expr):
            consumed_by = "[discarded]"            # result not used at all
            usage_points.append(UsagePoint(
                kind=consumed_by, lineno=parent.lineno,
                col_offset=parent.col_offset
            ))
        elif isinstance(parent, ast.Assign) or isinstance(parent, ast.AnnAssign):
            assigned_to = _get_assignment_target(parent)
            consumed_by = "[assigned]"

            # v2 addition: if assigned to a variable, track its downstream usage
            if assigned_to is not None:
                # Try to find the enclosing function body for forward scanning
                func_body = _get_enclosing_function_body(tree, node.lineno)
                if func_body is not None:
                    deferred_usages = _find_variable_usages(
                        func_body, assigned_to, node.lineno, code
                    )
                    usage_points.extend(deferred_usages)
                else:
                    # Top-level code: scan all top-level statements
                    deferred_usages = _find_variable_usages(
                        tree.body, assigned_to, node.lineno, code
                    )
                    usage_points.extend(deferred_usages)

        calls.append(APICall(
            full_expr=dotted,
            call_source=call_src,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            args_source=args_src,
            consumed_by=consumed_by,
            usage_points=usage_points,
            assigned_to=assigned_to,
        ))

    return calls


def check_code_for_misuse(code: str) -> list[dict]:
    """
    Analyzes code using AST to identify syntactic/usage-semantic intent misuses.
    Delegates to src.label_taxonomy.check_code_for_misuse for the full pattern suite.
    """
    from src.label_taxonomy import check_code_for_misuse as _check
    return _check(code)


if __name__ == "__main__":
    # --- Self-test with examples demonstrating both v1 and v2 detection ---
    samples = {
        # 1. Clean, correct usage (direct chain)
        "correct_chained": """\
import requests
resp = requests.get(url)
data = resp.json()['results']
""",
        # 2. Fabricated API (Hallucination Misuse) — existence-checkable
        "hallucination_misuse": """\
import pandas as pd
df = pd.read_excel_fast(path)
""",
        # 3. Intent Misuse — direct chain (v1 catches this)
        "intent_misuse_chained": """\
import requests
data = requests.get(url)['results']
""",
        # 4. Intent Misuse — assign then misuse (v1 MISSED this, v2 catches it)
        "intent_misuse_deferred": """\
import requests
def get_data(url):
    resp = requests.get(url)
    data = resp['results']
    return data
""",
        # 5. Correct deferred usage
        "correct_deferred": """\
import requests
def get_data(url):
    resp = requests.get(url)
    data = resp.json()
    return data['results']
""",
        # 6. Multiple deferred usages of same variable
        "multi_usage": """\
import requests
def process(url):
    resp = requests.get(url)
    status = resp.status_code
    body = resp.json()
    return status, body
""",
    }

    for label, code in samples.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for c in extract_api_calls(code):
            print(f"  call: {c.full_expr:25s} consumed_by: {c.consumed_by!s:15s} "
                  f"assigned_to: {c.assigned_to!s:10s}")
            if c.usage_points:
                for u in c.usage_points:
                    print(f"    +-- usage: {u.kind:20s} line {u.lineno}  "
                          f"src: {u.source[:50]}")
            else:
                print(f"    +-- (no downstream usage tracked)")
