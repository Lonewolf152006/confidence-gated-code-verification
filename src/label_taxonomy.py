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

import ast

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _get_call_name(node: ast.Call) -> str | None:
    """Extract dotted call name such as requests.get or np.abs."""
    func = node.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return None


def check_code_for_misuse(code: str) -> list[dict]:
    """
    Analyzes code using AST to identify syntactic/usage-semantic intent misuses:
      a. Subscripting a requests.Response object directly (resp["key"] or resp['key'])
         where resp was assigned from requests.get/post/put/etc., instead of resp.json()["key"]
      b. Calling the removed/obsolete list.sort() pattern incorrectly on a pandas object
         (df.sort(...) -- deprecated in favor of sort_values())
      c. Using np.abs() where surrounding context computes a vector magnitude / Euclidean norm
         (look for multi-dimensional array inputs going into abs() where norm-like usage is implied)
    """
    clean_code = code.split("```")[0]
    if "```python" in code:
        parts = code.split("```python")
        if len(parts) > 1:
            clean_code = parts[1].split("```")[0]

    try:
        tree = ast.parse(clean_code)
    except Exception:
        try:
            tree = ast.parse(f"def _wrap():\n{clean_code}")
        except Exception:
            return []

    misuses = []

    # Map each node to its parent
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent

    # 1. Track variable assignments to identify types
    var_types: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_names = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        target_names.append(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        for elt in t.elts:
                            if isinstance(elt, ast.Name):
                                target_names.append(elt.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_names.append(node.target.id)

            val = node.value
            if val is None:
                continue

            val_call = _get_call_name(val) if isinstance(val, ast.Call) else ""
            val_call = val_call or ""

            for t_name in target_names:
                # Requests response objects
                if any(val_call.startswith(p) for p in ("requests.get", "requests.post", "requests.put", "requests.delete", "requests.patch", "requests.request")):
                    var_types[t_name] = "requests_response"
                elif t_name.lower() in ("response", "resp", "r"):
                    var_types[t_name] = "requests_response"
                # Bytes / Digest objects
                elif val_call.endswith(".digest") or "digest" in val_call or t_name.lower() == "digest":
                    var_types[t_name] = "bytes_digest"
                # Python lists / sequences
                elif isinstance(val, (ast.List, ast.ListComp)) or val_call in ("zip", "list"):
                    var_types[t_name] = "python_list"
                # Uncast collection lookups (e.g. data_dict[key])
                elif isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and ("dict" in val.value.id.lower() or "data" in val.value.id.lower()):
                    var_types[t_name] = "dict_value_uncast"
                # Pandas objects
                elif any(val_call.startswith(p) for p in ("pd.read_", "pandas.read_", "pd.DataFrame", "pandas.DataFrame", "pd.Series", "pandas.Series", "pd.concat", "pd.merge")):
                    var_types[t_name] = "pandas_obj"
                elif t_name.lower() in ("df", "dataframe", "sales_df", "data_df") or t_name.lower().endswith(("_df", "_dataframe")):
                    var_types[t_name] = "pandas_obj"
                # NumPy arrays / vectors
                elif any(val_call.startswith(p) for p in ("np.array", "numpy.array", "np.zeros", "np.ones", "np.asarray")):
                    var_types[t_name] = "numpy_array"
                elif any(k in t_name.lower() for k in ("vector", "vec", "velocity", "coords")):
                    var_types[t_name] = "numpy_array"

    # Enclosing function names
    func_names = [n.name.lower() for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    has_magnitude_func = any(k in f for f in func_names for k in ("magnitude", "norm", "euclidean", "distance", "speed"))
    code_lower = code.lower()
    has_norm_text = any(k in code_lower for k in ("magnitude", "euclidean norm", "vector magnitude", "norm("))

    # 2. Walk AST to detect misuse patterns
    for node in ast.walk(tree):
        # Pattern a: Subscripting a requests.Response object directly
        if isinstance(node, ast.Subscript):
            val = node.value
            # a1: resp['key'] or resp["key"]
            if isinstance(val, ast.Name):
                var_name = val.id
                if var_types.get(var_name) == "requests_response" or var_name.lower() in ("response", "resp"):
                    slice_desc = ast.unparse(node.slice) if hasattr(ast, "unparse") else "key"
                    misuses.append({
                        "call": f"{var_name}[{slice_desc}]",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "requests_subscript_without_json",
                        "description": f"Direct subscript {var_name}[{slice_desc}] on Response object without calling .json()",
                    })
            # a2: chained requests.get(...)['key']
            elif isinstance(val, ast.Call):
                cname = _get_call_name(val) or ""
                if any(cname.startswith(p) for p in ("requests.get", "requests.post", "requests.put", "requests.delete", "requests.patch")):
                    slice_desc = ast.unparse(node.slice) if hasattr(ast, "unparse") else "key"
                    misuses.append({
                        "call": f"{cname}[{slice_desc}]",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "requests_subscript_without_json",
                        "description": f"Direct chained subscript on {cname} Response without calling .json()",
                    })
            # a3: resp.json['key'] (treating .json as attribute instead of method)
            elif isinstance(val, ast.Attribute) and val.attr == "json":
                caller = val.value
                caller_name = caller.id if isinstance(caller, ast.Name) else ""
                if var_types.get(caller_name) == "requests_response" or caller_name.lower() in ("response", "resp"):
                    slice_desc = ast.unparse(node.slice) if hasattr(ast, "unparse") else "key"
                    misuses.append({
                        "call": f"{caller_name}.json[{slice_desc}]",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "requests_missing_parentheses_json",
                        "description": f"Subscripting {caller_name}.json as attribute instead of calling method .json()[{slice_desc}]",
                    })

        # Pattern b: Calling obsolete or invalid methods on objects
        if isinstance(node, ast.Call):
            cname = _get_call_name(node) or ""

            # b0: Calling response.text() as method instead of string property
            if isinstance(node.func, ast.Attribute) and node.func.attr == "text":
                caller = node.func.value
                caller_name = caller.id if isinstance(caller, ast.Name) else ""
                if var_types.get(caller_name) == "requests_response" or caller_name.lower() in ("response", "resp", "r"):
                    misuses.append({
                        "call": f"{caller_name}.text()",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "requests_text_called_as_function",
                        "description": f"Calling {caller_name}.text() as a method, but .text is a string property in requests",
                    })

            # b1b: Calling removed df.append() in pandas (removed in Pandas 2.0)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "append":
                caller = node.func.value
                caller_name = caller.id if isinstance(caller, ast.Name) else ""
                has_pandas_kwargs = any(k.arg in ("ignore_index", "verify_integrity", "sort", "axis") for k in node.keywords if k.arg)
                is_subscript_df = isinstance(caller, ast.Subscript) and (
                    (isinstance(caller.value, ast.Name) and "df" in caller.value.id.lower())
                    or var_types.get(getattr(caller.value, "id", "")) == "pandas_obj"
                )
                is_pandas = (
                    var_types.get(caller_name) == "pandas_obj"
                    or caller_name.lower() in ("df", "dataframe", "series", "s", "s1", "s2", "data_df")
                    or caller_name.lower().endswith(("_df", "_dataframe"))
                    or is_subscript_df
                    or has_pandas_kwargs
                )
                if is_pandas:
                    call_label = caller_name if caller_name else ("df[...]" if is_subscript_df else "obj")
                    misuses.append({
                        "call": f"{call_label}.append",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "pandas_removed_append",
                        "description": "Calling DataFrame/Series.append() which was removed in Pandas 2.0 (use pd.concat)",
                    })

            # b1: Obsolete df.sort(...) on pandas object
            if isinstance(node.func, ast.Attribute) and node.func.attr == "sort":
                caller = node.func.value
                caller_name = caller.id if isinstance(caller, ast.Name) else ""
                is_pandas = (
                    var_types.get(caller_name) == "pandas_obj"
                    or caller_name.lower() in ("df", "dataframe", "series", "s", "data_df")
                    or caller_name.lower().endswith(("_df", "_dataframe"))
                )
                has_pandas_kwargs = any(k.arg in ("columns", "ascending", "inplace", "axis") for k in node.keywords if k.arg)
                has_positional_col = len(node.args) > 0 and isinstance(node.args[0], (ast.Constant, ast.List))
                if is_pandas or has_pandas_kwargs or has_positional_col:
                    misuses.append({
                        "call": f"{caller_name}.sort" if caller_name else "df.sort",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "pandas_obsolete_sort",
                        "description": "Calling obsolete DataFrame/Series.sort() instead of sort_values()",
                    })

            # b2: .hexdigest() called on bytes or hash.digest()
            if isinstance(node.func, ast.Attribute) and node.func.attr == "hexdigest":
                caller = node.func.value
                if isinstance(caller, ast.Call) and (_get_call_name(caller) or "").endswith(".digest"):
                    misuses.append({
                        "call": "digest().hexdigest",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "hexdigest_on_bytes",
                        "description": "Calling .hexdigest() on bytes returned by .digest() instead of calling on the hash object directly",
                    })
                elif isinstance(caller, ast.Name) and (var_types.get(caller.id) == "bytes_digest" or caller.id.lower() == "digest"):
                    misuses.append({
                        "call": f"{caller.id}.hexdigest",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "hexdigest_on_bytes",
                        "description": f"Calling .hexdigest() on bytes variable '{caller.id}' instead of the hash object",
                    })

            # b3: .reshape() called on uncast list, comprehension, or dict lookup
            if isinstance(node.func, ast.Attribute) and node.func.attr == "reshape":
                caller = node.func.value
                if isinstance(caller, (ast.List, ast.ListComp)):
                    misuses.append({
                        "call": "[...].reshape",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "reshape_on_dict_list",
                        "description": "Calling .reshape() directly on a Python list or list comprehension without converting to numpy array",
                    })
                elif isinstance(caller, ast.Subscript) and isinstance(caller.value, ast.Name) and "dict" in caller.value.id.lower():
                    misuses.append({
                        "call": "dict[key].reshape",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "reshape_on_dict_list",
                        "description": "Calling .reshape() directly on dict value without converting to numpy array",
                    })
                elif isinstance(caller, ast.Name):
                    cid = caller.id.lower()
                    is_uncast = (
                        var_types.get(caller.id) in ("python_list", "dict_value_uncast")
                        or cid in ("pairs", "weights", "future_dates", "data", "items", "counts", "values", "data_list", "val_list", "l")
                    ) and var_types.get(caller.id) != "numpy_array"
                    if is_uncast:
                        misuses.append({
                            "call": f"{caller.id}.reshape",
                            "line": getattr(node, "lineno", 1),
                            "misuse_type": "reshape_on_dict_list",
                            "description": f"Calling .reshape() on Python list variable '{caller.id}' without converting to numpy array",
                        })

            # b4: DictWriter with scalar/string row in writerow
            if cname.endswith("writerow") and len(node.args) > 0:
                arg = node.args[0]
                if isinstance(arg, ast.Name) and arg.id.lower() in ("row", "item"):
                    p = parent_map.get(id(node))
                    while p and not isinstance(p, ast.For):
                        p = parent_map.get(id(p))
                    if p and isinstance(p.iter, ast.Call) and (_get_call_name(p.iter) or "").endswith((".values", ".keys")):
                        misuses.append({
                            "call": "csv.DictWriter.writerow",
                            "line": getattr(node, "lineno", 1),
                            "misuse_type": "dictwriter_non_dict_row",
                            "description": "Passing scalar item from dict.values() to DictWriter.writerow() instead of a field-to-value dictionary mapping",
                        })

            # Pattern c: Using np.abs() where context computes vector magnitude / Euclidean norm / distance
            if cname in ("np.abs", "numpy.abs"):
                arg_is_vector = False
                if len(node.args) > 0:
                    arg = node.args[0]
                    if isinstance(arg, ast.Call) and (_get_call_name(arg) or "").startswith(("np.array", "numpy.array")):
                        arg_is_vector = True
                    elif isinstance(arg, (ast.List, ast.Tuple, ast.BinOp)):
                        arg_is_vector = True
                    elif isinstance(arg, ast.Name):
                        arg_name = arg.id.lower()
                        if var_types.get(arg.id) == "numpy_array" or any(k in arg_name for k in ("vector", "vec", "velocity", "coords", "points", "arr")):
                            arg_is_vector = True

                # Check if result is assigned to magnitude/norm/length/distance or passed to sum
                assigned_to_norm = False
                parent = parent_map.get(id(node))
                if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                    t_names = []
                    if isinstance(parent, ast.Assign):
                        for t in parent.targets:
                            if isinstance(t, ast.Name):
                                t_names.append(t.id.lower())
                    elif isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
                        t_names.append(parent.target.id.lower())
                    if any(any(k in tn for k in ("magnitude", "mag", "norm", "euclidean", "length", "dist", "speed", "distance")) for tn in t_names):
                        assigned_to_norm = True

                summed = False
                if isinstance(parent, ast.Call):
                    pname = _get_call_name(parent) or ""
                    if pname in ("np.sum", "numpy.sum", "sum"):
                        summed = True

                if assigned_to_norm or (arg_is_vector and (has_magnitude_func or summed or has_norm_text)):
                    misuses.append({
                        "call": cname,
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "numpy_abs_vector_norm_misuse",
                        "description": "Using np.abs() where vector magnitude/Euclidean norm (np.linalg.norm) is intended",
                    })

            # Check: os.listdir treated as mapping
            if cname == "os.listdir":
                parent = parent_map.get(id(node))
                if isinstance(parent, ast.Attribute) and parent.attr in ("keys", "values", "items"):
                    misuses.append({
                        "call": "os.listdir",
                        "line": getattr(node, "lineno", 1),
                        "misuse_type": "os_listdir_treated_as_mapping",
                        "description": f"os.listdir return value called with .{parent.attr}() as a dictionary mapping",
                    })

        # Pattern d: Subscripting loop variable when iterating over dictionary keys
        loop_var = None
        body_nodes = []
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name):
            loop_var = node.target.id if isinstance(node.target, ast.Name) else ""
            body_nodes = list(ast.walk(node))
        elif isinstance(node, ast.ListComp):
            for gen in node.generators:
                if isinstance(gen.iter, ast.Name) and isinstance(gen.target, ast.Name):
                    loop_var = gen.target.id
                    body_nodes = list(ast.walk(node.elt))

        if loop_var and body_nodes:
            for sub in body_nodes:
                if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == loop_var:
                    if isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
                        misuses.append({
                            "call": f"{loop_var}[{sub.slice.value}]",
                            "line": getattr(sub, "lineno", 1),
                            "misuse_type": "string_key_on_dict_key_iteration",
                            "description": f"Directly subscripting loop variable '{loop_var}' with string key '{sub.slice.value}' when iterating over a dict mapping",
                        })

        # Pattern e: Assigning result of inplace method call in pandas (evaluates to None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            val = node.value
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
                method_name = val.func.attr
                if method_name in ("drop", "dropna", "fillna", "reset_index", "sort_values", "rename"):
                    is_inplace_true = any(
                        kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                        for kw in val.keywords
                    )
                    if is_inplace_true:
                        caller = val.func.value
                        caller_name = caller.id if isinstance(caller, ast.Name) else "df"
                        misuses.append({
                            "call": f"{caller_name}.{method_name}(inplace=True)",
                            "line": getattr(node, "lineno", 1),
                            "misuse_type": "pandas_inplace_none_assignment",
                            "description": f"Assigning result of {method_name}(..., inplace=True) to variable, which evaluates to None in pandas",
                        })

    return misuses


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
        is_runtime_misuse = bool(exc_msg and any(p in exc_msg for p in (
            "has no attribute 'reshape'",     # list.reshape instead of np.array().reshape
            "has no attribute 'hexdigest'",   # bytes.hexdigest instead of hmac.hexdigest
            "has no attribute 'keys'",        # str.keys instead of json dict
            "has no attribute 'sort'",        # df.sort
            "has no attribute 'items'",       # list.items (os.listdir treated as dict)
        )))
        if len(ast_misuses) > 0 or is_runtime_misuse:
            category = "USAGE_SEMANTIC_MISUSE"
        else:
            category = "ATTRIBUTE_ERROR"
    elif exc_type == "TypeError":
        is_runtime_misuse = bool(exc_msg and any(p in exc_msg for p in (
            "not subscriptable",
            "string indices must be integers",
            "argument",
        )))
        if len(ast_misuses) > 0 or is_runtime_misuse:
            category = "USAGE_SEMANTIC_MISUSE"
        else:
            category = "TYPE_ERROR"
    elif exc_type in ("KeyError", "IndexError"):
        if len(ast_misuses) > 0:
            category = "USAGE_SEMANTIC_MISUSE"
        else:
            category = "KEY_INDEX_ERROR"
    elif exc_type == "ValueError":
        if len(ast_misuses) > 0:
            category = "USAGE_SEMANTIC_MISUSE"
        else:
            category = "VALUE_ERROR"
    elif exc_type == "AssertionError" or "AssertionError" in stderr:
        if len(ast_misuses) > 0:
            category = "USAGE_SEMANTIC_MISUSE"
        else:
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
