"""
repair.py -- Self-Healing AST Auto-Repair Engine for Usage-Semantic Hallucinations.

Automatically synthesizes AST transformations to repair detected Intent Misuses:
  1. Requests: Converts direct subscripting `resp['key']` -> `resp.json()['key']`
  2. Pandas: Replaces obsolete `df.sort(...)` -> `df.sort_values(...)`
  3. NumPy: Replaces `np.abs(v)` in vector norm contexts -> `np.linalg.norm(v)`
  4. Scikit-learn / NumPy: Wraps uncast lists in `np.array(x).reshape(...)`
  5. Hashlib: Corrects `.hexdigest()` called on bytes -> called on the hash object
  6. Dict / JSON iteration: Fixes string key iteration `for row in data:` -> `for row in data.values():`
"""

import ast
import difflib
from dataclasses import dataclass
from typing import Optional

from src.label_taxonomy import _get_call_name, check_code_for_misuse


@dataclass
class RepairResult:
    repaired: bool
    original_code: str
    repaired_code: str
    diff: str
    repaired_issues: list[dict]
    remaining_issues: list[dict]


class SemanticMisuseTransformer(ast.NodeTransformer):
    """
    Transforms AST nodes containing usage-semantic intent misuses into valid code.
    """
    def __init__(self):
        super().__init__()
        self.repaired_log = []
        self.var_types = {}
        self.hash_obj_map = {}  # maps digest_var -> hash_obj_name
        self.parent_map = {}

    def pre_analyze(self, tree: ast.AST):
        """Analyze assignments to record variable types and sources."""
        for p in ast.walk(tree):
            for child in ast.iter_child_nodes(p):
                self.parent_map[id(child)] = p

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
                    # Requests response
                    if any(val_call.startswith(p) for p in ("requests.get", "requests.post", "requests.put", "requests.delete", "requests.patch")):
                        self.var_types[t_name] = "requests_response"
                    elif t_name.lower() in ("response", "resp"):
                        self.var_types[t_name] = "requests_response"
                    # Hash object creation (e.g. hmac.new, hashlib.sha256)
                    elif any(k in val_call for k in ("hashlib.", "hmac.new")):
                        self.var_types[t_name] = "hash_object"
                    # Digest from hash object: digest = hash_obj.digest()
                    elif isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) and val.func.attr == "digest":
                        caller = val.func.value
                        caller_name = caller.id if isinstance(caller, ast.Name) else ""
                        self.var_types[t_name] = "bytes_digest"
                        if caller_name:
                            self.hash_obj_map[t_name] = caller_name
                    elif t_name.lower() == "digest":
                        self.var_types[t_name] = "bytes_digest"
                    # Python lists / sequences
                    elif isinstance(val, (ast.List, ast.ListComp)) or val_call in ("zip", "list"):
                        self.var_types[t_name] = "python_list"
                    # Dict lookups (values = data_dict[key])
                    elif isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and ("dict" in val.value.id.lower() or "data" in val.value.id.lower()):
                        self.var_types[t_name] = "dict_value_uncast"
                    elif t_name.lower() in ("values", "data_list", "val_list"):
                        self.var_types[t_name] = "dict_value_uncast"
                    # Pandas
                    elif any(val_call.startswith(p) for p in ("pd.read_", "pandas.read_", "pd.DataFrame", "pandas.DataFrame", "pd.Series")):
                        self.var_types[t_name] = "pandas_obj"
                    elif t_name.lower() in ("df", "dataframe", "sales_df") or t_name.lower().endswith(("_df", "_dataframe")):
                        self.var_types[t_name] = "pandas_obj"

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        val = node.value

        # 1. Requests: resp['key'] -> resp.json()['key']
        if isinstance(val, ast.Name):
            var_name = val.id
            if self.var_types.get(var_name) == "requests_response" or var_name.lower() in ("response", "resp"):
                json_call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id=var_name, ctx=ast.Load()), attr="json", ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )
                node.value = json_call
                self.repaired_log.append({
                    "type": "requests_subscript",
                    "line": getattr(node, "lineno", 1),
                    "action": f"Injected .json() call before subscripting on '{var_name}'"
                })
                return node

        # 2. Chained requests.get(...)['key'] -> requests.get(...).json()['key']
        elif isinstance(val, ast.Call):
            cname = _get_call_name(val) or ""
            if any(cname.startswith(p) for p in ("requests.get", "requests.post", "requests.put", "requests.delete", "requests.patch")):
                json_call = ast.Call(
                    func=ast.Attribute(value=val, attr="json", ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )
                node.value = json_call
                self.repaired_log.append({
                    "type": "requests_subscript",
                    "rule": "requests_subscript_without_json",
                    "line": getattr(node, "lineno", 1),
                    "action": f"Injected .json() call after '{cname}' before subscripting"
                })
                return node

        # 2b. Missing parentheses on .json: resp.json['key'] -> resp.json()['key']
        elif isinstance(val, ast.Attribute) and val.attr == "json":
            caller = val.value
            caller_name = caller.id if isinstance(caller, ast.Name) else ""
            if self.var_types.get(caller_name) == "requests_response" or caller_name.lower() in ("response", "resp"):
                json_call = ast.Call(
                    func=val,
                    args=[],
                    keywords=[]
                )
                node.value = json_call
                self.repaired_log.append({
                    "type": "requests_missing_parentheses_json",
                    "rule": "requests_missing_parentheses_json",
                    "line": getattr(node, "lineno", 1),
                    "action": f"Converted '{caller_name}.json' attribute access to method call '{caller_name}.json()'"
                })
                return node

        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        val = node.value
        if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
            method_name = val.func.attr
            if method_name in ("drop", "dropna", "fillna", "reset_index", "sort_values", "rename"):
                new_kws = [kw for kw in val.keywords if not (kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True)]
                if len(new_kws) != len(val.keywords):
                    val.keywords = new_kws
                    self.repaired_log.append({
                        "type": "pandas_inplace_none_assignment",
                        "rule": "pandas_inplace_none_assignment",
                        "line": getattr(node, "lineno", 1),
                        "action": f"Removed conflicting 'inplace=True' from assigned {method_name}() call so result is returned to variable"
                    })
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        cname = _get_call_name(node) or ""

        # 2c. Requests: resp.text() -> resp.text (property called as method)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "text":
            caller = node.func.value
            caller_name = caller.id if isinstance(caller, ast.Name) else ""
            if self.var_types.get(caller_name) == "requests_response" or caller_name.lower() in ("response", "resp", "r"):
                self.repaired_log.append({
                    "type": "requests_text_called_as_function",
                    "rule": "requests_text_called_as_function",
                    "line": getattr(node, "lineno", 1),
                    "action": f"Removed call parentheses from requests string property '{caller_name}.text'"
                })
                return node.func

        # 3b. Pandas: df.append(other) -> pd.concat([df, other])
        if isinstance(node.func, ast.Attribute) and node.func.attr == "append":
            caller = node.func.value
            caller_name = caller.id if isinstance(caller, ast.Name) else ""
            has_pandas_kwargs = any(k.arg in ("ignore_index", "verify_integrity", "sort", "axis") for k in node.keywords if k.arg)
            is_subscript_df = isinstance(caller, ast.Subscript) and (
                (isinstance(caller.value, ast.Name) and "df" in caller.value.id.lower())
                or self.var_types.get(getattr(caller.value, "id", "")) == "pandas_obj"
            )
            is_pandas = (
                self.var_types.get(caller_name) == "pandas_obj"
                or caller_name.lower() in ("df", "dataframe", "series", "s", "s1", "s2", "data_df")
                or caller_name.lower().endswith(("_df", "_dataframe"))
                or is_subscript_df
                or has_pandas_kwargs
            )
            if is_pandas and len(node.args) > 0:
                other_arg = node.args[0]
                concat_list = ast.List(elts=[caller, other_arg], ctx=ast.Load())
                new_kws = [kw for kw in node.keywords if kw.arg in ("ignore_index", "axis", "sort")]
                concat_call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="pd", ctx=ast.Load()), attr="concat", ctx=ast.Load()),
                    args=[concat_list],
                    keywords=new_kws
                )
                call_label = caller_name if caller_name else ("df[...]" if is_subscript_df else "obj")
                self.repaired_log.append({
                    "type": "pandas_removed_append",
                    "rule": "pandas_removed_append",
                    "line": getattr(node, "lineno", 1),
                    "action": f"Replaced removed '{call_label}.append()' with modern 'pd.concat([{call_label}, ...])'"
                })
                return concat_call

        # 3. Pandas: df.sort(...) -> df.sort_values(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "sort":
            caller = node.func.value
            caller_name = caller.id if isinstance(caller, ast.Name) else ""
            is_pandas = (
                self.var_types.get(caller_name) == "pandas_obj"
                or caller_name.lower() in ("df", "dataframe", "series", "s", "data_df")
                or caller_name.lower().endswith(("_df", "_dataframe"))
            )
            has_pandas_kwargs = any(k.arg in ("columns", "ascending", "inplace", "axis") for k in node.keywords if k.arg)
            has_positional_col = len(node.args) > 0 and isinstance(node.args[0], (ast.Constant, ast.List))
            if is_pandas or has_pandas_kwargs or has_positional_col:
                node.func.attr = "sort_values"
                self.repaired_log.append({
                    "type": "pandas_obsolete_sort",
                    "line": getattr(node, "lineno", 1),
                    "action": f"Replaced obsolete '{caller_name}.sort()' with '{caller_name}.sort_values()'"
                })
                return node

        # 4. Hashlib: digest.hexdigest() -> hash_object.hexdigest()
        if isinstance(node.func, ast.Attribute) and node.func.attr == "hexdigest":
            caller = node.func.value
            # Case 4a: hash_obj.digest().hexdigest() -> hash_obj.hexdigest()
            if isinstance(caller, ast.Call) and isinstance(caller.func, ast.Attribute) and caller.func.attr == "digest":
                hash_target = caller.func.value
                node.func.value = hash_target
                self.repaired_log.append({
                    "type": "hexdigest_on_bytes",
                    "line": getattr(node, "lineno", 1),
                    "action": "Bypassed intermediate .digest() call, calling .hexdigest() directly on the hash object"
                })
                return node
            # Case 4b: digest.hexdigest() where digest was assigned from hash_obj.digest()
            elif isinstance(caller, ast.Name) and caller.id in self.hash_obj_map:
                hash_obj_name = self.hash_obj_map[caller.id]
                node.func.value = ast.Name(id=hash_obj_name, ctx=ast.Load())
                self.repaired_log.append({
                    "type": "hexdigest_on_bytes",
                    "line": getattr(node, "lineno", 1),
                    "action": f"Redirected .hexdigest() from bytes '{caller.id}' to source hash object '{hash_obj_name}'"
                })
                return node

        # 5. Reshape: values.reshape(...) -> np.array(values).reshape(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "reshape":
            caller = node.func.value
            needs_array_wrap = False
            # e.g., [...].reshape(...)
            if isinstance(caller, (ast.List, ast.ListComp)):
                needs_array_wrap = True
            # e.g., data_dict[key].reshape(...)
            elif isinstance(caller, ast.Subscript):
                needs_array_wrap = True
            # e.g., values.reshape(...) where values was from dict lookup or list
            elif isinstance(caller, ast.Name):
                cid = caller.id.lower()
                if (
                    self.var_types.get(caller.id) in ("python_list", "dict_value_uncast")
                    or cid in ("pairs", "weights", "future_dates", "data", "items", "counts", "values", "data_list", "val_list", "l")
                ) and self.var_types.get(caller.id) != "numpy_array":
                    needs_array_wrap = True

            if needs_array_wrap:
                # Wrap caller with np.array(caller)
                array_call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="array", ctx=ast.Load()),
                    args=[caller],
                    keywords=[]
                )
                node.func.value = array_call
                self.repaired_log.append({
                    "type": "reshape_on_dict_list",
                    "rule": "reshape_on_dict_list",
                    "line": getattr(node, "lineno", 1),
                    "action": "Wrapped uncast Python list/sequence in 'np.array()' before calling .reshape()"
                })
                return node

        # 6. NumPy norm: np.abs(vector) in norm context -> np.linalg.norm(vector)
        if cname in ("np.abs", "numpy.abs") and len(node.args) > 0:
            arg = node.args[0]
            arg_name = arg.id.lower() if isinstance(arg, ast.Name) else ""
            is_vector_like = any(k in arg_name for k in ("vector", "vec", "velocity", "coords", "points")) or isinstance(arg, (ast.List, ast.Tuple, ast.BinOp))

            parent = self.parent_map.get(id(node))
            assigned_to_norm = False
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

            if is_vector_like or assigned_to_norm:
                norm_func = ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="linalg", ctx=ast.Load()),
                    attr="norm",
                    ctx=ast.Load()
                )
                node.func = norm_func
                self.repaired_log.append({
                    "type": "numpy_abs_vector_norm",
                    "rule": "numpy_abs_vector_norm_misuse",
                    "line": getattr(node, "lineno", 1),
                    "action": "Replaced 'np.abs()' with 'np.linalg.norm()' for vector magnitude computation"
                })
                return node

        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        # 7. Dict iteration with subscripting: for row in data: row['key'] -> for row in data.values():
        if isinstance(node.iter, ast.Name):
            loop_var = node.target.id if isinstance(node.target, ast.Name) else ""
            if loop_var:
                has_str_subscript = False
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == loop_var:
                        if isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
                            has_str_subscript = True
                            break
                if has_str_subscript:
                    # Change node.iter from data -> data.values()
                    iter_name = node.iter.id if isinstance(node.iter, ast.Name) else "dict"
                    values_call = ast.Call(
                        func=ast.Attribute(value=node.iter, attr="values", ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    )
                    node.iter = values_call
                    self.repaired_log.append({
                        "type": "string_key_on_dict_iteration",
                        "line": getattr(node, "lineno", 1),
                        "action": f"Converted iteration from '{iter_name}' to '{iter_name}.values()' so loop items are dictionary mappings"
                    })
                    return node

        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        self.generic_visit(node)
        for gen in node.generators:
            if isinstance(gen.iter, ast.Name) and isinstance(gen.target, ast.Name):
                loop_var = gen.target.id
                has_str_subscript = any(
                    isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == loop_var
                    and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str)
                    for sub in ast.walk(node.elt)
                )
                if has_str_subscript:
                    iter_name = gen.iter.id
                    values_call = ast.Call(
                        func=ast.Attribute(value=gen.iter, attr="values", ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    )
                    gen.iter = values_call
                    self.repaired_log.append({
                        "type": "string_key_on_dict_comprehension",
                        "line": getattr(node, "lineno", 1),
                        "action": f"Converted list comprehension iteration to '{iter_name}.values()'"
                    })
        return node


def repair_code(code: str) -> RepairResult:
    """
    Analyzes code for semantic intent misuses, applies AST transforms,
    and returns a structured RepairResult with colorable diff.
    """
    clean_code = code.split("```")[0]
    initial_misuses = check_code_for_misuse(clean_code)

    if not initial_misuses:
        return RepairResult(
            repaired=False,
            original_code=clean_code,
            repaired_code=clean_code,
            diff="",
            repaired_issues=[],
            remaining_issues=[]
        )

    try:
        tree = ast.parse(clean_code)
        wrapped = False
    except Exception:
        try:
            tree = ast.parse(f"def _wrap():\n{clean_code}")
            wrapped = True
        except Exception as e:
            return RepairResult(
                repaired=False,
                original_code=clean_code,
                repaired_code=clean_code,
                diff="",
                repaired_issues=[],
                remaining_issues=[{"error": f"Failed to parse AST: {e}"}]
            )

    transformer = SemanticMisuseTransformer()
    transformer.pre_analyze(tree)
    repaired_tree = transformer.visit(tree)
    ast.fix_missing_locations(repaired_tree)

    repaired_code = ast.unparse(repaired_tree)
    if wrapped and repaired_code.startswith("def _wrap():\n"):
        repaired_code = repaired_code[len("def _wrap():\n"):]
        # Remove 4 spaces of indentation
        repaired_lines = []
        for line in repaired_code.splitlines():
            if line.startswith("    "):
                repaired_lines.append(line[4:])
            else:
                repaired_lines.append(line)
        repaired_code = "\n".join(repaired_lines)

    # Compute unified diff
    orig_lines = clean_code.splitlines(keepends=True)
    repaired_lines = [l + "\n" for l in repaired_code.splitlines()]
    diff_gen = difflib.unified_diff(
        orig_lines,
        repaired_lines,
        fromfile="original.py",
        tofile="repaired.py",
        n=3
    )
    diff_str = "".join(diff_gen)

    remaining = check_code_for_misuse(repaired_code)

    return RepairResult(
        repaired=len(transformer.repaired_log) > 0,
        original_code=clean_code,
        repaired_code=repaired_code,
        diff=diff_str,
        repaired_issues=transformer.repaired_log,
        remaining_issues=remaining
    )
