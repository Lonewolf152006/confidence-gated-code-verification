"""
src/mutate.py -- AST-Driven Synthetic Intent-Misuse Bug Seeder.

Generates realistic usage-semantic bugs from clean reference solutions:
  1. Requests: Strips .json() from response subscripting -> resp['key']
  2. Requests: Drops parentheses from .json() -> resp.json['key']
  3. Pandas: Injects inplace=True into variable assignments -> df = df.dropna(inplace=True)
  4. Pandas: Mutates modern .sort_values(...) -> obsolete .sort(...)
  5. NumPy: Replaces np.linalg.norm(...) -> element-wise np.abs(...)
  6. NumPy: Drops np.array(...) wrapping before .reshape(...) -> val.reshape(...)
  7. Hashlib: Chains .hexdigest() onto bytes returned by .digest() -> h.digest().hexdigest()

Every seeded mutation is verified:
  - Detection: check_code_for_misuse() flags the violation
  - Self-Healing: repair_code() repairs the violation back to 0 issues
"""

import ast
import json
import difflib
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.label_taxonomy import check_code_for_misuse
from src.repair import repair_code


class MutationVisitor(ast.NodeTransformer):
    """
    AST transformer that mutates clean code by injecting semantic intent misuses.
    """
    def __init__(self, target_mutation: str):
        super().__init__()
        self.target_mutation = target_mutation
        self.mutated = False
        self.mutation_desc = ""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node

        # Target: Strip .json() from resp.json()['key'] -> resp['key']
        if self.target_mutation == "requests_subscript_without_json":
            val = node.value
            if (
                isinstance(val, ast.Call)
                and isinstance(val.func, ast.Attribute)
                and val.func.attr == "json"
            ):
                caller = val.func.value
                node.value = caller
                self.mutated = True
                self.mutation_desc = "Stripped .json() from response subscripting"
                return node

        # Target: Missing parentheses on resp.json: resp.json()['key'] -> resp.json['key']
        if self.target_mutation == "requests_missing_parentheses_json":
            val = node.value
            if (
                isinstance(val, ast.Call)
                and isinstance(val.func, ast.Attribute)
                and val.func.attr == "json"
            ):
                node.value = val.func
                self.mutated = True
                self.mutation_desc = "Dropped parentheses from .json() before subscripting"
                return node

        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node

        # Target: Mutate sort_values -> obsolete sort
        if self.target_mutation == "pandas_obsolete_sort":
            if isinstance(node.func, ast.Attribute) and node.func.attr == "sort_values":
                node.func.attr = "sort"
                self.mutated = True
                self.mutation_desc = "Replaced .sort_values() with obsolete .sort()"
                return node

        # Target: Mutate np.linalg.norm -> np.abs
        if self.target_mutation == "numpy_abs_vector_norm_misuse":
            if isinstance(node.func, ast.Attribute) and node.func.attr == "norm":
                caller = node.func.value
                caller_name = ""
                if isinstance(caller, ast.Attribute):
                    caller_name = caller.attr
                if caller_name == "linalg" or "norm" in node.func.attr:
                    node.func = ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr="abs",
                        ctx=ast.Load()
                    )
                    self.mutated = True
                    self.mutation_desc = "Replaced np.linalg.norm() with element-wise np.abs()"
                    return node

        # Target: Hashlib .hexdigest() chained on .digest()
        if self.target_mutation == "hexdigest_on_bytes":
            if isinstance(node.func, ast.Attribute) and node.func.attr == "hexdigest":
                caller = node.func.value
                if isinstance(caller, ast.Name):
                    digest_call = ast.Call(
                        func=ast.Attribute(value=caller, attr="digest", ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    )
                    node.func.value = digest_call
                    self.mutated = True
                    self.mutation_desc = "Chained .hexdigest() onto bytes from .digest()"
                    return node

        # Target: Unwrap np.array(x).reshape(...) -> x.reshape(...)
        if self.target_mutation == "reshape_on_dict_list":
            if isinstance(node.func, ast.Attribute) and node.func.attr == "reshape":
                caller = node.func.value
                if isinstance(caller, ast.Call) and isinstance(caller.func, ast.Attribute) and caller.func.attr in ("array", "asarray"):
                    if len(caller.args) > 0:
                        node.func.value = caller.args[0]
                        self.mutated = True
                        self.mutation_desc = "Removed np.array() cast before calling .reshape()"
                        return node

        # Target: Mutate pd.concat([df1, df2]) -> df1.append(df2)
        if self.target_mutation == "pandas_removed_append":
            cname = ""
            if isinstance(node.func, ast.Attribute):
                caller = node.func.value
                c_id = caller.id if isinstance(caller, ast.Name) else ""
                cname = f"{c_id}.{node.func.attr}" if c_id else node.func.attr
            if cname in ("pd.concat", "pandas.concat") and len(node.args) > 0:
                arg0 = node.args[0]
                if isinstance(arg0, (ast.List, ast.Tuple)) and len(arg0.elts) >= 2:
                    df1 = arg0.elts[0]
                    df2 = arg0.elts[1]
                    append_call = ast.Call(
                        func=ast.Attribute(value=df1, attr="append", ctx=ast.Load()),
                        args=[df2],
                        keywords=node.keywords
                    )
                    self.mutated = True
                    self.mutation_desc = "Replaced modern pd.concat() with obsolete df.append()"
                    return append_call

        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node

        # Target: Mutate response.text -> response.text()
        if self.target_mutation == "requests_text_called_as_function":
            if node.attr == "text" and isinstance(node.value, ast.Name):
                var_name = node.value.id.lower()
                if var_name in ("response", "resp", "r"):
                    call_node = ast.Call(func=node, args=[], keywords=[])
                    self.mutated = True
                    self.mutation_desc = f"Called requests property '{var_name}.text' as a function '{var_name}.text()'"
                    return call_node
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node

        # Target: Inject inplace=True into assignment: df = df.dropna() -> df = df.dropna(inplace=True)
        if self.target_mutation == "pandas_inplace_none_assignment":
            val = node.value
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
                method_name = val.func.attr
                if method_name in ("drop", "dropna", "fillna", "reset_index", "sort_values"):
                    has_inplace = any(k.arg == "inplace" for k in val.keywords)
                    if not has_inplace:
                        val.keywords.append(ast.keyword(arg="inplace", value=ast.Constant(value=True)))
                        self.mutated = True
                        self.mutation_desc = f"Injected inplace=True into assigned {method_name}() call"
                        return node

        return node


ALL_MUTATION_TYPES = [
    "pandas_inplace_none_assignment",
    "pandas_obsolete_sort",
    "pandas_removed_append",
    "requests_subscript_without_json",
    "requests_missing_parentheses_json",
    "requests_text_called_as_function",
    "numpy_abs_vector_norm_misuse",
    "reshape_on_dict_list",
    "hexdigest_on_bytes"
]


def seed_mutations(tasks_path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Scans reference tasks and generates synthetic intent-misuse mutations.
    """
    if not tasks_path.exists():
        raise FileNotFoundError(f"Reference tasks not found at {tasks_path}")

    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)

    mutated_samples = []

    for task in tasks:
        task_id = task.get("task_id", "")
        code = task.get("canonical_solution", "")
        if not code.strip():
            continue

        try:
            tree = ast.parse(code)
            wrapped = False
        except Exception:
            try:
                tree = ast.parse(f"def _wrap():\n{code}")
                wrapped = True
            except Exception:
                continue

        for mtype in ALL_MUTATION_TYPES:
            # Re-parse fresh AST
            if wrapped:
                current_tree = ast.parse(f"def _wrap():\n{code}")
            else:
                current_tree = ast.parse(code)

            mutator = MutationVisitor(target_mutation=mtype)
            mutated_tree = mutator.visit(current_tree)
            ast.fix_missing_locations(mutated_tree)

            if mutator.mutated:
                mutated_code = ast.unparse(mutated_tree)
                if wrapped and mutated_code.startswith("def _wrap():\n"):
                    lines = mutated_code[len("def _wrap():\n"):].splitlines()
                    mutated_code = "\n".join(l[4:] if l.startswith("    ") else l for l in lines)

                # Compute diff
                orig_lines = code.splitlines(keepends=True)
                mut_lines = [l + "\n" for l in mutated_code.splitlines()]
                diff_gen = difflib.unified_diff(
                    orig_lines, mut_lines, fromfile="reference.py", tofile="mutated.py", n=2
                )
                diff_str = "".join(diff_gen)

                # Verify detection
                misuses_detected = check_code_for_misuse(mutated_code)
                is_detected = any(m.get("misuse_type") == mtype for m in misuses_detected)

                # Verify auto-repair
                repair_res = repair_code(mutated_code)
                is_repaired = repair_res.repaired and len(repair_res.remaining_issues) == 0

                sample = {
                    "task_id": task_id,
                    "mutation_type": mtype,
                    "mutation_desc": mutator.mutation_desc,
                    "reference_code": code,
                    "mutated_code": mutated_code,
                    "diff": diff_str,
                    "detected_by_ast": is_detected,
                    "auto_repairable": is_repaired,
                    "remaining_issues_after_repair": len(repair_res.remaining_issues)
                }
                mutated_samples.append(sample)

                if max_samples and len(mutated_samples) >= max_samples:
                    return mutated_samples

    return mutated_samples


def main():
    parser = argparse.ArgumentParser(description="Synthetic Bug Seeder for Usage-Semantic Hallucinations")
    parser.add_argument("--tasks", type=str, default="data/raw_tasks/filtered_tasks.json", help="Path to reference tasks JSON")
    parser.add_argument("--output", type=str, default="data/labels/synthetic_mutations.json", help="Output path for mutated dataset")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of mutations to generate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tasks_path = project_root / args.tasks
    output_path = project_root / args.output

    print("=" * 70)
    print("  SYNTHETIC INTENT-MISUSE BUG SEEDER")
    print(f"  Source: {tasks_path}")
    print(f"  Target Output: {output_path}")
    print("=" * 70)

    samples = seed_mutations(tasks_path, max_samples=args.limit)
    print(f"\n[+] Generated {len(samples)} synthetic mutation instances across reference tasks.")

    # Breakdown by mutation type
    by_type = {}
    detected_count = 0
    repaired_count = 0

    for s in samples:
        mtype = s["mutation_type"]
        by_type[mtype] = by_type.get(mtype, 0) + 1
        if s["detected_by_ast"]:
            detected_count += 1
        if s["auto_repairable"]:
            repaired_count += 1

    print("\n[+] Breakdown by Mutation Pattern:")
    for mtype, cnt in by_type.items():
        print(f"    - {mtype:<36} : {cnt:>3} samples")

    det_rate = (detected_count / len(samples) * 100) if samples else 0
    rep_rate = (repaired_count / len(samples) * 100) if samples else 0

    print("\n[+] Benchmark Verification Metrics:")
    print(f"    - AST Detection Rate      : {detected_count}/{len(samples)} ({det_rate:.1f}%)")
    print(f"    - AST Self-Healing Rate   : {repaired_count}/{len(samples)} ({rep_rate:.1f}%)")

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"\n[OK] Successfully saved synthetic dataset to {output_path}\n")


if __name__ == "__main__":
    main()
