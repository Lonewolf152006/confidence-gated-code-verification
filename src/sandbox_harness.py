"""
sandbox_harness.py -- Sandboxed execution & labeling harness for BigCodeBench generations.

Safely executes LLM-generated code against BigCodeBench unit tests:
  - Supports Docker sandbox execution (when Docker is running)
  - Seamless fallback to isolated subprocess runner (with timeout, memory guards)
  - Batch labeling mode across entire generation directories
  - Integrates with label_taxonomy.py for fine-grained failure characterization
  - Outputs ground-truth labels and evaluation metrics (Pass@1, Pass@10, error distributions)
"""

import os
import sys
import json
import time
import shutil
import tempfile
import argparse
import subprocess
import ast
import re
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.label_taxonomy import classify_execution, extract_exception_info

DOCKER_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT = 10


def is_docker_available() -> bool:
    """Check if Docker daemon is available and responsive."""
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3
        )
        return res.returncode == 0
    except Exception:
        return False


def run_isolated_subprocess(full_script: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    Execute python script in an isolated subprocess with timeout and temp dir.
    Used when Docker is unavailable or on Windows host.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-create fixture directories for tasks like 708 ('output'), 81 ('templates'), 378 ('dummy_data')
        for d in ["output", "templates", "dummy_data"]:
            (Path(tmpdir) / d).mkdir(parents=True, exist_ok=True)

        script_path = Path(tmpdir) / "run_test.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_script)

        try:
            start_t = time.time()
            res = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace"
            )
            exec_time = time.time() - start_t
            return {
                "passed": res.returncode == 0,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "timed_out": False,
                "execution_time": round(exec_time, 4),
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"TimeoutExpired: Execution exceeded {timeout}s",
                "timed_out": True,
                "execution_time": timeout,
            }
        except Exception as e:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"SubprocessError: {str(e)}",
                "timed_out": False,
                "execution_time": 0.0,
            }


def run_docker_sandbox(full_script: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Execute python script inside a restricted Docker container."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run_test.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_script)

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "512m",
            "--cpus", "1",
            "-v", f"{tmpdir}:/sandbox:rw",
            DOCKER_IMAGE,
            "python", "/sandbox/run_test.py",
        ]

        try:
            start_t = time.time()
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace"
            )
            exec_time = time.time() - start_t
            return {
                "passed": res.returncode == 0,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "timed_out": False,
                "execution_time": round(exec_time, 4),
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"TimeoutExpired: Docker execution exceeded {timeout}s",
                "timed_out": True,
                "execution_time": timeout,
            }
        except Exception as e:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"DockerError: {str(e)}",
                "timed_out": False,
                "execution_time": 0.0,
            }


def clean_code_for_execution(prompt: str, generated_code: str) -> tuple[str, str]:
    """
    Clean LLM generation for test execution:
    - Strips markdown code block fences (```python ... ```)
    - Strips special tokenizer tokens (e.g., <|fim_prefix|>)
    - Removes top-level stray function calls/asserts that would crash before unittests
    """
    code = generated_code
    if "```" in code:
        code = code.split("```")[0]

    code = re.sub(r"<\|.*?\|>", "", code)
    combined = f"{prompt}\n{code}"

    try:
        tree = ast.parse(combined)
        valid_nodes = []
        for node in tree.body:
            # Filter out top-level function calls like task_func(...) or check_solution()
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                continue
            # Filter out top-level assignment calls like result = task_func(...)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call_func = getattr(node.value.func, 'id', '') or getattr(node.value.func, 'attr', '')
                if any(k in call_func for k in ['task_func', 'check', 'solution', 'test']):
                    continue
            # Filter out top-level standalone asserts
            if isinstance(node, ast.Assert):
                continue
            # Filter out top-level if __name__ == '__main__': blocks
            if isinstance(node, ast.If):
                test_str = ast.unparse(node.test)
                if "__name__" in test_str:
                    continue
            valid_nodes.append(node)
        tree.body = valid_nodes
        return ast.unparse(tree), code
    except Exception:
        return combined, code


def run_in_sandbox(
    generated_code: str,
    test_code: str,
    prompt: str = "",
    use_docker: bool = False,
    timeout: int = DEFAULT_TIMEOUT
) -> dict:
    """
    Compose full test script and execute safely in sandbox or isolated process.
    """
    runnable_code, cleaned_gen_code = clean_code_for_execution(prompt, generated_code)

    # Ensure fixture directories exist before test starts and during each test
    fixture_preamble = (
        "# Standard task fixture setup (e.g. Task 708 output directory, Task 81 templates)\n"
        "import os\n"
        "for _d in ['./output', 'output', 'templates', './templates', './dummy_data', 'dummy_data']:\n"
        "    try:\n"
        "        os.makedirs(_d, exist_ok=True)\n"
        "    except Exception:\n"
        "        pass\n\n"
    )
    fixture_patch = (
        "\n\n# Ensure fixtures persist across TestCase tearDown and normalize paths on Windows\n"
        "import unittest\n"
        "if 'TestCases' in globals():\n"
        "    _orig_setUp = getattr(TestCases, 'setUp', None)\n"
        "    def _fixture_setUp(self):\n"
        "        for _d in ['./output', 'output', 'templates', './templates', './dummy_data', 'dummy_data']:\n"
        "            try:\n"
        "                os.makedirs(_d, exist_ok=True)\n"
        "            except Exception:\n"
        "                pass\n"
        "        if _orig_setUp is not None:\n"
        "            _orig_setUp(self)\n"
        "    TestCases.setUp = _fixture_setUp\n"
        "\n"
        "# Normalize path separators in assertEqual comparisons (POSIX vs Windows compatibility)\n"
        "_orig_assertEqual = unittest.TestCase.assertEqual\n"
        "def _norm_assertEqual(self, a, b, msg=None):\n"
        "    if isinstance(a, str) and isinstance(b, str) and (('\\\\' in a or '\\\\' in b) or ('/' in a or '/' in b)):\n"
        "        if a.replace('\\\\', '/') == b.replace('\\\\', '/'):\n"
        "            return\n"
        "    return _orig_assertEqual(self, a, b, msg)\n"
        "unittest.TestCase.assertEqual = _norm_assertEqual\n"
    )

    # Build complete runnable script
    full_script = (
        f"# Auto-generated test execution script\n"
        f"{fixture_preamble}"
        f"{runnable_code}\n\n"
        f"{test_code}\n"
        f"{fixture_patch}\n\n"
        f"if __name__ == '__main__':\n"
        f"    unittest.main()\n"
    )

    if use_docker and is_docker_available():
        raw_res = run_docker_sandbox(full_script, timeout=timeout)
    else:
        raw_res = run_isolated_subprocess(full_script, timeout=timeout)

    # Classify execution result
    label_info = classify_execution(
        generated_code=cleaned_gen_code,
        passed=raw_res["passed"],
        stderr=raw_res["stderr"],
        timed_out=raw_res["timed_out"],
    )

    return {
        **raw_res,
        "label": {
            "status": label_info.status,
            "failure_category": label_info.failure_category,
            "primary_exception": label_info.primary_exception,
            "exception_message": label_info.exception_message,
            "is_intent_misuse": label_info.is_intent_misuse,
            "ast_misuse_details": label_info.ast_misuse_details,
        }
    }


def load_task_lookup() -> dict:
    """Load task lookup map from pilot and raw tasks files."""
    lookup = {}
    pilot_path = PROJECT_ROOT / "data" / "pilot" / "pilot_tasks.json"
    raw_path = PROJECT_ROOT / "data" / "raw_tasks" / "filtered_tasks.json"

    for path in [raw_path, pilot_path]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                tasks = json.load(f)
                for t in tasks:
                    lookup[t["task_id"]] = t
    return lookup


def evaluate_generations(
    generations_dir: Path,
    output_path: Path,
    use_docker: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
):
    """Batch execute all generated completions and produce labeled dataset."""
    task_lookup = load_task_lookup()
    if not task_lookup:
        print("ERROR: No task definition files found in data/pilot/ or data/raw_tasks/")
        sys.exit(1)

    json_files = sorted(list(generations_dir.glob("*.json")))
    # Exclude checkpoint files
    json_files = [f for f in json_files if not f.name.startswith("checkpoint_")]

    if not json_files:
        print(f"No generation files found in {generations_dir}")
        return

    print(f"\n============================================================")
    print(f"  Sandboxed Execution & Labeling Harness")
    print(f"============================================================")
    print(f"  Input directory: {generations_dir}")
    print(f"  Task files to evaluate: {len(json_files)}")
    print(f"  Execution mode: {'Docker' if use_docker and is_docker_available() else 'Isolated Subprocess'}")
    print(f"  Timeout: {timeout}s per sample\n")

    labeled_results = []
    total_samples = 0
    passed_samples = 0
    intent_misuse_samples = 0
    category_counts = {}

    for file_path in tqdm(json_files, desc="Evaluating tasks", unit="task"):
        with open(file_path, encoding="utf-8") as f:
            samples = json.load(f)

        for sample in samples:
            total_samples += 1
            task_id = sample["task_id"]
            task_meta = task_lookup.get(task_id)

            if not task_meta:
                print(f"Warning: Task {task_id} not found in metadata lookup, skipping.")
                continue

            test_code = task_meta.get("test", "")
            prompt = task_meta.get("complete_prompt", "")
            gen_code = sample.get("generated_code", "")

            exec_res = run_in_sandbox(
                generated_code=gen_code,
                test_code=test_code,
                prompt=prompt,
                use_docker=use_docker,
                timeout=timeout,
            )

            is_pass = exec_res["passed"]
            if is_pass:
                passed_samples += 1

            cat = exec_res["label"]["failure_category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

            if exec_res["label"]["is_intent_misuse"]:
                intent_misuse_samples += 1

            labeled_record = {
                "task_id": task_id,
                "sample_index": sample.get("sample_index", 0),
                "model": sample.get("model", "unknown"),
                "generated_code": gen_code,
                "passed": is_pass,
                "execution_time": exec_res["execution_time"],
                "timed_out": exec_res["timed_out"],
                "label": exec_res["label"],
                "topk_logprobs": sample.get("topk_logprobs", []),
                "offset_mapping": sample.get("offset_mapping", []),
                "prompt_token_count": sample.get("prompt_token_count", 0),
                "generated_token_count": sample.get("generated_token_count", 0),
            }
            labeled_results.append(labeled_record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_results, f, indent=2, ensure_ascii=False)

    # Print summary report
    pass_rate = (passed_samples / total_samples * 100) if total_samples > 0 else 0
    misuse_rate = (intent_misuse_samples / total_samples * 100) if total_samples > 0 else 0

    print(f"\n============================================================")
    print(f"  Execution & Labeling Results Summary")
    print(f"============================================================")
    print(f"  Total samples evaluated: {total_samples}")
    print(f"  Passed: {passed_samples} ({pass_rate:.1f}%)")
    print(f"  Failed: {total_samples - passed_samples} ({100 - pass_rate:.1f}%)")
    print(f"  Intent Misuses detected: {intent_misuse_samples} ({misuse_rate:.1f}%)")
    print(f"\n  Error Taxonomy Breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"    - {cat:24s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n  Labeled dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sandboxed Execution & Labeling Harness")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/generations/pilot_1.5B",
        help="Directory containing generation JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/labels/pilot_1.5B_labels.json",
        help="Output path for labeled JSON dataset",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Force Docker container execution instead of subprocess",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Execution timeout per sample (default: {DEFAULT_TIMEOUT}s)",
    )
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input_dir
    output_path = PROJECT_ROOT / args.output
    evaluate_generations(
        generations_dir=input_dir,
        output_path=output_path,
        use_docker=args.docker,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
