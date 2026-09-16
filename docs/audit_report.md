# Comprehensive Pilot Execution & Failure Audit Report

**Dataset Audited**: `data/labels/pilot_1.5B_labels.json`  
**Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
**Scale**: 100 completions across 10 BigCodeBench pilot tasks (10 samples/task)  
**Execution Environment**: Python 3.11 Subprocess Isolation with 10s Timeout  

> [!WARNING]
> **PILOT-SCALE RESULT (10 tasks) — NOT YET VALIDATED AT FULL SCALE, treat as provisional.**  
> This audit evaluates the initial 10-task pilot. With only 10 unique task groups, statistical power is low, and task-specific quirks (e.g., missing dependencies or missing prompt imports) disproportionately affect aggregate rates.

---

## 1. Overall Outcome Summary

| Metric | Count | Percentage | Description |
|---|:---:|:---:|---|
| **Total Evaluated** | 100 | 100.0% | Complete pilot cohort |
| **Passed Unit Tests** | 15 | 15.0% | Executed and passed all unittest assertions |
| **Failed Unit Tests** | 85 | 85.0% | Failed at execution or assertion stage |
| **Timed Out** | 0 | 0.0% | No executions exceeded the 10s timeout |

---

## 2. Fine-Grained Failure Taxonomy Breakdown

| Category | Count | Pct (%) | Primary Root Cause & Description |
|---|:---:|:---:|---|
| **`OTHER_RUNTIME_ERROR`** | 22 | 22.0% | Runtime exceptions during execution (16 `FileNotFoundError`, 4 `NameError`, 1 `FileExistsError`, 1 `Error`). Mainly caused by failure to `os.makedirs()` or unimported symbols like `BytesIO`. |
| **`ASSERTION_ERROR`** | 19 | 19.0% | Code executed cleanly without crashing, but failed unit test assertions (semantic / logic discrepancy). |
| **`SYNTAX_ERROR`** | 17 | 17.0% | Model hit max token budget mid-expression or generated unclosed triple-quotes / brackets. |
| **`NONE (PASS)`** | 15 | 15.0% | Completely correct implementation passing all test cases. |
| **`IMPORT_ERROR`** | 14 | 14.0% | `ModuleNotFoundError`: Missing environment package in execution environment (predominantly in BigCodeBench/81). |
| **`ATTRIBUTE_ERROR`** | 5 | 5.0% | Classic API hallucination: calling methods that do not exist on the returned object. |
| **`KEY_INDEX_ERROR`** | 5 | 5.0% | Container indexing mistakes: `KeyError` (3) and `IndexError` (2) on returned data structures. |
| **`VALUE_ERROR`** | 2 | 2.0% | Incompatible argument values passed to standard library calls. |
| **`TYPE_ERROR`** | 1 | 1.0% | Operating on incompatible types (e.g. passing int to function expecting path string). |

---

## 3. Detailed Exception Frequency (Failed Samples Only)

```text
AssertionError       : 19 (22.4%)
SyntaxError          : 17 (20.0%)
FileNotFoundError    : 16 (18.8%)
ModuleNotFoundError  : 14 (16.5%)
AttributeError       :  5 ( 5.9%)
NameError            :  4 ( 4.7%)
KeyError             :  3 ( 3.5%)
ValueError           :  2 ( 2.4%)
IndexError           :  2 ( 2.4%)
TypeError            :  1 ( 1.2%)
FileExistsError      :  1 ( 1.2%)
Error (generic)      :  1 ( 1.2%)
```

---

## 4. Per-Task Outcome Breakdown

| Task ID | Pass | Fail | Pass Rate | Top Failure Category | Root Cause Analysis |
|---|:---:|:---:|:---:|---|---|
| **BigCodeBench/1016** | 0 | 10 | 0.0% | `ASSERTION_ERROR` / `NameError` | Model frequently used `BytesIO` without importing from `io`, or failed pixel histogram assertions. |
| **BigCodeBench/151** | 1 | 9 | 10.0% | `SYNTAX_ERROR` | Long prompt causing model to exhaust token limit or emit markdown explanations mid-syntax. |
| **BigCodeBench/18** | 0 | 10 | 0.0% | `OTHER_RUNTIME_ERROR` | Shuffling files in directories that do not exist on Windows (`/fake/path`). |
| **BigCodeBench/339** | 3 | 7 | 30.0% | `SYNTAX_ERROR` | HMAC/URL encoding task; 3 solutions passed cleanly, 7 suffered syntax truncation. |
| **BigCodeBench/378** | 0 | 10 | 0.0% | `SYNTAX_ERROR` | CSV table drawing task; completions consistently ran out of tokens before closing defs. |
| **BigCodeBench/380** | 0 | 10 | 0.0% | `ASSERTION_ERROR` | Mathematical logic error in return formatting. |
| **BigCodeBench/390** | 9 | 1 | 90.0% | `SYNTAX_ERROR` | High model capability on this standard data manipulation task (9/10 passed). |
| **BigCodeBench/708** | 0 | 10 | 0.0% | `OTHER_RUNTIME_ERROR` | Base64/CSV extraction: 10/10 failed to call `os.makedirs(output_dir, exist_ok=True)` before opening file. |
| **BigCodeBench/743** | 2 | 8 | 20.0% | `ASSERTION_ERROR` | Mocked JSON prefix counting: logic errors when handling empty or non-existent directories. |
| **BigCodeBench/81** | 0 | 10 | 0.0% | `IMPORT_ERROR` | Environment missing specific auxiliary visualization package. |

---

## 5. Key Quality Insights & Limitations

1. **Environmental Noise in 1.5B Labels**:
   - `ModuleNotFoundError` (14%) and `NameError: BytesIO` (4%) represent missing symbols or missing packages, rather than subtle semantic misuse.
   - `SyntaxError` (17%) is an artifact of the 1.5B model hitting `MAX_NEW_TOKENS=512` or generating conversational text at the end of the completion.
2. **True Semantic Misuse Subset**:
   - `ASSERTION_ERROR` (19%), `ATTRIBUTE_ERROR` (5%), `KEY_INDEX_ERROR` (5%), `TYPE_ERROR` (1%), and `VALUE_ERROR` (2%) represent genuine functional and semantic errors (**32% of total samples**).
3. **Statistical Underpowering Warning**:
   - Because only 10 unique task prompts exist in the pilot, results are heavily clustered by prompt (e.g. Task 390 has 90% pass rate, while Task 708 has 0% due to directory creation).
   - Any classifier trained on this pilot data must be evaluated with `GroupKFold` strictly grouped by `task_id` and flagged as **provisional pilot findings** until scaled to 100+ tasks.
