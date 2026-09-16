# Implementation Plan: Usage-Semantic Hallucination Detection — Full Build

## Goal

Build the complete **Confidence-Gated Verification Cascade** for detecting Intent Misuse hallucinations in LLM-generated code. This covers: project reorganization, fixing the known `parse_calls.py` gap, pilot validation, BigCodeBench dataset integration, code generation pipeline (local GPU), sandboxed execution, signal extraction, classifier training, and the distilled specialist model.

**Timeline**: ~12 weeks of active work across the full semester.
**Compute**: Local RTX 4070/4080/4090 (12+ GB VRAM) — can run Qwen2.5-Coder-7B at 4-bit quantization comfortably.
**Docker**: Available for sandbox execution.

---

## User Review Required

> [!IMPORTANT]
> **Model choice**: The plan uses Qwen2.5-Coder-1.5B for the pilot (fast iteration) and 7B for the full run. With your RTX 4070+, we can run both locally — no Colab needed. Does this work, or do you have a preference?

> [!IMPORTANT]
> **Scope decision**: The full cascade (cheap signals → router → distilled specialist) is ambitious for a semester. The plan is structured so Phases 1-6 produce a complete, publishable result (cheap-signal benchmarking study). Phases 7-8 add the distilled specialist and cascade — these are designed as additive extensions. If time gets tight, you can publish with just Phases 1-6 and still have a solid paper.

> [!WARNING]
> **BigCodeBench+ caveat**: 768 of 1,140 original BigCodeBench tasks had quality issues. The plan includes a task-quality verification step, but you should decide: use the original BigCodeBench and manually spot-check your filtered subset, or find the BigCodeBench+ corrected version if it's publicly available?

---

## Open Questions

> [!IMPORTANT]
> **Which RTX specifically?** RTX 4070 (12GB), 4080 (16GB), or 4090 (24GB)? This determines whether 7B+4bit fits comfortably or tightly.

> [!NOTE]
> **Python version**: The sandbox harness targets Python 3.11 (Docker image). What Python version are you running locally? The code uses `str | None` union syntax which requires 3.10+.

---

## Proposed Changes

The plan is split into 8 phases, each with a clear checkpoint. **Phases are designed to be sequential** — each builds on the previous one.

---

### Phase 1: Project Reorganization & Environment Setup (Week 1)

Restructure the flat file layout into the organized format from the tech stack document, set up dependencies, and verify everything still runs.

#### [NEW] Project directory structure

```
ai_project/
├── data/
│   ├── raw_tasks/              # BigCodeBench filtered subset (Phase 3)
│   ├── generations/            # LLM outputs + logprobs (Phase 4)
│   ├── labels/                 # execution PASS/FAIL + taxonomy labels (Phase 5)
│   └── pilot/                  # pilot hand-written examples (Phase 2)
├── src/
│   ├── parse_calls.py          # ← moved, then fixed (Phase 2)
│   ├── baselines.py            # ← moved
│   ├── classifier.py           # ← moved
│   ├── generate.py             # ← renamed from colab_generate.py, adapted for local GPU
│   ├── sandbox_harness.py      # ← moved
│   ├── signals.py              # [NEW] entropy + consistency feature extraction
│   ├── filter_tasks.py         # [NEW] BigCodeBench task filtering
│   └── label_taxonomy.py       # [NEW] LLM-assisted taxonomy labeling
├── tests/
│   └── test_parse_calls.py     # [NEW] unit tests for the parser
├── notebooks/                  # exploratory analysis
├── paper/                      # LaTeX drafts
├── configs/
│   └── default.yaml            # [NEW] centralized config (model, paths, hyperparams)
├── requirements.txt            # [NEW] proper requirements file
├── README.md                   # ← updated
└── docs/
    ├── project_handoff_summary.md
    ├── literature_survey_hallucination_detection.md
    ├── tech_stack_hallucination_detection.md
    └── execution_roadmap.md
```

#### [NEW] `requirements.txt`
Pin actual versions for reproducibility:
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
pylint>=3.0
mypy>=1.8
transformers>=4.40
accelerate>=0.28
bitsandbytes>=0.43
torch>=2.2
vllm>=0.4
requests>=2.31
responses>=0.25
pyyaml>=6.0
datasets>=2.18         # for loading BigCodeBench from HuggingFace Hub
docker>=7.0            # Python Docker SDK (cleaner than subprocess)
tqdm>=4.66
```

#### [NEW] `configs/default.yaml`
Centralized configuration so hyperparameters aren't scattered across files:
```yaml
model:
  name: "Qwen/Qwen2.5-Coder-1.5B-Instruct"   # bump to 7B for full run
  quantization: null                            # "4bit" for 7B
  max_new_tokens: 256

generation:
  n_samples_per_task: 10
  temperature: 0.8
  top_p: 0.95
  top_k_logprobs: 10

dataset:
  source: "bigcodebench"
  filter_libraries: ["pandas", "requests", "os", "json"]
  pilot_task_count: 10
  full_task_count: 150

sandbox:
  docker_image: "python:3.11-slim"
  timeout_seconds: 30
  memory_limit: "512m"
  cpus: 1

classifier:
  n_splits: 5
  model: "logistic_regression"    # or "mlp"
  class_weight: "balanced"
```

---

### Phase 2: Fix `parse_calls.py` + Pilot Validation (Weeks 1-2)

This is the **critical pre-scaling step** the README warns about. Fix the known variable-tracking gap, create diverse test cases, and validate that baselines truly miss Intent Misuse.

#### [MODIFY] [parse_calls.py](file:///c:/Users/student/Downloads/ai_project/parse_calls.py)

**The variable-tracking fix**: Add a lightweight def-use chain that:
1. When a call's result is assigned to a name (e.g., `resp = requests.get(url)`), find where that name is next used within the same function
2. Treat *that* usage as the "decision point" too, not just same-line chained usage
3. Implementation: simple forward scan within the same function body — no full data-flow analysis needed

New logic:
```python
def _track_variable_usage(tree: ast.AST, assign_target: str, assign_lineno: int) -> str | None:
    """
    After `resp = requests.get(url)` on line N, find the next usage of `resp`
    in the same function body and return what happens to it.
    E.g. `resp.json()` → ".json()", `resp['key']` → "[subscript]"
    """
    # Forward scan: find next Name node matching `assign_target` after assign_lineno
    # Check its parent to determine usage pattern
```

This directly addresses the gap documented in the README — `resp = requests.get(url)` followed by `resp['results']` two lines later will now be caught.

#### [NEW] `data/pilot/intent_misuse_examples.py`

10 diverse hand-written Intent Misuse examples covering different patterns:
1. Dict subscript on Response object (`resp['results']` instead of `resp.json()['results']`)
2. Wrong method chain (`df.sort()` instead of `df.sort_values()`)
3. Wrong parameter assumptions (`pd.read_csv(path, header=True)` — `header` takes int, not bool)
4. Wrong return type assumption (`os.listdir()` treated as dict)
5. String method on bytes (`resp.content.split(',')` instead of `resp.text.split(',')`)
6. Wrong pandas aggregation (`df.count()` instead of `df.sum()`)
7. Missing `.json()` call (`data = requests.post(url).text` then `json.loads(data)` works, but `data = requests.post(url)` then `data['key']` doesn't)
8. Wrong numpy operation (`np.abs()` for vector magnitude instead of `np.linalg.norm()`)
9. Wrong file mode (`open(path, 'r')` for binary file then `.read()` expecting bytes)
10. `os.path.join()` result treated as a file object instead of string

#### [MODIFY] [baselines.py](file:///c:/Users/student/Downloads/ai_project/baselines.py)

Extend to run all 10 pilot examples through pylint + mypy, producing a summary table showing catch rates. Expected: pylint/mypy catch ≈ 0% of these — this is the core thesis validation.

#### [NEW] `tests/test_parse_calls.py`

Unit tests verifying:
- Direct-chain misuse is detected (existing behavior)
- Assign-then-misuse-later is detected (new behavior)
- Correct code doesn't produce false positives
- All 10 pilot examples are parsed correctly

---

### Phase 3: BigCodeBench Dataset Integration (Week 2-3)

Download and filter BigCodeBench to your target subset.

#### [NEW] `src/filter_tasks.py`

- Load BigCodeBench-Complete from HuggingFace Hub (`datasets` library)
- Filter tasks by library annotations → pandas, requests, os, json
- Extract: task_id, prompt (docstring + function signature), reference solution, unit test code
- Save filtered subset to `data/raw_tasks/`
- Also check BigCodeBench-Hard overlap — flag harder tasks
- Output statistics: total tasks found, library distribution, test complexity

#### [NEW] `data/raw_tasks/filtered_tasks.json`

Schema per task:
```json
{
  "task_id": "BigCodeBench/123",
  "prompt": "def func(...):\n    \"\"\"docstring\"\"\"",
  "reference_solution": "...",
  "test_code": "...",
  "libraries_used": ["pandas", "requests"],
  "is_hard": false
}
```

---

### Phase 4: Local Code Generation Pipeline (Weeks 3-4)

Adapt `colab_generate.py` for local GPU execution with your RTX 4070+.

#### [MODIFY] [generate.py](file:///c:/Users/student/Downloads/ai_project/colab_generate.py) (renamed from `colab_generate.py`)

Major changes:
- Load model locally with `bitsandbytes` 4-bit quantization (for 7B) or fp16 (for 1.5B)
- Add `offset_mapping` capture alongside logprobs — needed for token-to-source alignment in Phase 6
- Add checkpoint/resume support (save after each task, so crashes don't lose progress)
- Add progress tracking with `tqdm`
- Load tasks from `data/raw_tasks/filtered_tasks.json`
- Save outputs to `data/generations/`
- **Pilot mode**: first run on only 5-10 tasks to validate, before scaling

Output schema per generation:
```json
{
  "task_id": "BigCodeBench/123",
  "sample_index": 0,
  "generated_code": "...",
  "topk_logprobs": [{"tokens": [...], "probs": [...]}],
  "offset_mapping": [[0, 3], [3, 7], ...]
}
```

---

### Phase 5: Sandboxed Execution & Labeling (Weeks 4-5)

Run generated code against BigCodeBench's unit tests in Docker to get PASS/FAIL labels.

#### [MODIFY] [sandbox_harness.py](file:///c:/Users/student/Downloads/ai_project/sandbox_harness.py)

Enhancements:
- Use the Python Docker SDK (`docker` package) instead of subprocess — cleaner error handling
- Add `pip install` step inside container for tasks needing specific libraries (pandas, requests, etc.)
- Add batch execution mode: process all generations in `data/generations/`
- Save results to `data/labels/execution_labels.json`
- Add retry logic for flaky Docker failures
- Increase timeout to 30s (some pandas tasks are slow)

#### [NEW] `src/label_taxonomy.py`

For FAILED cases only — LLM-assisted taxonomy labeling:
- Feed the generated code + reference solution + test output to a local LLM (or use the same Qwen model)
- Classify into: Intent / Hallucination / Missing / Redundancy (Dr.Fix taxonomy)
- Save to `data/labels/taxonomy_labels.json`
- Flag low-confidence labels for manual review

---

### Phase 6: Signal Extraction & Classifier (Weeks 5-8)

This is the core scientific contribution — extracting the entropy/consistency signals and testing whether they predict Intent Misuse.

#### [NEW] `src/signals.py`

Two signal types:

**1. Token-level entropy at decision points (EPR-style)**:
- For each API call site found by `parse_calls.py`, use `offset_mapping` to find the corresponding token indices
- Compute Shannon entropy over the top-K logprobs at tokens immediately *after* the call (the "decision point")
- Aggregate: mean entropy, max entropy, entropy variance at decision points vs. non-decision-point tokens

**2. Self-consistency / usage-pattern diversity**:
- Across the 10 samples per task, extract the "usage pattern" after each API call (using `parse_calls.py`'s `consumed_by` field)
- Compute pattern entropy: if all 10 samples do `.json()` → low diversity (confident). If 6 do `.json()` and 4 do `['key']` → high diversity (uncertain)
- This is the signal that should distinguish "confidently wrong" from "uncertainly wrong"

**Output feature vector per generation**:
```python
{
    "task_id": str,
    "sample_index": int,
    "mean_decision_entropy": float,
    "max_decision_entropy": float,
    "entropy_ratio": float,          # decision-point entropy / overall entropy
    "usage_pattern_diversity": float, # across the 10 samples
    "n_api_calls": int,
    "code_length": int,
    "library_domain": str,           # categorical
    "execution_label": int,          # 0=pass, 1=fail
    "taxonomy_label": str | None,    # only for failures
}
```

#### [MODIFY] [classifier.py](file:///c:/Users/student/Downloads/ai_project/classifier.py)

Evolve from synthetic data to real features:
- Load real feature matrix from `signals.py` output
- Keep GroupKFold by task_id
- Add feature importance analysis (which signal matters most?)
- Add ablation: entropy-only vs. consistency-only vs. both
- Add the "naive whole-sequence consistency" baseline (ablation)
- Generate publication-ready results table + plots
- Bootstrap confidence intervals

#### [MODIFY] [baselines.py](file:///c:/Users/student/Downloads/ai_project/baselines.py)

Scale to full dataset:
- Run pylint/mypy on all generated code
- Record per-sample: did the baseline flag anything near the call site?
- Compare catch rates vs. classifier, broken down by taxonomy category

---

### Phase 7: Distilled Specialist Model (Weeks 8-10)

The "escalation path" in the cascade — a small model specifically trained to catch cases the cheap signals miss.

#### [NEW] `src/specialist_model.py`

- Architecture: small MLP or fine-tuned CodeBERT/GraphCodeBERT encoder
- Input features: code embedding (from a frozen small code encoder) + the cheap signal features + AST structural features
- Training data: focus on "confidently wrong" cases — failures where entropy was *low* (the blind spot)
- Training: standard binary classification with focal loss (addresses class imbalance)
- Evaluation: how many additional failures does it catch that cheap signals missed?

#### [NEW] `src/cascade.py`

The full confidence-gated pipeline:
1. Compute cheap signals (entropy + consistency)
2. If confidence is high (entropy below threshold) → ACCEPT
3. If confidence is low → escalate to specialist model
4. Report: accuracy vs. cost tradeoff curve (% of samples needing escalation vs. overall accuracy)
5. Compare against Dr.Fix's always-heavy approach

---

### Phase 8: Analysis, Visualization & Paper (Weeks 10-14)

#### [NEW] `notebooks/results_analysis.ipynb`

- Taxonomy-level breakdown: which signal catches which misuse category
- Cost/accuracy tradeoff curve vs. Dr.Fix
- Feature importance plots
- Error analysis: what the cascade still misses and why
- Confidence calibration plots

#### [NEW] `paper/` directory

- IEEE/ACM template LaTeX structure
- Figures directory with publication-ready plots
- Results tables generated from `classifier.py` output

---

## Verification Plan

### Automated Tests
Each phase has a verification checkpoint:

```bash
# Phase 1: environment works
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"

# Phase 2: parser fix validated
python -m pytest tests/test_parse_calls.py -v
python src/baselines.py  # confirm pylint/mypy still miss Intent Misuse

# Phase 3: dataset loaded
python src/filter_tasks.py  # should print task count + library distribution

# Phase 4: generation works (pilot)
python src/generate.py --pilot  # 5-10 tasks only

# Phase 5: sandbox works
python src/sandbox_harness.py --pilot  # verify PASS/FAIL on pilot data

# Phase 6: signals + classifier
python src/signals.py  # extract features
python src/classifier.py  # train + evaluate, print AUROC/AUPRC/F1

# Phase 7: specialist + cascade
python src/specialist_model.py  # train specialist
python src/cascade.py  # full pipeline evaluation
```

### Manual Verification
- **Phase 2**: Manually inspect the 10 pilot Intent Misuse examples to verify `parse_calls.py` catches the assign-then-misuse pattern
- **Phase 4**: Eyeball a few generated code samples — do they look like realistic code?
- **Phase 5**: Spot-check 10-20 PASS/FAIL labels against the generated code — are they correct?
- **Phase 6**: Sanity-check: do high-entropy spots actually correspond to plausible mistakes when you read the code?
