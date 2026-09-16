# Tech Stack: Usage-Semantic Hallucination Detection in Generated Code

---

## 1. Dataset

**Don't build the benchmark from scratch — adapt an existing one.** Building your own sandboxed test-harness dataset from zero is 3–4 weeks of pure infrastructure work. Instead:

| Option | What it gives you | Recommendation |
|---|---|---|
| **BigCodeBench** (bigcode-project, HF Hub + GitHub) | 1,140 tasks across 139 real libraries (pandas, requests, os, json included), each with unit tests already written and a working Docker-based execution harness | **Primary choice.** Filter down to the ~100–150 tasks that use pandas/requests/os/json — this concentrates your dataset on tasks actually prone to usage-semantic errors, not fabricated names. |
| **DS-1000** | 1,000 real Stack Overflow-derived data-science problems (numpy, pandas, sklearn, matplotlib, scipy) with execution-based correctness checks | Good secondary/cross-validation set if you want a second domain. |
| **Gorilla APIBench** | Specifically built to study API hallucination (originally for the Gorilla paper) | Worth citing as prior work — it's the closest existing dataset to your topic, though it targets a different failure type (API selection, not usage semantics). |

**Generation protocol**: for each filtered task, generate **N=10 completions per prompt** at temperature 0.7–1.0 (needed for the self-consistency signal — temperature 0 would give you identical samples every time). ~150 tasks × 10 samples = **1,500 generations total** — feasible to generate and execute in a day or two on free-tier compute, not weeks.

---

## 2. Code-generating LLM (the model whose hallucinations you're studying)

You need **local, open-weight model access** — not a closed API — because you need raw token logprobs for the entropy signal, and most hosted APIs (OpenAI, Anthropic) either don't expose full logprobs or cap them at top-20, which limits some entropy calculations.

| Model | Size | Why |
|---|---|---|
| **Qwen2.5-Coder-Instruct** | 1.5B or 7B | **Recommended primary.** Apache 2.0 license, strong code performance, widely used in the recent literature you already surveyed (several papers used Qwen family), good HF `transformers` support. Start with 1.5B for fast iteration, move to 7B once your pipeline works. |
| DeepSeek-Coder / StarCoder2 | 6.7B / 7B | Optional — only add if you have time for a cross-model generalization experiment (does the signal transfer across models?). Treat as stretch goal, not core scope. |

Run via **HuggingFace `transformers`** locally (full logit access, no API limits). For faster batched generation of your 1,500 samples, use **`vllm`** instead of raw `transformers` — it's substantially faster for repeated sampling and is now standard in this kind of work.

---

## 3. Parsing & signal extraction

| Component | Tool | Purpose |
|---|---|---|
| Code parsing | Python's built-in `ast` module | Walk the generated code's syntax tree, extract every function/method call site and its source line/column |
| Token-to-source alignment | Tokenizer's `offset_mapping` (HF tokenizers support this natively) | Map AST call-site positions back to token indices, so you know exactly which generated tokens correspond to "the tokens right after this API call" |
| Entropy signal | Custom, following EPR (Moslonka et al.) | At each token position after an API call, compute Shannon entropy over the top-K logits (`model.generate(output_scores=True)` gives you this directly) |
| Consistency signal | Custom, adapting HalluCodeDetector's idea | Across the 10 samples per prompt, extract the "usage pattern" following each API call (e.g. `.json()`, `[key]`, `.text` — represent as a short categorical string), then compute the entropy/diversity of these patterns across samples |

---

## 4. Sandboxed execution (ground-truth labeling)

**Do not `exec()` raw LLM output directly** — it's unsafe and non-reproducible (real network calls, filesystem writes).

| Component | Tool |
|---|---|
| Isolation | Docker container per run, `--network none`, memory/CPU/time limits — reuse **BigCodeBench's own open-source execution harness** rather than writing this from scratch |
| Network mocking | Python `responses` library (mocks `requests` calls with fixed fixture responses) so `requests`-based tasks are deterministic and don't need real network access |
| File I/O mocking | Small local fixture files (sample CSV/Excel) bundled with the task, not live external files |

---

## 5. Classifier

| Component | Tool | Why |
|---|---|---|
| Model | `scikit-learn` `LogisticRegression`, or a small `torch` MLP (2 hidden layers, <50k params) | Matches the "lightweight" theme of your whole survey — the Semantic Token-Group paper used exactly this pattern (small MLP on compressed features) |
| Features | Entropy score at decision points, consistency/diversity score, library-used (categorical), code length | Keep the feature set small and interpretable — this isn't a deep learning project |
| Validation | 5-fold cross-validation | Your dataset will be modest (hundreds, not thousands, of examples) — CV is necessary to get a stable estimate, and you should report the variance across folds honestly |

---

## 6. Baselines (required for a credible paper)

| Baseline | How to run it | Expected result |
|---|---|---|
| `pylint` | Subprocess call on generated code | Should catch ~0% of usage-semantic errors — this is your key evidence that static tools structurally can't see this failure class |
| `mypy` / `pyright` | Subprocess call | Same — near-0% catch rate expected (generated code usually lacks type annotations for these tools to leverage) |
| Your own fabricated-name checker (from the earlier idea) | `inspect`-based existence check | Should also score near-0% here, by construction — these are all real API names |
| Pure sampling-consistency (no decision-point weighting) | Ablation of your own method | Shows whether focusing specifically on post-call tokens adds value over naive whole-sequence consistency |

---

## 7. Evaluation

- **Metrics**: F1, AUROC, AUPRC (report AUPRC prominently — pass/fail labels will likely be imbalanced toward "pass")
- **Significance**: bootstrap confidence intervals or McNemar's test when comparing your method against baselines — reviewers will ask for this
- **Report negative results honestly** if the signal doesn't transfer well — a well-motivated negative result is still publishable and is far better than overstating a weak correlation

---

## 8. Repo structure (suggested)

```
project/
├── data/
│   ├── raw_tasks/          # filtered BigCodeBench subset
│   └── generations/        # LLM outputs + execution labels
├── src/
│   ├── generate.py         # vllm-based sampling
│   ├── parse.py            # ast-based call extraction
│   ├── signals.py          # entropy + consistency computation
│   ├── sandbox/             # Docker execution harness
│   ├── baselines.py        # pylint/mypy/existence-checker wrappers
│   └── classifier.py       # sklearn training + eval
├── notebooks/               # exploratory analysis
└── paper/                   # LaTeX (Overleaf), IEEE/ACM template
```

---

## 9. Honest compute & timeline check

- **Compute**: Google Colab free tier (T4 GPU, ~16GB) is enough for the 1.5B model comfortably, and the 7B with 4-bit quantization (`bitsandbytes`). Kaggle notebooks (free P100/T4) are a good backup if Colab session limits become annoying.
- **Timeline reality**: sandbox + dataset filtering setup is realistically 2–3 weeks even reusing BigCodeBench's harness. Signal extraction + classifier is faster once data exists, maybe 1–2 weeks. Budget accordingly — this is not a weekend project.
- **The biggest risk stays what I told you before**: if the entropy/consistency signals show near-zero correlation with actual execution failures, you don't have a positive result. Plan your writeup to be honest either way — a clear negative result on a well-motivated question is a legitimate paper; an overstated weak correlation is not.
