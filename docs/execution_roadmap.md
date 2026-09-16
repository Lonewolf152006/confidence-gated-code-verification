# Execution Roadmap: Usage-Semantic Hallucination Detection

Rough plan assuming a semester (~14 weeks). Compress or stretch based on how much total time you actually have — confirm with your professor first, since this affects how ambitious the final scope should be.

---

## Phase 0 — Setup (Week 1)
- Set up Python 3.10+, `transformers`, `vllm`, Docker, HuggingFace account/access
- Load Qwen2.5-Coder-1.5B-Instruct on Colab/Kaggle, confirm you can generate text AND access per-token logprobs (`output_scores=True`)
- **Checkpoint**: you can generate one piece of code and see its token probabilities printed out

## Phase 1 — Dataset prep (Weeks 2–3)
- Download BigCodeBench, filter to tasks using pandas/requests/os/json
- Pick a **tiny pilot subset first — 5 to 10 tasks only**
- Set up the Docker-based sandbox execution harness (reuse BigCodeBench's own harness rather than writing from scratch)
- Manually run a couple of known-correct and known-broken code snippets through the sandbox to confirm PASS/FAIL labeling actually works
- **Checkpoint**: you can feed one piece of code into the sandbox and get a correct PASS/FAIL label back

## Phase 2 — Generation pipeline (Weeks 3–4)
- Write a script that, for each prompt, samples N=10 completions at temperature 0.7–1.0 and saves full text + per-token logprobs
- Run it only on the pilot subset first
- **Checkpoint**: 10 varied completions per pilot task, all saved with their logprobs

## Phase 3 — Signal extraction (Weeks 4–6)
- Use Python's `ast` module to find every API call in the generated code
- Align call positions to token indices using the tokenizer's offset mapping
- Compute the entropy signal at the tokens right after each call (EPR-style)
- Compute the self-consistency signal: extract "usage pattern" after each call across the 10 samples, measure how much they disagree
- Manually eyeball a handful of examples — do high-entropy/high-disagreement spots actually look like plausible mistakes to a human reader? This sanity check matters more than any metric at this stage.
- **Checkpoint**: two numeric features (entropy score, disagreement score) computed per generation

## Phase 4 — Full run (Weeks 6–7)
- Scale the whole pipeline (generation → sandbox execution → signal extraction) up to your full filtered task set (~100–150 tasks × 10 samples ≈ 1,000–1,500 generations)
- Check class balance (how many PASS vs FAIL) — if it's extremely lopsided (e.g. 95% pass), note this now, since it affects which metrics matter later (AUPRC over accuracy)
- **Checkpoint**: a full labeled dataset — (generation, entropy score, disagreement score, PASS/FAIL) for every sample

## Phase 5 — Baselines (Weeks 7–8)
- Run `pylint` and `mypy` on every generation, log whether either flags anything near the API call site
- Run your existence-checker (from the earlier idea) as a second baseline
- Implement a naive "whole-sequence consistency" baseline (ablation of your own method, without the decision-point focus)
- **Checkpoint**: baseline catch-rates recorded, ready to compare against

## Phase 6 — Classifier + evaluation (Weeks 8–9)
- Build the feature matrix, train a logistic regression / small MLP with 5-fold cross-validation
- Report F1, AUROC, AUPRC — compare against all baselines
- Run a bootstrap CI or McNemar's test to check if differences are statistically meaningful, not noise
- **Checkpoint**: a results table you could put directly in a paper

## Phase 7 — Analysis & writing (Weeks 9–12)
- Do error analysis: which failure types does the detector catch, which does it miss, are there patterns (e.g. only catches `requests`-related errors, not `pandas`)?
- Draft the paper: Intro → Related Work (your 19-paper survey slots in directly here) → Method → Results → Limitations
- Share a draft with your professor well before the deadline, leave room to revise

## Phase 8 — Buffer & polish (Weeks 12–14)
- Fix whatever broke, tighten the writing, format for your target venue
- If time remains: attempt the repair-suggestion stretch goal, or a second model (DeepSeek-Coder/StarCoder2) for a generalization check

---

## The one rule that saves you the most time
**Do not scale past the pilot subset (Phase 1–3) until the pipeline works end-to-end on 5–10 tasks.** If the sandbox mislabels things, or the signals look like noise even on a handful of hand-checked examples, you want to find that out in week 3, not week 8 after generating 1,500 samples on a broken pipeline.
