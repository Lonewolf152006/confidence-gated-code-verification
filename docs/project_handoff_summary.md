# Project Handoff Summary: Lightweight Usage-Semantic Hallucination Detection in LLM-Generated Code

This document captures the full arc of project planning so far — use it to continue in a new chat without losing context. Three companion files were also created earlier in this process (see Section 9).

---

## 1. How We Got Here (evolution of the idea)

1. Started with a general need: a 4th-year AI project that also produces a publishable research paper, with real-world usefulness and resume value.
2. Explored 8 candidate topics (XAI, federated learning, low-resource NLP, edge AI/TinyML, AI for agriculture, hallucination detection, multimodal accessibility, agentic AI evaluation) — picked **hallucination detection in LLMs**.
3. Built a literature survey of 21 papers (via a structured ChatGPT-extraction prompt), found an apparent gap: no lightweight detector tested on **code generation** specifically.
4. Verified this gap via live web search — found it was **already partly claimed** by recent 2025–2026 papers on fabricated-API detection. Had to reframe.
5. Narrowed to the real open gap: **"Intent Misuse"** — the model uses a *real* API but the *wrong* one for the task (e.g. `np.abs` instead of `vx.magnitude`) — which static tools (pylint/mypy) structurally cannot catch, since nothing is syntactically wrong.
6. Found the closest prior work, **Dr.Fix (Zhuo et al.)**, which already defines this exact taxonomy but detects/repairs it using **heavy, multi-stage LLM prompting** (GPT-4o, 32B/70B models) — expensive, not lightweight.
7. Landed on the final novelty: test whether **cheap, self-generated uncertainty/consistency signals** (no extra LLM calls) can predict Intent Misuse, and use a **small distilled specialist model** (not another giant LLM) to catch cases where those cheap signals fail — specifically the "confidently wrong" blind spot that self-examination structurally cannot see (tied back to a theory paper proving detection needs external negative-labeled signal, not just self-examination).
8. Worked out full dataset details (BigCodeBench), tech stack, and a 14-week execution roadmap.
9. Verified BigCodeBench's credibility/adoption and confirmed one more adjacent (not overlapping) paper on library-name hallucination sensitivity to prompts.

---

## 2. Final Project Definition

**Working title**: *"Confidence-Gated Detection of Usage-Semantic Hallucinations in LLM-Generated Code: A Lightweight Alternative to LLM-Based Verification"*

**Problem**: LLMs writing code sometimes use a real, syntactically valid API — but the wrong one for the task ("Intent Misuse," per Zhuo et al.'s taxonomy: Intent, Hallucination, Missing, Redundancy). This is common (20–30% of method-level misuses) and static analysis tools cannot detect it, since nothing is syntactically incorrect.

**Novelty**: Existing lightweight hallucination detectors (entropy, self-consistency — see Tier 2 papers below) have never been tested on this failure type. The one paper that does study Intent Misuse in LLM code (Dr.Fix) uses expensive multi-stage LLM prompting for both detection and repair. Nobody has tested whether cheap, self-generated signals can predict it — and there's a real theoretical reason to expect they might *not* fully work: self-uncertainty signals are examining the model's own confidence, and Intent Misuse may be a **"confidently wrong"** error (a well-learned but incorrect association), which entropy-based methods are known to sometimes miss (this blind-spot phenomenon is explicitly noted in one of the survey papers, Semantic Energy). This motivates the architecture below — using an *externally trained*, distilled specialist model for the cases cheap self-signals can't catch, rather than relying purely on self-examination.

**Honest scope note**: this is workshop/student-conference tier, not top-tier — and that's appropriate for the assignment. There's a real risk the cheap signals show near-zero correlation with Intent Misuse; if so, that's still a legitimate, citable negative result, given how well-motivated the question is.

---

## 3. Proposed Architecture: Confidence-Gated Verification Cascade

A triage-style pipeline (visualized earlier as a flowchart):

1. **Layer 0+1 (cheap checks)**: static existence verification (via `inspect`) + token-level entropy/self-consistency signals (from the generating model itself) — near-free.
2. **Router**: a confidence-gated decision — most cases are accepted here at low cost.
3. **Escalation path**: uncertain/suspicious cases are sent to a **small distilled specialist model** — trained on labels combining (a) Dr.Fix's categorized misuse labels and (b) objective execution-based pass/fail labels — specifically to catch the "confidently wrong" cases Layer 0+1 misses.
4. **Output**: labeled result with misuse category + confidence.

This is explicitly positioned as a **verification/router agent that could sit inside an agentic coding assistant's generate→execute→observe loop** — tying it to the current "small models for narrow agentic subtasks, escalate to heavy models only when needed" industry trend, which strengthens the real-world relevance/resume angle.

**Novel empirical claims this design enables**:
- A taxonomy-level breakdown of which signal (entropy vs. consistency vs. distilled model) catches which misuse category — directly testing the "entropy is blind to confidently-wrong errors" hypothesis.
- A cost/accuracy tradeoff curve vs. Dr.Fix's always-on heavy approach (e.g., "we recover X% of Dr.Fix's accuracy at a fraction of the LLM-call cost").

---

## 4. Dataset Plan

**Source**: BigCodeBench (Zhuo et al., 2406.15877, ICLR 2025) — 1,140 Python tasks, Apache 2.0, 139 libraries, execution-based unit tests with real assertions (not just crash-checks), multi-label library annotations (avg 2.8 libraries/task, so filtering by library is a metadata lookup).

**Critical implementation detail**: use **BigCodeBench-Complete** (fixed function signature + docstring), NOT BigCodeBench-Instruct — Complete keeps the signature fixed so the existing unit tests plug in directly with no extra engineering. Instruct format lets the model choose its own function name/signature, which breaks automatic testing.

**Known caveat**: a curation project (BigCodeBench+) found 768 of 1,140 original tasks needed at least one fix (927 major issues). Either use BigCodeBench+ or budget time to manually spot-check your filtered subset.

**Filtering**: filter to tasks using pandas/requests/os/json via the library-annotation metadata. Also consider **BigCodeBench-Hard** (~150 curated harder tasks) since easy tasks may produce too few FAIL examples for classifier training.

**Generation protocol**: N=10 completions per task at temperature 0.7–1.0 (needed for self-consistency signal diversity), using Qwen2.5-Coder via `vllm`. Store only top-K (e.g. top-10) logprobs per token, not the full vocabulary.

**Ground truth — two layers**:
- **Execution PASS/FAIL** (primary, objective): run in Docker sandbox against BigCodeBench's real assertion-based unit tests. This does catch Intent Misuse too (wrong function → wrong value → failed assertion), not just crashes.
- **Taxonomy category** (Intent/Hallucination/Missing/Redundancy, enrichment layer, only on FAILED cases): LLM-assisted first pass + manual spot-check, following Zhuo et al.'s categories. Be upfront this secondary label has some LLM-judge involvement, unlike the primary PASS/FAIL label.

**Splitting**: split by **task ID** (use `GroupKFold`), not by individual generation — 10 samples per task are correlated, random sample-level splits would leak data across train/test.

**Suggested schema per generation**:
```json
{
  "task_id": "...",
  "library_domain": "pandas | requests | os | json",
  "sample_index": 1-10,
  "generated_code": "...",
  "topk_logprobs": "[...per token, top-10 only]",
  "execution_label": "pass | fail",
  "taxonomy_label": "intent | hallucination | missing | redundancy | null"
}
```

**Secondary datasets worth checking**: DS-1000 (generalization check), Zhuo et al.'s Dr.Fix replication package (6,452 annotated misuse cases — verify availability/license before relying on it).

---

## 5. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Code-generating LLM | Qwen2.5-Coder-Instruct (1.5B to start, 7B later) | Open-weight (need real logprobs), Apache 2.0, widely used in the literature surveyed |
| Inference | HuggingFace `transformers` (full logit access) + `vllm` for speed at scale | vllm needed once generating 1,000+ samples |
| Quantization | `bitsandbytes` (4-bit) | Fit 7B model on free-tier Colab (T4, ~16GB) |
| Parsing | Python's built-in `ast` module | Extract API call sites |
| Token alignment | Tokenizer's `offset_mapping` | Map call sites to token indices for entropy computation |
| Sandbox | Docker (`--network none`, resource limits), reuse BigCodeBench's own execution harness | Safe, reproducible execution of untrusted generated code |
| Network mocking | `responses` library | Deterministic testing of `requests`-based tasks |
| Static baselines | `pylint`, `mypy`/`pyright` (subprocess) | Expected to catch ~0% of Intent Misuse — key evidence for your contribution |
| Classifier | `scikit-learn` (LogisticRegression or small MLP) | Matches the "lightweight" theme; small feature set (entropy, consistency, library, code length) |
| Validation | `GroupKFold` (grouped by task ID) | Avoid data leakage from correlated samples |
| Compute | Google Colab free tier (T4 GPU) or Kaggle (P100/T4 backup) | Sufficient for 1.5B–7B models with quantization |

---

## 6. Execution Roadmap (condensed — full version in separate file)

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Setup | 1 | Model running, logprobs accessible |
| 1. Dataset prep | 2–3 | Sandbox working on 5–10 pilot tasks |
| 2. Generation pipeline | 3–4 | 10 samples/task generated on pilot subset |
| 3. Signal extraction | 4–6 | Entropy + consistency features computed, manually sanity-checked |
| 4. Full run | 6–7 | ~1,000–1,500 labeled generations |
| 5. Baselines | 7–8 | pylint/mypy/existence-checker catch rates recorded |
| 6. Classifier + eval | 8–9 | F1/AUROC/AUPRC results table vs. baselines |
| 7. Analysis & writing | 9–12 | Draft paper, error analysis |
| 8. Buffer & polish | 12–14 | Final revisions, optional stretch goals |

**Golden rule**: do not scale past the 5–10 task pilot until the full pipeline (generation → sandbox → signal extraction) works end to end. Find out early if the sandbox mislabels things or the signals look like noise.

---

## 7. Full Reading List (tiered)

**Tier 1 — read in full, these anchor the paper:**
- Zhuo, Vu, Chim, Hu, Yu, Widyasari, Yusuf, Zhan, He, Paul et al. — *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions* (ICLR 2025, arXiv:2406.15877) — dataset paper
- Zhuo, He, Sun, Xing, Lo, Grundy, Du — *Identifying and Mitigating API Misuse in Large Language Models* (Dr.Fix, arXiv:2503.22821) — closest related work, defines the Intent/Hallucination/Missing/Redundancy taxonomy
- Karbasi, Montasser, Sous, Velegkas — *(Im)possibility of Automated Hallucination Detection in Large Language Models* (arXiv:2504.17004) — theoretical motivation for needing external negative-labeled signal
- Twist, Zhang, Harman, Yannakoudakis — *Library Hallucinations in LLM-Generated Code: A Risk Analysis Grounded in Developer Queries* (arXiv:2509.22202) — adjacent (prompt-sensitivity of library-name hallucination), good for motivation, not overlapping

**Tier 2 — core lightweight-technique baselines to adapt:**
- Moslonka, Randrianarivo, Garnier, Malherbe — *Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate* (EPR/WEPR, arXiv:2509.04492)
- Faujdar, Kadvani — *How Far Can You Get Without a GPU?* (arXiv:2606.29809) — your closest methodological template
- Nguyen, Gupta, Le — *Probabilities Are All You Need* (PRO, AAAI-26)
- Bouchard, Chauhan — *Uncertainty Quantification for Language Models* (uqlm, TMLR/arXiv:2504.19254)
- Lee, Yoshinaga — *Lightweight Hallucination Detection via Semantic Token-Group Logit Trajectories*
- Yang, Huang — *Logic Matters in Lightweight Hallucination Classification for RAG Systems* (ACL 2026) — closest structural template (small model + evidence-checking)

**Tier 3 — supporting evidence for the core hypothesis:**
- Ma, Pan, Liu, Chen, Zhou, Wang, Hu, Wu, Zhang, Wang — *Semantic Energy: Detecting LLM Hallucination Beyond Entropy* (arXiv:2508.14496) — first paper to note entropy-based methods can be "blind" in certain failure modes, supporting the confidently-wrong hypothesis

**Full original 21-paper survey** (remaining papers, background/context) is preserved in the separate literature survey file — see Section 9.

---

## 8. Key Honest Risks & Caveats (do not lose these)

- **Don't compete on fabricated-name detection** — pylint/mypy/existence-checkers already solve that; your value is in the harder, structurally-invisible-to-static-tools category (Intent Misuse).
- **Real risk of a null result**: if entropy/consistency show near-zero correlation with Intent Misuse, that's still a legitimate, citable negative result given how well-motivated the question is — plan the writeup to handle this honestly either way.
- **BigCodeBench has known spec-quality issues** (768/1,140 tasks needed fixes per the BigCodeBench+ curation project) — verify or use the corrected version.
- **Must use BigCodeBench-Complete, not Instruct** — Instruct format breaks automatic testing since the model can choose its own function signature.
- **Split by task ID, not by sample** — GroupKFold, to avoid leaking correlated same-task samples across train/test.
- **Expect class imbalance** (more PASS than FAIL) — report AUPRC, not just accuracy/F1.
- **Realistic venue tier**: workshop or student/national conference, not a top-tier venue — set expectations accordingly with your professor.
- **Timeline reality**: sandbox + dataset setup alone is realistically 2–3 weeks even reusing BigCodeBench's harness — this is a semester-scale project, not a quick build.

---

## 9. Files Already Created (in this session's outputs)

1. `literature_survey_hallucination_detection.md` — full 21-paper comparison table + original gap analysis
2. `tech_stack_hallucination_detection.md` — detailed stack spec (superseded in parts by Section 5 above, which has corrections)
3. `execution_roadmap.md` — full 14-week phased plan (Section 6 above is the condensed version)

If continuing in a new chat, upload this handoff file plus the three above for full context.

---

## 10. Open Next Steps

- Read Dr.Fix and BigCodeBench in full before writing anything else.
- Confirm total available timeframe with professor to calibrate the roadmap's pace.
- Build and validate the pilot pipeline (5–10 tasks) before any full-scale run.
- Decide whether to pursue the full confidence-gated cascade + distilled model, or scope down to just the cheap-signal benchmarking study if time is tight (still valid, just less novel — say so honestly in the paper).
