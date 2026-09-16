# Literature Survey: Lightweight Hallucination Detection in LLMs
### Working toward: A lightweight hallucination detector for LLM-generated code

---

## 1. Full Paper Table

| # | Title (short) | Authors | Year/Venue | Type | Access | Lightweight? | Domain | Key Result | Relevance |
|---|---|---|---|---|---|---|---|---|---|
| 1 | EPR/WEPR (token entropy) | Moslonka et al. | 2026, arXiv | Detector | Black-box (top-K logprobs) | ✅ Yes, single-pass | QA / RAG | Entropy production rate flags high-uncertainty tokens | **Core baseline** |
| 2 | Agentic RAG Faithfulness | Papageorgiou et al. | 2025, MDPI BDCC | Eval framework | Heavy (LLM judges) | ❌ No | e-Governance RAG | Statement-level faithfulness via GPT-4.1/Claude/Gemini judges | Contrast point (what you're avoiding) |
| 3 | Narrative Consistency Survey | Park et al. | 2026, IEEE Access | Survey | N/A | N/A | Story generation | Taxonomy of 90 papers on narrative consistency | Background only |
| 4 | Systematic Review of Hallucinations | Woesle et al. | 2025, IEEE Access | PRISMA review | N/A | N/A | Broad (QA, multimodal, etc.) | 125 papers mapped; QA/multimodal ~48% of coverage | **Gap evidence** — shows domain imbalance |
| 5 | Semantic Energy | Ma et al. | 2025, arXiv | Detector | White-box (logits) | Moderate | General | Boltzmann-style energy fixes blind spots of semantic entropy | Comparison method |
| 6 | How Far Without a GPU? | Faujdar & Kadvani | 2026, arXiv | Benchmark | Black-box | ✅✅ CPU-only | QA, dialogue, summarization | Ensemble F1=0.792 (QA); near-random on summarization (AUC 0.47–0.57) | **Direct template for your methodology** |
| 7 | MEGA-RAG | Xu et al. | 2025, Frontiers Public Health | Mitigation | Heavy (multi-retrieval) | ❌ No | Public-health QA | 40%+ hallucination reduction, F1=0.79 | Contrast (heavy approach) |
| 8 | (Im)possibility of Automated Detection | Karbasi et al. | 2025, arXiv | Theory | N/A | N/A | Any generation task | Proves negative examples are *necessary* for learnability | **Theoretical anchor/motivation** |
| 9 | UQ Suite (uqlm) | Bouchard & Chauhan | 2025, TMLR | Toolkit | Mixed (black+white+judge) | Ensemble-tunable | Closed-book QA | Tunable ensemble beats individual scorers | Candidate baseline to adapt |
| 10 | Quantized LLM Internal Detection | Liu & Xu | 2026, Electronics | Detector | White-box (activations) | Lightweight if internal access | Edge/quantized (Qwen, Llama) | Best separability at 50–70% layer depth | Useful for on-device angle |
| 11 | STaRQ-Agent | Jiang et al. | 2026, IEEE Access | Multi-agent mitigation | LLM-based | Moderate/heavy | KGQA | +5.1%/+4.3%/+0.7% over baselines (few-shot) | Background |
| 12 | Routing Interpretability Probes | Javadov | 2026, arXiv | Interpretability study | White-box | N/A | Mechanistic interpretability | Routing mass ≠ causal importance (cautionary) | Background/caution |
| 13 | Semantic Token-Group Logit Trajectories | Lee & Yoshinaga | 2026, workshop | Detector | White-box (logits) | ✅ Very — 4096→192 features | Binary truthfulness | Small MLP ≈ matches heavy probes (SAPLMA) | Candidate technique |
| 14 | HALLUCITECHECKER | Sakai et al. | 2026, arXiv | Toolkit | Heuristic pipeline | ✅✅ CPU, offline, seconds | Citation verification | Practical, specialized tool | Shows CPU-only is viable for a narrow domain — precedent for your code-domain angle |
| 15 | Logic Matters | Yang & Huang | 2026, ACL | Detector | Small NLI + graph clustering | ✅ Very — 0.5B, 85ms latency | Multi-hop RAG | 82.4% acc (0.5B) beats sub-1B baselines by ~30%; 85.6% acc (1.5B) beats 11B TrueTeacher | **Strongest methodological template** |
| 16 | Coagulant Dosage Prediction | Li et al. | 2026, Water (MDPI) | Case study | Black-box LLM application | Moderate | Tabular regression, water treatment | Rationale-pattern hallucination audit in a real deployment | Example of domain-specific hallucination audit design |
| 17 | Mechanistic Interpretability Survey | Naseem | 2026, arXiv | Survey | N/A | N/A | Interpretability/alignment | Reviews probing, logit-lens, circuits | Background |
| 18 | SinkProbe (attention sinks) | Binkowski et al. | 2026, ICML | Detector | White-box (attention maps) | Probe is lightweight, needs internals | General (QA, summarization) | Compact probe on attention-sink features; claims SOTA | Candidate technique |
| 19 | HaMI (Multiple-Instance Learning) | Niu et al. | 2025, NeurIPS | Detector | White-box (internal reps) | Moderate/heavy | General | Adaptive token selection via MIL; outperforms SOTA on 4 benchmarks | Candidate technique (if internals available) |
| 20 | PRO (Probability-Only) | Nguyen et al. | 2026, AAAI | Detector | Black-box (top-K probs) | ✅✅ Very lightweight, training-free | Free-form QA | Outperforms expensive semantic-entropy baselines | **Strong candidate baseline** |
| 21 | Adaptive Bayesian Semantic Entropy | Sun et al. | 2026, AAAI | Detector | Black-box, multi-sample | Moderate (adaptive, fewer samples) | QA | ~50% fewer samples needed; AUROC +12.6% vs fixed-budget | Candidate technique |

*(Excluded: a mismatched "Osiris FHE accelerator" paper — unrelated to hallucination detection, likely a file mix-up; and one duplicate submission of Paper 6.)*

---

## 2. Categorized View

**Lightweight, black-box (API-only, no model internals needed):**
Papers 1, 6, 9 (partially), 14, 20, 21 — this is your most directly comparable baseline family.

**Lightweight, white-box (needs logits/activations, but small compute):**
Papers 5, 10, 13, 18, 19 — usable only if you have local model access (e.g., an open-weight model like Llama or Qwen run locally), not for closed APIs like GPT-4.

**Heavy / mitigation-focused (not what you're building, but useful contrast):**
Papers 2, 7, 11.

**Surveys / theory (background and motivation, not techniques to adapt):**
Papers 3, 4, 8, 12, 17.

**Domain-specific detectors (precedent for narrow-domain lightweight tools):**
Papers 14 (citations), 15 (multi-hop RAG), 16 (tabular/water treatment).

---

## 3. The Research Gap

Across all 19 valid papers, every lightweight or CPU-feasible hallucination detector targets **natural-language tasks**: QA (1, 6, 9, 20, 21), dialogue (6), summarization (6), multi-hop RAG (15), citations (14), general/binary truthfulness (13, 18, 19). Paper 4's own systematic review of 125 papers confirms this imbalance at the field level — QA and multimodal tasks dominate coverage.

**Not one paper in this set addresses hallucination detection in LLM-generated code** — e.g., fabricated library/API functions, invented parameters, functions that don't exist in the stated version of a library, or logically incorrect code that "looks" syntactically confident. This is a genuine, real-world problem (developers using Copilot/Claude/ChatGPT for code routinely encounter fabricated APIs) and it sits outside all 19 papers you've surveyed.

---

## 4. Proposed Project

**Working title:**
*"Benchmarking Lightweight Hallucination Detectors for LLM-Generated Code: A CPU-Feasible Approach to Detecting Fabricated APIs and Library Calls"*

**Problem statement:**
Existing lightweight/CPU-feasible hallucination detectors (entropy-based, NLI-based, probability-based) have been validated on QA, dialogue, and summarization — but never systematically tested on code generation, where hallucinations take a structurally different form (fabricated identifiers rather than false claims about the world).

**Methodology (mirrors Paper 6's structure, applied to a new domain):**
1. **Build a labeled dataset**: generate code completions from an open LLM (e.g., via a local model or API) for a set of programming prompts, then label spans as hallucinated (non-existent function/API/parameter, verified against real library documentation or a static analyzer) vs. faithful.
2. **Adapt existing lightweight detectors** to code tokens: try Paper 1's entropy approach, Paper 20's top-K probability method, Paper 6's NLI/similarity ensemble, and Paper 15's small-NLI + evidence-clustering idea (treating official API docs as the "retrieved evidence" to check code against).
3. **Evaluate** using the same protocol as Paper 6 (F1, AUC-ROC, per-category breakdown) to see which methods transfer and which fail — exactly the kind of systematic frontier-mapping Paper 6 did for text tasks.
4. **If time allows**: fine-tune or re-weight one method specifically for code (e.g., boosting entropy/uncertainty weight around function-call and import-statement tokens, since that's where fabrication concentrates).

**Baselines to directly compare against:** Papers 1, 6, 9, 15, 20 (all lightweight, all with published numbers you can benchmark relative to, even though on different domains — useful for framing "does the QA-domain frontier hold on code?").

**Theoretical framing (from Paper 8):** justify why your dataset needs explicit negative (hallucinated) examples, not just clean code — ties your empirical design directly to a cited theoretical result.

**Novelty claim:** First systematic, CPU-feasible benchmark of hallucination detection methods applied to the code-generation domain — extending the "how far can you get without a GPU" question (Paper 6) to a task nobody in the current literature has covered.

---

## 5. Suggested Next Steps
1. Confirm the code-hallucination gap still holds by doing 2–3 targeted searches specifically for "hallucination detection code generation" / "LLM fabricated API detection" — just to be certain nothing slipped past this 21-paper sample.
2. Read Paper 15 (Logic Matters) and Paper 6 (How Far Without a GPU) in full — these are your two closest methodological templates.
3. Decide on scope: full new detector, or a benchmarking/comparison study (the latter is more feasible for a 4th-year timeline and still clearly publishable).
