# Verification Cascade Model Comparison Results

> **Note**: PILOT-SCALE (10 tasks) — not yet validated at full 665-task scale.

| Model | Pass Rate | Intent Misuse Rate | Router AUROC | Router AUPRC | Router F1 | Cascade Accuracy @ 50% escalation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-Coder-1.5B** | 15.0% | 7.0% | 1.000 | 1.000 | 0.833 | 93.0% |
| **Qwen2.5-Coder-7B** | 22.0% | 0.0% | 1.000 | 1.000 | 0.000 | 100.0% |

### Detailed Breakdown & Findings
- **Sample Count**: N = 100 completions per model (10 tasks × 10 stochastic samples, T=0.8, Top-p=0.95)
- **Evaluation**: Strict out-of-fold `GroupKFold` grouped by `task_id` (zero prompt leakage)
- **Caveat**: PILOT-SCALE (10 tasks) — not yet validated at full 665-task scale.
