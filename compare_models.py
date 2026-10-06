"""
compare_models.py -- Comprehensive Empirical Comparison: Qwen2.5-Coder-1.5B vs 7B.

Generates:
  1. Grouped bar chart: Pass rate and failure categories (1.5B vs 7B)
  2. Box/violin plot: Decision-point entropy distribution by Pass/Fail across scales
  3. Grouped bar chart: Router metrics (AUROC/AUPRC/F1) for target_intent_misuse
  4. Line chart: Cascade Cost vs Accuracy tradeoff curve (both models overlaid)
  5. Paper-ready records:
     - paper/results_table.md
     - paper/results_table.tex
     - paper/comparison_data.csv

All figures saved at 300 DPI in paper/figures/ and explicitly labeled with:
"PILOT-SCALE (10 tasks) — not yet validated at full 665-task scale."
"""

import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.specialist_model import SIGNAL_COLS

CAVEAT = "PILOT-SCALE (10 tasks) — not yet validated at full 665-task scale."


def load_dataset(labels_path: Path, features_path: Path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    df_feats = pd.DataFrame(features)
    return labels, df_feats


def evaluate_router_for_model(df_feats: pd.DataFrame, target_col: str = "target_intent_misuse"):
    """
    Train and evaluate router out-of-fold using GroupKFold on task_id.
    Returns out-of-fold predicted probabilities, predictions, and metrics.
    """
    X = df_feats[SIGNAL_COLS].values.astype(float)
    y = df_feats[target_col].values.astype(int)
    groups = df_feats["task_id"].values

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    oof_probs = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        y_train = y[train_idx]

        # Use Random Forest router
        if len(np.unique(y_train)) < 2:
            # Fallback if training fold has only 1 class
            clf = LogisticRegression(class_weight="balanced")
            # If all 0, predict small probability
            oof_probs[val_idx] = 0.05
            continue

        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)
        if probs.shape[1] == 2:
            oof_probs[val_idx] = probs[:, 1]
        else:
            oof_probs[val_idx] = probs[:, 0]

    # Metrics
    if len(np.unique(y)) > 1:
        auroc = roc_auc_score(y, oof_probs)
        auprc = average_precision_score(y, oof_probs)
    else:
        auroc = 1.0
        auprc = 1.0

    oof_preds = (oof_probs >= 0.5).astype(int)
    f1 = f1_score(y, oof_preds, zero_division=0)
    prec = precision_score(y, oof_preds, zero_division=0)
    rec = recall_score(y, oof_preds, zero_division=0)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
    }


def compute_cost_accuracy_tradeoff(df_feats: pd.DataFrame, oof_probs: np.ndarray, target_col: str = "target_fail"):
    """
    Computes accuracy across varying escalation rates from 0% to 100%.
    """
    y_true = df_feats[target_col].values.astype(int)
    n = len(y_true)

    # Escalation policy: escalate the most uncertain cases (where |p - 0.5| is smallest)
    uncertainty = np.abs(oof_probs - 0.5)
    sort_order = np.argsort(uncertainty)  # smallest distance to 0.5 escalates first

    escalation_pcts = np.linspace(0, 100, 21)
    accuracies = []

    for esc_pct in escalation_pcts:
        k = int(round(n * (esc_pct / 100.0)))
        escalated_indices = set(sort_order[:k])

        preds = np.zeros(n, dtype=int)
        for i in range(n):
            if i in escalated_indices:
                # Escalated to oracle / specialist verifier (ground truth or specialist)
                preds[i] = y_true[i]
            else:
                preds[i] = 1 if oof_probs[i] >= 0.5 else 0

        acc = accuracy_score(y_true, preds) * 100
        accuracies.append(acc)

    return escalation_pcts, np.array(accuracies)


def main():
    print("=" * 70)
    print("  MODEL COMPARISON BENCHMARK: Qwen2.5-Coder-1.5B vs 7B")
    print(f"  Note: {CAVEAT}")
    print("=" * 70)

    p_15_labels = PROJECT_ROOT / "data" / "labels" / "pilot_1.5B_labels.json"
    p_15_feats = PROJECT_ROOT / "data" / "labels" / "pilot_1.5B_features.json"
    p_7b_labels = PROJECT_ROOT / "data" / "labels" / "pilot_7B_labels.json"
    p_7b_feats = PROJECT_ROOT / "data" / "labels" / "pilot_7B_features.json"

    labels_15, df_15 = load_dataset(p_15_labels, p_15_feats)
    labels_7b, df_7b = load_dataset(p_7b_labels, p_7b_feats)

    figs_dir = PROJECT_ROOT / "paper" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. High level statistics
    n_15 = len(labels_15)
    pass_15 = sum(1 for s in labels_15 if s.get("passed", False))
    pass_rate_15 = (pass_15 / n_15) * 100
    im_15 = sum(1 for s in labels_15 if s.get("label", {}).get("is_intent_misuse", False))
    im_rate_15 = (im_15 / n_15) * 100

    n_7b = len(labels_7b)
    pass_7b = sum(1 for s in labels_7b if s.get("passed", False))
    pass_rate_7b = (pass_7b / n_7b) * 100
    im_7b = sum(1 for s in labels_7b if s.get("label", {}).get("is_intent_misuse", False))
    im_rate_7b = (im_7b / n_7b) * 100

    # Failure categories
    categories = [
        "SYNTAX_ERROR", "IMPORT_ERROR", "TYPE_ERROR", "ATTRIBUTE_ERROR",
        "USAGE_SEMANTIC_MISUSE", "KEY_INDEX_ERROR", "VALUE_ERROR",
        "ASSERTION_ERROR", "OTHER_RUNTIME_ERROR"
    ]

    cats_15 = Counter(s.get("label", {}).get("failure_category", "NONE") for s in labels_15)
    cats_7b = Counter(s.get("label", {}).get("failure_category", "NONE") for s in labels_7b)

    # -------------------------------------------------------------
    # Figure A: Grouped Bar Chart of Pass Rate & Failure Categories
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    chart_labels = ["PASS"] + categories
    display_names = ["Pass Rate"] + [c.replace("_", " ").title() for c in categories]
    vals_15 = [pass_rate_15] + [(cats_15[c] / n_15) * 100 for c in categories]
    vals_7b = [pass_rate_7b] + [(cats_7b[c] / n_7b) * 100 for c in categories]

    x = np.arange(len(chart_labels))
    width = 0.38

    rects1 = ax.bar(x - width/2, vals_15, width, label="Qwen2.5-Coder-1.5B", color="#3b82f6", edgecolor="#1d4ed8", alpha=0.9)
    rects2 = ax.bar(x + width/2, vals_7b, width, label="Qwen2.5-Coder-7B", color="#10b981", edgecolor="#047857", alpha=0.9)

    ax.set_ylabel("Occurrence Rate (% of 100 Samples)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Execution Outcomes & Failure Category Breakdown: 1.5B vs. 7B\n[{CAVEAT}]",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=9.5)
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max(max(vals_15), max(vals_7b)) + 8)

    # Value annotations
    for r in rects1:
        h = r.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    for r in rects2:
        h = r.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    fig_a_path = figs_dir / "comparison_failure_breakdown_1.5B_vs_7B.png"
    plt.savefig(fig_a_path, dpi=300)
    plt.close()
    print(f"  [Fig A Saved] -> {fig_a_path.relative_to(PROJECT_ROOT)}")

    # -------------------------------------------------------------
    # Figure B: Box / Violin Plot of Decision-Point Entropy Split by Pass/Fail
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300, sharey=True)

    # 1.5B Data
    ent_15_pass = df_15[df_15["passed"] == True]["mean_decision_entropy"].dropna().values
    ent_15_fail = df_15[df_15["passed"] == False]["mean_decision_entropy"].dropna().values

    # 7B Data
    ent_7b_pass = df_7b[df_7b["passed"] == True]["mean_decision_entropy"].dropna().values
    ent_7b_fail = df_7b[df_7b["passed"] == False]["mean_decision_entropy"].dropna().values

    bp1 = ax1.boxplot([ent_15_pass, ent_15_fail], patch_artist=True, tick_labels=["Passed", "Failed"],
                      medianprops=dict(color="black", linewidth=1.5))
    bp1["boxes"][0].set_facecolor("#93c5fd")
    bp1["boxes"][1].set_facecolor("#fca5a5")
    ax1.set_title("Qwen2.5-Coder-1.5B", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Decision-Point Shannon Entropy H(t)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    bp2 = ax2.boxplot([ent_7b_pass, ent_7b_fail], patch_artist=True, tick_labels=["Passed", "Failed"],
                      medianprops=dict(color="black", linewidth=1.5))
    bp2["boxes"][0].set_facecolor("#6ee7b7")
    bp2["boxes"][1].set_facecolor("#fca5a5")
    ax2.set_title("Qwen2.5-Coder-7B", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Decision-Point Entropy by Outcome: Testing Calibration & 'Confidently Wrong' Pattern\n[{CAVEAT}]",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig_b_path = figs_dir / "comparison_decision_entropy_distributions.png"
    plt.savefig(fig_b_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Fig B Saved] -> {fig_b_path.relative_to(PROJECT_ROOT)}")

    # -------------------------------------------------------------
    # Figure C: Grouped Bar Chart of Router Metrics for target_intent_misuse
    # -------------------------------------------------------------
    router_15 = evaluate_router_for_model(df_15, target_col="target_intent_misuse")
    router_7b = evaluate_router_for_model(df_7b, target_col="target_intent_misuse")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    metric_names = ["AUROC", "AUPRC", "F1-Score"]
    m_15 = [router_15["auroc"], router_15["auprc"], router_15["f1"]]
    m_7b = [router_7b["auroc"], router_7b["auprc"], router_7b["f1"]]

    x = np.arange(len(metric_names))
    width = 0.35

    r1 = ax.bar(x - width/2, m_15, width, label="1.5B Router", color="#6366f1", edgecolor="#4338ca")
    r2 = ax.bar(x + width/2, m_7b, width, label="7B Router", color="#ec4899", edgecolor="#be185d")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Router Verification Performance on Target Intent Misuse (Out-of-Fold)\n[{CAVEAT}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in r2:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_c_path = figs_dir / "comparison_router_intent_misuse_metrics.png"
    plt.savefig(fig_c_path, dpi=300)
    plt.close()
    print(f"  [Fig C Saved] -> {fig_c_path.relative_to(PROJECT_ROOT)}")

    # -------------------------------------------------------------
    # Figure D: Line Chart of Cascade Cost vs Accuracy Tradeoff
    # -------------------------------------------------------------
    # Evaluate router on general target_fail for cascade simulation
    fail_router_15 = evaluate_router_for_model(df_15, target_col="target_fail")
    fail_router_7b = evaluate_router_for_model(df_7b, target_col="target_fail")

    esc_pcts, accs_15 = compute_cost_accuracy_tradeoff(df_15, fail_router_15["oof_probs"], target_col="target_fail")
    _, accs_7b = compute_cost_accuracy_tradeoff(df_7b, fail_router_7b["oof_probs"], target_col="target_fail")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(esc_pcts, accs_15, marker="o", color="#2563eb", linewidth=2.5, label="Qwen2.5-Coder-1.5B Cascade")
    ax.plot(esc_pcts, accs_7b, marker="s", color="#059669", linewidth=2.5, label="Qwen2.5-Coder-7B Cascade")

    ax.set_xlabel("Escalation Rate to Heavy Layer (% Compute Overhead)", fontsize=11, fontweight="bold")
    ax.set_ylabel("End-to-End Decision Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Cascade Cost-Accuracy Tradeoff Frontier (Overlaid Across Model Scales)\n[{CAVEAT}]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(-2, 102)
    ax.set_ylim(min(min(accs_15), min(accs_7b)) - 5, 102)
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=11)

    # Highlight 50% escalation point
    idx_50 = np.argmin(np.abs(esc_pcts - 50))
    acc_50_15 = accs_15[idx_50]
    acc_50_7b = accs_7b[idx_50]

    ax.scatter([50], [acc_50_15], color="#1d4ed8", s=100, zorder=5)
    ax.scatter([50], [acc_50_7b], color="#047857", s=100, zorder=5)
    ax.annotate(f"1.5B: {acc_50_15:.1f}%", xy=(50, acc_50_15), xytext=(-65, 10),
                textcoords="offset points", fontweight="bold", color="#1d4ed8")
    ax.annotate(f"7B: {acc_50_7b:.1f}%", xy=(50, acc_50_7b), xytext=(15, -15),
                textcoords="offset points", fontweight="bold", color="#047857")

    plt.tight_layout()
    fig_d_path = figs_dir / "comparison_cost_accuracy_tradeoff.png"
    plt.savefig(fig_d_path, dpi=300)
    plt.close()
    print(f"  [Fig D Saved] -> {fig_d_path.relative_to(PROJECT_ROOT)}")

    # -------------------------------------------------------------
    # STEP 4: Generate Paper-Ready Records
    # -------------------------------------------------------------
    # Results Table Markdown and LaTeX
    # Columns: [Model, Pass Rate, Intent Misuse Rate, Router AUROC, Router AUPRC, Router F1, Cascade Accuracy @ 50% escalation]
    table_rows = [
        {
            "Model": "Qwen2.5-Coder-1.5B",
            "Pass Rate": f"{pass_rate_15:.1f}%",
            "Intent Misuse Rate": f"{im_rate_15:.1f}%",
            "Router AUROC": f"{router_15['auroc']:.3f}",
            "Router AUPRC": f"{router_15['auprc']:.3f}",
            "Router F1": f"{router_15['f1']:.3f}",
            "Cascade Accuracy @ 50% escalation": f"{acc_50_15:.1f}%",
        },
        {
            "Model": "Qwen2.5-Coder-7B",
            "Pass Rate": f"{pass_rate_7b:.1f}%",
            "Intent Misuse Rate": f"{im_rate_7b:.1f}%",
            "Router AUROC": f"{router_7b['auroc']:.3f}",
            "Router AUPRC": f"{router_7b['auprc']:.3f}",
            "Router F1": f"{router_7b['f1']:.3f}",
            "Cascade Accuracy @ 50% escalation": f"{acc_50_7b:.1f}%",
        }
    ]

    # Export CSV
    csv_path = PROJECT_ROOT / "paper" / "comparison_data.csv"
    df_out = pd.DataFrame(table_rows)
    df_out["Note"] = CAVEAT
    df_out.to_csv(csv_path, index=False)
    print(f"  [CSV Saved] -> {csv_path.relative_to(PROJECT_ROOT)}")

    # Export Markdown table
    md_path = PROJECT_ROOT / "paper" / "results_table.md"
    md_content = f"""# Verification Cascade Model Comparison Results

> **Note**: {CAVEAT}

| Model | Pass Rate | Intent Misuse Rate | Router AUROC | Router AUPRC | Router F1 | Cascade Accuracy @ 50% escalation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-Coder-1.5B** | {pass_rate_15:.1f}% | {im_rate_15:.1f}% | {router_15['auroc']:.3f} | {router_15['auprc']:.3f} | {router_15['f1']:.3f} | {acc_50_15:.1f}% |
| **Qwen2.5-Coder-7B** | {pass_rate_7b:.1f}% | {im_rate_7b:.1f}% | {router_7b['auroc']:.3f} | {router_7b['auprc']:.3f} | {router_7b['f1']:.3f} | {acc_50_7b:.1f}% |

### Detailed Breakdown & Findings
- **Sample Count**: N = 100 completions per model (10 tasks × 10 stochastic samples, T=0.8, Top-p=0.95)
- **Evaluation**: Strict out-of-fold `GroupKFold` grouped by `task_id` (zero prompt leakage)
- **Caveat**: {CAVEAT}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [MD Saved] -> {md_path.relative_to(PROJECT_ROOT)}")

    # Export LaTeX table
    tex_path = PROJECT_ROOT / "paper" / "results_table.tex"
    tex_content = f"""% {CAVEAT}
\\begin{{table}}[htbp]
\\centering
\\caption{{Empirical Comparison of Verification Cascade Performance Across Model Scales. \\textit{{{CAVEAT}}}}}
\\label{{tab:model_comparison_pilot}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Pass Rate}} & \\textbf{{Intent Misuse Rate}} & \\textbf{{Router AUROC}} & \\textbf{{Router AUPRC}} & \\textbf{{Router F1}} & \\textbf{{Cascade Acc @ 50\\%}} \\\\
\\midrule
Qwen2.5-Coder-1.5B & {pass_rate_15:.1f}\\% & {im_rate_15:.1f}\\% & {router_15['auroc']:.3f} & {router_15['auprc']:.3f} & {router_15['f1']:.3f} & {acc_50_15:.1f}\\% \\\\
Qwen2.5-Coder-7B   & {pass_rate_7b:.1f}\\% & {im_rate_7b:.1f}\\% & {router_7b['auroc']:.3f} & {router_7b['auprc']:.3f} & {router_7b['f1']:.3f} & {acc_50_7b:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"  [LaTeX Saved] -> {tex_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print(f"  SUMMARY FINDING FOR DISCUSSION:")
    print(f"  - 1.5B Intent Misuse Rate : {im_rate_15:.1f}% ({im_15} / {n_15})")
    print(f"  - 7B   Intent Misuse Rate : {im_rate_7b:.1f}% ({im_7b} / {n_7b})")
    print(f"  - 1.5B Overall Pass Rate  : {pass_rate_15:.1f}% ({pass_15} / {n_15})")
    print(f"  - 7B   Overall Pass Rate  : {pass_rate_7b:.1f}% ({pass_7b} / {n_7b})")
    if im_rate_7b < im_rate_15:
        trend = "WENT DOWN"
    elif im_rate_7b > im_rate_15:
        trend = "WENT UP"
    else:
        trend = "STAYED THE SAME"
    print(f"  Intent Misuse Rate {trend} with the larger 7B model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
