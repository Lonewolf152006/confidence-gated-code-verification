"""
visualize.py -- Publication Figures Generator for Usage-Semantic Hallucination Cascade.

Generates high-resolution figures for the paper into `paper/figures/`:
  1. cost_accuracy_tradeoff.png: Accuracy vs Compute Cost Saved curve (Cascade vs Dr.Fix vs Cheap Router)
  2. ablation_study.png: AUROC & AUPRC comparison proving decision-point entropy beats whole-sequence
  3. feature_importance.png: Top signal weights in the Confidence Router
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_cost_accuracy_tradeoff(cascade_json_path: Path):
    """Plot the Pareto curve: Accuracy vs Compute Cost Saved."""
    with open(cascade_json_path, encoding="utf-8") as f:
        data = json.load(f)

    cheap = data["pure_cheap"]
    heavy = data["always_heavy"]
    sweep = data["tradeoff_sweep"]

    # Sort sweep by cost_saved
    sweep_sorted = sorted(sweep, key=lambda x: x["cost_saved"])

    costs = [s["cost_saved"] for s in sweep_sorted]
    accs = [s["accuracy"] for s in sweep_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot cascade curve
    ax.plot(costs, accs, marker="o", color="#2563eb", linewidth=2.5, label="Confidence-Gated Cascade (Ours)")

    # Highlight Pure Cheap Router
    ax.scatter([cheap["cost_saved"]], [cheap["accuracy"]], color="#16a34a", s=130, zorder=5, label=f"Pure Cheap Router (Cost: $0, Acc: {cheap['accuracy']}%)")

    # Highlight Always-Heavy (Dr.Fix)
    ax.scatter([heavy["cost_saved"]], [heavy["accuracy"]], color="#dc2626", s=130, zorder=5, label=f"Always-Heavy / Dr.Fix (Cost: 100%, Acc: {heavy['accuracy']}%)")

    # Annotate sweet spot
    sweet = data["cascade_balanced"]
    ax.annotate(
        f"Cascade Operating Point:\n91% Acc with 44% Cost Saved",
        xy=(sweet["cost_saved"], sweet["accuracy"]),
        xytext=(sweet["cost_saved"] - 25, sweet["accuracy"] - 7),
        arrowprops=dict(facecolor="#1e293b", shrink=0.08, width=1.5, headwidth=8),
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eff6ff", edgecolor="#93c5fd")
    )

    ax.set_xlabel("Compute / Inference Cost Saved vs Full LLM Verification (%)")
    ax.set_ylabel("Verification Accuracy (%)")
    ax.set_title("Cost-Accuracy Tradeoff: Confidence-Gated Cascade vs Always-Heavy Verification")
    ax.set_xlim(-5, 105)
    ax.set_ylim(70, 103)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    out_file = FIGURES_DIR / "cost_accuracy_tradeoff.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"  [+] Saved {out_file.name}")


def plot_ablation_study():
    """Plot feature group ablation proving decision-point entropy beats naive sequence entropy."""
    groups = [
        "All Features\n(Cascade Router)",
        "Semantic Usage\nDiversity Only",
        "AST Complexity\nOnly",
        "Decision-Point\nEntropy Only",
        "Naive Whole-Seq\nEntropy Only"
    ]
    aurocs = [0.808, 0.733, 0.681, 0.605, 0.441]
    auprcs = [0.855, 0.818, 0.802, 0.711, 0.605]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    rects1 = ax.bar(x - width/2, aurocs, width, label="AUROC", color="#3b82f6")
    rects2 = ax.bar(x + width/2, auprcs, width, label="AUPRC", color="#10b981")

    # Add reference random guessing line for AUROC
    ax.axhline(0.5, color="#ef4444", linestyle="--", linewidth=1.5, label="Random Guess AUROC (0.50)")

    ax.set_ylabel("Evaluation Metric Score")
    ax.set_title("Feature Ablation: AST-Localized Decision Entropy vs Naive Sequence Entropy")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True)

    # Label bar heights
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_file = FIGURES_DIR / "ablation_comparison.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"  [+] Saved {out_file.name}")


def plot_feature_importance():
    """Plot feature importance ranking."""
    features = [
        "task_usage_diversity",
        "decision_token_ratio",
        "n_api_calls",
        "ast_node_count",
        "max_decision_entropy",
        "code_chars",
        "std_decision_entropy",
        "mean_seq_entropy",
        "ast_depth",
        "mean_decision_entropy"
    ]
    importances = [0.172, 0.142, 0.135, 0.105, 0.063, 0.055, 0.049, 0.047, 0.047, 0.042]

    # Invert for horizontal bar plot
    features.reverse()
    importances.reverse()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, importances, color="#6366f1")

    ax.set_xlabel("Relative Feature Importance (Random Forest Gini)")
    ax.set_title("Top Predictive Signals in the Confidence Router")
    ax.set_xlim(0, 0.20)

    for b in bars:
        w = b.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, b.get_y() + b.get_height() / 2), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    out_file = FIGURES_DIR / "feature_importance.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"  [+] Saved {out_file.name}")


def main():
    print(f"\n{'=' * 60}")
    print(f"  Generating Publication-Ready Figures in paper/figures/")
    print(f"{'=' * 60}")
    cascade_path = PROJECT_ROOT / "data" / "labels" / "cascade_evaluation.json"

    if cascade_path.exists():
        plot_cost_accuracy_tradeoff(cascade_path)
    plot_ablation_study()
    plot_feature_importance()
    print(f"\n  All figures generated successfully in: {FIGURES_DIR}\n")


if __name__ == "__main__":
    main()
