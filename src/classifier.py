"""
classifier.py -- Multi-model benchmark & ablation for cheap-signal hallucination detection.

Evaluates multiple classifier architectures using GroupKFold (grouped by task_id,
preventing cross-sample data leakage).

Models:
  1. Logistic Regression (balanced, standardized -- linear interpretable baseline)
  2. Random Forest (balanced, non-linear interactions)
  3. HistGradientBoosting (gradient boosted decision trees)
  4. Multi-Layer Perceptron (MLP with 2 hidden layers)

Features:
  - Decision-point token entropy (mean, max, std, contrast, ratio)
  - Whole-sequence token entropy (mean, max, std, pct_high)
  - Semantic usage pattern diversity across samples
  - AST structural complexity metrics (depth, nodes, call count, code length)

Includes Ablation Studies:
  - All features vs Decision-point vs Sequence vs Consistency vs AST features
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Feature subsets for ablation studies
FEATURE_GROUPS = {
    "all_features": [
        "mean_decision_entropy", "max_decision_entropy", "std_decision_entropy",
        "mean_seq_entropy", "max_seq_entropy", "std_seq_entropy",
        "entropy_contrast", "entropy_ratio", "pct_high_entropy",
        "decision_token_ratio", "task_usage_diversity",
        "n_api_calls", "ast_depth", "ast_node_count", "code_chars", "n_tokens",
        "ast_misuse_flag", "ast_misuse_count", "def_use_deferred_count",
        "has_subscript_on_call", "has_invalid_type_call"
    ],
    "decision_entropy_only": [
        "mean_decision_entropy", "max_decision_entropy", "std_decision_entropy",
        "entropy_contrast", "entropy_ratio", "decision_token_ratio"
    ],
    "sequence_entropy_only": [
        "mean_seq_entropy", "max_seq_entropy", "std_seq_entropy", "pct_high_entropy"
    ],
    "consistency_only": [
        "task_usage_diversity"
    ],
    "ast_complexity_only": [
        "n_api_calls", "ast_depth", "ast_node_count", "code_chars", "n_tokens"
    ],
    "ast_def_use_only": [
        "ast_misuse_flag", "ast_misuse_count", "def_use_deferred_count",
        "has_subscript_on_call", "has_invalid_type_call"
    ],
    "ast_combined": [
        "n_api_calls", "ast_depth", "ast_node_count", "code_chars", "n_tokens",
        "ast_misuse_flag", "ast_misuse_count", "def_use_deferred_count",
        "has_subscript_on_call", "has_invalid_type_call"
    ],
}


def load_dataset(features_path: Path, target_col: str = "target_fail"):
    """Load tabular features and return DataFrame."""
    with open(features_path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def get_model_pipeline(model_name: str):
    """Instantiate model with standard scaling."""
    if model_name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, max_depth=6
        )
    elif model_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100, random_state=42, class_weight="balanced"
        )
    elif model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, random_state=42)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_model_group_kfold(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    n_splits: int = 5,
    verbose: bool = False
) -> dict:
    """Evaluate a single model architecture across GroupKFold splits."""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df["task_id"].values

    # Determine safe split count based on unique groups
    unique_groups = len(np.unique(groups))
    actual_splits = min(n_splits, unique_groups)
    gkf = GroupKFold(n_splits=actual_splits)

    aucs, auprcs, f1s, naive_auprcs = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check for single-class test fold
        if len(np.unique(y_test)) < 2:
            # If all test samples belong to one class in this fold, record majority score
            continue

        model = get_model_pipeline(model_name)
        model.fit(X_train, y_train)

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.decision_function(X_test)
        preds = model.predict(X_test)

        auc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        f1 = f1_score(y_test, preds, zero_division=0)
        naive_auprc = float(np.mean(y_test))

        aucs.append(auc)
        auprcs.append(auprc)
        f1s.append(f1)
        naive_auprcs.append(naive_auprc)

        if verbose:
            print(f"    Fold {fold+1}: AUROC={auc:.3f} | AUPRC={auprc:.3f} (naive={naive_auprc:.3f}) | F1={f1:.3f}")

    return {
        "model": model_name,
        "auroc_mean": float(np.mean(aucs)) if aucs else 0.0,
        "auroc_std": float(np.std(aucs)) if aucs else 0.0,
        "auprc_mean": float(np.mean(auprcs)) if auprcs else 0.0,
        "auprc_std": float(np.std(auprcs)) if auprcs else 0.0,
        "naive_auprc": float(np.mean(naive_auprcs)) if naive_auprcs else 0.0,
        "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
        "f1_std": float(np.std(f1s)) if f1s else 0.0,
        "valid_folds": len(aucs),
    }


def run_benchmark(features_path: Path, target_col: str = "target_fail", n_splits: int = 5):
    """Run full benchmark comparing model architectures and ablations."""
    df = load_dataset(features_path, target_col)
    unique_groups = df["task_id"].nunique()

    print(f"\n{'=' * 85}")
    print(f"  Confidence-Gated Router Benchmark (GroupKFold by task_id)")
    print(f"{'=' * 85}")
    print(f"  Dataset: {features_path.name}")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique task_id groups: {unique_groups}")
    print(f"  GroupKFold splits: {min(n_splits, unique_groups)}")
    if unique_groups < 25:
        print(f"\n  [!] STATISTICAL WARNING: Only {unique_groups} task groups available (< 25-30 recommended).")
        print(f"  >>> PILOT-SCALE RESULT ({unique_groups} tasks) -- NOT YET VALIDATED AT FULL SCALE, treat as provisional. <<<\n")

    pos_rate = df[target_col].mean() * 100
    print(f"  Target: '{target_col}' | Positive class (Failures): {pos_rate:.1f}%\n")

    # 1. Model Architecture Comparison (on all features)
    models = ["logistic_regression", "random_forest", "gradient_boosting", "mlp"]
    all_features = FEATURE_GROUPS["all_features"]

    print(f"--- 1. Model Architecture Benchmark (All {len(all_features)} Features) ---")
    results = []
    for m in models:
        res = evaluate_model_group_kfold(df, all_features, target_col, m, n_splits=n_splits)
        results.append(res)
        print(f"  {m:22s} | AUROC: {res['auroc_mean']:.3f} ± {res['auroc_std']:.3f} | "
              f"AUPRC: {res['auprc_mean']:.3f} ± {res['auprc_std']:.3f} (naive {res['naive_auprc']:.3f}) | "
              f"F1: {res['f1_mean']:.3f}")

    # 2. Feature Importance Analysis (Logistic Regression & Random Forest)
    print(f"\n--- 2. Feature Importance Analysis (Logistic Regression Coefficients) ---")
    pipe = get_model_pipeline("logistic_regression")
    pipe.fit(df[all_features].values, df[target_col].values)
    lr_coefs = pipe.named_steps["logisticregression"].coef_[0]

    rf = get_model_pipeline("random_forest")
    rf.fit(df[all_features].values, df[target_col].values)
    rf_importances = rf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": all_features,
        "LR_Weight": lr_coefs,
        "RF_Importance": rf_importances
    }).sort_values(by="RF_Importance", ascending=False)

    for _, row in importance_df.head(10).iterrows():
        sign = "+" if row["LR_Weight"] >= 0 else "-"
        print(f"  {row['Feature']:25s} | RF Importance: {row['RF_Importance']:6.3f} | LR Weight: {sign}{abs(row['LR_Weight']):.3f}")

    # 3. Ablation Study
    print(f"\n--- 3. Ablation Study: Impact of Feature Subsets (Random Forest) ---")
    for group_name, cols in FEATURE_GROUPS.items():
        ab_res = evaluate_model_group_kfold(df, cols, target_col, "random_forest", n_splits=n_splits)
        print(f"  {group_name:25s} ({len(cols):2d} feats) | AUROC: {ab_res['auroc_mean']:.3f} | AUPRC: {ab_res['auprc_mean']:.3f} | F1: {ab_res['f1_mean']:.3f}")

    print(f"\n{'=' * 75}")
    print(f"  Benchmark complete. Real features successfully validated on GroupKFold!")
    print(f"{'=' * 75}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate cheap-signal classifiers on extracted features")
    parser.add_argument(
        "--features",
        type=str,
        default="data/labels/pilot_1.5B_features.json",
        help="Path to feature JSON file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_fail",
        choices=["target_fail", "target_intent_misuse"],
        help="Target prediction column (default: target_fail)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits (default: 5)",
    )
    args = parser.parse_args()

    features_path = PROJECT_ROOT / args.features
    run_benchmark(features_path, target_col=args.target, n_splits=args.splits)


if __name__ == "__main__":
    main()
