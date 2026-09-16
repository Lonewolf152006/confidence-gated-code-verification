"""
cascade.py -- Confidence-Gated Verification Cascade Evaluation (Leak-Free GroupKFold).

Implements the multi-layer confidence-gated verification cascade:
  1. Layer 1: Cheap signal extraction (decision entropy, usage diversity, AST metrics)
  2. Layer 2: Fast router (Random Forest)
  3. Layer 3: Specialist Verifier (Focal Loss + code TF-IDF + signal features)

CRITICAL METHODOLOGY NOTE:
  - Both p_cheap and p_specialist are evaluated strictly OUT-OF-FOLD using GroupKFold
    grouped by task_id to prevent any training leakage.
  - Compares:
      (a) Pure Cheap Router (0% escalation)
      (b) Always-Escalate (100% escalation to specialist_model.py alone)
      (c) Confidence-Gated Cascade (escalating only borderline cases)
  - Evaluates on pilot dataset (10 tasks, 100 samples) and prints explicit
    warnings regarding statistical underpowering.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import get_model_pipeline, FEATURE_GROUPS
from src.specialist_model import (
    SpecialistNetwork, BinaryFocalLoss, CodeSignalDataset, SIGNAL_COLS
)


def evaluate_cascade_at_thresholds(
    y_true: np.ndarray,
    p_cheap: np.ndarray,
    p_specialist: np.ndarray,
    tau_accept: float = 0.35,
    tau_reject: float = 0.85,
) -> dict:
    """
    Simulate cascade routing:
      - If p_cheap <= tau_accept  --> FAST_ACCEPT (predict 0: pass)
      - If p_cheap >= tau_reject  --> FAST_REJECT (predict 1: fail)
      - Else                      --> ESCALATE to Specialist (predict p_specialist >= 0.5)
    """
    n = len(y_true)
    decisions = np.zeros(n, dtype=int)
    escalated_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        if p_cheap[i] <= tau_accept:
            decisions[i] = 0  # Predict Pass
        elif p_cheap[i] >= tau_reject:
            decisions[i] = 1  # Predict Fail
        else:
            escalated_mask[i] = True
            decisions[i] = 1 if p_specialist[i] >= 0.5 else 0

    escalation_rate = float(np.mean(escalated_mask) * 100)
    cost_saved = 100.0 - escalation_rate
    acc = float(accuracy_score(y_true, decisions) * 100)
    f1 = float(f1_score(y_true, decisions, zero_division=0))
    prec = float(precision_score(y_true, decisions, zero_division=0))
    rec = float(recall_score(y_true, decisions, zero_division=0))

    return {
        "tau_accept": tau_accept,
        "tau_reject": tau_reject,
        "escalation_rate": round(escalation_rate, 1),
        "cost_saved": round(cost_saved, 1),
        "accuracy": round(acc, 2),
        "f1": round(f1, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "n_escalated": int(np.sum(escalated_mask)),
        "total_samples": n,
    }


def compute_outof_fold_predictions(
    df: pd.DataFrame,
    code_texts: list[str],
    n_splits: int = 5,
    epochs: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Train both Layer 2 (cheap router) and Layer 3 (specialist model) strictly out-of-fold.
    Guarantees ZERO data leakage between training and testing.
    """
    all_features = FEATURE_GROUPS["all_features"]
    X_cheap = df[all_features].values
    X_tab_sig = df[SIGNAL_COLS].values
    y = df["target_fail"].values
    groups = df["task_id"].values

    unique_groups = len(np.unique(groups))
    actual_splits = min(n_splits, unique_groups)

    print(f"  GroupKFold Setup:")
    print(f"    - Total samples: {len(y)}")
    print(f"    - Unique task groups (task_id): {unique_groups}")
    print(f"    - Number of splits (n_splits): {actual_splits}")
    print(f"    - Average tasks per test fold: {unique_groups / actual_splits:.1f}")

    if unique_groups < 25:
        print(f"\n  [!] STATISTICAL WARNING: Only {unique_groups} unique task groups available (recommended: >= 25).")
        print(f"  >>> PILOT-SCALE RESULT ({unique_groups} tasks) -- NOT YET VALIDATED AT FULL SCALE, treat as provisional. <<<\n")

    gkf = GroupKFold(n_splits=actual_splits)
    p_cheap = np.zeros(len(y))
    p_specialist = np.zeros(len(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_cheap, y, groups)):
        test_tasks = list(set(groups[test_idx]))
        print(f"  Training Fold {fold+1}/{actual_splits} (Test tasks: {test_tasks})...")

        # 1. Train Cheap Router (Random Forest) strictly on train_idx
        rf = get_model_pipeline("random_forest")
        rf.fit(X_cheap[train_idx], y[train_idx])
        p_cheap[test_idx] = rf.predict_proba(X_cheap[test_idx])[:, 1]

        # 2. Train Specialist Model strictly on train_idx (fit TFIDF & Scaler on train_idx ONLY)
        tfidf = TfidfVectorizer(max_features=48, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|\.[a-zA-Z_]\w*")
        X_train_code = tfidf.fit_transform([code_texts[i] for i in train_idx]).toarray()
        X_test_code = tfidf.transform([code_texts[i] for i in test_idx]).toarray()

        scaler = StandardScaler()
        X_train_tab = scaler.fit_transform(X_tab_sig[train_idx])
        X_test_tab = scaler.transform(X_tab_sig[test_idx])

        train_dataset = CodeSignalDataset(X_train_tab, X_train_code, y[train_idx])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        spec_model = SpecialistNetwork(tabular_dim=len(SIGNAL_COLS), code_dim=48).to(device)
        criterion = BinaryFocalLoss(alpha=0.5, gamma=2.0)
        optimizer = torch.optim.AdamW(spec_model.parameters(), lr=0.003, weight_decay=1e-3)

        spec_model.train()
        for _ in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(spec_model(bx), by)
                loss.backward()
                optimizer.step()

        # Out-of-fold prediction on test_idx
        spec_model.eval()
        with torch.no_grad():
            X_test_all = np.hstack([X_test_tab, X_test_code]).astype(np.float32)
            inp = torch.tensor(X_test_all, dtype=torch.float32).to(device)
            logits = spec_model(inp)
            p_specialist[test_idx] = torch.sigmoid(logits).cpu().numpy().flatten()

    return y, p_cheap, p_specialist, unique_groups


def run_cascade_benchmark(
    features_path: Path,
    labels_path: Path,
    output_path: Path = None,
    n_splits: int = 5,
):
    print(f"\n{'=' * 85}")
    print(f"  Confidence-Gated Verification Cascade Benchmark (Leak-Free Out-of-Fold)")
    print(f"{'=' * 85}")

    # Load datasets
    with open(features_path, encoding="utf-8") as f:
        feat_records = json.load(f)
    with open(labels_path, encoding="utf-8") as f:
        label_records = json.load(f)

    code_map = {(r["task_id"], r["sample_index"]): r.get("generated_code", "") for r in label_records}
    df = pd.DataFrame(feat_records)
    code_texts = [code_map.get((r["task_id"], r["sample_index"]), "").split("```")[0] for r in feat_records]

    # Compute strictly out-of-fold predictions
    y, p_cheap, p_specialist, n_groups = compute_outof_fold_predictions(df, code_texts, n_splits=n_splits)

    warning_tag = f"[PILOT-SCALE RESULT ({n_groups} tasks) -- NOT YET VALIDATED AT FULL SCALE, treat as provisional]"

    print(f"\n{'-'*85}")
    print(f"  {warning_tag}")
    print(f"{'-'*85}")

    # Out-of-fold metrics for individual models
    auc_cheap = roc_auc_score(y, p_cheap)
    auprc_cheap = average_precision_score(y, p_cheap)
    auc_spec = roc_auc_score(y, p_specialist)
    auprc_spec = average_precision_score(y, p_specialist)

    print(f"\n--- Out-of-Fold Model Generalization Metrics ---")
    print(f"  1. Cheap Router (Random Forest alone)       | AUROC: {auc_cheap:.3f} | AUPRC: {auprc_cheap:.3f}")
    print(f"  2. Specialist Verifier (Focal Net alone)    | AUROC: {auc_spec:.3f} | AUPRC: {auprc_spec:.3f}")

    print(f"\n--- Strategy Comparison Table ---")
    print(f"  {'Strategy':38s} | {'Escalation %':12s} | {'Cost Saved %':12s} | {'Accuracy %':10s} | {'F1-Score':8s}")
    print(f"  {'-'*38}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

    # Baseline 1: Pure Cheap Router (0% escalation)
    res_cheap = evaluate_cascade_at_thresholds(y, p_cheap, p_specialist, tau_accept=0.5, tau_reject=0.5)
    print(f"  {'1. Pure Cheap Router (0% Escalate)':38s} | {res_cheap['escalation_rate']:10.1f}% | {res_cheap['cost_saved']:10.1f}% | {res_cheap['accuracy']:8.1f}% | {res_cheap['f1']:.3f}")

    # Baseline 2: Always-Escalate (100% escalation to specialist_model.py alone)
    res_heavy = evaluate_cascade_at_thresholds(y, p_cheap, p_specialist, tau_accept=-0.01, tau_reject=1.01)
    print(f"  {'2. Always-Escalate (100% Specialist)':38s} | {res_heavy['escalation_rate']:10.1f}% | {res_heavy['cost_saved']:10.1f}% | {res_heavy['accuracy']:8.1f}% | {res_heavy['f1']:.3f}")

    # Candidate 3: Cascade (Balanced window)
    res_cascade_bal = evaluate_cascade_at_thresholds(y, p_cheap, p_specialist, tau_accept=0.30, tau_reject=0.75)
    print(f"  {'3. Confidence Cascade (Balanced)':38s} | {res_cascade_bal['escalation_rate']:10.1f}% | {res_cascade_bal['cost_saved']:10.1f}% | {res_cascade_bal['accuracy']:8.1f}% | {res_cascade_bal['f1']:.3f}")

    # Candidate 4: Cascade (High-Efficiency)
    res_cascade_eff = evaluate_cascade_at_thresholds(y, p_cheap, p_specialist, tau_accept=0.40, tau_reject=0.85)
    print(f"  {'4. Confidence Cascade (High-Efficiency)':38s} | {res_cascade_eff['escalation_rate']:10.1f}% | {res_cascade_eff['cost_saved']:10.1f}% | {res_cascade_eff['accuracy']:8.1f}% | {res_cascade_eff['f1']:.3f}")

    # Tradeoff Sweep
    print(f"\n--- Confidence Threshold Sweep ---")
    print(f"  {'Accept Tau':10s} | {'Reject Tau':10s} | {'Escalation %':12s} | {'Cost Saved %':12s} | {'Accuracy %':10s} | {'F1-Score':8s}")
    tradeoff_records = []
    for accept_tau in [0.20, 0.30, 0.40]:
        for reject_tau in [0.70, 0.80, 0.90]:
            sweep_res = evaluate_cascade_at_thresholds(y, p_cheap, p_specialist, tau_accept=accept_tau, tau_reject=reject_tau)
            tradeoff_records.append(sweep_res)
            print(f"  {accept_tau:10.2f} | {reject_tau:10.2f} | {sweep_res['escalation_rate']:10.1f}% | {sweep_res['cost_saved']:10.1f}% | {sweep_res['accuracy']:8.1f}% | {sweep_res['f1']:.3f}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pilot_warning": warning_tag,
            "unique_task_groups": n_groups,
            "pure_cheap_router": res_cheap,
            "always_escalate_specialist": res_heavy,
            "cascade_balanced": res_cascade_bal,
            "cascade_efficient": res_cascade_eff,
            "tradeoff_sweep": tradeoff_records,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        print(f"\n  Saved leak-free evaluation to: {output_path}")

    print(f"{'=' * 85}\n")


def main():
    parser = argparse.ArgumentParser(description="Leak-Free Cascade Benchmark")
    parser.add_argument("--features", type=str, default="data/labels/pilot_1.5B_features.json")
    parser.add_argument("--labels", type=str, default="data/labels/pilot_1.5B_labels.json")
    parser.add_argument("--output", type=str, default="data/labels/cascade_evaluation.json")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    feat_p = PROJECT_ROOT / args.features
    lab_p = PROJECT_ROOT / args.labels
    out_p = PROJECT_ROOT / args.output

    run_cascade_benchmark(feat_p, lab_p, out_p, n_splits=args.splits)


if __name__ == "__main__":
    main()
