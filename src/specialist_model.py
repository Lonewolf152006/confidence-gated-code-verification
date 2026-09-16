"""
specialist_model.py -- Distilled Specialist Verifier for Escalated Cases.

The specialist model acts as Layer 3 in the Confidence-Gated Verification Cascade.
It is trained specifically on hard, borderline, and "confidently wrong" cases
where cheap signals alone are uncertain.

Architecture:
  - Multimodal input:
      1. Tabular signal features (entropy, diversity, AST complexity)
      2. Code semantic representation (code TF-IDF n-grams covering API chains & syntax)
  - Deep verification head:
      Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> ReLU -> Sigmoid
  - Training Objective:
      Binary Focal Loss (gamma=2.0) to prioritize hard / borderline misuses
      over trivial easy examples.

Evaluation:
  - 5-fold GroupKFold by task_id
  - Measures performance specifically on borderline cases and overall AUROC/AUPRC.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SIGNAL_COLS = [
    "mean_decision_entropy", "max_decision_entropy", "std_decision_entropy",
    "mean_seq_entropy", "max_seq_entropy", "std_seq_entropy",
    "entropy_contrast", "entropy_ratio", "pct_high_entropy",
    "decision_token_ratio", "task_usage_diversity",
    "n_api_calls", "ast_depth", "ast_node_count", "code_chars", "n_tokens"
]


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for hard example mining:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and concentrates training on ambiguous / confidently-wrong cases.
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)

        # Numerical stability clamp
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        p_t = torch.clamp(p_t, 1e-7, 1.0 - 1e-7)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)

        bce = -torch.log(p_t)
        loss = focal_weight * bce
        return loss.mean()


class SpecialistNetwork(nn.Module):
    """Deep verification head combining tabular signal features and code representation."""
    def __init__(self, tabular_dim: int, code_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = tabular_dim + code_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CodeSignalDataset(Dataset):
    def __init__(self, X_tab: np.ndarray, X_code: np.ndarray, y: np.ndarray):
        combined = np.hstack([X_tab, X_code]).astype(np.float32)
        self.X = torch.tensor(combined, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SpecialistVerifier:
    """Production wrapper for inference in the cascade router."""
    def __init__(self, model_path: Path = None):
        self.scaler = None
        self.tfidf = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path and model_path.exists():
            self.load(model_path)

    def predict_proba(self, tabular_feats: np.ndarray, code_texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        self.model.eval()

        X_tab = self.scaler.transform(tabular_feats)
        X_code = self.tfidf.transform(code_texts).toarray()
        X_all = np.hstack([X_tab, X_code]).astype(np.float32)

        with torch.no_grad():
            inp = torch.tensor(X_all, dtype=torch.float32).to(self.device)
            logits = self.model(inp)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def save(self, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "tfidf": self.tfidf,
            "tabular_dim": len(SIGNAL_COLS),
            "code_dim": self.tfidf.max_features,
        }, save_path)
        print(f"  Specialist model checkpoint saved to: {save_path}")

    def load(self, load_path: Path):
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.scaler = checkpoint["scaler"]
        self.tfidf = checkpoint["tfidf"]
        self.model = SpecialistNetwork(
            tabular_dim=checkpoint["tabular_dim"],
            code_dim=checkpoint["code_dim"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()


def train_specialist(
    features_path: Path,
    labels_path: Path,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 0.003,
    save_path: Path = None
) -> dict:
    """Train specialist verifier with GroupKFold evaluation."""
    print(f"\n{'=' * 75}")
    print(f"  Training Distilled Specialist Verifier (Focal Loss + Code Features)")
    print(f"{'=' * 75}")

    with open(features_path, encoding="utf-8") as f:
        feat_records = json.load(f)
    with open(labels_path, encoding="utf-8") as f:
        label_records = json.load(f)

    # Match records
    code_map = {(r["task_id"], r["sample_index"]): r.get("generated_code", "") for r in label_records}

    X_tab_raw = []
    code_texts = []
    y_raw = []
    groups = []

    for r in feat_records:
        key = (r["task_id"], r["sample_index"])
        code = code_map.get(key, "")
        code_clean = code.split("```")[0]
        row_feats = [float(r.get(c, 0.0)) for c in SIGNAL_COLS]

        X_tab_raw.append(row_feats)
        code_texts.append(code_clean)
        y_raw.append(int(r["target_fail"]))
        groups.append(r["task_id"])

    X_tab_raw = np.array(X_tab_raw, dtype=float)
    y_raw = np.array(y_raw, dtype=float)
    groups = np.array(groups)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device} | Total samples: {len(y_raw)} | Fail rate: {y_raw.mean():.1%}")

    # 5-fold GroupKFold evaluation
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    fold_aucs, fold_auprcs, fold_f1s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_tab_raw, y_raw, groups)):
        y_test = y_raw[test_idx]
        if len(np.unique(y_test)) < 2:
            continue

        # Fit TF-IDF & Scaler only on train fold
        tfidf = TfidfVectorizer(max_features=48, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|\.[a-zA-Z_]\w*")
        X_train_code = tfidf.fit_transform([code_texts[i] for i in train_idx]).toarray()
        X_test_code = tfidf.transform([code_texts[i] for i in test_idx]).toarray()

        scaler = StandardScaler()
        X_train_tab = scaler.fit_transform(X_tab_raw[train_idx])
        X_test_tab = scaler.transform(X_tab_raw[test_idx])

        train_dataset = CodeSignalDataset(X_train_tab, X_train_code, y_raw[train_idx])
        test_dataset = CodeSignalDataset(X_test_tab, X_test_code, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = SpecialistNetwork(tabular_dim=len(SIGNAL_COLS), code_dim=48).to(device)
        criterion = BinaryFocalLoss(alpha=0.5, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for epoch in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate test fold
        model.eval()
        with torch.no_grad():
            X_test_tensor = test_dataset.X.to(device)
            logits = model(X_test_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        f1 = f1_score(y_test, preds, zero_division=0)

        fold_aucs.append(auc)
        fold_auprcs.append(auprc)
        fold_f1s.append(f1)
        print(f"  Fold {fold+1}: AUROC = {auc:.3f} | AUPRC = {auprc:.3f} | F1 = {f1:.3f}")

    print(f"\n  Specialist Verifier GroupKFold Performance:")
    print(f"    Mean AUROC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    print(f"    Mean AUPRC: {np.mean(fold_auprcs):.3f} ± {np.std(fold_auprcs):.3f}")
    print(f"    Mean F1:    {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")

    # Train final model on full data and save
    if save_path:
        final_tfidf = TfidfVectorizer(max_features=48, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|\.[a-zA-Z_]\w*")
        full_code = final_tfidf.fit_transform(code_texts).toarray()
        final_scaler = StandardScaler()
        full_tab = final_scaler.fit_transform(X_tab_raw)

        full_dataset = CodeSignalDataset(full_tab, full_code, y_raw)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        final_model = SpecialistNetwork(tabular_dim=len(SIGNAL_COLS), code_dim=48).to(device)
        crit = BinaryFocalLoss(alpha=0.5, gamma=2.0)
        opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        final_model.train()
        for epoch in range(epochs):
            for bx, by in full_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = crit(final_model(bx), by)
                loss.backward()
                opt.step()
            sched.step()

        final_verifier = SpecialistVerifier()
        final_verifier.model = final_model
        final_verifier.scaler = final_scaler
        final_verifier.tfidf = final_tfidf
        final_verifier.save(save_path)

    return {
        "auroc_mean": float(np.mean(fold_aucs)),
        "auprc_mean": float(np.mean(fold_auprcs)),
        "f1_mean": float(np.mean(fold_f1s)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Distilled Specialist Verifier")
    parser.add_argument("--features", type=str, default="data/labels/pilot_1.5B_features.json")
    parser.add_argument("--labels", type=str, default="data/labels/pilot_1.5B_labels.json")
    parser.add_argument("--save", type=str, default="data/models/specialist_model.pt")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    feat_p = PROJECT_ROOT / args.features
    lab_p = PROJECT_ROOT / args.labels
    save_p = PROJECT_ROOT / args.save

    train_specialist(feat_p, lab_p, epochs=args.epochs, save_path=save_p)


if __name__ == "__main__":
    main()
