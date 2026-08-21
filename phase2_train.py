"""
Phase 2 — Autoencoder Training (PyTorch)
Trains on ONLY the 4,114 normal windows from Phase 1 cache.
Uses target loss early stopping to keep the model as a slightly
weak reconstructor — the property the whole detection layer depends on.

After training it measures reconstruction error on all 41,276 windows
and reports the separation gap between normal and abnormal.
Sets the detection threshold at the 97th percentile of normal error.

Usage:
    python phase2_train.py

Output:
    cache/autoencoder.pt        — trained model weights
    cache/thresholds.json       — per-role error stats + threshold
    cache/phase2_results.csv    — error per window for analysis
    cache/phase2_report.txt     — human readable summary
"""

import json
import time
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config
BASE   = Path(__file__).parent
CACHE  = BASE / "cache"

TARGET_VAL_LOSS = 0.006
BATCH_SIZE      = 64
MAX_EPOCHS      = 200
PATIENCE        = 10
VAL_SPLIT       = 0.15
THRESHOLD_PCT   = 97
THRESHOLD_FLOOR = 0.002
THRESHOLD_CEIL  = 0.200
RANDOM_SEED     = 42
LR              = 1e-3
LR_PATIENCE     = 4
LR_FACTOR       = 0.5
MIN_LR          = 1e-5

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------ logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(CACHE / "phase2_log.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger()


# ------------------------------------------------------------------ model
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ------------------------------------------------------------------ load cache
def load_cache():
    log.info("Loading Phase 1 cache ...")
    npz  = np.load(CACHE / "features.npz")
    meta = pd.read_csv(CACHE / "metadata.csv")

    X = np.concatenate([
        npz["mel_mean"],
        npz["mel_std"],
        npz["mfcc_mean"],
        npz["mfcc_std"],
    ], axis=1).astype(np.float32)

    log.info(f"  Feature matrix shape : {X.shape}")
    log.info(f"  Metadata rows        : {len(meta)}")
    log.info(f"  Role distribution:\n{meta['role'].value_counts().to_string()}")
    log.info(f"  Device               : {DEVICE}")
    log.info(f"  PyTorch version      : {torch.__version__}")

    return X, meta


# ------------------------------------------------------------------ normalise
def normalise(X_train, X_all):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_all - mean) / std, mean, std


# ------------------------------------------------------------------ train
def train(X_normal_n, input_dim):
    n_val   = max(1, int(len(X_normal_n) * VAL_SPLIT))
    n_train = len(X_normal_n) - n_val

    tensor  = torch.tensor(X_normal_n, dtype=torch.float32)
    dataset = TensorDataset(tensor, tensor)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False)

    model     = Autoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"\n  Model parameters     : {n_params:,}")
    log.info(f"  Training windows     : {n_train}")
    log.info(f"  Validation windows   : {n_val}")
    log.info(f"  Target val loss      : {TARGET_VAL_LOSS}")
    log.info(f"  Max epochs           : {MAX_EPOCHS}")
    log.info("")

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    stopped_at = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                val_loss += criterion(model(xb), xb).item() * len(xb)
        val_loss /= n_val

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            log.info(f"  Epoch {epoch:3d}  "
                     f"train={train_loss:.6f}  "
                     f"val={val_loss:.6f}  "
                     f"lr={lr_now:.2e}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # target loss stop
        if val_loss <= TARGET_VAL_LOSS:
            log.info(f"\n  Target val_loss {TARGET_VAL_LOSS} reached "
                     f"at epoch {epoch}. Stopping.")
            stopped_at = epoch
            break

        if no_improve >= PATIENCE:
            log.info(f"\n  No improvement for {PATIENCE} epochs. Stopping.")
            stopped_at = epoch
            break

    model.load_state_dict(best_state)
    log.info(f"\n  Best val_loss   : {best_val:.6f}")
    log.info(f"  Stopped at epoch: {stopped_at}")

    return model, best_val, stopped_at


# ------------------------------------------------------------------ evaluate
def evaluate(model, X_all_n, meta):
    log.info("\nComputing reconstruction error on all windows ...")
    model.eval()
    tensor = torch.tensor(X_all_n, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=512, shuffle=False)
    errors = []

    with torch.no_grad():
        for (xb,) in loader:
            xb   = xb.to(DEVICE)
            pred = model(xb)
            err  = ((xb - pred) ** 2).mean(dim=1).cpu().numpy()
            errors.append(err)

    errors = np.concatenate(errors)
    meta   = meta.copy()
    meta["recon_error"] = errors

    log.info("\nReconstruction error by role:")
    for role in ["normal", "false_alarm", "threat", "background"]:
        subset = errors[meta["role"] == role]
        if len(subset):
            log.info(f"  {role:12}  n={len(subset):6}  "
                     f"mean={subset.mean():.6f}  "
                     f"std={subset.std():.6f}  "
                     f"max={subset.max():.6f}")

    return meta, errors


# ------------------------------------------------------------------ threshold
def set_threshold(errors_normal, all_errors, meta):
    raw_thresh = np.percentile(errors_normal, THRESHOLD_PCT)
    threshold  = float(np.clip(raw_thresh, THRESHOLD_FLOOR, THRESHOLD_CEIL))

    log.info(f"\nThreshold calibration ({THRESHOLD_PCT}th pct of normal):")
    log.info(f"  Raw {THRESHOLD_PCT}th percentile : {raw_thresh:.6f}")
    log.info(f"  After floor/ceil    : {threshold:.6f}")

    meta = meta.copy()
    meta["flagged"] = (meta["recon_error"] > threshold).astype(int)

    tp = ((meta["flagged"] == 1) & (meta["role"] == "threat")).sum()
    fp = ((meta["flagged"] == 1) &
          (meta["role"].isin(["normal", "false_alarm"]))).sum()
    fn = ((meta["flagged"] == 0) & (meta["role"] == "threat")).sum()

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    log.info(f"\nDetection performance at threshold {threshold:.6f}:")
    log.info(f"  True positives  : {tp}")
    log.info(f"  False positives : {fp}")
    log.info(f"  Precision       : {precision:.4f}")
    log.info(f"  Recall          : {recall:.4f}")
    log.info(f"  F1              : {f1:.4f}")

    normal_mean = all_errors[meta["role"] == "normal"].mean()
    threat_mean = all_errors[meta["role"] == "threat"].mean()
    gap         = threat_mean - normal_mean

    log.info(f"\nSeparation gap:")
    log.info(f"  Normal mean     : {normal_mean:.6f}")
    log.info(f"  Threat mean     : {threat_mean:.6f}")
    log.info(f"  Gap             : {gap:.6f}  "
             f"({'GOOD' if gap > 0.005 else 'WARN — consider retraining'})")

    stats = {
        "threshold"       : round(threshold, 6),
        "precision"       : round(float(precision), 4),
        "recall"          : round(float(recall), 4),
        "f1"              : round(float(f1), 4),
        "true_pos"        : int(tp),
        "false_pos"       : int(fp),
        "normal_mean"     : round(float(normal_mean), 6),
        "threat_mean"     : round(float(threat_mean), 6),
        "gap"             : round(float(gap), 6),
        "target_val_loss" : TARGET_VAL_LOSS,
        "threshold_pct"   : THRESHOLD_PCT,
    }

    return threshold, stats, meta


# ------------------------------------------------------------------ save
def save_results(model, stats, meta, mean, std):
    model_path = CACHE / "autoencoder.pt"
    torch.save(model.state_dict(), model_path)
    log.info(f"\n  Model saved      -> {model_path}")

    thresh_path = CACHE / "thresholds.json"
    with open(thresh_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    log.info(f"  Thresholds saved -> {thresh_path}")

    results_path = CACHE / "phase2_results.csv"
    meta.to_csv(results_path, index=False)
    log.info(f"  Results saved    -> {results_path}")

    scaler_path = CACHE / "scaler.npz"
    np.savez(scaler_path, mean=mean, std=std)
    log.info(f"  Scaler saved     -> {scaler_path}")

    report_path = CACHE / "phase2_report.txt"
    with open(report_path, "w") as fh:
        fh.write("PHASE 2 REPORT — AUTOENCODER TRAINING\n")
        fh.write("=" * 50 + "\n\n")
        for k, v in stats.items():
            fh.write(f"  {k:25} : {v}\n")
    log.info(f"  Report saved     -> {report_path}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    t0 = time.time()
    log.info("PERIMETER THREAT SYSTEM  —  Phase 2 Autoencoder Training (PyTorch)")

    X, meta     = load_cache()
    normal_mask = meta["role"] == "normal"
    X_normal    = X[normal_mask.values]

    log.info(f"\n  Normal windows for training : {len(X_normal)}")
    log.info(f"  All windows for evaluation  : {len(X)}")

    X_normal_n, X_all_n, mean, std = normalise(X_normal, X)

    model, best_val, stopped_at = train(X_normal_n, X.shape[1])

    meta_with_errors, all_errors = evaluate(model, X_all_n, meta)
    errors_normal = all_errors[normal_mask.values]

    threshold, stats, meta_final = set_threshold(
        errors_normal, all_errors, meta_with_errors)

    save_results(model, stats, meta_final, mean, std)

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    print("\n" + "=" * 60)
    print("  PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"  Threshold     : {stats['threshold']}")
    print(f"  Precision     : {stats['precision']}")
    print(f"  Recall        : {stats['recall']}")
    print(f"  F1            : {stats['f1']}")
    print(f"  False pos     : {stats['false_pos']}")
    print(f"  Normal mean   : {stats['normal_mean']}")
    print(f"  Threat mean   : {stats['threat_mean']}")
    print(f"  Gap           : {stats['gap']}")
    print()
    if stats["false_pos"] == 0:
        print("  Zero false positives on normal/false_alarm windows.")
    else:
        print(f"  {stats['false_pos']} false positives.")
        print("  If precision < 0.85, raise TARGET_VAL_LOSS to 0.008 and retrain.")
    print("\n  Ready for Phase 3 — frequency layer + severity scoring.")