"""
Phase 3 — Frequency Layer Classifier + Tiered Fusion
Builds the second detection layer on top of Phase 2's autoencoder.

Fusion rule (strict AND — both layers must agree for THREAT):

  THREAT  — AE fires AND freq layer confident >= 70%
  WARNING — exactly one of the two fires
  CLEAR   — neither fires

No single layer can unilaterally call THREAT. This is what keeps
the false alarm rate on animal/environmental sounds low.

Usage:
    python phase3_classifier.py

Output:
    cache/freq_classifier.pkl   — trained Random Forest
    cache/freq_scaler.pkl       — feature scaler
    cache/phase3_results.csv    — per-window fusion verdict
    cache/phase3_report.txt     — human readable summary
"""

import json
import time
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config
BASE  = Path(__file__).parent
CACHE = BASE / "cache"

RANDOM_SEED     = 42
N_ESTIMATORS    = 200
MAX_DEPTH       = 20
MIN_SAMPLES     = 5
FREQ_CONFIDENCE = 0.70   # freq layer fires only when >= 70% confident

BAND_NAMES = ["sub_bass", "bass", "vehicle", "low_mid",
              "mid", "high_mid", "high"]
BAND_FEATS = (
    [f"band_{b}_energy" for b in BAND_NAMES] +
    [f"band_{b}_ratio"  for b in BAND_NAMES]
)
SCALAR_FEATS = [
    "kurtosis", "rms", "zcr_mean", "zcr_std",
    "spectral_centroid_mean", "spectral_rolloff_mean",
]
ALL_FEATS = BAND_FEATS + SCALAR_FEATS

LABEL_TO_CLASS = {
    "footsteps"        : "person",
    "engine_idling"    : "vehicle",
    "engine"           : "vehicle",
    "car_horn"         : "vehicle",
    "jackhammer"       : "vehicle",
    "drilling"         : "vehicle",
    "chainsaw"         : "vehicle",
    "helicopter"       : "vehicle",
    "gunshot"          : "gunshot",
    "gun_shot"         : "gunshot",
    "explosion"        : "gunshot",
    "siren"            : "siren",
    "dog"              : "animal",
    "dog_bark"         : "animal",
    "cat"              : "animal",
    "frog"             : "animal",
    "insects"          : "animal",
    "chirping_birds"   : "animal",
    "hen"              : "animal",
    "pig"              : "animal",
    "cow"              : "animal",
    "sheep"            : "animal",
    "crow"             : "animal",
    "children_playing" : "animal",
    "casual"           : "animal",
    "wind"             : "environmental",
    "rain"             : "environmental",
    "sea_waves"        : "environmental",
    "thunderstorm"     : "environmental",
    "water_drops"      : "environmental",
    "crickets"         : "environmental",
    "air_conditioner"  : "environmental",
    "street_music"     : "environmental",
}

THREAT_CLASSES = {"person", "gunshot", "vehicle", "siren"}

# ------------------------------------------------------------------ logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(CACHE / "phase3_log.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger()


# ------------------------------------------------------------------ load
def load_data():
    log.info("Loading Phase 1 metadata and Phase 2 results ...")
    meta2 = pd.read_csv(CACHE / "phase2_results.csv")

    with open(CACHE / "thresholds.json") as fh:
        thresh_info = json.load(fh)

    threshold   = thresh_info["threshold"]
    normal_errs = meta2.loc[meta2["role"] == "normal", "recon_error"]
    healthy_max = float(np.percentile(normal_errs, 99))  # 99th pct, not max

    log.info(f"  Total windows     : {len(meta2)}")
    log.info(f"  AE threshold      : {threshold:.6f}")
    log.info(f"  Healthy max (99p) : {healthy_max:.6f}")
    log.info(f"  Freq confidence   : >= {FREQ_CONFIDENCE}")
    log.info(f"  Fusion rule       : STRICT AND (both layers must agree)")

    return meta2, threshold, healthy_max


# ------------------------------------------------------------------ features
def build_features(df):
    available = [c for c in ALL_FEATS if c in df.columns]
    X = df[available].copy().fillna(df[available].median())
    for col in X.columns:
        p1, p99 = X[col].quantile([0.01, 0.99])
        X[col]  = X[col].clip(p1, p99)
    return X.values.astype(np.float32), available


# ------------------------------------------------------------------ labels
def map_labels(df):
    labels = df["label"].map(LABEL_TO_CLASS)
    unmapped = labels.isna()
    if unmapped.sum():
        role_map = {
            "threat"      : "gunshot",
            "false_alarm" : "animal",
            "normal"      : "environmental",
            "background"  : "environmental",
        }
        labels[unmapped] = df.loc[unmapped, "role"].map(role_map)
    labels = labels.fillna("environmental")

    log.info("\n  Class distribution after mapping:")
    for cls, n in labels.value_counts().items():
        log.info(f"    {cls:15} : {n}")

    return labels.values


# ------------------------------------------------------------------ train
def train_classifier(X_train, y_train, X_test, y_test, feat_cols):
    log.info(f"\nTraining Random Forest ...")
    log.info(f"  Train : {len(X_train)}  Test : {len(X_test)}")
    log.info(f"  Features : {X_train.shape[1]}")

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_tr_s, y_train)

    y_pred = clf.predict(X_te_s)
    acc    = accuracy_score(y_test, y_pred)

    log.info(f"\n  Overall accuracy : {acc:.4f}")
    log.info("\n  Per class report:\n"
             + classification_report(y_test, y_pred, zero_division=0))

    log.info("  Per class accuracy:")
    for cls in sorted(set(y_test)):
        mask    = y_test == cls
        cls_acc = accuracy_score(y_test[mask], y_pred[mask])
        log.info(f"    {cls:15} : {cls_acc:.4f}  (n={mask.sum()})")

    importances = pd.Series(
        clf.feature_importances_, index=feat_cols
    ).sort_values(ascending=False)
    log.info("\n  Top 10 features:")
    for feat, imp in importances.head(10).items():
        log.info(f"    {feat:35} : {imp:.4f}")

    return clf, scaler, acc


# ------------------------------------------------------------------ fusion
def tiered_fusion(df, clf, scaler, threshold):
    """
    Strict AND fusion rule.
    THREAT  : AE fires AND freq fires (confidence >= FREQ_CONFIDENCE)
    WARNING : exactly one fires
    CLEAR   : neither fires

    No layer fires THREAT unilaterally. This is what prevents
    animal sounds from being escalated even though the AE
    correctly flags them as non-normal.
    """
    log.info("\nApplying strict AND tiered fusion to all windows ...")

    X_all, _ = build_features(df)
    X_all_s  = scaler.transform(X_all)

    freq_proba  = clf.predict_proba(X_all_s)
    freq_pred   = clf.predict(X_all_s)
    threat_idx  = [list(clf.classes_).index(c)
                   for c in THREAT_CLASSES if c in list(clf.classes_)]
    threat_conf = freq_proba[:, threat_idx].sum(axis=1)
    freq_fires  = threat_conf >= FREQ_CONFIDENCE

    ae_fires = df["recon_error"].values > threshold

    # strict AND — both must agree
    verdict = np.where(
        ae_fires & freq_fires,  "THREAT",
        np.where(ae_fires | freq_fires, "WARNING",
                                         "CLEAR")
    )

    df = df.copy()
    df["freq_class"]  = freq_pred
    df["freq_conf"]   = threat_conf
    df["freq_fires"]  = freq_fires
    df["ae_fires"]    = ae_fires
    df["verdict"]     = verdict

    log.info("\nFusion verdict distribution:")
    for v in ["THREAT", "WARNING", "CLEAR"]:
        n = (verdict == v).sum()
        log.info(f"  {v:7} : {n:6}  ({100*n/len(verdict):.1f}%)")

    return df


# ------------------------------------------------------------------ metrics
def fusion_metrics(df):
    log.info("\nFusion performance metrics:")

    tp = ((df["verdict"] == "THREAT") & (df["role"] == "threat")).sum()
    fp = ((df["verdict"] == "THREAT") &
          (df["role"].isin(["normal", "false_alarm"]))).sum()
    fn = ((df["verdict"] != "THREAT") & (df["role"] == "threat")).sum()

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    fa_on_animal = ((df["verdict"] == "THREAT") &
                    (df["role"] == "false_alarm")).sum()
    fa_on_normal = ((df["verdict"] == "THREAT") &
                    (df["role"] == "normal")).sum()

    log.info(f"  True positives       : {tp}")
    log.info(f"  False positives      : {fp}")
    log.info(f"    of which animal    : {fa_on_animal}")
    log.info(f"    of which normal    : {fa_on_normal}")
    log.info(f"  Precision            : {precision:.4f}")
    log.info(f"  Recall               : {recall:.4f}")
    log.info(f"  F1                   : {f1:.4f}")

    log.info("\n  Detection by label:")
    for lbl in ["gunshot", "gun_shot", "siren", "engine_idling",
                 "car_horn", "jackhammer", "drilling", "explosion"]:
        subset = df[df["label"] == lbl]
        if len(subset) == 0:
            continue
        detected = (subset["verdict"] == "THREAT").sum()
        log.info(f"    {lbl:20} : {detected}/{len(subset)} "
                 f"({100*detected/len(subset):.0f}%)")

    log.info("\n  False alarm check on animal/false_alarm class:")
    fa_df = df[df["role"] == "false_alarm"]
    if len(fa_df):
        alerted = (fa_df["verdict"] == "THREAT").sum()
        rate    = alerted / len(fa_df)
        log.info(f"    Total false_alarm windows : {len(fa_df)}")
        log.info(f"    Incorrectly alerted       : {alerted}")
        log.info(f"    Correctly suppressed      : {len(fa_df) - alerted}")
        log.info(f"    False alarm rate          : {rate:.4f}  "
                 f"({'GOOD' if rate < 0.10 else 'WARN — above 10%'})")

    # WARNING tier check — threats that appear at least as WARNING
    warn_or_threat = df["verdict"].isin(["THREAT", "WARNING"])
    tp_warn = ((warn_or_threat) & (df["role"] == "threat")).sum()
    recall_warn = tp_warn / (tp_warn + fn + 1e-10)
    log.info(f"\n  Recall at THREAT+WARNING : {recall_warn:.4f}")
    log.info(f"  (threats reaching operator attention at any severity)")

    stats = {
        "precision"      : round(float(precision), 4),
        "recall"         : round(float(recall), 4),
        "f1"             : round(float(f1), 4),
        "recall_warning" : round(float(recall_warn), 4),
        "true_pos"       : int(tp),
        "false_pos"      : int(fp),
        "fa_on_animal"   : int(fa_on_animal),
        "fa_on_normal"   : int(fa_on_normal),
    }
    return stats


# ------------------------------------------------------------------ save
def save_results(clf, scaler, df, stats, feat_cols, acc):
    with open(CACHE / "freq_classifier.pkl", "wb") as fh:
        pickle.dump(clf, fh)
    log.info(f"\n  Classifier saved -> {CACHE}/freq_classifier.pkl")

    with open(CACHE / "freq_scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    log.info(f"  Scaler saved     -> {CACHE}/freq_scaler.pkl")

    df.to_csv(CACHE / "phase3_results.csv", index=False)
    log.info(f"  Results saved    -> {CACHE}/phase3_results.csv")

    with open(CACHE / "thresholds.json") as fh:
        thresh = json.load(fh)
    thresh.update({
        "freq_confidence"   : FREQ_CONFIDENCE,
        "fusion_rule"       : "strict_AND",
        "fusion_precision"  : stats["precision"],
        "fusion_recall"     : stats["recall"],
        "fusion_f1"         : stats["f1"],
        "recall_at_warning" : stats["recall_warning"],
        "freq_layer_acc"    : round(acc, 4),
    })
    with open(CACHE / "thresholds.json", "w") as fh:
        json.dump(thresh, fh, indent=2)
    log.info(f"  Thresholds updated -> {CACHE}/thresholds.json")

    with open(CACHE / "phase3_report.txt", "w") as fh:
        fh.write("PHASE 3 REPORT — FREQUENCY LAYER + TIERED FUSION\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"  Fusion rule               : strict AND\n")
        fh.write(f"  Freq confidence threshold : {FREQ_CONFIDENCE}\n\n")
        fh.write("FUSION METRICS\n")
        for k, v in stats.items():
            fh.write(f"  {k:25} : {v}\n")
        fh.write(f"\n  freq_layer_accuracy       : {acc:.4f}\n")
    log.info(f"  Report saved     -> {CACHE}/phase3_report.txt")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    t0 = time.time()
    log.info("PERIMETER THREAT SYSTEM  —  Phase 3 Frequency Layer + Fusion")

    df, threshold, healthy_max = load_data()

    X, feat_cols = build_features(df)
    y = map_labels(df)

    test_mask  = df["fold"].astype(str).isin(["9", "10", "test"])
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    log.info(f"\n  Train : {train_mask.sum()}  Test : {test_mask.sum()}")

    clf, scaler, acc = train_classifier(
        X_train, y_train, X_test, y_test, feat_cols)

    df_fused = tiered_fusion(df, clf, scaler, threshold)

    stats = fusion_metrics(df_fused)

    save_results(clf, scaler, df_fused, stats, feat_cols, acc)

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    print("\n" + "=" * 60)
    print("  PHASE 3 COMPLETE")
    print("=" * 60)
    print(f"  Fusion rule          : strict AND (both layers required)")
    print(f"  Freq confidence      : >= {FREQ_CONFIDENCE}")
    print(f"  Freq layer accuracy  : {acc:.4f}")
    print(f"  Fusion precision     : {stats['precision']}")
    print(f"  Fusion recall        : {stats['recall']}")
    print(f"  Recall at WARNING+   : {stats['recall_warning']}")
    print(f"  Fusion F1            : {stats['f1']}")
    print(f"  True positives       : {stats['true_pos']}")
    print(f"  False positives      : {stats['false_pos']}")
    print(f"    on animal class    : {stats['fa_on_animal']}")
    print(f"    on normal class    : {stats['fa_on_normal']}")
    print()
    fa_rate = stats["fa_on_animal"] / 10485
    if fa_rate < 0.10:
        print(f"  False alarm rate : {fa_rate:.4f}  GOOD")
    else:
        print(f"  False alarm rate : {fa_rate:.4f}  WARN")
    print("\n  Ready for Phase 4 — severity scoring + localization.")