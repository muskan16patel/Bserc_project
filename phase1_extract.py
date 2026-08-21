"""
Phase 1 — Feature Extraction Pipeline
Reads all four datasets, extracts features from every audio file,
and saves them to cache/ as numpy arrays with a metadata CSV.

Features extracted per window:
  - Mel spectrogram (128 bands)
  - Band energies across 7 frequency bands
  - MFCC coefficients (13)
  - Kurtosis (impulsiveness — cheap footstep detector)
  - RMS energy

Everything is tagged with source, label, fold, and whether
the data is real or synthetic, so results can never be
accidentally mixed.

Usage:
    python phase1_extract.py

Output:
    cache/features.npz   — all feature arrays
    cache/metadata.csv   — one row per file, provenance attached
    cache/extract_log.txt — log of what was processed
"""

import os
import csv
import json
import time
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import librosa
import soundfile as sf
from scipy.stats import kurtosis
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config
BASE     = Path(__file__).parent
DATASETS = BASE / "datasets"
CACHE    = BASE / "cache"
CACHE.mkdir(exist_ok=True)

ESC50    = DATASETS / "ESC-50"
SESA     = DATASETS / "SESA"
GUNSHOT  = DATASETS / "edge-collected-gunshot-audio"
US8K     = DATASETS / "UrbanSound8K"

# audio processing
SR          = 22050   # target sample rate for all files
WIN_SEC     = 2.0     # window length in seconds
HOP_SEC     = 1.0     # hop between windows (50% overlap)
WIN_SAMPLES = int(WIN_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)
N_FFT       = 1024
HOP_FFT     = 256
N_MELS      = 128
N_MFCC      = 13

# frequency bands of interest (Hz)
# chosen from published field work on perimeter sensing
BANDS = {
    "sub_bass"  : (20,   60),    # seismic footstep energy
    "bass"      : (60,  250),    # footstep acoustic, vehicle low
    "vehicle"   : (10,   20),    # vehicle seismic dominant (~15 Hz)
    "low_mid"   : (250, 500),
    "mid"       : (500, 2000),
    "high_mid"  : (2000, 4000),
    "high"      : (4000, 8000),
}

# class mapping — what role each class plays in our system
# normal/baseline = what the autoencoder trains on
# false_alarm     = things the system must stay silent about
# threat          = things the system must detect
ESC50_ROLES = {
    # ---- false alarm (must suppress)
    "dog"               : "false_alarm",
    "cat"               : "false_alarm",
    "crow"              : "false_alarm",
    "frog"              : "false_alarm",
    "insects"           : "false_alarm",
    "chirping_birds"    : "false_alarm",
    "hen"               : "false_alarm",
    "pig"               : "false_alarm",
    "cow"               : "false_alarm",
    "sheep"             : "false_alarm",
    # ---- normal baseline
    "wind"              : "normal",
    "rain"              : "normal",
    "sea_waves"         : "normal",
    "crackling_fire"    : "normal",
    "water_drops"       : "normal",
    "thunderstorm"      : "normal",
    "crickets"          : "normal",
    # ---- threat adjacent
    "chainsaw"          : "threat",
    "helicopter"        : "threat",
    "engine"            : "threat",
    "car_horn"          : "threat",
    "siren"             : "threat",
    "door_wood_knock"   : "threat",
    "footsteps"         : "threat",
    # ---- background / other
    "clapping"          : "background",
    "brushing_teeth"    : "background",
    "snoring"           : "background",
    "can_opening"       : "background",
    "church_bells"      : "background",
    "clock_alarm"       : "background",
    "clock_tick"        : "background",
    "coughing"          : "background",
    "crying_baby"       : "background",
    "drinking_sipping"  : "background",
    "hand_saw"          : "background",
    "keyboard_typing"   : "background",
    "laughing"          : "background",
    "mouse_click"       : "background",
    "sneezing"          : "background",
    "toilet_flush"      : "background",
    "vacuum_cleaner"    : "background",
    "washing_machine"   : "background",
    "airplane"          : "background",
    "breathing"         : "background",
    "pouring_water"     : "background",
    "glass_breaking"    : "threat",
}

US8K_ROLES = {
    "air_conditioner" : "normal",
    "car_horn"        : "threat",
    "children_playing": "false_alarm",
    "dog_bark"        : "false_alarm",
    "drilling"        : "threat",
    "engine_idling"   : "threat",
    "gun_shot"        : "threat",
    "jackhammer"      : "threat",
    "siren"           : "threat",
    "street_music"    : "background",
}

SESA_ROLES = {
    "0": "false_alarm",   # casual
    "1": "threat",        # gunshot
    "2": "threat",        # explosion
    "3": "threat",        # siren
}

# ------------------------------------------------------------------ logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(CACHE / "extract_log.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger()


# ------------------------------------------------------------------ core extraction
def load_audio(path: Path, sr: int = SR):
    """Load audio file, convert to mono, resample to sr."""
    try:
        y, orig_sr = librosa.load(str(path), sr=sr, mono=True)
        return y, sr
    except Exception as e:
        log.warning(f"Failed to load {path.name}: {e}")
        return None, None


def compute_band_energy(y: np.ndarray, sr: int) -> dict:
    """Compute energy in each frequency band via FFT."""
    n      = len(y)
    fft    = np.abs(np.fft.rfft(y, n=n))
    freqs  = np.fft.rfftfreq(n, 1.0 / sr)
    energies = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        energies[name] = float(np.sum(fft[mask] ** 2)) if mask.any() else 0.0
    return energies


def extract_features(y: np.ndarray, sr: int) -> dict | None:
    """
    Extract all features from one audio window.
    Returns a flat dict of feature values.
    """
    if y is None or len(y) < N_FFT:
        return None

    # pad short windows
    if len(y) < WIN_SAMPLES:
        y = np.pad(y, (0, WIN_SAMPLES - len(y)))

    features = {}

    # --- mel spectrogram (flattened mean + std per band)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features["mel_mean"] = mel_db.mean(axis=1).tolist()   # N_MELS values
    features["mel_std"]  = mel_db.std(axis=1).tolist()

    # --- MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_FFT)
    features["mfcc_mean"] = mfcc.mean(axis=1).tolist()
    features["mfcc_std"]  = mfcc.std(axis=1).tolist()

    # --- band energies
    band_e = compute_band_energy(y, sr)
    total  = sum(band_e.values()) + 1e-10
    for name, e in band_e.items():
        features[f"band_{name}_energy"] = e
        features[f"band_{name}_ratio"]  = e / total

    # --- impulsiveness (kurtosis — key for footstep detection)
    features["kurtosis"]    = float(kurtosis(y, fisher=True))

    # --- RMS energy
    features["rms"]         = float(np.sqrt(np.mean(y ** 2)))

    # --- zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_FFT)
    features["zcr_mean"]    = float(zcr.mean())
    features["zcr_std"]     = float(zcr.std())

    # --- spectral centroid
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    features["spectral_centroid_mean"] = float(cent.mean())

    # --- spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    features["spectral_rolloff_mean"] = float(rolloff.mean())

    return features


def process_file(path: Path, label: str, role: str, fold: str,
                 source: str, data_type: str = "real") -> list[dict]:
    """
    Load a file, window it, extract features from each window.
    Returns a list of feature dicts, one per window.
    """
    y, sr = load_audio(path)
    if y is None:
        return []

    records = []
    starts  = range(0, max(1, len(y) - WIN_SAMPLES + 1), HOP_SAMPLES)

    for i, start in enumerate(starts):
        window = y[start: start + WIN_SAMPLES]
        feats  = extract_features(window, sr)
        if feats is None:
            continue

        record = {
            "source"    : source,
            "data_type" : data_type,
            "file"      : path.name,
            "label"     : label,
            "role"      : role,
            "fold"      : fold,
            "window_idx": i,
            "start_s"   : round(start / sr, 3),
        }
        record.update(feats)
        records.append(record)

    return records


# ------------------------------------------------------------------ dataset processors
def process_esc50(max_files=None):
    log.info("Processing ESC-50 ...")
    meta_path = ESC50 / "meta" / "esc50.csv"
    records   = []

    with open(meta_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if max_files:
        rows = rows[:max_files]

    for row in tqdm(rows, desc="ESC-50", ncols=70):
        fname    = row["filename"]
        category = row["category"]
        fold     = row["fold"]
        role     = ESC50_ROLES.get(category, "background")
        path     = ESC50 / "audio" / fname

        if not path.exists():
            continue

        recs = process_file(
            path, label=category, role=role,
            fold=fold, source="ESC50")
        records.extend(recs)

    log.info(f"  ESC-50: {len(rows)} files -> {len(records)} windows")
    return records


def process_sesa(max_files=None):
    log.info("Processing SESA ...")
    records = []

    # SESA has train/ and test/ subdirs
    # labels are encoded in filename or subfolder
    for split in ["train", "test"]:
        split_dir = SESA / split
        if not split_dir.exists():
            continue

        files = sorted(split_dir.glob("*.wav"))
        if max_files:
            files = files[:max_files // 2]

        for f in tqdm(files, desc=f"SESA {split}", ncols=70):
            # SESA filenames typically start with class id
            # 0=casual, 1=gunshot, 2=explosion, 3=siren
            stem = f.stem
            # try to extract class from first character
            cls_id = stem[0] if stem[0].isdigit() else "0"
            role   = SESA_ROLES.get(cls_id, "threat")
            label_map = {"0": "casual", "1": "gunshot",
                         "2": "explosion", "3": "siren"}
            label  = label_map.get(cls_id, "unknown")

            recs = process_file(
                f, label=label, role=role,
                fold=split, source="SESA")
            records.extend(recs)

    log.info(f"  SESA: -> {len(records)} windows")
    return records


def process_gunshot(max_files=None):
    log.info("Processing Gunshot DOA dataset ...")
    records   = []
    meta_path = GUNSHOT / "gunshot-audio-all-metadata.csv"

    meta = {}
    if meta_path.exists():
        with open(meta_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                meta[row["filename"]] = row

    files = sorted(GUNSHOT.rglob("*.wav"))
    if max_files:
        files = files[:max_files]

    for f in tqdm(files, desc="Gunshot", ncols=70):
        m      = meta.get(f.name, {})
        label  = "gunshot"
        role   = "threat"
        device = m.get("device_name", "unknown")
        fold   = m.get("firearm", f.parent.name)

        recs = process_file(
            f, label=label, role=role,
            fold=fold, source="GUNSHOT")

        # attach DOA metadata where available
        for r in recs:
            r["device"]    = device
            r["latitude"]  = m.get("latitude", "")
            r["longitude"] = m.get("longitude", "")
            r["firearm"]   = m.get("firearm", "")

        records.extend(recs)

    log.info(f"  Gunshot DOA: {len(files)} files -> {len(records)} windows")
    return records


def process_us8k(max_files=None):
    log.info("Processing UrbanSound8K ...")
    records  = []
    meta_path = US8K / "UrbanSound8K.csv"

    with open(meta_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if max_files:
        rows = rows[:max_files]

    for row in tqdm(rows, desc="US8K", ncols=70):
        fname = row["slice_file_name"]
        fold  = row["fold"]
        cls   = row["class"]
        role  = US8K_ROLES.get(cls, "background")
        path  = US8K / f"fold{fold}" / fname

        if not path.exists():
            continue

        recs = process_file(
            path, label=cls, role=role,
            fold=fold, source="US8K")
        records.extend(recs)

    log.info(f"  US8K: {len(rows)} files -> {len(records)} windows")
    return records


# ------------------------------------------------------------------ save
def save_cache(all_records: list[dict]):
    """
    Saves features to cache/ in two formats:
      1. metadata.csv    — provenance row per window (no arrays)
      2. features.npz    — numpy arrays for ML training
    """
    log.info(f"\nSaving {len(all_records)} total windows to cache/ ...")

    # separate scalar metadata from array features
    scalar_keys = ["source", "data_type", "file", "label", "role",
                   "fold", "window_idx", "start_s",
                   "kurtosis", "rms", "zcr_mean", "zcr_std",
                   "spectral_centroid_mean", "spectral_rolloff_mean",
                   ] + [f"band_{b}_{t}" for b in BANDS for t in ["energy", "ratio"]]

    meta_rows   = []
    feat_arrays = defaultdict(list)

    for r in all_records:
        # metadata row (CSV friendly)
        meta_row = {k: r.get(k, "") for k in scalar_keys}
        meta_rows.append(meta_row)

        # array features
        feat_arrays["mel_mean"].append(r.get("mel_mean", [0]*N_MELS))
        feat_arrays["mel_std"].append(r.get("mel_std",  [0]*N_MELS))
        feat_arrays["mfcc_mean"].append(r.get("mfcc_mean", [0]*N_MFCC))
        feat_arrays["mfcc_std"].append(r.get("mfcc_std",  [0]*N_MFCC))

    # save metadata CSV
    meta_path = CACHE / "metadata.csv"
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    log.info(f"  Metadata saved -> {meta_path}")

    # save numpy arrays
    npz_path = CACHE / "features.npz"
    np.savez_compressed(
        npz_path,
        mel_mean  = np.array(feat_arrays["mel_mean"],  dtype=np.float32),
        mel_std   = np.array(feat_arrays["mel_std"],   dtype=np.float32),
        mfcc_mean = np.array(feat_arrays["mfcc_mean"], dtype=np.float32),
        mfcc_std  = np.array(feat_arrays["mfcc_std"],  dtype=np.float32),
    )
    log.info(f"  Feature arrays saved -> {npz_path}")

    # print summary
    print("\n" + "="*60)
    print("  EXTRACTION COMPLETE")
    print("="*60)
    df = pd.DataFrame(meta_rows)
    print(f"\n  Total windows     : {len(df)}")
    print(f"\n  By source:")
    print(df["source"].value_counts().to_string())
    print(f"\n  By role:")
    print(df["role"].value_counts().to_string())
    print(f"\n  By label (top 15):")
    print(df["label"].value_counts().head(15).to_string())
    print(f"\n  Cache location    : {CACHE}")
    print(f"  metadata.csv      : {len(df)} rows")
    npz = np.load(npz_path)
    print(f"  features.npz      : mel_mean shape {npz['mel_mean'].shape}")
    print()


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    t0 = time.time()

    log.info("PERIMETER THREAT SYSTEM  —  Phase 1 Feature Extraction")
    log.info(f"Base: {BASE}")
    log.info(f"Cache: {CACHE}")
    log.info(f"Window: {WIN_SEC}s  Hop: {HOP_SEC}s  SR: {SR} Hz\n")

    all_records = []

    # Set MAX_FILES to a small number (e.g. 50) for a quick test run.
    # Set to None to process everything (takes 20-40 min for all 13k files).
    MAX_FILES = None

    all_records += process_esc50(MAX_FILES)
    all_records += process_sesa(MAX_FILES)
    all_records += process_gunshot(MAX_FILES)
    all_records += process_us8k(MAX_FILES)

    save_cache(all_records)

    elapsed = time.time() - t0
    log.info(f"Total time: {elapsed/60:.1f} minutes")
    log.info("Ready for Phase 2 — autoencoder training.")