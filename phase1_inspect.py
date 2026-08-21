"""
Phase 1 — Dataset Inspector
Run this first. It walks all four datasets and tells you exactly what is
inside each one: file counts, audio formats, duration samples, class labels.
No extraction happens here — this is just verification before we commit to
the full pipeline.

Usage:
    python phase1_inspect.py
"""

import os
import csv
import json
import random
from pathlib import Path
from collections import Counter

try:
    import soundfile as sf
    SF_OK = True
except Exception:
    SF_OK = False

# ------------------------------------------------------------------ config
BASE      = Path(__file__).parent
DATASETS  = BASE / "datasets"

ESC50     = DATASETS / "ESC-50"
SESA      = DATASETS / "SESA"
GUNSHOT   = DATASETS / "edge-collected-gunshot-audio"
US8K      = DATASETS / "UrbanSound8K"

AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aif"}


# ------------------------------------------------------------------ helpers
def find_audio(root: Path):
    files = []
    for f in root.rglob("*"):
        if f.suffix.lower() in AUDIO_EXT:
            files.append(f)
    return files


def sample_durations(files, n=10):
    """Return durations in seconds for up to n random files."""
    if not SF_OK:
        return []
    sample = random.sample(files, min(n, len(files)))
    durations = []
    for f in sample:
        try:
            info = sf.info(str(f))
            durations.append(round(info.duration, 2))
        except Exception:
            pass
    return sorted(durations)


def hr(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(label, condition, ok_msg="", fail_msg=""):
    icon = "OK  " if condition else "FAIL"
    msg  = ok_msg if condition else fail_msg
    print(f"  [{icon}]  {label}  {msg}")
    return condition


# ================================================================= ESC-50
def inspect_esc50():
    hr("ESC-50  —  environmental baseline and false-alarm classes")

    audio_dir = ESC50 / "audio"
    meta_file = ESC50 / "meta" / "esc50.csv"

    check("audio/ folder exists",    audio_dir.exists(), str(audio_dir))
    check("meta/esc50.csv exists",   meta_file.exists(), str(meta_file))

    files = find_audio(ESC50)
    print(f"\n  Total audio files found : {len(files)}")

    ext_counts = Counter(f.suffix.lower() for f in files)
    print(f"  Formats                 : {dict(ext_counts)}")

    if files:
        durs = sample_durations(files)
        print(f"  Sample durations (s)    : {durs}")

    # read meta if available
    if meta_file.exists():
        classes = set()
        folds   = set()
        with open(meta_file, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                classes.add(row.get("category", ""))
                folds.add(row.get("fold", ""))
        print(f"  Unique classes          : {len(classes)}")
        print(f"  Folds                   : {sorted(folds)}")
        print(f"  First 10 classes        : {sorted(classes)[:10]}")
    print()


# ================================================================= SESA
def inspect_sesa():
    hr("SESA  —  surveillance threat sounds")

    files = find_audio(SESA)
    print(f"\n  Total audio files found : {len(files)}")

    ext_counts = Counter(f.suffix.lower() for f in files)
    print(f"  Formats                 : {dict(ext_counts)}")

    # list subdirectories — class structure lives here
    subdirs = sorted([d.name for d in SESA.iterdir() if d.is_dir()])
    print(f"  Subdirectories          : {subdirs}")

    if files:
        durs = sample_durations(files)
        print(f"  Sample durations (s)    : {durs}")

    # class counts from folder names
    class_counts = {}
    for d in SESA.iterdir():
        if d.is_dir():
            n = len(find_audio(d))
            if n:
                class_counts[d.name] = n
    if class_counts:
        print(f"  Files per class         : {class_counts}")
    print()


# ================================================================= Gunshot
def inspect_gunshot():
    hr("Edge-collected gunshot audio  —  multi-device DOA dataset")

    files = find_audio(GUNSHOT)
    print(f"\n  Total audio files found : {len(files)}")

    ext_counts = Counter(f.suffix.lower() for f in files)
    print(f"  Formats                 : {dict(ext_counts)}")

    subdirs = sorted([d.name for d in GUNSHOT.iterdir() if d.is_dir()])
    print(f"  Subdirectories          : {subdirs}")

    # check CSVs
    csvs = list(GUNSHOT.glob("*.csv"))
    print(f"  CSV metadata files      : {[c.name for c in csvs]}")

    if csvs:
        with open(csvs[0], newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames
            rows    = list(reader)
        print(f"  CSV columns             : {headers}")
        print(f"  CSV rows                : {len(rows)}")

    if files:
        durs = sample_durations(files)
        print(f"  Sample durations (s)    : {durs}")
    print()


# ================================================================= US8K
def inspect_us8k():
    hr("UrbanSound8K  —  urban noise + gun_shot class")

    files = find_audio(US8K)
    print(f"\n  Total audio files found : {len(files)}")

    ext_counts = Counter(f.suffix.lower() for f in files)
    print(f"  Formats                 : {dict(ext_counts)}")

    folds = sorted([d.name for d in US8K.iterdir() if d.is_dir()])
    print(f"  Folds found             : {folds}")

    # read metadata CSV
    meta = US8K / "UrbanSound8K.csv"
    if meta.exists():
        class_counts = Counter()
        fold_counts  = Counter()
        with open(meta, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                class_counts[row.get("class", "")] += 1
                fold_counts[row.get("fold", "")] += 1
        print(f"  Unique classes          : {len(class_counts)}")
        print(f"  Classes                 : {sorted(class_counts.keys())}")
        print(f"  Files per fold          : { {k: fold_counts[k] for k in sorted(fold_counts)} }")
    else:
        print("  WARN: UrbanSound8K.csv not found in dataset root")

    if files:
        durs = sample_durations(files)
        print(f"  Sample durations (s)    : {durs}")
    print()


# ================================================================= summary
def summary():
    hr("SUMMARY")

    rows = [
        ("ESC-50",               ESC50,   "Baseline + false-alarm classes"),
        ("SESA",                 SESA,    "Surveillance threat events"),
        ("Gunshot DOA",          GUNSHOT, "Multi-device, direction-of-arrival"),
        ("UrbanSound8K",         US8K,    "Urban noise + gun_shot class"),
    ]

    all_ok = True
    for name, path, role in rows:
        exists  = path.exists()
        n_files = len(find_audio(path)) if exists else 0
        status  = "READY" if (exists and n_files > 0) else "MISSING"
        if status != "READY":
            all_ok = False
        print(f"  {status:7}  {name:25}  {n_files:5} files  —  {role}")

    print()
    if all_ok:
        print("  All datasets present and non-empty.")
        print("  Ready to run phase1_extract.py")
    else:
        print("  One or more datasets missing — check paths above.")
    print()


# ================================================================= main
if __name__ == "__main__":
    random.seed(42)
    print("\nPERIMETER THREAT SYSTEM  —  Phase 1 Dataset Inspector")
    print(f"Base path : {BASE}")
    print(f"Datasets  : {DATASETS}")

    for path, name in [(ESC50, "ESC-50"), (SESA, "SESA"),
                       (GUNSHOT, "edge-collected-gunshot-audio"),
                       (US8K, "UrbanSound8K")]:
        if not path.exists():
            print(f"\n  WARN: {name} not found at {path}")

    inspect_esc50()
    inspect_sesa()
    inspect_gunshot()
    inspect_us8k()
    summary()