"""
Extract eGeMAPSv02 LLD features from the proposed dataset clinical interviews.

Two modes:
  --mode diarised      → patient-only audio from interview_t1_patient_audio/wav/{PID}_patient.wav
  --mode non_diarised  → full interview audio from interview_t1_audio/{PID}_audio.wav

For each audio file:
  1. Extract 25 eGeMAPSv02 LLDs at frame level via openSMILE
  2. Aggregate per participant with 8 statistics:
     mean, std, min, max, median, IQR, skewness, kurtosis
  → 25 × 8 = 200 features per participant

Output saved to:
  FEATURES_BASE/proposed_dataset_egemapsv02_{mode}/proposed_dataset_egemapsv02_{mode}.csv
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
from scipy.stats import iqr, kurtosis, skew

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configure these paths for your environment ─────────────────────────────────
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")   # root of the clinical interview data
FEATURES_BASE     = DATASET_ROOT / "features_output"

AUDIO_ROOTS = {
    "diarised":     DATASET_ROOT / "interview_t1_patient_audio" / "wav",
    "non_diarised": DATASET_ROOT / "interview_t1_audio",
}
AUDIO_PATTERNS = {
    "diarised":     "{pid}_patient.wav",
    "non_diarised": "{pid}_audio.wav",
}

# Labels CSV provides the list of participant IDs
LABELS_PATH = Path("CONFIGURE_ME/labels/classi_labels.csv")
# ──────────────────────────────────────────────────────────────────────────────

STAT_FUNCS = {
    "mean":     np.mean,
    "std":      np.std,
    "min":      np.min,
    "max":      np.max,
    "median":   np.median,
    "iqr":      lambda x: float(iqr(x)),
    "skewness": lambda x: float(skew(x)),
    "kurtosis": lambda x: float(kurtosis(x)),
}


def extract_and_aggregate(audio_path: Path, smile: opensmile.Smile) -> dict:
    try:
        llds = smile.process_file(str(audio_path))
    except Exception as e:
        logging.warning(f"  openSMILE failed on {audio_path.name}: {e}")
        return {}

    if llds.empty:
        return {}

    feats = {}
    for col in llds.columns:
        vals = llds[col].values.astype(float)
        for stat_name, stat_fn in STAT_FUNCS.items():
            feats[f"{col}_{stat_name}"] = stat_fn(vals)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["diarised", "non_diarised"])
    args = parser.parse_args()

    mode = args.mode
    audio_root = AUDIO_ROOTS[mode]
    pattern = AUDIO_PATTERNS[mode]

    out_dir = FEATURES_BASE / f"proposed_dataset_egemapsv02_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"proposed_dataset_egemapsv02_{mode}.csv"

    labels = pd.read_csv(LABELS_PATH)
    pids = sorted(labels['patient_id'].unique())
    logging.info(f"Mode: {mode} | Participants: {len(pids)} | Audio root: {audio_root}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    lld_names = smile.feature_names
    logging.info(f"eGeMAPSv02 LLDs: {len(lld_names)} features")

    # Checkpoint: load already-processed patients to enable resume
    checkpoint_csv = out_dir / f"proposed_dataset_egemapsv02_{mode}_checkpoint.csv"
    done_pids = set()
    if checkpoint_csv.exists():
        try:
            done_df = pd.read_csv(checkpoint_csv)
            done_pids = set(done_df["patient_id"].tolist())
            logging.info(f"Resuming — {len(done_pids)} patients already done: skipping them")
        except Exception as e:
            logging.warning(f"Could not read checkpoint: {e} — starting from scratch")

    remaining = [pid for pid in pids if pid not in done_pids]
    logging.info(f"Patients remaining: {len(remaining)} / {len(pids)}")

    rows = []
    missing = []

    for pid in remaining:
        wav = audio_root / pattern.format(pid=f"{pid:03d}")
        if not wav.exists():
            logging.warning(f"  PID {pid}: audio not found → {wav}")
            missing.append(pid)
            continue

        feats = extract_and_aggregate(wav, smile)
        if not feats:
            missing.append(pid)
            continue

        feats["patient_id"] = pid
        rows.append(feats)
        logging.info(f"  PID {pid}: {len(feats)-1} features extracted")

        # Append to checkpoint CSV after every patient
        row_df = pd.DataFrame([feats])
        write_header = not checkpoint_csv.exists()
        row_df.to_csv(checkpoint_csv, mode='a', header=write_header, index=False)

    # Merge checkpoint (previously done) + current run
    all_rows_df = pd.read_csv(checkpoint_csv) if checkpoint_csv.exists() else pd.DataFrame(rows)
    meta = ["patient_id"]
    feat_cols = [c for c in all_rows_df.columns if c not in meta]
    all_rows_df = all_rows_df[meta + sorted(feat_cols)]
    all_rows_df = all_rows_df.drop_duplicates(subset=["patient_id"])
    all_rows_df.to_csv(out_csv, index=False)

    logging.info(f"\n{'='*60}")
    logging.info(f"EXTRACTION COMPLETE — {mode}")
    logging.info(f"  Processed this run: {len(rows)}")
    logging.info(f"  Total in output:    {len(all_rows_df)} / {len(pids)}")
    logging.info(f"  Missing this run:   {len(missing)} → {missing[:10]}{'...' if len(missing)>10 else ''}")
    logging.info(f"  Features:           {len(feat_cols)} ({len(lld_names)} LLDs × {len(STAT_FUNCS)} stats)")
    logging.info(f"  Saved:              {out_csv}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
