"""
Extract eGeMAPSv02 LLD features from E-DAIC audio (transcript-bounded).

For each participant:
  1. Load the full audio from {PID}_P/{PID}_AUDIO.wav
  2. Read {PID}_P/{PID}_Transcript.csv for Start_Time / End_Time boundaries
  3. Concatenate audio from Start_Time to End_Time of each transcript row
     (this captures patient speech, excluding interviewer instructions)
  4. Extract 25 eGeMAPSv02 LLDs at frame level via openSMILE on the concatenated audio
  5. Aggregate per participant with 8 statistics:
     mean, std, min, max, median, IQR, skewness, kurtosis
  → 25 × 8 = 200 features per participant

Output saved to:
  FEATURES_BASE/edaic_egemapsv02/edaic_egemapsv02.csv
"""

import logging
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from scipy.stats import iqr, kurtosis, skew

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configure these paths for your environment ─────────────────────────────────
EDAIC_ROOT  = Path("CONFIGURE_ME/E-DAIC/extracted")   # directory containing {PID}_P/ subdirectories
LABELS_PATH = Path("CONFIGURE_ME/labels/edaic_labels.csv")
FEATURES_BASE    = Path("CONFIGURE_ME/proposed_dataset/features_output")
OUT_DIR     = FEATURES_BASE / "edaic_egemapsv02"
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
MIN_DURATION = 0.1

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


def load_audio(wav_path: Path) -> tuple:
    signal, sr = sf.read(str(wav_path))
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return signal, sr


def get_transcript_segments(signal: np.ndarray, sr: int, transcript_path: Path) -> np.ndarray:
    df = pd.read_csv(transcript_path)
    if 'Start_Time' not in df.columns or 'End_Time' not in df.columns:
        raise ValueError(f"Missing Start_Time/End_Time columns in {transcript_path}")

    parts = []
    for _, row in df.iterrows():
        start_sample = int(row['Start_Time'] * sr)
        end_sample = int(row['End_Time'] * sr)
        if end_sample > len(signal):
            end_sample = len(signal)
        if start_sample >= len(signal):
            continue
        seg = signal[start_sample:end_sample]
        if len(seg) >= int(MIN_DURATION * sr):
            parts.append(seg)

    if not parts:
        raise ValueError(f"No valid segments from transcript {transcript_path}")

    return np.concatenate(parts)


def aggregate_llds(llds: pd.DataFrame) -> dict:
    feats = {}
    for col in llds.columns:
        vals = llds[col].values.astype(float)
        for stat_name, stat_fn in STAT_FUNCS.items():
            feats[f"{col}_{stat_name}"] = stat_fn(vals)
    return feats


def main():
    labels = pd.read_csv(LABELS_PATH)
    pids = sorted(labels['participant_id'].unique())
    logging.info(f"Participants to process: {len(pids)}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    lld_names = smile.feature_names
    logging.info(f"eGeMAPSv02 LLDs: {len(lld_names)} features")

    rows = []
    missing = []

    for pid in pids:
        pid_dir = EDAIC_ROOT / f"{pid}_P"
        wav_path = pid_dir / f"{pid}_AUDIO.wav"
        tsv_path = pid_dir / f"{pid}_Transcript.csv"

        if not wav_path.exists():
            logging.warning(f"  PID {pid}: audio not found → {wav_path}")
            missing.append(pid)
            continue

        try:
            signal, sr = load_audio(wav_path)
        except Exception as e:
            logging.error(f"  PID {pid}: audio load error: {e}")
            missing.append(pid)
            continue

        if not tsv_path.exists():
            logging.warning(f"  PID {pid}: transcript not found → {tsv_path}")
            missing.append(pid)
            continue

        try:
            concat_audio = get_transcript_segments(signal, sr, tsv_path)
        except Exception as e:
            logging.warning(f"  PID {pid}: transcript segmentation failed: {e}")
            missing.append(pid)
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, concat_audio, sr)
            try:
                llds = smile.process_file(tmp.name)
            except Exception as e:
                logging.error(f"  PID {pid}: openSMILE extraction failed: {e}")
                missing.append(pid)
                continue

        if llds.empty:
            logging.warning(f"  PID {pid}: empty LLD output")
            missing.append(pid)
            continue

        feats = aggregate_llds(llds)
        feats["participant_id"] = pid
        rows.append(feats)
        logging.info(f"  PID {pid}: {len(concat_audio)/sr:.1f}s of audio → {len(feats)-1} features")

    df = pd.DataFrame(rows)
    meta = ["participant_id"]
    feat_cols = [c for c in df.columns if c not in meta]
    df = df[meta + sorted(feat_cols)]
    out_csv = OUT_DIR / "edaic_egemapsv02.csv"
    df.to_csv(out_csv, index=False)

    logging.info(f"\n{'='*60}")
    logging.info(f"EXTRACTION COMPLETE — E-DAIC transcript-bounded eGeMAPSv02")
    logging.info(f"  Processed: {len(rows)} / {len(pids)}")
    logging.info(f"  Missing:   {len(missing)} → {missing[:10]}{'...' if len(missing)>10 else ''}")
    logging.info(f"  Features:  {len(feat_cols)} ({len(lld_names)} LLDs × {len(STAT_FUNCS)} stats)")
    logging.info(f"  Saved:     {out_csv}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
