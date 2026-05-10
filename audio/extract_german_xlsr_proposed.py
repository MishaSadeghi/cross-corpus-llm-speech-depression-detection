"""
Extract jonatasgrosman/wav2vec2-large-xlsr-53-german features from the proposed dataset.

This script is used for within-corpus nested cross-validation experiments
(Tables 1 & 2 in the paper).

Two modes:
  --mode diarised      → patient-only audio: interview_t1_patient_audio/wav/{PID}_patient.wav
  --mode non_diarised  → full interview audio: interview_t1_audio/{PID}_audio.wav

Feature extraction:
  - Model : jonatasgrosman/wav2vec2-large-xlsr-53-german  (1024-dim hidden states)
  - Audio is processed in 20-sec chunks; last_hidden_state mean-pooled per chunk
  - 6 statistics across chunks: min, max, mean, var, skew, kurt
  - Output: 6 × 1024 = 6144 features per participant

Output saved to:
  FEATURES_BASE/proposed_dataset_german_xlsr_{mode}/proposed_dataset_german_xlsr_{mode}.csv
"""

import argparse
import logging
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import kurtosis, skew
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configure these paths for your environment ─────────────────────────────────
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")   # root of the clinical interview data
FEATURES_BASE     = DATASET_ROOT / "features_output"

# Audio roots for the two extraction modes
AUDIO_ROOTS = {
    "diarised":     DATASET_ROOT / "interview_t1_patient_audio" / "wav",
    "non_diarised": DATASET_ROOT / "interview_t1_audio",
}
AUDIO_PATTERNS = {
    "diarised":     "{pid:03d}_patient.wav",
    "non_diarised": "{pid:03d}_audio.wav",
}

# Labels CSV (binary depression labels); provides the list of participant IDs
LABELS_PATH = Path("CONFIGURE_ME/labels/classi_labels.csv")

# Local model directory (optional; falls back to HuggingFace Hub)
MODEL_LOCAL = Path("CONFIGURE_ME/models/wav2vec2-large-xlsr-53-german")
MODEL_HUB   = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
MODEL_CACHE = Path("CONFIGURE_ME/models/hf_cache")
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SR     = 16_000
CHUNK_SECONDS = 20.0


def load_audio(wav_path: Path) -> np.ndarray:
    signal, sr = sf.read(str(wav_path))
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
    return signal.astype(np.float32)


@torch.no_grad()
def extract_chunks(signal: np.ndarray, feature_extractor, model, device) -> np.ndarray:
    """Extract mean-pooled last_hidden_state per 20-sec chunk. Returns (N_chunks × 1024)."""
    chunk_size = int(CHUNK_SECONDS * TARGET_SR)
    min_samples = TARGET_SR // 2  # skip chunks shorter than 0.5 s
    embeddings = []

    for start in range(0, len(signal), chunk_size):
        chunk = signal[start: start + chunk_size]
        if len(chunk) < min_samples:
            continue
        inputs = feature_extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hs = model(**inputs).last_hidden_state  # (1, T, 1024)
        emb = hs.mean(dim=1).squeeze(0).cpu().numpy()  # (1024,)
        embeddings.append(emb)

    return np.array(embeddings) if embeddings else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def aggregate_embeddings(embs: np.ndarray) -> dict:
    """6 statistics (min/max/mean/var/skew/kurt) per dimension → 6144 features."""
    feats = {}
    for i in range(embs.shape[1]):
        col = embs[:, i]
        feats[f"min_{i}"]  = float(np.min(col))
        feats[f"max_{i}"]  = float(np.max(col))
        feats[f"mean_{i}"] = float(np.mean(col))
        feats[f"var_{i}"]  = float(np.var(col))
        feats[f"skew_{i}"] = float(skew(col))
        feats[f"kurt_{i}"] = float(kurtosis(col))
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["diarised", "non_diarised"])
    args = parser.parse_args()
    mode = args.mode

    out_dir = FEATURES_BASE / f"proposed_dataset_german_xlsr_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"proposed_dataset_german_xlsr_{mode}.csv"

    audio_root = AUDIO_ROOTS[mode]
    audio_pat  = AUDIO_PATTERNS[mode]

    labels = pd.read_csv(LABELS_PATH)
    pids   = sorted(labels["patient_id"].unique())
    logging.info(f"Participants: {len(pids)}  |  Mode: {mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    model_name   = str(MODEL_LOCAL) if MODEL_LOCAL.exists() else MODEL_HUB
    load_kwargs  = {} if MODEL_LOCAL.exists() else {"cache_dir": str(MODEL_CACHE)}
    logging.info(f"Loading model: {model_name}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, **load_kwargs)
    model             = Wav2Vec2Model.from_pretrained(model_name, **load_kwargs).to(device)
    model.eval()
    hidden_size = model.config.hidden_size
    logging.info(f"Hidden size: {hidden_size}")

    rows, missing = [], []

    for pid in tqdm(pids, desc="Participants"):
        wav_path = audio_root / audio_pat.format(pid=pid)
        if not wav_path.exists():
            logging.warning(f"  PID {pid}: audio not found at {wav_path}")
            missing.append(pid)
            continue
        try:
            signal = load_audio(wav_path)
        except Exception as e:
            logging.error(f"  PID {pid}: audio load error: {e}")
            missing.append(pid)
            continue

        try:
            embs = extract_chunks(signal, feature_extractor, model, device)
        except Exception as e:
            logging.error(f"  PID {pid}: extraction error: {e}")
            missing.append(pid)
            continue

        if embs.shape[0] == 0:
            logging.warning(f"  PID {pid}: no valid chunks extracted")
            missing.append(pid)
            continue

        feats = aggregate_embeddings(embs)
        feats["patient_id"]  = pid
        feats["n_segments"]  = embs.shape[0]
        rows.append(feats)
        logging.info(f"  PID {pid}: {embs.shape[0]} chunks → {len(feats)-2} features")

    if not rows:
        logging.error("No participants processed. Exiting.")
        return

    df = pd.DataFrame(rows)
    meta = ["patient_id", "n_segments"]
    feat_cols = [c for c in df.columns if c not in meta]
    df = df[meta + feat_cols]
    df.to_csv(out_csv, index=False)

    logging.info(f"\n{'='*60}")
    logging.info(f"EXTRACTION COMPLETE — Proposed Dataset German XLSR-53 ({mode})")
    logging.info(f"  Model:     {model_name}")
    logging.info(f"  Processed: {len(rows)} / {len(pids)}")
    logging.info(f"  Missing:   {len(missing)} → {missing[:10]}")
    logging.info(f"  Features:  {len(feat_cols)} (6 stats × {hidden_size} dims)")
    logging.info(f"  Saved:     {out_csv}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
