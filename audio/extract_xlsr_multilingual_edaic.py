"""
Extract facebook/wav2vec2-large-xlsr-53 (multilingual) features from E-DAIC.

Uses participant-speech segments defined by transcript timestamps (Start_Time / End_Time)
to isolate patient speech.

Feature extraction:
  - Model : facebook/wav2vec2-large-xlsr-53  (1024-dim hidden states)
  - Per segment: run model → mean-pool last_hidden_state → 1024-dim embedding
  - 6 statistics across all segments: min, max, mean, var, skew, kurt
  - Output: 6 × 1024 = 6144 features per participant

Output saved to:
  FEATURES_BASE/edaic_xlsr53_multilingual/edaic_xlsr53_multilingual.csv
"""

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
EDAIC_ROOT  = Path("CONFIGURE_ME/E-DAIC/extracted")   # directory containing {PID}_P/ subdirectories
LABELS_PATH = Path("CONFIGURE_ME/labels/edaic_labels.csv")

FEATURES_BASE = Path("CONFIGURE_ME/proposed_dataset/features_output")
OUT_DIR  = FEATURES_BASE / "edaic_xlsr53_multilingual"

# Local model directory (optional; falls back to HuggingFace Hub)
MODEL_LOCAL = Path("CONFIGURE_ME/models/wav2vec2-large-xlsr-53")
MODEL_HUB   = "facebook/wav2vec2-large-xlsr-53"
MODEL_CACHE = Path("CONFIGURE_ME/models/hf_cache")
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SR    = 16_000
MIN_DURATION = 0.3   # seconds — skip segments shorter than this


def load_audio(wav_path: Path) -> tuple:
    signal, sr = sf.read(str(wav_path))
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return signal.astype(np.float32), sr


def get_transcript_segments(signal: np.ndarray, sr: int, transcript_path: Path) -> list:
    """Return list of audio arrays for participant speech segments."""
    df = pd.read_csv(transcript_path)
    if "Start_Time" not in df.columns or "End_Time" not in df.columns:
        return []

    segments = []
    for _, row in df.iterrows():
        start = int(row["Start_Time"] * sr)
        end   = int(row["End_Time"]   * sr)
        end   = min(end, len(signal))
        if start >= len(signal):
            continue
        seg = signal[start:end]
        if len(seg) >= int(MIN_DURATION * sr):
            segments.append(seg)
    return segments


@torch.no_grad()
def embed_segment(segment: np.ndarray, feature_extractor, model, device) -> np.ndarray:
    """Run XLSR-53 on one segment; return mean-pooled 1024-dim embedding."""
    inputs = feature_extractor(segment, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hs  = model(**inputs).last_hidden_state  # (1, T, 1024)
    emb = hs.mean(dim=1).squeeze(0).cpu().numpy()  # (1024,)
    return emb


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
    labels = pd.read_csv(LABELS_PATH)
    pids   = sorted(labels["participant_id"].unique())
    logging.info(f"Participants: {len(pids)}")

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
        pid_dir  = EDAIC_ROOT / f"{pid}_P"
        wav_path = pid_dir / f"{pid}_AUDIO.wav"
        tsv_path = pid_dir / f"{pid}_Transcript.csv"

        if not wav_path.exists():
            logging.warning(f"  PID {pid}: audio not found")
            missing.append(pid)
            continue

        try:
            signal, sr = load_audio(wav_path)
        except Exception as e:
            logging.error(f"  PID {pid}: audio load error: {e}")
            missing.append(pid)
            continue

        if not tsv_path.exists():
            logging.warning(f"  PID {pid}: transcript not found, skipping")
            missing.append(pid)
            continue

        segments = get_transcript_segments(signal, sr, tsv_path)
        if not segments:
            logging.warning(f"  PID {pid}: no valid segments from transcript")
            missing.append(pid)
            continue

        embs = []
        for seg in segments:
            try:
                emb = embed_segment(seg, feature_extractor, model, device)
                embs.append(emb)
            except Exception as e:
                logging.debug(f"  PID {pid}: segment error: {e}")

        if not embs:
            logging.warning(f"  PID {pid}: all segments failed")
            missing.append(pid)
            continue

        embs_np = np.array(embs)
        feats   = aggregate_embeddings(embs_np)
        feats["participant_id"] = pid
        feats["n_segments"]     = len(embs)
        rows.append(feats)
        logging.info(f"  PID {pid}: {len(embs)} segments → {len(feats)-2} features")

    if not rows:
        logging.error("No participants processed. Exiting.")
        return

    df = pd.DataFrame(rows)
    meta      = ["participant_id", "n_segments"]
    feat_cols = [c for c in df.columns if c not in meta]
    df = df[meta + feat_cols]
    out_csv = OUT_DIR / "edaic_xlsr53_multilingual.csv"
    df.to_csv(out_csv, index=False)

    logging.info(f"\n{'='*60}")
    logging.info(f"EXTRACTION COMPLETE — E-DAIC XLSR-53 Multilingual")
    logging.info(f"  Model:     {model_name}")
    logging.info(f"  Processed: {len(rows)} / {len(pids)}")
    logging.info(f"  Missing:   {len(missing)} → {missing[:10]}")
    logging.info(f"  Features:  {len(feat_cols)} (6 stats × {hidden_size} dims)")
    logging.info(f"  Saved:     {out_csv}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
