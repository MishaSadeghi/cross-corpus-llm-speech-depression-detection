#!/usr/bin/env python3
"""
Speaker Diarization for Proposed Dataset — generates RTTM files

Uses pyannote/speaker-diarization-3.1 to diarize full interview WAV files.
One RTTM file is written per participant; downstream `diarize_proposed.py`
reads these RTTM files to extract and enhance patient-only audio.

Pipeline position:
    raw WAV  →  [this script]  →  RTTM files
                                     ↓
                              diarize_proposed.py  →  patient WAV
                                                          ↓
                                           extract_*_proposed.py  →  feature CSVs

Requirements:
    pip install pyannote.audio torch

HuggingFace access:
    The model is gated. Accept the license at:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    Then set your token:  export HF_TOKEN="hf_..."

Configure the paths below before running.
"""

import multiprocessing as mp
import os
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration — update these paths before running
# =============================================================================

# Directory containing full interview WAV files ({PID}_audio.wav)
RAW_AUDIO_DIR = Path("CONFIGURE_ME/interview_audio")

# Directory where RTTM files will be written ({PID}_audio.rttm)
OUTPUT_RTTM_DIR = Path("CONFIGURE_ME/diarization/rttm")

# CSV with at minimum a 'patient_id' column listing valid participant IDs
LABELS_CSV = Path("CONFIGURE_ME/labels/classi_labels.csv")

# HuggingFace token — can also be set via HF_TOKEN environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Force exactly 2 speakers (interviewer + patient); set to None for auto-detect
NUM_SPEAKERS = 2

# =============================================================================


def normalize_id(pid) -> str | None:
    """Zero-pad numeric IDs to 3 digits; leave longer IDs as-is."""
    if pd.isna(pid):
        return None
    try:
        pid_int = int(float(pid))
        return str(pid_int).zfill(3) if pid_int < 100 else str(pid_int)
    except (ValueError, TypeError):
        return None


def get_valid_participant_ids() -> set:
    """Load valid participant IDs from the labels CSV."""
    df = pd.read_csv(LABELS_CSV)
    ids = df["patient_id"].map(normalize_id).dropna().tolist()
    return set(ids)


def save_rttm_file(annotation: Annotation, output_path: Path) -> None:
    """Write pyannote Annotation to standard RTTM format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_id = output_path.stem
    with open(output_path, "w") as f:
        for segment, _, speaker_label in annotation.itertracks(yield_label=True):
            f.write(
                f"SPEAKER {file_id} 1 {segment.start:.3f} {segment.duration:.3f}"
                f" <NA> <NA> {speaker_label} <NA> <NA>\n"
            )


def process_one_file(args: Tuple[Path, int, int]) -> Tuple[bool, str]:
    """Worker: diarize one audio file and write its RTTM."""
    warnings.filterwarnings("ignore")
    wav_path, worker_id, num_gpus = args

    device = (
        torch.device(f"cuda:{worker_id % num_gpus}")
        if num_gpus > 0
        else torch.device("cpu")
    )

    try:
        token = HF_TOKEN or None
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        pipeline.to(device)

        kwargs = {"num_speakers": NUM_SPEAKERS} if NUM_SPEAKERS is not None else {}
        diarization = pipeline(str(wav_path), **kwargs)

        out_path = OUTPUT_RTTM_DIR / f"{wav_path.stem}.rttm"
        save_rttm_file(diarization, out_path)

        return True, f"[OK]    {wav_path.name} → {out_path.name}"

    except Exception as e:
        return False, f"[ERROR] {wav_path.name}: {e}"


def main():
    OUTPUT_RTTM_DIR.mkdir(parents=True, exist_ok=True)

    if not HF_TOKEN:
        print(
            "WARNING: HF_TOKEN is not set. The pyannote model is gated — "
            "set export HF_TOKEN='hf_...' before running."
        )

    valid_pids = get_valid_participant_ids()
    print(f"Valid participants: {len(valid_pids)}")

    wav_files: List[Path] = []
    for wav_path in sorted(RAW_AUDIO_DIR.glob("*_audio.wav")):
        subject_id = normalize_id(wav_path.stem.replace("_audio", ""))
        if subject_id in valid_pids:
            wav_files.append(wav_path)
        else:
            print(f"  Skipping {wav_path.name} (not in labels CSV)")

    # Skip already-processed files
    wav_files = [
        p for p in wav_files
        if not (OUTPUT_RTTM_DIR / f"{p.stem}.rttm").exists()
    ]

    if not wav_files:
        print("Nothing to process — all RTTM files already exist.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\n--- Diarization configuration ---")
    print(f"  Input  : {RAW_AUDIO_DIR}")
    print(f"  Output : {OUTPUT_RTTM_DIR}")
    print(f"  Model  : pyannote/speaker-diarization-3.1")
    print(f"  Speakers: {NUM_SPEAKERS if NUM_SPEAKERS else 'auto-detect'}")
    print(f"  Files to process: {len(wav_files)}")
    print(f"  GPUs available  : {num_gpus if num_gpus > 0 else 'none (CPU)'}")
    print("-" * 35)

    tasks = [(wav, i, num_gpus) for i, wav in enumerate(wav_files)]

    if num_gpus > 1:
        with mp.Pool(processes=num_gpus) as pool:
            results = pool.map(process_one_file, tasks)
    else:
        results = [process_one_file(t) for t in tasks]

    success = sum(1 for ok, _ in results if ok)
    for _, msg in results:
        print(msg)

    print(f"\n--- Complete ---")
    print(f"  Processed : {len(wav_files)}")
    print(f"  Succeeded : {success}")
    print(f"  Failed    : {len(wav_files) - success}")
    print(f"  RTTM dir  : {OUTPUT_RTTM_DIR}")
    print("\nNext step: run audio/diarize_proposed.py to extract patient WAVs from the RTTM files.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
