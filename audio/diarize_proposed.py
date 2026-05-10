#!/usr/bin/env python3
"""
Extract Patient Speech from RTTM Diarization Files

This script:
1. Parses RTTM diarization files to extract speaker segments
2. Identifies the patient speaker (speaker with the longest total duration)
3. Extracts and concatenates only the patient's audio segments
4. Optionally applies audio enhancement: bandpass filter, spectral gating,
   peak normalization, and resampling to 16 kHz
5. Saves per-participant patient WAV, segment timestamps CSV, and metadata JSON

Configure the paths below before running.
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.effects import normalize
from scipy import signal
from scipy.io import wavfile

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration — update these paths before running
# =============================================================================

# Directory containing per-participant RTTM files ({PID}_audio.rttm)
RTTM_DIR = Path("CONFIGURE_ME/diarization/rttm")

# Directory containing full interview WAV files ({PID}_audio.wav)
RAW_AUDIO_DIR = Path("CONFIGURE_ME/interview_audio")

# Output directory; subdirs wav/, segments/, metadata/ are created automatically
OUTPUT_DIR = Path("CONFIGURE_ME/patient_audio")

# CSV with at minimum a 'patient_id' column listing valid participant IDs
LABELS_CSV = Path("CONFIGURE_ME/labels/classi_labels.csv")

# Audio settings
TARGET_SAMPLE_RATE = 16000
HIGHPASS_CUTOFF = 80    # Hz — remove low-frequency rumble
LOWPASS_CUTOFF = 8000   # Hz — remove noise above speech range

# =============================================================================
# Setup output directories
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "wav").mkdir(exist_ok=True)
(OUTPUT_DIR / "segments").mkdir(exist_ok=True)
(OUTPUT_DIR / "metadata").mkdir(exist_ok=True)


# =============================================================================
# Audio enhancement
# =============================================================================

def apply_bandpass_filter(audio_data, sample_rate, lowcut=80, highcut=8000):
    """Apply 4th-order Butterworth bandpass filter."""
    nyquist = sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(4, [low, high], btype="band", output="sos")
    return signal.sosfilt(sos, audio_data)


def reduce_noise_simple(audio_data, sample_rate, noise_reduction_strength=0.3):
    """Spectral gating: attenuate frames below the 20th-percentile energy threshold."""
    frame_length = int(sample_rate * 0.05)  # 50 ms frames
    hop_length = frame_length // 2
    num_frames = (len(audio_data) - frame_length) // hop_length + 1

    energies = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= len(audio_data):
            energies[i] = np.sqrt(np.mean(audio_data[start:end] ** 2))

    noise_threshold = np.percentile(energies, 20)
    result = audio_data.copy()
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= len(audio_data):
            if energies[i] < noise_threshold * (1 + noise_reduction_strength):
                attenuation = max(0.1, 1.0 - noise_reduction_strength)
                result[start:end] *= attenuation
    return result


def enhance_audio(audio_segment):
    """
    Full enhancement pipeline:
      1. Bandpass filter (80–8000 Hz)
      2. Spectral gating noise reduction
      3. Peak normalization
      4. Resample to TARGET_SAMPLE_RATE if needed
    """
    print("    Applying audio enhancement...", flush=True)

    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    if audio_segment.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif audio_segment.sample_width == 1:
        samples = samples.astype(np.float32) / 128.0

    sr = audio_segment.frame_rate
    samples = apply_bandpass_filter(samples, sr, HIGHPASS_CUTOFF, LOWPASS_CUTOFF)
    samples = reduce_noise_simple(samples, sr, noise_reduction_strength=0.3)
    samples = samples / (np.abs(samples).max() + 1e-8)
    samples = (samples * 32767).astype(np.int16)

    enhanced = AudioSegment(
        samples.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )
    enhanced = normalize(enhanced)
    if sr != TARGET_SAMPLE_RATE:
        enhanced = enhanced.set_frame_rate(TARGET_SAMPLE_RATE)

    print("    Enhancement complete!", flush=True)
    return enhanced


# =============================================================================
# Helper functions
# =============================================================================

def normalize_id(pid):
    """Zero-pad numeric IDs to 3 digits; leave longer IDs as-is."""
    if pd.isna(pid):
        return None
    try:
        pid_int = int(float(pid))
        return str(pid_int).zfill(3) if pid_int < 100 else str(pid_int)
    except (ValueError, TypeError):
        return None


def get_valid_participant_ids():
    """Load valid participant IDs from the labels CSV."""
    df = pd.read_csv(LABELS_CSV)
    ids = df["patient_id"].map(normalize_id).dropna().tolist()
    return set(ids)


def parse_rttm_file(rttm_path):
    """Parse an RTTM file; return list of {start, duration, end, speaker} dicts."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            segments.append({
                "start": start,
                "duration": dur,
                "end": start + dur,
                "speaker": parts[7],
            })
    return segments


def identify_patient_speaker(segments):
    """Patient = speaker with longest cumulative duration."""
    stats = defaultdict(lambda: {"duration": 0.0, "segments": 0})
    for seg in segments:
        stats[seg["speaker"]]["duration"] += seg["duration"]
        stats[seg["speaker"]]["segments"] += 1
    patient = max(stats, key=lambda s: stats[s]["duration"])
    return patient, dict(stats)


def extract_patient_segments(segments, patient_speaker):
    return [s for s in segments if s["speaker"] == patient_speaker]


def extract_patient_audio(audio_path, patient_segments, output_path, use_enhancement=False):
    """Concatenate patient segments from the full interview WAV and export."""
    print(f"  Loading audio: {audio_path.name}", flush=True)
    audio = AudioSegment.from_wav(str(audio_path))

    extracted = AudioSegment.empty()
    for seg in patient_segments:
        extracted += audio[int(seg["start"] * 1000):int(seg["end"] * 1000)]

    original_duration = len(audio) / 1000
    extracted_duration = len(extracted) / 1000

    if use_enhancement:
        extracted = enhance_audio(extracted)

    print(f"  Saving patient audio...", flush=True)
    extracted.export(str(output_path), format="wav")
    return extracted_duration, original_duration


def save_segments_file(patient_segments, output_path):
    pd.DataFrame([
        {"start": s["start"], "end": s["end"], "duration": s["duration"]}
        for s in patient_segments
    ]).to_csv(output_path, index=False)


def save_metadata(subject_id, patient_speaker, speaker_stats,
                  extracted_duration, original_duration, output_path):
    metadata = {
        "subject_id": subject_id,
        "patient_speaker": patient_speaker,
        "speaker_statistics": speaker_stats,
        "extracted_duration_seconds": extracted_duration,
        "original_duration_seconds": original_duration,
        "extraction_ratio": extracted_duration / original_duration if original_duration > 0 else 0,
        "audio_settings": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "highpass_filter_hz": HIGHPASS_CUTOFF,
            "lowpass_filter_hz": LOWPASS_CUTOFF,
        },
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# Per-subject processing
# =============================================================================

def process_single_subject(subject_id, force=False, use_enhancement=False):
    rttm_file = RTTM_DIR / f"{subject_id}_audio.rttm"
    audio_file = RAW_AUDIO_DIR / f"{subject_id}_audio.wav"
    wav_out = OUTPUT_DIR / "wav" / f"{subject_id}_patient.wav"
    seg_out = OUTPUT_DIR / "segments" / f"{subject_id}_segments.csv"
    meta_out = OUTPUT_DIR / "metadata" / f"{subject_id}_metadata.json"

    if wav_out.exists() and not force:
        print(f"✓ Already processed: {subject_id}", flush=True)
        return True

    for fpath, label in [(rttm_file, "RTTM"), (audio_file, "audio")]:
        if not fpath.exists():
            print(f"✗ {label} file not found: {fpath}", flush=True)
            return False

    print(f"\n>>> Processing: {subject_id}", flush=True)
    try:
        segments = parse_rttm_file(rttm_file)
        print(f"  Found {len(segments)} total segments", flush=True)

        patient_speaker, speaker_stats = identify_patient_speaker(segments)
        for spk, stats in sorted(speaker_stats.items()):
            tag = " <- PATIENT" if spk == patient_speaker else ""
            print(f"    {spk}: {stats['duration']:.1f}s, "
                  f"{stats['segments']} segments{tag}", flush=True)

        patient_segments = extract_patient_segments(segments, patient_speaker)
        print(f"  Patient segments: {len(patient_segments)}", flush=True)

        ext_dur, orig_dur = extract_patient_audio(
            audio_file, patient_segments, wav_out, use_enhancement=use_enhancement
        )
        print(f"  Extracted {ext_dur:.1f}s / {orig_dur:.1f}s "
              f"({100 * ext_dur / orig_dur:.1f}%)", flush=True)

        save_segments_file(patient_segments, seg_out)
        save_metadata(subject_id, patient_speaker, speaker_stats,
                      ext_dur, orig_dur, meta_out)

        print(f"✓ Done: {subject_id}", flush=True)
        return True

    except Exception as e:
        import traceback
        print(f"✗ ERROR processing {subject_id}: {e}", flush=True)
        traceback.print_exc()
        return False


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract patient speech from RTTM diarization files"
    )
    parser.add_argument("--test-single", type=str,
                        help="Process only a single participant ID (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess participants even if output already exists")
    parser.add_argument("--enhance", action="store_true",
                        help="Apply audio enhancement (bandpass filter + noise reduction + normalization)")
    args = parser.parse_args()

    print(f"[CONFIG] Audio enhancement: {'ENABLED' if args.enhance else 'DISABLED'}", flush=True)

    valid_pids = get_valid_participant_ids()
    print(f"Valid participants: {len(valid_pids)}", flush=True)

    rttm_files = sorted(RTTM_DIR.glob("*_audio.rttm"))
    print(f"RTTM files found: {len(rttm_files)}", flush=True)

    processed = failed = 0
    for rttm_file in rttm_files:
        subject_id = normalize_id(rttm_file.stem.replace("_audio", ""))

        if args.test_single and subject_id != args.test_single:
            continue

        if subject_id not in valid_pids:
            print(f"Skipping {subject_id} (not in labels CSV)", flush=True)
            continue

        success = process_single_subject(
            subject_id, force=args.force, use_enhancement=args.enhance
        )
        if success:
            processed += 1
        else:
            failed += 1

    print(f"\n{'='*60}", flush=True)
    print(f"Processing complete!", flush=True)
    print(f"  Processed : {processed}", flush=True)
    print(f"  Failed    : {failed}", flush=True)
    print(f"  Output    : {OUTPUT_DIR}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
