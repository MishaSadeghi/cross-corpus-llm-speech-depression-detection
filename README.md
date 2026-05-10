# Cross-Cultural Depression Detection in Structured Clinical Interviews

Code for the paper:

> **Cross-Cultural Depression Detection in Structured Clinical Interviews**  
> *Under review*

---

## Overview

This repository contains the full pipeline for multimodal depression detection
from clinical interview data, combining LLM-based text features, wav2vec2 speech
features, and eGeMAPSv02 acoustic features.

The pipeline includes:
- **ASR transcripts** → LLM (Llama) summary + diagnostic Q&A → DepRoBERTa probabilities → 14-D text feature vector
- **Wav2Vec2 XLSR-53** audio embeddings (6144-D, German or multilingual)
- **eGeMAPSv02** LLD-based acoustic features (200-D)
- **Nested cross-validation** and **fixed-split** evaluation frameworks
- **Cross-corpus** transfer experiments

---

## Repository Structure

```
audio/          Diarization, patient audio extraction, and feature extraction scripts
text/           LLM-based text feature extraction pipeline
nested_cv/      Within-corpus nested 5-fold cross-validation experiments (Tables 1–2)
fixed_split/    Fixed-split evaluations for proposed dataset and E-DAIC (Table 4–5)
cross_corpus/   Cross-dataset transfer classification (Table 5)
utils/          Shared utilities (split saving)
```

---

## Configuration

All scripts use `CONFIGURE_ME/...` path placeholders. Before running, replace
these with your actual paths by editing the configuration section at the top of
each script.

### Key path constants

| Placeholder | Description |
|---|---|
| `CONFIGURE_ME/proposed_dataset` | Root of the clinical interview dataset |
| `CONFIGURE_ME/proposed_dataset/features_output` | Pre-extracted feature CSVs and results |
| `CONFIGURE_ME/labels/classi_labels.csv` | Binary depression labels (SCID-5-CV) |
| `CONFIGURE_ME/labels/regress_labels.csv` | PHQ-9 continuous scores |
| `CONFIGURE_ME/labels/edaic_labels.csv` | E-DAIC labels (train/dev/test split info) |
| `CONFIGURE_ME/storage` | Large-file storage (models, results) |
| `CONFIGURE_ME/storage/models/hf_cache` | HuggingFace model cache |
| `CONFIGURE_ME/LLM_summaries/proposed_dataset/summaries_3108` | Llama 3.1-8B output summaries |
| `CONFIGURE_ME/LLM_summaries/proposed_dataset/summaries_3170` | Llama 3.1-70B output summaries |
| `CONFIGURE_ME/LLM_summaries/proposed_dataset/summaries_3370` | Llama 3.3-70B output summaries |
| `CONFIGURE_ME/features/text_features_llama*.csv` | Pre-computed 11-D question score CSVs |
| `CONFIGURE_ME/E-DAIC/extracted` | E-DAIC audio directory |

---

## Dataset

**Proposed dataset (German):** A novel German-language clinical interview corpus
collected under institutional ethical approval. Labels are not included in this
repository and will be released separately after publication. The dataset
consists of 252 participants (130 MDD+, 122 controls) diagnosed using SCID-5-CV.

**E-DAIC:** Publicly available English interview dataset.
See [E-DAIC](https://dcapswoz.ict.usc.edu/).

---

## Audio Pipeline — Proposed Dataset

The proposed dataset requires speaker diarization before feature extraction because
the full interview recording contains both the interviewer and the patient.

```
Step 1  audio/run_diarization_proposed.py   raw WAV  →  RTTM files
                                            (pyannote/speaker-diarization-3.1)

Step 2  audio/diarize_proposed.py           RTTM  →  patient-only WAV
                                            (longest-duration speaker heuristic +
                                             bandpass filter + normalization)

Step 3  Feature extraction — choose one or more:
        audio/extract_german_xlsr_proposed.py       →  6144-D  (nested CV, Tables 1–2)
        audio/extract_xlsr_multilingual_proposed.py →  6144-D  (fixed split / cross-corpus)
        audio/extract_egemaps_proposed.py           →   200-D  (all settings)
```

**Note on Step 1:** The pyannote model is gated on HuggingFace Hub.
Accept the license at `https://huggingface.co/pyannote/speaker-diarization-3.1`
and export your token before running: `export HF_TOKEN="hf_..."`.

## Audio Pipeline — E-DAIC

E-DAIC provides transcript turn timestamps, so no diarization is needed.
Features are extracted directly from the full audio using transcript boundaries.

```
audio/extract_xlsr_multilingual_edaic.py   →  6144-D  (cross-corpus, Table 5)
audio/extract_egemaps_edaic.py             →   200-D  (cross-corpus, Table 5)
```

## Audio Models

| Use case | Model | Script |
|---|---|---|
| Within-corpus nested CV (Tables 1–2) | `jonatasgrosman/wav2vec2-large-xlsr-53-german` | `audio/extract_german_xlsr_proposed.py` |
| Fixed split + cross-corpus (Tables 4–5) | `facebook/wav2vec2-large-xlsr-53` | `audio/extract_xlsr_multilingual_proposed.py` |
| E-DAIC | `facebook/wav2vec2-large-xlsr-53` | `audio/extract_xlsr_multilingual_edaic.py` |
| All datasets (eGeMAPS) | openSMILE eGeMAPSv02 | `audio/extract_egemaps_*.py` |

---

## Text Pipeline

1. `text/llm_extract_summaries.py` — Llama generates ~300-word clinical summaries
2. `text/llm_extract_questions.py` — Llama answers 11 DSM-5 diagnostic questions
3. `text/parse_llm_answers.py` — Parses free-form LLM answers to canonical labels
4. `text/map_answers_to_scores.py` — Maps labels to numeric scores with mean imputation
5. `text/deproberta_inference.py` — DepRoBERTa (`rafalposwiata/deproberta-large-v1`) inference
6. `text/aggregate_deproberta_probs.py` — Aggregates JSON predictions to a single CSV

The 14-D text feature vector = 3 DepRoBERTa probabilities + 11 question scores.

**Note:** DepRoBERTa must be fine-tuned inside each CV fold to prevent data
leakage. The `nested_cv/` scripts handle this automatically.

---

## Excluded Participants

Patient IDs `{177, 207, 299}` are always excluded (consent/technical issues).

---

## Requirements

```
torch >= 2.0
transformers >= 4.35
scikit-learn >= 1.3
pandas, numpy, scipy
opensmile
librosa, soundfile
xgboost, lightgbm, catboost
shap
tqdm
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{anonymous2025cross,
  title={Cross-Cultural Depression Detection in Structured Clinical Interviews},
  year={2026}
}
```
