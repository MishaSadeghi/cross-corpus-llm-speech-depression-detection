# File Mapping: Original → Public Repository

This document maps every original script (with its full HPC path) to the new
clean filename in this public release.  Only final scripts used for the paper
are included; superseded experiments are omitted.

## audio/

| New filename | Original internal filename |
|---|---|
| `audio/run_diarization_proposed.py` | `internal_scripts/audio/Audio_diarization.py` |
| `audio/diarize_proposed.py` | `internal_scripts/audio/extract_patient_from_rttm.py` |
| `audio/extract_egemaps_proposed.py` | `internal_scripts/extract_proposed_dataset_egemapsv02.py` |
| `audio/extract_egemaps_edaic.py` | `internal_scripts/extract_edaic_egemapsv02.py` |
| `audio/extract_german_xlsr_proposed.py` | *(new — written for this release)* |
| `audio/extract_xlsr_multilingual_proposed.py` | `internal_scripts/Audio_new/extract_proposed_dataset_xlsr53_multilingual.py` |
| `audio/extract_xlsr_multilingual_edaic.py` | `internal_scripts/Audio_new/extract_edaic_xlsr53_multilingual.py` |

## text/

| New filename | Original path |
|---|---|
| `text/llm_extract_summaries.py` | `Text/Text_extract_summary.py` |
| `text/llm_extract_questions.py` | `Text/Text_extract_questions.py` |
| `text/parse_llm_answers.py` | `Sup/Sup_LLL_ans_aggregate.py` |
| `text/map_answers_to_scores.py` | `Sup/Sup_LLM_ans_score_mapping.py` |
| `text/deproberta_inference.py` | `Text/Text_DepRoBERTa_inference.py` |
| `text/aggregate_deproberta_probs.py` | `Sup/Sup_DepRoBERTa_probs_aggrerate.py` |

## nested_cv/

| New filename | Original path |
|---|---|
| `nested_cv/text_classification.py` | `Text/maximal_text_classification_cv_deproberta.py` |
| `nested_cv/text_regression.py` | `Text/maximal_text_regression_cv_deproberta.py` |
| `nested_cv/audio_classification.py` | `Audio/wav2vec_experiments/maximal_audio_classification_v2.py` |
| `nested_cv/audio_regression.py` | `Audio/wav2vec_experiments/maximal_audio_regression_v2.py` |
| `nested_cv/multimodal_classification.py` | `Multi/multimodal_classification_cv_deproberta.py` |
| `nested_cv/multimodal_regression.py` | `Multi/multimodal_regression_cv_deproberta.py` |

## fixed_split/

| New filename | Original path |
|---|---|
| `fixed_split/proposed_text.py` | `internal_scripts/proposed_dataset_text_fixed_split_llm.py` |
| `fixed_split/proposed_audio.py` | `internal_scripts/proposed_dataset_audio_fixed_split_xlsr53ml.py` |
| `fixed_split/proposed_multimodal.py` | `internal_scripts/proposed_dataset_multimodal_fixed_split_xlsr53ml.py` |
| `fixed_split/edaic_text.py` | `internal_scripts/edaic_text_classification_devopt.py` |
| `fixed_split/edaic_text_regression.py` | `internal_scripts/edaic_text_regression_devopt.py` |
| `fixed_split/edaic_audio.py` | `internal_scripts/edaic_audio_within_xlsr53ml.py` |
| `fixed_split/edaic_multimodal.py` | `internal_scripts/edaic_multimodal_within_llm.py` |

## cross_corpus/

| New filename | Original path |
|---|---|
| `cross_corpus/classification.py` | `internal_scripts/Audio_new/cross_corpus_classification_v2_llm_xlsr53ml.py` |
| `cross_corpus/classification_phq8.py` | `internal_scripts/cross_corpus_classification_v2_llm_phq8.py` |
| `cross_corpus/shap_analysis.py` | `internal_scripts/Audio_new/cross_corpus_shap_best_multimodal.py` |

## utils/

| New filename | Original path |
|---|---|
| `utils/save_splits.py` | `internal_scripts/save_splits.py` |

---

## Changes made during anonymization

1. **Identifiers removed**: all occurrences of the private corpus name, internal
   project folder names, and author username removed from file contents and names.
2. **Absolute paths replaced**: all absolute storage paths replaced with
   configurable `Path("CONFIGURE_ME/...")` placeholders at the top of each file.
   Set these before running.
3. **Label files not included**: `classi_labels.csv` and `regress_labels.csv`
   will be released separately upon paper acceptance.
4. **Job files excluded**: SLURM `.sh` and `.sh.o*` files omitted.
5. **Superseded scripts excluded**: older versions with data leakage or incomplete
   feature selection are not included.
6. **New script added**: `audio/extract_german_xlsr_proposed.py` extracts
   `jonatasgrosman/wav2vec2-large-xlsr-53-german` features (used for Tables 1–2).

## Notes on audio feature models

- **Tables 1–2 (nested CV, within-corpus)**: German XLSR
  `jonatasgrosman/wav2vec2-large-xlsr-53-german` → `audio/extract_german_xlsr_proposed.py`
- **Table 5 (fixed split + cross-corpus)**: Multilingual XLSR
  `facebook/wav2vec2-large-xlsr-53` → `audio/extract_xlsr_multilingual_proposed.py`
  (same model used for E-DAIC to ensure language-agnostic comparability)
