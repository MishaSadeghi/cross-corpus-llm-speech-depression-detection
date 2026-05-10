"""
Save the fixed train/test split (Fold 1 = median fold) for reproducibility.

The median fold was selected from 5-fold StratifiedKFold CV on the text
classification task (Llama 3.3 70B, no_filter). Folds ranked by mean F1
across all classifiers:
    Fold 3: 0.784 (worst)
    Fold 0: 0.792
    Fold 1: 0.845  ← MEDIAN
    Fold 2: 0.876
    Fold 4: 0.885 (best)

Also saves E-DAIC official train/dev/test split IDs.

Outputs saved to:
  FEATURES_BASE/splits/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ── Configure these paths for your environment ─────────────────────────────────
FEATURES_BASE = Path("CONFIGURE_ME/proposed_dataset/features_output")
SPLITS_DIR = FEATURES_BASE / "splits"

PROPOSED_CLASSI_LABELS  = Path("CONFIGURE_ME/labels/classi_labels.csv")
PROPOSED_REGRESS_LABELS = Path("CONFIGURE_ME/labels/regress_labels.csv")
EDAIC_LABELS            = Path("CONFIGURE_ME/labels/edaic_labels.csv")
# ──────────────────────────────────────────────────────────────────────────────

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

MEDIAN_FOLD = 1
CV_SEED = 42
N_SPLITS = 5


def save_proposed_split():
    labels = pd.read_csv(PROPOSED_CLASSI_LABELS)
    all_ids = labels['patient_id'].values
    y = labels['depressed'].values

    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(all_ids, y)):
        if fold == MEDIAN_FOLD:
            train_ids = sorted(all_ids[train_idx].tolist())
            test_ids = sorted(all_ids[test_idx].tolist())

            split_info = {
                "dataset": "proposed_dataset",
                "fold": MEDIAN_FOLD,
                "cv_n_splits": N_SPLITS,
                "cv_random_state": CV_SEED,
                "selection_criterion": "median mean-F1 across classifiers (Llama 3.3 70B, no_filter)",
                "train_ids": train_ids,
                "test_ids": test_ids,
                "train_n": len(train_ids),
                "test_n": len(test_ids),
                "train_depressed": int(sum(y[train_idx])),
                "train_healthy": int(len(train_idx) - sum(y[train_idx])),
                "test_depressed": int(sum(y[test_idx])),
                "test_healthy": int(len(test_idx) - sum(y[test_idx])),
            }

            out = SPLITS_DIR / "proposed_dataset_fold1_split.json"
            with open(out, 'w') as f:
                json.dump(split_info, f, indent=2)
            print(f"Proposed dataset split saved to {out}")
            print(f"  Train: {split_info['train_n']} ({split_info['train_depressed']} dep / {split_info['train_healthy']} healthy)")
            print(f"  Test:  {split_info['test_n']} ({split_info['test_depressed']} dep / {split_info['test_healthy']} healthy)")

            # Also save PHQ-thresholded labels for cross-corpus setting 2
            regress = pd.read_csv(PROPOSED_REGRESS_LABELS)
            phq_scores = regress.set_index('patient_id')['PHQ9-Score']
            phq_labels = {}
            for pid in train_ids + test_ids:
                score = phq_scores.get(pid, np.nan)
                if pd.notna(score):
                    phq_labels[str(pid)] = int(float(score) >= 10)
            split_info["phq_threshold_labels"] = phq_labels
            out2 = SPLITS_DIR / "proposed_dataset_fold1_split_with_phq_labels.json"
            with open(out2, 'w') as f:
                json.dump(split_info, f, indent=2)
            print(f"  PHQ-thresholded labels also saved to {out2}")
            return


def save_edaic_split():
    labels = pd.read_csv(EDAIC_LABELS)

    split_info = {
        "dataset": "edaic",
        "note": "Official E-DAIC train/dev/test split",
    }
    for s in ['train', 'dev', 'test']:
        subset = labels[labels['split'] == s]
        ids = sorted(subset['participant_id'].tolist())
        n_dep = int(subset['depressed'].sum())
        split_info[f"{s}_ids"] = ids
        split_info[f"{s}_n"] = len(ids)
        split_info[f"{s}_depressed"] = n_dep
        split_info[f"{s}_healthy"] = len(ids) - n_dep
        print(f"  E-DAIC {s}: {len(ids)} ({n_dep} dep / {len(ids)-n_dep} healthy)")

    out = SPLITS_DIR / "edaic_official_split.json"
    with open(out, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"E-DAIC split saved to {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("SAVING SPLITS FOR REPRODUCIBILITY")
    print("=" * 60)
    save_proposed_split()
    print()
    save_edaic_split()
    print("\nDone.")
