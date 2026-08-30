#!/usr/bin/env python
"""
Generate the CANONICAL 5-fold cross-validation splits for the analysis.

Single source of truth: every within-corpus and cross-corpus script reads folds
from cv_folds.json so that all analyses in the paper use byte-for-byte identical
splits. This file is meant to be shared alongside the dataset for replication.

Folds: StratifiedKFold(5, shuffle=True, random_state=42) on the binary
SCID labels, with participants sorted by patient_id (deterministic ordering).
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RR = Path(__file__).resolve().parent
LABELS = Path("CONFIGURE_ME/labels/classi_labels.csv")
OUT = RR / "cv_folds.json"

df = pd.read_csv(LABELS)
df["patient_id"] = df["patient_id"].astype(int)
df = df.sort_values("patient_id").reset_index(drop=True)   # deterministic order
pids = df["patient_id"].to_numpy()
y = df["depressed"].to_numpy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = {}
for k, (tr, te) in enumerate(skf.split(pids, y)):
    folds[f"fold_{k}"] = {
        "train_ids": [int(x) for x in pids[tr]],
        "test_ids":  [int(x) for x in pids[te]],
    }

meta = {
    "n_participants": int(len(pids)),
    "n_mdd_pos": int(y.sum()),
    "n_mdd_neg": int((y == 0).sum()),
    "ordering": "sorted_by_patient_id",
    "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "label_source": "classi_labels.csv (SCID-5-CV binary labels)",
    "folds": folds,
}
with open(OUT, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Wrote {OUT}")
for k in folds:
    tr, te = folds[k]["train_ids"], folds[k]["test_ids"]
    print(f"  {k}: train={len(tr)} test={len(te)} test_pos={sum(1 for p in te if int(df.set_index('patient_id').loc[p,'depressed'])==1)}")
