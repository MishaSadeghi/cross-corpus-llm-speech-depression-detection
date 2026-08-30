"""Shared loader for the canonical CV splits (cv_folds.json).

Every within-corpus and cross-corpus script imports load_folds() so that ALL
analyses use byte-for-byte identical splits. Returns a list of
(train_ids, test_ids) with integer patient_ids, fold 0..4.
"""
import json
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent / "cv_folds.json"

def load_folds(path=None):
    p = Path(path) if path is not None else _DEFAULT
    d = json.load(open(p))["folds"]
    n = len(d)
    return [([int(x) for x in d[f"fold_{k}"]["train_ids"]],
             [int(x) for x in d[f"fold_{k}"]["test_ids"]]) for k in range(n)]
