"""
Cross-Corpus Depression Classification — v2 XLSR-53 Multilingual
=================================================================
Variant of cross_corpus_classification_v2_llm.py that adds the
facebook/wav2vec2-large-xlsr-53 (language-agnostic multilingual)
audio features on top of the existing audio sources:

  ProposedDataset:
    xlsr53_ml_diarised     → proposed_dataset_xlsr53_multilingual_diarised.csv     (6144-dim)
    xlsr53_ml_non_diarised → proposed_dataset_xlsr53_multilingual_non_diarised.csv (6144-dim)

  E-DAIC:
    xlsr53_ml_partitioned  → edaic_xlsr53_multilingual.csv                 (6144-dim)

All other existing audio, text and multimodal modalities are preserved.
Feature selection, classifiers, output format are identical to v2.

Usage:
  python cross_corpus_classification_v2_llm_xlsr53ml.py \\
      --direction proposed_dataset_to_edaic --llm llama_3.1_8B
  python cross_corpus_classification_v2_llm_xlsr53ml.py \\
      --direction edaic_to_proposed_dataset --llm llama_3.1_70B

Output:
  cross_corpus_{direction}_v2_xlsr53ml_{llm}.csv
"""

import argparse
import json
import logging
import random
import warnings
from pathlib import Path
from scipy import stats

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from lightgbm import LGBMClassifier
    LGBM = True
except ImportError:
    LGBM = False
try:
    from catboost import CatBoostClassifier
    CB = True
except ImportError:
    CB = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================================
# FIXED PATHS
# ============================================================================
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")
FEATURES_BASE   = DATASET_ROOT / "features_output"
WOODY_BASE = Path("CONFIGURE_ME/storage")
CROSS_DIR  = Path("CONFIGURE_ME/cross_corpus_data")

PROPOSED_SPLIT   = FEATURES_BASE / "splits" / "proposed_dataset_fold1_split.json"
EDAIC_SPLIT     = FEATURES_BASE / "splits" / "edaic_official_split.json"
PROPOSED_CLASSI  = Path("CONFIGURE_ME/labels/classi_labels.csv")
PROPOSED_REGRESS = Path("CONFIGURE_ME/labels/regress_labels.csv")
EDAIC_LABELS    = Path("CONFIGURE_ME/labels/edaic_labels.csv")

# ── Audio paths ────────────────────────────────────────────────────────────────
# ProposedDataset: existing eGeMAPS + wav2vec sources  +  NEW XLSR-53 multilingual
PROPOSED_AUDIO = {
    # eGeMAPS LLD (200-dim)
    "egemaps_diarised":         FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    "egemaps_non_diarised":     FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    # language-specific XLSR fine-tuned (audeering backbone, 6144-dim)
    "wav2vec_diarised":         FEATURES_BASE / "proposed_dataset_german_xlsr_diarised/proposed_dataset_german_xlsr_diarised.csv",
    "wav2vec_non_diarised":     FEATURES_BASE / "proposed_dataset_german_xlsr_non_diarised/proposed_dataset_german_xlsr_non_diarised.csv",
    # NEW — facebook/wav2vec2-large-xlsr-53 multilingual (6144-dim)
    "xlsr53_ml_diarised":       FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_diarised/proposed_dataset_xlsr53_multilingual_diarised.csv",
    "xlsr53_ml_non_diarised":   FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_non_diarised/proposed_dataset_xlsr53_multilingual_non_diarised.csv",
}

# E-DAIC: existing eGeMAPS + XLSR-English  +  NEW XLSR-53 multilingual
EDAIC_AUDIO = {
    # eGeMAPS LLD (200-dim)
    "egemaps_partitioned":      FEATURES_BASE / "edaic_egemapsv02/edaic_egemapsv02.csv",
    # language-specific XLSR-53-English (6144-dim)
    "xlsr_english_partitioned": FEATURES_BASE / "edaic_xlsr_english_partitioned/edaic_xlsr_english_partitioned.csv",
    # NEW — facebook/wav2vec2-large-xlsr-53 multilingual (6144-dim)
    "xlsr53_ml_partitioned":    FEATURES_BASE / "edaic_xlsr53_multilingual/edaic_xlsr53_multilingual.csv",
}

OUTPUT_DIR = FEATURES_BASE / "results" / "cross_corpus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORR_DIR = OUTPUT_DIR / "feature_correlations"
CORR_DIR.mkdir(parents=True, exist_ok=True)

DEPROBERTA_MAX_LENGTH = 512

# ============================================================================
# LLM-DEPENDENT PATH BUILDER  (unchanged from v2)
# ============================================================================
LLM_CODES = {
    "llama_3.1_8B":  "3108",
    "llama_3.1_70B": "3170",
    "llama_3.3_70B": "3370",
}


def build_llm_paths(llm: str) -> dict:
    code = LLM_CODES[llm]
    return {
        "proposed_dataset_deproberta": WOODY_BASE / "deproberta_cv_models" / llm / "fold_1",
        "edaic_deproberta":   WOODY_BASE / "deproberta_edaic_models" / llm / "train_split",
        "proposed_dataset_summaries":  (Path(f"CONFIGURE_ME/LLM_summaries/proposed_dataset/summaries_{code}")),
        "edaic_summaries":    Path(f"CONFIGURE_ME/LLM_summaries/edaic/summaries_{code}"),
        "proposed_dataset_q_feats":    Path(f"CONFIGURE_ME/features/text_features_llama_{code}.csv"),
        "edaic_q_feats":      CROSS_DIR / f"E-DAIC_data/Llama{code}_combined_text_r2_case1.csv",
    }


# ============================================================================
# FEATURE SELECTION PARAMETERS  (unchanged)
# ============================================================================
EGEMAPS_K      = 30
W2V_PRESEL_K   = 200
W2V_MM_PCA_N   = 50
W2V_MM_POSTK   = 15
REDUND_THRESH  = 0.70
SPEARMAN_SIG   = 0.10
PCA_COMPONENTS = [10, 20, 50, 100]

# ============================================================================
# CLASSIFIER GRIDS  (unchanged)
# ============================================================================
GRIDS = {
    "LogisticRegression": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__solver": ["liblinear", "saga"]},
    "SVC":                {"clf__C": [0.1, 1, 10, 100], "clf__gamma": [1, 0.1, 0.01, 0.001], "clf__kernel": ["rbf", "poly", "sigmoid"]},
    "RandomForest":       {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "XGBoost":            {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 0.2], "clf__max_depth": [3, 5, 7]},
    "AdaBoost":           {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 1.0]},
    "DecisionTree":       {"clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "KNN":                {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
    "ExtraTrees":         {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20]},
    "GradientBoosting":   {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7]},
    "MLP":                {"clf__hidden_layer_sizes": [(50,), (100,), (50, 50)], "clf__activation": ["relu", "tanh"], "clf__alpha": [0.0001, 0.001, 0.01]},
}
if LGBM: GRIDS["LightGBM"] = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1]}
if CB:   GRIDS["CatBoost"] = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}


def make_classifier(name):
    m = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=10000, random_state=42),
        "SVC":                lambda: SVC(probability=True, random_state=42),
        "RandomForest":       lambda: RandomForestClassifier(random_state=42),
        "XGBoost":            lambda: XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42),
        "AdaBoost":           lambda: AdaBoostClassifier(random_state=42),
        "DecisionTree":       lambda: DecisionTreeClassifier(random_state=42),
        "KNN":                lambda: KNeighborsClassifier(),
        "ExtraTrees":         lambda: ExtraTreesClassifier(random_state=42),
        "GradientBoosting":   lambda: GradientBoostingClassifier(random_state=42),
        "MLP":                lambda: MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
    }
    if LGBM: m["LightGBM"] = lambda: LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
    if CB:   m["CatBoost"] = lambda: CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    return m[name]()


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return s if s.ndim == 1 else s[:, 1]
    return model.predict(X).astype(float)


# ============================================================================
# DEPROBERTA HELPERS  (unchanged from v2)
# ============================================================================
def extract_deproberta_probs(pids, model_dir, summaries, device):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        logging.warning(f"  DepRoBERTa model not found: {model_dir}")
        return pd.DataFrame({"pid": pids,
                             "prob_severe": np.nan,
                             "prob_moderate": np.nan,
                             "prob_not_depression": np.nan})

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    rows = []
    for pid in pids:
        text = summaries.get(pid, "")
        if not text:
            rows.append({"pid": pid, "prob_severe": np.nan,
                         "prob_moderate": np.nan, "prob_not_depression": np.nan})
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=DEPROBERTA_MAX_LENGTH, padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        rows.append({"pid": pid,
                     "prob_severe": float(probs[0]) if len(probs) > 0 else np.nan,
                     "prob_moderate": float(probs[1]) if len(probs) > 1 else np.nan,
                     "prob_not_depression": float(probs[2]) if len(probs) > 2 else np.nan})
    return pd.DataFrame(rows)


def load_proposed_dataset_summaries(pids, llm_paths):
    summaries = {}
    summ_dir = Path(llm_paths["proposed_dataset_summaries"])
    for pid in pids:
        for fname in [f"{pid}.txt", f"VP{pid:03d}.txt", f"{pid:03d}.txt"]:
            p = summ_dir / fname
            if p.exists():
                summaries[pid] = p.read_text(encoding="utf-8").strip()
                break
    return summaries


def load_edaic_summaries(pids, llm_paths):
    summaries = {}
    summ_dir = Path(llm_paths["edaic_summaries"])
    for pid in pids:
        for fname in [f"{pid}.txt", f"{pid}_P.txt"]:
            p = summ_dir / fname
            if p.exists():
                summaries[pid] = p.read_text(encoding="utf-8").strip()
                break
    return summaries


def build_text_features(pids, dataset, device, llm_paths):
    if dataset == "proposed_dataset":
        summaries = load_proposed_dataset_summaries(pids, llm_paths)
        model_dir = llm_paths["proposed_dataset_deproberta"]
        q_df = pd.read_csv(llm_paths["proposed_dataset_q_feats"])
        q_df = q_df.rename(columns={"patient_id": "pid"})
    else:
        summaries = load_edaic_summaries(pids, llm_paths)
        model_dir = llm_paths["edaic_deproberta"]
        q_df = pd.read_csv(llm_paths["edaic_q_feats"])
        # Normalise ID column to 'pid' regardless of original column name
        for id_col in ["participant_id", "patient_id"]:
            if id_col in q_df.columns:
                q_df = q_df.rename(columns={id_col: "pid"})
                break
    prob_df = extract_deproberta_probs(pids, model_dir, summaries, device)
    q_cols = [f"q{i}" for i in range(1, 12)]
    q_df = q_df[["pid"] + q_cols]
    merged = prob_df.merge(q_df, on="pid", how="inner")
    feat_cols = ["prob_severe", "prob_moderate", "prob_not_depression"] + q_cols
    return merged, feat_cols


def load_audio_df(path):
    df = pd.read_csv(path)
    if "patient_id" in df.columns:
        df = df.rename(columns={"patient_id": "pid"})
    elif "participant_id" in df.columns:
        df = df.rename(columns={"participant_id": "pid"})
    elif "file_id" in df.columns:
        df["pid"] = df["file_id"].astype(str).str.extract(r"(\d+)").astype(int)
        df = df.drop(columns=["file_id"])
    meta = ["pid", "n_segments"]
    feat_cols = [c for c in df.columns if c not in meta]
    return df[["pid"] + feat_cols], feat_cols


def get_labels(pids, dataset, label_setting):
    if dataset == "proposed_dataset":
        cls = pd.read_csv(PROPOSED_CLASSI).rename(columns={"patient_id": "pid"})
        if label_setting == "scid":
            labels = cls.set_index("pid")["depressed"]
        else:
            reg = pd.read_csv(PROPOSED_REGRESS).rename(columns={"patient_id": "pid"})
            phq = reg.set_index("pid")["PHQ9-Score"]
            labels = (phq >= 10).astype(int)
    else:
        edaic = pd.read_csv(EDAIC_LABELS).rename(columns={"participant_id": "pid"})
        labels = edaic.set_index("pid")["depressed"]
    return np.array([labels.get(pid, np.nan) for pid in pids])


# ============================================================================
# SPEARMAN CORRELATION & FEATURE SELECTION HELPERS  (unchanged from v2)
# ============================================================================
def compute_spearman_correlations(X, y, feature_names, save_path=None):
    rows = []
    for i, fname in enumerate(feature_names):
        try:
            r, p = stats.spearmanr(X[:, i], y)
        except Exception:
            r, p = 0.0, 1.0
        rows.append({"feature": fname, "spearman_r": r, "p_value": p})
    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def select_audio_features(X_tr, y_tr, X_te, k=30, feature_names=None):
    k = min(k, X_tr.shape[1])
    sel = SelectKBest(f_classif, k=k)
    X_tr_new = sel.fit_transform(X_tr, y_tr)
    X_te_new = sel.transform(X_te)
    if feature_names:
        sel_names = [feature_names[i] for i in sel.get_support(indices=True)]
    else:
        sel_names = [f"feat_{i}" for i in sel.get_support(indices=True)]
    return X_tr_new, X_te_new, sel_names


def remove_redundant_features(X_tr, X_te, feature_names, threshold=0.70):
    if X_tr.shape[1] <= 1:
        return X_tr, X_te, feature_names
    corr_matrix = np.corrcoef(X_tr.T)
    np.fill_diagonal(corr_matrix, 0)
    keep = list(range(X_tr.shape[1]))
    drop = set()
    for i in range(len(keep)):
        if i in drop:
            continue
        for j in range(i + 1, len(keep)):
            if j in drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                drop.add(j)
    keep_idx = [i for i in range(X_tr.shape[1]) if i not in drop]
    kept_names = [feature_names[i] for i in keep_idx]
    return X_tr[:, keep_idx], X_te[:, keep_idx], kept_names


# ============================================================================
# EVALUATION  (unchanged from v2)
# ============================================================================
def evaluate_modality(X_train, y_train, X_test, y_test, modality_name, direction,
                      label_setting, feature_names=None):
    results = []
    best_f1, best_model_obj, best_model_name = -1, None, ""
    for name, grid in GRIDS.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_classifier(name))])
            gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=43),
                              scoring="f1", n_jobs=1, refit=True)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            yp = best.predict(X_test)
            ys = get_scores(best, X_test)
            row = {
                "direction": direction, "label_setting": label_setting,
                "modality": modality_name, "model": name,
                "best_params": str(gs.best_params_),
                "ap":        average_precision_score(y_test, ys),
                "auc":       roc_auc_score(y_test, ys),
                "bal_acc":   balanced_accuracy_score(y_test, yp),
                "f1":        f1_score(y_test, yp, zero_division=0),
                "precision": precision_score(y_test, yp, zero_division=0),
                "recall":    recall_score(y_test, yp, zero_division=0),
            }
            results.append(row)
            logging.info(f"    {name:<20} F1={row['f1']:.3f}  AUC={row['auc']:.3f}")
            if row["f1"] > best_f1:
                best_f1, best_model_obj, best_model_name = row["f1"], best, name
        except Exception as e:
            logging.warning(f"    {name} FAILED: {e}")
    return results


# ============================================================================
# MULTIMODAL FUSION  (unchanged from v2)
# ============================================================================
def fuse_multimodal(X_tr_audio, X_te_audio,
                    X_tr_text, X_te_text,
                    y_tr, y_te,
                    modality_name, direction, label_setting):
    # PCA on audio → select top features, then concat with text
    n_pca = min(W2V_MM_PCA_N, X_tr_audio.shape[0] - 1, X_tr_audio.shape[1])
    sc_a  = StandardScaler()
    X_tr_a = sc_a.fit_transform(X_tr_audio)
    X_te_a = sc_a.transform(X_te_audio)
    pca = PCA(n_components=n_pca, random_state=42)
    X_tr_ap = pca.fit_transform(X_tr_a)
    X_te_ap = pca.transform(X_te_a)
    k = min(W2V_MM_POSTK, X_tr_ap.shape[1])
    sel = SelectKBest(f_classif, k=k)
    X_tr_as = sel.fit_transform(X_tr_ap, y_tr)
    X_te_as = sel.transform(X_te_ap)
    sc_t = StandardScaler()
    X_tr_t = sc_t.fit_transform(X_tr_text)
    X_te_t = sc_t.transform(X_te_text)
    X_tr_mm = np.hstack([X_tr_as, X_tr_t])
    X_te_mm = np.hstack([X_te_as, X_te_t])
    return evaluate_modality(X_tr_mm, y_tr, X_te_mm, y_te,
                             modality_name, direction, label_setting)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", required=True,
                        choices=["proposed_dataset_to_edaic", "edaic_to_proposed_dataset"])
    parser.add_argument("--llm", required=True,
                        choices=["llama_3.1_8B", "llama_3.1_70B", "llama_3.3_70B"])
    args = parser.parse_args()

    out_file = OUTPUT_DIR / f"cross_corpus_{args.direction}_v2_xlsr53ml_{args.llm}.csv"
    logging.info(f"Output: {out_file}")

    llm_paths = build_llm_paths(args.llm)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    with open(PROPOSED_SPLIT) as f:
        proposed_dataset_sp = json.load(f)
    with open(EDAIC_SPLIT) as f:
        edaic_sp = json.load(f)

    # Split files use 'train_ids' / 'test_ids' as keys (not 'train'/'test')
    proposed_dataset_train = [int(x) for x in proposed_dataset_sp.get("train_ids", proposed_dataset_sp.get("train", proposed_dataset_sp.get("Train", [])))]
    proposed_dataset_test  = [int(x) for x in proposed_dataset_sp.get("test_ids",  proposed_dataset_sp.get("test",  proposed_dataset_sp.get("Test",  [])))]
    edaic_train   = sorted([int(x) for x in edaic_sp.get("train_ids", edaic_sp.get("train", edaic_sp.get("Train", [])))])
    edaic_test    = sorted([int(x) for x in edaic_sp.get("test_ids",  edaic_sp.get("test",  edaic_sp.get("Test",  [])))])

    all_results = []

    for label_setting in ["scid", "phq9_10"]:
        logging.info(f"\n{'='*70}")
        logging.info(f"Direction: {args.direction}  |  Label: {label_setting}")
        logging.info(f"{'='*70}")

        if args.direction == "proposed_dataset_to_edaic":
            train_pids, test_pids = proposed_dataset_train, edaic_test
            train_ds,   test_ds   = "proposed_dataset", "edaic"
        else:
            train_pids, test_pids = edaic_train, proposed_dataset_test
            train_ds,   test_ds   = "edaic", "proposed_dataset"

        y_train_all = get_labels(train_pids, train_ds, label_setting)
        y_test_all  = get_labels(test_pids,  test_ds,  label_setting)
        valid_tr = ~np.isnan(y_train_all); valid_te = ~np.isnan(y_test_all)
        train_pids_v = [p for p, v in zip(train_pids, valid_tr) if v]
        test_pids_v  = [p for p, v in zip(test_pids,  valid_te) if v]
        y_train = y_train_all[valid_tr].astype(int)
        y_test  = y_test_all[valid_te].astype(int)
        logging.info(f"Train: {len(train_pids_v)} ({sum(y_train)} dep), "
                     f"Test: {len(test_pids_v)} ({sum(y_test)} dep)")

        # ------------------------------------------------------------------ #
        # TEXT                                                                 #
        # ------------------------------------------------------------------ #
        logging.info("\n--- TEXT ---")
        train_text = test_text = None
        text_cols = []
        try:
            train_text, text_cols = build_text_features(train_pids_v, train_ds, device, llm_paths)
            test_text,  _         = build_text_features(test_pids_v,  test_ds,  device, llm_paths)
            tr_tid = set(train_text["pid"].tolist())
            te_tid = set(test_text["pid"].tolist())
            X_tr = train_text.set_index("pid").loc[
                [p for p in train_pids_v if p in tr_tid], text_cols].values
            y_tr = get_labels([p for p in train_pids_v if p in tr_tid],
                               train_ds, label_setting).astype(int)
            X_te = test_text.set_index("pid").loc[
                [p for p in test_pids_v if p in te_tid], text_cols].values
            y_te = get_labels([p for p in test_pids_v if p in te_tid],
                               test_ds, label_setting).astype(int)
            res = evaluate_modality(X_tr, y_tr, X_te, y_te, "text",
                                    args.direction, label_setting, text_cols)
            all_results.extend(res)
        except Exception as e:
            logging.error(f"Text failed: {e}")

        # ------------------------------------------------------------------ #
        # AUDIO (eGeMAPS) — no_fs and selectk30                               #
        # ------------------------------------------------------------------ #
        logging.info("\n--- AUDIO (eGeMAPS) ---")
        train_audio_paths = PROPOSED_AUDIO if train_ds == "proposed_dataset" else EDAIC_AUDIO
        test_audio_paths  = EDAIC_AUDIO   if test_ds  == "edaic"   else PROPOSED_AUDIO

        egemap_tr_keys = [k for k in train_audio_paths if "egemaps" in k]
        egemap_te_keys = [k for k in test_audio_paths  if "egemaps" in k]

        for tk in egemap_tr_keys:
            for tek in egemap_te_keys:
                base_name = f"egemaps_{tk}_to_{tek}"
                try:
                    tr_df, tr_cols = load_audio_df(train_audio_paths[tk])
                    te_df, te_cols = load_audio_df(test_audio_paths[tek])
                    common_cols = sorted(set(tr_cols) & set(te_cols))
                    if not common_cols:
                        continue
                    tr_ids = [p for p in train_pids_v if p in set(tr_df["pid"])]
                    te_ids = [p for p in test_pids_v  if p in set(te_df["pid"])]
                    X_tr_raw = np.nan_to_num(tr_df.set_index("pid").loc[tr_ids, common_cols].values.astype(float))
                    y_tr_eg  = get_labels(tr_ids, train_ds, label_setting).astype(int)
                    X_te_raw = np.nan_to_num(te_df.set_index("pid").loc[te_ids, common_cols].values.astype(float))
                    y_te_eg  = get_labels(te_ids, test_ds,  label_setting).astype(int)

                    corr_path = CORR_DIR / f"feature_corr_{args.direction}_{label_setting}_{base_name}_{args.llm}.csv"
                    compute_spearman_correlations(X_tr_raw, y_tr_eg, common_cols, save_path=corr_path)

                    # no feature selection
                    mod_no_fs = f"{base_name}_no_fs"
                    logging.info(f"  {mod_no_fs}: train={X_tr_raw.shape}, test={X_te_raw.shape}")
                    res = evaluate_modality(X_tr_raw, y_tr_eg, X_te_raw, y_te_eg,
                                            mod_no_fs, args.direction, label_setting, common_cols)
                    all_results.extend(res)

                    # selectk30
                    mod_sk = f"{base_name}_selectk{EGEMAPS_K}"
                    X_tr_sk, X_te_sk, sk_names = select_audio_features(
                        X_tr_raw, y_tr_eg, X_te_raw, k=EGEMAPS_K, feature_names=common_cols)
                    logging.info(f"  {mod_sk}: train={X_tr_sk.shape}, test={X_te_sk.shape}")
                    res = evaluate_modality(X_tr_sk, y_tr_eg, X_te_sk, y_te_eg,
                                            mod_sk, args.direction, label_setting, sk_names)
                    all_results.extend(res)

                except Exception as e:
                    logging.error(f"  {base_name} FAILED: {e}")

        # ------------------------------------------------------------------ #
        # AUDIO (wav2vec / XLSR — all variants incl. new XLSR-53 multilingual)#
        # ------------------------------------------------------------------ #
        logging.info("\n--- AUDIO (wav2vec / XLSR — all variants) ---")
        # The existing logic matches train keys containing "wav2vec" or "xlsr"
        # against test keys containing "wav2vec" or "xlsr".
        # New keys "xlsr53_ml_*" contain "xlsr" so they are automatically included.
        w2v_tr_keys = [k for k in train_audio_paths if "wav2vec" in k or "xlsr" in k]
        w2v_te_keys = [k for k in test_audio_paths  if "wav2vec" in k or "xlsr" in k]

        for tk in w2v_tr_keys:
            for tek in w2v_te_keys:
                base_name = f"w2v_{tk}_to_{tek}"
                try:
                    tr_path = train_audio_paths[tk]
                    te_path = test_audio_paths[tek]
                    if not tr_path.exists():
                        logging.warning(f"  SKIP {base_name}: train file missing: {tr_path}")
                        continue
                    if not te_path.exists():
                        logging.warning(f"  SKIP {base_name}: test file missing: {te_path}")
                        continue

                    tr_df, tr_cols = load_audio_df(tr_path)
                    te_df, te_cols = load_audio_df(te_path)
                    common_cols = sorted(set(tr_cols) & set(te_cols))
                    if not common_cols:
                        logging.warning(f"  SKIP {base_name}: no common feature columns")
                        continue

                    tr_ids = [p for p in train_pids_v if p in set(tr_df["pid"])]
                    te_ids = [p for p in test_pids_v  if p in set(te_df["pid"])]
                    X_tr_raw = np.nan_to_num(tr_df.set_index("pid").loc[tr_ids, common_cols].values.astype(float))
                    y_tr_w2v = get_labels(tr_ids, train_ds, label_setting).astype(int)
                    X_te_raw = np.nan_to_num(te_df.set_index("pid").loc[te_ids, common_cols].values.astype(float))
                    y_te_w2v = get_labels(te_ids, test_ds,  label_setting).astype(int)

                    scaler_raw = StandardScaler()
                    X_tr_sc = scaler_raw.fit_transform(X_tr_raw)
                    X_te_sc = scaler_raw.transform(X_te_raw)

                    # PCA-only conditions
                    for n_pca in PCA_COMPONENTS:
                        eff = min(n_pca, X_tr_sc.shape[0] - 1, X_tr_sc.shape[1])
                        pca = PCA(n_components=eff, random_state=42)
                        X_tr_pca = pca.fit_transform(X_tr_sc)
                        X_te_pca = pca.transform(X_te_sc)
                        pca_names = [f"PC{i}" for i in range(eff)]

                        pca_mod = f"{base_name}_pca{n_pca}_no_fs"
                        logging.info(f"  {pca_mod}: train={X_tr_pca.shape}, test={X_te_pca.shape}")

                        if n_pca == PCA_COMPONENTS[0]:
                            corr_path = CORR_DIR / f"feature_corr_{args.direction}_{label_setting}_{base_name}_pca{n_pca}_{args.llm}.csv"
                            compute_spearman_correlations(X_tr_pca, y_tr_w2v, pca_names, save_path=corr_path)

                        res = evaluate_modality(X_tr_pca, y_tr_w2v, X_te_pca, y_te_w2v,
                                                pca_mod, args.direction, label_setting, pca_names)
                        all_results.extend(res)

                    # selectk200 + redundancy pruning + PCA conditions
                    k_pre = min(W2V_PRESEL_K, X_tr_raw.shape[1])
                    X_tr_sk, X_te_sk, sk_names = select_audio_features(
                        X_tr_raw, y_tr_w2v, X_te_raw, k=k_pre, feature_names=common_cols)
                    X_tr_rd, X_te_rd, rd_names = remove_redundant_features(
                        X_tr_sk, X_te_sk, sk_names, threshold=REDUND_THRESH)

                    for n_pca in PCA_COMPONENTS:
                        eff = min(n_pca, X_tr_rd.shape[0] - 1, X_tr_rd.shape[1])
                        if eff < 1:
                            continue
                        sc2 = StandardScaler()
                        X_tr_rd_sc = sc2.fit_transform(X_tr_rd)
                        X_te_rd_sc = sc2.transform(X_te_rd)
                        pca2 = PCA(n_components=eff, random_state=42)
                        X_tr_p2 = pca2.fit_transform(X_tr_rd_sc)
                        X_te_p2 = pca2.transform(X_te_rd_sc)
                        pca_names2 = [f"PC{i}" for i in range(eff)]
                        mod_p2 = f"{base_name}_selectk{k_pre}_dedup_pca{n_pca}"
                        logging.info(f"  {mod_p2}: train={X_tr_p2.shape}, test={X_te_p2.shape}")
                        res = evaluate_modality(X_tr_p2, y_tr_w2v, X_te_p2, y_te_w2v,
                                                mod_p2, args.direction, label_setting, pca_names2)
                        all_results.extend(res)

                    # ----------------------------------------------------------------
                    # MULTIMODAL: audio + text  (only when text data is available)
                    # ----------------------------------------------------------------
                    if train_text is not None and test_text is not None:
                        try:
                            tr_both = [p for p in tr_ids if p in set(train_text["pid"])]
                            te_both = [p for p in te_ids if p in set(test_text["pid"])]
                            if tr_both and te_both:
                                X_tr_mm_audio = np.nan_to_num(
                                    tr_df.set_index("pid").loc[tr_both, common_cols].values.astype(float))
                                X_te_mm_audio = np.nan_to_num(
                                    te_df.set_index("pid").loc[te_both, common_cols].values.astype(float))
                                X_tr_mm_text = train_text.set_index("pid").loc[tr_both, text_cols].values
                                X_te_mm_text = test_text.set_index("pid").loc[te_both, text_cols].values
                                y_tr_mm = get_labels(tr_both, train_ds, label_setting).astype(int)
                                y_te_mm = get_labels(te_both, test_ds,  label_setting).astype(int)
                                mm_name = f"multimodal_{base_name}"
                                logging.info(f"\n--- MULTIMODAL: {mm_name} ---")
                                res = fuse_multimodal(X_tr_mm_audio, X_te_mm_audio,
                                                      X_tr_mm_text,  X_te_mm_text,
                                                      y_tr_mm, y_te_mm,
                                                      mm_name, args.direction, label_setting)
                                all_results.extend(res)
                        except Exception as e:
                            logging.error(f"  Multimodal {base_name} FAILED: {e}")

                except Exception as e:
                    logging.error(f"  {base_name} FAILED: {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(out_file, index=False)
        logging.info(f"\nSaved {len(df_out)} rows → {out_file}")

        logging.info("\n=== TOP 15 BY F1 ===")
        top = df_out.sort_values("f1", ascending=False).head(15)
        logging.info(top[["direction", "label_setting", "modality", "model", "f1", "auc"]].to_string(index=False))
    else:
        logging.warning("No results to save.")


if __name__ == "__main__":
    main()
