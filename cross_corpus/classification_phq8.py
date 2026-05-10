"""
Cross-Corpus Depression Classification (Train set → Test set) — v2, PHQ-8 Label Variant
========================================================================================
Identical to cross_corpus_classification_v2_llm.py except ProposedDataset binary labels
are derived from PHQ-8 scores (threshold ≥ 10) instead of SCID or PHQ-9.

E-DAIC labels remain unchanged (official 'depressed' column, PHQ-8 based).

Label setting evaluated:
  phq8_proposed_dataset : ProposedDataset side uses PHQ-8 ≥ 10 (from proposed_dataset_phq8.csv)
                 E-DAIC  side uses its standard 'depressed' column

PHQ-8 label file: proposed_dataset_phq8.csv  (semicolon-separated, columns: ID;PHQ-8)
  - IDs present in the file but not in the split → silently ignored
  - IDs present in the split but missing PHQ-8 values → WARNING printed, excluded

Output: cross_corpus_{direction}_v2_phq8_{llm}.csv
  (does NOT overwrite previous scid/phq result files)

Split files are identical (same train/test IDs as all other cross-corpus runs).

Usage:
  python cross_corpus_classification_v2_llm_phq8.py --direction proposed_dataset_to_edaic --llm llama_3.1_8B
  python cross_corpus_classification_v2_llm_phq8.py --direction edaic_to_proposed_dataset --llm llama_3.1_70B
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

PROPOSED_SPLIT = FEATURES_BASE / "splits" / "proposed_dataset_fold1_split.json"
EDAIC_SPLIT   = FEATURES_BASE / "splits" / "edaic_official_split.json"
PROPOSED_PHQ8  = Path("CONFIGURE_ME/labels/proposed_dataset_phq8.csv")
EDAIC_LABELS  = Path("CONFIGURE_ME/labels/edaic_labels.csv")

PHQ8_THRESHOLD = 10  # ≥ 10 → depressed (standard PHQ-8 clinical cutoff)

PROPOSED_AUDIO = {
    "egemaps_diarised":         FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    "egemaps_non_diarised":     FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    "wav2vec_diarised":         FEATURES_BASE / "proposed_dataset_german_xlsr_diarised/proposed_dataset_german_xlsr_diarised.csv",
    "wav2vec_non_diarised":     FEATURES_BASE / "proposed_dataset_german_xlsr_non_diarised/proposed_dataset_german_xlsr_non_diarised.csv",
    # NEW — facebook/wav2vec2-large-xlsr-53 multilingual (6144-dim)
    "xlsr53ml_diarised":        FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_diarised/proposed_dataset_xlsr53_multilingual_diarised.csv",
    "xlsr53ml_non_diarised":    FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_non_diarised/proposed_dataset_xlsr53_multilingual_non_diarised.csv",
}
EDAIC_AUDIO = {
    "egemaps_partitioned":      FEATURES_BASE / "edaic_egemapsv02/edaic_egemapsv02.csv",
    "xlsr_english_partitioned": FEATURES_BASE / "edaic_xlsr_english_partitioned/edaic_xlsr_english_partitioned.csv",
    # NEW — facebook/wav2vec2-large-xlsr-53 multilingual (6144-dim)
    "xlsr53ml_partitioned":     FEATURES_BASE / "edaic_xlsr53_multilingual/edaic_xlsr53_multilingual.csv",
}

OUTPUT_DIR = FEATURES_BASE / "results" / "cross_corpus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORR_DIR = OUTPUT_DIR / "feature_correlations"
CORR_DIR.mkdir(parents=True, exist_ok=True)

DEPROBERTA_MAX_LENGTH = 512

# ============================================================================
# LLM-DEPENDENT PATH BUILDER  (identical to v2_llm.py)
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
# FEATURE SELECTION PARAMETERS
# ============================================================================
EGEMAPS_K      = 30
W2V_PRESEL_K   = 200
W2V_MM_PCA_N   = 50
W2V_MM_POSTK   = 15
REDUND_THRESH  = 0.70
SPEARMAN_SIG   = 0.10
PCA_COMPONENTS = [10, 20, 50, 100]

# ============================================================================
# CLASSIFIER GRIDS
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
# LABEL LOADING  (PHQ-8 based for ProposedDataset, standard depressed for E-DAIC)
# ============================================================================
def load_proposed_dataset_phq8_labels():
    """
    Load proposed_dataset_phq8.csv and return a dict {patient_id -> binary label (1=depressed)}.
    Threshold: PHQ-8 >= 10.
    Rows with missing PHQ-8 values are silently skipped (already notified below).
    """
    df = pd.read_csv(PROPOSED_PHQ8, sep=";")
    df.columns = ["patient_id", "PHQ8"]
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
    df = df.dropna(subset=["patient_id"])
    df["patient_id"] = df["patient_id"].astype(int)
    df["PHQ8"] = pd.to_numeric(df["PHQ8"], errors="coerce")
    # Strip any annotation text from values like "548;4 (5 Missings)"
    valid = df.dropna(subset=["PHQ8"])
    labels = {int(row["patient_id"]): int(row["PHQ8"] >= PHQ8_THRESHOLD)
              for _, row in valid.iterrows()}
    return labels


def get_phq8_labels(pids, dataset, proposed_dataset_phq8_map, edaic_labels_df):
    """
    Return binary labels for pids:
      - ProposedDataset: from PHQ-8 >= 10 map; WARNING for any pid with data but no label
      - E-DAIC:  from official 'depressed' column (unchanged)
    """
    if dataset == "proposed_dataset":
        result = []
        for pid in pids:
            if pid in proposed_dataset_phq8_map:
                result.append(float(proposed_dataset_phq8_map[pid]))
            else:
                logging.warning(f"  [PHQ-8 MISSING] ProposedDataset patient_id={pid} has data "
                                f"but NO PHQ-8 label — excluded from analysis")
                result.append(np.nan)
        return np.array(result)
    else:
        edaic_idx = edaic_labels_df.set_index("participant_id")["depressed"]
        result = []
        for pid in pids:
            val = edaic_idx.get(pid, np.nan)
            result.append(float(val))
        return np.array(result)


# ============================================================================
# HELPERS: Spearman, feature selection, redundancy pruning
# ============================================================================
def compute_spearman_correlations(X_tr, y_tr, feat_names, save_path=None):
    rows = []
    for i, fn in enumerate(feat_names):
        rho, pval = stats.spearmanr(X_tr[:, i], y_tr)
        rows.append({
            "feature": fn, "spearman_r": float(rho), "abs_r": float(abs(rho)),
            "p_value": float(pval), "significant": bool(abs(rho) >= SPEARMAN_SIG),
        })
    df = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    if save_path is not None:
        df.to_csv(save_path, index=False)
        logging.info(f"  Saved correlation analysis → {save_path.name}")
    return df


def select_audio_features(X_tr, y_tr, X_te, k, feature_names=None):
    k_eff = min(k, X_tr.shape[1])
    sel = SelectKBest(f_classif, k=k_eff)
    X_tr_sel = sel.fit_transform(X_tr, y_tr)
    X_te_sel = sel.transform(X_te)
    sel_names = ([feature_names[i] for i in sel.get_support(indices=True)]
                 if feature_names else [f"f{i}" for i in sel.get_support(indices=True)])
    return X_tr_sel, X_te_sel, sel_names


def prune_audio_vs_text(X_audio_tr, audio_names, X_text_tr, threshold=REDUND_THRESH):
    keep = np.ones(X_audio_tr.shape[1], dtype=bool)
    for ai in range(X_audio_tr.shape[1]):
        for ti in range(X_text_tr.shape[1]):
            r, _ = stats.pearsonr(X_audio_tr[:, ai], X_text_tr[:, ti])
            if abs(r) > threshold:
                keep[ai] = False; break
    logging.info(f"  Audio-text redundancy pruning: {int((~keep).sum())}/{X_audio_tr.shape[1]} "
                 f"audio features removed (|r|>{threshold})")
    return keep


# ============================================================================
# FEATURE LOADING  (identical to v2_llm.py)
# ============================================================================
def load_proposed_dataset_summaries(pids, llm_paths):
    out = {}
    for pid in pids:
        for suf in ["_transcription.json", "_whisper.json"]:
            p = llm_paths["proposed_dataset_summaries"] / f"{pid:03d}{suf}"
            if p.exists():
                try:
                    with open(p) as f: s = json.load(f).get("summary", "").strip()
                    if s: out[pid] = s; break
                except Exception: pass
    return out


def load_edaic_summaries(pids, llm_paths):
    out = {}
    for pid in pids:
        p = llm_paths["edaic_summaries"] / f"{pid}.json"
        if p.exists():
            try:
                with open(p) as f: s = json.load(f).get("summary", "").strip()
                if s: out[pid] = s
            except Exception: pass
    return out


def extract_deproberta_probs(pids, model_dir, summaries, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    rows = []
    for pid in pids:
        text = summaries.get(pid)
        if text is None:
            rows.append([pid, 0.0, 0.0, 0.0]); continue
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=DEPROBERTA_MAX_LENGTH).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
        rows.append([pid, float(probs[0]), float(probs[1]), float(probs[2])])
    return pd.DataFrame(rows, columns=["pid", "prob_severe", "prob_moderate", "prob_not_depression"])


def build_text_features(pids, dataset, device, llm_paths):
    if dataset == "proposed_dataset":
        summaries = load_proposed_dataset_summaries(pids, llm_paths)
        model_dir = llm_paths["proposed_dataset_deproberta"]
        q_df = pd.read_csv(llm_paths["proposed_dataset_q_feats"]).rename(columns={"patient_id": "pid"})
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


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_modality(X_train, y_train, X_test, y_test, modality_name, direction,
                      feature_names=None):
    results = []
    best_f1, best_model_obj, best_model_name = -1, None, ""
    for name, grid in GRIDS.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_classifier(name))])
            gs = GridSearchCV(pipe, grid,
                              cv=StratifiedKFold(3, shuffle=True, random_state=43),
                              scoring="f1", n_jobs=1, refit=True)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            yp = best.predict(X_test)
            ys = get_scores(best, X_test)
            row = {
                "direction":   direction,
                "label_setting": "phq8_proposed_dataset",
                "modality":    modality_name,
                "model":       name,
                "best_params": str(gs.best_params_),
                "ap":          average_precision_score(y_test, ys),
                "auc":         roc_auc_score(y_test, ys),
                "bal_acc":     balanced_accuracy_score(y_test, yp),
                "f1":          f1_score(y_test, yp),
                "precision":   precision_score(y_test, yp),
                "recall":      recall_score(y_test, yp),
            }
            results.append(row)
            logging.info(f"    {name}: F1={row['f1']:.3f} AUC={row['auc']:.3f}")
            if row["f1"] > best_f1:
                best_f1, best_model_obj, best_model_name = row["f1"], best, name
        except Exception as e:
            logging.error(f"    {name} failed: {e}")
    if SHAP_AVAILABLE and best_model_obj and feature_names and "multimodal" in modality_name:
        _generate_shap(best_model_obj, X_test, feature_names, modality_name,
                       direction, best_model_name)
    return results


def _generate_shap(model, X_test, feature_names, modality, direction, model_name):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap_dir = OUTPUT_DIR / "shap_plots"; shap_dir.mkdir(exist_ok=True)
        clf = model.named_steps["clf"]
        X_scaled = model.named_steps["scaler"].transform(X_test)
        explainer = shap.Explainer(clf, X_scaled, feature_names=feature_names)
        shap_values = explainer(X_scaled)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title(f"SHAP — {modality} | {direction} | phq8_proposed_dataset | {model_name}")
        plt.tight_layout()
        fname = f"shap_v2_phq8_{direction}_{modality}_{model_name}.pdf"
        plt.savefig(shap_dir / fname, dpi=150, bbox_inches="tight"); plt.close()
        logging.info(f"  SHAP saved: {shap_dir / fname}")
    except Exception as e:
        logging.warning(f"  SHAP failed: {e}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", required=True,
                        choices=["proposed_dataset_to_edaic", "edaic_to_proposed_dataset"])
    parser.add_argument("--llm", required=True,
                        choices=list(LLM_CODES.keys()),
                        help="LLM used for text summaries and DepRoBERTa fine-tuning")
    args = parser.parse_args()

    logging.info(f"LLM: {args.llm}  (code: {LLM_CODES[args.llm]})")
    llm_paths = build_llm_paths(args.llm)
    for key, path in llm_paths.items():
        status = "OK" if Path(path).exists() else "MISSING"
        logging.info(f"  [{status}] {key}: {path}")

    random.seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load fixed data
    with open(PROPOSED_SPLIT) as f: emp_split = json.load(f)
    with open(EDAIC_SPLIT)   as f: edaic_split = json.load(f)
    emp_train_ids   = emp_split["train_ids"]
    emp_test_ids    = emp_split["test_ids"]
    edaic_train_ids = edaic_split["train_ids"]
    edaic_test_ids  = edaic_split["test_ids"]

    # Load PHQ-8 label map for ProposedDataset (once, not per-participant)
    proposed_dataset_phq8_map = load_proposed_dataset_phq8_labels()
    logging.info(f"\nProposedDataset PHQ-8 label map: {len(proposed_dataset_phq8_map)} participants "
                 f"({sum(proposed_dataset_phq8_map.values())} depressed at threshold≥{PHQ8_THRESHOLD})")

    edaic_labels_df = pd.read_csv(EDAIC_LABELS)

    # Verify coverage: warn for any split ID with data but no PHQ-8 label
    all_emp_ids = set(emp_train_ids + emp_test_ids)
    missing_phq8 = sorted(all_emp_ids - set(proposed_dataset_phq8_map.keys()))
    if missing_phq8:
        logging.warning(f"\n[PHQ-8 COVERAGE WARNING] {len(missing_phq8)} ProposedDataset split IDs "
                        f"have NO PHQ-8 label and will be excluded:")
        for pid in missing_phq8:
            logging.warning(f"  patient_id={pid}")
    else:
        logging.info("\n[PHQ-8 COVERAGE] All ProposedDataset split IDs have PHQ-8 labels. ✅")

    if args.direction == "proposed_dataset_to_edaic":
        train_pids, test_pids = emp_train_ids, edaic_test_ids
        train_ds,   test_ds   = "proposed_dataset", "edaic"
    else:
        train_pids, test_pids = edaic_train_ids, emp_test_ids
        train_ds,   test_ds   = "edaic", "proposed_dataset"

    y_train_all = get_phq8_labels(train_pids, train_ds, proposed_dataset_phq8_map, edaic_labels_df)
    y_test_all  = get_phq8_labels(test_pids,  test_ds,  proposed_dataset_phq8_map, edaic_labels_df)

    valid_tr = ~np.isnan(y_train_all)
    valid_te = ~np.isnan(y_test_all)
    train_pids_v = [p for p, v in zip(train_pids, valid_tr) if v]
    test_pids_v  = [p for p, v in zip(test_pids,  valid_te) if v]
    y_train = y_train_all[valid_tr].astype(int)
    y_test  = y_test_all[valid_te].astype(int)

    logging.info(f"\nDirection: {args.direction} | Label: phq8_proposed_dataset | LLM: {args.llm}")
    logging.info(f"Train ({train_ds}): n={len(train_pids_v)}, depressed={sum(y_train)}")
    logging.info(f"Test  ({test_ds}):  n={len(test_pids_v)},  depressed={sum(y_test)}")

    all_results = []

    # ---------------------------------------------------------------------- #
    # TEXT                                                                     #
    # ---------------------------------------------------------------------- #
    logging.info("\n--- TEXT ---")
    train_text = test_text = None
    text_cols = []
    try:
        train_text, text_cols = build_text_features(train_pids_v, train_ds, device, llm_paths)
        test_text,  _         = build_text_features(test_pids_v,  test_ds,  device, llm_paths)
        tr_pids_t = [p for p in train_pids_v if p in set(train_text["pid"])]
        te_pids_t = [p for p in test_pids_v  if p in set(test_text["pid"])]
        X_tr = train_text.set_index("pid").loc[tr_pids_t, text_cols].values
        y_tr = get_phq8_labels(tr_pids_t, train_ds, proposed_dataset_phq8_map, edaic_labels_df)
        mask_tr = ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr].astype(int)
        X_te = test_text.set_index("pid").loc[te_pids_t, text_cols].values
        y_te = get_phq8_labels(te_pids_t, test_ds, proposed_dataset_phq8_map, edaic_labels_df)
        mask_te = ~np.isnan(y_te)
        X_te, y_te = X_te[mask_te], y_te[mask_te].astype(int)
        res = evaluate_modality(X_tr, y_tr, X_te, y_te, "text", args.direction, text_cols)
        all_results.extend(res)
    except Exception as e:
        logging.error(f"Text failed: {e}")

    train_audio_paths = PROPOSED_AUDIO if train_ds == "proposed_dataset" else EDAIC_AUDIO
    test_audio_paths  = EDAIC_AUDIO   if test_ds  == "edaic"   else PROPOSED_AUDIO

    # ---------------------------------------------------------------------- #
    # AUDIO (eGeMAPS)                                                          #
    # ---------------------------------------------------------------------- #
    logging.info("\n--- AUDIO (eGeMAPS) ---")
    egemap_tr_keys = [k for k in train_audio_paths if "egemaps" in k]
    egemap_te_keys = [k for k in test_audio_paths  if "egemaps" in k]

    for tk in egemap_tr_keys:
        for tek in egemap_te_keys:
            base_name = f"egemaps_{tk}_to_{tek}"
            try:
                tr_df, tr_cols = load_audio_df(train_audio_paths[tk])
                te_df, te_cols = load_audio_df(test_audio_paths[tek])
                common_cols = sorted(set(tr_cols) & set(te_cols))
                if not common_cols: continue

                tr_ids = [p for p in train_pids_v if p in set(tr_df["pid"])]
                te_ids = [p for p in test_pids_v  if p in set(te_df["pid"])]
                X_tr_raw = np.nan_to_num(tr_df.set_index("pid").loc[tr_ids, common_cols].values.astype(float))
                y_tr_eg  = get_phq8_labels(tr_ids, train_ds, proposed_dataset_phq8_map, edaic_labels_df)
                X_te_raw = np.nan_to_num(te_df.set_index("pid").loc[te_ids, common_cols].values.astype(float))
                y_te_eg  = get_phq8_labels(te_ids, test_ds, proposed_dataset_phq8_map, edaic_labels_df)

                # Filter NaN labels
                mtr = ~np.isnan(y_tr_eg); mte = ~np.isnan(y_te_eg)
                X_tr_raw, y_tr_eg = X_tr_raw[mtr], y_tr_eg[mtr].astype(int)
                X_te_raw, y_te_eg = X_te_raw[mte], y_te_eg[mte].astype(int)

                corr_path = CORR_DIR / f"feature_corr_{args.direction}_phq8_{base_name}_{args.llm}.csv"
                compute_spearman_correlations(X_tr_raw, y_tr_eg, common_cols, save_path=corr_path)

                mod_no_fs = f"{base_name}_no_fs"
                logging.info(f"  {mod_no_fs}: train={X_tr_raw.shape}, test={X_te_raw.shape}")
                all_results.extend(evaluate_modality(X_tr_raw, y_tr_eg, X_te_raw, y_te_eg,
                                                     mod_no_fs, args.direction, common_cols))

                mod_sk = f"{base_name}_selectk{EGEMAPS_K}"
                X_tr_sk, X_te_sk, sk_names = select_audio_features(
                    X_tr_raw, y_tr_eg, X_te_raw, k=EGEMAPS_K, feature_names=common_cols)
                logging.info(f"  {mod_sk}: train={X_tr_sk.shape}, test={X_te_sk.shape}")
                all_results.extend(evaluate_modality(X_tr_sk, y_tr_eg, X_te_sk, y_te_eg,
                                                     mod_sk, args.direction, sk_names))
            except Exception as e:
                logging.error(f"  {base_name} failed: {e}")

    # ---------------------------------------------------------------------- #
    # AUDIO (wav2vec / XLSR)                                                   #
    # ---------------------------------------------------------------------- #
    logging.info("\n--- AUDIO (wav2vec / XLSR) ---")
    w2v_tr_keys = [k for k in train_audio_paths if "wav2vec" in k or "xlsr" in k]
    w2v_te_keys = [k for k in test_audio_paths  if "wav2vec" in k or "xlsr" in k]

    for tk in w2v_tr_keys:
        for tek in w2v_te_keys:
            base_name = f"w2v_{tk}_to_{tek}"
            try:
                tr_df, tr_cols = load_audio_df(train_audio_paths[tk])
                te_df, te_cols = load_audio_df(test_audio_paths[tek])
                common_cols = sorted(set(tr_cols) & set(te_cols))
                if not common_cols: continue

                tr_ids = [p for p in train_pids_v if p in set(tr_df["pid"])]
                te_ids = [p for p in test_pids_v  if p in set(te_df["pid"])]
                X_tr_raw = np.nan_to_num(tr_df.set_index("pid").loc[tr_ids, common_cols].values.astype(float))
                y_tr_w2v = get_phq8_labels(tr_ids, train_ds, proposed_dataset_phq8_map, edaic_labels_df)
                X_te_raw = np.nan_to_num(te_df.set_index("pid").loc[te_ids, common_cols].values.astype(float))
                y_te_w2v = get_phq8_labels(te_ids, test_ds, proposed_dataset_phq8_map, edaic_labels_df)

                mtr = ~np.isnan(y_tr_w2v); mte = ~np.isnan(y_te_w2v)
                X_tr_raw, y_tr_w2v = X_tr_raw[mtr], y_tr_w2v[mtr].astype(int)
                X_te_raw, y_te_w2v = X_te_raw[mte], y_te_w2v[mte].astype(int)

                scaler_raw = StandardScaler()
                X_tr_sc = scaler_raw.fit_transform(X_tr_raw)
                X_te_sc = scaler_raw.transform(X_te_raw)

                for n_pca in PCA_COMPONENTS:
                    eff = min(n_pca, X_tr_sc.shape[0], X_tr_sc.shape[1])
                    pca = PCA(n_components=eff, random_state=42)
                    X_tr_pca = pca.fit_transform(X_tr_sc)
                    X_te_pca = pca.transform(X_te_sc)
                    pca_names = [f"PC{i}" for i in range(eff)]
                    pca_mod = f"{base_name}_pca{n_pca}_no_fs"
                    logging.info(f"  {pca_mod}: train={X_tr_pca.shape}, test={X_te_pca.shape}")
                    if n_pca == PCA_COMPONENTS[0]:
                        corr_path = CORR_DIR / f"feature_corr_{args.direction}_phq8_{base_name}_pca{n_pca}_{args.llm}.csv"
                        compute_spearman_correlations(X_tr_pca, y_tr_w2v, pca_names, save_path=corr_path)
                    all_results.extend(evaluate_modality(X_tr_pca, y_tr_w2v, X_te_pca, y_te_w2v,
                                                         pca_mod, args.direction, pca_names))

                k_raw = min(W2V_PRESEL_K, X_tr_sc.shape[1])
                X_tr_presel, X_te_presel, presel_names = select_audio_features(
                    X_tr_sc, y_tr_w2v, X_te_sc, k=k_raw, feature_names=common_cols)
                logging.info(f"  wav2vec selectk{k_raw}: {X_tr_sc.shape[1]} → {X_tr_presel.shape[1]} features")
                for n_pca in PCA_COMPONENTS:
                    eff = min(n_pca, X_tr_presel.shape[0], X_tr_presel.shape[1])
                    pca = PCA(n_components=eff, random_state=42)
                    X_tr_pca = pca.fit_transform(X_tr_presel)
                    X_te_pca = pca.transform(X_te_presel)
                    pca_names = [f"PC{i}" for i in range(eff)]
                    sk_mod = f"{base_name}_selectk{k_raw}_pca{n_pca}"
                    logging.info(f"  {sk_mod}: train={X_tr_pca.shape}, test={X_te_pca.shape}")
                    all_results.extend(evaluate_modality(X_tr_pca, y_tr_w2v, X_te_pca, y_te_w2v,
                                                         sk_mod, args.direction, pca_names))
            except Exception as e:
                logging.error(f"  {base_name} failed: {e}")

    # ---------------------------------------------------------------------- #
    # MULTIMODAL                                                               #
    # ---------------------------------------------------------------------- #
    logging.info("\n--- MULTIMODAL ---")

    if train_text is None:
        logging.warning("  Skipping multimodal — text failed earlier")
    else:
        # a & b: text + eGeMAPS
        for tk in egemap_tr_keys:
            for tek in egemap_te_keys:
                try:
                    tr_audio_df, tr_acols = load_audio_df(train_audio_paths[tk])
                    te_audio_df, te_acols = load_audio_df(test_audio_paths[tek])
                    common_audio = sorted(set(tr_acols) & set(te_acols))
                    if not common_audio: continue

                    tr_merged = train_text.merge(tr_audio_df, on="pid")
                    te_merged = test_text.merge(te_audio_df, on="pid")
                    tr_ids = [p for p in train_pids_v if p in set(tr_merged["pid"])]
                    te_ids = [p for p in test_pids_v  if p in set(te_merged["pid"])]

                    X_tr_text  = np.nan_to_num(tr_merged.set_index("pid").loc[tr_ids, text_cols].values.astype(float))
                    X_te_text  = np.nan_to_num(te_merged.set_index("pid").loc[te_ids, text_cols].values.astype(float))
                    X_tr_audio = np.nan_to_num(tr_merged.set_index("pid").loc[tr_ids, common_audio].values.astype(float))
                    X_te_audio = np.nan_to_num(te_merged.set_index("pid").loc[te_ids, common_audio].values.astype(float))
                    y_tr_mm = get_phq8_labels(tr_ids, train_ds, proposed_dataset_phq8_map, edaic_labels_df)
                    y_te_mm = get_phq8_labels(te_ids, test_ds, proposed_dataset_phq8_map, edaic_labels_df)
                    mtr = ~np.isnan(y_tr_mm); mte = ~np.isnan(y_te_mm)
                    X_tr_text, X_tr_audio = X_tr_text[mtr], X_tr_audio[mtr]
                    X_te_text, X_te_audio = X_te_text[mte], X_te_audio[mte]
                    y_tr_mm, y_te_mm = y_tr_mm[mtr].astype(int), y_te_mm[mte].astype(int)

                    # a: no_fs
                    mm_no_fs = f"multimodal_text+egemaps_{tk}_to_{tek}_no_fs"
                    X_tr_mm  = np.hstack([X_tr_text, X_tr_audio])
                    X_te_mm  = np.hstack([X_te_text, X_te_audio])
                    logging.info(f"  {mm_no_fs}: train={X_tr_mm.shape}, test={X_te_mm.shape}")
                    all_results.extend(evaluate_modality(X_tr_mm, y_tr_mm, X_te_mm, y_te_mm,
                                                         mm_no_fs, args.direction, text_cols + common_audio))

                    # b: selectk30 + redundancy pruning
                    X_tr_ask, X_te_ask, ask_names = select_audio_features(
                        X_tr_audio, y_tr_mm, X_te_audio, k=EGEMAPS_K, feature_names=common_audio)
                    keep = prune_audio_vs_text(X_tr_ask, ask_names, X_tr_text)
                    if keep.sum() == 0:
                        logging.warning("  All eGeMAPS features pruned — skipping selectk MM")
                    else:
                        X_tr_ask = X_tr_ask[:, keep]; X_te_ask = X_te_ask[:, keep]
                        ask_names = [n for n, k in zip(ask_names, keep) if k]
                        mm_sk = f"multimodal_text+egemaps_{tk}_to_{tek}_selectk{EGEMAPS_K}_pruned"
                        X_tr_mm2 = np.hstack([X_tr_text, X_tr_ask])
                        X_te_mm2 = np.hstack([X_te_text, X_te_ask])
                        logging.info(f"  {mm_sk}: train={X_tr_mm2.shape} ({keep.sum()} audio feats)")
                        all_results.extend(evaluate_modality(X_tr_mm2, y_tr_mm, X_te_mm2, y_te_mm,
                                                             mm_sk, args.direction, text_cols + ask_names))
                except Exception as e:
                    logging.error(f"  multimodal egemaps {tk}_to_{tek} failed: {e}")

        # c: text + wav2vec PCA→selectk15 + redundancy pruning
        for tk in w2v_tr_keys:
            for tek in w2v_te_keys:
                try:
                    tr_audio_df, tr_acols = load_audio_df(train_audio_paths[tk])
                    te_audio_df, te_acols = load_audio_df(test_audio_paths[tek])
                    common_audio = sorted(set(tr_acols) & set(te_acols))
                    if not common_audio: continue

                    tr_merged = train_text.merge(tr_audio_df, on="pid")
                    te_merged = test_text.merge(te_audio_df, on="pid")
                    tr_ids = [p for p in train_pids_v if p in set(tr_merged["pid"])]
                    te_ids = [p for p in test_pids_v  if p in set(te_merged["pid"])]

                    X_tr_text      = np.nan_to_num(tr_merged.set_index("pid").loc[tr_ids, text_cols].values.astype(float))
                    X_te_text      = np.nan_to_num(te_merged.set_index("pid").loc[te_ids, text_cols].values.astype(float))
                    X_tr_audio_raw = np.nan_to_num(tr_merged.set_index("pid").loc[tr_ids, common_audio].values.astype(float))
                    X_te_audio_raw = np.nan_to_num(te_merged.set_index("pid").loc[te_ids, common_audio].values.astype(float))
                    y_tr_mm = get_phq8_labels(tr_ids, train_ds, proposed_dataset_phq8_map, edaic_labels_df)
                    y_te_mm = get_phq8_labels(te_ids, test_ds, proposed_dataset_phq8_map, edaic_labels_df)
                    mtr = ~np.isnan(y_tr_mm); mte = ~np.isnan(y_te_mm)
                    X_tr_text,      X_tr_audio_raw = X_tr_text[mtr],      X_tr_audio_raw[mtr]
                    X_te_text,      X_te_audio_raw = X_te_text[mte],      X_te_audio_raw[mte]
                    y_tr_mm, y_te_mm = y_tr_mm[mtr].astype(int), y_te_mm[mte].astype(int)

                    scaler_mm = StandardScaler()
                    X_tr_asc = scaler_mm.fit_transform(X_tr_audio_raw)
                    X_te_asc = scaler_mm.transform(X_te_audio_raw)

                    eff_pca = min(W2V_MM_PCA_N, X_tr_asc.shape[0], X_tr_asc.shape[1])
                    pca_mm = PCA(n_components=eff_pca, random_state=42)
                    X_tr_apca = pca_mm.fit_transform(X_tr_asc)
                    X_te_apca = pca_mm.transform(X_te_asc)
                    pca_names = [f"PC{i}" for i in range(eff_pca)]

                    k_post = min(W2V_MM_POSTK, X_tr_apca.shape[1])
                    X_tr_ask, X_te_ask, ask_names = select_audio_features(
                        X_tr_apca, y_tr_mm, X_te_apca, k=k_post, feature_names=pca_names)

                    keep = prune_audio_vs_text(X_tr_ask, ask_names, X_tr_text)
                    if keep.sum() == 0:
                        logging.warning(f"  All wav2vec features pruned for {tk}_to_{tek} — skipping")
                        continue

                    X_tr_ask = X_tr_ask[:, keep]; X_te_ask = X_te_ask[:, keep]
                    ask_names = [n for n, k in zip(ask_names, keep) if k]
                    mm_w2v = (f"multimodal_text+w2v_{tk}_to_{tek}"
                              f"_pca{eff_pca}_selectk{k_post}_pruned")
                    X_tr_mm = np.hstack([X_tr_text, X_tr_ask])
                    X_te_mm = np.hstack([X_te_text, X_te_ask])
                    logging.info(f"  {mm_w2v}: train={X_tr_mm.shape} ({keep.sum()} wav2vec retained)")
                    all_results.extend(evaluate_modality(X_tr_mm, y_tr_mm, X_te_mm, y_te_mm,
                                                         mm_w2v, args.direction, text_cols + ask_names))
                except Exception as e:
                    logging.error(f"  multimodal wav2vec {tk}_to_{tek} failed: {e}")

    # ---------------------------------------------------------------------- #
    # SAVE RESULTS  (distinct filename — no overwrite)                         #
    # ---------------------------------------------------------------------- #
    if all_results:
        df = pd.DataFrame(all_results).sort_values("f1", ascending=False)
        out = OUTPUT_DIR / f"cross_corpus_{args.direction}_v2_phq8_xlsr53ml_{args.llm}.csv"
        df.to_csv(out, index=False)
        logging.info(f"\nResults → {out}")
        logging.info(f"Best: {df.iloc[0]['modality']} / {df.iloc[0]['model']} "
                     f"→ F1={df.iloc[0]['f1']:.3f}")


if __name__ == "__main__":
    main()
