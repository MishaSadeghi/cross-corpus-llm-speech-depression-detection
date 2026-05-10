"""
E-DAIC Multimodal Depression Classification + Regression (Within-Dataset)
Multi-LLM version: accepts --llm argument to select the LLM model.

Same logic as edaic_multimodal_within.py but with dynamic LLM-dependent paths.
Output files are suffixed with the llm name:
  classification_results_{llm}.csv / regression_results_{llm}.csv

LLM model codes:
  llama_3.1_8B  → 3108
  llama_3.1_70B → 3170
  llama_3.3_70B → 3370

Usage:
  python edaic_multimodal_within_llm.py --llm llama_3.1_8B
  python edaic_multimodal_within_llm.py --llm llama_3.1_70B
  python edaic_multimodal_within_llm.py --llm llama_3.3_70B
"""

import argparse
import json
import logging
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM = True
except ImportError:
    LGBM = False
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CB = True
except ImportError:
    CB = False

# ============================================================================
# FIXED PATHS
# ============================================================================
BASE_DIR   = Path("CONFIGURE_ME/repo")
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")
FEATURES_BASE   = DATASET_ROOT / "features_output"
WOODY_BASE = Path("CONFIGURE_ME/storage")
CROSS_DIR  = Path("CONFIGURE_ME/cross_corpus_data")

EDAIC_LABELS = Path("CONFIGURE_ME/labels/edaic_labels.csv")
DEPROBERTA_MAX_LENGTH = 512

AUDIO_SETS = {
    # ---- eGeMAPS (commented out — results already exist) ----
    # "egemaps_partitioned": {
    #     "path": FEATURES_BASE / "edaic_egemapsv02/edaic_egemapsv02.csv",
    #     "id_col": "participant_id", "high_dim": False,
    # },
    "xlsr_english_partitioned": {
        "path": FEATURES_BASE / "edaic_xlsr_english_partitioned/edaic_xlsr_english_partitioned.csv",
        "id_col": "participant_id", "high_dim": True,
    },
    # ---- NEW: XLSR-53 Multilingual ----
    "xlsr53ml_partitioned": {
        "path": FEATURES_BASE / "edaic_xlsr53_multilingual/edaic_xlsr53_multilingual.csv",
        "id_col": "participant_id", "high_dim": True,
    },
}

OUTPUT_DIR = FEATURES_BASE / "results" / "edaic_multimodal_within"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORR_DIR = OUTPUT_DIR / "feature_correlations"
CORR_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LLM-DEPENDENT PATH BUILDER
# ============================================================================
LLM_CODES = {
    "llama_3.1_8B":  "3108",
    "llama_3.1_70B": "3170",
    "llama_3.3_70B": "3370",
}


def build_llm_paths(llm: str) -> dict:
    code = LLM_CODES[llm]
    return {
        "deproberta_model_dir": WOODY_BASE / "deproberta_edaic_models" / llm / "train_split",
        "llm_summary_dir":      Path(f"CONFIGURE_ME/LLM_summaries/edaic/summaries_{code}"),
        "question_feats":       CROSS_DIR / f"E-DAIC_data/Llama{code}_combined_text_r2_case1.csv",
    }


# ============================================================================
# FEATURE SELECTION PARAMETERS  (identical to original)
# ============================================================================
EGEMAPS_K     = 30
W2V_PRESEL_K  = 200
W2V_MM_PCA_N  = 50
W2V_MM_POSTK  = 15
REDUND_THRESH = 0.70
SPEARMAN_SIG  = 0.10
PCA_FOR_AUDIO = [20, 50, 100]

# ============================================================================
# GRIDS  (identical to original)
# ============================================================================
CLASSI_GRIDS = {
    "LogisticRegression": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__solver": ["liblinear", "saga"]},
    "SVC": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": [1, 0.1, 0.01, 0.001], "clf__kernel": ["rbf", "poly", "sigmoid"]},
    "RandomForest": {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "XGBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 0.2], "clf__max_depth": [3, 5, 7]},
    "AdaBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 1.0]},
    "DecisionTree": {"clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
    "ExtraTrees": {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20]},
    "GradientBoosting": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7]},
    "MLP": {"clf__hidden_layer_sizes": [(50,), (100,), (50, 50)], "clf__activation": ["relu", "tanh"], "clf__alpha": [0.0001, 0.001, 0.01]},
}
REGRESS_GRIDS = {
    "Ridge": {"clf__alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"clf__alpha": [0.01, 0.1, 1, 10]},
    "ElasticNet": {"clf__alpha": [0.01, 0.1, 1, 10], "clf__l1_ratio": [0.2, 0.5, 0.8]},
    "SVR": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.1, 0.01], "clf__kernel": ["rbf", "poly"]},
    "RandomForest": {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20]},
    "AdaBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 1.0]},
    "DecisionTree": {"clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
    "ExtraTrees": {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20]},
    "GradientBoosting": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7]},
    "MLP": {"clf__hidden_layer_sizes": [(50,), (100,), (50, 50)], "clf__activation": ["relu", "tanh"], "clf__alpha": [0.0001, 0.001, 0.01]},
    "XGBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 0.2], "clf__max_depth": [3, 5, 7]},
}
if LGBM:
    CLASSI_GRIDS["LightGBM"] = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1]}
    REGRESS_GRIDS["LightGBM"] = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1]}
if CB:
    CLASSI_GRIDS["CatBoost"] = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}
    REGRESS_GRIDS["CatBoost"] = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


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


def make_regressor(name):
    m = {
        "Ridge":            lambda: Ridge(max_iter=10000),
        "Lasso":            lambda: Lasso(max_iter=10000),
        "ElasticNet":       lambda: ElasticNet(max_iter=10000),
        "SVR":              lambda: SVR(),
        "RandomForest":     lambda: RandomForestRegressor(random_state=42),
        "AdaBoost":         lambda: AdaBoostRegressor(random_state=42),
        "DecisionTree":     lambda: DecisionTreeRegressor(random_state=42),
        "KNN":              lambda: KNeighborsRegressor(),
        "ExtraTrees":       lambda: ExtraTreesRegressor(random_state=42),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
        "MLP":              lambda: MLPRegressor(random_state=42, max_iter=1000, early_stopping=True),
        "XGBoost":          lambda: XGBRegressor(objective="reg:squarederror", random_state=42),
    }
    if LGBM: m["LightGBM"] = lambda: LGBMRegressor(random_state=42, verbose=-1, force_col_wise=True)
    if CB:   m["CatBoost"] = lambda: CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
    return m[name]()


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return s if s.ndim == 1 else s[:, 1]
    return model.predict(X).astype(float)


def compute_spearman_correlations(X_train, y_train, feat_names, save_path=None):
    rows = []
    for i, fn in enumerate(feat_names):
        rho, pval = stats.spearmanr(X_train[:, i], y_train)
        rows.append({"feature": fn, "spearman_r": float(rho), "abs_r": float(abs(rho)),
                     "p_value": float(pval), "significant": bool(abs(rho) >= SPEARMAN_SIG)})
    df = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    if save_path is not None:
        df.to_csv(save_path, index=False); logging.info(f"  Correlation CSV → {save_path.name}")
    return df


def prune_audio_vs_text(X_audio_train, X_text_train, threshold=REDUND_THRESH):
    keep = np.ones(X_audio_train.shape[1], dtype=bool)
    for ai in range(X_audio_train.shape[1]):
        for ti in range(X_text_train.shape[1]):
            r, _ = stats.pearsonr(X_audio_train[:, ai], X_text_train[:, ti])
            if abs(r) > threshold:
                keep[ai] = False; break
    logging.info(f"  Redundancy pruning: {int((~keep).sum())}/{X_audio_train.shape[1]} audio dims removed (|r|>{threshold})")
    return keep


# ============================================================================
# TEXT FEATURES (LLM-path-aware)
# ============================================================================
def load_edaic_summaries(pids, llm_paths):
    out = {}
    for pid in pids:
        for fname in [f"{pid:03d}_whisper.json", f"{pid}_whisper.json", f"{pid:03d}_transcription.json"]:
            p = llm_paths["llm_summary_dir"] / fname
            if p.exists():
                try:
                    with open(p) as f:
                        s = json.load(f).get("summary", "").strip()
                    if s: out[pid] = s; break
                except Exception: pass
    return out


def extract_text_features(all_ids, llm_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = llm_paths["deproberta_model_dir"]
    logging.info(f"Extracting DepRoBERTa text features from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    summaries = load_edaic_summaries(all_ids, llm_paths)
    rows = []
    for pid in all_ids:
        text = summaries.get(pid)
        if text is None:
            rows.append({"participant_id": pid, "prob_severe": 0.0,
                         "prob_moderate": 0.0, "prob_not_depression": 0.0})
            continue
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=DEPROBERTA_MAX_LENGTH).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
        rows.append({"participant_id": pid, "prob_severe": float(probs[0]),
                     "prob_moderate": float(probs[1]), "prob_not_depression": float(probs[2])})
    prob_df = pd.DataFrame(rows)
    q_df = pd.read_csv(llm_paths["question_feats"]).rename(columns={"patient_id": "participant_id"})
    q_cols = [f"q{i}" for i in range(1, 12)]
    q_df = q_df[["participant_id"] + q_cols]
    merged = prob_df.merge(q_df, on="participant_id", how="inner")
    text_feat_cols = ["prob_severe", "prob_moderate", "prob_not_depression"] + q_cols
    logging.info(f"Text features: {len(text_feat_cols)} cols, {len(merged)} patients")
    return merged, text_feat_cols


def load_audio_features(aset_name):
    cfg = AUDIO_SETS[aset_name]
    df = pd.read_csv(cfg["path"])
    if cfg["id_col"] != "participant_id":
        df = df.rename(columns={cfg["id_col"]: "participant_id"})
    meta = ["participant_id", "n_segments"]
    feat_cols = [c for c in df.columns if c not in meta]
    return df[["participant_id"] + feat_cols], feat_cols, cfg["high_dim"]


# ============================================================================
# RUN MODELS  (identical to original)
# ============================================================================
def run_mm_models(X_td, y_td, X_test, y_test, X_train_only, y_train_only,
                  ps, combo_label, grids, make_fn, score_fn, task):
    results = []
    for name, grid in grids.items():
        try:
            pipe = Pipeline([("clf", make_fn(name))])
            gs = GridSearchCV(pipe, grid, cv=ps, scoring=score_fn, n_jobs=1, refit=True)
            gs.fit(X_td, y_td)
            gs.best_estimator_.fit(X_train_only, y_train_only)
            yp = gs.best_estimator_.predict(X_test)
            row = {"combo": combo_label, "model": name, "best_params": str(gs.best_params_)}
            if task == "cls":
                ys = get_scores(gs.best_estimator_, X_test)
                row.update({"ap": average_precision_score(y_test, ys),
                             "auc": roc_auc_score(y_test, ys),
                             "bal_acc": balanced_accuracy_score(y_test, yp),
                             "f1": f1_score(y_test, yp),
                             "precision": precision_score(y_test, yp),
                             "recall": recall_score(y_test, yp)})
                logging.info(f"    CLS {name}: F1={row['f1']:.3f}")
            else:
                row.update({"r2": r2_score(y_test, yp),
                             "mae": mean_absolute_error(y_test, yp),
                             "rmse": float(np.sqrt(mean_squared_error(y_test, yp)))})
                logging.info(f"    REG {name}: MAE={row['mae']:.2f}")
            results.append(row)
        except Exception as e:
            logging.error(f"    {task.upper()} {name} [{combo_label}] failed: {e}")
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True,
                        choices=list(LLM_CODES.keys()),
                        help="LLM used for text summaries and DepRoBERTa fine-tuning")
    args = parser.parse_args()

    logging.info(f"LLM: {args.llm}  (code: {LLM_CODES[args.llm]})")
    llm_paths = build_llm_paths(args.llm)
    for key, path in llm_paths.items():
        status = "OK" if Path(path).exists() else "MISSING"
        logging.info(f"  [{status}] {key}: {path}")

    set_seed(42)

    labels = pd.read_csv(EDAIC_LABELS)
    train_df = labels[labels['split'] == 'train']
    dev_df   = labels[labels['split'] == 'dev']
    test_df  = labels[labels['split'] == 'test']
    logging.info(f"train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")

    all_ids = labels['participant_id'].tolist()
    text_df_all, text_cols = extract_text_features(all_ids, llm_paths)

    all_cls, all_reg = [], []

    for aset_name, cfg in AUDIO_SETS.items():
        if not cfg["path"].exists():
            logging.warning(f"Audio not found: {cfg['path']} — skipping {aset_name}"); continue

        audio_df, audio_cols, high_dim = load_audio_features(aset_name)
        logging.info(f"\n{'='*70}\nE-DAIC Multimodal [{args.llm}]: text + {aset_name} ({len(audio_cols)} audio feats)")

        merged = (labels[["participant_id", "split", "depressed", "PHQ_score"]]
                  .merge(text_df_all, on="participant_id", how="inner")
                  .merge(audio_df,    on="participant_id", how="inner"))

        td      = merged[merged['split'].isin(['train', 'dev'])]
        test_m  = merged[merged['split'] == 'test']
        train_m = merged[merged['split'] == 'train']

        fold_idx = np.array([-1 if s == 'train' else 0 for s in td['split']])
        ps_cls   = PredefinedSplit(fold_idx)
        ps_reg   = PredefinedSplit(fold_idx)

        text_td  = td[text_cols].values.astype(float)
        audio_td = np.nan_to_num(td[audio_cols].values.astype(float))
        text_te  = test_m[text_cols].values.astype(float)
        audio_te = np.nan_to_num(test_m[audio_cols].values.astype(float))
        text_tr  = train_m[text_cols].values.astype(float)
        audio_tr = np.nan_to_num(train_m[audio_cols].values.astype(float))

        train_mask_td = (td['split'] == 'train').values

        y_td_cls    = td['depressed'].values;     y_test_cls  = test_m['depressed'].values
        y_td_reg    = td['PHQ_score'].values;     y_test_reg  = test_m['PHQ_score'].values
        y_train_cls = train_m['depressed'].values; y_train_reg = train_m['PHQ_score'].values

        sc_t2 = StandardScaler().fit(text_td[train_mask_td])
        text_td_s = sc_t2.transform(text_td)
        text_te_s = sc_t2.transform(text_te)
        text_tr_s = sc_t2.transform(text_tr)

        sc_a2 = StandardScaler().fit(audio_td[train_mask_td])
        audio_td_s = sc_a2.transform(audio_td)
        audio_te_s = sc_a2.transform(audio_te)
        audio_tr_s = sc_a2.transform(audio_tr)

        corr_path = CORR_DIR / f"feature_corr_{aset_name}_{args.llm}.csv"
        compute_spearman_correlations(audio_td_s[train_mask_td], y_td_cls[train_mask_td],
                                      audio_cols, save_path=corr_path)

        if not high_dim:
            # eGeMAPS — Condition 1: no_fs
            combo_nofs = f"text+{aset_name}_no_fs"
            X_td_mm = np.hstack([text_td_s, audio_td_s])
            X_te_mm = np.hstack([text_te_s, audio_te_s])
            X_tr_mm = np.hstack([text_tr_s, audio_tr_s])
            logging.info(f"  [{combo_nofs}] td={X_td_mm.shape}, test={X_te_mm.shape}")
            all_cls += run_mm_models(X_td_mm, y_td_cls, X_te_mm, y_test_cls, X_tr_mm, y_train_cls,
                                     ps_cls, combo_nofs, CLASSI_GRIDS, make_classifier, "f1", "cls")
            all_reg += run_mm_models(X_td_mm, y_td_reg, X_te_mm, y_test_reg, X_tr_mm, y_train_reg,
                                     ps_reg, combo_nofs, REGRESS_GRIDS, make_regressor, "r2", "reg")

            # eGeMAPS — Condition 2: selectk30 + redundancy pruning
            k_eff = min(EGEMAPS_K, audio_td_s.shape[1])
            sel = SelectKBest(f_classif, k=k_eff).fit(audio_td_s[train_mask_td],
                                                       y_td_cls[train_mask_td])
            a_td_sk = sel.transform(audio_td_s)
            a_te_sk = sel.transform(audio_te_s)
            a_tr_sk = sel.transform(audio_tr_s)

            keep = prune_audio_vs_text(a_tr_sk, text_tr_s)
            if keep.sum() == 0:
                logging.warning("  All eGeMAPS features pruned — skipping selectk combo")
            else:
                combo_sk = f"text+{aset_name}_selectk{k_eff}_pruned"
                X_td_mm = np.hstack([text_td_s, a_td_sk[:, keep]])
                X_te_mm = np.hstack([text_te_s, a_te_sk[:, keep]])
                X_tr_mm = np.hstack([text_tr_s, a_tr_sk[:, keep]])
                logging.info(f"  [{combo_sk}] td={X_td_mm.shape} ({keep.sum()} audio feats retained)")
                all_cls += run_mm_models(X_td_mm, y_td_cls, X_te_mm, y_test_cls, X_tr_mm, y_train_cls,
                                         ps_cls, combo_sk, CLASSI_GRIDS, make_classifier, "f1", "cls")
                all_reg += run_mm_models(X_td_mm, y_td_reg, X_te_mm, y_test_reg, X_tr_mm, y_train_reg,
                                         ps_reg, combo_sk, REGRESS_GRIDS, make_regressor, "r2", "reg")

        else:
            # XLSR — Condition 1: pca_only
            for n_pca in PCA_FOR_AUDIO:
                eff = min(n_pca, audio_td_s[train_mask_td].shape[0], audio_td_s.shape[1])
                pca = PCA(n_components=eff, random_state=42).fit(audio_td_s[train_mask_td])
                a_td_pca = pca.transform(audio_td_s)
                a_te_pca = pca.transform(audio_te_s)
                a_tr_pca = pca.transform(audio_tr_s)
                combo_pca = f"text+{aset_name}_pca{n_pca}_no_fs"
                X_td_mm = np.hstack([text_td_s, a_td_pca])
                X_te_mm = np.hstack([text_te_s, a_te_pca])
                X_tr_mm = np.hstack([text_tr_s, a_tr_pca])
                logging.info(f"  [{combo_pca}] td={X_td_mm.shape}, test={X_te_mm.shape}")
                all_cls += run_mm_models(X_td_mm, y_td_cls, X_te_mm, y_test_cls, X_tr_mm, y_train_cls,
                                         ps_cls, combo_pca, CLASSI_GRIDS, make_classifier, "f1", "cls")
                all_reg += run_mm_models(X_td_mm, y_td_reg, X_te_mm, y_test_reg, X_tr_mm, y_train_reg,
                                         ps_reg, combo_pca, REGRESS_GRIDS, make_regressor, "r2", "reg")

            # XLSR — Condition 2: selectk_pca + redundancy pruning
            k_raw = min(W2V_PRESEL_K, audio_td_s.shape[1])
            sel_pre = SelectKBest(f_classif, k=k_raw).fit(audio_td_s[train_mask_td], y_td_cls[train_mask_td])
            a_td_pre = sel_pre.transform(audio_td_s)
            a_te_pre = sel_pre.transform(audio_te_s)
            a_tr_pre = sel_pre.transform(audio_tr_s)

            eff_pca = min(W2V_MM_PCA_N, a_tr_pre.shape[0], a_tr_pre.shape[1])
            pca_mm = PCA(n_components=eff_pca, random_state=42).fit(a_tr_pre)
            a_td_pca = pca_mm.transform(a_td_pre)
            a_te_pca = pca_mm.transform(a_te_pre)
            a_tr_pca = pca_mm.transform(a_tr_pre)

            k_post = min(W2V_MM_POSTK, a_td_pca.shape[1])
            sel_post = SelectKBest(f_classif, k=k_post).fit(a_tr_pca, y_train_cls)
            a_td_sk = sel_post.transform(a_td_pca)
            a_te_sk = sel_post.transform(a_te_pca)
            a_tr_sk = sel_post.transform(a_tr_pca)

            keep = prune_audio_vs_text(a_tr_sk, text_tr_s)
            if keep.sum() == 0:
                logging.warning("  All XLSR features pruned — skipping selectk_pca combo")
            else:
                combo_skpca = f"text+{aset_name}_selectk{k_raw}_pca{eff_pca}_k{k_post}_pruned"
                X_td_mm = np.hstack([text_td_s, a_td_sk[:, keep]])
                X_te_mm = np.hstack([text_te_s, a_te_sk[:, keep]])
                X_tr_mm = np.hstack([text_tr_s, a_tr_sk[:, keep]])
                logging.info(f"  [{combo_skpca}] td={X_td_mm.shape} ({keep.sum()} audio dims retained)")
                all_cls += run_mm_models(X_td_mm, y_td_cls, X_te_mm, y_test_cls, X_tr_mm, y_train_cls,
                                         ps_cls, combo_skpca, CLASSI_GRIDS, make_classifier, "f1", "cls")
                all_reg += run_mm_models(X_td_mm, y_td_reg, X_te_mm, y_test_reg, X_tr_mm, y_train_reg,
                                         ps_reg, combo_skpca, REGRESS_GRIDS, make_regressor, "r2", "reg")

    if all_cls:
        pd.DataFrame(all_cls).sort_values("f1", ascending=False).to_csv(
            OUTPUT_DIR / f"classification_results_{args.llm}.csv", index=False)
    if all_reg:
        pd.DataFrame(all_reg).sort_values("mae").to_csv(
            OUTPUT_DIR / f"regression_results_{args.llm}.csv", index=False)
    logging.info(f"\nResults → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
