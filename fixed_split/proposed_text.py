"""
ProposedDataset Text-Based Depression Classification + Regression — Fixed Split (Fold 1)
Multi-LLM version: accepts --llm argument to select the LLM model.

Uses the median fold (Fold 1) from the original 5-fold CV as a fixed train/test
split. Reuses the DepRoBERTa model already fine-tuned on the Fold 1 training
set (no data leakage).

Tasks:
  - Classification: binary depression (SCID-based) using 14 features
    (3 DepRoBERTa probs + 11 question scores)
  - Regression: PHQ-9 score prediction using the same 14 features

LLM model codes:
  llama_3.1_8B  → 3108
  llama_3.1_70B → 3170
  llama_3.3_70B → 3370

Output files are suffixed with the llm name to avoid overwriting:
  classification_results_{llm}.csv
  regression_results_{llm}.csv

Usage:
  python proposed_dataset_text_fixed_split_llm.py --llm llama_3.1_8B
  python proposed_dataset_text_fixed_split_llm.py --llm llama_3.1_70B
  python proposed_dataset_text_fixed_split_llm.py --llm llama_3.3_70B
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
from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
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
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset/features_output")
WOODY_BASE = Path("CONFIGURE_ME/storage")

SPLIT_FILE     = DATASET_ROOT / "splits" / "proposed_dataset_fold1_split.json"
CLASSI_LABELS  = Path("CONFIGURE_ME/labels/classi_labels.csv")
REGRESS_LABELS = Path("CONFIGURE_ME/labels/regress_labels.csv")

CACHE_DIR  = Path("CONFIGURE_ME/storage/models/hf_cache")
OUTPUT_DIR = DATASET_ROOT / "results" / "proposed_dataset_text_fixed_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEPROBERTA_MAX_LENGTH = 512

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
        "deproberta_model_dir": WOODY_BASE / "deproberta_cv_models" / llm / "fold_1",
        "llm_summary_dir":      (Path(f"CONFIGURE_ME/LLM_summaries/proposed_dataset/summaries_{code}")),
        "question_feats":       Path(f"CONFIGURE_ME/features/text_features_llama_{code}.csv"),
    }


# ============================================================================
# GRIDS  (identical to original)
# ============================================================================
CLASSI_GRIDS = {
    "LogisticRegression": {"C": [0.001, 0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "saga"]},
    "SVC": {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001], "kernel": ["rbf", "poly", "sigmoid"]},
    "RandomForest": {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]},
    "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
    "DecisionTree": {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "criterion": ["gini", "entropy"]},
    "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan", "minkowski"]},
    "ExtraTrees": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
    "GradientBoosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 5, 7], "subsample": [0.8, 1.0]},
    "MLP": {"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)], "activation": ["relu", "tanh"], "alpha": [0.0001, 0.001, 0.01], "learning_rate": ["constant", "adaptive"]},
}
REGRESS_GRIDS = {
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"alpha": [0.01, 0.1, 1, 10]},
    "ElasticNet": {"alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.2, 0.5, 0.8]},
    "SVR": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 0.01], "kernel": ["rbf", "poly"]},
    "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
    "DecisionTree": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
    "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    "ExtraTrees": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
    "GradientBoosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]},
    "MLP": {"hidden_layer_sizes": [(50,), (100,), (50, 50)], "activation": ["relu", "tanh"], "alpha": [0.0001, 0.001, 0.01]},
    "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
}

if LGBM:
    CLASSI_GRIDS["LightGBM"] = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7, -1], "num_leaves": [15, 31, 63], "min_child_samples": [10, 20, 30]}
    REGRESS_GRIDS["LightGBM"] = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7, -1], "num_leaves": [15, 31, 63]}
if CB:
    CLASSI_GRIDS["CatBoost"] = {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "depth": [3, 5, 7], "l2_leaf_reg": [1, 3, 5], "border_count": [32, 64, 128]}
    REGRESS_GRIDS["CatBoost"] = {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "depth": [3, 5, 7], "l2_leaf_reg": [1, 3, 5]}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


# ============================================================================
# FEATURE EXTRACTION  (LLM-path-aware)
# ============================================================================
def load_summaries(patient_ids, llm_paths):
    summaries = {}
    for pid in patient_ids:
        for suffix in ["_transcription.json", "_whisper.json"]:
            p = llm_paths["llm_summary_dir"] / f"{pid:03d}{suffix}"
            if p.exists():
                try:
                    with open(p) as f:
                        s = json.load(f).get("summary", "").strip()
                    if s:
                        summaries[pid] = s
                        break
                except Exception:
                    pass
    logging.info(f"  Loaded {len(summaries)}/{len(patient_ids)} summaries")
    return summaries


def extract_deproberta_probs(patient_ids, llm_paths):
    model_dir = llm_paths["deproberta_model_dir"]
    logging.info(f"Extracting DepRoBERTa probs from {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    summaries = load_summaries(patient_ids, llm_paths)
    rows = []
    for pid in patient_ids:
        text = summaries.get(pid)
        if text is None:
            rows.append({"patient_id": pid, "prob_severe": 0.0,
                         "prob_moderate": 0.0, "prob_not_depression": 0.0})
            continue
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=DEPROBERTA_MAX_LENGTH).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
        rows.append({"patient_id": pid, "prob_severe": float(probs[0]),
                     "prob_moderate": float(probs[1]), "prob_not_depression": float(probs[2])})
    return pd.DataFrame(rows)


def build_feature_matrix(patient_ids, llm_paths):
    prob_df = extract_deproberta_probs(patient_ids, llm_paths)
    q_df = pd.read_csv(llm_paths["question_feats"])
    q_cols = [f"q{i}" for i in range(1, 12)]
    q_df = q_df[["patient_id"] + q_cols]
    merged = prob_df.merge(q_df, on="patient_id", how="inner")
    feat_cols = ["prob_severe", "prob_moderate", "prob_not_depression"] + q_cols
    return merged, feat_cols


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

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_ids = split["train_ids"]
    test_ids  = split["test_ids"]
    all_ids   = train_ids + test_ids
    logging.info(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

    feat_df, feat_cols = build_feature_matrix(all_ids, llm_paths)
    classi_labels  = pd.read_csv(CLASSI_LABELS)
    regress_labels = pd.read_csv(REGRESS_LABELS)

    # ===================== CLASSIFICATION =====================
    logging.info("=" * 70)
    logging.info(f"CLASSIFICATION (binary depression, SCID-based) | {args.llm}")
    logging.info("=" * 70)

    df_cls = feat_df.merge(classi_labels[["patient_id", "depressed"]], on="patient_id")
    train_mask = df_cls["patient_id"].isin(train_ids)
    X_train_cls = df_cls.loc[train_mask,  feat_cols].values
    y_train_cls = df_cls.loc[train_mask,  "depressed"].values
    X_test_cls  = df_cls.loc[~train_mask, feat_cols].values
    y_test_cls  = df_cls.loc[~train_mask, "depressed"].values

    scaler_cls  = StandardScaler()
    X_train_cls = scaler_cls.fit_transform(X_train_cls)
    X_test_cls  = scaler_cls.transform(X_test_cls)

    cls_results = []
    for name, grid in CLASSI_GRIDS.items():
        try:
            gs = GridSearchCV(make_classifier(name), grid,
                              cv=StratifiedKFold(3, shuffle=True, random_state=43),
                              scoring="f1", n_jobs=1, refit=True)
            gs.fit(X_train_cls, y_train_cls)
            best   = gs.best_estimator_
            y_pred = best.predict(X_test_cls)
            y_score = get_scores(best, X_test_cls)
            row = {
                "llm": args.llm, "model": name, "best_params": str(gs.best_params_),
                "ap":       average_precision_score(y_test_cls, y_score),
                "auc":      roc_auc_score(y_test_cls, y_score),
                "bal_acc":  balanced_accuracy_score(y_test_cls, y_pred),
                "f1":       f1_score(y_test_cls, y_pred),
                "precision": precision_score(y_test_cls, y_pred),
                "recall":   recall_score(y_test_cls, y_pred),
            }
            cls_results.append(row)
            logging.info(f"  {name}: F1={row['f1']:.3f}  AUC={row['auc']:.3f}")
        except Exception as e:
            logging.error(f"  {name} failed: {e}")

    cls_df = pd.DataFrame(cls_results).sort_values("f1", ascending=False)
    cls_out = OUTPUT_DIR / f"classification_results_{args.llm}.csv"
    cls_df.to_csv(cls_out, index=False)
    logging.info(f"Classification results → {cls_out}")

    # ===================== REGRESSION =====================
    logging.info("=" * 70)
    logging.info(f"REGRESSION (PHQ-9 score prediction) | {args.llm}")
    logging.info("=" * 70)

    df_reg = feat_df.merge(regress_labels[["patient_id", "PHQ9-Score"]], on="patient_id")
    train_mask_r = df_reg["patient_id"].isin(train_ids)
    X_train_reg = df_reg.loc[train_mask_r,  feat_cols].values
    y_train_reg = df_reg.loc[train_mask_r,  "PHQ9-Score"].values
    X_test_reg  = df_reg.loc[~train_mask_r, feat_cols].values
    y_test_reg  = df_reg.loc[~train_mask_r, "PHQ9-Score"].values

    scaler_reg  = StandardScaler()
    X_train_reg = scaler_reg.fit_transform(X_train_reg)
    X_test_reg  = scaler_reg.transform(X_test_reg)

    reg_results = []
    for name, grid in REGRESS_GRIDS.items():
        try:
            gs = GridSearchCV(make_regressor(name), grid,
                              cv=KFold(3, shuffle=True, random_state=43),
                              scoring="r2", n_jobs=1, refit=True)
            gs.fit(X_train_reg, y_train_reg)
            best   = gs.best_estimator_
            y_pred = best.predict(X_test_reg)
            row = {
                "llm": args.llm, "model": name, "best_params": str(gs.best_params_),
                "r2":   r2_score(y_test_reg, y_pred),
                "mae":  mean_absolute_error(y_test_reg, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test_reg, y_pred))),
            }
            reg_results.append(row)
            logging.info(f"  {name}: R²={row['r2']:.3f}  MAE={row['mae']:.2f}")
        except Exception as e:
            logging.error(f"  {name} failed: {e}")

    reg_df = pd.DataFrame(reg_results).sort_values("mae")
    reg_out = OUTPUT_DIR / f"regression_results_{args.llm}.csv"
    reg_df.to_csv(reg_out, index=False)
    logging.info(f"Regression results → {reg_out}")

    logging.info("\nDone.")


if __name__ == "__main__":
    main()
