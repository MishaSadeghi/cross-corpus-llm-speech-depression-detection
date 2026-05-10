"""
ProposedDataset Multimodal Depression Classification + Regression — Fixed Split
XLSR-53 Multilingual Edition
=========================================================================
Derived from proposed_dataset_multimodal_fixed_split_v2.py, but:
  - Only uses XLSR-53 multilingual audio (eGeMAPS + old wav2vec commented out)
  - Adds both diarised and non-diarised xlsr53ml sources
  - Same text features (DepRoBERTa probs + 11 question features via llama_3.3_70B)
  - Same fixed fold-1 split, same models, same audio conditions:
      Condition 1: pca_only [20, 50, 100]
      Condition 2: selectk200 → PCA → selectk15 → redundancy pruning

Results saved to:
  FEATURES_BASE/results/proposed_dataset_multimodal_fixed_split/
    classification_results_xlsr53ml.csv
    regression_results_xlsr53ml.csv
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
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
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
# PATHS  (same as v2)
# ============================================================================
BASE_DIR   = Path("CONFIGURE_ME/repo")
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")
FEATURES_BASE   = DATASET_ROOT / "features_output"
WOODY_BASE = Path("CONFIGURE_ME/storage")

SPLIT_FILE    = FEATURES_BASE / "splits" / "proposed_dataset_fold1_split.json"
CLASSI_LABELS = BASE_DIR / "Other/classi_labels.csv"
REGRESS_LABELS = Path("CONFIGURE_ME/labels/regress_labels.csv")
DEPROBERTA_MAX_LENGTH = 512

# LLM-specific paths — resolved at runtime from --llm argument
LLM_CODES = {
    "llama_3.1_8B":  "3108",
    "llama_3.1_70B": "3170",
    "llama_3.3_70B": "3370",
}


def build_llm_paths(llm: str) -> dict:
    code = LLM_CODES[llm]
    return {
        "question_feats":    Path(f"CONFIGURE_ME/features/text_features_llama_{code}.csv"),
        "deproberta_dir":    WOODY_BASE / f"deproberta_cv_models/{llm}/fold_1",
        "summary_dir":       Path(f"CONFIGURE_ME/LLM_summaries/proposed_dataset/summaries_{code}"),
    }

# ============================================================================
# AUDIO SETS  — xlsr53ml only (eGeMAPS + old wav2vec commented out)
# ============================================================================
AUDIO_SETS = {
    # ---- eGeMAPS (commented out — results already exist from v2) ----
    # "egemaps_diarised": {
    #     "path": FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    #     "id_col": "patient_id", "high_dim": False,
    # },
    # "egemaps_non_diarised": {
    #     "path": FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    #     "id_col": "patient_id", "high_dim": False,
    # },
    # ---- Old wav2vec (commented out — results already exist from v2) ----
    # "wav2vec_diarised": {
    #     "path": FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_diarised/proposed_dataset_xlsr53_multilingual_diarised.csv",
    #     "id_col": "patient_id", "high_dim": True,
    # },
    # "wav2vec_non_diarised": {
    #     "path": FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_non_diarised/proposed_dataset_xlsr53_multilingual_non_diarised.csv",
    #     "id_col": "file_id", "high_dim": True,
    # },
    # ---- NEW: XLSR-53 Multilingual ----
    "xlsr53ml_diarised": {
        "path": FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_diarised/proposed_dataset_xlsr53_multilingual_diarised.csv",
        "id_col": "patient_id", "high_dim": True,
    },
    "xlsr53ml_non_diarised": {
        "path": FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_non_diarised/proposed_dataset_xlsr53_multilingual_non_diarised.csv",
        "id_col": "patient_id", "high_dim": True,
    },
}

OUTPUT_DIR = FEATURES_BASE / "results" / "proposed_dataset_multimodal_fixed_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORR_DIR   = OUTPUT_DIR / "feature_correlations"
CORR_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FEATURE SELECTION PARAMETERS
# ============================================================================
W2V_PRESEL_K   = 200
W2V_MM_PCA_N   = 50
W2V_MM_POSTK   = 15
REDUND_THRESH  = 0.70
SPEARMAN_SIG   = 0.10
PCA_FOR_AUDIO  = [20, 50, 100]

# ============================================================================
# GRIDS  (identical to proposed_dataset_multimodal_fixed_split_v2.py)
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
    CLASSI_GRIDS["LightGBM"]  = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1]}
    REGRESS_GRIDS["LightGBM"] = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1]}
if CB:
    CLASSI_GRIDS["CatBoost"]  = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}
    REGRESS_GRIDS["CatBoost"] = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}


# ============================================================================
# HELPERS  (identical to proposed_dataset_multimodal_fixed_split_v2.py)
# ============================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_classifier(name):
    m = {"LogisticRegression": lambda: LogisticRegression(max_iter=10000, random_state=42),
         "SVC": lambda: SVC(probability=True, random_state=42),
         "RandomForest": lambda: RandomForestClassifier(random_state=42),
         "XGBoost": lambda: XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42),
         "AdaBoost": lambda: AdaBoostClassifier(random_state=42),
         "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
         "KNN": lambda: KNeighborsClassifier(),
         "ExtraTrees": lambda: ExtraTreesClassifier(random_state=42),
         "GradientBoosting": lambda: GradientBoostingClassifier(random_state=42),
         "MLP": lambda: MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)}
    if LGBM: m["LightGBM"] = lambda: LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
    if CB:   m["CatBoost"] = lambda: CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    return m[name]()


def make_regressor(name):
    m = {"Ridge": lambda: Ridge(max_iter=10000), "Lasso": lambda: Lasso(max_iter=10000),
         "ElasticNet": lambda: ElasticNet(max_iter=10000), "SVR": lambda: SVR(),
         "RandomForest": lambda: RandomForestRegressor(random_state=42),
         "AdaBoost": lambda: AdaBoostRegressor(random_state=42),
         "DecisionTree": lambda: DecisionTreeRegressor(random_state=42),
         "KNN": lambda: KNeighborsRegressor(),
         "ExtraTrees": lambda: ExtraTreesRegressor(random_state=42),
         "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
         "MLP": lambda: MLPRegressor(random_state=42, max_iter=1000, early_stopping=True),
         "XGBoost": lambda: XGBRegressor(objective="reg:squarederror", random_state=42)}
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
        df.to_csv(save_path, index=False)
        logging.info(f"  Correlation CSV → {save_path.name}")
    return df


def prune_audio_vs_text(X_audio_train, X_text_train, threshold=REDUND_THRESH):
    """Return keep-mask: drop audio dims with Pearson |r|>threshold vs any text dim."""
    keep = np.ones(X_audio_train.shape[1], dtype=bool)
    for ai in range(X_audio_train.shape[1]):
        for ti in range(X_text_train.shape[1]):
            r, _ = stats.pearsonr(X_audio_train[:, ai], X_text_train[:, ti])
            if abs(r) > threshold:
                keep[ai] = False; break
    n_pruned = int((~keep).sum())
    logging.info(f"  Redundancy pruning: {n_pruned}/{X_audio_train.shape[1]} audio dims removed (|r|>{threshold})")
    return keep


def run_mm(X_train, y_train, X_test, y_test, combo_label, grids, make_fn, cv, task):
    results = []
    for name, grid in grids.items():
        try:
            pipe = Pipeline([("clf", make_fn(name))])
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1" if task == "cls" else "r2",
                              n_jobs=1, refit=True)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_; yp = best.predict(X_test)
            row = {"combo": combo_label, "model": name, "best_params": str(gs.best_params_)}
            if task == "cls":
                ys = get_scores(best, X_test)
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
# TEXT FEATURES  (identical to proposed_dataset_multimodal_fixed_split_v2.py)
# ============================================================================
def load_summaries(pids, summary_dir):
    out = {}
    for pid in pids:
        for suf in ["_transcription.json", "_whisper.json"]:
            p = summary_dir / f"{pid:03d}{suf}"
            if p.exists():
                try:
                    with open(p) as f:
                        s = json.load(f).get("summary", "").strip()
                    if s: out[pid] = s; break
                except Exception: pass
    return out


def extract_text_features(all_ids, llm_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(llm_paths["deproberta_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(llm_paths["deproberta_dir"]).to(device)
    model.eval()
    summaries = load_summaries(all_ids, llm_paths["summary_dir"])
    rows = []
    for pid in all_ids:
        text = summaries.get(pid)
        if text is None:
            rows.append({"patient_id": pid, "prob_severe": 0.0, "prob_moderate": 0.0, "prob_not_depression": 0.0})
            continue
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=DEPROBERTA_MAX_LENGTH).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
        rows.append({"patient_id": pid, "prob_severe": float(probs[0]),
                     "prob_moderate": float(probs[1]), "prob_not_depression": float(probs[2])})
    prob_df = pd.DataFrame(rows)
    q_df = pd.read_csv(llm_paths["question_feats"])
    q_cols = [f"q{i}" for i in range(1, 12)]
    merged = prob_df.merge(q_df[["patient_id"] + q_cols], on="patient_id", how="inner")
    text_feat_cols = ["prob_severe", "prob_moderate", "prob_not_depression"] + q_cols
    return merged, text_feat_cols


def load_audio_features(aset_name):
    cfg = AUDIO_SETS[aset_name]
    df = pd.read_csv(cfg["path"])
    id_col = cfg["id_col"]
    if id_col == "file_id":
        df["patient_id"] = df[id_col].astype(str).str.extract(r"(\d+)").astype(int)
        df = df.drop(columns=[id_col])
    meta = ["patient_id", "n_segments"]
    feat_cols = [c for c in df.columns if c not in meta]
    return df[["patient_id"] + feat_cols], feat_cols, cfg["high_dim"]


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True, choices=list(LLM_CODES.keys()),
                        help="LLM used for text summaries and DepRoBERTa fine-tuning")
    args = parser.parse_args()

    llm_paths = build_llm_paths(args.llm)
    logging.info(f"LLM: {args.llm}")
    for k, v in llm_paths.items():
        logging.info(f"  [{('OK' if Path(v).exists() else 'MISSING')}] {k}: {v}")

    set_seed(42)

    print("=" * 60)
    print("ProposedDataset Multimodal Fixed-Split — XLSR-53 Multilingual Only")
    print("=" * 60)

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_ids = set(split["train_ids"])
    test_ids  = set(split["test_ids"])
    all_ids   = sorted(train_ids | test_ids)

    classi_labels  = pd.read_csv(CLASSI_LABELS).set_index("patient_id")
    regress_labels = pd.read_csv(REGRESS_LABELS).set_index("patient_id")

    text_df, text_cols = extract_text_features(all_ids, llm_paths)
    logging.info(f"Text features: {len(text_cols)} cols, {len(text_df)} patients")

    cv_cls = StratifiedKFold(3, shuffle=True, random_state=43)
    cv_reg = KFold(3, shuffle=True, random_state=43)

    all_cls, all_reg = [], []

    for aset_name, cfg in AUDIO_SETS.items():
        if not cfg["path"].exists():
            logging.warning(f"Audio not found: {cfg['path']} — skipping {aset_name}"); continue

        audio_df, audio_cols, high_dim = load_audio_features(aset_name)
        logging.info(f"\n{'='*70}\nMultimodal xlsr53ml: text + {aset_name} ({len(audio_cols)} audio feats)")

        merged = text_df.merge(audio_df, on="patient_id", how="inner")
        train_mask = merged["patient_id"].isin(train_ids)

        text_arr  = merged[text_cols].values.astype(float)
        audio_arr = np.nan_to_num(merged[audio_cols].values.astype(float))

        sc_t = StandardScaler(); sc_a = StandardScaler()
        text_tr   = sc_t.fit_transform(text_arr[train_mask])
        text_te   = sc_t.transform(text_arr[~train_mask])
        audio_tr  = sc_a.fit_transform(audio_arr[train_mask])
        audio_te  = sc_a.transform(audio_arr[~train_mask])

        y_train_cls = classi_labels.loc[merged.loc[train_mask,  "patient_id"].values, "depressed"].values
        y_test_cls  = classi_labels.loc[merged.loc[~train_mask, "patient_id"].values, "depressed"].values
        y_train_reg = regress_labels.loc[merged.loc[train_mask,  "patient_id"].values, "PHQ9-Score"].values
        y_test_reg  = regress_labels.loc[merged.loc[~train_mask, "patient_id"].values, "PHQ9-Score"].values

        corr_path = CORR_DIR / f"feature_corr_{aset_name}.csv"
        compute_spearman_correlations(audio_tr, y_train_cls, audio_cols, save_path=corr_path)

        # All xlsr53ml sources are high_dim → only wav2vec-style conditions
        # ----------------------------------------------------------------
        # Condition 1: pca_only
        # ----------------------------------------------------------------
        for n_pca in PCA_FOR_AUDIO:
            eff = min(n_pca, audio_tr.shape[0], audio_tr.shape[1])
            pca = PCA(n_components=eff, random_state=42)
            audio_tr_pca = pca.fit_transform(audio_tr); audio_te_pca = pca.transform(audio_te)
            combo_pca = f"text+{aset_name}_pca{n_pca}_no_fs"
            X_tr = np.hstack([text_tr, audio_tr_pca]); X_te = np.hstack([text_te, audio_te_pca])
            logging.info(f"  [{combo_pca}] train={X_tr.shape}, test={X_te.shape}")
            all_cls += run_mm(X_tr, y_train_cls, X_te, y_test_cls, combo_pca, CLASSI_GRIDS, make_classifier, cv_cls, "cls")
            all_reg += run_mm(X_tr, y_train_reg, X_te, y_test_reg, combo_pca, REGRESS_GRIDS, make_regressor, cv_reg, "reg")

        # ----------------------------------------------------------------
        # Condition 2: selectk_pca + redundancy pruning
        # ----------------------------------------------------------------
        k_raw = min(W2V_PRESEL_K, audio_tr.shape[1])
        sel_pre = SelectKBest(f_classif, k=k_raw).fit(audio_tr, y_train_cls)
        audio_tr_pre = sel_pre.transform(audio_tr); audio_te_pre = sel_pre.transform(audio_te)

        eff_pca = min(W2V_MM_PCA_N, audio_tr_pre.shape[0], audio_tr_pre.shape[1])
        pca_mm  = PCA(n_components=eff_pca, random_state=42)
        audio_tr_pca = pca_mm.fit_transform(audio_tr_pre); audio_te_pca = pca_mm.transform(audio_te_pre)

        k_post = min(W2V_MM_POSTK, audio_tr_pca.shape[1])
        sel_post = SelectKBest(f_classif, k=k_post).fit(audio_tr_pca, y_train_cls)
        audio_tr_sk = sel_post.transform(audio_tr_pca); audio_te_sk = sel_post.transform(audio_te_pca)

        keep = prune_audio_vs_text(audio_tr_sk, text_tr)
        if keep.sum() == 0:
            logging.warning(f"  [{aset_name} selectk_pca] All features pruned — skipping")
        else:
            audio_tr_pr = audio_tr_sk[:, keep]; audio_te_pr = audio_te_sk[:, keep]
            combo_skpca = f"text+{aset_name}_selectk{k_raw}_pca{eff_pca}_k{k_post}_pruned"
            X_tr = np.hstack([text_tr, audio_tr_pr]); X_te = np.hstack([text_te, audio_te_pr])
            logging.info(f"  [{combo_skpca}] train={X_tr.shape}, test={X_te.shape} ({keep.sum()} dims retained)")
            all_cls += run_mm(X_tr, y_train_cls, X_te, y_test_cls, combo_skpca, CLASSI_GRIDS, make_classifier, cv_cls, "cls")
            all_reg += run_mm(X_tr, y_train_reg, X_te, y_test_reg, combo_skpca, REGRESS_GRIDS, make_regressor, cv_reg, "reg")

    if all_cls:
        out_cls = OUTPUT_DIR / f"classification_results_xlsr53ml_{args.llm}.csv"
        pd.DataFrame(all_cls).sort_values("f1", ascending=False).to_csv(out_cls, index=False)
        logging.info(f"Classification results → {out_cls}")
    if all_reg:
        out_reg = OUTPUT_DIR / f"regression_results_xlsr53ml_{args.llm}.csv"
        pd.DataFrame(all_reg).sort_values("mae").to_csv(out_reg, index=False)
        logging.info(f"Regression results → {out_reg}")
    logging.info(f"\nAll results → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
