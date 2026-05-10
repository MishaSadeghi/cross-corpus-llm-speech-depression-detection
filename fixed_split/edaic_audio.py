"""
E-DAIC Audio-Based Depression Classification + Regression (Within-Dataset)
XLSR-53 Multilingual Edition
===========================================================================
Same structure as edaic_audio_within_v2.py, but uses the new XLSR-53
multilingual features (facebook/wav2vec2-large-xlsr-53) instead of the
English-partitioned XLSR features / eGeMAPS.

Feature file:
  CONFIGURE_ME/proposed_dataset/
  features_output/edaic_xlsr53_multilingual/
  edaic_xlsr53_multilingual.csv
  → 275 participants × 6144 features  (6 stats × 1024 dims)
  → columns: participant_id, n_segments, min_0 … kurt_1023

Uses the official E-DAIC train/dev/test splits (loaded from edaic_labels.csv)
with PredefinedSplit for HP tuning (dev = validation fold).

Two feature-selection conditions applied (matching v2 behaviour for XLSR):
  1. xlsr53ml_pca{N}       – PCA with N ∈ [10,20,50,100] components
  2. xlsr53ml_selectk200_pca{N} – SelectKBest(f_classif, k=200) → PCA

Results saved to:
  FEATURES_BASE/results/edaic_audio_within/classification_results_xlsr53ml.csv
  FEATURES_BASE/results/edaic_audio_within/regression_results_xlsr53ml.csv
"""

import logging
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
# PATHS
# ============================================================================
BASE_DIR = Path("CONFIGURE_ME/repo")
FEATURES_BASE = Path("CONFIGURE_ME/proposed_dataset/features_output")

EDAIC_LABELS = Path("CONFIGURE_ME/labels/edaic_labels.csv")

FEATURE_PATH = FEATURES_BASE / "edaic_xlsr53_multilingual/edaic_xlsr53_multilingual.csv"

OUTPUT_DIR = FEATURES_BASE / "results" / "edaic_audio_within"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORR_DIR = OUTPUT_DIR / "feature_correlations"
CORR_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FEATURE SELECTION PARAMETERS  (matching edaic_audio_within_v2.py)
# ============================================================================
W2V_PRESEL_K   = 200              # SelectKBest k on raw features before PCA
SPEARMAN_SIG   = 0.10             # |ρ| threshold for annotation
PCA_COMPONENTS = [10, 20, 50, 100]

# ============================================================================
# GRIDS  (identical to edaic_audio_within_v2.py)
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


# ============================================================================
# HELPERS  (identical to edaic_audio_within_v2.py)
# ============================================================================
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
    if CB: m["CatBoost"] = lambda: CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
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
    if CB: m["CatBoost"] = lambda: CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
    return m[name]()


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return s if s.ndim == 1 else s[:, 1]
    return model.predict(X).astype(float)


def compute_spearman_correlations(X_train, y_train, feat_names, save_path=None):
    """Compute Spearman ρ(feature, label) on train split and optionally save to CSV."""
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


def apply_selectk(X_td, y_train_portion, X_test, train_mask, k, feat_names):
    """Fit SelectKBest on train portion of X_td, apply to full X_td and X_test."""
    k_eff = min(k, X_td.shape[1])
    sel = SelectKBest(f_classif, k=k_eff)
    sel.fit(X_td[train_mask], y_train_portion)
    X_td_sel = sel.transform(X_td)
    X_test_sel = sel.transform(X_test)
    sel_names = [feat_names[i] for i in sel.get_support(indices=True)]
    return X_td_sel, X_test_sel, sel_names


def run_models(X_td, y_td, X_test, y_test, X_train_only, y_train_only,
               ps, fset_label, task, grids, make_fn, score_fn):
    """Run all classifiers/regressors for one feature condition."""
    results = []
    for name, grid in grids.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_fn(name))])
            gs = GridSearchCV(pipe, grid, cv=ps, scoring=score_fn, n_jobs=1, refit=True)
            gs.fit(X_td, y_td)
            # Retrain best on train-only (no dev leakage)
            gs.best_estimator_.fit(X_train_only, y_train_only)
            yp = gs.best_estimator_.predict(X_test)
            row = {"feature_set": fset_label, "model": name, "best_params": str(gs.best_params_)}
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
            logging.error(f"    {task.upper()} {name} [{fset_label}] failed: {e}")
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    random.seed(42); np.random.seed(42)

    print("=" * 60)
    print("E-DAIC XLSR-53 Multilingual — Within-Dataset Analysis")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load labels
    # ------------------------------------------------------------------
    if not EDAIC_LABELS.exists():
        logging.error(f"Labels file not found: {EDAIC_LABELS}")
        return
    labels = pd.read_csv(EDAIC_LABELS)

    # ------------------------------------------------------------------
    # Load feature file
    # ------------------------------------------------------------------
    if not FEATURE_PATH.exists():
        logging.error(f"Feature file not found: {FEATURE_PATH}")
        return

    feat_df = pd.read_csv(FEATURE_PATH)
    logging.info(f"Loaded features: {feat_df.shape}  →  {FEATURE_PATH.name}")

    # ID column normalisation
    if "participant_id" not in feat_df.columns:
        for alias in ["patient_id", "file_id"]:
            if alias in feat_df.columns:
                feat_df = feat_df.rename(columns={alias: "participant_id"})
                break

    meta_cols = ["participant_id", "n_segments"]
    feat_cols = [c for c in feat_df.columns if c not in meta_cols]
    logging.info(f"Feature columns: {len(feat_cols)}")

    # ------------------------------------------------------------------
    # Merge with labels
    # ------------------------------------------------------------------
    merged = feat_df.merge(
        labels[["participant_id", "split", "depressed", "PHQ_score"]],
        on="participant_id")
    logging.info(f"After merge: {len(merged)} rows  "
                 f"(train={sum(merged['split']=='train')}, "
                 f"dev={sum(merged['split']=='dev')}, "
                 f"test={sum(merged['split']=='test')})")

    td         = merged[merged['split'].isin(['train', 'dev'])]
    test_data  = merged[merged['split'] == 'test']
    train_only = merged[merged['split'] == 'train']

    train_mask_td = (td['split'] == 'train').values
    fold_idx  = np.array([-1 if s == 'train' else 0 for s in td['split']])
    ps        = PredefinedSplit(fold_idx)

    X_td_raw   = np.nan_to_num(td[feat_cols].values.astype(float))
    y_td_cls   = td['depressed'].values
    y_td_reg   = td['PHQ_score'].values
    X_test_raw = np.nan_to_num(test_data[feat_cols].values.astype(float))
    y_test_cls = test_data['depressed'].values
    y_test_reg = test_data['PHQ_score'].values

    X_train_only_raw = np.nan_to_num(train_only[feat_cols].values.astype(float))
    y_train_cls      = train_only['depressed'].values
    y_train_reg      = train_only['PHQ_score'].values

    fset_name = "xlsr53ml"

    # ---- Spearman correlation on train portion ----
    corr_path = CORR_DIR / f"feature_corr_{fset_name}.csv"
    compute_spearman_correlations(X_td_raw[train_mask_td], y_td_cls[train_mask_td],
                                  feat_cols, save_path=corr_path)

    all_cls, all_reg = [], []

    # Scale once (shared across PCA configs) for efficiency
    scaler = StandardScaler()
    X_td_sc   = scaler.fit_transform(X_td_raw)
    X_test_sc = scaler.transform(X_test_raw)

    # ---- Condition 1: pca_only ----
    for n_pca in PCA_COMPONENTS:
        eff = min(n_pca, X_td_sc.shape[0], X_td_sc.shape[1])
        pca = PCA(n_components=eff, random_state=42)
        X_td_pca   = pca.fit_transform(X_td_sc)
        X_test_pca = pca.transform(X_test_sc)
        pca_names  = [f"PC{i}" for i in range(eff)]

        if n_pca == PCA_COMPONENTS[0]:
            corr_pca_path = CORR_DIR / f"feature_corr_{fset_name}_pca{n_pca}.csv"
            compute_spearman_correlations(X_td_pca[train_mask_td],
                                          y_td_cls[train_mask_td],
                                          pca_names, save_path=corr_pca_path)

        label_pca = f"{fset_name}_pca{n_pca}_no_fs"
        logging.info(f"\n  [{label_pca}]")
        all_cls += run_models(X_td_pca, y_td_cls, X_test_pca, y_test_cls,
                              X_td_pca[train_mask_td], y_train_cls, ps,
                              label_pca, "cls", CLASSI_GRIDS, make_classifier, "f1")
        all_reg += run_models(X_td_pca, y_td_reg, X_test_pca, y_test_reg,
                              X_td_pca[train_mask_td], y_train_reg, ps,
                              label_pca, "reg", REGRESS_GRIDS, make_regressor, "r2")

    # ---- Condition 2: selectk_pca (supervised pre-filter → PCA) ----
    k_raw = min(W2V_PRESEL_K, X_td_sc.shape[1])
    X_td_presel, X_test_presel, _ = apply_selectk(
        X_td_sc, y_train_cls, X_test_sc, train_mask_td, k_raw, feat_cols)
    logging.info(f"  XLSR pre-selectk{k_raw}: {X_td_sc.shape[1]} → {X_td_presel.shape[1]}")

    for n_pca in PCA_COMPONENTS:
        eff = min(n_pca, X_td_presel.shape[0], X_td_presel.shape[1])
        pca = PCA(n_components=eff, random_state=42)
        X_td_pca   = pca.fit_transform(X_td_presel)
        X_test_pca = pca.transform(X_test_presel)

        label_sk_pca = f"{fset_name}_selectk{k_raw}_pca{n_pca}"
        logging.info(f"\n  [{label_sk_pca}]")
        all_cls += run_models(X_td_pca, y_td_cls, X_test_pca, y_test_cls,
                              X_td_pca[train_mask_td], y_train_cls, ps,
                              label_sk_pca, "cls", CLASSI_GRIDS, make_classifier, "f1")
        all_reg += run_models(X_td_pca, y_td_reg, X_test_pca, y_test_reg,
                              X_td_pca[train_mask_td], y_train_reg, ps,
                              label_sk_pca, "reg", REGRESS_GRIDS, make_regressor, "r2")

    # ---- Save results ----
    if all_cls:
        out_cls = OUTPUT_DIR / "classification_results_xlsr53ml.csv"
        pd.DataFrame(all_cls).sort_values("f1", ascending=False).to_csv(out_cls, index=False)
        logging.info(f"\nClassification results → {out_cls}")
    if all_reg:
        out_reg = OUTPUT_DIR / "regression_results_xlsr53ml.csv"
        pd.DataFrame(all_reg).sort_values("mae").to_csv(out_reg, index=False)
        logging.info(f"Regression results     → {out_reg}")

    logging.info(f"\nDone. Results → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
