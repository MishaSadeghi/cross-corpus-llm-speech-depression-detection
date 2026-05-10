"""
ProposedDataset Audio-Based Depression Classification + Regression — Fixed Split
XLSR-53 Multilingual Edition
=========================================================================
Derived from proposed_dataset_audio_fixed_split_v2.py, but:
  - Only uses XLSR-53 multilingual features (eGeMAPS commented out)
  - Adds both diarised and non-diarised xlsr53ml sources
  - Same fixed fold-1 split, same model grids, same PCA conditions

Feature sources (6144-dim each):
  xlsr53ml_diarised     → proposed_dataset_xlsr53_multilingual_diarised.csv
  xlsr53ml_non_diarised → proposed_dataset_xlsr53_multilingual_non_diarised.csv

Results saved to:
  FEATURES_BASE/results/proposed_dataset_audio_fixed_split/
    classification_results_xlsr53ml.csv
    regression_results_xlsr53ml.csv
"""

import json
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
BASE_DIR   = Path("CONFIGURE_ME/repo")
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")
FEATURES_BASE   = DATASET_ROOT / "features_output"

SPLIT_FILE     = FEATURES_BASE / "splits" / "proposed_dataset_fold1_split.json"
CLASSI_LABELS  = Path("CONFIGURE_ME/labels/classi_labels.csv")
REGRESS_LABELS = Path("CONFIGURE_ME/labels/regress_labels.csv")

# ============================================================================
# FEATURE SETS  — xlsr53ml only (eGeMAPS commented out)
# ============================================================================
FEATURE_SETS = {
    # ---- eGeMAPS (commented out — results already exist from v2) ----
    # "egemaps_diarised": {
    #     "path": FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    #     "id_col": "patient_id", "high_dim": False,
    # },
    # "egemaps_non_diarised": {
    #     "path": FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    #     "id_col": "patient_id", "high_dim": False,
    # },
    # ---- Original wav2vec (commented out — results already exist from v2) ----
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

OUTPUT_DIR = FEATURES_BASE / "results" / "proposed_dataset_audio_fixed_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORR_DIR   = OUTPUT_DIR / "feature_correlations"
CORR_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FEATURE SELECTION PARAMETERS
# ============================================================================
W2V_PRESEL_K   = 200
SPEARMAN_SIG   = 0.10
PCA_COMPONENTS = [10, 20, 50, 100]

# ============================================================================
# GRIDS  (identical to proposed_dataset_audio_fixed_split_v2.py)
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
    CLASSI_GRIDS["LightGBM"]  = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1], "clf__num_leaves": [15, 31, 63]}
    REGRESS_GRIDS["LightGBM"] = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1], "clf__num_leaves": [15, 31, 63]}
if CB:
    CLASSI_GRIDS["CatBoost"]  = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}
    REGRESS_GRIDS["CatBoost"] = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}

# ============================================================================
# HELPERS
# ============================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)


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


def load_features(fset_name):
    cfg = FEATURE_SETS[fset_name]
    df = pd.read_csv(cfg["path"])
    id_col = cfg["id_col"]
    if id_col == "file_id":
        df["patient_id"] = df[id_col].astype(str).str.extract(r"(\d+)").astype(int)
        df = df.drop(columns=[id_col])
    meta = ["patient_id", "n_segments"]
    feat_cols = [c for c in df.columns if c not in meta]
    return df[["patient_id"] + feat_cols], feat_cols, cfg["high_dim"]


def run_classification(X_train, y_train, X_test, y_test, fset_label, cv):
    results = []
    for name, grid in CLASSI_GRIDS.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_classifier(name))])
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=1, refit=True)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            yp = best.predict(X_test); ys = get_scores(best, X_test)
            results.append({"feature_set": fset_label, "model": name,
                             "best_params": str(gs.best_params_),
                             "ap": average_precision_score(y_test, ys),
                             "auc": roc_auc_score(y_test, ys),
                             "bal_acc": balanced_accuracy_score(y_test, yp),
                             "f1": f1_score(y_test, yp),
                             "precision": precision_score(y_test, yp),
                             "recall": recall_score(y_test, yp)})
            logging.info(f"    CLS {name}: F1={results[-1]['f1']:.3f}")
        except Exception as e:
            logging.error(f"    CLS {name} [{fset_label}] failed: {e}")
    return results


def run_regression(X_train, y_train, X_test, y_test, fset_label, cv):
    results = []
    for name, grid in REGRESS_GRIDS.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_regressor(name))])
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="r2", n_jobs=1, refit=True)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_; yp = best.predict(X_test)
            results.append({"feature_set": fset_label, "model": name,
                             "best_params": str(gs.best_params_),
                             "r2": r2_score(y_test, yp),
                             "mae": mean_absolute_error(y_test, yp),
                             "rmse": float(np.sqrt(mean_squared_error(y_test, yp)))})
            logging.info(f"    REG {name}: MAE={results[-1]['mae']:.2f}")
        except Exception as e:
            logging.error(f"    REG {name} [{fset_label}] failed: {e}")
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    set_seed(42)

    print("=" * 60)
    print("ProposedDataset Audio Fixed-Split — XLSR-53 Multilingual Only")
    print("=" * 60)

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_ids = set(split["train_ids"])

    classi_labels  = pd.read_csv(CLASSI_LABELS)
    regress_labels = pd.read_csv(REGRESS_LABELS)

    cv_cls = StratifiedKFold(3, shuffle=True, random_state=43)
    cv_reg = KFold(3, shuffle=True, random_state=43)

    all_cls, all_reg = [], []

    for fset_name in FEATURE_SETS:
        try:
            feat_df, feat_cols, high_dim = load_features(fset_name)
        except FileNotFoundError as e:
            logging.error(f"Feature file not found for {fset_name}: {e}"); continue

        logging.info(f"\n{'='*70}\n{fset_name}: {len(feat_cols)} features, high_dim={high_dim}")

        df_cls = feat_df.merge(classi_labels[["patient_id", "depressed"]], on="patient_id")
        df_reg = feat_df.merge(regress_labels[["patient_id", "PHQ9-Score"]], on="patient_id")

        train_mask_cls = df_cls["patient_id"].isin(train_ids)
        X_train_cls = np.nan_to_num(df_cls.loc[train_mask_cls, feat_cols].values.astype(float))
        y_train_cls = df_cls.loc[train_mask_cls, "depressed"].values
        X_test_cls  = np.nan_to_num(df_cls.loc[~train_mask_cls, feat_cols].values.astype(float))
        y_test_cls  = df_cls.loc[~train_mask_cls, "depressed"].values

        train_mask_reg = df_reg["patient_id"].isin(train_ids)
        X_train_reg = np.nan_to_num(df_reg.loc[train_mask_reg, feat_cols].values.astype(float))
        y_train_reg = df_reg.loc[train_mask_reg, "PHQ9-Score"].values
        X_test_reg  = np.nan_to_num(df_reg.loc[~train_mask_reg, feat_cols].values.astype(float))
        y_test_reg  = df_reg.loc[~train_mask_reg, "PHQ9-Score"].values

        corr_path = CORR_DIR / f"feature_corr_{fset_name}.csv"
        compute_spearman_correlations(X_train_cls, y_train_cls, feat_cols, save_path=corr_path)

        # All xlsr53ml sources are high_dim → always use PCA conditions
        # ----------------------------------------------------------------
        # Condition 1: pca_only (PCA inside pipeline)
        # ----------------------------------------------------------------
        for n_pca in PCA_COMPONENTS:
            lbl_pca = f"{fset_name}_pca{n_pca}_no_fs"
            eff_cls = min(n_pca, X_train_cls.shape[0], X_train_cls.shape[1])
            eff_reg = min(n_pca, X_train_reg.shape[0], X_train_reg.shape[1])

            if n_pca == PCA_COMPONENTS[0]:
                sc_tmp  = StandardScaler()
                pca_tmp = PCA(n_components=eff_cls, random_state=42)
                X_tmp   = pca_tmp.fit_transform(sc_tmp.fit_transform(X_train_cls))
                pca_names = [f"PC{i}" for i in range(eff_cls)]
                compute_spearman_correlations(X_tmp, y_train_cls, pca_names,
                                              save_path=CORR_DIR / f"feature_corr_{fset_name}_pca{n_pca}.csv")

            logging.info(f"\n  [{lbl_pca}]")
            for name, grid in CLASSI_GRIDS.items():
                try:
                    pipe = Pipeline([("scaler", StandardScaler()),
                                     ("pca", PCA(n_components=eff_cls, random_state=42)),
                                     ("clf", make_classifier(name))])
                    gs = GridSearchCV(pipe, grid, cv=cv_cls, scoring="f1", n_jobs=1, refit=True)
                    gs.fit(X_train_cls, y_train_cls)
                    best = gs.best_estimator_; yp = best.predict(X_test_cls); ys = get_scores(best, X_test_cls)
                    all_cls.append({"feature_set": lbl_pca, "model": name, "n_pca": n_pca,
                                    "best_params": str(gs.best_params_),
                                    "ap": average_precision_score(y_test_cls, ys),
                                    "auc": roc_auc_score(y_test_cls, ys),
                                    "bal_acc": balanced_accuracy_score(y_test_cls, yp),
                                    "f1": f1_score(y_test_cls, yp),
                                    "precision": precision_score(y_test_cls, yp),
                                    "recall": recall_score(y_test_cls, yp)})
                    logging.info(f"    CLS {name}: F1={all_cls[-1]['f1']:.3f}")
                except Exception as e:
                    logging.error(f"    CLS {name} [{lbl_pca}] failed: {e}")
            for name, grid in REGRESS_GRIDS.items():
                try:
                    pipe = Pipeline([("scaler", StandardScaler()),
                                     ("pca", PCA(n_components=eff_reg, random_state=42)),
                                     ("clf", make_regressor(name))])
                    gs = GridSearchCV(pipe, grid, cv=cv_reg, scoring="r2", n_jobs=1, refit=True)
                    gs.fit(X_train_reg, y_train_reg)
                    best = gs.best_estimator_; yp = best.predict(X_test_reg)
                    all_reg.append({"feature_set": lbl_pca, "model": name, "n_pca": n_pca,
                                    "best_params": str(gs.best_params_),
                                    "r2": r2_score(y_test_reg, yp),
                                    "mae": mean_absolute_error(y_test_reg, yp),
                                    "rmse": float(np.sqrt(mean_squared_error(y_test_reg, yp)))})
                    logging.info(f"    REG {name}: MAE={all_reg[-1]['mae']:.2f}")
                except Exception as e:
                    logging.error(f"    REG {name} [{lbl_pca}] failed: {e}")

        # ----------------------------------------------------------------
        # Condition 2: selectk_pca (supervised pre-select → PCA)
        # ----------------------------------------------------------------
        k_raw = min(W2V_PRESEL_K, X_train_cls.shape[1])
        sel_pre = SelectKBest(f_classif, k=k_raw).fit(X_train_cls, y_train_cls)
        X_tr_pre_cls = sel_pre.transform(X_train_cls); X_te_pre_cls = sel_pre.transform(X_test_cls)
        X_tr_pre_reg = sel_pre.transform(X_train_reg); X_te_pre_reg = sel_pre.transform(X_test_reg)
        logging.info(f"  selectk{k_raw}: {X_train_cls.shape[1]} → {k_raw}")

        for n_pca in PCA_COMPONENTS:
            lbl_sk_pca = f"{fset_name}_selectk{k_raw}_pca{n_pca}"
            eff_cls = min(n_pca, X_tr_pre_cls.shape[0], X_tr_pre_cls.shape[1])
            eff_reg = min(n_pca, X_tr_pre_reg.shape[0], X_tr_pre_reg.shape[1])
            logging.info(f"\n  [{lbl_sk_pca}]")
            for name, grid in CLASSI_GRIDS.items():
                try:
                    pipe = Pipeline([("scaler", StandardScaler()),
                                     ("pca", PCA(n_components=eff_cls, random_state=42)),
                                     ("clf", make_classifier(name))])
                    gs = GridSearchCV(pipe, grid, cv=cv_cls, scoring="f1", n_jobs=1, refit=True)
                    gs.fit(X_tr_pre_cls, y_train_cls)
                    best = gs.best_estimator_; yp = best.predict(X_te_pre_cls); ys = get_scores(best, X_te_pre_cls)
                    all_cls.append({"feature_set": lbl_sk_pca, "model": name, "n_pca": n_pca,
                                    "best_params": str(gs.best_params_),
                                    "ap": average_precision_score(y_test_cls, ys),
                                    "auc": roc_auc_score(y_test_cls, ys),
                                    "bal_acc": balanced_accuracy_score(y_test_cls, yp),
                                    "f1": f1_score(y_test_cls, yp),
                                    "precision": precision_score(y_test_cls, yp),
                                    "recall": recall_score(y_test_cls, yp)})
                    logging.info(f"    CLS {name}: F1={all_cls[-1]['f1']:.3f}")
                except Exception as e:
                    logging.error(f"    CLS {name} [{lbl_sk_pca}] failed: {e}")
            for name, grid in REGRESS_GRIDS.items():
                try:
                    pipe = Pipeline([("scaler", StandardScaler()),
                                     ("pca", PCA(n_components=eff_reg, random_state=42)),
                                     ("clf", make_regressor(name))])
                    gs = GridSearchCV(pipe, grid, cv=cv_reg, scoring="r2", n_jobs=1, refit=True)
                    gs.fit(X_tr_pre_reg, y_train_reg)
                    best = gs.best_estimator_; yp = best.predict(X_te_pre_reg)
                    all_reg.append({"feature_set": lbl_sk_pca, "model": name, "n_pca": n_pca,
                                    "best_params": str(gs.best_params_),
                                    "r2": r2_score(y_test_reg, yp),
                                    "mae": mean_absolute_error(y_test_reg, yp),
                                    "rmse": float(np.sqrt(mean_squared_error(y_test_reg, yp)))})
                    logging.info(f"    REG {name}: MAE={all_reg[-1]['mae']:.2f}")
                except Exception as e:
                    logging.error(f"    REG {name} [{lbl_sk_pca}] failed: {e}")

    if all_cls:
        pd.DataFrame(all_cls).sort_values("f1", ascending=False).to_csv(
            OUTPUT_DIR / "classification_results_xlsr53ml.csv", index=False)
    if all_reg:
        pd.DataFrame(all_reg).sort_values("mae").to_csv(
            OUTPUT_DIR / "regression_results_xlsr53ml.csv", index=False)
    logging.info(f"\nResults → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
