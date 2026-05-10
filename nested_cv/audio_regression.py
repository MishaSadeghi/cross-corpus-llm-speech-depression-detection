"""
MAXIMAL Audio Regression v2 - All Audio Sources + Enhanced Feature Selection

v2 changes vs v1:
  - Loops over 4 ProposedDataset audio sources (eGeMAPS diar/non-diar + wav2vec diar/non-diar)
  - Adds 'selectk_pca_pruned': SelectKBest -> PCA cascade
  - Saves results with _v2 suffix; old script untouched
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                               ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR   = Path("CONFIGURE_ME/repo")  # root of this repository
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")
FEATURES_BASE   = DATASET_ROOT / "features_output"

LABELS_CSV = BASE_DIR / "regress_labels.csv"

# v2: all four sources (v1 used only wav2vec_non_diarised — kept as comment):
# FEATURES_CSV = BASE_DIR / "proposed_dataset_wav2vec2_with_diarization_features.csv"
AUDIO_SOURCES = {
    "egemaps_diarised":     FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    "egemaps_non_diarised": FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    "wav2vec_diarised":     FEATURES_BASE / "proposed_dataset_german_xlsr_diarised/proposed_dataset_german_xlsr_diarised.csv",
    "wav2vec_non_diarised": FEATURES_BASE / "proposed_dataset_german_xlsr_non_diarised/proposed_dataset_german_xlsr_non_diarised.csv",
}

OUTPUT_BASE = Path("CONFIGURE_ME/repo/Audio/maximal_regression_v2")
QUALITY_FILTERS = {"no_filter": None}
ID_COL, Y_COL = "patient_id", "PHQ9-Score"
ORIGINAL_EXCLUDE = {177, 207, 299}

FEATURE_METHODS_V2 = ['pca', 'anova', 'mutual_info', 'selectk_pca_pruned']

GRIDS = {
    "Ridge": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
              'regressor__alpha': [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3],
              'regressor__solver': ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]},
    "Lasso": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
              'regressor__alpha': [1e-3, 1e-2, 1e-1, 1.0, 10.0]},
    "ElasticNet": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                   'regressor__alpha': [1e-3, 1e-2, 1e-1, 1.0, 10.0],
                   'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
    "SVR": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
            'regressor__C': [0.1, 1, 10, 100], 'regressor__gamma': [1, 0.1, 0.01, 0.001],
            'regressor__kernel': ["rbf", "poly", "sigmoid"], 'regressor__epsilon': [0.1, 0.2]},
    "RandomForest": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                     'regressor__n_estimators': [50, 100, 200],
                     'regressor__max_depth': [None, 10, 20, 30],
                     'regressor__min_samples_split': [2, 5, 10]},
    "AdaBoost": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                 'regressor__n_estimators': [50, 100, 200],
                 'regressor__learning_rate': [0.01, 0.1, 1.0],
                 'regressor__loss': ["linear", "square", "exponential"]},
    "DecisionTree": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                     'regressor__max_depth': [None, 10, 20, 30],
                     'regressor__min_samples_split': [2, 5, 10],
                     'regressor__criterion': ["squared_error", "friedman_mse", "absolute_error"]},
    "KNN": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
            'regressor__n_neighbors': [3, 5, 7, 9],
            'regressor__weights': ["uniform", "distance"],
            'regressor__metric': ["euclidean", "manhattan", "minkowski"]},
    "ExtraTrees": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                   'regressor__n_estimators': [50, 100, 200],
                   'regressor__max_depth': [None, 10, 20, 30],
                   'regressor__min_samples_split': [2, 5, 10],
                   'regressor__min_samples_leaf': [1, 2, 4]},
    "GradientBoosting": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                         'regressor__n_estimators': [50, 100, 200],
                         'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
                         'regressor__max_depth': [3, 5, 7],
                         'regressor__subsample': [0.8, 1.0]},
    "MLP": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
            'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'regressor__activation': ['relu', 'tanh'],
            'regressor__alpha': [0.0001, 0.001, 0.01],
            'regressor__learning_rate': ['constant', 'adaptive']},
    "XGBoost": {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7],
                'regressor__subsample': [0.8, 1.0]},
}

if LIGHTGBM_AVAILABLE:
    GRIDS["LightGBM"] = {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                         'regressor__n_estimators': [50, 100, 200],
                         'regressor__learning_rate': [0.01, 0.05, 0.1],
                         'regressor__max_depth': [3, 5, 7, -1],
                         'regressor__num_leaves': [15, 31, 63],
                         'regressor__min_child_samples': [10, 20, 30]}

if CATBOOST_AVAILABLE:
    GRIDS["CatBoost"] = {'feature_method': FEATURE_METHODS_V2, 'n_components': [50, 100, 200],
                         'regressor__iterations': [50, 100, 200],
                         'regressor__learning_rate': [0.01, 0.05, 0.1],
                         'regressor__depth': [3, 5, 7],
                         'regressor__l2_leaf_reg': [1, 3, 5]}

# ============================================================================
# FEATURE SELECTOR
# ============================================================================

class FlexibleFeatureSelectorV2:
    """v2: pca / anova / mutual_info (v1) + selectk_pca_pruned (new)."""
    def __init__(self, method='pca', n_components=100, presel_k=200):
        self.method = method
        self.n_components = n_components
        self.presel_k = presel_k
        self.selector = None
        self.scaler = StandardScaler()
        self._presel = None
        self._pca    = None

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        n  = min(self.n_components, X.shape[1], len(y) - 1)
        if self.method == 'anova':
            self.selector = SelectKBest(f_regression, k=min(n, Xs.shape[1])).fit(Xs, y)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_regression, k=min(n, Xs.shape[1])).fit(Xs, y)
        elif self.method == 'pca':
            self.selector = PCA(n_components=n).fit(Xs)
        elif self.method == 'selectk_pca_pruned':
            k = min(self.presel_k, Xs.shape[1])
            self._presel = SelectKBest(f_regression, k=k).fit(Xs, y)
            Xp = self._presel.transform(Xs)
            np_c = min(n, Xp.shape[0] - 1, Xp.shape[1])
            self._pca = PCA(n_components=np_c, random_state=42).fit(Xp)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def transform(self, X):
        Xs = self.scaler.transform(X)
        if self.method == 'selectk_pca_pruned':
            return self._pca.transform(self._presel.transform(Xs))
        return self.selector.transform(Xs)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {'method': self.method, 'n_components': self.n_components,
                'presel_k': self.presel_k}

    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

# ============================================================================
# HELPERS
# ============================================================================

def load_audio_source(source_name):
    df = pd.read_csv(AUDIO_SOURCES[source_name])
    if 'file_id' in df.columns:
        df['patient_id'] = df['file_id'].astype(str).str.extract(r'(\d+)').astype(int)
        df = df.drop(columns=['file_id'])
    elif 'patient_id' not in df.columns:
        fc = df.columns[0]; df['patient_id'] = df[fc].astype(int); df = df.drop(columns=[fc])
    else:
        df['patient_id'] = df['patient_id'].astype(int)
    df = df[[c for c in df.columns if c != 'n_segments']]
    return df[~df['patient_id'].isin(ORIGINAL_EXCLUDE)]


def load_data(source_name, quality_filter):
    feats = load_audio_source(source_name)
    labs  = pd.read_csv(LABELS_CSV)
    labs['patient_id'] = labs['patient_id'].astype(int)
    labs[Y_COL] = pd.to_numeric(labs[Y_COL], errors='coerce')
    labs = labs.dropna(subset=[Y_COL])
    exclude = ORIGINAL_EXCLUDE.copy()
    if quality_filter is not None and hasattr(quality_filter, 'exists') and quality_filter.exists():
        with open(quality_filter) as f:
            exclude.update(int(l.strip()) for l in f if l.strip())
    feats = feats[~feats[ID_COL].isin(exclude)]
    df = feats.merge(labs[[ID_COL, Y_COL]], on=ID_COL, how='inner')
    fc = [c for c in df.columns if c not in [ID_COL, Y_COL]]
    return np.nan_to_num(df[fc].to_numpy(float)), df[Y_COL].to_numpy(float), len(exclude)


def make_estimator(name):
    regs = {
        'Ridge': Ridge(max_iter=100000), 'Lasso': Lasso(max_iter=100000),
        'ElasticNet': ElasticNet(max_iter=100000), 'SVR': SVR(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'KNN': KNeighborsRegressor(),
        'ExtraTrees': ExtraTreesRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'MLP': MLPRegressor(random_state=42, max_iter=1000, early_stopping=True),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
    }
    if LIGHTGBM_AVAILABLE: regs['LightGBM'] = LGBMRegressor(random_state=42, verbose=-1, force_col_wise=True)
    if CATBOOST_AVAILABLE: regs['CatBoost'] = CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
    if name not in regs: raise ValueError(f"Unknown: {name}")
    return Pipeline([('feature_selection', FlexibleFeatureSelectorV2()), ('regressor', regs[name])])


def compute_metrics(y_true, y_pred):
    return {'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def run_experiment(source_name, quality_name, quality_filter, output_dir):
    print(f"\n{'='*70}\nExperiment: {source_name} + {quality_name}\n{'='*70}")
    X, y, n_exc = load_data(source_name, quality_filter)
    print(f"Samples: {len(y)}, Features: {X.shape[1]}, Excluded: {n_exc}")
    print(f"PHQ-9 range: {y.min():.1f}-{y.max():.1f}, mean: {y.mean():.1f}±{y.std():.1f}")
    if len(y) < 50:
        print("⚠️  Too few samples - skipping"); return None

    # Spearman correlation analysis
    corr_path = output_dir / f"spearman_corr_{source_name}_{quality_name}.csv"
    if not corr_path.exists():
        corrs = [abs(scipy_stats.spearmanr(X[:, i], y)[0]) for i in range(min(X.shape[1], 500))]
        pd.DataFrame({'feature_idx': range(len(corrs)), 'spearman_abs_r': corrs}).to_csv(corr_path, index=False)
        print(f"  Correlation saved → {corr_path}")

    results = []
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name in GRIDS.keys():
        print(f"\n  {model_name}:")
        model = make_estimator(model_name)
        per_fold, best_ps = [], []
        for fold, (tr, te) in enumerate(outer_cv.split(X, y), 1):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            pg = GRIDS[model_name].copy()
            pg['feature_selection__method'] = pg.pop('feature_method')
            pg['feature_selection__n_components'] = pg.pop('n_components')
            gs = RandomizedSearchCV(model, pg, n_iter=50,
                                    cv=KFold(3, shuffle=True, random_state=42+fold),
                                    scoring='r2', n_jobs=1, refit=True, random_state=42+fold)
            gs.fit(Xtr, ytr)
            m = compute_metrics(yte, gs.best_estimator_.predict(Xte))
            per_fold.append(m); best_ps.append(gs.best_params_)
            print(f"    Fold {fold}: R2={m['r2']:.3f} MAE={m['mae']:.2f} "
                  f"[{gs.best_params_['feature_selection__method']}/"
                  f"{gs.best_params_['feature_selection__n_components']}]")

        agg = {}
        for key in per_fold[0]:
            vals = np.array([m[key] for m in per_fold])
            agg[f"{key}_mean"] = vals.mean(); agg[f"{key}_std"] = vals.std(ddof=1)
        methods = [p['feature_selection__method'] for p in best_ps]
        best_m = max(set(methods), key=methods.count)
        results.append({"audio_source": source_name, "quality_filter": quality_name,
                         "model": model_name, "best_feature_method": best_m,
                         "n_samples": len(y), "n_features_original": X.shape[1], **agg})
        print(f"    → {best_m}: R2={agg['r2_mean']:.3f}±{agg['r2_std']:.3f} MAE={agg['mae_mean']:.2f}")
    return results


def main():
    print("="*70)
    print("MAXIMAL AUDIO REGRESSION v2 — ALL SOURCES + ENHANCED FEATURE SELECTION")
    print("="*70)
    for name, path in AUDIO_SOURCES.items():
        print(f"  · {name}: {path}")
    print(f"Feature methods: {FEATURE_METHODS_V2}  |  Regressors: {len(GRIDS)}\n")

    all_results = []
    for source_name, source_path in AUDIO_SOURCES.items():
        if not source_path.exists():
            print(f"  ⚠️  Not found, skipping: {source_path}"); continue
        output_dir = OUTPUT_BASE / source_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for qual_name, qual_filter in QUALITY_FILTERS.items():
            results = run_experiment(source_name, qual_name, qual_filter, output_dir)
            if results:
                all_results.extend(results)
                df_r = pd.DataFrame(results)
                fp = output_dir / f"maximal_audio_regression_v2_{source_name}_{qual_name}.csv"
                df_r.to_csv(fp, index=False)
                print(f"  Saved → {fp}")

    if all_results:
        df = pd.DataFrame(all_results)
        out = OUTPUT_BASE / "maximal_audio_regression_v2_all.csv"
        df.to_csv(out, index=False)
        print(f"\n📊 TOP 15 BY R²:")
        print(df.sort_values('r2_mean', ascending=False)
              [['audio_source','model','best_feature_method','r2_mean','r2_std','mae_mean']]
              .head(15).to_string(index=False))
        print("\n🏆 BEST PER SOURCE:")
        for src in AUDIO_SOURCES:
            sub = df[df['audio_source'] == src]
            if len(sub) > 0:
                b = sub.loc[sub['r2_mean'].idxmax()]
                print(f"  {src:30s}: R²={b['r2_mean']:.3f}±{b['r2_std']:.3f} ({b['model']}, {b['best_feature_method']})")
        print(f"\n✓ Saved → {out}")
    print(f"\n✓ Completed {len(all_results)} experiments")


if __name__ == "__main__":
    main()
