"""
MAXIMAL Audio Classification v2 - All Audio Sources + Enhanced Feature Selection

v2 changes vs v1:
  - Loops over 4 ProposedDataset audio sources (eGeMAPS diar/non-diar + wav2vec diar/non-diar)
  - Adds 'selectk_pca_pruned' feature selection: SelectKBest → PCA → no redundancy pruning
    (audio-only, no text to prune against, so simply SelectKBest→PCA cascade)
  - Only 'no_filter' quality condition used (consistent with multimodal scripts)
  - Results saved per audio source, no overwrite of v1

Old script kept: maximal_audio_classification.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR   = Path("CONFIGURE_ME/repo")  # root of this repository
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")
FEATURES_BASE   = DATASET_ROOT / "features_output"

LABELS_CSV = Path("CONFIGURE_ME/labels/classi_labels.csv")

# v1 used single source (commented out):
# FEATURES_CSV = BASE_DIR / "proposed_dataset_wav2vec2_with_diarization_features.csv"

# v2: all four ProposedDataset audio feature sets
AUDIO_SOURCES = {
    "egemaps_diarised":     FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    "egemaps_non_diarised": FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    "wav2vec_diarised":     FEATURES_BASE / "proposed_dataset_german_xlsr_diarised/proposed_dataset_german_xlsr_diarised.csv",
    "wav2vec_non_diarised": FEATURES_BASE / "proposed_dataset_german_xlsr_non_diarised/proposed_dataset_german_xlsr_non_diarised.csv",
}

# v2 output: separate sub-dirs per source, results have _v2 suffix
OUTPUT_BASE = Path("CONFIGURE_ME/storage/results/audio_nested_cv")

# Only no_filter (consistent with multimodal pipeline in Multi/)
QUALITY_FILTERS = {
    "no_filter": None,
}

ID_COL, Y_COL = "patient_id", "depressed"
ORIGINAL_EXCLUDE = {177, 207, 299}

# ============================================================================
# GRIDS — v2 adds 'selectk_pca_pruned' to feature_method
# ============================================================================

# v1 grids used: ['pca', 'anova', 'mutual_info']
# v2 adds:       'selectk_pca_pruned'  (SelectKBest → PCA cascade)
FEATURE_METHODS_V2 = ['pca', 'anova', 'mutual_info', 'selectk_pca_pruned']

GRIDS = {
    "LogisticRegression": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'saga']
    },
    "SVC": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': [1, 0.1, 0.01, 0.001],
        'classifier__kernel': ['rbf', 'poly', 'sigmoid']
    },
    "RandomForest": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__n_estimators': [10, 50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    },
    "XGBoost": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    "AdaBoost": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1.0]
    },
    "DecisionTree": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__criterion': ['gini', 'entropy']
    },
    "KNN": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    },
    "ExtraTrees": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    "GradientBoosting": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0]
    },
    "MLP": {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate': ['constant', 'adaptive']
    },
}

if LIGHTGBM_AVAILABLE:
    GRIDS["LightGBM"] = {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7, -1],
        'classifier__num_leaves': [15, 31, 63],
        'classifier__min_child_samples': [10, 20, 30]
    }

if CATBOOST_AVAILABLE:
    GRIDS["CatBoost"] = {
        'feature_method': FEATURE_METHODS_V2,
        'n_components': [50, 100, 200],
        'classifier__iterations': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__depth': [3, 5, 7],
        'classifier__l2_leaf_reg': [1, 3, 5],
        'classifier__border_count': [32, 64, 128]
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_audio_source(source_name: str) -> pd.DataFrame:
    """Load audio features for a named source, normalise patient_id."""
    path = AUDIO_SOURCES[source_name]
    df = pd.read_csv(path)
    if 'file_id' in df.columns:
        df['patient_id'] = df['file_id'].astype(str).str.extract(r'(\d+)').astype(int)
        df = df.drop(columns=['file_id'])
    elif 'patient_id' not in df.columns:
        first_col = df.columns[0]
        df['patient_id'] = df[first_col].astype(int)
        df = df.drop(columns=[first_col])
    else:
        df['patient_id'] = df['patient_id'].astype(int)
    df = df[[c for c in df.columns if c != 'n_segments']]
    df = df[~df['patient_id'].isin(ORIGINAL_EXCLUDE)]
    return df


def load_data(source_name: str, quality_filter):
    feats = load_audio_source(source_name)
    labs = pd.read_csv(LABELS_CSV)
    labs['patient_id'] = labs['patient_id'].astype(int)

    exclude_ids = ORIGINAL_EXCLUDE.copy()
    if quality_filter == "inverse_good":
        for fname in ["nisqa_good_quality_ids.txt", "nisqa_very_good_quality_ids.txt"]:
            p = BASE_DIR / fname
            if p.exists():
                with open(p) as f:
                    exclude_ids.update(int(l.strip()) for l in f if l.strip())
    elif quality_filter is not None and quality_filter.exists():
        with open(quality_filter) as f:
            exclude_ids.update(int(l.strip()) for l in f if l.strip())

    feats = feats[~feats[ID_COL].isin(exclude_ids)]
    df = feats.merge(labs[[ID_COL, Y_COL]], on=ID_COL, how="inner")
    feature_cols = [col for col in df.columns if col not in [ID_COL, Y_COL]]
    X = np.nan_to_num(df[feature_cols].to_numpy(dtype=float))
    y = df[Y_COL].to_numpy(dtype=int)
    return X, y, len(exclude_ids)


class FlexibleFeatureSelectorV2:
    """
    v2: adds 'selectk_pca_pruned' method (SelectKBest → PCA cascade).
    For audio-only context there are no text features to prune against,
    so selectk_pca_pruned simply performs SelectKBest(presel_k) → PCA(n_components).
    """
    def __init__(self, method='anova', n_components=100, presel_k=200):
        self.method = method
        self.n_components = n_components
        self.presel_k = presel_k
        self.selector = None
        self.scaler = StandardScaler()
        # extra state for selectk_pca_pruned
        self._presel = None
        self._pca    = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        n = min(self.n_components, X.shape[1], len(y) - 1)
        if self.method == 'anova':
            self.selector = SelectKBest(f_classif, k=min(n, X.shape[1]))
            self.selector.fit(X_scaled, y)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=min(n, X.shape[1]))
            self.selector.fit(X_scaled, y)
        elif self.method == 'pca':
            self.selector = PCA(n_components=n)
            self.selector.fit(X_scaled)
        elif self.method == 'selectk_pca_pruned':
            # Step 1: SelectKBest pre-filter
            k_pre = min(self.presel_k, X_scaled.shape[1])
            self._presel = SelectKBest(f_classif, k=k_pre).fit(X_scaled, y)
            X_pre = self._presel.transform(X_scaled)
            # Step 2: PCA
            n_pca = min(n, X_pre.shape[0] - 1, X_pre.shape[1])
            self._pca = PCA(n_components=n_pca, random_state=42).fit(X_pre)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.method == 'selectk_pca_pruned':
            return self._pca.transform(self._presel.transform(X_scaled))
        return self.selector.transform(X_scaled)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {'method': self.method, 'n_components': self.n_components,
                'presel_k': self.presel_k}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def make_estimator(name):
    if name == "LogisticRegression":
        clf = LogisticRegression(max_iter=10000, random_state=42)
    elif name == "SVC":
        clf = SVC(probability=True, random_state=42)
    elif name == "RandomForest":
        clf = RandomForestClassifier(random_state=42)
    elif name == "XGBoost":
        # use_label_encoder removed (deprecated)
        clf = XGBClassifier(objective="binary:logistic", eval_metric='logloss', random_state=42)
    elif name == "AdaBoost":
        clf = AdaBoostClassifier(random_state=42)
    elif name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=42)
    elif name == "KNN":
        clf = KNeighborsClassifier()
    elif name == "ExtraTrees":
        clf = ExtraTreesClassifier(random_state=42)
    elif name == "GradientBoosting":
        clf = GradientBoostingClassifier(random_state=42)
    elif name == "MLP":
        clf = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
    elif name == "LightGBM" and LIGHTGBM_AVAILABLE:
        clf = LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
    elif name == "CatBoost" and CATBOOST_AVAILABLE:
        clf = CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    else:
        raise ValueError(f"Unknown or unavailable model: {name}")
    return Pipeline([
        ('feature_selection', FlexibleFeatureSelectorV2()),
        ('classifier', clf)
    ])


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores if scores.ndim == 1 else scores[:, 1]
    return model.predict(X).astype(float)


def compute_metrics(y_true, y_score, y_pred):
    return {
        "average_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def run_experiment(source_name: str, quality_name: str, quality_filter, output_dir: Path):
    print(f"\n{'='*70}")
    print(f"Experiment: {source_name} + {quality_name}")
    print(f"{'='*70}")

    X, y, n_excluded = load_data(source_name, quality_filter)
    print(f"Samples: {len(y)}, Features: {X.shape[1]}, Excluded: {n_excluded}")
    print(f"Class distribution: {np.bincount(y)}")

    if len(np.unique(y)) < 2 or len(y) < 50:
        print("⚠️  Insufficient data - skipping")
        return None

    # Correlation analysis (Spearman |r| per feature vs label)
    corr_path = output_dir / f"spearman_corr_{source_name}_{quality_name}.csv"
    if not corr_path.exists():
        corrs = [abs(scipy_stats.spearmanr(X[:, i], y)[0]) for i in range(min(X.shape[1], 500))]
        pd.DataFrame({'feature_idx': range(len(corrs)), 'spearman_abs_r': corrs}).to_csv(
            corr_path, index=False)
        print(f"  Correlation analysis saved → {corr_path}")

    results = []
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name in GRIDS.keys():
        print(f"\n  {model_name}:")
        model = make_estimator(model_name)
        per_fold_metrics = []
        best_params_per_fold = []

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + fold)

            param_grid = GRIDS[model_name].copy()
            param_grid['feature_selection__method'] = param_grid.pop('feature_method')
            param_grid['feature_selection__n_components'] = param_grid.pop('n_components')

            gs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=50,
                cv=inner_cv,
                scoring="f1",
                n_jobs=1,
                refit=True,
                random_state=42 + fold
            )
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
            y_score = get_scores(best_model, X_test)

            metrics = compute_metrics(y_test, y_score, y_pred)
            per_fold_metrics.append(metrics)
            best_params_per_fold.append(gs.best_params_)

            print(f"    Fold {fold}: F1={metrics['f1']:.3f}, "
                  f"Method={gs.best_params_['feature_selection__method']}, "
                  f"K={gs.best_params_['feature_selection__n_components']}")

        agg_metrics = {}
        for key in per_fold_metrics[0].keys():
            vals = np.array([m[key] for m in per_fold_metrics])
            agg_metrics[f"{key}_mean"] = vals.mean()
            agg_metrics[f"{key}_std"] = vals.std(ddof=1)

        methods = [p['feature_selection__method'] for p in best_params_per_fold]
        best_method = max(set(methods), key=methods.count)

        results.append({
            "audio_source":       source_name,
            "quality_filter":     quality_name,
            "model":              model_name,
            "best_feature_method": best_method,
            "n_samples":          len(y),
            "n_features_original": X.shape[1],
            **agg_metrics
        })

        print(f"    → Best: {best_method}, F1={agg_metrics['f1_mean']:.3f}±{agg_metrics['f1_std']:.3f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("MAXIMAL AUDIO CLASSIFICATION v2 - ALL SOURCES + ENHANCED FEATURE SELECTION")
    print("="*70)
    print(f"\nAudio sources ({len(AUDIO_SOURCES)}):")
    for name, path in AUDIO_SOURCES.items():
        print(f"  · {name}: {path}")
    print(f"\nFeature selection methods: {FEATURE_METHODS_V2}")
    print(f"Classifiers: {len(GRIDS)}")
    print()

    all_results = []

    for source_name, source_path in AUDIO_SOURCES.items():
        if not source_path.exists():
            print(f"  ⚠️  Source not found, skipping: {source_path}")
            continue

        output_dir = OUTPUT_BASE / source_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for qual_name, qual_filter in QUALITY_FILTERS.items():
            results = run_experiment(source_name, qual_name, qual_filter, output_dir)
            if results:
                all_results.extend(results)
                # Save intermediate per-source results
                src_df = pd.DataFrame(results)
                src_file = output_dir / f"maximal_audio_classification_v2_{source_name}_{qual_name}.csv"
                src_df.to_csv(src_file, index=False)
                print(f"  Saved → {src_file}")

    if all_results:
        df = pd.DataFrame(all_results)
        output_file = OUTPUT_BASE / "maximal_audio_classification_v2_all.csv"
        df.to_csv(output_file, index=False)

        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)

        print("\n📊 TOP 15 BY F1 (across all sources):")
        df_sorted = df.sort_values('f1_mean', ascending=False)
        print(df_sorted[['audio_source', 'model', 'best_feature_method',
                          'f1_mean', 'f1_std', 'roc_auc_mean']].head(15).to_string(index=False))

        print("\n🏆 BEST PER AUDIO SOURCE:")
        for src in AUDIO_SOURCES:
            sub = df[df['audio_source'] == src]
            if len(sub) > 0:
                best = sub.loc[sub['f1_mean'].idxmax()]
                print(f"  {src:30s}: F1={best['f1_mean']:.3f}±{best['f1_std']:.3f} "
                      f"({best['model']}, {best['best_feature_method']})")

        print(f"\n✓ Combined results saved to: {output_file}")

    print(f"\n✓ Completed {len(all_results)} experiments")


if __name__ == "__main__":
    main()
