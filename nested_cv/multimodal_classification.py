"""
MULTIMODAL Classification with CV-Aware DepRoBERTa

Combines:
  - AUDIO: wav2vec2 aggregated features (6144-dim)
           Reduced via AudioFeatureSelector (PCA / ANOVA / mutual_info)
           n_components in [50, 100, 200] — jointly optimised by inner CV
  - TEXT:  3 DepRoBERTa probs (fine-tuned per fold) + 11 question features = 14 text features

Key design (no data leakage):
  For each outer CV fold:
    1. Fine-tune DepRoBERTa on TRAIN patients only
    2. Extract 3 probability features for ALL patients using fold-specific model
    3. Combine with 11 static question features → 14-dim text array
    4. Inside inner RandomizedSearchCV:
       a. AudioFeatureSelector(method, n_components) fitted on TRAIN audio
       b. Reduced audio concat with 14 text features
       c. Classifier fitted on combined features
    5. Evaluate best pipeline on TEST fold

Total: 3 Llama models × 12 classifiers × 5 folds
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedTokenizer)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)

# Optional imports
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("⚠️  LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("⚠️  CatBoost not available")


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("CONFIGURE_ME/repo")
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")
FEATURES_BASE   = DATASET_ROOT / "features_output"

# ---- v1 (single audio source, non-diarised wav2vec) — kept for reference ----
# AUDIO_CSV = FEATURES_BASE / "proposed_dataset_german_xlsr_non_diarised/proposed_dataset_german_xlsr_non_diarised.csv"

# ---- v2: all four ProposedDataset audio feature sets --------------------------------
AUDIO_SOURCES = {
    "egemaps_diarised":     FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    "egemaps_non_diarised": FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    "wav2vec_diarised":     FEATURES_BASE / "proposed_dataset_german_xlsr_diarised/proposed_dataset_german_xlsr_diarised.csv",
    "wav2vec_non_diarised": FEATURES_BASE / "proposed_dataset_german_xlsr_non_diarised/proposed_dataset_german_xlsr_non_diarised.csv",
}
# High-dim flag per source (True = wav2vec-scale, will try full PCA range)
AUDIO_HIGH_DIM = {
    "egemaps_diarised":     False,
    "egemaps_non_diarised": False,
    "wav2vec_diarised":     True,
    "wav2vec_non_diarised": True,
}

# PHQ-9 scores for DepRoBERTa fine-tuning (3-class: severe/moderate/not-depressed)
PHQ_LABELS_CSV = Path("CONFIGURE_ME/labels/regress_labels.csv")

# Binary depression labels for final classification
BINARY_LABELS_CSV = Path("CONFIGURE_ME/labels/classi_labels.csv")

# Output directory for classification results — v2 suffix, no overwrite of v1
OUTPUT_DIR = BASE_DIR / "Multi/multimodal_cv_deproberta_results_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CONFIGURE_ME: STORAGE_ROOT should be on a large filesystem
WOODY_BASE = Path("CONFIGURE_ME/storage")

# Fine-tuned DepRoBERTa models (namespaced under 'multimodal' to avoid collision
# with text-only script models)
MODEL_SAVE_DIR = WOODY_BASE / "deproberta_cv_models_multimodal"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Extracted DepRoBERTa probability features (per fold)
FEATURES_SAVE_DIR = WOODY_BASE / "deproberta_extracted_features_multimodal"
FEATURES_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# LLM summary directories (input text for DepRoBERTa fine-tuning)
LLM_SUMMARY_DIRS = {
    "llama_3.1_8B":  BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3108",
    "llama_3.1_70B": BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3170",
    "llama_3.3_70B": BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3370",
}

# DepRoBERTa configuration
CACHE_DIR = Path("CONFIGURE_ME/storage/models/hf_cache")
BASE_DEPROBERTA_MODEL = "rafalposwiata/deproberta-large-v1"
DEPROBERTA_EPOCHS = 10
DEPROBERTA_BATCH_SIZE = 32
DEPROBERTA_LR_CLASSIFIER = 5e-6
DEPROBERTA_LR_ENCODER = 1e-5
DEPROBERTA_MAX_LENGTH = 512

ID_COL, Y_COL = "patient_id", "depressed"
ORIGINAL_EXCLUDE = {177, 207, 299}

# 11 question feature files (q1-q11, already leakage-free as they are
# per-patient static scores not derived from the training split)
QUESTION_FEATURE_FILES = {
    "llama_3.1_8B":  Path("CONFIGURE_ME/features/text_features_llama3.1_8B.csv"),
    "llama_3.1_70B": Path("CONFIGURE_ME/features/text_features_llama3.1_70B.csv"),
    "llama_3.3_70B": Path("CONFIGURE_ME/features/text_features_llama3.3_70B.csv"),
}

# No separate outer loop over audio methods — these are tuned inside inner CV
# via AudioFeatureSelector params in GRIDS.
#
# v1 (kept for reference):
# AUDIO_SELECTION_PARAMS = {
#     'preprocessor__audio__method':       ['pca', 'anova', 'mutual_info'],
#     'preprocessor__audio__n_components': [50, 100, 200],
# }
#
# v2: adds supervised SelectKBest → PCA → text-redundancy-pruning method:
AUDIO_SELECTION_PARAMS = {
    'preprocessor__audio__method':        ['pca', 'anova', 'mutual_info', 'selectk_pca_pruned'],
    'preprocessor__audio__n_components':  [50, 100, 200],
    # used only by selectk_pca_pruned — pre-filter k before PCA
    'preprocessor__audio__presel_k':      [200, 500],
    # pearson |r| threshold for audio-text redundancy pruning
    'preprocessor__audio__redund_thresh': [0.70],
}

# ============================================================================
# CLASSIFIER GRIDS
# ============================================================================

# Each grid includes audio feature selection params (passed to AudioFeatureSelector
# inside the pipeline) as well as the classifier's own hyperparams.
# The audio_* params are prefixed with 'audio_selector__' in run_experiment.

CLF_GRIDS = {
    "LogisticRegression": {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__solver': ['liblinear', 'saga']
    },
    "SVC": {
        'clf__C': [0.1, 1, 10, 100],
        'clf__gamma': [1, 0.1, 0.01, 0.001],
        'clf__kernel': ['rbf', 'poly', 'sigmoid']
    },
    "RandomForest": {
        'clf__n_estimators': [10, 50, 100, 200],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10]
    },
    "XGBoost": {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [3, 5, 7]
    },
    "AdaBoost": {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 1.0]
    },
    "DecisionTree": {
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__criterion': ['gini', 'entropy']
    },
    "KNN": {
        'clf__n_neighbors': [3, 5, 7, 9],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan', 'minkowski']
    },
    "ExtraTrees": {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    },
    "GradientBoosting": {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.8, 1.0]
    },
    "MLP": {
        'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'clf__activation': ['relu', 'tanh'],
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__learning_rate': ['constant', 'adaptive']
    },
}

if LIGHTGBM_AVAILABLE:
    CLF_GRIDS["LightGBM"] = {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 5, 7, -1],
        'clf__num_leaves': [15, 31, 63],
        'clf__min_child_samples': [10, 20, 30]
    }

if CATBOOST_AVAILABLE:
    CLF_GRIDS["CatBoost"] = {
        'clf__iterations': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__depth': [3, 5, 7],
        'clf__l2_leaf_reg': [1, 3, 5],
        'clf__border_count': [32, 64, 128]
    }


# ============================================================================
# UTILITIES
# ============================================================================


class AudioFeatureSelector:
    """
    Sklearn-compatible audio feature selector. v2 adds 'selectk_pca_pruned' method.

    Methods:
      'pca'               - PCA (v1)
      'anova'             - SelectKBest(f_classif) (v1)
      'mutual_info'       - SelectKBest(mutual_info_classif) (v1)
      'selectk_pca_pruned'- NEW (v2): SelectKBest(k=presel_k) → PCA(n_components)
                            → drop components with Pearson |r|>redund_thresh vs text.
                            Requires X_text_train to be stored before transform.

    Fitted only on training data (leakage-free).
    """

    def __init__(self, method: str = 'pca', n_components: int = 100,
                 presel_k: int = 200, redund_thresh: float = 0.70):
        self.method = method
        self.n_components = n_components
        self.presel_k = presel_k
        self.redund_thresh = redund_thresh
        self.scaler = StandardScaler()
        self.selector = None
        # v2 extra state for selectk_pca_pruned
        self._presel = None
        self._pca = None
        self._keep_mask = None  # pruning mask, set during fit
        self._X_text_train = None  # text features stored during fit for pruning

    def fit(self, X: np.ndarray, y: np.ndarray = None, X_text_train: np.ndarray = None):
        X_scaled = self.scaler.fit_transform(X)
        n = min(self.n_components, X.shape[1], X.shape[0] - 1)
        if self.method == 'pca':
            self.selector = PCA(n_components=n)
            self.selector.fit(X_scaled)
        elif self.method == 'anova':
            self.selector = SelectKBest(f_classif, k=n)
            self.selector.fit(X_scaled, y)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=n)
            self.selector.fit(X_scaled, y)
        elif self.method == 'selectk_pca_pruned':
            # Step 1: SelectKBest pre-filter
            k_pre = min(self.presel_k, X_scaled.shape[1])
            self._presel = SelectKBest(f_classif, k=k_pre).fit(X_scaled, y)
            X_pre = self._presel.transform(X_scaled)
            # Step 2: PCA
            n_pca = min(n, X_pre.shape[0] - 1, X_pre.shape[1])
            self._pca = PCA(n_components=n_pca, random_state=42).fit(X_pre)
            X_pca = self._pca.transform(X_pre)
            # Step 3: Redundancy pruning vs text (only if text provided)
            if X_text_train is not None and X_text_train.shape[1] > 0:
                from scipy import stats as _stats
                keep = np.ones(X_pca.shape[1], dtype=bool)
                for ai in range(X_pca.shape[1]):
                    for ti in range(X_text_train.shape[1]):
                        r, _ = _stats.pearsonr(X_pca[:, ai], X_text_train[:, ti])
                        if abs(r) > self.redund_thresh:
                            keep[ai] = False; break
                n_pruned = int((~keep).sum())
                logging.info(f"  [selectk_pca_pruned] Pruned {n_pruned}/{X_pca.shape[1]} "
                             f"audio PCA dims (|r|>{self.redund_thresh})")
                self._keep_mask = keep
            else:
                self._keep_mask = np.ones(X_pca.shape[1], dtype=bool)
        else:
            raise ValueError(f"Unknown audio method: {self.method}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if self.method == 'selectk_pca_pruned':
            X_pre = self._presel.transform(X_scaled)
            X_pca = self._pca.transform(X_pre)
            return X_pca[:, self._keep_mask]
        return self.selector.transform(X_scaled)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None,
                      X_text_train: np.ndarray = None) -> np.ndarray:
        return self.fit(X, y, X_text_train=X_text_train).transform(X)

    def get_params(self, deep=True):
        return {'method': self.method, 'n_components': self.n_components,
                'presel_k': self.presel_k, 'redund_thresh': self.redund_thresh}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def make_estimator(name):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=10000, random_state=42)
    elif name == "SVC":
        return SVC(probability=True, random_state=42)
    elif name == "RandomForest":
        # return RandomForestClassifier(random_state=42)  # v1
        return RandomForestClassifier(random_state=42, n_jobs=1)  # v2: n_jobs=1 avoids OpenMP segfault on GPU nodes
    elif name == "XGBoost":
        # NOTE: use_label_encoder removed (deprecated in newer XGBoost)
        # return XGBClassifier(objective="binary:logistic", eval_metric='logloss', random_state=42)  # v1
        # v2: nthread=1 + device=cpu fixes 'super has no __sklearn_tags__' on sklearn>=1.6
        return XGBClassifier(objective="binary:logistic", eval_metric='logloss',
                             random_state=42, nthread=1, device="cpu")
    elif name == "AdaBoost":
        return AdaBoostClassifier(random_state=42)
    elif name == "DecisionTree":
        return DecisionTreeClassifier(random_state=42)
    elif name == "KNN":
        return KNeighborsClassifier()
    elif name == "ExtraTrees":
        # return ExtraTreesClassifier(random_state=42)  # v1
        return ExtraTreesClassifier(random_state=42, n_jobs=1)  # v2: n_jobs=1
    elif name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42)  # single-threaded by default
    elif name == "MLP":
        return MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
    elif name == "LightGBM" and LIGHTGBM_AVAILABLE:
        # return LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)  # v1
        return LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, device="cpu")  # v2: force CPU, single thread
    elif name == "CatBoost" and CATBOOST_AVAILABLE:
        return CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    else:
        raise ValueError(f"Unknown or unavailable model: {name}")


# ============================================================================
# AUDIO DATA LOADING
# ============================================================================

# ---- v1 load_audio_features (single source) ---- kept for reference ----
# def load_audio_features() -> pd.DataFrame:
#     audio_df = pd.read_csv(AUDIO_CSV)
#     if 'file_id' in audio_df.columns:
#         audio_df['patient_id'] = audio_df['file_id'].str.split('_').str[0].astype(int)
#         audio_df = audio_df.drop(columns=['file_id'])
#     elif 'patient_id' not in audio_df.columns:
#         first_col = audio_df.columns[0]
#         audio_df['patient_id'] = audio_df[first_col].astype(int)
#         audio_df = audio_df.drop(columns=[first_col])
#     else:
#         audio_df['patient_id'] = audio_df['patient_id'].astype(int)
#     audio_df = audio_df[~audio_df['patient_id'].isin(ORIGINAL_EXCLUDE)]
#     return audio_df

def load_audio_features(source_name: str) -> pd.DataFrame:
    """Load audio features for a given source key from AUDIO_SOURCES. v2."""
    path = AUDIO_SOURCES[source_name]
    audio_df = pd.read_csv(path)
    # Normalise ID column
    if 'file_id' in audio_df.columns:
        audio_df['patient_id'] = audio_df['file_id'].astype(str).str.extract(r'(\d+)').astype(int)
        audio_df = audio_df.drop(columns=['file_id'])
    elif 'patient_id' not in audio_df.columns:
        first_col = audio_df.columns[0]
        audio_df['patient_id'] = audio_df[first_col].astype(int)
        audio_df = audio_df.drop(columns=[first_col])
    else:
        audio_df['patient_id'] = audio_df['patient_id'].astype(int)
    meta = ['patient_id', 'n_segments']
    audio_df = audio_df[[c for c in audio_df.columns if c not in ['n_segments']] ]
    audio_df = audio_df[~audio_df['patient_id'].isin(ORIGINAL_EXCLUDE)]
    return audio_df


def load_question_features(llama_model: str) -> pd.DataFrame:
    """Load 11 static question features (q1-q11) — already leakage-free."""
    feature_file = QUESTION_FEATURE_FILES[llama_model]
    df = pd.read_csv(feature_file)
    df['patient_id'] = df['patient_id'].astype(int)
    q_cols = [f'q{i}' for i in range(1, 12)]
    return df[['patient_id'] + q_cols]


def combine_text_features(
    prob_df: pd.DataFrame, question_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge 3 DepRoBERTa probability features + 11 question features = 14 text features."""
    combined = prob_df.merge(question_df, on='patient_id', how='inner')
    # Ensure fixed column order
    cols = (
        ['patient_id', 'prob_severe', 'prob_moderate', 'prob_not_depression']
        + [f'q{i}' for i in range(1, 12)]
    )
    return combined[cols]


# ============================================================================
# FUSION TRANSFORMERS  (unchanged from enhanced_multimodal_classification.py)
# ============================================================================

class PCAFusionTransformer:
    """PCA on audio features + concatenate with text probability features."""

    def __init__(self, target_components=100):
        self.target_components = target_components
        self.audio_pca = None
        self.scaler = StandardScaler()
        self.n_audio_selected = None
        self.n_text_features = None

    def fit(self, X_audio, X_text, y=None):
        n_samples = X_audio.shape[0]
        n_features = X_audio.shape[1]
        self.n_audio_selected = min(self.target_components, n_features, n_samples - 1)
        self.n_text_features = X_text.shape[1]

        self.audio_pca = PCA(n_components=self.n_audio_selected)
        X_audio_reduced = self.audio_pca.fit_transform(X_audio)

        X_fused = np.hstack([X_audio_reduced, X_text])
        self.scaler.fit(X_fused)
        return self

    def transform(self, X_audio, X_text):
        X_audio_reduced = self.audio_pca.transform(X_audio)
        X_fused = np.hstack([X_audio_reduced, X_text])
        return self.scaler.transform(X_fused)

    def fit_transform(self, X_audio, X_text, y=None):
        self.fit(X_audio, X_text, y)
        return self.transform(X_audio, X_text)

    def get_feature_info(self):
        return {
            'method': 'PCA',
            'n_audio_selected': self.n_audio_selected,
            'n_text_selected': self.n_text_features,
            'total_features': self.n_audio_selected + self.n_text_features,
            'audio_variance_explained': (
                self.audio_pca.explained_variance_ratio_.sum() if self.audio_pca else 0
            )
        }


class RFEFusionTransformer:
    """RFE across concatenated audio + text features."""

    def __init__(self, target_features=100):
        self.target_features = target_features
        self.scaler = StandardScaler()
        self.rfe = None
        self.selected_indices = None
        self.n_audio_features = None
        self.n_text_features = None

    def fit(self, X_audio, X_text, y):
        self.n_audio_features = X_audio.shape[1]
        self.n_text_features = X_text.shape[1]

        X_concat = np.hstack([X_audio, X_text])
        X_scaled = self.scaler.fit_transform(X_concat)

        n_total_features = X_concat.shape[1]
        n_samples = X_concat.shape[0]
        n_select = min(self.target_features, n_total_features, n_samples - 1)

        estimator = LogisticRegression(max_iter=1000, random_state=42)
        self.rfe = RFE(estimator=estimator, n_features_to_select=n_select, step=10)
        self.rfe.fit(X_scaled, y)
        self.selected_indices = np.where(self.rfe.support_)[0]
        return self

    def transform(self, X_audio, X_text):
        X_concat = np.hstack([X_audio, X_text])
        X_scaled = self.scaler.transform(X_concat)
        return X_scaled[:, self.selected_indices]

    def fit_transform(self, X_audio, X_text, y):
        self.fit(X_audio, X_text, y)
        return self.transform(X_audio, X_text)

    def get_feature_info(self):
        n_audio_selected = int(np.sum(self.selected_indices < self.n_audio_features))
        n_text_selected = int(np.sum(self.selected_indices >= self.n_audio_features))
        return {
            'method': 'RFE',
            'n_audio_selected': n_audio_selected,
            'n_text_selected': n_text_selected,
            'total_features': len(self.selected_indices),
            'audio_selection_rate': (
                n_audio_selected / self.n_audio_features if self.n_audio_features > 0 else 0
            ),
            'text_selection_rate': (
                n_text_selected / self.n_text_features if self.n_text_features > 0 else 0
            )
        }


# ============================================================================
# DEPROBERTA FINE-TUNING  (ported verbatim from text script)
# ============================================================================

@dataclass
class DepRoBERTaConfig:
    """Configuration for DepRoBERTa fine-tuning."""
    base_model_path: str = BASE_DEPROBERTA_MODEL
    max_token_length: int = DEPROBERTA_MAX_LENGTH
    num_labels: int = 3
    label_map: Dict[int, str] = field(default_factory=lambda: {
        0: "Severe", 1: "Moderate", 2: "Not Depressed"
    })
    batch_size: int = DEPROBERTA_BATCH_SIZE
    epochs: int = DEPROBERTA_EPOCHS
    lr_classifier: float = DEPROBERTA_LR_CLASSIFIER
    lr_encoder: float = DEPROBERTA_LR_ENCODER
    last_n_layers_to_unfreeze: int = 6
    early_stopping_patience: int = 10
    random_seed: int = 42
    validation_size: float = 0.15


class ClinicalTextDataset(Dataset):
    """PyTorch Dataset for clinical text."""
    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer: PreTrainedTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer.encode_plus(
            self.texts[idx], add_special_tokens=True, max_length=self.max_length,
            padding="max_length", truncation=True, return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_summaries_for_patients(llama_model: str,
                                patient_ids: List[int]) -> Dict[int, str]:
    """Load LLM summaries for given patient IDs."""
    summary_dir = LLM_SUMMARY_DIRS[llama_model]
    summaries = {}

    for pid in patient_ids:
        possible_files = [
            summary_dir / f"{pid:03d}_transcription.json",
            summary_dir / f"{pid:03d}_whisper.json",
        ]
        for json_file in possible_files:
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        summary = data.get("summary", "").strip()
                        if summary:
                            summaries[pid] = summary
                            break
                except Exception as e:
                    logging.warning(f"Error loading {json_file}: {e}")

    return summaries


def map_phq_to_label(score: int) -> int:
    """Map PHQ-8/9 score to 3-class categorical label."""
    if score >= 14:
        return 0  # Severe
    if score >= 7:
        return 1  # Moderate
    return 2      # Not Depressed / Mild


def fine_tune_deproberta_for_fold(
    llama_model: str,
    fold: int,
    train_patient_ids: List[int],
    save_dir: Path,
    config: DepRoBERTaConfig
) -> Path:
    """Fine-tune DepRoBERTa on training data for a specific fold."""

    logging.info(f"\n{'='*70}")
    logging.info(f"Fine-tuning DepRoBERTa: {llama_model}, Fold {fold}")
    logging.info(f"{'='*70}")

    # PHQ-9 labels for fine-tuning (3-class)
    labels_df = pd.read_csv(PHQ_LABELS_CSV)
    phq_col = 'PHQ9-Score'
    labels_df[phq_col] = pd.to_numeric(labels_df[phq_col], errors='coerce')
    labels_df.dropna(subset=[phq_col], inplace=True)
    labels_df['label'] = labels_df[phq_col].apply(map_phq_to_label)

    # Restrict to training patients for this fold
    train_df = labels_df[labels_df['patient_id'].isin(train_patient_ids)].copy()

    logging.info(f"Loading summaries for {len(train_df)} training patients...")
    summaries = load_summaries_for_patients(llama_model, train_df['patient_id'].tolist())

    train_df['text'] = train_df['patient_id'].map(summaries)
    train_df = train_df.dropna(subset=['text'])

    logging.info(f"Training samples with summary: {len(train_df)}")
    logging.info(f"Class distribution: {train_df['label'].value_counts().to_dict()}")

    if len(train_df) < 10:
        raise ValueError("Insufficient training data for DepRoBERTa fine-tuning")

    # Train / validation split (from training fold only)
    train_subset, val_subset = train_test_split(
        train_df, test_size=config.validation_size,
        stratify=train_df['label'], random_state=config.random_seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path, cache_dir=str(CACHE_DIR)
    )

    train_dataset = ClinicalTextDataset(
        train_subset['text'].tolist(), train_subset['label'].tolist(),
        tokenizer, config.max_token_length
    )
    val_dataset = ClinicalTextDataset(
        val_subset['text'].tolist(), val_subset['label'].tolist(),
        tokenizer, config.max_token_length
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_path, num_labels=config.num_labels, cache_dir=str(CACHE_DIR)
    )
    model.to(device)

    # Freeze all; unfreeze classifier + last N encoder layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    num_layers = model.roberta.config.num_hidden_layers
    layers_to_unfreeze = range(num_layers - config.last_n_layers_to_unfreeze, num_layers)
    for name, param in model.named_parameters():
        if any(f"encoder.layer.{i}." in name for i in layers_to_unfreeze):
            param.requires_grad = True

    optimizer = AdamW([
        {"params": model.classifier.parameters(),
         "lr": config.lr_classifier},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "encoder.layer" in n],
         "lr": config.lr_encoder},
    ])

    # Training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                total_val_loss += outputs.loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss   = total_val_loss   / len(val_loader)
        logging.info(
            f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logging.info(f"✓ Best model saved to {save_dir}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    logging.info(f"Fine-tuning complete. Best val loss: {best_val_loss:.4f}")
    return save_dir


def extract_deproberta_probabilities(
    llama_model: str,
    fold: int,
    patient_ids: List[int],
    model_dir: Path
) -> pd.DataFrame:
    """Extract 3 probability features for all patients using a fold-specific model."""

    logging.info(f"\nExtracting DepRoBERTa probs: {llama_model}, Fold {fold}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    summaries = load_summaries_for_patients(llama_model, patient_ids)

    results = []
    for pid in tqdm(patient_ids, desc="Extracting probabilities"):
        summary = summaries.get(pid)
        if summary is None:
            # No summary available → zero features
            results.append({
                'patient_id': pid,
                'prob_severe': 0.0,
                'prob_moderate': 0.0,
                'prob_not_depression': 0.0,
            })
            continue

        inputs = tokenizer(
            summary,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=DEPROBERTA_MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        results.append({
            'patient_id': pid,
            'prob_severe': float(probs[0]),
            'prob_moderate': float(probs[1]),
            'prob_not_depression': float(probs[2]),
        })

    # Free GPU memory after extraction
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ============================================================================
# MULTIMODAL PIPELINE
# ============================================================================

def make_multimodal_pipeline(audio_selector_params, classifier, n_audio_features):
    """
    Creates a standard sklearn Pipeline that:
      1. Applies AudioFeatureSelector to the first `n_audio_features` columns.
      2. Passes the remaining 14 text columns through untouched.
      3. Classifies with the given classifier.
    """
    from sklearn.compose import ColumnTransformer
    
    # Extract audio component parameters from the flattened param dict
    method = audio_selector_params.get('feature_selection__method', 'pca')
    n_components = audio_selector_params.get('feature_selection__n_components', 100)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('audio', AudioFeatureSelector(method=method, n_components=n_components), slice(0, n_audio_features)),
            ('text', 'passthrough', slice(n_audio_features, None))
        ]
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(llama_model: str) -> List[Dict]:
    """
    v2: Full CV experiment for one llama_model, iterating over all audio sources.

    For each of 5 outer folds × 4 audio sources:
      1. Fine-tune DepRoBERTa on train patient IDs (done once per fold, shared)
      2. Extract 3 DepRoBERTa probs for ALL patients
      3. Combine with 11 question features → 14-dim text array
      4. Inner RandomizedSearchCV (n_iter=50) jointly optimises:
         - AudioFeatureSelector(method, n_components)  [includes selectk_pca_pruned]
         - Classifier hyperparams
      5. Evaluate best pipeline on test fold
    """

    print(f"\n{'='*70}")
    print(f"Experiment: {llama_model}")
    print(f"{'='*70}")

    # ----- Load labels -----
    labels_df = pd.read_csv(BINARY_LABELS_CSV)
    labels_df['patient_id'] = labels_df['patient_id'].astype(int)
    labels_df = labels_df[~labels_df['patient_id'].isin(ORIGINAL_EXCLUDE)]

    # ----- Load 11 question features (static, leakage-free) -----
    question_df = load_question_features(llama_model)
    question_df = question_df[~question_df['patient_id'].isin(ORIGINAL_EXCLUDE)]

    # ----- CV setup -----
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    deproberta_config = DepRoBERTaConfig()
    results = []

    # We iterate over all audio sources (v2 — was single source in v1)
    for audio_source_name in AUDIO_SOURCES:
        if not AUDIO_SOURCES[audio_source_name].exists():
            logging.warning(f"Audio source not found, skipping: {AUDIO_SOURCES[audio_source_name]}")
            continue

        print(f"\n{'='*60}")
        print(f"Audio source: {audio_source_name}")
        print(f"{'='*60}")

        # Load audio for this source
        audio_df = load_audio_features(audio_source_name)
        merged = labels_df.merge(audio_df, on='patient_id', how='inner')
        all_patient_ids = merged['patient_id'].values
        y_all            = merged[Y_COL].values
        audio_feature_cols = [c for c in audio_df.columns if c != 'patient_id']
        X_audio_all = np.nan_to_num(merged[audio_feature_cols].to_numpy(dtype=float))

        print(f"Total patients (labels ∩ audio): {len(all_patient_ids)}")
        print(f"Audio feature dims: {X_audio_all.shape[1]}")
        print(f"Class distribution: {np.bincount(y_all)}")

        if len(np.unique(y_all)) < 2 or len(y_all) < 50:
            print("⚠️  Insufficient data — skipping this audio source")
            continue

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(all_patient_ids, y_all)):
            print(f"\n  --- Fold {fold} | {audio_source_name} ---")
            train_pids = all_patient_ids[train_idx].tolist()
            print(f"    Train: {len(train_pids)}  |  Test: {len(test_idx)}")

            # STEP 1: Fine-tune DepRoBERTa on this fold's training patients
            model_save_path = MODEL_SAVE_DIR / llama_model / f"fold_{fold}"
            if not model_save_path.exists():
                try:
                    fine_tune_deproberta_for_fold(
                        llama_model=llama_model,
                        fold=fold,
                        train_patient_ids=train_pids,
                        save_dir=model_save_path,
                        config=deproberta_config,
                    )
                except Exception as e:
                    logging.error(f"Fine-tuning failed for fold {fold}: {e}")
                    continue
            else:
                logging.info(f"Reusing existing model: {model_save_path}")

            # STEP 2: Extract 3 DepRoBERTa probability features for ALL patients
            prob_df = extract_deproberta_probabilities(
                llama_model=llama_model,
                fold=fold,
                patient_ids=all_patient_ids.tolist(),
                model_dir=model_save_path,
            )
            # Save for later analysis
            feat_out_dir = FEATURES_SAVE_DIR / llama_model
            feat_out_dir.mkdir(parents=True, exist_ok=True)
            prob_df.to_csv(feat_out_dir / f"fold_{fold}_probabilities.csv", index=False)

            # STEP 3: Combine 3 probs + 11 question features = 14 text features
            combined_text_df = combine_text_features(prob_df, question_df)
            all_text_feature_cols = (
                ['prob_severe', 'prob_moderate', 'prob_not_depression']
                + [f'q{i}' for i in range(1, 12)]
            )
            combined_indexed = combined_text_df.set_index('patient_id')
            X_text_all = np.zeros((len(all_patient_ids), 14), dtype=float)
            for i, pid in enumerate(all_patient_ids):
                if pid in combined_indexed.index:
                    X_text_all[i] = combined_indexed.loc[pid, all_text_feature_cols].values

            # STEP 4: Train/test split using concatenated arrays
            # Text features go after audio so ColumnTransformer slices work
            X_concat_all = np.hstack([X_audio_all, X_text_all])
            X_train  = X_concat_all[train_idx]
            X_test   = X_concat_all[test_idx]
            y_train = y_all[train_idx]
            y_test  = y_all[test_idx]

            # Text train subset (for redundancy pruning in selectk_pca_pruned)
            X_text_train = X_text_all[train_idx]
            n_audio_feats = X_audio_all.shape[1]

            # STEP 5: Per-classifier inner RandomizedSearchCV
            for clf_name in CLF_GRIDS.keys():
                try:
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True,
                                               random_state=42 + fold)

                    mapped_param_dist = {**AUDIO_SELECTION_PARAMS}
                    for k, v in CLF_GRIDS[clf_name].items():
                        mapped_k = k.replace('clf__', 'classifier__') if k.startswith('clf__') else k
                        mapped_param_dist[mapped_k] = v

                    clf_est = make_estimator(clf_name)

                    # Build AudioFeatureSelector with v2 constructor signature
                    audio_sel = AudioFeatureSelector(method='pca', n_components=100,
                                                     presel_k=200, redund_thresh=0.70)
                    # For selectk_pca_pruned, the selector needs X_text_train at fit time.
                    # We pass it via a custom fit_params dict in GridSearchCV;
                    # however, ColumnTransformer doesn't forward fit_params to sub-transformers.
                    # Workaround: store text train on the selector before fitting.
                    audio_sel._X_text_train = X_text_train

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('audio', audio_sel, slice(0, n_audio_feats)),
                            ('text', 'passthrough', slice(n_audio_feats, X_concat_all.shape[1]))
                        ]
                    )
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', clf_est)
                    ])

                    rs = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=mapped_param_dist,
                        n_iter=50,
                        cv=inner_cv,
                        scoring="f1",
                        n_jobs=1,
                        refit=True,
                        random_state=42 + fold,
                    )
                    rs.fit(X_train, y_train)

                    best_pipeline = rs.best_estimator_
                    y_pred  = best_pipeline.predict(X_test)

                    if hasattr(best_pipeline, 'predict_proba'):
                        y_score = best_pipeline.predict_proba(X_test)
                        if y_score.ndim == 2:
                            y_score = y_score[:, 1]
                    else:
                        scores = best_pipeline.decision_function(X_test)
                        y_score = scores if scores.ndim == 1 else scores[:, 1]

                    metrics = compute_metrics(y_test, y_score, y_pred)

                    best_audio_method = rs.best_params_.get('preprocessor__audio__method', '?')
                    best_n_components = rs.best_params_.get('preprocessor__audio__n_components', '?')

                    results.append({
                        "llama_model":       llama_model,
                        "audio_source":      audio_source_name,   # v2: tracks which audio set
                        "classifier":        clf_name,
                        "fold":              fold,
                        "n_train":           len(y_train),
                        "n_test":            len(y_test),
                        "best_audio_method": best_audio_method,
                        "best_n_components": best_n_components,
                        "n_text_features":   14,
                        **metrics,
                    })

                    print(f"    {clf_name:<25s} "
                          f"F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}  "
                          f"[{best_audio_method}/{best_n_components}]")

                except Exception as e:
                    logging.error(f"Classifier {clf_name} failed on fold {fold} "
                                  f"[{audio_source_name}]: {e}")
                    continue

    return results



    # ----- Load labels -----
    labels_df = pd.read_csv(BINARY_LABELS_CSV)
    labels_df['patient_id'] = labels_df['patient_id'].astype(int)
    labels_df = labels_df[~labels_df['patient_id'].isin(ORIGINAL_EXCLUDE)]

    # ----- Load audio features -----
    audio_df = load_audio_features()

    # Merge labels with audio to get a consistent patient universe
    merged = labels_df.merge(audio_df, on='patient_id', how='inner')
    all_patient_ids = merged['patient_id'].values
    y_all            = merged[Y_COL].values

    audio_feature_cols = [c for c in audio_df.columns if c != 'patient_id']
    X_audio_all = merged[audio_feature_cols].to_numpy(dtype=float)

    print(f"Total patients (labels ∩ audio): {len(all_patient_ids)}")
    print(f"Audio feature dims: {X_audio_all.shape[1]}")
    print(f"Class distribution: {np.bincount(y_all)}")

    if len(np.unique(y_all)) < 2 or len(y_all) < 50:
        print("⚠️  Insufficient data — skipping")
        return []

    # ----- CV setup -----
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    deproberta_config = DepRoBERTaConfig()
    results = []

    # Load 11 question features once — they are static per-patient scores
    # that do NOT depend on the fold split, so leakage-free to load here.
    question_df = load_question_features(llama_model)
    question_df = question_df[~question_df['patient_id'].isin(ORIGINAL_EXCLUDE)]

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(all_patient_ids, y_all)):
        print(f"\n--- Fold {fold} ---")
        train_pids = all_patient_ids[train_idx].tolist()
        test_pids  = all_patient_ids[test_idx].tolist()
        print(f"  Train: {len(train_pids)}  |  Test: {len(test_pids)}")

        # STEP 1: Fine-tune DepRoBERTa on this fold's training patients
        model_save_path = MODEL_SAVE_DIR / llama_model / f"fold_{fold}"

        if not model_save_path.exists():
            try:
                fine_tune_deproberta_for_fold(
                    llama_model=llama_model,
                    fold=fold,
                    train_patient_ids=train_pids,
                    save_dir=model_save_path,
                    config=deproberta_config,
                )
            except Exception as e:
                logging.error(f"Fine-tuning failed for fold {fold}: {e}")
                continue
        else:
            logging.info(f"Reusing existing model: {model_save_path}")

        # STEP 2: Extract 3 DepRoBERTa probability features for ALL patients
        prob_df = extract_deproberta_probabilities(
            llama_model=llama_model,
            fold=fold,
            patient_ids=all_patient_ids.tolist(),
            model_dir=model_save_path,
        )

        # Save per-fold probabilities for later SHAP analysis
        feat_out_dir = FEATURES_SAVE_DIR / llama_model / fusion_method
        feat_out_dir.mkdir(parents=True, exist_ok=True)
        prob_df.to_csv(feat_out_dir / f"fold_{fold}_probabilities.csv", index=False)

        # STEP 3: Combine 3 DepRoBERTa probs + 11 question features = 14 text features
        #   - question features are static per-patient scores (leakage-free)
        #   - we load them once per fold iteration but they don't change across folds
        combined_text_df = combine_text_features(prob_df, question_df)

        # Align with all_patient_ids order (some patients may lack question features)
        all_text_feature_cols = (
            ['prob_severe', 'prob_moderate', 'prob_not_depression']
            + [f'q{i}' for i in range(1, 12)]
        )  # 14 columns total
        combined_indexed = combined_text_df.set_index('patient_id')
        X_text_all = np.zeros((len(all_patient_ids), 14), dtype=float)
        for i, pid in enumerate(all_patient_ids):
            if pid in combined_indexed.index:
                X_text_all[i] = combined_indexed.loc[pid, all_text_feature_cols].values

        # STEP 4: Split into train/test arrays
        X_audio_train, X_audio_test = X_audio_all[train_idx], X_audio_all[test_idx]
        X_text_train,  X_text_test  = X_text_all[train_idx],  X_text_all[test_idx]
        y_train, y_test             = y_all[train_idx],        y_all[test_idx]

        # STEP 5: Apply fusion (fit on train, transform train+test)
        if fusion_method == "pca":
            fusion = PCAFusionTransformer(target_components=target_audio_components)
            X_train_fused = fusion.fit_transform(X_audio_train, X_text_train, y_train)
        else:  # rfe
            # RFE target = audio components + 14 text features
            rfe_target = target_audio_components + 14
            fusion = RFEFusionTransformer(target_features=rfe_target)
            X_train_fused = fusion.fit_transform(X_audio_train, X_text_train, y_train)

        X_test_fused = fusion.transform(X_audio_test, X_text_test)
        feat_info = fusion.get_feature_info()

        logging.info(
            f"  Fold {fold} feature info: {feat_info['total_features']} total dims "
            f"({feat_info['n_audio_selected']} audio + {feat_info['n_text_selected']} text)"
        )

        # STEP 6: Run all classifiers with inner GridSearch
        for clf_name in GRIDS.keys():
            try:
                clf = make_estimator(clf_name)
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True,
                                           random_state=42 + fold)
                gs = GridSearchCV(
                    estimator=clf,
                    param_grid=GRIDS[clf_name],
                    cv=inner_cv,
                    scoring="f1",
                    n_jobs=1,
                    refit=True,
                )
                gs.fit(X_train_fused, y_train)

                best_clf = gs.best_estimator_
                y_pred  = best_clf.predict(X_test_fused)
                y_score = get_scores(best_clf, X_test_fused)
                metrics = compute_metrics(y_test, y_score, y_pred)

                results.append({
                    "llama_model":             llama_model,
                    "fusion_method":           fusion_method,
                    "target_audio_components": target_audio_components,
                    "classifier":              clf_name,
                    "fold":                    fold,
                    "n_train":                 len(y_train),
                    "n_test":                  len(y_test),
                    "n_audio_selected":        feat_info['n_audio_selected'],
                    "n_text_selected":         feat_info['n_text_selected'],
                    "n_total_features":        feat_info['total_features'],
                    **metrics,
                })

                print(f"    {clf_name:<25s} "
                      f"F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}")

            except Exception as e:
                logging.error(f"Classifier {clf_name} failed on fold {fold}: {e}")
                continue

    return results


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def generate_shap_plots(df: pd.DataFrame) -> None:
    """
    SHAP beeswarm plot for the best (llama_model, classifier) configuration.

    Uses fold-0 saved probabilities + question features to rebuild the data,
    refits AudioFeatureSelector + classifier, then runs KernelExplainer.
    Feature names are fully labelled in the plot:
      - audio_PC1..N (PCA) or audio_feat_0001..N (ANOVA / mutual_info)
      - prob_severe, prob_moderate, prob_not_depression, q1..q11
    """
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)

    shap_output_dir = OUTPUT_DIR / "shap_analysis"
    shap_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Find best configuration ----
    agg = df.groupby(['llama_model', 'classifier'])['f1'].mean()
    best_idx = agg.idxmax()
    best_llama, best_clf_name = best_idx
    best_audio_method = (
        df[df['llama_model'] == best_llama]
        .groupby('best_audio_method')['f1'].mean().idxmax()
    )
    best_n_components = int(
        df[(df['llama_model'] == best_llama) & (df['best_audio_method'] == best_audio_method)]
        .groupby('best_n_components')['f1'].mean().idxmax()
    )
    best_f1 = agg[best_idx]

    print(f"\n  Best configuration:")
    print(f"   Llama model:      {best_llama}")
    print(f"   Classifier:       {best_clf_name}")
    print(f"   Audio method:     {best_audio_method}")
    print(f"   Audio components: {best_n_components}")
    print(f"   Mean F1 (5-fold): {best_f1:.3f}")

    # ---- 2. Reconstruct dataset ----
    labels_df = pd.read_csv(BINARY_LABELS_CSV)
    labels_df['patient_id'] = labels_df['patient_id'].astype(int)
    labels_df = labels_df[~labels_df['patient_id'].isin(ORIGINAL_EXCLUDE)]

    audio_df = load_audio_features()
    merged = labels_df.merge(audio_df, on='patient_id', how='inner')
    all_patient_ids = merged['patient_id'].values
    y_all = merged[Y_COL].values
    audio_feature_cols = [c for c in audio_df.columns if c != 'patient_id']
    X_audio_all = merged[audio_feature_cols].to_numpy(dtype=float)

    # ---- 3. Reproduce fold-0 split ----
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(iter(outer_cv.split(all_patient_ids, y_all)))

    # Load fold-0 probabilities
    fold_prob_csv = FEATURES_SAVE_DIR / best_llama / "fold_0_probabilities.csv"
    if not fold_prob_csv.exists():
        print(f"  Fold-0 probabilities not found at:\n  {fold_prob_csv}")
        print("  Skipping SHAP analysis.")
        return

    prob_df_raw = pd.read_csv(fold_prob_csv)
    question_df_shap = load_question_features(best_llama)
    question_df_shap = question_df_shap[~question_df_shap['patient_id'].isin(ORIGINAL_EXCLUDE)]
    combined_text_df = combine_text_features(prob_df_raw, question_df_shap)

    all_text_feature_cols = (
        ['prob_severe', 'prob_moderate', 'prob_not_depression']
        + [f'q{i}' for i in range(1, 12)]
    )
    combined_indexed = combined_text_df.set_index('patient_id')
    X_text_all = np.zeros((len(all_patient_ids), 14), dtype=float)
    for i, pid in enumerate(all_patient_ids):
        if pid in combined_indexed.index:
            X_text_all[i] = combined_indexed.loc[pid, all_text_feature_cols].values

    X_train_concat = np.hstack([X_audio_all[train_idx], X_text_all[train_idx]])
    X_test_concat  = np.hstack([X_audio_all[test_idx], X_text_all[test_idx]])
    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]

    # ---- 4. Retrain best pipeline ----
    print(f"\n  Retraining {best_clf_name} (audio={best_audio_method}/{best_n_components})...")
    clf_est = make_estimator(best_clf_name)
    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('audio', AudioFeatureSelector(method=best_audio_method, n_components=best_n_components), slice(0, X_audio_all.shape[1])),
            ('text', 'passthrough', slice(X_audio_all.shape[1], X_audio_all.shape[1] + X_text_all.shape[1]))
        ]
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf_est)
    ])

    # Combined param search on fold-0
    mapped_param_dist = {**AUDIO_SELECTION_PARAMS}
    for k, v in CLF_GRIDS[best_clf_name].items():
        mapped_k = k.replace('clf__', 'classifier__') if k.startswith('clf__') else k
        mapped_param_dist[mapped_k] = v

    mapped_param_dist['preprocessor__audio__method']       = [best_audio_method]
    mapped_param_dist['preprocessor__audio__n_components'] = [best_n_components]

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=pipeline, param_distributions=mapped_param_dist,
        n_iter=30, cv=inner_cv, scoring='f1', n_jobs=1, refit=True, random_state=42
    )
    rs.fit(X_train_concat, y_train)
    best_pipeline = rs.best_estimator_

    y_pred  = best_pipeline.predict(X_test_concat)
    if hasattr(best_pipeline, 'predict_proba'):
        y_score = best_pipeline.predict_proba(X_test_concat)
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
    else:
        scores = best_pipeline.decision_function(X_test_concat)
        if scores.ndim == 1:
            y_score = scores
        else:
            y_score = scores[:, 1]
            
    test_m  = compute_metrics(y_test, y_score, y_pred)
    print(f"  Test F1={test_m['f1']:.3f}  AUC={test_m['roc_auc']:.3f}")

    # ---- 5. Build named feature matrix for SHAP ----
    n_audio = best_n_components
    if best_audio_method == 'pca':
        audio_names = [f'audio_PC{i+1}' for i in range(n_audio)]
    else:
        audio_names = [f'audio_feat_{i:04d}' for i in range(n_audio)]
    text_names = (
        ['prob_severe', 'prob_moderate', 'prob_not_depression']
        + [f'q{i}' for i in range(1, 12)]
    )
    feature_names = audio_names + text_names
    
    X_train_reduced = best_pipeline.named_steps['preprocessor'].transform(X_train_concat)
    X_test_reduced  = best_pipeline.named_steps['preprocessor'].transform(X_test_concat)
    # Trim feature_names to actual output dim (in case n_components was capped)
    n_actual = X_train_reduced.shape[1]
    feature_names = feature_names[:n_actual]

    X_train_df = pd.DataFrame(X_train_reduced, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_reduced,  columns=feature_names)

    # ---- 6. SHAP via KernelExplainer + kmeans background ----
    print("\n  Computing SHAP values (KernelExplainer + kmeans, ~5 min)...")
    background = shap.kmeans(X_train_df, min(50, len(X_train_df)))

    clf_step = best_pipeline.named_steps['classifier']

    def predict_proba_fn(X_arr):
        proba = clf_step.predict_proba(X_arr) \
            if hasattr(clf_step, 'predict_proba') \
            else np.column_stack([-clf_step.decision_function(X_arr),
                                   clf_step.decision_function(X_arr)])
        return proba

    explainer = shap.KernelExplainer(predict_proba_fn, background)
    shap_values_all = explainer.shap_values(X_test_df)

    # Robust extraction of class-1 SHAP values regardless of SHAP version:
    #  - list [class0(n,f), class1(n,f)]       → shap_values_all[1]
    #  - ndarray (2, n_test, n_feat)            → arr[1]           (old SHAP)
    #  - ndarray (n_test, n_feat, 2)            → arr[:, :, 1]     (newer SHAP)
    #  - ndarray (n_test, n_feat)               → arr              (single output)
    shap_arr = np.array(shap_values_all)
    if isinstance(shap_values_all, list):
        shap_vals = np.array(shap_values_all[1])
    elif shap_arr.ndim == 3:
        if shap_arr.shape[0] == 2:
            # (n_classes=2, n_test, n_feat) — old SHAP format
            shap_vals = shap_arr[1]
        elif shap_arr.shape[2] == 2:
            # (n_test, n_feat, n_classes=2) — newer SHAP format
            shap_vals = shap_arr[:, :, 1]
        else:
            shap_vals = shap_arr[0]
    else:
        shap_vals = shap_arr
    print(f"  shap_vals shape after extraction: {shap_vals.shape}")

    # ---- 7. Feature importance CSV ----
    n_text = X_text_all.shape[1]
    n_audio_feat = X_train_reduced.shape[1] - n_text
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    print(f"  feature_names={len(feature_names)}, mean_abs_shap={mean_abs_shap.shape}, "
          f"n_audio_feat={n_audio_feat}, n_text={n_text}")
    modality = (['audio'] * n_audio_feat + ['text'] * n_text)[:len(feature_names)]
    importance_df = pd.DataFrame({
        'feature':       feature_names,
        'modality':      modality,
        'mean_abs_shap': mean_abs_shap[:len(feature_names)],
    }).sort_values('mean_abs_shap', ascending=False)

    importance_csv = shap_output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    print(f"\n  Top features by mean |SHAP|:")
    max_val = mean_abs_shap.max() if mean_abs_shap.max() > 0 else 1.0
    for _, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['mean_abs_shap'] / max_val * 20)
        print(f"   {row['feature']:<32s} [{row['modality']:5s}]  {row['mean_abs_shap']:.5f}  {bar}")

    # ---- Beeswarm PDF with named axes ----
    TOP_N = min(len(feature_names), 10)
    top_idx = np.argsort(mean_abs_shap)[-TOP_N:][::-1]

    shap_explanation = shap.Explanation(
        values=shap_vals[:, top_idx],
        base_values=np.zeros(len(shap_vals)),
        data=X_test_df.iloc[:, top_idx].values,
        feature_names=[feature_names[i] for i in top_idx],
    )

    plt.figure(figsize=(11, max(6, TOP_N * 0.6)))
    shap.plots.beeswarm(shap_explanation, max_display=TOP_N + 1, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    for ax in fig.get_axes():
        ax.patch.set_facecolor('white')

    title = (
        "Multimodal Depression — SHAP Feature Importance (Beeswarm)\n"
        f"{best_llama}  |  {best_clf_name}  |  "
        f"audio: {best_audio_method}/{best_n_components} components\n"
        f"Fold-0 test set  |  F1={test_m['f1']:.3f}  AUC={test_m['roc_auc']:.3f}  "
        f"|  {n_audio_feat} audio + 14 text features"
    )
    plt.suptitle(title, fontsize=9, y=1.01)
    plt.tight_layout()

    pdf_path = shap_output_dir / "shap_beeswarm_best_config.pdf"
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n  SHAP beeswarm PDF: {pdf_path}")
    print(f"  Feature importance CSV: {importance_csv}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Classification with CV-Aware DepRoBERTa"
    )
    parser.add_argument(
        "--llama_model",
        type=str,
        default=None,
        choices=list(LLM_SUMMARY_DIRS.keys()),
        help=(
            "Run only this Llama model. "
            "If omitted, all 3 models are run sequentially (~24h). "
            "Use this flag to split the workload across 3 separate jobs (~9h each)."
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MULTIMODAL CLASSIFICATION — CV-AWARE DEPROBERTA")
    print("=" * 70)
    print()

    set_seed(42)

    models_to_run = (
        {args.llama_model: LLM_SUMMARY_DIRS[args.llama_model]}
        if args.llama_model
        else LLM_SUMMARY_DIRS
    )

    n_clf = len(CLF_GRIDS)
    print(f"Classifiers:       {n_clf}")
    print(f"Llama models:      {len(models_to_run)}" +
          (f"  [{args.llama_model}]" if args.llama_model else "  [all]"))
    print(f"Audio selection:   PCA / ANOVA / mutual_info, n_components in [50, 100, 200]")
    print(f"Text features:     14 (3 DepRoBERTa probs + 11 question features)")
    print(f"Outer folds:       5")
    print(f"Inner iter:        RandomizedSearchCV n_iter=50")
    print()

    all_results = []
    done = 0
    total = len(models_to_run)

    for llama_model in models_to_run.keys():
        done += 1
        print(f"\n[{done}/{total}] {llama_model}")
        results = run_experiment(llama_model)
        all_results.extend(results)

        # Save per-model results immediately so progress is not lost if a later
        # model fails or the job hits the time limit.
        if results:
            model_df = pd.DataFrame(results)
            model_file = OUTPUT_DIR / f"multimodal_cv_deproberta_results_{llama_model}.csv"
            model_df.to_csv(model_file, index=False)
            print(f"  Intermediate results saved: {model_file}")

    # ---- Save combined results ----
    if all_results:
        df = pd.DataFrame(all_results)
        if args.llama_model:
            output_file = OUTPUT_DIR / f"multimodal_cv_deproberta_results_v2_{args.llama_model}.csv"
        else:
            output_file = OUTPUT_DIR / "multimodal_cv_deproberta_results_v2.csv"
        df.to_csv(output_file, index=False)

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        agg_df = df.groupby(['audio_source', 'llama_model', 'classifier']).agg({
            'f1': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std'],
            'best_audio_method': lambda x: x.mode()[0],
            'best_n_components': lambda x: int(x.mode()[0]),
        }).round(3)

        print("\nTOP 15 CONFIGURATIONS BY MEAN F1:")
        top_configs = agg_df.sort_values(('f1', 'mean'), ascending=False).head(15)
        print(top_configs.to_string())

        print(f"\n  Results saved to: {output_file}")
        print(f"  Rows: {len(df)}")

        # --- Save aggregated summary CSV (mean±std across folds, per audio source) ---
        summary_rows = []
        for (audio_src, llama, clf), grp in df.groupby(['audio_source', 'llama_model', 'classifier']):
            def safe_mode(s):
                m = s.mode()
                return m.iloc[0] if len(m) > 0 else '?'
            summary_rows.append({
                'audio_source':       audio_src,
                'llama_model':        llama,
                'classifier':         clf,
                'n_folds':            len(grp),
                'f1_mean':            round(grp['f1'].mean(), 4),
                'f1_std':             round(grp['f1'].std(), 4),
                'roc_auc_mean':       round(grp['roc_auc'].mean(), 4),
                'roc_auc_std':        round(grp['roc_auc'].std(), 4),
                'balanced_acc_mean':  round(grp['balanced_accuracy'].mean(), 4),
                'balanced_acc_std':   round(grp['balanced_accuracy'].std(), 4),
                'best_audio_method':  safe_mode(grp['best_audio_method']),
                'best_n_components':  int(safe_mode(grp['best_n_components'])),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values('f1_mean', ascending=False)
        if args.llama_model:
            summary_file = OUTPUT_DIR / f"summary_mean_std_{args.llama_model}.csv"
        else:
            summary_file = OUTPUT_DIR / "summary_mean_std.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"  Aggregated summary saved to: {summary_file}")
        print(f"  (columns: audio_source, llama_model, classifier, f1_mean/std, roc_auc_mean/std, balanced_acc_mean/std)")
        print("\n  TOP 15 BY F1_MEAN:")
        print(summary_df[['audio_source','llama_model','classifier',
                           'f1_mean','f1_std','roc_auc_mean','best_audio_method']].head(15).to_string(index=False))


        if args.llama_model:
            print(
                "\n  SHAP analysis skipped (single-model run).\n"
                "  After all 3 models finish, run:\n"
                "    python multimodal_classification_cv_deproberta_merge_shap.py\n"
                "  to merge CSVs and generate cross-model SHAP plots."
            )
        else:
            try:
                generate_shap_plots(df)
            except Exception as e:
                logging.error(f"SHAP analysis failed (non-fatal): {e}")
                print("  SHAP analysis skipped — check log for details.")

    print(f"\n  Completed {len(all_results)} classifier evaluations")




if __name__ == "__main__":
    main()
