"""
MULTIMODAL PHQ-9 Regression with CV-Aware DepRoBERTa

Combines:
  - AUDIO: wav2vec2 aggregated features (6144-dim)
           Reduced via AudioFeatureSelector (PCA / ANOVA / mutual_info)
           n_components in [50, 100, 200] — jointly optimised by inner CV
  - TEXT:  3 DepRoBERTa probs (fine-tuned per fold) + 11 question features = 14 text features

Key design (no data leakage):
  For each of 5 outer folds:
    1. Fine-tune DepRoBERTa on TRAIN patients only
    2. Extract 3 probability features for ALL patients using fold-specific model
    3. Combine with 11 static question features → 14-dim text array
    4. Inside inner RandomizedSearchCV:
       a. AudioFeatureSelector(method, n_components) fitted on TRAIN audio
       b. Reduced audio concat with 14 text features
       c. Regressor fitted on combined features
    5. Evaluate on test fold → R², MAE, RMSE

Total: 3 Llama models × 12 regressors × 5 folds
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                               ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedTokenizer)
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)

# Optional imports
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("⚠️  LightGBM not available")

try:
    from catboost import CatBoostRegressor
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

# PHQ-9 scores (used for DepRoBERTa fine-tuning AND regression target)
PHQ_LABELS_CSV = Path("CONFIGURE_ME/labels/regress_labels.csv")

# Output directory — v2 suffix, no overwrite of v1
OUTPUT_DIR = BASE_DIR / "Multi/multimodal_cv_deproberta_regression_results_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CONFIGURE_ME: STORAGE_ROOT should be on a large filesystem
WOODY_BASE = Path("CONFIGURE_ME/storage")

# Fine-tuned DepRoBERTa models (namespaced under 'multimodal_regression')
MODEL_SAVE_DIR = WOODY_BASE / "deproberta_cv_models_multimodal_regression"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Extracted DepRoBERTa features (per fold)
FEATURES_SAVE_DIR = WOODY_BASE / "deproberta_extracted_features_multimodal_regression"
FEATURES_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# LLM summary directories
LLM_SUMMARY_DIRS = {
    "llama_3.1_8B":  BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3108",
    "llama_3.1_70B": BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3170",
    "llama_3.3_70B": BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3370",
}

# 11 question feature files
QUESTION_FEATURE_FILES = {
    "llama_3.1_8B":  Path("CONFIGURE_ME/features/text_features_llama3.1_8B.csv"),
    "llama_3.1_70B": Path("CONFIGURE_ME/features/text_features_llama3.1_70B.csv"),
    "llama_3.3_70B": Path("CONFIGURE_ME/features/text_features_llama3.3_70B.csv"),
}

# DepRoBERTa configuration
CACHE_DIR = Path("CONFIGURE_ME/storage/models/hf_cache")
BASE_DEPROBERTA_MODEL = "rafalposwiata/deproberta-large-v1"
DEPROBERTA_EPOCHS = 10
DEPROBERTA_BATCH_SIZE = 32
DEPROBERTA_LR_CLASSIFIER = 5e-6
DEPROBERTA_LR_ENCODER = 1e-5
DEPROBERTA_MAX_LENGTH = 512

ID_COL, Y_COL = "patient_id", "PHQ9-Score"
ORIGINAL_EXCLUDE = set()  # participant IDs to exclude (configure for your data)

# v1 Audio selection params (kept for reference):
# AUDIO_SELECTION_PARAMS = {
#     'preprocessor__audio__method':       ['pca', 'anova', 'mutual_info'],
#     'preprocessor__audio__n_components': [50, 100, 200],
# }
# v2: adds selectk_pca_pruned method:
AUDIO_SELECTION_PARAMS = {
    'preprocessor__audio__method':        ['pca', 'anova', 'mutual_info', 'selectk_pca_pruned'],
    'preprocessor__audio__n_components':  [50, 100, 200],
    'preprocessor__audio__presel_k':      [200, 500],
    'preprocessor__audio__redund_thresh': [0.70],
}

# ============================================================================
# REGRESSOR GRIDS  (prefixed with reg__ for MultimodalPipeline)
# ============================================================================

REG_GRIDS = {
    "Ridge": {
        'reg__alpha': [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3],
        'reg__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
    },
    "Lasso": {
        'reg__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    },
    "ElasticNet": {
        'reg__alpha':    [1e-3, 1e-2, 1e-1, 1.0, 10.0],
        'reg__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "SVR": {
        'reg__C':       [0.1, 1, 10, 100],
        'reg__gamma':   [1, 0.1, 0.01, 0.001],
        'reg__kernel':  ['rbf', 'poly', 'sigmoid'],
        'reg__epsilon': [0.1, 0.2],
    },
    "RandomForest": {
        'reg__n_estimators':    [50, 100, 200],
        'reg__max_depth':       [None, 10, 20, 30],
        'reg__min_samples_split': [2, 5, 10],
    },
    "XGBoost": {
        'reg__n_estimators':  [50, 100, 200],
        'reg__learning_rate': [0.01, 0.1, 0.2],
        'reg__max_depth':     [3, 5, 7],
        'reg__subsample':     [0.8, 1.0],
    },
    "AdaBoost": {
        'reg__n_estimators':  [50, 100, 200],
        'reg__learning_rate': [0.01, 0.1, 1.0],
        'reg__loss':          ['linear', 'square', 'exponential'],
    },
    "DecisionTree": {
        'reg__max_depth':        [None, 10, 20, 30],
        'reg__min_samples_split': [2, 5, 10],
        'reg__criterion':        ['squared_error', 'friedman_mse', 'absolute_error'],
    },
    "KNN": {
        'reg__n_neighbors': [3, 5, 7, 9],
        'reg__weights':     ['uniform', 'distance'],
        'reg__metric':      ['euclidean', 'manhattan', 'minkowski'],
    },
    "ExtraTrees": {
        'reg__n_estimators':    [50, 100, 200],
        'reg__max_depth':       [None, 10, 20, 30],
        'reg__min_samples_split': [2, 5, 10],
        'reg__min_samples_leaf':  [1, 2, 4],
    },
    "GradientBoosting": {
        'reg__n_estimators':  [50, 100, 200],
        'reg__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'reg__max_depth':     [3, 5, 7],
        'reg__subsample':     [0.8, 1.0],
    },
    "MLP": {
        'reg__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'reg__activation':  ['relu', 'tanh'],
        'reg__alpha':       [0.0001, 0.001, 0.01],
        'reg__learning_rate': ['constant', 'adaptive'],
    },
}

if LIGHTGBM_AVAILABLE:
    REG_GRIDS["LightGBM"] = {
        'reg__n_estimators':    [50, 100, 200],
        'reg__learning_rate':   [0.01, 0.05, 0.1],
        'reg__max_depth':       [3, 5, 7, -1],
        'reg__num_leaves':      [15, 31, 63],
        'reg__min_child_samples': [10, 20, 30],
    }

if CATBOOST_AVAILABLE:
    REG_GRIDS["CatBoost"] = {
        'reg__iterations':    [50, 100, 200],
        'reg__learning_rate': [0.01, 0.05, 0.1],
        'reg__depth':         [3, 5, 7],
        'reg__l2_leaf_reg':   [1, 3, 5],
    }


# ============================================================================
# AUDIO FEATURE SELECTOR
# ============================================================================

class AudioFeatureSelector:
    """
    Sklearn-compatible audio feature selector. v2 adds 'selectk_pca_pruned'.

    Methods:
      'pca'               - PCA (v1)
      'anova'             - SelectKBest(f_regression) (v1)
      'mutual_info'       - SelectKBest(mutual_info_regression) (v1)
      'selectk_pca_pruned'- NEW (v2): SelectKBest(k=presel_k) → PCA(n_components)
                            → drop components with Pearson |r|>redund_thresh vs text.
    """

    def __init__(self, method: str = 'pca', n_components: int = 100,
                 presel_k: int = 200, redund_thresh: float = 0.70):
        self.method = method
        self.n_components = n_components
        self.presel_k = presel_k
        self.redund_thresh = redund_thresh
        self.scaler = StandardScaler()
        self.selector = None
        self._presel = None
        self._pca = None
        self._keep_mask = None
        self._X_text_train = None

    def fit(self, X: np.ndarray, y: np.ndarray = None, X_text_train: np.ndarray = None):
        X_scaled = self.scaler.fit_transform(X)
        n = min(self.n_components, X.shape[1], X.shape[0] - 1)
        if self.method == 'pca':
            self.selector = PCA(n_components=n)
            self.selector.fit(X_scaled)
        elif self.method == 'anova':
            self.selector = SelectKBest(f_regression, k=n)
            self.selector.fit(X_scaled, y)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_regression, k=n)
            self.selector.fit(X_scaled, y)
        elif self.method == 'selectk_pca_pruned':
            k_pre = min(self.presel_k, X_scaled.shape[1])
            self._presel = SelectKBest(f_regression, k=k_pre).fit(X_scaled, y)
            X_pre = self._presel.transform(X_scaled)
            n_pca = min(n, X_pre.shape[0] - 1, X_pre.shape[1])
            self._pca = PCA(n_components=n_pca, random_state=42).fit(X_pre)
            X_pca = self._pca.transform(X_pre)
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


# ============================================================================
# MULTIMODAL PIPELINE
# ============================================================================

def make_multimodal_pipeline(audio_selector_params, regressor, n_audio_features):
    """
    Creates a standard sklearn Pipeline that:
      1. Applies AudioFeatureSelector to the first `n_audio_features` columns.
      2. Passes the remaining 14 text columns through untouched.
      3. Regresses with the given regressor.
    """
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
        ('regressor', regressor)
    ])


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2":   r2_score(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def make_regressor(name: str):
    if name == "Ridge":
        return Ridge(max_iter=10000)
    elif name == "Lasso":
        return Lasso(max_iter=10000)
    elif name == "ElasticNet":
        return ElasticNet(max_iter=10000)
    elif name == "SVR":
        return SVR()
    elif name == "RandomForest":
        # return RandomForestRegressor(random_state=42)  # v1
        return RandomForestRegressor(random_state=42, n_jobs=1)  # v2: n_jobs=1 avoids OpenMP segfault on GPU nodes
    elif name == "XGBoost":
        # return XGBRegressor(objective="reg:squarederror", random_state=42)  # v1
        return XGBRegressor(objective="reg:squarederror", random_state=42, nthread=1, device="cpu")  # v2: force CPU, single thread
    elif name == "AdaBoost":
        return AdaBoostRegressor(random_state=42)
    elif name == "DecisionTree":
        return DecisionTreeRegressor(random_state=42)
    elif name == "KNN":
        return KNeighborsRegressor()
    elif name == "ExtraTrees":
        # return ExtraTreesRegressor(random_state=42)  # v1
        return ExtraTreesRegressor(random_state=42, n_jobs=1)  # v2: n_jobs=1
    elif name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=42)  # single-threaded by default
    elif name == "MLP":
        return MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
    elif name == "LightGBM" and LIGHTGBM_AVAILABLE:
        # return LGBMRegressor(random_state=42, verbose=-1, force_col_wise=True)  # v1
        return LGBMRegressor(random_state=42, verbose=-1, n_jobs=1, device="cpu")  # v2: force CPU, single thread
    elif name == "CatBoost" and CATBOOST_AVAILABLE:
        return CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
    else:
        raise ValueError(f"Unknown or unavailable regressor: {name}")


# ============================================================================
# AUDIO DATA LOADING
# ============================================================================

# ---- v1 load_audio_features (single source) — kept for reference ----
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
    if 'file_id' in audio_df.columns:
        audio_df['patient_id'] = audio_df['file_id'].astype(str).str.extract(r'(\d+)').astype(int)
        audio_df = audio_df.drop(columns=['file_id'])
    elif 'patient_id' not in audio_df.columns:
        first_col = audio_df.columns[0]
        audio_df['patient_id'] = audio_df[first_col].astype(int)
        audio_df = audio_df.drop(columns=[first_col])
    else:
        audio_df['patient_id'] = audio_df['patient_id'].astype(int)
    # drop n_segments if present
    audio_df = audio_df[[c for c in audio_df.columns if c != 'n_segments']]
    audio_df = audio_df[~audio_df['patient_id'].isin(ORIGINAL_EXCLUDE)]
    return audio_df


def load_question_features(llama_model: str) -> pd.DataFrame:
    """Load 11 static question features (q1-q11)."""
    df = pd.read_csv(QUESTION_FEATURE_FILES[llama_model])
    df['patient_id'] = df['patient_id'].astype(int)
    return df[['patient_id'] + [f'q{i}' for i in range(1, 12)]]


def combine_text_features(prob_df: pd.DataFrame, question_df: pd.DataFrame) -> pd.DataFrame:
    """Merge 3 DepRoBERTa probs + 11 question features = 14 text features."""
    combined = prob_df.merge(question_df, on='patient_id', how='inner')
    cols = (
        ['patient_id', 'prob_severe', 'prob_moderate', 'prob_not_depression']
        + [f'q{i}' for i in range(1, 12)]
    )
    return combined[cols]


# ============================================================================
# DEPROBERTA FINE-TUNING
# ============================================================================

@dataclass
class DepRoBERTaConfig:
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
    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer: PreTrainedTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer.encode_plus(
            self.texts[idx], add_special_tokens=True,
            max_length=self.max_length, padding="max_length",
            truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_summaries_for_patients(llama_model: str,
                                patient_ids: List[int]) -> Dict[int, str]:
    summary_dir = LLM_SUMMARY_DIRS[llama_model]
    summaries = {}
    for pid in patient_ids:
        for json_file in [
            summary_dir / f"{pid:03d}_transcription.json",
            summary_dir / f"{pid:03d}_whisper.json",
        ]:
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f).get("summary", "").strip()
                        if summary:
                            summaries[pid] = summary
                            break
                except Exception as e:
                    logging.warning(f"Error loading {json_file}: {e}")
    return summaries


def map_phq_to_label(score: int) -> int:
    if score >= 14: return 0   # Severe
    if score >= 7:  return 1   # Moderate
    return 2                   # Not Depressed / Mild


def fine_tune_deproberta_for_fold(
    llama_model: str, fold: int, train_patient_ids: List[int],
    save_dir: Path, config: DepRoBERTaConfig
) -> Path:
    """Fine-tune DepRoBERTa (3-class PHQ) on training fold only."""
    logging.info(f"\n{'='*70}")
    logging.info(f"Fine-tuning DepRoBERTa: {llama_model}, Fold {fold}")
    logging.info(f"{'='*70}")

    labels_df = pd.read_csv(PHQ_LABELS_CSV)
    labels_df['PHQ9-Score'] = pd.to_numeric(labels_df['PHQ9-Score'], errors='coerce')
    labels_df.dropna(subset=['PHQ9-Score'], inplace=True)
    labels_df['label'] = labels_df['PHQ9-Score'].apply(map_phq_to_label)
    train_df = labels_df[labels_df['patient_id'].isin(train_patient_ids)].copy()

    summaries = load_summaries_for_patients(llama_model, train_df['patient_id'].tolist())
    train_df['text'] = train_df['patient_id'].map(summaries)
    train_df = train_df.dropna(subset=['text'])

    logging.info(f"Training samples with summary: {len(train_df)}")
    if len(train_df) < 10:
        raise ValueError("Insufficient training data for fine-tuning")

    train_subset, val_subset = train_test_split(
        train_df, test_size=config.validation_size,
        stratify=train_df['label'], random_state=config.random_seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path, cache_dir=str(CACHE_DIR)
    )
    train_loader = DataLoader(
        ClinicalTextDataset(train_subset['text'].tolist(), train_subset['label'].tolist(),
                            tokenizer, config.max_token_length),
        batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ClinicalTextDataset(val_subset['text'].tolist(), val_subset['label'].tolist(),
                            tokenizer, config.max_token_length),
        batch_size=config.batch_size, shuffle=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_path, num_labels=config.num_labels, cache_dir=str(CACHE_DIR)
    ).to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    num_layers = model.roberta.config.num_hidden_layers
    for name, param in model.named_parameters():
        for i in range(num_layers - config.last_n_layers_to_unfreeze, num_layers):
            if f"encoder.layer.{i}." in name:
                param.requires_grad = True

    optimizer = AdamW([
        {"params": model.classifier.parameters(), "lr": config.lr_classifier},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "encoder.layer" in n],
         "lr": config.lr_encoder},
    ])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(config.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                total_val_loss += out.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logging.info(f"✓ Best model saved → {save_dir}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    logging.info(f"Fine-tuning complete. Best val loss: {best_val_loss:.4f}")
    return save_dir


def extract_deproberta_probabilities(
    llama_model: str, fold: int, patient_ids: List[int], model_dir: Path
) -> pd.DataFrame:
    """Extract 3 probability features for all patients using fold-specific model."""
    logging.info(f"Extracting DepRoBERTa probs: {llama_model}, Fold {fold}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    summaries = load_summaries_for_patients(llama_model, patient_ids)
    results = []
    for pid in tqdm(patient_ids, desc="Extracting probabilities"):
        summary = summaries.get(pid)
        if summary is None:
            results.append({'patient_id': pid, 'prob_severe': 0.0,
                            'prob_moderate': 0.0, 'prob_not_depression': 0.0})
            continue
        inputs = tokenizer(
            summary, return_tensors="pt", padding=True,
            truncation=True, max_length=DEPROBERTA_MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
        results.append({
            'patient_id': pid,
            'prob_severe':         float(probs[0]),
            'prob_moderate':       float(probs[1]),
            'prob_not_depression': float(probs[2]),
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(results)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(llama_model: str) -> List[Dict]:
    """
    v2: Full CV experiment for one llama_model, iterating over all audio sources.

    For each of 5 outer folds × 4 audio sources:
      1. Fine-tune DepRoBERTa on TRAIN patients (3-class PHQ-9 categorical)
      2. Extract 3 probability features for ALL patients
      3. Combine with 11 question features → 14-dim text array
      4. Inner RandomizedSearchCV (n_iter=50):
         - AudioFeatureSelector(method, n_components) [incl. selectk_pca_pruned]
         - Regressor (PHQ-9 continuous)
      5. Evaluate: R², MAE, RMSE on test fold
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {llama_model}")
    print(f"{'='*70}")

    # ----- Load PHQ-9 labels -----
    labels_df = pd.read_csv(PHQ_LABELS_CSV)
    labels_df['patient_id'] = labels_df['patient_id'].astype(int)
    labels_df['PHQ9-Score'] = pd.to_numeric(labels_df['PHQ9-Score'], errors='coerce')
    labels_df.dropna(subset=['PHQ9-Score'], inplace=True)
    labels_df = labels_df[~labels_df['patient_id'].isin(ORIGINAL_EXCLUDE)]

    # ----- Load 11 question features (static, leakage-free) -----
    question_df = load_question_features(llama_model)
    question_df = question_df[~question_df['patient_id'].isin(ORIGINAL_EXCLUDE)]

    # ----- CV setup -----
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    deproberta_config = DepRoBERTaConfig()
    results = []

    all_text_feature_cols = (
        ['prob_severe', 'prob_moderate', 'prob_not_depression']
        + [f'q{i}' for i in range(1, 12)]
    )

    # Iterate over all audio sources (v2)
    for audio_source_name in AUDIO_SOURCES:
        if not AUDIO_SOURCES[audio_source_name].exists():
            logging.warning(f"Audio source not found, skipping: {AUDIO_SOURCES[audio_source_name]}")
            continue

        print(f"\n{'='*60}")
        print(f"Audio source: {audio_source_name}")
        print(f"{'='*60}")

        audio_df = load_audio_features(audio_source_name)
        merged = labels_df.merge(audio_df, on='patient_id', how='inner')
        all_patient_ids = merged['patient_id'].values
        y_all            = merged[Y_COL].values
        audio_feature_cols = [c for c in audio_df.columns if c != 'patient_id']
        X_audio_all = np.nan_to_num(merged[audio_feature_cols].to_numpy(dtype=float))

        print(f"Total patients (labels ∩ audio): {len(all_patient_ids)}")
        print(f"Audio feature dims: {X_audio_all.shape[1]}")
        print(f"PHQ-9 range: {y_all.min():.1f}–{y_all.max():.1f}, mean: {y_all.mean():.1f}±{y_all.std():.1f}")

        if len(y_all) < 50:
            print("⚠️  Insufficient data — skipping")
            continue

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(all_patient_ids)):
            print(f"\n  --- Fold {fold} | {audio_source_name} ---")
            train_pids = all_patient_ids[train_idx].tolist()
            print(f"    Train: {len(train_pids)}  |  Test: {len(test_idx)}")

            # STEP 1: Fine-tune DepRoBERTa
            model_save_path = MODEL_SAVE_DIR / llama_model / f"fold_{fold}"
            if not model_save_path.exists():
                try:
                    fine_tune_deproberta_for_fold(
                        llama_model=llama_model, fold=fold,
                        train_patient_ids=train_pids,
                        save_dir=model_save_path, config=deproberta_config,
                    )
                except Exception as e:
                    logging.error(f"Fine-tuning failed for fold {fold}: {e}")
                    continue
            else:
                logging.info(f"Reusing existing model: {model_save_path}")

            # STEP 2: Extract 3 DepRoBERTa probability features
            prob_df = extract_deproberta_probabilities(
                llama_model=llama_model, fold=fold,
                patient_ids=all_patient_ids.tolist(), model_dir=model_save_path,
            )
            feat_out_dir = FEATURES_SAVE_DIR / llama_model
            feat_out_dir.mkdir(parents=True, exist_ok=True)
            prob_df.to_csv(feat_out_dir / f"fold_{fold}_probabilities.csv", index=False)

            # STEP 3: Combine 3 probs + 11 question features = 14 text features
            combined_text_df = combine_text_features(prob_df, question_df)
            combined_indexed = combined_text_df.set_index('patient_id')
            X_text_all = np.zeros((len(all_patient_ids), 14), dtype=float)
            for i, pid in enumerate(all_patient_ids):
                if pid in combined_indexed.index:
                    X_text_all[i] = combined_indexed.loc[pid, all_text_feature_cols].values

            # STEP 4: Train/test split
            X_concat_all = np.hstack([X_audio_all, X_text_all])
            X_train  = X_concat_all[train_idx]
            X_test   = X_concat_all[test_idx]
            y_train = y_all[train_idx]
            y_test  = y_all[test_idx]

            # Text train subset (for redundancy pruning)
            X_text_train = X_text_all[train_idx]
            n_audio_feats = X_audio_all.shape[1]

            # STEP 5: Per-regressor inner RandomizedSearchCV
            for reg_name in REG_GRIDS.keys():
                try:
                    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42 + fold)
                    mapped_param_dist = {**AUDIO_SELECTION_PARAMS}
                    for k, v in REG_GRIDS[reg_name].items():
                        mapped_k = k.replace('reg__', 'regressor__') if k.startswith('reg__') else k
                        mapped_param_dist[mapped_k] = v

                    reg_est = make_regressor(reg_name)

                    audio_sel = AudioFeatureSelector(method='pca', n_components=100,
                                                     presel_k=200, redund_thresh=0.70)
                    audio_sel._X_text_train = X_text_train

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('audio', audio_sel, slice(0, n_audio_feats)),
                            ('text', 'passthrough', slice(n_audio_feats, X_concat_all.shape[1]))
                        ]
                    )
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', reg_est)
                    ])

                    rs = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=mapped_param_dist,
                        n_iter=50,
                        cv=inner_cv,
                        scoring="neg_mean_absolute_error",
                        n_jobs=1,
                        refit=True,
                        random_state=42 + fold,
                    )
                    rs.fit(X_train, y_train)

                    best_pipeline = rs.best_estimator_
                    y_pred  = best_pipeline.predict(X_test)
                    metrics = compute_metrics(y_test, y_pred)

                    best_audio_method = rs.best_params_.get('preprocessor__audio__method', '?')
                    best_n_components = rs.best_params_.get('preprocessor__audio__n_components', '?')

                    results.append({
                        "llama_model":        llama_model,
                        "audio_source":        audio_source_name,   # v2
                        "regressor":          reg_name,
                        "fold":               fold,
                        "n_train":            len(y_train),
                        "n_test":             len(y_test),
                        "best_audio_method":  best_audio_method,
                        "best_n_components":  best_n_components,
                        "n_text_features":    14,
                        **metrics,
                    })

                    print(f"    {reg_name:<25s} "
                          f"R²={metrics['r2']:.3f}  MAE={metrics['mae']:.2f}  "
                          f"[{best_audio_method}/{best_n_components}]")

                except Exception as e:
                    logging.error(f"Regressor {reg_name} failed on fold {fold} "
                                  f"[{audio_source_name}]: {e}")
                    continue

    return results


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def generate_shap_plots(df: pd.DataFrame) -> None:
    """
    SHAP analysis for the best (llama_model, regressor) configuration.

    Feature names in the beeswarm plot:
      - audio_PC1..N (PCA) or audio_feat_0001..N (ANOVA / mutual_info)
      - prob_severe, prob_moderate, prob_not_depression, q1..q11
    """
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)

    shap_output_dir = OUTPUT_DIR / "shap_analysis"
    shap_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Find best configuration ----
    agg = df.groupby(['llama_model', 'regressor'])['r2'].mean()
    best_idx = agg.idxmax()
    best_llama, best_reg_name = best_idx
    best_audio_method = (
        df[df['llama_model'] == best_llama]
        .groupby('best_audio_method')['r2'].mean().idxmax()
    )
    best_n_components = int(
        df[(df['llama_model'] == best_llama) &
           (df['best_audio_method'] == best_audio_method)]
        .groupby('best_n_components')['r2'].mean().idxmax()
    )
    best_r2 = agg[best_idx]

    print(f"\n  Best configuration:")
    print(f"   Llama model:      {best_llama}")
    print(f"   Regressor:        {best_reg_name}")
    print(f"   Audio method:     {best_audio_method}")
    print(f"   Audio components: {best_n_components}")
    print(f"   Mean R² (5-fold): {best_r2:.3f}")

    # ---- 2. Reconstruct dataset ----
    labels_df = pd.read_csv(PHQ_LABELS_CSV)
    labels_df['patient_id'] = labels_df['patient_id'].astype(int)
    labels_df['PHQ9-Score'] = pd.to_numeric(labels_df['PHQ9-Score'], errors='coerce')
    labels_df.dropna(subset=['PHQ9-Score'], inplace=True)
    labels_df = labels_df[~labels_df['patient_id'].isin(ORIGINAL_EXCLUDE)]

    audio_df = load_audio_features()
    merged = labels_df.merge(audio_df, on='patient_id', how='inner')
    all_patient_ids = merged['patient_id'].values
    y_all = merged[Y_COL].values
    audio_feature_cols = [c for c in audio_df.columns if c != 'patient_id']
    X_audio_all = merged[audio_feature_cols].to_numpy(dtype=float)

    # ---- 3. Reproduce fold-0 split ----
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(iter(outer_cv.split(all_patient_ids)))

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

    X_audio_train = X_audio_all[train_idx]
    X_audio_test  = X_audio_all[test_idx]
    X_text_train  = X_text_all[train_idx]
    X_text_test   = X_text_all[test_idx]
    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]

    # ---- 4. Retrain best pipeline ----
    print(f"\n  Retraining {best_reg_name} (audio={best_audio_method}/{best_n_components})...")
    reg_est   = make_regressor(best_reg_name)
    
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
        ('regressor', reg_est)
    ])

    mapped_param_dist = {**AUDIO_SELECTION_PARAMS}
    for k, v in REG_GRIDS[best_reg_name].items():
        mapped_k = k.replace('reg__', 'regressor__') if k.startswith('reg__') else k
        mapped_param_dist[mapped_k] = v

    mapped_param_dist['preprocessor__audio__method']       = [best_audio_method]
    mapped_param_dist['preprocessor__audio__n_components'] = [best_n_components]

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=pipeline, param_distributions=mapped_param_dist,
        n_iter=30, cv=inner_cv, scoring='neg_mean_absolute_error',
        n_jobs=1, refit=True, random_state=42,
    )
    X_train_concat = np.hstack([X_audio_train, X_text_train])
    X_test_concat  = np.hstack([X_audio_test, X_text_test])
    rs.fit(X_train_concat, y_train)
    best_pipeline = rs.best_estimator_

    y_pred = best_pipeline.predict(X_test_concat)
    test_m = compute_metrics(y_test, y_pred)
    print(f"  Test R²={test_m['r2']:.3f}  MAE={test_m['mae']:.2f}  RMSE={test_m['rmse']:.2f}")

    # ---- 5. Build named feature matrix ----
    # Preprocessor extracts method inside so we reconstruct names
    n_audio = best_n_components
    if best_audio_method == 'pca':
        audio_names = [f'audio_PC{i+1}' for i in range(n_audio)]
    else:
        # Fallback names
        audio_names = [f'audio_feat_{i:04d}' for i in range(n_audio)]
    text_names = (
        ['prob_severe', 'prob_moderate', 'prob_not_depression']
        + [f'q{i}' for i in range(1, 12)]
    )
    feature_names = audio_names + text_names
    
    X_train_reduced = best_pipeline.named_steps['preprocessor'].transform(X_train_concat)
    X_test_reduced  = best_pipeline.named_steps['preprocessor'].transform(X_test_concat)
    n_actual = X_train_reduced.shape[1]
    feature_names = feature_names[:n_actual]

    X_train_df = pd.DataFrame(X_train_reduced, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_reduced,  columns=feature_names)

    # ---- 6. SHAP via KernelExplainer (regression → single output) ----
    print("\n  Computing SHAP values (KernelExplainer + kmeans background, ~5 min)...")
    background = shap.kmeans(X_train_df, min(50, len(X_train_df)))

    def predict_fn(X_arr):
        return best_pipeline.regressor.predict(X_arr)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_vals = explainer.shap_values(X_test_df)   # shape: (n_test, n_features)
    shap_arr  = np.array(shap_vals)

    # ---- 7. Feature importance CSV ----
    n_audio_feat  = X_train_reduced.shape[1] - 14
    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    modality = ['audio'] * n_audio_feat + ['text'] * 14
    importance_df = pd.DataFrame({
        'feature':       feature_names,
        'modality':      modality[:len(feature_names)],
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)

    importance_csv = shap_output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    print(f"\n  Top features by mean |SHAP|:")
    max_val = mean_abs_shap.max() if mean_abs_shap.max() > 0 else 1.0
    for _, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['mean_abs_shap'] / max_val * 20)
        print(f"   {row['feature']:<32s} [{row['modality']:5s}]  {row['mean_abs_shap']:.5f}  {bar}")

    # ---- Beeswarm PDF with named axes ----
    TOP_N = min(len(feature_names), 20)
    top_idx = np.argsort(mean_abs_shap)[-TOP_N:][::-1]

    shap_explanation = shap.Explanation(
        values=shap_arr[:, top_idx],
        base_values=np.full(len(shap_arr), explainer.expected_value),
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
        "Multimodal PHQ-9 Regression — SHAP Feature Importance (Beeswarm)\n"
        f"{best_llama}  |  {best_reg_name}  |  "
        f"audio: {best_audio_method}/{best_n_components} components\n"
        f"Fold-0 test  |  R²={test_m['r2']:.3f}  MAE={test_m['mae']:.2f}  "
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

def summarize_results(df: pd.DataFrame, llama_label: str = None) -> None:
    # v2: group by audio_source too so each audio source gets its own row in summary
    grouped = df.groupby(['audio_source', 'llama_model', 'regressor'])
    rows = []
    for (audio_src, llama, reg), grp in grouped:
        def safe_mode(s):
            m = s.mode()
            return m.iloc[0] if len(m) > 0 else '?'
        rows.append({
            'audio_source':  audio_src,
            'llama_model':   llama,
            'regressor':     reg,
            'n_folds':       len(grp),
            'r2_mean':       round(grp['r2'].mean(), 4),
            'r2_std':        round(grp['r2'].std(), 4),
            'mae_mean':      round(grp['mae'].mean(), 4),
            'mae_std':       round(grp['mae'].std(), 4),
            'rmse_mean':     round(grp['rmse'].mean(), 4),
            'rmse_std':      round(grp['rmse'].std(), 4),
            'best_audio_method': safe_mode(grp['best_audio_method']),
            'best_n_components': int(safe_mode(grp['best_n_components']))
                                  if grp['best_n_components'].dtype != object else safe_mode(grp['best_n_components']),
        })
    summary_df = pd.DataFrame(rows).sort_values('r2_mean', ascending=False)
    if llama_label:
        summary_file = OUTPUT_DIR / f"summary_mean_std_{llama_label}.csv"
    else:
        summary_file = OUTPUT_DIR / "summary_mean_std.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n" + "=" * 70)
    print("TOP 10 REGRESSORS (by R²)")
    print("=" * 70)
    for _, row in summary_df.head(10).iterrows():
        print(f"  {row['regressor']:<22s} ({row['llama_model']}, {row['audio_source']},  "
              f"{row['best_audio_method']}/{row['best_n_components']})  "
              f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}  "
              f"MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f}")
    print(f"\n  Summary saved → {summary_file}")



def main(args=None):
    print("=" * 70)
    print("MULTIMODAL PHQ-9 REGRESSION — CV-AWARE DEPROBERTA")
    print("=" * 70)
    print()

    set_seed(42)

    # --- v2: support running a single llama model (for per-job execution) ---
    if args is not None and args.llama_model:
        models_to_run = {args.llama_model: LLM_SUMMARY_DIRS[args.llama_model]}
    else:
        models_to_run = LLM_SUMMARY_DIRS

    n_reg = len(REG_GRIDS)
    print(f"Regressors:        {n_reg}")
    print(f"Llama models:      {len(models_to_run)}" +
          (f"  [{args.llama_model}]" if (args and args.llama_model) else "  [all]"))
    print(f"Audio selection:   PCA / ANOVA / mutual_info / selectk_pca_pruned")
    print(f"Text features:     14 (3 DepRoBERTa probs + 11 question features)")
    print(f"Outer folds:       5  (KFold, regression)")
    print(f"Inner iter:        RandomizedSearchCV n_iter=50")
    print(f"Target:            PHQ-9 continuous score (0–27)")
    print()

    all_results = []
    total = len(models_to_run)

    for done, llama_model in enumerate(models_to_run.keys(), 1):
        print(f"\n[{done}/{total}] {llama_model}")
        results = run_experiment(llama_model)
        all_results.extend(results)

        # Save per-model results immediately — avoids losing progress on time limit
        if results:
            model_df = pd.DataFrame(results)
            model_file = OUTPUT_DIR / f"multimodal_cv_deproberta_regression_results_{llama_model}.csv"
            model_df.to_csv(model_file, index=False)
            print(f"  Intermediate results saved: {model_file}")

    if all_results:
        df = pd.DataFrame(all_results)
        # Per-llama or combined output — never overwrites
        if args is not None and args.llama_model:
            output_file = OUTPUT_DIR / f"multimodal_cv_deproberta_regression_results_v2_{args.llama_model}.csv"
        else:
            output_file = OUTPUT_DIR / "multimodal_cv_deproberta_regression_results_v2.csv"
        df.to_csv(output_file, index=False)
        print(f"\n  Results saved → {output_file}  ({len(df)} rows)")

        summarize_results(df, llama_label=args.llama_model if args else None)

        try:
            generate_shap_plots(df)
        except Exception as e:
            logging.error(f"SHAP analysis failed (non-fatal): {e}")
            print("  SHAP analysis skipped — check log for details.")

    print(f"\n  Completed {len(all_results)} regressor evaluations")


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="Multimodal regression (per-llama mode)")
    _parser.add_argument(
        "--llama_model", type=str, default=None,
        choices=list(LLM_SUMMARY_DIRS.keys()),
        help="Run only this llama model (omit to run all)"
    )
    _args = _parser.parse_args()
    main(_args)
