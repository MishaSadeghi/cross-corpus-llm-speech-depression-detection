"""
MAXIMAL Text-Based Depression Classification with CV-Aware DepRoBERTa

This script fixes data leakage by fine-tuning DepRoBERTa separately for each
cross-validation fold, using only the training data from that fold.

Key differences from maximal_text_classification.py:
- Fine-tunes DepRoBERTa for each CV fold (training data only)
- Extracts 3 probability features using fold-specific models
- Merges with existing 11 question features (already leakage-free)
- Everything integrated in one script

Total: 3 Llama models × 4 quality filters × 12 classifiers × 5 folds
"""
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
# TWO different label files:
# 1. PHQ-9 scores for DepRoBERTa fine-tuning (3-class categorical)
PHQ_LABELS_CSV = Path("CONFIGURE_ME/labels/regress_labels.csv")
# 2. Binary depression labels for FINAL classification (binary)
BINARY_LABELS_CSV = Path("CONFIGURE_ME/labels/classi_labels.csv")
OUTPUT_DIR = Path("CONFIGURE_ME/storage/results/text_nested_cv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CONFIGURE_ME: STORAGE_ROOT should be on a large filesystem
WOODY_BASE = Path("CONFIGURE_ME/storage")

# Model storage: Fine-tuned DepRoBERTa models for each fold
MODEL_SAVE_DIR = WOODY_BASE / "deproberta_cv_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Feature storage: Extracted probability features from DepRoBERTa
FEATURES_SAVE_DIR = WOODY_BASE / "deproberta_extracted_features"
FEATURES_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# LLM summary directories (for DepRoBERTa input)
LLM_SUMMARY_DIRS = {
    "llama_3.1_8B": BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3108",
    "llama_3.1_70B": BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3170",
    "llama_3.3_70B": BASE_DIR / "LLM_summaries/proposed_dataset/summaries_3370",
}

# Existing feature files (for 11 question features)
QUESTION_FEATURE_FILES = {
    "llama_3.1_8B": Path("CONFIGURE_ME/features/text_features_llama3.1_8B.csv"),
    "llama_3.1_70B": Path("CONFIGURE_ME/features/text_features_llama3.1_70B.csv"),
    "llama_3.3_70B": Path("CONFIGURE_ME/features/text_features_llama3.3_70B.csv"),
}

# No quality filters - use all patients
QUALITY_FILTERS = {
    "no_filter": None,
}

ID_COL, Y_COL = "patient_id", "depressed"
ORIGINAL_EXCLUDE = set()  # participant IDs to exclude (configure for your data)

# DepRoBERTa configuration
# Model is cached at CONFIGURE_ME/storage/models/hf_cache
CACHE_DIR = Path("CONFIGURE_ME/storage/models/hf_cache")
BASE_DEPROBERTA_MODEL = "rafalposwiata/deproberta-large-v1"  # DepRoBERTa for depression detection
DEPROBERTA_EPOCHS = 10  # Increased from 5 for better performance within 24h
DEPROBERTA_BATCH_SIZE = 32
DEPROBERTA_LR_CLASSIFIER = 5e-6
DEPROBERTA_LR_ENCODER = 1e-5
DEPROBERTA_MAX_LENGTH = 512

# Classifier hyperparameter grids
GRIDS = {
    "LogisticRegression": {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    "SVC": {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    "RandomForest": {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    "DecisionTree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    "ExtraTrees": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "GradientBoosting": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    },
}

if LIGHTGBM_AVAILABLE:
    GRIDS["LightGBM"] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, -1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 30]
    }

if CATBOOST_AVAILABLE:
    GRIDS["CatBoost"] = {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [3, 5, 7],
        'l2_leaf_reg': [1, 3, 5],
        'border_count': [32, 64, 128]
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_quality_ids(filter_path):
    if filter_path is None or not filter_path.exists():
        return set()
    with open(filter_path, 'r') as f:
        return {int(line.strip()) for line in f if line.strip()}

def get_exclude_ids(quality_filter):
    """Get set of patient IDs to exclude based on quality filter."""
    exclude_ids = ORIGINAL_EXCLUDE.copy()
    
    if quality_filter == "inverse_good":
        good_ids = load_quality_ids(Path("CONFIGURE_ME/quality_filters/nisqa_good_quality_ids.txt"))
        very_good_ids = load_quality_ids(Path("CONFIGURE_ME/quality_filters/nisqa_very_good_quality_ids.txt"))
        exclude_ids.update(good_ids | very_good_ids)
    elif quality_filter is not None:
        exclude_ids.update(load_quality_ids(quality_filter))
    
    return exclude_ids

# ============================================================================
# DEPROBERTA FINE-TUNING
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
    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer.encode_plus(
            self.texts[idx], add_special_tokens=True, max_length=self.max_length,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def load_summaries_for_patients(llama_model: str, patient_ids: List[int]) -> Dict[int, str]:
    """Load LLM summaries for given patient IDs."""
    summary_dir = LLM_SUMMARY_DIRS[llama_model]
    summaries = {}
    
    for pid in patient_ids:
        # Try different filename patterns
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
    """Map PHQ-8/9 score to categorical label."""
    if score >= 14: return 0  # Severe
    if score >= 7: return 1   # Moderate
    return 2  # Not Depressed / Mild

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
    
    # Load PHQ-9 labels for fine-tuning DepRoBERTa (3-class categorical)
    labels_df = pd.read_csv(PHQ_LABELS_CSV)
    
    # PHQ-9 score column
    phq_col = 'PHQ9-Score'
    labels_df[phq_col] = pd.to_numeric(labels_df[phq_col], errors='coerce')
    labels_df.dropna(subset=[phq_col], inplace=True)
    labels_df['label'] = labels_df[phq_col].apply(map_phq_to_label)
    
    # Filter to training patients
    train_df = labels_df[labels_df['patient_id'].isin(train_patient_ids)].copy()
    
   # Load summaries for training patients
    logging.info(f"Loading summaries for {len(train_df)} training patients...")
    summaries = load_summaries_for_patients(llama_model, train_df['patient_id'].tolist())
    
    # Filter to patients with summaries
    train_df['text'] = train_df['patient_id'].map(summaries)
    train_df = train_df.dropna(subset=['text'])
    
    logging.info(f"Training samples: {len(train_df)}")
    logging.info(f"Class distribution: {train_df['label'].value_counts().to_dict()}")
    
    if len(train_df) < 10:
        logging.error("Too few training samples!")
        raise ValueError("Insufficient training data")
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_subset, val_subset = train_test_split(
        train_df, test_size=config.validation_size, 
        stratify=train_df['label'], random_state=config.random_seed
    )
    
    # Create datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, cache_dir=str(CACHE_DIR))
    
    train_dataset = ClinicalTextDataset(
        train_subset['text'].tolist(),
        train_subset['label'].tolist(),
        tokenizer,
        config.max_token_length
    )
    val_dataset = ClinicalTextDataset(
        val_subset['text'].tolist(),
        val_subset['label'].tolist(),
        tokenizer,
        config.max_token_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_path, num_labels=config.num_labels, cache_dir=str(CACHE_DIR)
    )
    model.to(device)
    
    # Configure optimizer (freeze most layers)
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    num_layers = model.roberta.config.num_hidden_layers
    layers_to_unfreeze = range(num_layers - config.last_n_layers_to_unfreeze, num_layers)
    for name, param in model.named_parameters():
        if any(f"encoder.layer.{i}." in name for i in layers_to_unfreeze):
            param.requires_grad = True
    
    optimizer_grouped_parameters = [
        {"params": model.classifier.parameters(), "lr": config.lr_classifier},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "encoder.layer" in n], "lr": config.lr_encoder},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        
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
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                total_val_loss += outputs.loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save model
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logging.info(f"Model saved to {save_dir}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logging.info(f"Fine-tuning complete. Best val loss: {best_val_loss:.4f}")
    return save_dir

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_deproberta_probabilities(
    llama_model: str,
    fold: int,
    patient_ids: List[int],
    model_dir: Path
) -> pd.DataFrame:
    """Extract 3 probability features using fold-specific DepRoBERTa model."""
    
    logging.info(f"\nExtracting DepRoBERTa probabilities: {llama_model}, Fold {fold}")
    logging.info(f"Model: {model_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Load summaries for all patients
    summaries = load_summaries_for_patients(llama_model, patient_ids)
    
    # Extract probabilities
    results = []
    for pid in tqdm(patient_ids, desc="Extracting probabilities"):
        summary = summaries.get(pid)
        if summary is None:
            # Missing summary - use zeros
            results.append({
                'patient_id': pid,
                'prob_severe': 0.0,
                'prob_moderate': 0.0,
                'prob_not_depression': 0.0
            })
            continue
        
        # Tokenize and predict
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
            'prob_not_depression': float(probs[2])
        })
    
    return pd.DataFrame(results)

def load_question_features(llama_model: str) -> pd.DataFrame:
    """Load existing 11 question features (already leakage-free)."""
    feature_file = QUESTION_FEATURE_FILES[llama_model]
    df = pd.read_csv(feature_file)
    
    # Extract question columns
    q_cols = [f'q{i}' for i in range(1, 12)]
    return df[['patient_id'] + q_cols]

def combine_features(prob_df: pd.DataFrame, question_df: pd.DataFrame) -> pd.DataFrame:
    """Combine 3 probability features + 11 question features."""
    combined = prob_df.merge(question_df, on='patient_id', how='inner')
    
    # Ensure correct column order
    cols = ['patient_id', 'prob_severe', 'prob_moderate', 'prob_not_depression'] + \
           [f'q{i}' for i in range(1, 12)]
    
    return combined[cols]

# ============================================================================
# CLASSIFICATION
# ============================================================================

def make_estimator(name):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=10000, random_state=42)
    elif name == "SVC":
        return SVC(probability=True, random_state=42)
    elif name == "RandomForest":
        return RandomForestClassifier(random_state=42)
    elif name == "XGBoost":
        return XGBClassifier(objective="binary:logistic", eval_metric='logloss', random_state=42, use_label_encoder=False)
    elif name == "AdaBoost":
        return AdaBoostClassifier(random_state=42)
    elif name == "DecisionTree":
        return DecisionTreeClassifier(random_state=42)
    elif name == "KNN":
        return KNeighborsClassifier()
    elif name == "ExtraTrees":
        return ExtraTreesClassifier(random_state=42)
    elif name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42)
    elif name == "MLP":
        return MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
    elif name == "LightGBM" and LIGHTGBM_AVAILABLE:
        return LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
    elif name == "CatBoost" and CATBOOST_AVAILABLE:
        return CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    else:
        raise ValueError(f"Unknown or unavailable model: {name}")

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

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(llama_model: str, quality_name: str, quality_filter):
    """Run CV experiment with integrated DepRoBERTa fine-tuning."""
    
    print(f"\n{'='*70}")
    print(f"Experiment: {llama_model} + {quality_name}")
    print(f"{'='*70}")
    
    # Load BINARY labels for classification task
    labels_df = pd.read_csv(BINARY_LABELS_CSV)  # Contains 'patient_id' and 'depressed' (0/1)
    
    # No need to create labels - already binary in classi_labels.csv
    # labels_df already has 'depressed' column with 0/1 values
    
    # Apply quality filter
    exclude_ids = get_exclude_ids(quality_filter)
    labels_df = labels_df[~labels_df['patient_id'].isin(exclude_ids)]
    
    # Get all patient IDs and labels
    all_patient_ids = labels_df['patient_id'].values
    y = labels_df[Y_COL].values
    
    print(f"Total patients: {len(all_patient_ids)}")
    print(f"Excluded: {len(exclude_ids)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    if len(np.unique(y)) < 2 or len(y) < 50:
        print("⚠️  Insufficient data - skipping")
        return None
    
    # Load question features (already leakage-free)
    question_features = load_question_features(llama_model)
    
    # Outer CV
    results = []
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    deproberta_config = DepRoBERTaConfig()
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(all_patient_ids, y)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}")
        print(f"{'='*70}")
        
        train_patient_ids = all_patient_ids[train_idx].tolist()
        test_patient_ids = all_patient_ids[test_idx].tolist()
        
        print(f"Train: {len(train_patient_ids)} patients")
        print(f"Test: {len(test_patient_ids)} patients")
        
        # STEP 1: Fine-tune DepRoBERTa on training data
        model_save_path = MODEL_SAVE_DIR / llama_model / f"fold_{fold}"
        
        if not model_save_path.exists():
            try:
                fine_tune_deproberta_for_fold(
                    llama_model=llama_model,
                    fold=fold,
                    train_patient_ids=train_patient_ids,
                    save_dir=model_save_path,
                    config=deproberta_config
                )
            except Exception as e:
                logging.error(f"Fine-tuning failed for fold {fold}: {e}")
                continue
        else:
            logging.info(f"Using existing model: {model_save_path}")
        
        # STEP 2: Extract probabilities for ALL patients using fold-specific model
        prob_df = extract_deproberta_probabilities(
            llama_model=llama_model,
            fold=fold,
            patient_ids=all_patient_ids.tolist(),
            model_dir=model_save_path
        )
        
        # Save extracted features to CSV for later analysis
        features_output_dir = FEATURES_SAVE_DIR / llama_model / quality_name
        features_output_dir.mkdir(parents=True, exist_ok=True)
        features_csv = features_output_dir / f"fold_{fold}_probabilities.csv"
        prob_df.to_csv(features_csv, index=False)
        logging.info(f"Saved extracted features to {features_csv}")
        
        # STEP 3: Combine with question features
        combined_features = combine_features(prob_df, question_features)
        
        # Merge with labels
        fold_data = combined_features.merge(labels_df[['patient_id', Y_COL]], on='patient_id')
        
        # Extract features for this fold
        feature_cols = [c for c in fold_data.columns if c not in ['patient_id', Y_COL]]
        X_fold = fold_data[feature_cols].values
        y_fold = fold_data[Y_COL].values
        
        # Map patient IDs to indices
        pid_to_idx = {pid: i for i, pid in enumerate(fold_data['patient_id'].values)}
        train_fold_idx = [pid_to_idx[pid] for pid in train_patient_ids if pid in pid_to_idx]
        test_fold_idx = [pid_to_idx[pid] for pid in test_patient_ids if pid in pid_to_idx]
        
        X_train, X_test = X_fold[train_fold_idx], X_fold[test_fold_idx]
        y_train, y_test = y_fold[train_fold_idx], y_fold[test_fold_idx]
        
        # Scale features for better classifier performance
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Feature matrix: {X_fold.shape}")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # STEP 4: Run classification with all models
        for model_name in GRIDS.keys():
            print(f"\n  {model_name}:")
            
            try:
                model = make_estimator(model_name)
                
                # Inner CV for hyperparameter tuning
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + fold)
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=GRIDS[model_name],
                    cv=inner_cv,
                    scoring="f1",
                    n_jobs=1,
                    refit=True
                )
                gs.fit(X_train, y_train)
                
                # Evaluate on test set
                best_model = gs.best_estimator_
                y_pred = best_model.predict(X_test)
                y_score = get_scores(best_model, X_test)
                
                metrics = compute_metrics(y_test, y_score, y_pred)
                
                results.append({
                    "llama_model": llama_model,
                    "quality_filter": quality_name,
                    "classifier": model_name,
                    "fold": fold,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_features": X_fold.shape[1],
                    **metrics
                })
                
                print(f"    F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}")
                
            except Exception as e:
                logging.error(f"Classification failed for {model_name}: {e}")
                continue
    
    return results

def main():
    print("="*70)
    print("MAXIMAL TEXT CLASSIFICATION - CV-AWARE DEPROBERTA")
    print("="*70)
    
    set_seed(42)
    
    n_classifiers = len(GRIDS)
    print(f"\nTesting {n_classifiers} classifiers")
    print(f"3 Llama models × 4 Quality filters × {n_classifiers} classifiers × 5 folds")
    print(f"Total experiments: {3 * 4 * n_classifiers * 5}")
    
    all_results = []
    total = len(LLM_SUMMARY_DIRS) * len(QUALITY_FILTERS)
    completed = 0
    
    for llama_model in LLM_SUMMARY_DIRS.keys():
        for quality_name, quality_filter in QUALITY_FILTERS.items():
            completed += 1
            print(f"\n[{completed}/{total}] {llama_model} + {quality_name}")
            
            results = run_experiment(llama_model, quality_name, quality_filter)
            if results:
                all_results.extend(results)
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = OUTPUT_DIR / "maximal_text_classification_cv_deproberta_results.csv"
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        # Aggregate per configuration
        agg_df = df.groupby(['llama_model', 'quality_filter', 'classifier']).agg({
            'f1': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std']
        }).round(3)
        
        print("\n📊 TOP 15 CONFIGURATIONS BY F1:")
        top_configs = agg_df.sort_values(('f1', 'mean'), ascending=False).head(15)
        print(top_configs)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    print(f"\n✓ Completed {len(all_results)} experiments")

if __name__ == "__main__":
    main()
