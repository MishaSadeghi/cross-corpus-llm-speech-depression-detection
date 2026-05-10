"""
E-DAIC Text-Based PHQ-8 Regression (Within-Dataset Evaluation)
-- Dev-Optimised DepRoBERTa variant --

Uses the official E-DAIC train / dev / test splits — NO cross-validation.
Pipeline:
  1. Fine-tune DepRoBERTa on E-DAIC TRAIN set, using the official DEV set
     as the validation set for early stopping (instead of a random 15% split).
     TEST SET IS NEVER TOUCHED DURING FINE-TUNING.
  2. Extract 3 probability features for all participants
  3. Load 11 question-score features (label-free, from LLM answers)
  4. GridSearchCV with PredefinedSplit (train=-1, dev=0) for HP tuning
  5. Retrain best model on train set → evaluate on dev and test

Results saved as edaic_text_regr_{llama_model}_devopt.csv to avoid
overwriting the original results.

Run one job per Llama model:
  python edaic_text_regression_devopt.py --llama_model llama_3.3_70B
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available")

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = Path("CONFIGURE_ME/repo")
CROSS_DIR   = Path("CONFIGURE_ME/cross_corpus_data")
EDAIC_LABELS = Path("CONFIGURE_ME/labels/edaic_labels.csv")

LLM_SUMMARY_DIRS = {
    "llama_3.1_8B":  Path("CONFIGURE_ME/LLM_summaries/edaic/summaries_llama3.1_8B"),
    "llama_3.1_70B": Path("CONFIGURE_ME/LLM_summaries/edaic/summaries_llama3.1_70B"),
    "llama_3.3_70B": Path("CONFIGURE_ME/LLM_summaries/edaic/summaries_llama3.3_70B"),
}

QUESTION_FEATURE_FILES = {
    "llama_3.1_8B":  Path("CONFIGURE_ME/features/edaic_text_features_llama3.1_8B.csv"),
    "llama_3.1_70B": Path("CONFIGURE_ME/features/edaic_text_features_llama3.1_70B.csv"),
    "llama_3.3_70B": Path("CONFIGURE_ME/features/edaic_text_features_llama3.3_70B.csv"),
}

# Results saved with _devopt suffix to avoid overwriting original results
OUTPUT_DIR = BASE_DIR / "EDAIC_evaluation/results/regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WOODY_BASE = Path("CONFIGURE_ME/storage")
# Saved under train_dev_split/ to distinguish from train-only model
MODEL_SAVE_DIR    = WOODY_BASE / "deproberta_edaic_models"
FEATURES_SAVE_DIR = WOODY_BASE / "deproberta_edaic_features"
CACHE_DIR         = Path("CONFIGURE_ME/storage/models/hf_cache")
for d in [MODEL_SAVE_DIR, FEATURES_SAVE_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEPROBERTA CONFIG  (identical to edaic_text_regression.py)
# ============================================================================
BASE_DEPROBERTA_MODEL    = "rafalposwiata/deproberta-large-v1"
DEPROBERTA_EPOCHS        = 10
DEPROBERTA_BATCH_SIZE    = 32
DEPROBERTA_LR_CLASSIFIER = 5e-6
DEPROBERTA_LR_ENCODER    = 1e-5
DEPROBERTA_MAX_LENGTH    = 512

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

# ============================================================================
# REGRESSOR GRIDS  (same as edaic_text_regression.py)
# ============================================================================
GRIDS = {
    "Ridge": {
        "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
    },
    "Lasso": {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]},
    "ElasticNet": {
        "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "SVR": {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["rbf", "poly", "sigmoid"],
        "epsilon": [0.1, 0.2],
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
        "loss": ["linear", "square", "exponential"],
    },
    "DecisionTree": {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    },
    "ExtraTrees": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    },
}
if LIGHTGBM_AVAILABLE:
    GRIDS["LightGBM"] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, -1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 30],
    }
if CATBOOST_AVAILABLE:
    GRIDS["CatBoost"] = {
        "iterations": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7],
        "l2_leaf_reg": [1, 3, 5],
        "border_count": [32, 64, 128],
    }

# ============================================================================
# HELPERS
# ============================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ClinicalTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts; self.labels = labels
        self.tokenizer = tokenizer; self.max_length = max_length

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.texts[idx], add_special_tokens=True, max_length=self.max_length,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

def map_phq_to_label(score: float) -> int:
    if score >= 14: return 0
    if score >= 7:  return 1
    return 2

def load_summaries(llama_model: str, patient_ids: List[int]) -> Dict[int, str]:
    summary_dir = LLM_SUMMARY_DIRS[llama_model]
    summaries = {}
    for pid in patient_ids:
        for fname in [f"{pid:03d}_whisper.json", f"{pid}_whisper.json",
                      f"{pid:03d}_transcription.json"]:
            p = summary_dir / fname
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding='utf-8'))
                    text = data.get("summary", "").strip()
                    if text:
                        summaries[pid] = text; break
                except Exception as e:
                    logging.warning(f"Error reading {p}: {e}")
    logging.info(f"Loaded {len(summaries)}/{len(patient_ids)} summaries")
    return summaries

# ============================================================================
# DEPROBERTA FINE-TUNING — KEY CHANGE vs. edaic_text_regression.py:
#   Uses the official E-DAIC dev split as the validation set for early
#   stopping rather than a random 15% carved from train.
#   TEST SET IS NEVER TOUCHED HERE.
# ============================================================================
def fine_tune_deproberta(llama_model: str,
                          train_ids: List[int],
                          dev_ids: List[int],
                          labels_df: pd.DataFrame,
                          config: DepRoBERTaConfig) -> Path:
    """Fine-tune DepRoBERTa on E-DAIC train split, validated on dev split."""
    save_dir = MODEL_SAVE_DIR / llama_model / "train_dev_split"
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_dir.exists() and any(save_dir.iterdir()):
        logging.info(f"Reusing existing dev-optimised DepRoBERTa model: {save_dir}")
        return save_dir

    logging.info(f"\n{'='*70}")
    logging.info(f"Fine-tuning DepRoBERTa (dev-optimised): {llama_model}")
    logging.info(f"  Train: {len(train_ids)} participants | Val (dev): {len(dev_ids)} participants")
    logging.info(f"{'='*70}")

    # --- Build train dataframe ---
    train_df = labels_df[labels_df['participant_id'].isin(train_ids)].copy()
    train_df['phq_num'] = pd.to_numeric(train_df['PHQ_score'], errors='coerce')
    train_df.dropna(subset=['phq_num'], inplace=True)
    train_df['label'] = train_df['phq_num'].apply(map_phq_to_label)
    train_summaries = load_summaries(llama_model, train_df['participant_id'].tolist())
    train_df['text'] = train_df['participant_id'].map(train_summaries)
    train_df = train_df.dropna(subset=['text'])
    logging.info(f"Train samples with text: {len(train_df)}")
    logging.info(f"Train class distribution: {train_df['label'].value_counts().to_dict()}")

    # --- Build dev (validation) dataframe ---
    dev_df = labels_df[labels_df['participant_id'].isin(dev_ids)].copy()
    dev_df['phq_num'] = pd.to_numeric(dev_df['PHQ_score'], errors='coerce')
    dev_df.dropna(subset=['phq_num'], inplace=True)
    dev_df['label'] = dev_df['phq_num'].apply(map_phq_to_label)
    dev_summaries = load_summaries(llama_model, dev_df['participant_id'].tolist())
    dev_df['text'] = dev_df['participant_id'].map(dev_summaries)
    dev_df = dev_df.dropna(subset=['text'])
    logging.info(f"Dev (val) samples with text: {len(dev_df)}")

    if len(train_df) < 10:
        raise ValueError("Insufficient training data")
    if len(dev_df) < 2:
        raise ValueError("Insufficient dev data for validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, cache_dir=str(CACHE_DIR))

    tr_ds  = ClinicalTextDataset(train_df['text'].tolist(), train_df['label'].tolist(),
                                  tokenizer, config.max_token_length)
    val_ds = ClinicalTextDataset(dev_df['text'].tolist(),   dev_df['label'].tolist(),
                                  tokenizer, config.max_token_length)
    tr_dl  = DataLoader(tr_ds,  batch_size=config.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_path, num_labels=config.num_labels, cache_dir=str(CACHE_DIR)
    ).to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    n_layers = model.roberta.config.num_hidden_layers
    for name, param in model.named_parameters():
        if any(f"encoder.layer.{i}." in name
               for i in range(n_layers - config.last_n_layers_to_unfreeze, n_layers)):
            param.requires_grad = True

    optimizer = AdamW([
        {"params": model.classifier.parameters(),            "lr": config.lr_classifier},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "encoder.layer" in n], "lr": config.lr_encoder},
    ])

    best_val_loss = float('inf'); no_improve = 0
    for epoch in range(config.epochs):
        model.train(); total_tr = 0
        for batch in tqdm(tr_dl, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward(); optimizer.step(); total_tr += out.loss.item()

        model.eval(); total_val = 0
        with torch.no_grad():
            for batch in val_dl:
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
                total_val += out.loss.item()

        avg_tr = total_tr / len(tr_dl); avg_val = total_val / len(val_dl)
        logging.info(f"Epoch {epoch+1}: Train={avg_tr:.4f}, Dev Val={avg_val:.4f}")
        if avg_val < best_val_loss:
            best_val_loss = avg_val; no_improve = 0
            model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
            logging.info(f"  Best model saved to {save_dir}")
        else:
            no_improve += 1
            if no_improve >= config.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch+1}"); break

    logging.info(f"Best dev val loss: {best_val_loss:.4f}"); return save_dir

# ============================================================================
# FEATURE EXTRACTION  (identical to edaic_text_regression.py)
# ============================================================================
def extract_deproberta_features(llama_model: str, patient_ids: List[int],
                                 model_dir: Path) -> pd.DataFrame:
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model    = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    summaries = load_summaries(llama_model, patient_ids)
    results = []
    for pid in tqdm(patient_ids, desc="Extracting DepRoBERTa probs"):
        s = summaries.get(pid)
        if s is None:
            results.append({'patient_id': pid, 'prob_severe': 0.0,
                             'prob_moderate': 0.0, 'prob_not_depression': 0.0})
            continue
        inputs = tokenizer(s, return_tensors="pt", padding=True,
                           truncation=True, max_length=DEPROBERTA_MAX_LENGTH).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
        results.append({'patient_id': pid, 'prob_severe': float(probs[0]),
                        'prob_moderate': float(probs[1]), 'prob_not_depression': float(probs[2])})
    return pd.DataFrame(results)

# ============================================================================
# REGRESSOR FACTORY  (same as edaic_text_regression.py)
# ============================================================================
def make_estimator(name):
    if name == "Ridge":          return Ridge()
    elif name == "Lasso":        return Lasso(max_iter=10000)
    elif name == "ElasticNet":   return ElasticNet(max_iter=10000)
    elif name == "SVR":          return SVR()
    elif name == "RandomForest": return RandomForestRegressor(random_state=42)
    elif name == "AdaBoost":     return AdaBoostRegressor(random_state=42)
    elif name == "DecisionTree": return DecisionTreeRegressor(random_state=42)
    elif name == "KNN":          return KNeighborsRegressor()
    elif name == "ExtraTrees":   return ExtraTreesRegressor(random_state=42)
    elif name == "GradientBoosting": return GradientBoostingRegressor(random_state=42)
    elif name == "MLP":          return MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
    elif name == "XGBoost":      return XGBRegressor(random_state=42)
    elif name == "LightGBM" and LIGHTGBM_AVAILABLE:
        return LGBMRegressor(random_state=42, verbose=-1, force_col_wise=True)
    elif name == "CatBoost" and CATBOOST_AVAILABLE:
        return CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
    else:
        raise ValueError(f"Unknown or unavailable model: {name}")

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2":   r2_score(y_true, y_pred),
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment(llama_model: str):
    print(f"\n{'='*70}")
    print(f"E-DAIC Text Regression (Dev-Optimised DepRoBERTa)  |  {llama_model}")
    print(f"{'='*70}")

    labels = pd.read_csv(EDAIC_LABELS)
    train_df = labels[labels['split'] == 'train']
    dev_df   = labels[labels['split'] == 'dev']
    test_df  = labels[labels['split'] == 'test']
    logging.info(f"train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")

    config = DepRoBERTaConfig()

    # STEP 1: Fine-tune DepRoBERTa — train on TRAIN, validate (early-stop) on DEV
    model_dir = fine_tune_deproberta(
        llama_model,
        train_df['participant_id'].tolist(),
        dev_df['participant_id'].tolist(),
        labels,
        config,
    )

    # STEP 2: Extract probs for ALL participants
    all_ids = labels['participant_id'].tolist()
    prob_df = extract_deproberta_features(llama_model, all_ids, model_dir)
    prob_df = prob_df.rename(columns={'patient_id': 'participant_id'})

    # Save extracted probs for inspection
    feat_path = FEATURES_SAVE_DIR / f"edaic_{llama_model}_train_dev_split_probs.csv"
    if not feat_path.exists():
        prob_df.to_csv(feat_path, index=False)

    # STEP 3: Load Q scores (q1-q11) from pre-computed file (label-free)
    q_df   = pd.read_csv(QUESTION_FEATURE_FILES[llama_model])
    Q_COLS = [f"q{i}" for i in range(1, 12)]
    q_df   = q_df[['patient_id'] + Q_COLS].rename(columns={'patient_id': 'participant_id'})

    merged = (labels
              .merge(prob_df, on='participant_id', how='left')
              .merge(q_df,    on='participant_id', how='left'))

    FEATURE_COLS = ['prob_severe', 'prob_moderate', 'prob_not_depression'] + Q_COLS
    Y_COL = 'PHQ_score'

    tr  = merged[merged['split'] == 'train']
    dev = merged[merged['split'] == 'dev']
    tst = merged[merged['split'] == 'test']

    X_train = tr[FEATURE_COLS].values;  y_train = tr[Y_COL].values.astype(float)
    X_dev   = dev[FEATURE_COLS].values; y_dev   = dev[Y_COL].values.astype(float)
    X_test  = tst[FEATURE_COLS].values; y_test  = tst[Y_COL].values.astype(float)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s   = scaler.transform(X_dev)
    X_test_s  = scaler.transform(X_test)

    X_traindev    = np.vstack([X_train_s, X_dev_s])
    y_traindev    = np.concatenate([y_train, y_dev])
    fold_indicator = np.concatenate([
        np.full(len(X_train_s), -1), np.zeros(len(X_dev_s), dtype=int)
    ])
    ps = PredefinedSplit(fold_indicator)

    results = []
    for reg_name in GRIDS:
        logging.info(f"\n--- {reg_name} ---")
        try:
            reg = make_estimator(reg_name)
            gs  = GridSearchCV(reg, GRIDS[reg_name], cv=ps,
                               scoring='neg_mean_absolute_error', n_jobs=1, refit=False)
            gs.fit(X_traindev, y_traindev)
            best_params = gs.best_params_
            logging.info(f"  Best params: {best_params}")

            reg = make_estimator(reg_name)
            reg.set_params(**best_params)
            reg.fit(X_train_s, y_train)

            for split_name, X_s, y_s in [("dev", X_dev_s, y_dev), ("test", X_test_s, y_test)]:
                y_pred = reg.predict(X_s)
                m = compute_metrics(y_s, y_pred)
                logging.info(f"  {split_name}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:.3f}")
                results.append({
                    'llama_model': llama_model, 'regressor': reg_name,
                    'split': split_name, 'best_params': str(best_params), **m
                })
        except Exception as e:
            import traceback
            logging.error(f"  ERROR in {reg_name}: {e}"); traceback.print_exc()

    # Save results with _devopt suffix to avoid overwriting original results
    out_path = OUTPUT_DIR / f"edaic_text_regr_{llama_model}_devopt.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    logging.info(f"\nResults saved to {out_path}")

    df_test = pd.DataFrame(results)
    df_test = df_test[df_test['split'] == 'test'].sort_values('mae')
    print("\n=== TEST SET RESULTS (sorted by MAE) ===")
    print(df_test[['regressor','mae','rmse','r2']].to_string(index=False))
    return df_test

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_model", required=True,
                        choices=["llama_3.1_8B", "llama_3.1_70B", "llama_3.3_70B"])
    args = parser.parse_args()
    set_seed(42)
    run_experiment(args.llama_model)
