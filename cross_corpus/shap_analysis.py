"""
Cross-Corpus SHAP Analysis — Best Multimodal Classification Models
===================================================================
For each cross-corpus direction (proposed_dataset_to_edaic, edaic_to_proposed_dataset):
  - Reads all standard-label (SCID) cross-corpus classification CSVs
  - Finds the best multimodal row (highest F1) for:
      (a) eGeMAPS + text
      (b) wav2vec/XLSR + text  (non-diarised)
  - Re-builds and re-fits the exact pipeline on training data
  - Generates high-quality SHAP beeswarm plots saved as PDF

SHAP output: ./shap_cross_corpus/
Usage:
  python cross_corpus_shap_best_multimodal.py
"""

import ast
import json
import logging
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================================
# PATHS
# ============================================================================
DATASET_ROOT = Path("CONFIGURE_ME/proposed_dataset")
FEATURES_BASE   = DATASET_ROOT / "features_output"
WOODY_BASE = Path("CONFIGURE_ME/storage")
CROSS_DIR  = Path("CONFIGURE_ME/cross_corpus_data")
RESULTS_DIR = FEATURES_BASE / "results" / "cross_corpus"

PROPOSED_SPLIT  = FEATURES_BASE / "splits" / "proposed_dataset_fold1_split.json"
EDAIC_SPLIT    = FEATURES_BASE / "splits" / "edaic_official_split.json"
EDAIC_LABELS   = Path("CONFIGURE_ME/labels/edaic_labels.csv")
PROPOSED_LABELS = Path("CONFIGURE_ME/labels/classi_labels.csv")

SHAP_OUTPUT_DIR = Path(__file__).parent / "shap_cross_corpus"
SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEPROBERTA_MAX_LENGTH = 512

LLM_CODES = {
    "llama_3.1_8B":  "3108",
    "llama_3.1_70B": "3170",
    "llama_3.3_70B": "3370",
}

# ============================================================================
# AUDIO SOURCE PATHS
# ============================================================================
PROPOSED_AUDIO = {
    "egemaps_diarised":      FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    "egemaps_non_diarised":  FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    "wav2vec_diarised":      FEATURES_BASE / "proposed_dataset_german_xlsr_diarised/proposed_dataset_german_xlsr_diarised.csv",
    "wav2vec_non_diarised":  FEATURES_BASE / "proposed_dataset_german_xlsr_non_diarised/proposed_dataset_german_xlsr_non_diarised.csv",
    "xlsr53ml_diarised":     FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_diarised/proposed_dataset_xlsr53_multilingual_diarised.csv",
    "xlsr53ml_non_diarised": FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_non_diarised/proposed_dataset_xlsr53_multilingual_non_diarised.csv",
}
EDAIC_AUDIO = {
    "egemaps_partitioned":       FEATURES_BASE / "edaic_egemapsv02/edaic_egemapsv02.csv",
    "xlsr_english_partitioned":  FEATURES_BASE / "edaic_xlsr_english_partitioned/edaic_xlsr_english_partitioned.csv",
    "xlsr53ml_partitioned":      FEATURES_BASE / "edaic_xlsr53_multilingual/edaic_xlsr53_multilingual.csv",
}

# ============================================================================
# FEATURE NAME MAPPINGS
# ============================================================================
# eGeMAPS LLD abbreviations
LLD_SHORT = {
    "F0semitoneFrom27.5Hz": "F0",
    "F1amplitudeLogRelF0":  "F1 Amp (LogRel)",
    "F1bandwidth":          "F1 BW",
    "F1frequency":          "F1 Freq",
    "F2amplitudeLogRelF0":  "F2 Amp (LogRel)",
    "F2bandwidth":          "F2 BW",
    "F2frequency":          "F2 Freq",
    "F3amplitudeLogRelF0":  "F3 Amp (LogRel)",
    "F3bandwidth":          "F3 BW",
    "F3frequency":          "F3 Freq",
    "HNRdBACF":             "HNR",
    "Loudness":             "Loudness",
    "alphaRatio":           "Alpha Ratio",
    "hammarbergIndex":      "Hammarberg Idx",
    "jitterLocal":          "Jitter",
    "logRelF0-H1-A3":       "H1\u2013A3 Ratio",
    "logRelF0-H1-H2":       "H1\u2013H2 Ratio",
    "mfcc1":                "MFCC 1",
    "mfcc2":                "MFCC 2",
    "mfcc3":                "MFCC 3",
    "mfcc4":                "MFCC 4",
    "shimmerLocaldB":       "Shimmer (dB)",
    "slope0-500":           "Slope [0\u2013500 Hz]",
    "slope500-1500":        "Slope [500\u20131500 Hz]",
    "spectralFlux":         "Spectral Flux",
}

STAT_SHORT = {
    "iqr":      "IQR",
    "kurtosis": "Kurt.",
    "max":      "Max",
    "mean":     "Mean",
    "median":   "Median",
    "min":      "Min",
    "skewness": "Skew.",
    "std":      "Std.",
    "amean":    "Mean",
    "stddev":   "Std.",
}

TEXT_FEAT_NAMES = {
    "prob_severe":         "DepRoBERTa: Severe",
    "prob_moderate":       "DepRoBERTa: Moderate",
    "prob_not_depression": "DepRoBERTa: Non-dep.",
    **{f"q{i}": f"LLM Q{i}" for i in range(1, 12)},
}

# Sorted LLDs by length (longest first) for unambiguous prefix matching
_LLD_SORTED = sorted(LLD_SHORT.items(), key=lambda x: -len(x[0]))


def clean_feature_name(name: str) -> str:
    """Return a short, readable version of an eGeMAPS/text/PCA feature name."""
    # Text features
    if name in TEXT_FEAT_NAMES:
        return TEXT_FEAT_NAMES[name]
    # PCA components (1-indexed for display)
    m = re.match(r"^PC(\d+)$", name)
    if m:
        return f"XLSR PC{int(m.group(1)) + 1}"
    # eGeMAPS LLD: {LLD}_sma3nz_{stat}  or  {LLD}_sma3_{stat}
    for lld, short in _LLD_SORTED:
        if name.startswith(lld):
            rest = name[len(lld):]
            rest = re.sub(r"^_sma3(nz)?_", "_", rest)
            stat = rest.lstrip("_")
            stat_clean = STAT_SHORT.get(stat, stat)
            return f"{short}: {stat_clean}"
    return name


# ============================================================================
# CLASSIFIER GRIDS  (same as cross_corpus_classification_v2_llm.py)
# ============================================================================
GRIDS = {
    "LogisticRegression": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__solver": ["liblinear", "saga"]},
    "SVC":                {"clf__C": [0.1, 1, 10, 100], "clf__gamma": [1, 0.1, 0.01, 0.001], "clf__kernel": ["rbf", "poly", "sigmoid"]},
    "RandomForest":       {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "AdaBoost":           {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 1.0]},
    "DecisionTree":       {"clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "KNN":                {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
    "ExtraTrees":         {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20]},
    "GradientBoosting":   {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7]},
    "MLP":                {"clf__hidden_layer_sizes": [(50,), (100,), (50, 50)], "clf__activation": ["relu", "tanh"], "clf__alpha": [0.0001, 0.001, 0.01]},
}
try:
    from lightgbm import LGBMClassifier
    GRIDS["LightGBM"] = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1]}
    LGBM = True
except ImportError:
    LGBM = False
try:
    from catboost import CatBoostClassifier
    GRIDS["CatBoost"] = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}
    CB = True
except ImportError:
    CB = False


def make_classifier(name):
    m = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=10000, random_state=42),
        "SVC":                lambda: SVC(probability=True, random_state=42),
        "RandomForest":       lambda: RandomForestClassifier(random_state=42),
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


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================
def load_audio_df(path):
    df = pd.read_csv(path)
    if "patient_id" in df.columns:
        df = df.rename(columns={"patient_id": "pid"})
    elif "participant_id" in df.columns:
        df = df.rename(columns={"participant_id": "pid"})
    elif "file_id" in df.columns:
        df["pid"] = df["file_id"].astype(str).str.extract(r"(\d+)").astype(int)
        df = df.drop(columns=["file_id"])
    meta = ["pid", "n_segments"]
    feat_cols = [c for c in df.columns if c not in meta]
    return df[["pid"] + feat_cols], feat_cols


def load_proposed_dataset_summaries(pids, llm):
    code = LLM_CODES[llm]
    summary_dir = Path(f"CONFIGURE_ME/LLM_summaries/proposed_dataset/summaries_{code}")
    out = {}
    for pid in pids:
        for suf in ["_transcription.json", "_whisper.json"]:
            p = summary_dir / f"{pid:03d}{suf}"
            if p.exists():
                try:
                    with open(p) as f:
                        s = json.load(f).get("summary", "").strip()
                    if s:
                        out[pid] = s
                        break
                except Exception:
                    pass
    return out


def load_edaic_summaries(pids, llm):
    code = LLM_CODES[llm]
    summary_dir = Path(f"CONFIGURE_ME/LLM_summaries/edaic/summaries_{code}")
    out = {}
    for pid in pids:
        p = summary_dir / f"{pid}.json"
        if p.exists():
            try:
                with open(p) as f:
                    s = json.load(f).get("summary", "").strip()
                if s:
                    out[pid] = s
            except Exception:
                pass
    return out


def extract_deproberta_probs(pids, model_dir, summaries, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    rows = []
    for pid in pids:
        text = summaries.get(pid)
        if text is None:
            rows.append([pid, 0.0, 0.0, 0.0])
            continue
        inputs = tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=DEPROBERTA_MAX_LENGTH
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
        rows.append([pid, float(probs[0]), float(probs[1]), float(probs[2])])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["pid", "prob_severe", "prob_moderate", "prob_not_depression"])


def build_text_features(pids, dataset, llm, device):
    code = LLM_CODES[llm]
    if dataset == "proposed_dataset":
        summaries = load_proposed_dataset_summaries(pids, llm)
        model_dir = WOODY_BASE / "deproberta_cv_models" / llm / "fold_1"
        q_df = pd.read_csv(Path(f"CONFIGURE_ME/features/text_features_llama_{code}.csv")).rename(columns={"patient_id": "pid"})
    else:
        summaries = load_edaic_summaries(pids, llm)
        model_dir = WOODY_BASE / "deproberta_edaic_models" / llm / "train_split"
        q_df = pd.read_csv(CROSS_DIR / f"E-DAIC_data/Llama{code}_combined_text_r2_case1.csv")
        for id_col in ["participant_id", "patient_id"]:
            if id_col in q_df.columns:
                q_df = q_df.rename(columns={id_col: "pid"})
                break
    prob_df = extract_deproberta_probs(pids, model_dir, summaries, device)
    q_cols = [f"q{i}" for i in range(1, 12)]
    q_df = q_df[["pid"] + q_cols]
    merged = prob_df.merge(q_df, on="pid", how="inner")
    feat_cols = ["prob_severe", "prob_moderate", "prob_not_depression"] + q_cols
    return merged, feat_cols


def get_scid_labels(pids, dataset, edaic_labels_df, proposed_dataset_labels_df):
    if dataset == "proposed_dataset":
        idx = proposed_dataset_labels_df.set_index("patient_id")["depressed"]
        return np.array([float(idx.get(p, np.nan)) for p in pids])
    else:
        idx = edaic_labels_df.set_index("participant_id")["depressed"]
        return np.array([float(idx.get(p, np.nan)) for p in pids])


# ============================================================================
# FEATURE RECONSTRUCTION
# ============================================================================
EGEMAPS_K    = 30
W2V_PRESEL_K = 200
W2V_MM_PCA_N = 50
W2V_MM_POSTK = 15
REDUND_THRESH = 0.70


def parse_modality(modality):
    """Extract (audio_type, train_audio_key, test_audio_key, condition) from modality string.

    Handles both naming conventions:
      Standard CSV:  multimodal_text+egemaps_{tr}_to_{te}_{cond}
                     multimodal_text+w2v_{tr}_to_{te}_{cond}
      xlsr53ml CSV:  multimodal_w2v_{tr}_to_{te}[_{cond}]
    """
    # Normalise prefix
    rest = modality
    if rest.startswith("multimodal_text+egemaps_"):
        audio_type = "egemaps"
        rest2 = rest[len("multimodal_text+egemaps_"):]
    elif rest.startswith("multimodal_text+w2v_"):
        audio_type = "w2v"
        rest2 = rest[len("multimodal_text+w2v_"):]
    elif rest.startswith("multimodal_w2v_"):
        audio_type = "w2v"
        rest2 = rest[len("multimodal_w2v_"):]
    else:
        raise ValueError(f"Unknown modality prefix: {modality}")

    # Split on "_to_"
    if "_to_" not in rest2:
        raise ValueError(f"Cannot find '_to_' separator in: {rest2}")
    parts = rest2.split("_to_")
    train_key = parts[0]
    te_rest   = parts[1]

    # Match test_key (longest first)
    test_key_candidates = list(EDAIC_AUDIO.keys()) + list(PROPOSED_AUDIO.keys())
    test_key = None
    condition = ""
    for tk in sorted(test_key_candidates, key=len, reverse=True):
        if te_rest.startswith(tk):
            test_key = tk
            condition = te_rest[len(tk):].lstrip("_")
            break
    if test_key is None:
        m = re.match(r"([a-z0-9_]*partitioned)(.*)", te_rest)
        if m:
            test_key  = m.group(1)
            condition = m.group(2).lstrip("_")
        else:
            raise ValueError(f"Cannot parse test_key from: {te_rest}")

    # Normalise underscore variants (xlsr53_ml ↔ xlsr53ml)
    def _norm(k):
        return k.replace("xlsr53_ml", "xlsr53ml")

    train_key = _norm(train_key)
    test_key  = _norm(test_key)

    return audio_type, train_key, test_key, condition


def finalise_audio_features(audio_type, condition, X_tr_audio, X_te_audio, X_tr_txt, y_train, common_cols):
    """Apply feature selection on audio; return (X_tr_red, X_te_red, feat_names)."""

    if audio_type == "egemaps" and "no_fs" in condition:
        # No selection — scale and return all eGeMAPS features
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr_audio)
        Xte_s = scaler.transform(X_te_audio)
        logging.info(f"  eGeMAPS no_fs: {len(common_cols)} features (all retained)")
        return Xtr_s, Xte_s, list(common_cols)

    if audio_type == "egemaps" and "selectk" in condition:
        k = min(EGEMAPS_K, X_tr_audio.shape[1])
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr_audio)
        Xte_s = scaler.transform(X_te_audio)
        sel = SelectKBest(f_classif, k=k).fit(Xtr_s, y_train)
        Xtr_sel = sel.transform(Xtr_s)
        Xte_sel = sel.transform(Xte_s)
        names_sel = [common_cols[i] for i in sel.get_support(indices=True)]
        # Redundancy pruning vs text
        keep = np.ones(Xtr_sel.shape[1], dtype=bool)
        for ai in range(Xtr_sel.shape[1]):
            for ti in range(X_tr_txt.shape[1]):
                r, _ = stats.pearsonr(Xtr_sel[:, ai], X_tr_txt[:, ti])
                if abs(r) > REDUND_THRESH:
                    keep[ai] = False
                    break
        Xtr_sel = Xtr_sel[:, keep]
        Xte_sel = Xte_sel[:, keep]
        names_sel = [n for n, k_ in zip(names_sel, keep) if k_]
        logging.info(f"  eGeMAPS selectk: {int(keep.sum())} features retained after pruning")
        return Xtr_sel, Xte_sel, names_sel

    if audio_type == "w2v":
        # Already StandardScaler-transformed before this call
        k_pre = min(W2V_PRESEL_K, X_tr_audio.shape[1])
        sel_pre = SelectKBest(f_classif, k=k_pre).fit(X_tr_audio, y_train)
        Xtr_pre = sel_pre.transform(X_tr_audio)
        Xte_pre = sel_pre.transform(X_te_audio)
        eff = min(W2V_MM_PCA_N, Xtr_pre.shape[0] - 1, Xtr_pre.shape[1])
        pca = PCA(n_components=eff, random_state=42)
        Xtr_pca = pca.fit_transform(Xtr_pre)
        Xte_pca = pca.transform(Xte_pre)
        k_post = min(W2V_MM_POSTK, Xtr_pca.shape[1])
        sel_post = SelectKBest(f_classif, k=k_post).fit(Xtr_pca, y_train)
        Xtr_sk = sel_post.transform(Xtr_pca)
        Xte_sk = sel_post.transform(Xte_pca)
        pca_names = [f"PC{i}" for i in sel_post.get_support(indices=True)]
        keep = np.ones(Xtr_sk.shape[1], dtype=bool)
        for ai in range(Xtr_sk.shape[1]):
            for ti in range(X_tr_txt.shape[1]):
                r, _ = stats.pearsonr(Xtr_sk[:, ai], X_tr_txt[:, ti])
                if abs(r) > REDUND_THRESH:
                    keep[ai] = False
                    break
        Xtr_sk = Xtr_sk[:, keep]
        Xte_sk = Xte_sk[:, keep]
        pca_names = [n for n, k_ in zip(pca_names, keep) if k_]
        logging.info(f"  wav2vec/xlsr: {int(keep.sum())} PCA dims retained after pruning")
        return Xtr_sk, Xte_sk, pca_names

    raise ValueError(f"Unknown audio_type='{audio_type}', condition='{condition}'")


# ============================================================================
# SHAP COMPUTATION
# ============================================================================
def _get_shap_values(clf, X_scaled):
    """
    Extract SHAP values (n_samples × n_features) for the positive class.
    Uses TreeExplainer for tree models; falls back to KernelExplainer.
    Returns (shap_values_2d, expected_value_float, explainer).
    """
    TREE_TYPES = (
        "RandomForest", "ExtraTrees", "GradientBoosting",
        "DecisionTree", "XGBoost", "LightGBM", "CatBoost",
    )
    clf_name = type(clf).__name__
    is_tree = any(t in clf_name for t in TREE_TYPES)

    if is_tree:
        try:
            # Use default TreeExplainer — compatible with all sklearn tree models
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(X_scaled)
            # Binary classification: sv is list [sv_c0, sv_c1] or ndarray (n, f, 2)
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            ev = expl.expected_value
            base = ev[1] if (isinstance(ev, (list, np.ndarray)) and len(ev) > 1) else float(ev)
            return sv, base, expl
        except Exception as e1:
            logging.warning(f"  TreeExplainer failed ({e1}), falling back to KernelExplainer")

    # Fallback: KernelExplainer (model-agnostic)
    bg = shap.sample(X_scaled, min(50, len(X_scaled)))
    expl = shap.KernelExplainer(lambda x: clf.predict_proba(x)[:, 1], bg)
    sv = expl.shap_values(X_scaled, nsamples=150)
    return sv, float(expl.expected_value), expl


# ============================================================================
# BEESWARM PLOT  (high-quality, publication-ready)
# ============================================================================
def generate_beeswarm(
    clf_fitted,
    X_train_raw,
    feature_names,
    title,
    save_path,
    max_display=20,
):
    """
    Generate a single high-quality SHAP beeswarm plot and save as PDF.

    Parameters
    ----------
    clf_fitted   : fitted sklearn Pipeline with 'scaler' and 'clf' steps
    X_train_raw  : raw (unscaled) training data, shape (n, d)
    feature_names: list of raw feature name strings
    title        : plot title string
    save_path    : Path or str — output PDF path
    max_display  : maximum number of features to show
    """
    try:
        scaler = clf_fitted.named_steps.get("scaler")
        clf    = clf_fitted.named_steps["clf"]
        X_sc   = scaler.transform(X_train_raw) if scaler is not None else X_train_raw

        # Clean feature names for display
        clean_names = [clean_feature_name(fn) for fn in feature_names]
        n_disp = min(max_display, len(clean_names))

        logging.info(f"  Computing SHAP values for {clf.__class__.__name__} "
                     f"({X_sc.shape[0]} samples, {X_sc.shape[1]} features)…")
        sv, base_val, _ = _get_shap_values(clf, X_sc)

        # Build shap.Explanation — pass raw (unscaled) values for interpretable colorbar
        expl = shap.Explanation(
            values=sv,
            base_values=np.full(len(sv), base_val),
            data=X_train_raw,          # original feature values → informative colour scale
            feature_names=clean_names,
        )

        # ---------- Publication-quality matplotlib settings ----------
        plt.rcParams.update({
            "font.family":       "sans-serif",
            "font.size":         11,
            "axes.titlesize":    12,
            "axes.labelsize":    11,
            "xtick.labelsize":   10,
            "ytick.labelsize":   10,
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })

        fig_h = max(7.0, n_disp * 0.46 + 3.0)
        plt.close("all")

        # SHAP 0.40+ beeswarm — creates its own figure internally
        shap.plots.beeswarm(
            expl,
            max_display=n_disp,
            show=False,
            color_bar=True,
        )

        fig = plt.gcf()
        fig.set_size_inches(13, fig_h)

        ax = fig.axes[0]

        # Symmetric x-axis so the zero line is visually centred
        max_abs = np.abs(sv).max()
        margin  = max_abs * 1.12          # 12 % padding on each side
        ax.set_xlim(-margin, margin)

        # Bold title above the plot
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12, loc="left",
                     wrap=True)

        # Slightly larger y-tick labels for readability
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=9)

        # Tighten layout and save at 300 DPI
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close("all")
        logging.info(f"  Beeswarm saved → {save_path}")

    except Exception as e:
        logging.warning(f"  Beeswarm generation failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    with open(PROPOSED_SPLIT) as f:
        emp_split = json.load(f)
    with open(EDAIC_SPLIT) as f:
        edaic_split = json.load(f)
    emp_train   = emp_split["train_ids"]
    emp_test    = emp_split["test_ids"]
    edaic_train = edaic_split["train_ids"]
    edaic_test  = edaic_split["test_ids"]

    edaic_labels_df   = pd.read_csv(EDAIC_LABELS)
    proposed_dataset_labels_df = pd.read_csv(PROPOSED_LABELS)

    # -----------------------------------------------------------------------
    # Load all SCID cross-corpus result CSVs
    #
    # cross_corpus_{direction}_v2.csv  (no LLM suffix) = Llama-3.3-70B results
    # cross_corpus_{direction}_v2_{llm}.csv             = Llama-3.1-8B / 3.1-70B
    # -----------------------------------------------------------------------
    logging.info("Loading result CSVs…")
    all_results = []
    for direction in ["proposed_dataset_to_edaic", "edaic_to_proposed_dataset"]:
        # L3.3-70B: stored in v2.csv (no LLM suffix) — these match the published table
        f_v2 = RESULTS_DIR / f"cross_corpus_{direction}_v2.csv"
        if f_v2.exists():
            df = pd.read_csv(f_v2)
            df["llm"] = "llama_3.3_70B"
            df["src"] = "v2_llama33"
            if "direction" not in df.columns:
                df["direction"] = direction
            all_results.append(df)
            logging.info(f"  Loaded L3.3-70B (v2.csv): {f_v2.name}")

        # L3.1-8B and L3.1-70B: stored in v2_{llm}.csv files
        for llm in ["llama_3.1_8B", "llama_3.1_70B"]:
            for suffix, src_tag in [("", "standard"), ("_xlsr53ml", "xlsr53ml")]:
                f = RESULTS_DIR / f"cross_corpus_{direction}_v2{suffix}_{llm}.csv"
                if f.exists():
                    df = pd.read_csv(f)
                    df["llm"] = llm
                    df["src"] = src_tag
                    if "direction" not in df.columns:
                        df["direction"] = direction
                    all_results.append(df)

    if not all_results:
        logging.error("No result CSVs found!")
        return

    all_df = pd.concat(all_results, ignore_index=True)
    mm_df = all_df[
        (all_df["label_setting"] == "scid") &
        (all_df["modality"].str.contains("multimodal", case=False, na=False))
    ].copy()
    logging.info(f"Total SCID multimodal rows: {len(mm_df)}")

    # -----------------------------------------------------------------------
    # For each direction × audio_type, find best model by F1.
    # Priority: Llama-3.3-70B (v2.csv) results first — these correspond to the
    # published table values.  Fall back to all LLMs if not available.
    # -----------------------------------------------------------------------
    def _best_row(pool, audio_pattern):
        """Return the highest-F1 row matching audio_pattern, or None."""
        sub = pool[pool["modality"].str.contains(audio_pattern, case=False, regex=True,
                                                   na=False)]
        return sub.nlargest(1, "f1").iloc[0] if len(sub) else None

    def _prefer_llama33(sub_all, sub_33, audio_pattern):
        """Return best row, preferring L3.3-70B; fall back to all LLMs."""
        row = _best_row(sub_33, audio_pattern)
        if row is None:
            row = _best_row(sub_all, audio_pattern)
        return row

    targets = []
    for direction in ["proposed_dataset_to_edaic", "edaic_to_proposed_dataset"]:
        sub         = mm_df[mm_df["direction"] == direction]
        sub_llama33 = sub[sub["llm"] == "llama_3.3_70B"]   # published-table models

        # (a) Best eGeMAPS + text  — prefer L3.3-70B
        best_eg = _prefer_llama33(sub, sub_llama33, "egemaps")
        if best_eg is not None:
            targets.append({
                "direction":        direction,
                "audio_type_label": "eGeMAPS",
                "modality":         best_eg["modality"],
                "model":            best_eg["model"],
                "llm":              best_eg["llm"],
                "f1":               best_eg["f1"],
                "best_params":      best_eg["best_params"],
            })
            logging.info(
                f"  [{direction}] Best eGeMAPS+text: {best_eg['modality']} | "
                f"{best_eg['model']} (LLM={best_eg['llm']}, F1={best_eg['f1']:.3f})"
            )

        # (b) Best wav2vec/XLSR + text  — prefer L3.3-70B
        best_w2v = _prefer_llama33(sub, sub_llama33, r"w2v|wav2vec|xlsr")
        if best_w2v is not None:
            targets.append({
                "direction":        direction,
                "audio_type_label": "wav2vec/XLSR",
                "modality":         best_w2v["modality"],
                "model":            best_w2v["model"],
                "llm":              best_w2v["llm"],
                "f1":               best_w2v["f1"],
                "best_params":      best_w2v["best_params"],
            })
            logging.info(
                f"  [{direction}] Best wav2vec/XLSR+text: {best_w2v['modality']} | "
                f"{best_w2v['model']} (LLM={best_w2v['llm']}, F1={best_w2v['f1']:.3f})"
            )

    if not targets:
        logging.error("No target models found!")
        return

    # -----------------------------------------------------------------------
    # For each target: build features, re-fit best model, generate beeswarm
    # -----------------------------------------------------------------------
    for t in targets:
        direction = t["direction"]
        llm       = t["llm"]
        modality  = t["modality"]
        clf_name  = t["model"]
        best_params = ast.literal_eval(t["best_params"])

        logging.info(f"\n{'='*70}")
        logging.info(f"SHAP: {direction} | {t['audio_type_label']} | {clf_name} | LLM={llm}")
        logging.info(f"Modality: {modality}")

        if direction == "proposed_dataset_to_edaic":
            train_pids, test_pids = emp_train,   edaic_test
            train_ds,  test_ds   = "proposed_dataset",    "edaic"
        else:
            train_pids, test_pids = edaic_train, emp_test
            train_ds,  test_ds   = "edaic",      "proposed_dataset"

        # --- Text features ---
        logging.info("Building text features…")
        try:
            train_text_df, txt_cols = build_text_features(train_pids, train_ds, llm, device)
            test_text_df,  _        = build_text_features(test_pids,  test_ds,  llm, device)
        except Exception as e:
            logging.error(f"  Text features failed: {e}")
            continue

        tr_pids_t = [p for p in train_pids if p in set(train_text_df["pid"])]
        te_pids_t = [p for p in test_pids  if p in set(test_text_df["pid"])]
        X_text_tr = np.nan_to_num(train_text_df.set_index("pid").loc[tr_pids_t, txt_cols].values.astype(float))
        X_text_te = np.nan_to_num(test_text_df.set_index("pid").loc[te_pids_t, txt_cols].values.astype(float))

        y_tr = get_scid_labels(tr_pids_t, train_ds, edaic_labels_df, proposed_dataset_labels_df)
        y_te = get_scid_labels(te_pids_t, test_ds,  edaic_labels_df, proposed_dataset_labels_df)
        mtr = ~np.isnan(y_tr); mte = ~np.isnan(y_te)
        tr_pids_t = [p for p, v in zip(tr_pids_t, mtr) if v]
        te_pids_t = [p for p, v in zip(te_pids_t, mte) if v]
        X_text_tr = X_text_tr[mtr]; y_tr = y_tr[mtr].astype(int)
        X_text_te = X_text_te[mte]; y_te = y_te[mte].astype(int)

        # --- Parse modality → audio pipeline ---
        try:
            audio_type, train_audio_key, test_audio_key, condition = parse_modality(modality)
        except Exception as e:
            logging.error(f"  Cannot parse modality '{modality}': {e}")
            continue

        # --- Load raw audio ---
        tr_audio_paths = PROPOSED_AUDIO if train_ds == "proposed_dataset" else EDAIC_AUDIO
        te_audio_paths = EDAIC_AUDIO   if test_ds  == "edaic"   else PROPOSED_AUDIO

        try:
            tr_df, tr_cols = load_audio_df(tr_audio_paths[train_audio_key])
            te_df, te_cols = load_audio_df(te_audio_paths[test_audio_key])
        except KeyError as e:
            logging.error(f"  Audio key not found: {e}")
            continue

        common_cols = sorted(set(tr_cols) & set(te_cols))
        tr_ids = [p for p in tr_pids_t if p in set(tr_df["pid"])]
        te_ids = [p for p in te_pids_t if p in set(te_df["pid"])]
        X_tr_audio = np.nan_to_num(tr_df.set_index("pid").loc[tr_ids, common_cols].values.astype(float))
        X_te_audio = np.nan_to_num(te_df.set_index("pid").loc[te_ids, common_cols].values.astype(float))

        # Align text to same pid order as audio
        tr_txt_idx = {p: i for i, p in enumerate(tr_pids_t)}
        te_txt_idx = {p: i for i, p in enumerate(te_pids_t)}
        X_tr_txt = X_text_tr[[tr_txt_idx[p] for p in tr_ids if p in tr_txt_idx]]
        X_te_txt = X_text_te[[te_txt_idx[p] for p in te_ids if p in te_txt_idx]]
        y_tr_a = y_tr[[tr_pids_t.index(p) for p in tr_ids if p in tr_txt_idx]]
        y_te_a = y_te[[te_pids_t.index(p) for p in te_ids if p in te_txt_idx]]

        # --- Pre-scale audio for w2v pipeline ---
        if audio_type == "w2v":
            sc_a = StandardScaler()
            X_tr_audio = sc_a.fit_transform(X_tr_audio)
            X_te_audio = sc_a.transform(X_te_audio)

        # --- Feature selection on audio ---
        try:
            X_tr_ared, X_te_ared, audio_feat_names = finalise_audio_features(
                audio_type, condition, X_tr_audio, X_te_audio, X_tr_txt, y_tr_a, common_cols
            )
        except Exception as e:
            logging.error(f"  Audio feature selection failed: {e}")
            continue

        if X_tr_ared.shape[1] == 0:
            logging.warning("  All audio features pruned — skipping")
            continue

        # --- Combine text + audio ---
        X_tr_mm = np.hstack([X_tr_txt, X_tr_ared])
        X_te_mm = np.hstack([X_te_txt, X_te_ared])
        feat_names = txt_cols + audio_feat_names
        logging.info(f"  Combined features: train={X_tr_mm.shape}, test={X_te_mm.shape}")

        # --- Re-fit best model ---
        logging.info(f"  Fitting {clf_name} with params: {best_params}")
        try:
            clf_params = {k.replace("clf__", ""): v for k, v in best_params.items()}
            clf = make_classifier(clf_name)
            clf.set_params(**clf_params)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X_tr_mm, y_tr_a)
            yp = pipe.predict(X_te_mm)
            f1 = f1_score(y_te_a, yp)
            logging.info(f"  Re-fit F1 on test: {f1:.3f} (original: {t['f1']:.3f})")
        except Exception as e:
            logging.error(f"  Model re-fit failed: {e}")
            continue

        # --- Build title and file name ---
        dir_label   = direction.replace("proposed_dataset", "ProposedDataset").replace("edaic", "E-DAIC").replace("_to_", " → ")
        audio_label = t["audio_type_label"]
        safe_dir    = direction.replace("_", "-")
        safe_audio  = t["audio_type_label"].lower().replace("/", "-").replace(" ", "_")
        safe_clf    = clf_name
        safe_llm    = llm.replace(".", "").replace("_", "-")
        base_name   = f"beeswarm_{safe_dir}_{safe_audio}_{safe_clf}_{safe_llm}"

        # Training-set beeswarm (original)
        title_train = (
            f"{dir_label}  |  {audio_label} + Text  ·  {clf_name}  ·  LLM: {llm}  [TRAIN]\n"
            f"F1 = {t['f1']:.3f}  |  {X_tr_mm.shape[0]} samples  ·  "
            f"{X_tr_mm.shape[1]} features (top {min(20, len(feat_names))} shown)"
        )
        generate_beeswarm(
            pipe, X_tr_mm, feat_names, title_train,
            SHAP_OUTPUT_DIR / f"{base_name}_train.pdf",
            max_display=20,
        )

        # Test-set beeswarm (target-domain view)
        title_test = (
            f"{dir_label}  |  {audio_label} + Text  ·  {clf_name}  ·  LLM: {llm}  [TEST]\n"
            f"F1 = {t['f1']:.3f}  |  {X_te_mm.shape[0]} samples  ·  "
            f"{X_te_mm.shape[1]} features (top {min(20, len(feat_names))} shown)"
        )
        generate_beeswarm(
            pipe, X_te_mm, feat_names, title_test,
            SHAP_OUTPUT_DIR / f"{base_name}_test.pdf",
            max_display=20,
        )

    logging.info(f"\n{'='*70}")
    logging.info(f"SHAP beeswarm analysis complete → {SHAP_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
