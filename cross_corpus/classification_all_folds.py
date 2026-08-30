"""
Cross-Corpus Depression Classification — All Folds
=======================================================================
Addresses reviewer concern: "The cross-corpus results depend on a single
selected split of the proposed dataset."

Runs the cross-corpus classification for all 5 outer CV folds and reports:
  - Mean ± SD across folds
  - Worst-case (min F1) fold

Uses pre-computed DepRoBERTa probabilities from CSV — no model reloading.
Classifier grids and audio feature selection EXACTLY match classification.py
in the public repo (cross_corpus/classification.py).

Usage:
  python classification_all_folds.py --llm llama_3.3_70B
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_utils import load_folds  # canonical shared CV splits (cv_folds.json)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier; LGBM = True
except ImportError:
    LGBM = False
try:
    from catboost import CatBoostClassifier; CB = True
except ImportError:
    CB = False

# ── Paths (replace every CONFIGURE_ME/... with your local path) ───────────
FEATURES_BASE = Path("CONFIGURE_ME/proposed_dataset/features_output")
STORAGE_BASE  = Path("CONFIGURE_ME/storage")
CROSS_DIR     = Path("CONFIGURE_ME/cross_corpus_data/E-DAIC_data")

# Pre-computed per-fold DepRoBERTa probabilities (no model reloading at run time)
PROPOSED_PROBS_DIR = STORAGE_BASE / "deproberta_features"
EDAIC_PROBS_DIR    = STORAGE_BASE / "deproberta_edaic_features"

LLM_CODES = {"llama_3.1_8B": "3108", "llama_3.1_70B": "3170", "llama_3.3_70B": "3370"}

PROPOSED_Q_FEATS = {llm: Path(f"CONFIGURE_ME/features/text_features_{code}.csv")
                    for llm, code in LLM_CODES.items()}
EDAIC_Q_FEATS    = {llm: CROSS_DIR / f"Llama{code}_combined_text_r2_case1.csv"
                    for llm, code in LLM_CODES.items()}

PROPOSED_CLASSI  = Path("CONFIGURE_ME/labels/classi_labels.csv")        # SCID-5-CV binary labels
PROPOSED_PHQ8    = Path("CONFIGURE_ME/labels/regress_labels_phq8.csv")  # PHQ-8 (Setting 2)
EDAIC_LABELS     = CROSS_DIR / "edaic_labels.csv"

PROPOSED_AUDIO = {
    "egemaps_diarised":     FEATURES_BASE / "proposed_dataset_egemapsv02_diarised/proposed_dataset_egemapsv02_diarised.csv",
    "egemaps_non_diarised": FEATURES_BASE / "proposed_dataset_egemapsv02_non_diarised/proposed_dataset_egemapsv02_non_diarised.csv",
    "wav2vec_diarised":     FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_diarised/proposed_dataset_xlsr53_multilingual_diarised.csv",
    "wav2vec_non_diarised": FEATURES_BASE / "proposed_dataset_xlsr53_multilingual_non_diarised/proposed_dataset_xlsr53_multilingual_non_diarised.csv",
}
EDAIC_AUDIO = {
    "egemaps":  FEATURES_BASE / "edaic_egemapsv02/edaic_egemapsv02.csv",
    "xlsr_eng": FEATURES_BASE / "edaic_xlsr_english_partitioned/edaic_xlsr_english_partitioned.csv",
}

EDAIC_SPLIT = FEATURES_BASE / "splits" / "edaic_official_split.json"
OUTPUT_DIR  = STORAGE_BASE / "results" / "cross_corpus_all_folds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature selection constants — identical to classification.py ───────────────
EGEMAPS_K      = 30
W2V_PRESEL_K   = 200
W2V_MM_PCA_N   = 50
W2V_MM_POSTK   = 15
REDUND_THRESH  = 0.70
PCA_COMPONENTS = [10, 20, 50, 100]

# ── Classifier grids — identical to classification.py ─────────────────────────
GRIDS = {
    "LogisticRegression": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__solver": ["liblinear", "saga"]},
    "SVC":                {"clf__C": [0.1, 1, 10, 100], "clf__gamma": [1, 0.1, 0.01, 0.001], "clf__kernel": ["rbf", "poly", "sigmoid"]},
    "RandomForest":       {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "XGBoost":            {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 0.2], "clf__max_depth": [3, 5, 7]},
    "AdaBoost":           {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 1.0]},
    "DecisionTree":       {"clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
    "KNN":                {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
    "ExtraTrees":         {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20]},
    "GradientBoosting":   {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7]},
    "MLP":                {"clf__hidden_layer_sizes": [(50,), (100,), (50, 50)], "clf__activation": ["relu", "tanh"], "clf__alpha": [0.0001, 0.001, 0.01]},
}
if LGBM: GRIDS["LightGBM"] = {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [3, 5, 7, -1]}
if CB:   GRIDS["CatBoost"]  = {"clf__iterations": [50, 100, 200], "clf__learning_rate": [0.01, 0.05, 0.1], "clf__depth": [3, 5, 7]}


def make_classifier(name):
    m = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=10000, random_state=42),
        "SVC":         lambda: SVC(probability=True, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(random_state=42),
        "XGBoost":      lambda: XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42),
        "AdaBoost":     lambda: AdaBoostClassifier(random_state=42),
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
        "KNN":          lambda: KNeighborsClassifier(),
        "ExtraTrees":   lambda: ExtraTreesClassifier(random_state=42),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=42),
        "MLP": lambda: MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
    }
    if LGBM: m["LightGBM"] = lambda: LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
    if CB:   m["CatBoost"]  = lambda: CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    return m[name]()


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return s if s.ndim == 1 else s[:, 1]
    return model.predict(X).astype(float)


# ── Audio feature selection helpers — identical to classification.py ───────────
def select_audio_features(X_tr, y_tr, X_te, k=30, feature_names=None):
    k = min(k, X_tr.shape[1])
    sel = SelectKBest(f_classif, k=k)
    X_tr_new = sel.fit_transform(X_tr, y_tr)
    X_te_new = sel.transform(X_te)
    sel_names = ([feature_names[i] for i in sel.get_support(indices=True)]
                 if feature_names else [f"feat_{i}" for i in sel.get_support(indices=True)])
    return X_tr_new, X_te_new, sel_names


def remove_redundant_features(X_tr, X_te, feature_names, threshold=0.70):
    if X_tr.shape[1] <= 1:
        return X_tr, X_te, feature_names
    corr = np.corrcoef(X_tr.T)
    np.fill_diagonal(corr, 0)
    drop = set()
    for i in range(X_tr.shape[1]):
        if i in drop: continue
        for j in range(i + 1, X_tr.shape[1]):
            if j in drop: continue
            if abs(corr[i, j]) > threshold:
                drop.add(j)
    keep = [i for i in range(X_tr.shape[1]) if i not in drop]
    return X_tr[:, keep], X_te[:, keep], [feature_names[i] for i in keep]


# ── Evaluation — identical logic to classification.py ─────────────────────────
def evaluate_modality(X_tr, y_tr, X_te, y_te, modality, direction, label, llm, fold):
    rows = []
    cv3 = StratifiedKFold(3, shuffle=True, random_state=43)
    for name, grid in GRIDS.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_classifier(name))])
            gs = GridSearchCV(pipe, grid, cv=cv3, scoring="f1", n_jobs=1, refit=True)
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_
            yp = best.predict(X_te)
            ys = get_scores(best, X_te)
            rows.append({
                "fold": fold, "direction": direction, "label": label, "llm": llm,
                "modality": modality, "model": name,
                "best_params": str(gs.best_params_),
                "ap":        average_precision_score(y_te, ys),
                "auc":       roc_auc_score(y_te, ys),
                "bal_acc":   balanced_accuracy_score(y_te, yp),
                "f1":        f1_score(y_te, yp, zero_division=0),
                "precision": precision_score(y_te, yp, zero_division=0),
                "recall":    recall_score(y_te, yp, zero_division=0),
            })
            log.info(f"    {name:<20} F1={rows[-1]['f1']:.3f}  AUC={rows[-1]['auc']:.3f}")
        except Exception as e:
            log.warning(f"    {name} FAILED: {e}")
    return rows


def fuse_multimodal(X_tr_audio, X_te_audio, X_tr_text, X_te_text,
                    y_tr, y_te, modality, direction, label, llm, fold):
    """Fusion — identical to classification.py fuse_multimodal."""
    n_pca = min(W2V_MM_PCA_N, X_tr_audio.shape[0] - 1, X_tr_audio.shape[1])
    sc_a  = StandardScaler()
    pca   = PCA(n_components=n_pca, random_state=42)
    X_tr_ap = pca.fit_transform(sc_a.fit_transform(X_tr_audio))
    X_te_ap = pca.transform(sc_a.transform(X_te_audio))
    k = min(W2V_MM_POSTK, X_tr_ap.shape[1])
    sel = SelectKBest(f_classif, k=k)
    X_tr_as = sel.fit_transform(X_tr_ap, y_tr)
    X_te_as = sel.transform(X_te_ap)
    sc_t = StandardScaler()
    X_tr_mm = np.hstack([X_tr_as, sc_t.fit_transform(X_tr_text)])
    X_te_mm = np.hstack([X_te_as, sc_t.transform(X_te_text)])
    return evaluate_modality(X_tr_mm, y_tr, X_te_mm, y_te,
                             modality, direction, label, llm, fold)


# ── Feature loading ────────────────────────────────────────────────────────────
def load_proposed_text(pids, llm, fold):
    probs = (pd.read_csv(PROPOSED_PROBS_DIR / llm / "no_filter" / f"fold_{fold}_probabilities.csv")
             .rename(columns={"patient_id": "pid"}))
    q_df  = (pd.read_csv(PROPOSED_Q_FEATS[llm])
             .rename(columns={"patient_id": "pid"})
             [["pid"] + [f"q{i}" for i in range(1, 12)]])
    merged = probs.merge(q_df, on="pid").pipe(lambda d: d[d["pid"].isin(pids)])
    fcols = ["prob_severe", "prob_moderate", "prob_not_depression"] + [f"q{i}" for i in range(1, 12)]
    return merged.set_index("pid"), fcols


def load_edaic_text(pids, llm):
    probs = (pd.read_csv(EDAIC_PROBS_DIR / f"edaic_{llm}_train_split_probs.csv")
             .rename(columns={"participant_id": "pid"}))
    q_df  = (pd.read_csv(EDAIC_Q_FEATS[llm])
             .rename(columns={"patient_id": "pid", "participant_id": "pid"})
             [["pid"] + [f"q{i}" for i in range(1, 12)]])
    merged = probs.merge(q_df, on="pid").pipe(lambda d: d[d["pid"].isin(pids)])
    fcols = ["prob_severe", "prob_moderate", "prob_not_depression"] + [f"q{i}" for i in range(1, 12)]
    return merged.set_index("pid"), fcols


def load_audio_df(path):
    df = pd.read_csv(path)
    if "patient_id" in df.columns:
        df = df.rename(columns={"patient_id": "pid"})
    elif "participant_id" in df.columns:
        df = df.rename(columns={"participant_id": "pid"})
    elif "file_id" in df.columns:
        df["pid"] = df["file_id"].astype(str).str.extract(r"(\d+)").astype(int)
        df = df.drop(columns=["file_id"])
    df["pid"] = df["pid"].astype(int)
    meta = {"pid", "n_segments"}
    fcols = [c for c in df.columns if c not in meta]
    return df[["pid"] + fcols].set_index("pid"), fcols


def get_labels(pids, dataset, label_setting):
    if dataset == "proposed":
        cls_df = pd.read_csv(PROPOSED_CLASSI).rename(columns={"patient_id": "pid"})
        if label_setting == "scid":
            lmap = cls_df.set_index("pid")["depressed"]
        else:
            # Setting 2: PHQ-8 >= 10 for proposed corpus (matches E-DAIC PHQ-8 threshold)
            # PHQ-8 file: comma-separated, columns patient_id,PHQ8-Score
            phq8_df = pd.read_csv(PROPOSED_PHQ8)
            phq8_df.columns = ["pid", "PHQ8"]
            phq8_df["pid"] = pd.to_numeric(phq8_df["pid"], errors="coerce")
            phq8_df["PHQ8"] = pd.to_numeric(phq8_df["PHQ8"], errors="coerce")
            phq8_df = phq8_df.dropna(subset=["pid", "PHQ8"])
            phq8_df["pid"] = phq8_df["pid"].astype(int)
            lmap = (phq8_df.set_index("pid")["PHQ8"] >= 10).astype(int)
    else:
        df = pd.read_csv(EDAIC_LABELS).rename(columns={"participant_id": "pid"})
        lmap = df.set_index("pid")["depressed"]
    return np.array([lmap.get(p, np.nan) for p in pids])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True,
                        choices=["llama_3.1_8B", "llama_3.1_70B", "llama_3.3_70B"])
    parser.add_argument("--labels", nargs="+", default=["scid", "phq8_10"],
                        choices=["scid", "phq8_10"],
                        help="Label settings to run (default: both)")
    args = parser.parse_args()
    llm = args.llm

    # CANONICAL SHARED SPLITS: read the exact same proposed-side folds used by
    # the within-corpus experiments (cv_folds.json). Single source of truth.
    fold_splits = load_folds()

    # E-DAIC fixed split
    with open(EDAIC_SPLIT) as f:
        edaic_sp = json.load(f)
    edaic_train = [int(x) for x in edaic_sp["train_ids"]]
    edaic_test  = [int(x) for x in edaic_sp["test_ids"]]

    # Pre-load audio DataFrames once (same files for all folds)
    prop_audio, edaic_audio = {}, {}
    for key, path in PROPOSED_AUDIO.items():
        if path.exists():
            prop_audio[key], _ = load_audio_df(path)
            log.info(f"Loaded proposed audio {key}: {prop_audio[key].shape}")
    for key, path in EDAIC_AUDIO.items():
        if path.exists():
            edaic_audio[key], _ = load_audio_df(path)
            log.info(f"Loaded E-DAIC audio {key}: {edaic_audio[key].shape}")

    all_results = []

    for fold, (prop_train, prop_test) in enumerate(fold_splits):
        log.info(f"\n{'='*70}\nFold {fold}  |  proposed_train={len(prop_train)}, "
                 f"proposed_test={len(prop_test)}\n{'='*70}")

        for label in args.labels:
            for direction in ["proposed_to_edaic", "edaic_to_proposed"]:

                if direction == "proposed_to_edaic":
                    tr_pids, tr_ds = prop_train, "proposed"
                    te_pids, te_ds = edaic_test,  "edaic"
                else:
                    tr_pids, tr_ds = edaic_train, "edaic"
                    te_pids, te_ds = prop_test,   "proposed"

                y_tr_all = get_labels(tr_pids, tr_ds, label)
                y_te_all = get_labels(te_pids, te_ds, label)
                ok_tr = ~np.isnan(y_tr_all); ok_te = ~np.isnan(y_te_all)
                tr_pids_v = [p for p, v in zip(tr_pids, ok_tr) if v]
                te_pids_v = [p for p, v in zip(te_pids, ok_te) if v]
                y_tr = y_tr_all[ok_tr].astype(int)
                y_te = y_te_all[ok_te].astype(int)

                log.info(f"\ndir={direction}  label={label}  "
                         f"train={len(tr_pids_v)}({sum(y_tr)}+)  "
                         f"test={len(te_pids_v)}({sum(y_te)}+)")

                # ── TEXT ──────────────────────────────────────────────────────
                log.info("  → TEXT")
                if direction == "proposed_to_edaic":
                    tr_txt, tcols = load_proposed_text(tr_pids_v, llm, fold)
                    te_txt, _     = load_edaic_text(te_pids_v, llm)
                else:
                    tr_txt, tcols = load_edaic_text(tr_pids_v, llm)
                    te_txt, _     = load_proposed_text(te_pids_v, llm, fold)

                tr_txt_pids = [p for p in tr_pids_v if p in tr_txt.index]
                te_txt_pids = [p for p in te_pids_v if p in te_txt.index]
                y_tr_t = get_labels(tr_txt_pids, tr_ds, label).astype(int)
                y_te_t = get_labels(te_txt_pids, te_ds, label).astype(int)
                X_tr_t = tr_txt.loc[tr_txt_pids, tcols].values
                X_te_t = te_txt.loc[te_txt_pids, tcols].values

                res = evaluate_modality(X_tr_t, y_tr_t, X_te_t, y_te_t,
                                        f"text", direction, label, llm, fold)
                all_results.extend(res)

                # ── AUDIO (eGeMAPS) — no_fs and selectk30 ────────────────────
                tr_egem_keys = [k for k in prop_audio if "egemaps" in k] if tr_ds == "proposed" else ["egemaps"]
                te_egem_keys = ["egemaps"] if te_ds == "edaic" else [k for k in prop_audio if "egemaps" in k]
                tr_egem_src  = prop_audio if tr_ds == "proposed" else edaic_audio
                te_egem_src  = edaic_audio if te_ds == "edaic" else prop_audio

                for tk in tr_egem_keys:
                    for sk in te_egem_keys:
                        if tk not in tr_egem_src or sk not in te_egem_src:
                            continue
                        tr_adf = tr_egem_src[tk]
                        te_adf = te_egem_src[sk]
                        common = sorted(set(tr_adf.columns) & set(te_adf.columns))
                        if not common: continue
                        tr_ids_a = [p for p in tr_pids_v if p in tr_adf.index]
                        te_ids_a = [p for p in te_pids_v if p in te_adf.index]
                        X_tra = np.nan_to_num(tr_adf.loc[tr_ids_a, common].values.astype(float))
                        X_tea = np.nan_to_num(te_adf.loc[te_ids_a, common].values.astype(float))
                        y_tra = get_labels(tr_ids_a, tr_ds, label).astype(int)
                        y_tea = get_labels(te_ids_a, te_ds, label).astype(int)
                        base  = f"egemaps_{tk}_to_{sk}"

                        log.info(f"  → AUDIO {base} no_fs")
                        res = evaluate_modality(X_tra, y_tra, X_tea, y_tea,
                                                f"{base}_no_fs", direction, label, llm, fold)
                        all_results.extend(res)

                        log.info(f"  → AUDIO {base} selectk{EGEMAPS_K}")
                        X_tr_sk, X_te_sk, _ = select_audio_features(
                            X_tra, y_tra, X_tea, k=EGEMAPS_K, feature_names=common)
                        res = evaluate_modality(X_tr_sk, y_tra, X_te_sk, y_tea,
                                                f"{base}_selectk{EGEMAPS_K}", direction, label, llm, fold)
                        all_results.extend(res)

                        # eGeMAPS multimodal fusion
                        tr_both = [p for p in tr_ids_a if p in tr_txt.index]
                        te_both = [p for p in te_ids_a if p in te_txt.index]
                        if tr_both and te_both:
                            X_tra_mm = np.nan_to_num(tr_adf.loc[tr_both, common].values.astype(float))
                            X_tea_mm = np.nan_to_num(te_adf.loc[te_both, common].values.astype(float))
                            X_trt_mm = tr_txt.loc[tr_both, tcols].values
                            X_tet_mm = te_txt.loc[te_both, tcols].values
                            y_tr_mm  = get_labels(tr_both, tr_ds, label).astype(int)
                            y_te_mm  = get_labels(te_both, te_ds, label).astype(int)
                            log.info(f"  → FUSION {base}")
                            res = fuse_multimodal(X_tra_mm, X_tea_mm, X_trt_mm, X_tet_mm,
                                                  y_tr_mm, y_te_mm,
                                                  f"fusion_{base}", direction, label, llm, fold)
                            all_results.extend(res)

                # ── AUDIO (wav2vec / XLSR) — PCA variants + selectk+dedup ────
                tr_w2v_keys = [k for k in prop_audio if "wav2vec" in k] if tr_ds == "proposed" else ["xlsr_eng"]
                te_w2v_keys = ["xlsr_eng"] if te_ds == "edaic" else [k for k in prop_audio if "wav2vec" in k]
                tr_w2v_src  = prop_audio if tr_ds == "proposed" else edaic_audio
                te_w2v_src  = edaic_audio if te_ds == "edaic" else prop_audio

                for tk in tr_w2v_keys:
                    for sk in te_w2v_keys:
                        if tk not in tr_w2v_src or sk not in te_w2v_src:
                            continue
                        tr_adf = tr_w2v_src[tk]
                        te_adf = te_w2v_src[sk]
                        common = sorted(set(tr_adf.columns) & set(te_adf.columns))
                        if not common: continue
                        tr_ids_a = [p for p in tr_pids_v if p in tr_adf.index]
                        te_ids_a = [p for p in te_pids_v if p in te_adf.index]
                        X_tra_r = np.nan_to_num(tr_adf.loc[tr_ids_a, common].values.astype(float))
                        X_tea_r = np.nan_to_num(te_adf.loc[te_ids_a, common].values.astype(float))
                        y_tra   = get_labels(tr_ids_a, tr_ds, label).astype(int)
                        y_tea   = get_labels(te_ids_a, te_ds, label).astype(int)
                        base    = f"w2v_{tk}_to_{sk}"

                        sc_raw = StandardScaler()
                        X_tr_sc = sc_raw.fit_transform(X_tra_r)
                        X_te_sc = sc_raw.transform(X_tea_r)

                        # PCA-only variants
                        for n_pca in PCA_COMPONENTS:
                            eff = min(n_pca, X_tr_sc.shape[0] - 1, X_tr_sc.shape[1])
                            pca = PCA(n_components=eff, random_state=42)
                            X_tr_pca = pca.fit_transform(X_tr_sc)
                            X_te_pca = pca.transform(X_te_sc)
                            mod = f"{base}_pca{n_pca}_no_fs"
                            log.info(f"  → AUDIO {mod}")
                            res = evaluate_modality(X_tr_pca, y_tra, X_te_pca, y_tea,
                                                    mod, direction, label, llm, fold)
                            all_results.extend(res)

                        # selectk200 + dedup + PCA variants
                        k_pre = min(W2V_PRESEL_K, X_tra_r.shape[1])
                        X_tr_sk, X_te_sk, sk_names = select_audio_features(
                            X_tra_r, y_tra, X_tea_r, k=k_pre, feature_names=common)
                        X_tr_rd, X_te_rd, rd_names = remove_redundant_features(
                            X_tr_sk, X_te_sk, sk_names, threshold=REDUND_THRESH)
                        for n_pca in PCA_COMPONENTS:
                            eff = min(n_pca, X_tr_rd.shape[0] - 1, X_tr_rd.shape[1])
                            if eff < 1: continue
                            sc2 = StandardScaler()
                            pca2 = PCA(n_components=eff, random_state=42)
                            X_tr_p2 = pca2.fit_transform(sc2.fit_transform(X_tr_rd))
                            X_te_p2 = pca2.transform(sc2.transform(X_te_rd))
                            mod = f"{base}_selectk{k_pre}_dedup_pca{n_pca}"
                            log.info(f"  → AUDIO {mod}")
                            res = evaluate_modality(X_tr_p2, y_tra, X_te_p2, y_tea,
                                                    mod, direction, label, llm, fold)
                            all_results.extend(res)

                        # wav2vec multimodal fusion
                        tr_both = [p for p in tr_ids_a if p in tr_txt.index]
                        te_both = [p for p in te_ids_a if p in te_txt.index]
                        if tr_both and te_both:
                            X_tra_mm = np.nan_to_num(tr_adf.loc[tr_both, common].values.astype(float))
                            X_tea_mm = np.nan_to_num(te_adf.loc[te_both, common].values.astype(float))
                            X_trt_mm = tr_txt.loc[tr_both, tcols].values
                            X_tet_mm = te_txt.loc[te_both, tcols].values
                            y_tr_mm  = get_labels(tr_both, tr_ds, label).astype(int)
                            y_te_mm  = get_labels(te_both, te_ds, label).astype(int)
                            log.info(f"  → FUSION {base}")
                            res = fuse_multimodal(X_tra_mm, X_tea_mm, X_trt_mm, X_tet_mm,
                                                  y_tr_mm, y_te_mm,
                                                  f"fusion_{base}", direction, label, llm, fold)
                            all_results.extend(res)

        # Save incrementally after each fold
        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / f"all_folds_raw_{llm}.csv", index=False)
        log.info(f"Fold {fold} done — running total saved.")

    # ── Aggregate across folds ─────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    best_per_fold = (df.sort_values("f1", ascending=False)
                       .groupby(["fold", "direction", "label", "llm", "modality"])
                       .first().reset_index())

    agg = (best_per_fold
           .groupby(["direction", "label", "llm", "modality"])
           .agg(f1_mean=("f1","mean"), f1_std=("f1","std"), f1_min=("f1","min"),
                auc_mean=("auc","mean"), auc_std=("auc","std"),
                rec_mean=("recall","mean"), rec_std=("recall","std"),
                n_folds=("fold","count"))
           .reset_index()
           .sort_values(["direction","label","modality"]))

    agg.to_csv(OUTPUT_DIR / f"all_folds_aggregated_{llm}.csv", index=False)
    log.info(f"\nAggregated results → {OUTPUT_DIR / f'all_folds_aggregated_{llm}.csv'}")

    for direction in ["proposed_to_edaic", "edaic_to_proposed"]:
        for label in args.labels:
            sub = agg[(agg["direction"]==direction) & (agg["label"]==label)]
            if sub.empty: continue
            log.info(f"\n{direction}  |  label={label}")
            log.info(sub[["modality","f1_mean","f1_std","f1_min","auc_mean","rec_mean"]]
                       .to_string(index=False))


if __name__ == "__main__":
    main()
