#!/usr/bin/env python
"""
Cross-corpus SHAP beeswarm plots (two panels, one per transfer direction).
Interpretable eGeMAPS + text fusion with semantic feature names and styling:
  (a) E-DAIC -> Proposed : ANOVA k=30 + cPCA (|r|>0.70) ; Extra Trees (n=50, depth=10)
  (b) Proposed -> E-DAIC : no feature selection (214)    ; CatBoost (depth=3, iter=100, lr=0.05)
Median-performing fold per direction; SHAP over source-training data.
"""
import sys, json, warnings, logging, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classification_all_folds as cc
OUT = Path("CONFIGURE_ME/storage/results/shap"); OUT.mkdir(parents=True, exist_ok=True)
LLM = "llama_3.3_70B"; EGEMAPS_K = 30; REDUND = 0.70

# ---- semantic feature name maps ----
QMAP = {1:"general_wellbeing",2:"mood_changes",3:"sleep_issues",4:"concentration_issues",
        5:"loss_of_interest",6:"depression_history",7:"PTSD_diagnosis",8:"financial_problems",
        9:"social_withdrawal",10:"suicidal_thoughts",11:"military_service"}
TEXT_NAMES = {"prob_severe":"Severe depression","prob_moderate":"Moderate depression",
              "prob_not_depression":"Not depression", **{f"q{i}":f"Q{i}: {QMAP[i]}" for i in range(1,12)}}
LLD = {"F0semitoneFrom27.5Hz":"F0","F1amplitudeLogRelF0":"F1 Amp","F1bandwidth":"F1 BW","F1frequency":"F1 Freq",
"F2amplitudeLogRelF0":"F2 Amp","F2bandwidth":"F2 BW","F2frequency":"F2 Freq","F3amplitudeLogRelF0":"F3 Amp",
"F3bandwidth":"F3 BW","F3frequency":"F3 Freq","HNRdBACF":"HNR","Loudness":"Loudness","alphaRatio":"Alpha Ratio",
"hammarbergIndex":"Hammarberg Idx","jitterLocal":"Jitter","logRelF0-H1-A3":"H1–A3 Ratio",
"logRelF0-H1-H2":"H1–H2 Ratio","mfcc1":"MFCC 1","mfcc2":"MFCC 2","mfcc3":"MFCC 3","mfcc4":"MFCC 4",
"shimmerLocaldB":"Shimmer (dB)","slope0-500":"Slope [0–500 Hz]","slope500-1500":"Slope [500–1500 Hz]",
"spectralFlux":"Spectral Flux"}
STAT = {"iqr":"IQR","kurtosis":"Kurt.","max":"Max","mean":"Mean","median":"Median","min":"Min",
        "skewness":"Skew.","std":"Std.","amean":"Mean","stddev":"Std."}
_LLD = sorted(LLD.items(), key=lambda x:-len(x[0]))
def clean(name):
    if name in TEXT_NAMES: return TEXT_NAMES[name]
    for lld,short in _LLD:
        if name.startswith(lld):
            stat = re.sub(r"^_sma3(nz)?_","_", name[len(lld):]).lstrip("_")
            return f"{short}: {STAT.get(stat,stat)}"
    return name

folds = cc.load_folds()
edaic_sp = json.load(open(cc.EDAIC_SPLIT))
edaic_train=[int(x) for x in edaic_sp["train_ids"]]; edaic_test=[int(x) for x in edaic_sp["test_ids"]]
prop_eg,_=cc.load_audio_df(cc.PROPOSED_AUDIO["egemaps_non_diarised"])
edaic_eg,_=cc.load_audio_df(cc.EDAIC_AUDIO["egemaps"])

def assemble(direction, fold):
    prop_train, prop_test = folds[fold]
    if direction=="proposed_to_edaic":
        tr_ds,tr_pids,te_ds,te_pids="proposed",prop_train,"edaic",edaic_test
        tr_txt,tcols=cc.load_proposed_text(tr_pids,LLM,fold); te_txt,_=cc.load_edaic_text(te_pids,LLM)
        src_eg,tgt_eg=prop_eg,edaic_eg
    else:
        tr_ds,tr_pids,te_ds,te_pids="edaic",edaic_train,"proposed",prop_test
        tr_txt,tcols=cc.load_edaic_text(tr_pids,LLM); te_txt,_=cc.load_proposed_text(te_pids,LLM,fold)
        src_eg,tgt_eg=edaic_eg,prop_eg
    common=sorted(set(src_eg.columns)&set(tgt_eg.columns))
    tr=[p for p in tr_pids if p in tr_txt.index and p in src_eg.index]
    te=[p for p in te_pids if p in te_txt.index and p in tgt_eg.index]
    y_tr=cc.get_labels(tr,tr_ds,"scid").astype(int); y_te=cc.get_labels(te,te_ds,"scid").astype(int)
    Xtr_txt=tr_txt.loc[tr,tcols].values.astype(float); Xte_txt=te_txt.loc[te,tcols].values.astype(float)
    Xtr_eg=np.nan_to_num(src_eg.loc[tr,common].values.astype(float)); Xte_eg=np.nan_to_num(tgt_eg.loc[te,common].values.astype(float))
    return Xtr_txt,Xte_txt,Xtr_eg,Xte_eg,y_tr,y_te,list(tcols),list(common)

def build_features(direction, fold):
    Xtr_txt,Xte_txt,Xtr_eg,Xte_eg,y_tr,y_te,tcols,common=assemble(direction,fold)
    if direction=="edaic_to_proposed":     # (a) ANOVA k=30 + cPCA
        sc=StandardScaler().fit(Xtr_eg); Atr=sc.transform(Xtr_eg); Ate=sc.transform(Xte_eg)
        sel=SelectKBest(f_classif,k=min(EGEMAPS_K,Atr.shape[1])).fit(Atr,y_tr)
        Atr_s=sel.transform(Atr); Ate_s=sel.transform(Ate)
        names=[common[i] for i in sel.get_support(indices=True)]
        keep=np.ones(Atr_s.shape[1],bool)
        for ai in range(Atr_s.shape[1]):
            for ti in range(Xtr_txt.shape[1]):
                if abs(stats.pearsonr(Atr_s[:,ai],Xtr_txt[:,ti])[0])>REDUND: keep[ai]=False; break
        Atr_s=Atr_s[:,keep]; Ate_s=Ate_s[:,keep]; names=[n for n,k in zip(names,keep) if k]
    else:                                   # (b) no FS
        sc=StandardScaler().fit(Xtr_eg); Atr_s=sc.transform(Xtr_eg); Ate_s=sc.transform(Xte_eg); names=list(common)
    Xtr=np.hstack([Xtr_txt,Atr_s]); Xte=np.hstack([Xte_txt,Ate_s])
    return Xtr,Xte,y_tr,y_te,[clean(c) for c in tcols]+[clean(n) for n in names]

def model(direction):
    # Interpretable tree model used for both transfer directions.
    return ExtraTreesClassifier(n_estimators=200, random_state=42)

def sv_pos(clf,X,nf):
    a=np.array(shap.TreeExplainer(clf).shap_values(X))
    if a.ndim==3: a=a[...,1] if a.shape[-1]==2 else a[1]
    if a.shape[1]==nf+1: a=a[:,:nf]
    return a

# ---- per-direction median-performing fold ----
med={}
for d in ["edaic_to_proposed","proposed_to_edaic"]:
    f1={}
    for fold in range(len(folds)):
        Xtr,Xte,y_tr,y_te,names=build_features(d,fold)
        pipe=Pipeline([("sc",StandardScaler()),("clf",model(d))]).fit(Xtr,y_tr)
        f1[fold]=f1_score(y_te,pipe.predict(Xte),zero_division=0)
    order=sorted(f1,key=f1.get); mf=order[len(order)//2]; med[d]=(mf,f1[mf],f1)
    print(f"{d}: {dict((k,round(v,3)) for k,v in f1.items())} -> median fold {mf} F1={f1[mf]:.3f}")

# ---- native two-panel vector figure via shap.plots.beeswarm(ax=...) ----
plt.rcParams.update({"font.family":"sans-serif","font.size":13,"axes.titlesize":16,
    "axes.labelsize":14,"xtick.labelsize":12,"ytick.labelsize":13,
    "axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"ps.fonttype":42})
titles={"edaic_to_proposed":"(a) E-DAIC","proposed_to_edaic":"(b) EKSpression"}
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(26, 11))
for ax, d in zip(axes, ["edaic_to_proposed","proposed_to_edaic"]):
    mf,f1v,_=med[d]
    Xtr,Xte,y_tr,y_te,names=build_features(d,mf)
    pipe=Pipeline([("sc",StandardScaler()),("clf",model(d))]).fit(Xtr,y_tr)
    Xs=pipe.named_steps["sc"].transform(Xtr)
    sv=sv_pos(pipe.named_steps["clf"],Xs,Xtr.shape[1])
    expl=shap.Explanation(values=sv, base_values=np.zeros(len(sv)), data=Xtr, feature_names=names)
    shap.plots.beeswarm(expl, max_display=20, show=False, color_bar=True, ax=ax, plot_size=None)
    mx=np.abs(sv).max()*1.12; ax.set_xlim(-mx,mx)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=14)
    ax.set_title(titles[d], fontsize=16, loc="center", pad=12)
fig.subplots_adjust(wspace=0.42, left=0.13, right=0.97, top=0.93, bottom=0.08)
fig.savefig(OUT/"shap_faithful.pdf", bbox_inches="tight")           # pure vector
fig.savefig(OUT/"shap_faithful.png", dpi=200, bbox_inches="tight")  # crisp preview
print("saved:", OUT/"shap_faithful.pdf")
