# -*- coding: utf-8 -*-
"""G1: 框架重基准的三个缺口闭口（skill Step 1/6/7）
① Step 7 内验校准四件套（slope/intercept/ECE10/Brier + intercept-only 更新）
② Step 1 Riley pmsampsize 样本量核算（患者级口径 + EPPP）
③ Step 6 SMOTE 折内敏感性（vs class_weight 主法；HP 冻结不重调）
输出：results/g1_gap_closers.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

import duckdb
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm

R = "E:/TBI subtype/09_tbi_aki/results"
out = {"probe": "G1 gap closers (framework re-baseline)", "date": "2026-09-21"}

# ============ ① 内验校准四件套 ============
d = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet")
y, p = d.y.values, d.p.values
z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
cal = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
slope, intercept = float(cal.params[1]), float(cal.params[0])
dfc = pd.DataFrame({"y": y, "p": p})
dfc["b"] = pd.qcut(dfc.p, 10, duplicates="drop")
ece10 = float((dfc.groupby("b", observed=True).apply(
    lambda g: len(g) / len(dfc) * abs(g.y.mean() - g.p.mean()), include_groups=False)).sum())
brier = float(brier_score_loss(y, p))
p_io = 1 / (1 + np.exp(-(intercept + z)))
dfc2 = pd.DataFrame({"y": y, "p": p_io})
dfc2["b"] = pd.qcut(dfc2.p, 10, duplicates="drop")
ece_io = float((dfc2.groupby("b", observed=True).apply(
    lambda g: len(g) / len(dfc2) * abs(g.y.mean() - g.p.mean()), include_groups=False)).sum())
out["internal_calibration"] = {
    "slope": round(slope, 3), "intercept": round(intercept, 3),
    "ece10": round(ece10, 4), "brier": round(brier, 4),
    "brier_null": round(brier_score_loss(y, np.full_like(p, y.mean())), 4),
    "intercept_only_update_ece10": round(ece_io, 4)}
print("① 内验校准:", out["internal_calibration"])

# ============ ② pmsampsize（患者级；参数动态读自当前冻结模型——审计修正） ============
grid = pd.read_parquet(f"{R}/n3_m4_abi_grid.parquet")
pat_evt = grid.groupby("stay_id").label.max()
n_pat, n_evt = len(pat_evt), int(pat_evt.sum())
prev = n_evt / n_pat
# 患者级 AUROC 从当前 preds 实算（不再硬编码）
_pat = d.groupby("stay_id").agg(y=("y", "max"), p=("p", "max"))
pat_auc_live = float(roc_auc_score(_pat.y, _pat.p))
import joblib as _jl
_nfeat = len(_jl.load(f"{R}/n6_abi_gbm_frozen.joblib")["feats"])
from pmsampsize.pmsampsize import pmsampsize as _pms
ps = _pms(type="b", cstatistic=pat_auc_live, parameters=_nfeat, prevalence=prev)
out["pmsampsize"] = {
    "unit": "患者级（检查点聚类，保守口径）",
    "n_patients_grid": n_pat, "n_patients_event": n_evt, "prevalence": round(prev, 4),
    "cstatistic_input": round(pat_auc_live, 4), "parameters_input": int(_nfeat),
    "riley_min_n": int(ps["sample_size"]), "riley_min_events": int(ps["events"]),
    "actual_meets": bool(n_pat >= ps["sample_size"] and n_evt >= ps["events"]),
    "eppp_checkpoints": round(grid.label.sum() / _nfeat, 1)}
print("② pmsampsize:", out["pmsampsize"])

# ============ ③ SMOTE 敏感性（HP 冻结） ============
import joblib
import lightgbm as lgb
frozen = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")
feats = frozen["feats"]
F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
con = duckdb.connect()
adm = con.execute(f"""
SELECT c.stay_id, c.subject_id, a.admittime, p.anchor_year
FROM read_parquet('{R}/n1_m4_abi_cohort.parquet') c
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/admissions.csv.gz') a ON c.hadm_id = a.hadm_id
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz') p ON c.subject_id = p.subject_id
""").df()
pat2 = pd.read_csv("E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz", usecols=["subject_id", "anchor_year_group"])
_g = pat2["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
pat2["group_mid"] = (_g[0].astype(int) + _g[1].astype(int)) / 2.0
coh = adm.merge(pat2[["subject_id", "group_mid"]], on="subject_id", how="left")
coh["year"] = (coh.group_mid + (coh.admittime.dt.year - coh.anchor_year)).round().astype(int)
F = F.merge(coh[["stay_id", "year"]], on="stay_id", how="left")
tr = F[F.year <= 2017]
iv = F[(F.year >= 2020) & (F.year <= 2022)]
Xtr, ytr = tr[feats].values, tr.label.values
Xiv, yiv = iv[feats].values, iv.label.values
# SMOTE 不能处理 NaN → 训练集中位数插补（仅敏感性用；主法 NaN 原生）
med = np.nanmedian(Xtr, axis=0)
med = np.where(np.isfinite(med), med, 0.0)
Xtr_i, Xiv_i = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xiv), med, Xiv)
from imblearn.over_sampling import SMOTE
Xr, yr = SMOTE(random_state=42).fit_resample(Xtr_i, ytr)
m_s = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.02, num_leaves=127,
                         random_state=42, verbose=-1, n_jobs=-1)
m_s.fit(Xr, yr)
p_sm = m_s.predict_proba(Xiv_i)[:, 1]
out["smote_sensitivity"] = {
    "main_class_weight_auroc": round(float(roc_auc_score(yiv, pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet").set_index(
        ["stay_id", "t_hr"]).loc[list(zip(iv.stay_id, iv.t_hr)), "p"].values)), 4),
    "smote_auroc": round(float(roc_auc_score(yiv, p_sm)), 4),
    "delta": None,  # 下方填
    "note": "HP 冻结不重调；SMOTE 仅训练集（中位插补后）；验证集真实分布"}
out["smote_sensitivity"]["delta"] = round(out["smote_sensitivity"]["smote_auroc"]
                                           - out["smote_sensitivity"]["main_class_weight_auroc"], 4)
print("③ SMOTE 敏感性:", out["smote_sensitivity"])

with open(f"{R}/g1_gap_closers.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("SAVED g1_gap_closers.json")
