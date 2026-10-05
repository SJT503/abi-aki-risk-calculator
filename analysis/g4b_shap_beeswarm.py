# -*- coding: utf-8 -*-
"""G4b: SHAP beeswarm 数据层（PI 2026-09-23 指令：Fig4a 改蜂群图）
==========================================================
复刻 g4 的抽样与解释器（rng(42)/8,000 行/TreeExplainer），落盘 top-15 特征的
个体级 SHAP 值 + 特征值矩阵（供 fig4 蜂群图；g4 原版只存聚合）。
说明：SHAP 值为模型派生统计量（非原始 EHR 数据），落盘 8000×15 不违 DUA 精神。
输出：results/g4b_shap_beeswarm.parquet + 顶部特征序（与 g4 top20 一致性校验）
"""
import os
import sys
import warnings

import duckdb
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

import shap

R = "E:/TBI subtype/09_tbi_aki/results"
frozen = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")
feats, model = frozen["feats"], frozen["model"]

F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
con = duckdb.connect()
adm = con.execute(f"""
SELECT c.stay_id, c.subject_id, a.admittime, p.anchor_year
FROM read_parquet('{R}/n1_m4_abi_cohort.parquet') c
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/admissions.csv.gz') a ON c.hadm_id=a.hadm_id
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz') p ON c.subject_id=p.subject_id
""").df()
pat2 = pd.read_csv("E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz", usecols=["subject_id", "anchor_year_group"])
_g = pat2["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
pat2["group_mid"] = (_g[0].astype(int) + _g[1].astype(int)) / 2.0
coh = adm.merge(pat2[["subject_id", "group_mid"]], on="subject_id", how="left")
coh["year"] = (coh.group_mid + (coh.admittime.dt.year - coh.anchor_year)).round().astype(int)
F = F.merge(coh[["stay_id", "year"]], on="stay_id", how="left")
iv = F[(F.year >= 2020) & (F.year <= 2022)].reset_index(drop=True)
Xiv = iv[feats]
print(f"intval: {len(iv)} rows")

rng = np.random.default_rng(42)
_sidx = rng.integers(0, len(iv), 8000)
Xs = Xiv.iloc[_sidx]
expl = shap.TreeExplainer(model)
sv = expl.shap_values(Xs)
if isinstance(sv, list):
    sv = sv[1]
imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
top15 = list(imp.head(15).index)

out = pd.DataFrame(sv, columns=feats)[top15]
for f in top15:
    out[f + "__value"] = Xs[f].values
out.to_parquet(f"{R}/g4b_shap_beeswarm.parquet", index=False)

# 与 g4 top20 一致性门禁
import json
g4 = json.load(open(f"{R}/g4_shap_fairness.json", encoding="utf-8"))
g4_top15 = list(g4["shap"]["top20_mean_abs"].keys())[:15]
match = sum(1 for a, b in zip(top15, g4_top15) if a == b)
print(f"top15 与 g4 一致: {match}/15")
print("top15:", top15)
if match < 14:
    raise SystemExit("GATE FAIL: 蜂群数据 top15 与 g4 聚合版不一致")
print("→ results/g4b_shap_beeswarm.parquet")
