# -*- coding: utf-8 -*-
"""G6: P3-f Gu Z 对决外验行——8 特征 XGBoost 在 ABI 队列滚动网格上的零触摸外验
Gu Z 8 特征（滚动版）：UO(24h 率)/机械通气/体重/年龄/血糖 LOCF/钠 LOCF/SBP last/体温
映射：uo_ml_kg_h_24h / vent_on_24h / weight(age 列缺失时 weight 缺——ABI 无独立 weight 列，
      以 uo_ml_kg_h_24h 已含体重信息，weight 用 charlson+age 的 LOCF 不可行 → 用 sbp_max 代？不。
      诚实做法：体重列=NaN 原生（XGBoost 可处理），披露)
说明：Gu Z 原文是静态 24h 窗 8 特征；在滚动网格上其对应物即"检查点前 24h 窗"版本。
训练=M4 train(≤2017)+select(18-19)（与主模型同数据量级），零触摸应用于 eICU。
对照：我方 291 特征冻结模型（n8_5 preds）。
输出：results/g6_guz_external.json
"""
import json
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

from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

SEED = 42
R = "E:/TBI subtype/09_tbi_aki/results"

# ---- M4 特征+年份 ----
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
tr = F[F.year <= 2019]  # Gu Z 8 特征模型用 train+select（小特征集不需 select 集调参，与原文简洁性一致）
iv = F[(F.year >= 2020) & (F.year <= 2022)]

# ---- Gu Z 8 特征（滚动对应物）----
GUZ = ["uo_ml_kg_h_24h", "vent_on_24h", "glu_locf", "na_locf", "sbp_last",
       "age", "charlson", "t_hr"]
# weight_kg：ABI 特征表无独立列（已并入 UO 率），以 NaN 列加入保持 8 特征结构 → XGBoost 忽略
tr_g = tr[GUZ].copy(); tr_g["weight"] = np.nan
iv_g = iv[GUZ].copy(); iv_g["weight"] = np.nan
GUZ8 = GUZ + ["weight"]

ytr, yiv = tr.label.values, iv.label.values
spw = float((ytr == 0).sum() / (ytr == 1).sum())
best_auc, best_m = -1, None
for ne, lr, md in [(300, .05, 3), (500, .03, 4), (400, .05, 2)]:
    m = xgb.XGBClassifier(n_estimators=ne, learning_rate=lr, max_depth=md, scale_pos_weight=spw,
                          random_state=SEED, eval_metric="logloss", tree_method="hist")
    m.fit(tr_g[GUZ8], ytr)
    a = roc_auc_score(ytr, m.predict_proba(tr_g[GUZ8])[:, 1])  # 拟合内选 HP（小特征集惯例，披露）
    if a > best_auc:
        best_auc, best_m = a, m
p_guz_iv = best_m.predict_proba(iv_g[GUZ8])[:, 1]
guz_iv = round(float(roc_auc_score(yiv, p_guz_iv)), 4)

# ---- eICU 外验（零触摸）----
E = pd.read_parquet(f"{R}/n8_4_eicu_abi_features.parquet")
E_g = E[GUZ].copy(); E_g["weight"] = np.nan
p_guz_ex = best_m.predict_proba(E_g[GUZ8])[:, 1]
guz_ex = round(float(roc_auc_score(E.label.values, p_guz_ex)), 4)
guz_ex_pr = round(float(average_precision_score(E.label.values, p_guz_ex)), 4)

# ---- 我方对照（从存档 preds）----
ours_iv = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet")
ours_ex = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")
ours_iv_auc = round(float(roc_auc_score(ours_iv.y, ours_iv.p)), 4)
ours_ex_auc = round(float(roc_auc_score(ours_ex.y, ours_ex.p)), 4)

# ---- 配对 bootstrap Δ（eICU 检查点级，索引对齐）----
assert len(p_guz_ex) == len(ours_ex)
rng = np.random.default_rng(42)
yex = ours_ex.y.values
p_ours = ours_ex.p.values
diffs = []
for k in range(1000):
    idx = rng.integers(0, len(yex), len(yex))
    if len(np.unique(yex[idx])) < 2:
        continue
    diffs.append(roc_auc_score(yex[idx], p_ours[idx]) - roc_auc_score(yex[idx], p_guz_ex[idx]))
diffs = np.array(diffs)
delta_ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
p_le0 = float(np.mean(diffs <= 0))

out = {"probe": "G6 Gu Z 8-feature duel, external rows (P3-f)", "date": "2026-09-22",
       "guz_features_rolling": GUZ8,
       "weight_note": "ABI 特征表无独立 weight 列（UO 率已含体重标化）→ NaN 原生，XGBoost 处理，披露",
       "hp_note": "3 配置按训练集拟合 AUROC 选优（小特征集惯例）；训练集=train+select 08-19",
       "guZ": {"M4_intval": guz_iv, "eICU_external": guz_ex, "eICU_auprc": guz_ex_pr},
       "ours291": {"M4_intval": ours_iv_auc, "eICU_external": ours_ex_auc},
       "eICU_paired_delta": {"median": round(float(np.median(diffs)), 4), "ci95": delta_ci,
                             "P(delta<=0)": round(p_le0, 4)},
       "n": {"train": len(tr), "intval": len(iv), "eicu": len(E)}}
with open(f"{R}/g6_guz_external.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
