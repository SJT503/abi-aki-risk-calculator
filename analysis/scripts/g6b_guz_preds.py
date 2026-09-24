# -*- coding: utf-8 -*-
"""G6b: Gu Z 对决逐检查点预测落盘（Fig2b 改 ROC 叠加需要曲线数据）
==========================================================
逐字复刻 g6 训练协议（同特征/同 3 HP/同拟合内选优/同 seed=42，确定性），
额外保存 intval + eICU 逐检查点概率 → g6b_guz_preds.parquet
一致性门禁：重算 AUROC 必须与 g6 json 聚合值完全一致（0.684/0.6174），否则 stop
"""
import json
import os
import sys
import warnings

import duckdb
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
tr = F[F.year <= 2019]
iv = F[(F.year >= 2020) & (F.year <= 2022)]

GUZ = ["uo_ml_kg_h_24h", "vent_on_24h", "glu_locf", "na_locf", "sbp_last",
       "age", "charlson", "t_hr"]
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
    a = roc_auc_score(ytr, m.predict_proba(tr_g[GUZ8])[:, 1])
    if a > best_auc:
        best_auc, best_m = a, m
p_iv = best_m.predict_proba(iv_g[GUZ8])[:, 1]
auc_iv = round(float(roc_auc_score(yiv, p_iv)), 4)

E = pd.read_parquet(f"{R}/n8_4_eicu_abi_features.parquet")
E_g = E[GUZ].copy(); E_g["weight"] = np.nan
p_ex = best_m.predict_proba(E_g[GUZ8])[:, 1]
auc_ex = round(float(roc_auc_score(E.label.values, p_ex)), 4)
pr_ex = round(float(average_precision_score(E.label.values, p_ex)), 4)

# ---- 一致性门禁 ----
g6 = json.load(open(f"{R}/g6_guz_external.json", encoding="utf-8"))
ok = (auc_iv == round(float(g6["guZ"]["M4_intval"]), 4)) and (auc_ex == round(float(g6["guZ"]["eICU_external"]), 4))
print(f"重算 intval={auc_iv} (g6={g6['guZ']['M4_intval']}) | eICU={auc_ex} (g6={g6['guZ']['eICU_external']})")
if not ok:
    raise SystemExit("GATE FAIL: g6b 复刻与 g6 聚合值不一致")

out = pd.DataFrame({
    "stay_id": list(iv.stay_id.values) + list(E.stay_id.values),
    "t_hr": list(iv.t_hr.values) + list(E.t_hr.values),
    "y": list(yiv) + list(E.label.values),
    "p": np.concatenate([p_iv, p_ex]),
    "set": ["M4_intval"] * len(iv) + ["eICU_external"] * len(E),
})
out.to_parquet(f"{R}/g6b_guz_preds.parquet", index=False)
print(f"→ results/g6b_guz_preds.parquet ({len(out)} rows)")
