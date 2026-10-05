# -*- coding: utf-8 -*-
"""G10: EPPP 双口径重算 + 患者级聚类 bootstrap CI（R2 审稿 Must-Fix #2 子项）
=====================================================
① EPPP 诚实口径：开发样本（train+select 2008-2019）事件 / 291 参数
   - 检查点级：开发集阳性检查点数 / 291（旧 23.4 = 全网格含内验阳性，口径混杂，废弃）
   - 患者级：开发集事件患者数 / 291（保守单位）
② 患者级聚类 bootstrap 95% CI：检查点级 AUROC/AUPRC（内验 + 外验）
   现有 CI 为检查点级 bootstrap/DeLong（忽略患者聚类 → 偏窄）；聚类 bootstrap 为披露口径
输出：results/g10_eppp_cluster_ci.json
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

from sklearn.metrics import roc_auc_score, average_precision_score

R = "E:/TBI subtype/09_tbi_aki/results"
N_PARAM = 291
N_BOOT = 1000
SEED = 42

# ---------- 年份重建（n6/n7 同法） ----------
import duckdb
con = duckdb.connect()
adm = con.execute(f"""
SELECT c.stay_id, c.subject_id, a.admittime, p.anchor_year
FROM read_parquet('{R}/n1_m4_abi_cohort.parquet') c
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/admissions.csv.gz') a ON c.hadm_id=a.hadm_id
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz') p ON c.subject_id=p.subject_id
""").df()
pat = pd.read_csv("E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz",
                  usecols=["subject_id", "anchor_year_group"])
_g = pat["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
pat["group_mid"] = (_g[0].astype(int) + _g[1].astype(int)) / 2.0
adm = adm.merge(pat[["subject_id", "group_mid"]], on="subject_id", how="left")
adm["year"] = (adm.group_mid + (adm.admittime.dt.year - adm.anchor_year)).round().astype(int)
ymap = adm.set_index("stay_id").year

# ---------- ① EPPP 双口径（开发样本 = train+select） ----------
g = pd.read_parquet(f"{R}/n3_m4_abi_grid.parquet")
g["year"] = g.stay_id.map(ymap)
dev = g[(g.year <= 2019)]                      # train 2008-17 + select 2018-19
iv = g[(g.year >= 2020) & (g.year <= 2022)]
dev_pos_ckpt = int(dev.label.sum())
# 患者级事件口径 = 网格内 ≥1 阳性检查点的患者（与手稿 1,582/g3/g9 同口径）
pos_pat = g.groupby("stay_id").label.max()
dev_event_pat = int(((pos_pat == 1) & pos_pat.index.isin(dev.stay_id.unique())).sum())
n_dev_pat = int(dev.stay_id.nunique())
eppp = {
    "definition": "development sample = train(2008-17)+select(2018-19); parameters=291; "
                  "patient-level event = >=1 positive checkpoint in grid (manuscript caliber, total 1582)",
    "checkpoint_level": {"events": dev_pos_ckpt, "eppp": round(dev_pos_ckpt / N_PARAM, 1)},
    "patient_level": {"events": dev_event_pat, "eppp": round(dev_event_pat / N_PARAM, 2)},
    "deprecated_whole_grid_eppp": 23.4,
    "note": "旧 23.4 = 全网格 6821/291（含内验阳性，口径混杂）——手稿改为开发集双口径",
}

# ---------- ② 患者级聚类 bootstrap ----------
def cluster_boot(pred, n_boot=N_BOOT, seed=SEED):
    """pred: DataFrame(stay_id, y, p)；患者整簇重采样"""
    rng = np.random.default_rng(seed)
    stays = pred.stay_id.unique()
    G = {s: (g.y.to_numpy(int), g.p.to_numpy(float))
         for s, g in pred.groupby("stay_id", sort=False)}
    aucs, auprcs = [], []
    for _ in range(n_boot):
        pick = rng.choice(stays, size=len(stays), replace=True)
        ys = np.concatenate([G[s][0] for s in pick])
        ps = np.concatenate([G[s][1] for s in pick])
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        aucs.append(roc_auc_score(ys, ps))
        auprcs.append(average_precision_score(ys, ps))
    aucs, auprcs = np.array(aucs), np.array(auprcs)
    return {"auroc": round(float(aucs.mean()), 4),
            "auroc_ci95": [round(float(np.percentile(aucs, 2.5)), 4),
                            round(float(np.percentile(aucs, 97.5)), 4)],
            "auprc": round(float(auprcs.mean()), 4),
            "auprc_ci95": [round(float(np.percentile(auprcs, 2.5)), 4),
                            round(float(np.percentile(auprcs, 97.5)), 4)],
            "n_resamples_valid": int(len(aucs))}


# 内验：n6 preds（y 即检查点 label）
p_iv = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet")[["stay_id", "y", "p"]]
p_iv.columns = ["stay_id", "y", "p"]
# 外验：n8_5 preds + grid label
p_ex = (pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")[["stay_id", "t_hr", "p"]]
        .merge(pd.read_parquet(f"{R}/n8_3_eicu_abi_grid.parquet")[["stay_id", "t_hr", "label"]],
               on=["stay_id", "t_hr"]))
p_ex = p_ex.rename(columns={"label": "y"})[["stay_id", "y", "p"]]

print("EPPP:", json.dumps(eppp, indent=1))
print("cluster bootstrap 内验 …")
ci_iv = cluster_boot(p_iv)
print(" 内验", ci_iv)
print("cluster bootstrap 外验 …")
ci_ex = cluster_boot(p_ex)
print(" 外验", ci_ex)

out = {"probe": "G10 EPPP dual-caliber + patient-cluster bootstrap CI (reviewer Must-Fix 2)",
       "date": "2026-09-23", "n_param": N_PARAM, "n_boot": N_BOOT, "seed": SEED,
       "eppp": eppp,
       "m4_intval_cluster_boot": ci_iv,
       "eicu_external_cluster_boot": ci_ex,
       "existing_ckpt_level_ci": {"m4_intval": [0.727, 0.747], "eicu": [0.703, 0.716],
                                   "note": "检查点级 bootstrap/DeLong，忽略患者聚类，偏窄"}}
with open(f"{R}/g10_eppp_cluster_ci.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("→ results/g10_eppp_cluster_ci.json")
