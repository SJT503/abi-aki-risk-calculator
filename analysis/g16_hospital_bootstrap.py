# -*- coding: utf-8 -*-
"""G16: 医院两层 bootstrap（A3 双口径）+ 医院级 AUROC 分布
==========================================================
背景：外验 171 医院、中位 13 stay/院、max 563——检查点在医院内高度聚类，
     单层（检查点级）bootstrap CI 过窄。已报告 stay 聚类 CI（0.697–0.724）。
设计：
  ① 两层 bootstrap（固定规模）：抽医院（有放回）→ 院内抽 stay（有放回）
     → 汇聚检查点重算 AUROC，1000 次 → 95% CI
  ② 医院级 AUROC 分布：每家 ≥20 阳性检查点且两类皆有的医院单独算 AUROC
     → 落盘 CSV（供分布图/表）+ 汇总统计（IQR/range/医院数）
  ③ 对照：stay 单层聚类 CI 复算（同 seed 同协议，供三口径并列）
门禁：
  - 医院数必须 = 171，检查点总数必须 = 84,866
  - 总体 AUROC 必须复现 0.7094
数据真实性：全部数字来自本脚本实跑输出，落盘 g16_hospital_bootstrap.json
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

from sklearn.metrics import roc_auc_score

SEED = 42
R = "E:/TBI subtype/09_tbi_aki/results"

ex = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")
coh = pd.read_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet")[["stay_id", "hospitalid"]]
ex = ex.merge(coh, on="stay_id", how="left")
assert ex.hospitalid.notna().all(), "hospitalid 缺失"

n_hosp = ex.hospitalid.nunique()
auc_all = round(float(roc_auc_score(ex.y, ex.p)), 4)
print(f"[gate] hospitals={n_hosp} (期望171) | ckpt={len(ex)} (期望84866) | auroc={auc_all} (期望0.7094)")
if n_hosp != 171 or len(ex) != 84866:
    raise SystemExit("GATE FAIL: 外验规模不符")
if auc_all != 0.7094:
    raise SystemExit("GATE FAIL: AUROC 未复现")

d = ex.reset_index(drop=True)
y, p = d.y.values, d.p.values

# ---- 索引预分组（两层：医院 → stay → 检查点） ----
hosp_ids = d.hospitalid.unique()
stays_by_hosp = {h: g.stay_id.unique() for h, g in d.groupby("hospitalid")}
idx_by_stay = {s: g.index.to_numpy() for s, g in d.groupby("stay_id", sort=False)}
n_stay_total = d.stay_id.nunique()

rng = np.random.default_rng(SEED)


def two_level_boot(n_boot=1000):
    aucs = []
    for _ in range(n_boot):
        hs = rng.choice(hosp_ids, size=len(hosp_ids), replace=True)
        idx_parts = []
        for h in hs:
            ss = stays_by_hosp[h]
            pick = rng.choice(ss, size=len(ss), replace=True)
            idx_parts.extend(idx_by_stay[s] for s in pick)
        idx = np.concatenate(idx_parts)
        yy = y[idx]
        if len(set(yy)) > 1:
            aucs.append(roc_auc_score(yy, p[idx]))
    return aucs


def stay_boot(n_boot=1000):
    aucs = []
    stays = d.stay_id.unique()
    for _ in range(n_boot):
        pick = rng.choice(stays, size=len(stays), replace=True)
        idx = np.concatenate([idx_by_stay[s] for s in pick])
        yy = y[idx]
        if len(set(yy)) > 1:
            aucs.append(roc_auc_score(yy, p[idx]))
    return aucs


print("两层 bootstrap 运行中（1000 次）...")
aucs2 = two_level_boot()
print("stay 单层 bootstrap 运行中（1000 次）...")
aucs1 = stay_boot()

ci_two = [round(float(np.percentile(aucs2, 2.5)), 4), round(float(np.percentile(aucs2, 97.5)), 4)]
ci_stay = [round(float(np.percentile(aucs1, 2.5)), 4), round(float(np.percentile(aucs1, 97.5)), 4)]

# ---- 医院级 AUROC 分布（≥20 阳性检查点） ----
rows = []
for h, g in d.groupby("hospitalid"):
    yy, pp = g.y.values, g.p.values
    if yy.sum() >= 20 and len(set(yy)) > 1:
        rows.append({"hospitalid": int(h), "n_ckpt": len(g), "n_stay": g.stay_id.nunique(),
                     "n_pos": int(yy.sum()), "event_rate": round(float(yy.mean()), 4),
                     "auroc": round(float(roc_auc_score(yy, pp)), 4)})
dist = pd.DataFrame(rows).sort_values("auroc").reset_index(drop=True)
dist.to_csv(f"{R}/g16_hospital_auroc_distribution.csv", index=False)
dist_desc = {"n_hospitals_evaluable": len(dist),
             "min_pos_ckpt_threshold": 20,
             "auroc_iqr": [round(float(dist.auroc.quantile(.25)), 4), round(float(dist.auroc.quantile(.75)), 4)],
             "auroc_range": [round(float(dist.auroc.min()), 4), round(float(dist.auroc.max()), 4)],
             "auroc_median": round(float(dist.auroc.median()), 4),
             "n_below_0.60": int((dist.auroc < 0.60).sum()),
             "n_below_0.65": int((dist.auroc < 0.65).sum())}

out = {"probe": "G16 hospital two-level bootstrap (A3 dual-caliber)", "date": "2026-09-23",
       "n_hospitals": int(n_hosp), "n_stays": int(n_stay_total), "n_ckpt": int(len(d)),
       "auroc_point": auc_all,
       "ci_checkpoint_level": [0.703, 0.716],
       "ci_stay_cluster": ci_stay,
       "ci_hospital_two_level": ci_two,
       "widths": {"checkpoint": round(0.716 - 0.703, 4),
                  "stay": round(ci_stay[1] - ci_stay[0], 4),
                  "hospital_two_level": round(ci_two[1] - ci_two[0], 4)},
       "hospital_level_distribution": dist_desc,
       "csv": "results/g16_hospital_auroc_distribution.csv"}
with open(f"{R}/g16_hospital_bootstrap.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("\n=== G16 完成 → results/g16_hospital_bootstrap.json ===")
print(json.dumps({"ci_stay": ci_stay, "ci_two_level": ci_two, "dist": dist_desc}, indent=1))
