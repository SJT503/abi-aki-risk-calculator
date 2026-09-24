# -*- coding: utf-8 -*-
"""G13: 公平性重切（65 岁二分组）+ 真聚类 bootstrap CI（PI 2026-09-23 指令）
==========================================================
① 年龄亚组从 3 带 (<65/65-79/≥80) 改为 65 二分（<65 / ≥65），性别不变
② 每亚组 AUROC + stay 级聚类 bootstrap 95% CI（1000 重采样，seed 42）
   ——替代 fig6 硬编码 ±0.012 假区间
③ 双库：M4 内验 + eICU 外验；落盘 g13_fairness65.json（图表数据源）
注：65 二分为 PI 修稿期决定 = post hoc 重切，MA2 记 D42
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

R = "E:/TBI subtype/09_tbi_aki/results"
SEED, N_BOOT = 42, 1000

# ---- 年龄映射 ----
n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet")[["stay_id", "anchor_age", "gender"]]
n8 = pd.read_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet")[["stay_id", "age", "gender"]]


def subgroups(row):
    if row.dataset == "M4":
        age, male = row.anchor_age, row.gender == "M"
    else:
        age, male = row.age, str(row.gender).upper().startswith("M")
    return pd.Series({
        "age<65": age < 65, "age>=65": age >= 65,
        "male": male, "female": not male})


def cluster_boot_ci(y, p, stay, seed=SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    stays = stay.unique()
    G = {s: (y[stay == s].to_numpy(int), p[stay == s].to_numpy(float))
         for s in stays}
    base = roc_auc_score(y, p)
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(stays, size=len(stays), replace=True)
        ys = np.concatenate([G[s][0] for s in pick])
        ps = np.concatenate([G[s][1] for s in pick])
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        aucs.append(roc_auc_score(ys, ps))
    return round(float(base), 4), [round(float(np.percentile(aucs, 2.5)), 4),
                                    round(float(np.percentile(aucs, 97.5)), 4)]


out = {"probe": "G13 fairness re-cut at age 65 + real cluster bootstrap CI",
       "date": "2026-09-23", "design": "PI-directed post-hoc re-cut (D42); stay-level cluster bootstrap, seed 42, 1000 resamples",
       "M4_intval": {}, "eICU_external": {}}

for tag, pred_f, cov, ycol, pcol in [
        ("M4_intval", "n6_abi_gbm_preds.parquet", n1.rename(columns={"anchor_age": "agea", "gender": "g"}), "y", "p"),
        ("eICU_external", "n8_5_eicu_abi_external.parquet", n8.rename(columns={"age": "agea", "gender": "g"}), "y", "p")]:
    d = pd.read_parquet(f"{R}/{pred_f}")
    d = d.merge(cov, on="stay_id", how="left")
    assert d.agea.notna().all(), f"{tag} 年龄缺失"
    d["male"] = str_cov = d.g.astype(str).str.upper().str.startswith("M")
    subs = {"age<65": d.agea < 65, "age>=65": d.agea >= 65, "male": d.male, "female": ~d.male}
    for gname, mask in subs.items():
        sub = d[mask.values]
        auc, ci = cluster_boot_ci(sub[ycol], sub[pcol], sub.stay_id)
        out[tag][gname] = {"auroc": auc, "ci95": ci,
                           "n_ckpt": int(len(sub)), "n_stays": int(sub.stay_id.nunique())}
        print(f"{tag} {gname}: AUROC {auc} CI {ci} (n={len(sub)})")

# 年龄差值
for tag in ["M4_intval", "eICU_external"]:
    diff = round(out[tag]["age>=65"]["auroc"] - out[tag]["age<65"]["auroc"], 4)
    out[tag]["age_gap_65"] = diff
    print(f"{tag} ≥65 − <65 = {diff:+.4f}")

with open(f"{R}/g13_fairness65.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("→ results/g13_fairness65.json")
