# -*- coding: utf-8 -*-
"""G12: M4 subject 级聚类 bootstrap AUROC CI（患者级最保守口径，审稿二轮发现3溯源需求）
==========================================================
背景：手稿新增句 "resampling unique patients instead of stays moved the internal-validation
interval by less than 0.003 (0.711–0.763)" 需要可复现工件（数字必须有工具出处铁律）。
设计：与 g10 同种子(seed=42)同次数(1000)，仅把重采样单元从 stay_id 换成 subject_id（n1 队列映射）。
输出：results/g12_subject_bootstrap.json
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

p = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet").rename(columns={"y": "label"})
n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet")[["stay_id", "subject_id"]]
p = p.merge(n1, on="stay_id", how="left")
assert p.subject_id.notna().all(), "subject 映射缺失"

rng = np.random.default_rng(SEED)
subs = p.subject_id.unique()
G = {s: (g_.label.to_numpy(int), g_.p.to_numpy(float)) for s, g_ in p.groupby("subject_id", sort=False)}
aucs = []
for _ in range(N_BOOT):
    pick = rng.choice(subs, size=len(subs), replace=True)
    ys = np.concatenate([G[s][0] for s in pick])
    ps = np.concatenate([G[s][1] for s in pick])
    if ys.sum() in (0, len(ys)):
        continue
    aucs.append(roc_auc_score(ys, ps))
aucs = np.array(aucs)
out = {
    "probe": "G12 subject-level cluster bootstrap (M4 internal validation)",
    "date": "2026-09-23",
    "design": "identical to g10 cluster_boot but resampling unit = subject_id (n1 mapping); seed 42, 1000 resamples",
    "n_subjects": int(len(subs)),
    "n_stays": int(p.stay_id.nunique()),
    "auroc_mean": round(float(aucs.mean()), 4),
    "auroc_ci95": [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)],
    "n_valid_resamples": int(len(aucs)),
    "reference_stay_level_ci": [0.7133, 0.7607],
}
with open(f"{R}/g12_subject_bootstrap.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, indent=1))
