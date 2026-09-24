# -*- coding: utf-8 -*-
"""G5b: 修复 g5 两处缺陷（审计自查）
① DeLong 方差公式错误（漏除 n²，z 被压成 0）→ 换经典快速 DeLong（Sun-Xu midrank）
② 随机分割敏感性原为检查点级 70/30（同患者跨集泄漏→0.9714 虚高）→ 患者级 70/30
③ seeds 全同注记：LGBM 默认无 feature/bagging 子采样=构造性确定，非"随机稳定"证据
回写 g5_robustness.json 对应字段
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

from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

R = "E:/TBI subtype/09_tbi_aki/results"
frozen = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")
feats, model = frozen["feats"], frozen["model"]
HP = dict(n_estimators=800, learning_rate=0.02, num_leaves=127)

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
tr, iv = F[F.year <= 2017], F[(F.year >= 2020) & (F.year <= 2022)]
Xiv, yiv = iv[feats], iv.label.values


# ---- 经典快速 DeLong（Sun & Xu 2014 midrank；配对两模型） ----
def _midrank(z):
    J = np.argsort(z, kind="mergesort")
    Z = z[J]
    N = len(z)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def delong_paired(y, p1, p2):
    y = np.asarray(y, dtype=bool)
    pos1, neg1 = p1[y], p1[~y]
    pos2, neg2 = p2[y], p2[~y]
    m, n = len(pos1), len(neg1)
    tx = np.concatenate([pos1, neg1])
    ty = np.concatenate([pos2, neg2])
    tx_midrank = _midrank(tx)
    ty_midrank = _midrank(ty)
    a1 = (tx_midrank[:m].sum() / m - (m + 1) / 2.0) / n
    a2 = (ty_midrank[:m].sum() / m - (m + 1) / 2.0) / n
    v10_1 = (tx_midrank[:m] - np.arange(1, m + 1)) / n   # 归一化结构成分（修复：漏 ÷n/÷m）
    v01_1 = (tx_midrank[m:] - np.arange(1, n + 1)) / m
    v10_2 = (ty_midrank[:m] - np.arange(1, m + 1)) / n
    v01_2 = (ty_midrank[m:] - np.arange(1, n + 1)) / m
    s10 = np.var(v10_1, ddof=1) / m + np.var(v10_2, ddof=1) / m - 2 * np.cov(v10_1, v10_2, ddof=1)[0, 1] / m
    s01 = np.var(v01_1, ddof=1) / n + np.var(v01_2, ddof=1) / n - 2 * np.cov(v01_1, v01_2, ddof=1)[0, 1] / n
    var_d = s10 + s01
    z = (a1 - a2) / np.sqrt(var_d)
    return {"auc1": round(float(a1), 4), "auc2": round(float(a2), 4),
            "z": round(float(z), 2), "p": float(f"{2 * (1 - norm.cdf(abs(z))):.3g}")}


# ① DeLong 修复
BASE_FEATS = ["age", "male", "cr_base", "charlson"]
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr[BASE_FEATS], tr.label.values)
p_lr = lr.predict_proba(iv[BASE_FEATS])[:, 1]
p_gb = model.predict_proba(Xiv)[:, 1]
dl = delong_paired(yiv, p_gb, p_lr)
print("DeLong(修):", dl)
# sanity: 与已知单模型 AUROC 一致
assert abs(dl["auc1"] - 0.7369) < 0.0002 and abs(dl["auc2"] - 0.6421) < 0.0002, "DeLong AUC 与独立计算不符"
# 交叉验证：bootstrap 配对检验（1000 次重采样 AUROC 差分布）
rng_b = np.random.default_rng(7)
diffs = []
for k in range(1000):
    idx = rng_b.integers(0, len(yiv), len(yiv))
    if len(np.unique(yiv[idx])) < 2:
        continue
    diffs.append(roc_auc_score(yiv[idx], p_gb[idx]) - roc_auc_score(yiv[idx], p_lr[idx]))
diffs = np.array(diffs)
boot_p = float(np.mean(diffs <= 0) * 2)
print(f"bootstrap 配对: Δ中位 {np.median(diffs):.4f} | 2·P(Δ<=0) = {boot_p:.4g}")
dl["bootstrap_crosscheck"] = {"delta_median": round(float(np.median(diffs)), 4), "p_two_sided": round(boot_p, 6),
                              "note": "1000 次重采样交叉验证 DeLong"}

# ② 患者级随机分割
rng = np.random.default_rng(42)
stays = F.stay_id.unique()
rtr_stays = set(rng.choice(stays, size=int(len(stays) * 0.70), replace=False))
rtr, rte = F[F.stay_id.isin(rtr_stays)], F[~F.stay_id.isin(rtr_stays)]
m = lgb.LGBMClassifier(**HP, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)
m.fit(rtr[feats], rtr.label.values)
rand_auc = float(roc_auc_score(rte.label.values, m.predict_proba(rte[feats])[:, 1]))
print(f"患者级随机分割 70/30: {rand_auc:.4f}")

# ③ 回写 json
out = json.load(open(f"{R}/g5_robustness.json", encoding="utf-8"))
out["delong_gbm_vs_baselineLR"] = dl
out["random_vs_temporal"] = {
    "random_patientlevel_70_30": round(rand_auc, 4),
    "temporal_reference": 0.7369,
    "note": "患者级随机分割（g5 检查点级 0.9714 系同患者跨集泄漏作废）；差异含年份漂移，非纯分割方式效应"}
out["seed_stability"]["note"] = ("LGBM 默认无 feature_fraction/bagging 子采样=构造性确定（五 seed 全同非随机稳定性证据，"
                                 "而是无随机成分）；稳健性证据实际来自学习曲线平台期+bootstrap CI")
json.dump(out, open(f"{R}/g5_robustness.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("g5_robustness.json 回写完成")
