# -*- coding: utf-8 -*-
"""G5: skill 对照查漏补缺——稳健性缺口闭口（SAP §4 三重论证补全 + 对照 A + 分带验证 + 阈值扫描）
① 5-seed 稳定性（终选 HP 重训，intval AUROC range）
② 学习曲线（train 25/50/75/100% 患者级抽样）
③ 随机分割 vs 时间分割敏感性（Zhang 2026 先例）
④ 对照 A：静态基线 LR（age/male/cr_base/charlson；SOFA 未入 ABI 特征线→偏离记录）+ DeLong 配对检验 vs GBM
⑤ 风险分带实测事件率（M4 内验 + eICU 外验，select-fit 校准概率；分带 <0.10/0.10-0.20/≥0.20）
⑥ 阈值扫描合并表（thr 0.05-0.30：sens/spec/PPV/NNE=1/PPV，eICU 校准概率）
输出：results/g5_robustness.json
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import statsmodels.api as sm

R = "E:/TBI subtype/09_tbi_aki/results"
out = {"probe": "G5 robustness gap-closers (skill audit)", "date": "2026-09-22"}

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
tr = F[F.year <= 2017]
iv = F[(F.year >= 2020) & (F.year <= 2022)]
Xiv, yiv = iv[feats], iv.label.values
print(f"train {len(tr)} / intval {len(iv)} ({int(yiv.sum())}+)")


def train_gbm(df, seed):
    m = lgb.LGBMClassifier(**HP, class_weight="balanced", random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(df[feats], df.label.values)
    return m


def delong_test(y, p1, p2):
    """DeLong 配对 AUROC 检验（自包含实现）"""
    def auc_var_components(y, p):
        pos = p[y == 1]
        neg = p[y == 0]
        m, n = len(pos), len(neg)
        # V10 (pos) 与 V01 (neg) 的结构成分
        tx = np.sort(pos)
        ty = np.sort(neg)
        def rank(x, v):
            # 大于 v 的个数与等于 v 的处理（含并列 0.5）
            hi = np.searchsorted(x, v, side="right")
            lo = np.searchsorted(x, v, side="left")
            return hi, lo
        V10 = np.zeros(m)
        for i, v in enumerate(pos):
            hi, lo = rank(ty, v)
            V10[i] = hi - (hi - lo) / 2.0
        V01 = np.zeros(n)
        for j, v in enumerate(neg):
            hi, lo = rank(tx, v)
            V01[j] = hi - (hi - lo) / 2.0
        a = V10.sum() / (m * n)
        s10 = np.var(V10, ddof=1) / m
        s01 = np.var(V01, ddof=1) / n
        return a, s10, s01, V10 / n, V01 / m
    a1, s10_1, s01_1, v10_1, v01_1 = auc_var_components(y, p1)
    a2, s10_2, s01_2, v10_2, v01_2 = auc_var_components(y, p2)
    # 协方差（同一样本两组预测）
    cov10 = np.cov(v10_1, v10_2, ddof=1)[0, 1] / len(p1[y == 1])
    cov01 = np.cov(v01_1, v01_2, ddof=1)[0, 1] / len(p1[y == 0])
    var_d = s10_1 + s01_1 + s10_2 + s01_2 - 2 * (cov10 + cov01)
    z = (a1 - a2) / np.sqrt(var_d)
    from scipy.stats import norm
    return {"auc1": round(a1, 4), "auc2": round(a2, 4), "z": round(float(z), 3),
            "p": round(float(2 * (1 - norm.cdf(abs(z)))), 5)}


# ---- ① 5-seed 稳定性 ----
seed_aucs = {}
for sd in [42, 7, 123, 999, 2024]:
    m = train_gbm(tr, sd)
    seed_aucs[sd] = round(float(roc_auc_score(yiv, m.predict_proba(Xiv)[:, 1])), 4)
    print(f"seed {sd}: {seed_aucs[sd]}", flush=True)
out["seed_stability"] = {"aucs": seed_aucs, "range": round(max(seed_aucs.values()) - min(seed_aucs.values()), 4),
                         "mean": round(float(np.mean(list(seed_aucs.values()))), 4)}

# ---- ② 学习曲线 ----
rng = np.random.default_rng(42)
pts = tr.stay_id.unique()
lc = []
for frac in [0.25, 0.50, 0.75, 1.0]:
    sub_pts = rng.choice(pts, size=int(len(pts) * frac), replace=False)
    sub = tr[tr.stay_id.isin(sub_pts)]
    m = train_gbm(sub, 42)
    lc.append({"frac": frac, "n_train_ckpt": len(sub), "n_train_pos": int(sub.label.sum()),
               "intval_auroc": round(float(roc_auc_score(yiv, m.predict_proba(Xiv)[:, 1])), 4)})
    print("LC", lc[-1], flush=True)
out["learning_curve"] = lc

# ---- ③ 随机分割敏感性 ----
F2 = F.sample(frac=1.0, random_state=42)
n_tr = int(len(F2) * 0.70)
rtr, rte = F2.iloc[:n_tr], F2.iloc[n_tr:]
m = train_gbm(rtr, 42)
out["random_vs_temporal"] = {
    "random_70_30": round(float(roc_auc_score(rte.label.values, m.predict_proba(rte[feats])[:, 1])), 4),
    "temporal_reference": 0.7369,
    "note": "随机分割检查点级 70/30（Zhang 2026 先例：分割方式敏感性 <0.02 为稳定）；与时间分割非同分布对照，差异含年份漂移效应"}

# ---- ④ 对照 A + DeLong ----
BASE_FEATS = ["age", "male", "cr_base", "charlson"]
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr[BASE_FEATS], tr.label.values)
p_lr = lr.predict_proba(iv[BASE_FEATS])[:, 1]
p_gb = model.predict_proba(Xiv)[:, 1]
out["control_A_baseline_LR"] = {
    "features": BASE_FEATS, "note": "SAP 对照 A 原定含 SOFA——ABI 特征线未建 SOFA（偏离记录：以 Charlson 代）",
    "intval_auroc": round(float(roc_auc_score(yiv, p_lr)), 4)}
out["delong_gbm_vs_baselineLR"] = delong_test(yiv, p_gb, p_lr)

# ---- ⑤ 风险分带实测事件率 ----
layer = json.load(open(f"{R}/g2_recalibration_dca.json", encoding="utf-8"))["recal_layer_frozen"]
A_, B_ = float(layer["a_intercept"]), float(layer["b_slope"])
def apply_layer(p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(A_ + B_ * z)))
def band_rates(y, p, tag):
    pc = apply_layer(p)
    r = {}
    for nm, m_ in [("<0.10", pc < 0.10), ("0.10-0.20", (pc >= 0.10) & (pc < 0.20)), (">=0.20", pc >= 0.20)]:
        r[nm] = {"n": int(m_.sum()), "observed_event_rate": round(float(y[m_].mean()), 4) if m_.sum() else None}
    return r
out["band_event_rates"] = {
    "M4_intval": band_rates(yiv, p_gb, "M4"),
    "eICU_external": band_rates(
        pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet").y.values,
        pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet").p.values, "eICU")}

# ---- ⑥ 阈值扫描合并表（eICU 校准概率，检查点级） ----
exd = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")
p_ex = apply_layer(exd.p.values)
y_ex = exd.y.values
rows = []
for t in [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
    pred = p_ex >= t
    tp = int((pred & (y_ex == 1)).sum())
    fp = int((pred & (y_ex == 0)).sum())
    fn = int((~pred & (y_ex == 1)).sum())
    tn = int((~pred & (y_ex == 0)).sum())
    ppv = tp / max(tp + fp, 1)
    rows.append({"thr": t, "sens": round(tp / max(tp + fn, 1), 3), "spec": round(tn / max(tn + fp, 1), 3),
                 "ppv": round(ppv, 3), "nne_ckpt": round(1 / ppv, 2) if ppv > 0 else None})
out["threshold_sweep_eicu"] = rows

with open(f"{R}/g5_robustness.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("SAVED g5_robustness.json")
print(json.dumps({k: out[k] for k in ("seed_stability", "control_A_baseline_LR", "delong_gbm_vs_baselineLR", "band_event_rates")}, ensure_ascii=False, indent=1))
