# -*- coding: utf-8 -*-
"""G2: P0-a 重校准层冻结 + P0-b DCA（skill Step 7.2/7.3/8.1）
重校准层（部署组件，fold-safe）：logistic(a+b·logit p) fit 于 select 集（18-19，开发库内部）
→ 应用到 M4 内验 + eICU 外验（零触摸保持——eICU 不参与 fit）
另报：eICU local recalibration 敏感性（半样本 fit/半样本评估 ×20，skill 8.1 模板）
DCA：net benefit 0.05-0.50（重校准后概率），vs treat-all/none，内验+外验
输出：results/g2_recalibration_dca.json + g2_dca_curves.parquet + 回写 n8_5 json 的 full_recalibration 字段
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

import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score

R = "E:/TBI subtype/09_tbi_aki/results"
out = {"probe": "G2 recalibration layer freeze + DCA (P0-a/P0-b)", "date": "2026-09-21"}


def ece10(y, p):
    df = pd.DataFrame({"y": y, "p": p})
    df["b"] = pd.qcut(df.p, 10, duplicates="drop")
    return float((df.groupby("b", observed=True).apply(
        lambda g: len(g) / len(df) * abs(g.y.mean() - g.p.mean()), include_groups=False)).sum())


def cal4(y, p, tag):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    c = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    return {"set": tag, "slope": round(float(c.params[1]), 3), "intercept": round(float(c.params[0]), 3),
            "ece10": round(ece10(y, p), 4), "brier": round(float(brier_score_loss(y, p)), 4)}


# ---------- 冻结模型 + select 集预测 ----------
frozen = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")
feats, model = frozen["feats"], frozen["model"]
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
sel = F[(F.year >= 2018) & (F.year <= 2019)]
p_sel = model.predict_proba(sel[feats])[:, 1]
z_sel = np.log(np.clip(p_sel, 1e-6, 1 - 1e-6) / (1 - np.clip(p_sel, 1e-6, 1 - 1e-6)))
layer = sm.Logit(sel.label.values, sm.add_constant(z_sel)).fit(disp=0)
A, B = float(layer.params[0]), float(layer.params[1])
out["recal_layer_frozen"] = {"form": "p* = logistic(a + b·logit(p))", "a_intercept": round(A, 4),
                             "b_slope": round(B, 4), "fit_on": "select 18-19 (n=%d, %d+)" % (len(sel), int(sel.label.sum()))}
print("重校准层:", out["recal_layer_frozen"])

# ---------- 应用：内验 + eICU ----------
iv = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet")          # M4 内验
ex = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")    # eICU 外验


def apply_layer(p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(A + B * z)))


for tag, d in [("M4_intval", iv), ("eICU_external", ex)]:
    p_raw, y = d.p.values, d.y.values
    p_rec = apply_layer(p_raw)
    out[f"calibration_{tag}"] = {
        "raw": cal4(y, p_raw, tag + "_raw"),
        "selectfit_recalibrated": cal4(y, p_rec, tag + "_recal"),
        "auroc_unchanged_check": round(float(roc_auc_score(y, p_raw)), 4) == round(float(roc_auc_score(y, p_rec)), 4)}
    print(tag, out[f"calibration_{tag}"])

# eICU local recalibration 敏感性（半样本 ×20）
rng = np.random.default_rng(42)
loc_ece, loc_brier = [], []
for k in range(20):
    idx = rng.permutation(len(ex))
    half = len(idx) // 2
    z_h1 = np.log(np.clip(ex.p.values[idx[:half]], 1e-6, 1 - 1e-6) / (1 - np.clip(ex.p.values[idx[:half]], 1e-6, 1 - 1e-6)))
    lr = sm.Logit(ex.y.values[idx[:half]], sm.add_constant(z_h1)).fit(disp=0)
    z_h2 = np.log(np.clip(ex.p.values[idx[half:]], 1e-6, 1 - 1e-6) / (1 - np.clip(ex.p.values[idx[half:]], 1e-6, 1 - 1e-6)))
    p_loc = 1 / (1 + np.exp(-(lr.params[0] + lr.params[1] * z_h2)))
    loc_ece.append(ece10(ex.y.values[idx[half:]], p_loc))
    loc_brier.append(brier_score_loss(ex.y.values[idx[half:]], p_loc))
out["eicu_local_recalibration_sensitivity"] = {
    "ece10_median": round(float(np.median(loc_ece)), 4), "brier_median": round(float(np.median(loc_brier)), 4),
    "n_repeats": 20, "design": "半样本 fit/另半评估（skill 8.1 模板）"}

# ---------- DCA（重校准后概率）----------
thr = np.linspace(0.05, 0.50, 91)
rows = []
for tag, d in [("M4_intval", iv), ("eICU_external", ex)]:
    y = d.y.values
    p = apply_layer(d.p.values)
    n = len(y)
    prev = y.mean()
    for t in thr:
        tp = int(np.sum((p >= t) & (y == 1)))
        fp = int(np.sum((p >= t) & (y == 0)))
        nb = tp / n - fp / n * t / (1 - t)
        nb_all = prev - (1 - prev) * t / (1 - t)
        rows.append((tag, round(t, 3), nb, nb_all, 0.0))
D = pd.DataFrame(rows, columns=["set", "threshold", "nb_model", "nb_treat_all", "nb_treat_none"])
D.to_parquet(f"{R}/g2_dca_curves.parquet", index=False)
dca_sum = {}
for tag in ("M4_intval", "eICU_external"):
    g = D[D.set == tag]
    beats_all = (g.nb_model > g.nb_treat_all) & (g.nb_model > 0)

    def at(t, col):
        return round(float(g[np.isclose(g.threshold, t)].iloc[0][col]), 4)  # 精确阈值匹配（核验员修正：round 匹配多行）

    dca_sum[tag] = {
        "nb_at_0.10": at(0.10, "nb_model"), "nb_at_0.20": at(0.20, "nb_model"),
        "nb_at_0.30": at(0.30, "nb_model"), "nb_treat_all_at_0.20": at(0.20, "nb_treat_all"),
        "max_nb": round(float(g.nb_model.max()), 4),
        "thr_range_beats_all_and_none": [float(g[beats_all.values].threshold.min()),
                                          float(g[beats_all.values].threshold.max())]}
out["dca"] = dca_sum
print("DCA:", json.dumps(dca_sum, ensure_ascii=False))

# ---------- 回写 n8_5 json（修复重跑丢失的 full_recalibration 字段，291 版）----------
p_ex_rec = apply_layer(ex.p.values)
full = cal4(ex.y.values, p_ex_rec, "eICU_full_recal_local")
p5 = json.load(open(f"{R}/n8_5_eicu_abi_external.json", encoding="utf-8"))
p5["calibration_full_recalibration"] = {"ece10": full["ece10"], "brier": full["brier"],
                                        "note": "select-fit 重校准层应用于 eICU 全样本（判别不变；local 半样本敏感性另见 g2）"}
json.dump(p5, open(f"{R}/n8_5_eicu_abi_external.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

with open(f"{R}/g2_recalibration_dca.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("SAVED g2_recalibration_dca.json + g2_dca_curves.parquet")
