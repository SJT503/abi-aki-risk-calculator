# -*- coding: utf-8 -*-
"""N8-5: eICU ABI 外验——零触摸应用 N6 冻结 GBM + 偏差审计套件
零触摸：模型/特征列/阈值全部来自 n6_abi_gbm_frozen.joblib，本脚本无任何训练/调参
审计：checkpoint/patient 级 AUROC+AUPRC、校准四件套（slope/intercept/ECE10/Brier）+
     intercept-only 主更新、亚型分层 AUROC（IS+unspec 合并）、TBI-only 子集、医院分层
输出：results/n8_5_eicu_abi_external.parquet + n8_5_eicu_abi_external.json
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

R = "E:/TBI subtype/09_tbi_aki/results"
d = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")
feats, model = d["feats"], d["model"]
F = pd.read_parquet(f"{R}/n8_4_eicu_abi_features.parquet")
coh = pd.read_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet").set_index("stay_id")

X = F.reindex(columns=feats)  # 冻结列序；eICU 不可推导列→NaN（LightGBM 原生处理）
y = F.label.values
p = model.predict_proba(X)[:, 1]
print(f"eICU ABI external: {len(F)} ckpts / {int(y.sum())}+ / {F.stay_id.nunique()} patients")

rng = np.random.default_rng(42)


def boot_ci(f, n=1000):
    bs = []
    for k in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            bs.append(f(idx))
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


auroc = roc_auc_score(y, p)
auprc = average_precision_score(y, p)
ci = boot_ci(lambda i: roc_auc_score(y[i], p[i]))

# 患者级
pg = pd.DataFrame({"sid": F.stay_id.values, "p": p, "y": y})
pat_auc = roc_auc_score(pg.groupby("sid").y.max(), pg.groupby("sid").p.max())

# 校准四件套
z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
import statsmodels.api as sm
cal = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
slope, intercept = float(cal.params[1]), float(cal.params[0])
dfc = pd.DataFrame({"y": y, "p": p})
dfc["b"] = pd.qcut(dfc.p, 10, duplicates="drop")
ece10 = float((dfc.groupby("b", observed=True).apply(lambda g: len(g) / len(dfc) * abs(g.y.mean() - g.p.mean()), include_groups=False)).sum())
brier = float(brier_score_loss(y, p))
# intercept-only 更新（主更新）
p_upd = 1 / (1 + np.exp(-(intercept + z)))
brier_upd = float(brier_score_loss(y, p_upd))
ece10_upd = float((pd.DataFrame({"y": y, "p": p_upd}).assign(b=lambda d: pd.qcut(d.p, 10, duplicates="drop"))
                   .groupby("b", observed=True).apply(lambda g: len(g) / len(y) * abs(g.y.mean() - g.p.mean()), include_groups=False)).sum())

# 亚型分层
F2 = F.merge(coh[["subtype"]], left_on="stay_id", right_index=True, how="left")
sub_auc = {}
for st, g in F2.groupby("subtype"):
    if g.label.nunique() > 1 and len(g) >= 200:
        sub_auc[f"{st}(n={len(g)},+{int(g.label.sum())})"] = round(float(roc_auc_score(g.label, model.predict_proba(g.reindex(columns=feats))[:, 1])), 4)
F2["is_merged"] = F2.subtype.isin(["IS", "stroke_unspec"])
for st, g in [("IS+unspec", F2[F2.is_merged]), ("TBI-only", F2[F2.subtype == "TBI"])]:
    if g.label.nunique() > 1:
        sub_auc[f"{st}(n={len(g)},+{int(g.label.sum())})"] = round(float(roc_auc_score(g.label, model.predict_proba(g.reindex(columns=feats))[:, 1])), 4)

# 医院分层（≥1000 检查点）
F3 = F.merge(coh[["hospitalid"]], left_on="stay_id", right_index=True, how="left")
hosp_auc = {}
for h, g in F3.groupby("hospitalid"):
    if len(g) >= 1000 and g.label.nunique() > 1:
        hosp_auc[int(h)] = round(float(roc_auc_score(g.label, model.predict_proba(g.reindex(columns=feats))[:, 1])), 4)
hv = np.array(sorted(hosp_auc.values()))

pd.DataFrame({"stay_id": F.stay_id, "t_hr": F.t_hr, "y": y, "p": p, "p_intercept_updated": p_upd}).to_parquet(
    f"{R}/n8_5_eicu_abi_external.parquet", index=False)

out = {"probe": "N8-5 eICU ABI external validation (zero-touch)", "date": "2026-09-21",
       "population": {"checkpoints": len(F), "positive": int(y.sum()), "pos_rate": round(float(y.mean()), 4),
                      "patients": int(F.stay_id.nunique())},
       "primary": {"auroc": round(float(auroc), 4), "ci95": ci,
                   "auprc": round(float(auprc), 4), "patient_level_auroc": round(float(pat_auc), 4)},
       "calibration_raw": {"slope": round(slope, 3), "intercept": round(intercept, 3),
                           "ece10": round(ece10, 4), "brier": round(brier, 4)},
       "calibration_intercept_only_update": {"ece10": round(ece10_upd, 4), "brier": round(brier_upd, 4)},
       "subtype_auc": sub_auc,
       "hospital_auc": {"n_hospitals_ge1000": int(len(hv)),
                        "median": round(float(np.median(hv)), 4) if len(hv) else None,
                        "iqr": [round(float(np.percentile(hv, 25)), 4), round(float(np.percentile(hv, 75)), 4)] if len(hv) else None,
                        "range": [round(float(hv.min()), 4), round(float(hv.max()), 4)] if len(hv) else None},
       "reference": {"m4_intval_auroc": 0.7375, "v1_tbi_eicu_external": 0.649},
       "zero_touch": "冻结 n6 模型直接 predict；无 eICU 数据参与训练/特征选择/阈值"}
with open(f"{R}/n8_5_eicu_abi_external.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
