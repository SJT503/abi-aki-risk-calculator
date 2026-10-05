# -*- coding: utf-8 -*-
"""G5c: 修复 g5b 的②——其"患者级 70/30"实现实为 stay 级（rng.choice(stays)），
同 subject 多次入科仍可跨集。本脚本仅换分割单位为 subject_id，其余协议
（frozen 291 feats、HP、class_weight、seed、label）与 g5b_fix.py 逐字一致。
回写 g5_robustness.json：random_vs_temporal 换真 subject 级值 + 作废注记。
"""
import json, warnings
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

R = "E:/TBI subtype/09_tbi_aki/results"
frozen = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")
feats, HP = frozen["feats"], dict(n_estimators=800, learning_rate=0.02, num_leaves=127)

F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))

rng = np.random.default_rng(42)
subjects = np.array(sorted({s2sub[s] for s in F.stay_id.unique()}))
rtr_subj = set(rng.choice(subjects, size=int(len(subjects) * 0.70), replace=False))
tr_mask = F.stay_id.map(s2sub).isin(rtr_subj).values
rtr, rte = F[tr_mask], F[~tr_mask]
# 审计：subject 跨集必须为 0
tr_s = set(rtr.stay_id.map(s2sub)); te_s = set(rte.stay_id.map(s2sub))
assert len(tr_s & te_s) == 0, "subject leak!"
print(f"train {len(tr_s)} subj/{rtr.stay_id.nunique()} stays/{len(rtr)} ckpts | "
      f"test {len(te_s)} subj/{rte.stay_id.nunique()} stays/{len(rte)} ckpts | overlap=0")

m = lgb.LGBMClassifier(**HP, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)
m.fit(rtr[feats], rtr.label.values)
auc = float(roc_auc_score(rte.label.values, m.predict_proba(rte[feats])[:, 1]))
print(f"TRUE subject-level random 70/30 AUROC = {auc:.4f}  (g5b stay-level value was 0.7679)")

# 回写 json
out = json.load(open(f"{R}/g5_robustness.json", encoding="utf-8"))
old = out.get("random_vs_temporal", {})
out["random_vs_temporal"] = {
    "random_subjectlevel_70_30": round(auc, 4),
    "temporal_reference": old.get("temporal_reference", 0.7369),
    "note": ("真 subject 级随机分割 70/30（2026-09-28 修正：g5b 所记 0.7679 实为 stay 级选择，"
             "同 subject 重复入科可跨集；更早的检查点级 0.9714 亦因患者跨集泄漏作废）。"
             "与时间分割非同分布对照，差异含年份漂移效应"),
    "superseded": {"random_patientlevel_70_30_stay_units": old.get("random_patientlevel_70_30", 0.7679),
                   "random_checkpointlevel_void": 0.9714},
}
json.dump(out, open(f"{R}/g5_robustness.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("g5_robustness.json updated")
