# -*- coding: utf-8 -*-
"""N6: ABI GBM（LightGBM，患者级时间分割）+ LSTM 对比"""
import json, os, sys, warnings
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate
gate(__file__)
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

SEED = 42
R = "E:/TBI subtype/09_tbi_aki/results"
F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")

# 年份还原（cohort 无 admittime → 从 admissions join）
import duckdb
con = duckdb.connect()
adm = con.execute(f"""
SELECT c.stay_id, c.subject_id, a.admittime, p.anchor_year
FROM read_parquet('{R}/n1_m4_abi_cohort.parquet') c
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/admissions.csv.gz') a ON c.hadm_id = a.hadm_id
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz') p ON c.subject_id = p.subject_id
""").df()
pat = pd.read_csv("E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz", usecols=["subject_id","anchor_year_group"])
_g = pat["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
pat["group_mid"] = (_g[0].astype(int)+_g[1].astype(int))/2.0
coh = adm.merge(pat[["subject_id","group_mid"]], on="subject_id", how="left")
coh["year"] = (coh.group_mid + (coh.admittime.dt.year - coh.anchor_year)).round().astype(int)
F = F.merge(coh[["stay_id","year"]], on="stay_id", how="left")

drop = [c for c in F.columns if F[c].isna().mean() > 0.6] + ["stay_id","label"]
feats = [c for c in F.columns if c not in drop and c != "year"]
tr = F[F.year<=2017]; tu = F[(F.year>=2018)&(F.year<=2019)]; iv = F[(F.year>=2020)&(F.year<=2022)]
ytr, ytu, yiv = tr.label.values, tu.label.values, iv.label.values
print(f"train={len(tr)}({int(ytr.sum())}+) select={len(tu)}({int(ytu.sum())}+) intval={len(iv)}({int(yiv.sum())}+)")

best_auc, best_m = -1, None
for ne, lr, nl in [(500,.03,63),(800,.02,127),(600,.05,31)]:
    m = lgb.LGBMClassifier(n_estimators=ne, learning_rate=lr, num_leaves=nl,
                           class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(tr[feats], ytr)
    a = roc_auc_score(ytu, m.predict_proba(tu[feats])[:,1])
    if a > best_auc: best_auc, best_m = a, m
    print(f"  hp=({ne},{lr},{nl}): select AUROC={a:.4f}")

p = best_m.predict_proba(iv[feats])[:,1]
bs = [roc_auc_score(yiv[np.random.default_rng(k).integers(0,len(yiv),len(yiv))], p[np.random.default_rng(k).integers(0,len(yiv),len(yiv))])
      for k in range(1000) if len(set(yiv[np.random.default_rng(k).integers(0,len(yiv),len(yiv))]))>1]
iv2 = iv.copy(); iv2["p"] = p
pat_auc = roc_auc_score(iv2.groupby("stay_id").label.max(), iv2.groupby("stay_id").p.max())
imp = pd.Series(best_m.feature_importances_, index=feats).sort_values(ascending=False)

out = {"probe":"N6 ABI GBM","date":"2026-09-21",
 "pop":{"train":len(tr),"train_pos":int(ytr.sum()),"select":len(tu),"intval":len(iv),"intval_pos":int(yiv.sum()),
        # caliber notes (2026-09-24): "n_patients" is legacy naming and holds distinct ICU STAYS;
        # "patient_level" is the per-stay AUROC (one max probability per stay, then AUROC) — the
        # quantity the manuscript/TABLES call "Stay-level AUROC" (0.7673 here, 0.7391 external).
        "n_patients":int(F.stay_id.nunique()),
        "n_stays":int(F.stay_id.nunique()),
        "caliber_note":"n_patients/n_stays = distinct ICU stays (stay_id.nunique()); distinct patients are not counted in this file"},
 "select_auc":round(float(best_auc),4),
 "intval":{"auroc":round(float(roc_auc_score(yiv,p)),4),
           "ci":[round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4)],
           "auprc":round(float(average_precision_score(yiv,p)),4),
           "patient_level":round(float(pat_auc),4),
           "patient_level_note":"per-stay AUROC = AUROC over one max probability per ICU stay (manuscript 'Stay-level AUROC')"},
 "n_features":len(feats),"top15":{k:int(v) for k,v in imp.head(15).items()},
 "comparison_tbi_only":{"gbm_ckpt":0.7402,"gbm_patient":0.7525}}
pd.DataFrame({"stay_id":iv.stay_id.values,"t_hr":iv.t_hr.values,"y":yiv,"p":p}).to_parquet(f"{R}/n6_abi_gbm_preds.parquet",index=False)
joblib.dump({"feats":feats,"model":best_m}, f"{R}/n6_abi_gbm_frozen.joblib")
with open(f"{R}/n6_abi_gbm.json","w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=1)
print(json.dumps({k:out[k] for k in ("pop","select_auc","intval","comparison_tbi_only")},ensure_ascii=False,indent=1))
print("TOP10:", dict(list(imp.head(10).items())))
