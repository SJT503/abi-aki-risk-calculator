# -*- coding: utf-8 -*-
"""s1e: hospital mapping (from eICU-CRD source icustays) + per-hospital AUROC
+ two-level hospital bootstrap CI + sex subgroup for the >=Stage 1 external
validation. Updates s1_stage_primary.json in place."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

R = "results"
pe = pd.read_parquet(f"{R}/s1_preds_external.parquet")
hmap = pd.read_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet")[["stay_id", "hospitalid"]]
hmap = hmap[hmap["stay_id"].isin(set(pe["stay_id"]))]
pe = pe.merge(hmap, on="stay_id", how="left")
assert pe["hospitalid"].notna().all()
pe[["stay_id", "hospitalid"]].drop_duplicates().to_parquet(f"{R}/s1_hospital_map.parquet", index=False)
print("hospitals:", pe["hospitalid"].nunique(), "stays:", pe["stay_id"].nunique())

hp = pe.groupby("hospitalid").apply(lambda g: roc_auc_score(g.y, g.p) if g.y.sum() >= 20 else np.nan).dropna()
extra_h = {"n_ge20": int(len(hp)), "median": round(float(hp.median()), 3),
           "min": round(float(hp.min()), 3), "max": round(float(hp.max()), 3),
           "n_below_060": int((hp < 0.60).sum())}
print("per-hospital:", extra_h)

# two-level bootstrap: hospitals with replacement, then stays within
r = np.random.default_rng(42)
hs = pd.unique(pe["hospitalid"].values)
idx_by_h = {h: np.where(pe["hospitalid"].values == h)[0] for h in hs}
yv, pv = pe["y"].values, pe["p"].values
aucs = []
for _ in range(1000):
    sel = r.choice(hs, size=len(hs), replace=True)
    idx = np.concatenate([idx_by_h[h] for h in sel])
    yy, pp = yv[idx], pv[idx]
    if yy.sum() > 0 and (1 - yy).sum() > 0:
        aucs.append(roc_auc_score(yy, pp))
ci2 = np.percentile(aucs, [2.5, 97.5]).round(4).tolist()
print("2-level CI:", ci2)

# sex subgroup ('male' feature)
eif_cols = ["stay_id", "male"]
sex = pd.read_parquet("E:/TBI subtype/09_tbi_aki/results/p1_expanded_features.parquet",
                      columns=eif_cols).drop_duplicates("stay_id")
pe2 = pd.read_parquet(f"{R}/s1_preds_external.parquet").merge(sex, on="stay_id", how="left")
sub = {}
for v, nm in [(1.0, "male=1"), (0.0, "male=0")]:
    g = pe2[pe2["male"] == v]
    sub[nm] = round(float(roc_auc_score(g.y, g.p)), 3)
print("sex subgroups:", sub)

s1 = json.load(open(f"{R}/s1_stage_primary.json"))
s1["extra"]["hospital_auroc"] = extra_h
s1["results"]["ge1"]["ci_ext_hospital2level"] = ci2
s1["extra"].setdefault("subgroups_ext_ge1", {}).update(sub)
json.dump(s1, open(f"{R}/s1_stage_primary.json", "w"), indent=2)
hp.reset_index().to_csv(f"{R}/s1_hospital_auroc.csv", index=False)
print("json + csv updated")
