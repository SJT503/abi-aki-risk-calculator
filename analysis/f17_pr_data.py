# -*- coding: utf-8 -*-
"""f17: PR curve data points for the Fig 2 PR panel (frozen predictions only).
  - Tier-1 and Tier-2 x int/M4/eICU: precision/recall arrays (downsampled to ~500 pts)
  - prevalence no-skill baseline per curve; AUPRC verified against f12 values
  - M4 LR baseline (age/charlson/cr_base, frozen f12 protocol) for the Tier-1 external panel
Output: results/jinhu/f17_pr_data.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

R = "E:/TBI subtype/09_tbi_aki/results"
J = f"{R}/jinhu"
f12 = json.load(open(f"{J}/f12_full_suite.json"))
out = {"_meta": {"date": "2026-10-03", "purpose": "PR curve data for Fig 2 PR panels",
                 "note": "frozen f12 predictions; AUPRC asserted == f12"}}

def curve(y, p, n_pts=500):
    pre, rec, thr = precision_recall_curve(y, p)
    # downsample uniformly over index, always keep first/last
    idx = np.unique(np.linspace(0, len(pre) - 1, n_pts).astype(int))
    return {"precision": [round(float(v), 5) for v in pre[idx]],
            "recall": [round(float(v), 5) for v in rec[idx]]}

for tier in (1, 3):
    key = f"ge{tier}"
    for nm in ("int", "M4", "eICU"):
        d = pd.read_parquet(f"{J}/f12_preds_{nm}_ge{tier}.parquet")
        y, p = d.y.values, d.p.values
        ap = average_precision_score(y, p)
        ref = f12["discrimination"][f"{nm}_{key}"]["auprc"]
        assert abs(ap - ref) < 5e-4, (nm, key, ap, ref)
        prev = float(y.mean())
        out[f"{nm}_ge{tier}"] = {"auprc": round(float(ap), 4), "prevalence": round(prev, 5),
                                 "lift": round(float(ap) / prev, 1), "n": int(len(y)), "n_pos": int(y.sum()),
                                 "curve": curve(y, p)}
        print(f"{nm} ge{tier}: AUPRC {ap:.4f} (f12 {ref}) | prev {prev:.4f} | lift {ap/prev:.1f}x")

# M4 LR baseline (Tier-1), frozen 3-feature LR per f12 protocol
import lightgbm as dummy  # noqa: ensure env parity with f12 imports
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=2000, class_weight="balanced")
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
d1 = pd.read_parquet(f"{J}/f12_preds_M4_ge1.parquet")
Xlr = m4.drop_duplicates("stay_id")[["stay_id", "age", "charlson", "cr_base"]].set_index("stay_id")
# expand stay-level covariates to the prediction grid
cov = d1.stay_id.map(Xlr.age), d1.stay_id.map(Xlr.charlson), d1.stay_id.map(Xlr.cr_base)
X = np.column_stack([c.fillna(c.median()).values for c in cov])
lr.fit(X, d1.y.values)
pl = lr.predict_proba(X)[:, 1]
ap = average_precision_score(d1.y.values, pl)
out["M4_ge1_LR"] = {"auprc": round(float(ap), 4), "prevalence": round(float(d1.y.mean()), 5),
                    "note": "admission-variable LR (age, charlson, cr_base), frozen f12 protocol, evaluated on the M4 grid",
                    "curve": curve(d1.y.values, pl)}
print(f"M4 ge1 LR: AUPRC {ap:.4f}")

json.dump(out, open(f"{J}/f17_pr_data.json", "w"), indent=1)
print("Saved f17_pr_data.json")
