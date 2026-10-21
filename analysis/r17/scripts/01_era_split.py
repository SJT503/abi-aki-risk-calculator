# -*- coding: utf-8 -*-
"""r17-1  Era (calendar-time) sensitivity for the frozen external MIMIC-IV predictions.

Question: does discrimination erode across the recruitment period (practice
evolution), as Zhang 2026 showed robustness across MIMIC-III vs MIMIC-IV eras?
MIMIC-IV dates are per-patient shifted, so eras are defined RELATIVELY: split
stays at the median admission time into first / second half of the period.

Canary (must pass before any era number is trusted):
  - overall ge1 checkpoint AUROC on this parquet == frozen 0.7694 (manuscript)
  - stay/checkpoint denominators == 7,442 / 75,638
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
R = Path("E:/TBI subtype/09_tbi_aki/results")
OUT = Path("E:/TBI subtype/09_tbi_aki/_analysis_r17/out")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_BOOT = 500  # f12 stay_boot convention for single AUROCs

coh = pd.read_parquet(R / "n1_m4_abi_cohort.parquet")[["stay_id", "intime"]]
coh["ts"] = pd.to_datetime(coh.intime)

res = {}
for t in (1, 3):
    d = pd.read_parquet(R / f"jinhu/f12_preds_M4_ge{t}.parquet")
    d = d.merge(coh[["stay_id", "ts"]], on="stay_id", how="left")
    assert d.ts.notna().all(), "missing intime"

    # ---- canary: overall AUROC + denominators ----
    overall = roc_auc_score(d.y, d.p)
    if t == 1:
        assert abs(overall - 0.7694) < 0.0005, f"canary fail ge1: {overall}"
        assert d.stay_id.nunique() == 7442 and len(d) == 75638, "denominator mismatch"
        print(f"canary ge1 overall={overall:.4f} (frozen 0.7694) stays=7442 ckpts=75638 OK")
    res[f"ge{t}_overall"] = round(float(overall), 4)

    # ---- era split at median admission timestamp (stay level) ----
    stay_ts = d.groupby("stay_id").ts.first()
    med = stay_ts.median()
    d["era"] = np.where(d.ts <= med, "early", "late")

    def stay_boot(dd, n=N_BOOT, seed=SEED):
        rng = np.random.default_rng(seed)
        uniq = dd.stay_id.unique()
        idx = {s: np.where(dd.stay_id.values == s)[0] for s in uniq}
        vals = []
        for _ in range(n):
            sel = rng.choice(uniq, size=len(uniq), replace=True)
            ii = np.concatenate([idx[s] for s in sel])
            if dd.y.values[ii].sum() > 0:
                vals.append(roc_auc_score(dd.y.values[ii], dd.p.values[ii]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return [round(float(lo), 3), round(float(hi), 3)]

    for era in ("early", "late"):
        dd = d[d.era == era]
        a = roc_auc_score(dd.y, dd.p)
        res[f"ge{t}_{era}"] = {"auroc": round(float(a), 4), "ci": stay_boot(dd),
                               "stays": int(dd.stay_id.nunique()),
                               "ckpts": int(len(dd)),
                               "pos_ckpt": int(dd.y.sum()),
                               "event_stays": int(dd[dd.y == 1].stay_id.nunique()),
                               "period": [str(dd.ts.min().date()), str(dd.ts.max().date())]}
        print(f"ge{t} {era}: AUROC={a:.4f} {res[f'ge{t}_{era}']['ci']} "
              f"stays={res[f'ge{t}_{era}']['stays']} events={res[f'ge{t}_{era}']['event_stays']}")

    # ---- unpaired bootstrap of late-early difference (independent resamples) ----
    de, dl = d[d.era == "early"], d[d.era == "late"]
    rng = np.random.default_rng(SEED)
    ue, ul = de.stay_id.unique(), dl.stay_id.unique()
    ie = {s: np.where(de.stay_id.values == s)[0] for s in ue}
    il = {s: np.where(dl.stay_id.values == s)[0] for s in ul}
    diffs = []
    for _ in range(1000):
        je = np.concatenate([ie[s] for s in rng.choice(ue, size=len(ue), replace=True)])
        jl = np.concatenate([il[s] for s in rng.choice(ul, size=len(ul), replace=True)])
        if de.y.values[je].sum() > 0 and dl.y.values[jl].sum() > 0:
            diffs.append(roc_auc_score(dl.y.values[jl], dl.p.values[jl])
                         - roc_auc_score(de.y.values[je], de.p.values[je]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    res[f"ge{t}_late_minus_early"] = {"delta": round(float(diffs.mean()), 3),
                                      "ci": [round(float(lo), 3), round(float(hi), 3)],
                                      "p": round(float(p), 3)}
    print(f"ge{t} late-early: {res[f'ge{t}_late_minus_early']}")

res["split"] = {"rule": "median stay admission timestamp (dates per-patient shifted in MIMIC-IV)",
                "median_ts": str(med)}
json.dump(res, open(OUT / "era_split.json", "w"), indent=1)
print("written:", OUT / "era_split.json")
