# -*- coding: utf-8 -*-
"""f9: Tier-2 (severe AKI) OOF calibration layer + threshold mini-table (round-9 B8 fill-in).
GroupKFold-by-subject OOF recalibration identical to the Tier-1 protocol (f2),
applied to the internal test set and eICU external. Deployment via the documented
v2 definition anchored on the final positive ge3 checkpoint.
Output: results/f9_tier2_calibration.json
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

R = "results"
SEED = 42
SEL = json.load(open(f"{R}/_reselected_features.json"))
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight="balanced", random_state=SEED, verbose=-1)
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[["stay_id"] + SEL]
eicu = pd.read_parquet(f"{R}/p1_expanded_features.parquet")[["stay_id"] + SEL]

def labels(labs_parquet, feat_df):
    labs = pd.read_parquet(labs_parquet)
    if "sid" in labs.columns:
        cr = labs[labs.labname == "creatinine"].rename(columns={"sid": "stay_id", "labresult": "cr_val"})
    else:
        cr = labs.rename(columns={"valuenum": "cr_val"})
    cr = cr[cr.cr_val.between(0.1, 30)].sort_values(["stay_id", "hr"])
    cr["t_hr"] = (cr.hr // 6) * 6
    cb = cr.groupby(["stay_id", "t_hr"]).cr_val.max().reset_index().sort_values(["stay_id", "t_hr"])
    mg = feat_df[["stay_id", "t_hr", "cr_base"]].drop_duplicates().merge(cb, on=["stay_id", "t_hr"], how="left").sort_values(["stay_id", "t_hr"])
    mg["f"] = np.nan
    for s in range(9):
        sh = mg.groupby("stay_id").cr_val.shift(-s)
        mg["f"] = pd.DataFrame({"c": mg["f"], "n": sh}).max(axis=1)
    fo = mg["f"] / mg.cr_base.clip(lower=0.1)
    ri = mg["f"] - mg.cr_base
    ak = (fo >= 1.5) | (ri >= 0.3)
    st = pd.Series(0, index=mg.index)
    st[ak.fillna(False)] = 1
    st[fo >= 2.0] = 2
    st[(fo >= 3.0) | ((mg["f"] >= 4.0) & ak)] = 3
    st[mg["f"].isna()] = 0
    mg["ge3"] = (st >= 3).astype(int)
    return mg[["stay_id", "t_hr", "ge3"]]

m4f = m4.merge(labels(f"{R}/n2_m4_abi_cr_serial.parquet", m4), on=["stay_id", "t_hr"], how="left")
eif = eicu.merge(labels(f"{R}/n8_2_labs_series.parquet", eicu), on=["stay_id", "t_hr"], how="left")
assert m4f.ge3.sum() == 226 and eif.ge3.sum() == 269, (m4f.ge3.sum(), eif.ge3.sum())

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=["stay_id", "subject_id"])
s2sub = dict(zip(n1.stay_id, n1.subject_id))
subs = np.array(sorted({s2sub[s] for s in m4f.stay_id.unique()}))
np.random.seed(SEED)
perm = np.random.permutation(len(subs))
train_subj = set(subs[perm[:int(0.8 * len(subs))]])
tr = m4f.stay_id.map(s2sub).isin(train_subj).values
te = ~tr
ytr = m4f.ge3.values[tr]
Xtr = m4f[SEL].values[tr]
gtr = m4f.stay_id.map(s2sub).values[tr]

mod = lgb.LGBMClassifier(**PARAMS)
mod.fit(Xtr, ytr)
pte = mod.predict_proba(m4f[SEL].values[te])[:, 1]
pee = mod.predict_proba(eif[SEL].values)[:, 1]
yte, yee = m4f.ge3.values[te], eif.ge3.values

oof = np.zeros(len(ytr))
for k, (a, b) in enumerate(GroupKFold(5).split(Xtr, ytr, gtr), 1):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xtr[a], ytr[a])
    oof[b] = m.predict_proba(Xtr[b])[:, 1]
    print(f"OOF fold {k}/5", flush=True)

def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

cal = LogisticRegression(C=1e6).fit(logit(oof).reshape(-1, 1), ytr)
a3, b3 = float(cal.intercept_[0]), float(cal.coef_[0][0])

def ece10(y, p):
    q = pd.qcut(pd.Series(p), 10, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return round(float((g.size() / len(y) * np.abs(g.y.mean() - g.p.mean())).sum()), 4)

def mets(y, p):
    lr_ = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return {"slope": round(float(lr_.coef_[0][0]), 3), "ece": ece10(y, p)}

res = {"_meta": {"protocol": "Tier-2 OOF recalibration, GroupKFold(5) by subject on training split (identical to Tier-1 f2 protocol)",
                 "date": "2026-10-01"},
       "a": round(a3, 6), "b": round(b3, 6),
       "int_test": {"raw": mets(yte, pte), "recal": mets(yte, sigmoid(a3 + b3 * logit(pte)))},
       "ext": {"raw": mets(yee, pee), "recal": mets(yee, sigmoid(a3 + b3 * logit(pee)))},
       "event_rate_ext": round(float(yee.mean()), 4)}
print("Tier-2 recal a,b =", res["a"], res["b"])
print("int:", res["int_test"])
print("ext:", res["ext"])

pe3 = pd.DataFrame({"stay_id": eif.stay_id.values, "y": yee, "p": pee,
                    "p_rec": sigmoid(a3 + b3 * logit(pee)), "t_hr": eif.t_hr.values})

def deploy_v2(df, thr):
    d = df.copy()
    d["alert"] = d["p_rec"] >= thr
    lastpos = d[d.y == 1].groupby("stay_id")["t_hr"].max()
    firstpos = d[d.y == 1].groupby("stay_id")["t_hr"].min()
    capt, leads = [], []
    for sid, g in d[d.stay_id.isin(lastpos.index)].groupby("stay_id"):
        al = g[g["alert"] & (g["t_hr"] < lastpos[sid])]
        if len(al):
            capt.append(sid)
            leads.append(lastpos[sid] - al["t_hr"].min())
    alerts = set(d.loc[d["alert"], "stay_id"])
    nc = len(capt)
    nev = d[~d.stay_id.isin(lastpos.index)].groupby("stay_id")["t_hr"].max().clip(lower=6).div(24).sum()
    evf = firstpos.clip(lower=6).div(24).sum()
    return {"thr": thr, "capture": round(nc / len(lastpos), 3),
            "lead_median_h": round(float(np.median(leads)), 1) if leads else None,
            "lead_iqr": [round(float(np.percentile(leads, 25)), 1), round(float(np.percentile(leads, 75)), 1)] if leads else None,
            "NNE": round(len(alerts) / nc, 2) if nc else None,
            "FA_per_100ptd": round(100 * len(alerts - set(lastpos.index)) / (nev + evf), 2),
            "pct_stays_alerted": round(100 * len(alerts) / d.stay_id.nunique(), 1)}

res["deployment_ext_ge3"] = {f"thr{t}": deploy_v2(pe3, t) for t in [0.05, 0.10, 0.15, 0.20]}
for t, v in res["deployment_ext_ge3"].items():
    print(t, v)
json.dump(res, open(f"{R}/f9_tier2_calibration.json", "w"), indent=2)
print("saved f9_tier2_calibration.json")
