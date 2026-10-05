# -*- coding: utf-8 -*-
"""f14: Round-11.1 gap fill (three items Biomni could not complete).
  A. Jinhua upstream funnel from f4/f5 reports + local parquet verification
  B. Near-miss creatinine share among false-alert stays at thr 0.15 (Paper-1 SM5 protocol:
     fold>=1.3 or rise>=0.2 vs baseline; denominator = never-event alerted stays) - eICU + M4
  C. Tier-2 (ge3) OOF recalibration layer under the ORIGINAL f9 protocol:
     5-fold GroupKFold-by-subject OOF on the Jinhua training split -> frozen logistic layer
     -> slopes/ECE in int/M4/eICU + threshold views (0.01-0.20)
Output: results/jinhu/f14_gap_fill.json
Deterministic rebuild of the f12 machinery (seed 42); no other refitting.
"""
import json, warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

R = Path("E:/TBI subtype/09_tbi_aki/results")
J = R / "jinhu"
SEED = 42
A1, B1 = 0.005067, 0.268467   # frozen Tier-1 layer (f12_full_suite.json)
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight="balanced", random_state=SEED, verbose=-1)

def logit(p): p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))

out = {}

# ============ A. Jinhua upstream funnel ============
print("=== A. funnel ===", flush=True)
f4 = json.load(open(J / "f4_extract_report.json"))
f5 = json.load(open(J / "f5_report.json"))
labs = pd.read_parquet(J / "jinhu_labs.parquet")
cr = labs[labs.fam == "cr"].copy()
cr["hadm_id"] = cr.hadm_id.astype(str)
coh = pd.read_parquet(J / "jinhu_cohort.parquet").set_index("hadm_id")
coh.index = coh.index.astype(str)
cr["hr"] = (cr.tmin - cr.hadm_id.map(coh.icu_start)) / 60.0
crw = cr[cr.hr.between(-24, 168) & cr.val.between(0.1, 30)]
n_with_cr = int(crw.hadm_id.nunique())
funnel = {
    "n_hadm_extracted": f4["n_hadms"],
    "n_with_cr_in_window": n_with_cr,
    "axis_status": f5["axis_status"],
    "rrt_le24h_excluded": f5["rrt_excluded"],
    "ok_after_rrt": f5["axis_status"]["ok"] - f5["rrt_excluded"],
    "grid_stays": f5["grid_stays"],
    "grid_attrition_unexplained": f5["axis_status"]["ok"] - f5["rrt_excluded"] - f5["grid_stays"],
    "grid_checkpoints": f5["grid_checkpoints"],
    "source": "f4_extract_report.json + f5_report.json + jinhu_labs/cohort recount",
}
print(f"  {funnel['n_hadm_extracted']} extracted -> {n_with_cr} with Cr(-24,168] -> ok {funnel['axis_status']['ok']} "
      f"(prevalent {funnel['axis_status']['prevalent_24h']}, no_base {funnel['axis_status']['no_base']}) "
      f"-> -{funnel['rrt_le24h_excluded']} RRT -> {funnel['grid_stays']} grid stays "
      f"(attrition {funnel['grid_attrition_unexplained']})", flush=True)
out["funnel"] = funnel

# ============ B. near-miss creatinine among false-alert stays (thr 0.15) ============
print("=== B. near-miss Cr ===", flush=True)
def near_miss(tag, preds_path, cr_df, base_df, key):
    d = pd.read_parquet(preds_path)
    d["stay_id"] = d.stay_id.astype(str)
    d["p_rec"] = sigmoid(A1 + B1 * logit(d.p.values))
    d["alert"] = d.p_rec >= 0.15
    alerts = set(d.loc[d.alert, "stay_id"])
    lastpos = set(d[d.y == 1].stay_id)
    false_stays = alerts - lastpos
    base = base_df.drop_duplicates("stay_id").set_index("stay_id").cr_base
    c = cr_df.copy()
    c["fold"] = c.cr_val / c.stay_id.map(base).clip(lower=0.1)
    c["rise"] = c.cr_val - c.stay_id.map(base)
    c = c[c.stay_id.isin(false_stays)]
    per = c.groupby("stay_id")[["fold", "rise"]].max()
    n_nm = int(((per.fold >= 1.3) | (per.rise >= 0.2)).sum()); n_f = len(false_stays)
    return {"cohort": tag, "false_alert_stays": n_f, "near_miss_stays": n_nm,
            "near_miss_share": round(n_nm / n_f, 3) if n_f else None,
            "definition": ">=1 measured creatinine with fold>=1.3 or rise>=0.2 mg/dL vs baseline (Paper-1 SM5 protocol)"}

# eICU
elab = pd.read_parquet(R / "n8_2_labs_series.parquet")
ecr = elab[(elab.labname == "creatinine") & (elab.labtypeid == 1)].rename(columns={"sid": "stay_id", "labresult": "cr_val"})
ecr = ecr[ecr.cr_val.between(0.1, 30)][["stay_id", "cr_val"]].copy()
ecr["stay_id"] = ecr.stay_id.astype(str)
eif = pd.read_parquet(R / "p1_expanded_features.parquet")[["stay_id", "cr_base"]]
eif["stay_id"] = eif.stay_id.astype(str)
out["near_miss_eICU"] = near_miss("eICU", J / "f12_preds_eICU_ge1.parquet", ecr, eif, "stay_id")
print("  eICU:", out["near_miss_eICU"]["near_miss_stays"], "/", out["near_miss_eICU"]["false_alert_stays"],
      "=", out["near_miss_eICU"]["near_miss_share"], flush=True)

# M4
mlab = pd.read_parquet(R / "n2_m4_abi_cr_serial.parquet").rename(columns={"valuenum": "cr_val"})
mlab = mlab[mlab.cr_val.between(0.1, 30)][["stay_id", "cr_val"]].copy()
mlab["stay_id"] = mlab.stay_id.astype(str)
m4b = pd.read_parquet(R / "n5_abi_rolling_features.parquet")[["stay_id", "cr_base"]]
m4b["stay_id"] = m4b.stay_id.astype(str)
out["near_miss_M4"] = near_miss("M4", J / "f12_preds_M4_ge1.parquet", mlab, m4b, "stay_id")
print("  M4:", out["near_miss_M4"]["near_miss_stays"], "/", out["near_miss_M4"]["false_alert_stays"],
      "=", out["near_miss_M4"]["near_miss_share"], flush=True)

# ============ C. Tier-2 OOF layer (original f9 protocol) ============
print("=== C. Tier-2 OOF layer (f12 machinery rebuild) ===", flush=True)
jf = pd.read_parquet(J / "jinhu_features.parquet").rename(columns={"stay_id": "hadm_id"})
jf["hadm_id"] = jf.hadm_id.astype(str)
all_nan = [c for c in jf.columns if c not in ("hadm_id", "t_hr") and jf[c].isna().all()]
CAND = [c for c in jf.columns if c not in ("hadm_id", "t_hr") and c not in all_nan]

con = duckdb.connect(); con.execute("SET threads=4")
DATA = "F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data"
sub = con.execute(f"SELECT hadm_id, subject_id FROM read_csv_auto('{DATA}/medical_record_front_page.csv', all_varchar=true)").df()
sub["hadm_id"] = sub.hadm_id.astype(str)
jf["subject"] = jf.hadm_id.map(dict(zip(sub.hadm_id, sub.subject_id)))

crj = labs[labs.fam == "cr"][["hadm_id", "tmin", "val"]].copy()
crj["hadm_id"] = crj.hadm_id.astype(str)
crj["hr"] = (crj.tmin - crj.hadm_id.map(coh.icu_start)) / 60.0
crj = crj[crj.val.between(0.1, 30)].sort_values(["hadm_id", "hr"])
crj["t_hr"] = (crj.hr // 6) * 6
cb = crj.groupby(["hadm_id", "t_hr"]).val.max().reset_index()
mg = jf[["hadm_id", "t_hr", "cr_base"]].drop_duplicates().merge(cb, on=["hadm_id", "t_hr"], how="left").sort_values(["hadm_id", "t_hr"])
mg["f"] = np.nan
for s in range(9):
    sh = mg.groupby("hadm_id").val.shift(-s)
    mg["f"] = pd.DataFrame({"c": mg["f"], "n": sh}).max(axis=1)
fold = mg.f / mg.cr_base.clip(lower=0.1); rise = mg.f - mg.cr_base; aki = (fold >= 1.5) | (rise >= 0.3)
st = pd.Series(0, index=mg.index)
st[aki.fillna(False)] = 1; st[fold >= 2.0] = 2; st[(fold >= 3.0) | ((mg.f >= 4.0) & aki)] = 3; st[mg.f.isna()] = 0
for t in [1, 2, 3]: mg[f"ge{t}"] = (st >= t).astype(int)
jf = jf.merge(mg[["hadm_id", "t_hr", "ge1", "ge2", "ge3"]], on=["hadm_id", "t_hr"], how="left")

subjects = np.array(sorted(jf.subject.unique()))
np.random.seed(SEED); perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = jf.subject.isin(train_subj).values; te = ~tr

Xtr_sel = jf[CAND].values[tr]; ytr_sel = jf.ge1.values[tr]; groups_sel = jf.subject.values[tr]
acc = np.zeros(len(CAND))
for a, b in GroupKFold(5).split(Xtr_sel, ytr_sel, groups_sel):
    m = lgb.LGBMClassifier(**{**PARAMS, "importance_type": "gain"}); m.fit(Xtr_sel[a], ytr_sel[a])
    imp = m.feature_importances_.astype(float); acc += imp / imp.sum()
imp_mean = acc / 5
order = np.argsort(imp_mean)[::-1]
cum = np.cumsum(imp_mean[order]) / imp_mean.sum()
SEL = [CAND[i] for i in order[:int(np.searchsorted(cum, 0.99) + 1)]]
print(f"  rebuilt selection: {len(CAND)} -> {len(SEL)} (must be 166)", flush=True)
assert len(SEL) == 166, "SELECTION MISMATCH vs f12"

m3 = lgb.LGBMClassifier(**PARAMS); m3.fit(jf[SEL].values[tr], jf.ge3.values[tr])

# Tier-2 OOF on the training split (the missing f9-protocol piece)
ytr3 = jf.ge3.values[tr]
oof3 = np.zeros(len(ytr3))
for a, b in GroupKFold(5).split(Xtr_sel, ytr3, groups_sel):
    m = lgb.LGBMClassifier(**PARAMS); m.fit(jf[SEL].values[tr][a], ytr3[a])
    oof3[b] = m.predict_proba(jf[SEL].values[tr][b])[:, 1]
cal3 = LogisticRegression(C=1e6).fit(logit(oof3).reshape(-1, 1), ytr3)
A3, B3 = float(cal3.intercept_[0]), float(cal3.coef_[0][0])
print(f"  Tier-2 OOF layer: a={A3:.6f} b={B3:.6f} (OOF positives {int(ytr3.sum())})", flush=True)

def ece10(y, p):
    q = pd.qcut(pd.Series(p), 10, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return round(float((g.size() / len(y) * np.abs(g.y.mean() - g.p.mean())).sum()), 4)
def cal_mets(y, p):
    lr_ = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return {"slope": round(float(lr_.coef_[0][0]), 3), "ece": ece10(y, p)}

def deploy_v2(df, thr):
    d = df.copy(); d["alert"] = sigmoid(A3 + B3 * logit(d.p)) >= thr
    lastpos = d[d.y == 1].groupby("s").t_hr.max(); firstpos = d[d.y == 1].groupby("s").t_hr.min()
    capt, leads = [], []
    for sid, g in d[d.s.isin(lastpos.index)].groupby("s"):
        al = g[g["alert"] & (g.t_hr < lastpos[sid])]
        if len(al): capt.append(sid); leads.append(lastpos[sid] - al.t_hr.min())
    alerts = set(d.loc[d["alert"], "s"]); nc = len(capt)
    nev = d[~d.s.isin(lastpos.index)].groupby("s").t_hr.max().clip(lower=6).div(24).sum()
    evf = firstpos.clip(lower=6).div(24).sum()
    return {"thr": thr, "capture": round(nc / len(lastpos), 3),
            "lead_median_h": round(float(np.median(leads)), 1) if leads else None,
            "NNE": round(len(alerts) / nc, 2) if nc else None,
            "FA_per_100ptd": round(100 * len(alerts - set(lastpos.index)) / (nev + evf), 2),
            "pct_stays_alerted": round(100 * len(alerts) / d.s.nunique(), 1),
            "n_event_stays": int(len(lastpos)), "n_captured": int(nc)}

tier2 = {"layer": {"a": round(A3, 6), "b": round(B3, 6), "oof_pos_ckpt": int(ytr3.sum()),
                   "oof_pos_stays": int(jf.loc[tr & (jf.ge3 == 1), "hadm_id"].nunique()),
                   "protocol": "5-fold GroupKFold by subject on Jinhua training split (f9/Tier-1 identical)"}}
for nm, key in [("int", "s"), ("M4", "stay_id"), ("eICU", "stay_id")]:
    if nm == "int":
        d = pd.DataFrame({"s": jf.hadm_id.values[te], "t_hr": jf.t_hr.values[te],
                          "y": jf.ge3.values[te], "p": m3.predict_proba(jf[SEL].values[te])[:, 1]})
    else:
        src = {"M4": None, "eICU": None}
        df_ext = pd.read_parquet(J / f"f12_preds_{nm}_ge3.parquet")
        d = df_ext.rename(columns={"stay_id": "s"})
    y, p = d.y.values, d.p.values
    pr = sigmoid(A3 + B3 * logit(p))
    tier2[nm] = {"raw": cal_mets(y, p), "layer_applied": cal_mets(y, pr),
                 "n_pos_ckpt": int(y.sum()), "n_event_stays": int(d[d.y == 1].s.nunique())}
    tier2[nm]["thresholds"] = {f"thr{t}": deploy_v2(d, t) for t in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]}
    print(f"  {nm}: raw {tier2[nm]['raw']} -> layer {tier2[nm]['layer_applied']}", flush=True)
out["tier2_oof_layer"] = tier2

out["_meta"] = {"date": "2026-10-03", "purpose": "round-11.1 gap fill: funnel / near-miss Cr / Tier-2 OOF layer (original protocol)",
                "deterministic": "f12 machinery rebuild seed 42; selection asserted == 166"}
json.dump(out, open(J / "f14_gap_fill.json", "w"), indent=2)
print("\nSaved f14_gap_fill.json")
print("DONE")
