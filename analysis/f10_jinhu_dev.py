# -*- coding: utf-8 -*-
"""f10: STANDALONE Jinhua-development pilot — replicates the exact M4 pipeline (f1/f2 protocol)
with JinhuaNSICU as the development dataset. Not part of the M4/eICU manuscript.

Protocol (identical machinery):
 1. Candidate pool = features computable in Jinhua (drop the 43 all-NaN columns from the 268)
 2. E02-corrected labels (same code), window [t, t+54 h)
 3. Subject-level 80/20 split, seed 42, zero-overlap asserted
 4. Feature selection: 5-fold GroupKFold(by subject) CV on TRAIN only,
    LightGBM gain importance (same hyperparameters), cumulative 99% rule -> N features
 5. Tier-1/2/3 models on the selected set; internal-test AUROC + stay-cluster bootstrap CI
 6. Tier-1 OOF recalibration layer (GroupKFold-by-subject, same as f2) + slope/ECE
Output: results/jinhu/f10_jinhua_dev_results.json
"""
import json, warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

R = Path("E:/TBI subtype/09_tbi_aki/results")
J = R / "jinhu"
SEED = 42
CUM_KEEP = 0.99

# ---------------- 1. features + subject map ----------------
jf = pd.read_parquet(J / "jinhu_features.parquet").rename(columns={"stay_id": "hadm_id"})
all_nan = [c for c in jf.columns if c not in ("hadm_id", "t_hr") and jf[c].isna().all()]
CAND = [c for c in jf.columns if c not in ("hadm_id", "t_hr") and c not in all_nan]
print(f"candidate pool (Jinhua-computable): {len(CAND)} (dropped {len(all_nan)} all-NaN)")

con = duckdb.connect(); con.execute("SET threads=4")
DATA = "F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data"
sub = con.execute(f"""SELECT hadm_id, subject_id FROM read_csv_auto('{DATA}/medical_record_front_page.csv', all_varchar=true)""").df()
sub["hadm_id"] = sub["hadm_id"].astype(str)
jf["hadm_id"] = jf["hadm_id"].astype(str)
s2sub = dict(zip(sub.hadm_id, sub.subject_id))
jf["subject"] = jf.hadm_id.map(s2sub)
assert jf.subject.notna().all(), "unmapped hadms"
n_hadm, n_subj = jf.hadm_id.nunique(), jf.subject.nunique()
print(f"stays {n_hadm} | subjects {n_subj}")

# ---------------- 2. labels (f6 machinery, verbatim) ----------------
labs = pd.read_parquet(J / "jinhu_labs.parquet")
coh = pd.read_parquet(J / "jinhu_cohort.parquet").set_index("hadm_id")
cr = labs[labs.fam == "cr"][["hadm_id", "tmin", "val"]].copy()
cr["hadm_id"] = cr.hadm_id.astype(str)
cr["hr"] = (cr.tmin - cr.hadm_id.map(coh.icu_start)) / 60.0
cr = cr[cr.val.between(0.1, 30)].sort_values(["hadm_id", "hr"])
cr["t_hr"] = (cr.hr // 6) * 6
cb = cr.groupby(["hadm_id", "t_hr"]).val.max().reset_index()
mg = jf[["hadm_id", "t_hr", "cr_base"]].drop_duplicates().merge(cb, on=["hadm_id", "t_hr"], how="left").sort_values(["hadm_id", "t_hr"])
mg["f"] = np.nan
for s in range(0, 9):
    sh = mg.groupby("hadm_id").val.shift(-s)
    mg["f"] = pd.DataFrame({"c": mg["f"], "n": sh}).max(axis=1)
fold = mg["f"] / mg.cr_base.clip(lower=0.1)
rise = mg["f"] - mg.cr_base
aki = (fold >= 1.5) | (rise >= 0.3)
st = pd.Series(0, index=mg.index)
st[aki.fillna(False)] = 1
st[fold >= 2.0] = 2
st[(fold >= 3.0) | ((mg["f"] >= 4.0) & aki)] = 3
st[mg["f"].isna()] = 0
for t in [1, 2, 3]:
    mg[f"ge{t}"] = (st >= t).astype(int)
print(f"labels: ge1={int(mg.ge1.sum())} ge2={int(mg.ge2.sum())} ge3={int(mg.ge3.sum())} (checkpoints)")

jf = jf.merge(mg[["hadm_id", "t_hr", "ge1", "ge2", "ge3"]], on=["hadm_id", "t_hr"], how="left")

# ---------------- 3. subject-level split ----------------
subjects = np.array(sorted(jf.subject.unique()))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = jf.subject.isin(train_subj).values
te = ~tr
assert not (set(jf.subject[tr]) & set(jf.subject[te]))
print(f"split: train {tr.sum()} / test {te.sum()} checkpoints "
      f"({len(train_subj)}/{len(subjects)-len(train_subj)} subjects); overlap asserted 0")

# ---------------- 4. selection: CV gain, cumulative 99% ----------------
PARAMS_SEL = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
                  class_weight="balanced", random_state=SEED, verbose=-1, importance_type="gain")
Xtr = jf[CAND].values[tr]
ytr = jf.ge1.values[tr]
groups = jf.subject.values[tr]
acc = np.zeros(len(CAND))
gkf = GroupKFold(n_splits=5)
for k, (a, b) in enumerate(gkf.split(Xtr, ytr, groups), 1):
    m = lgb.LGBMClassifier(**PARAMS_SEL)
    m.fit(Xtr[a], ytr[a])
    imp = m.feature_importances_.astype(float)
    acc += imp / imp.sum()
    print(f"  fold {k}/5", flush=True)
imp = acc / 5
order = np.argsort(imp)[::-1]
cum = np.cumsum(imp[order]) / imp.sum()
curve = {f"{p:.1%}": int(np.searchsorted(cum, p) + 1) for p in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)}
nk = min(int(np.searchsorted(cum, CUM_KEEP) + 1), len(CAND))
SEL = [CAND[i] for i in order[:nk]]
print(f"curve: {curve}")
print(f"99% rule -> {nk} features selected")
(feat_share_all_nan := None)

# ---------------- 5. three endpoint models ----------------
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight="balanced", random_state=SEED, verbose=-1)

def stay_boot(y, p, sid, n=500, seed=SEED):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"s": sid, "y": y, "p": p})
    uniq = df.s.unique()
    idx = {s_: np.where(df.s.values == s_)[0] for s_ in uniq}
    yv, pv, sv = df.y.values, df.p.values, df.s.values
    aucs = []
    for _ in range(n):
        sel = r.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[s_] for s_ in sel])
        if yv[ii].sum() > 0 and (1 - yv[ii]).sum() > 0:
            aucs.append(roc_auc_score(yv[ii], pv[ii]))
    return np.percentile(aucs, [2.5, 97.5]).round(4).tolist()

res = {}
sid_te = jf.hadm_id.values[te]
for t in [1, 2, 3]:
    mod = lgb.LGBMClassifier(**PARAMS)
    mod.fit(jf[SEL].values[tr], jf[f"ge{t}"].values[tr])
    pte = mod.predict_proba(jf[SEL].values[te])[:, 1]
    yte = jf[f"ge{t}"].values[te]
    if yte.sum() > 0 and (1 - yte).sum() > 0:
        au = round(float(roc_auc_score(yte, pte)), 4)
        ap = round(float(average_precision_score(yte, pte)), 4)
        ci = stay_boot(yte, pte, sid_te)
    else:
        au, ap, ci = None, None, None
    res[f"ge{t}"] = {"internal_test_auroc": au, "ci": ci, "auprc": ap,
                     "test_pos_ckpt": int(yte.sum()),
                     "test_event_stays": int(jf.loc[te & (jf[f'ge{t}'] == 1), 'hadm_id'].nunique())}
    print(f"ge{t}: AUROC={au} CI={ci} AUPRC={ap} pos={int(yte.sum())}")

# ---------------- 6. Tier-1 OOF calibration ----------------
oof = np.zeros(len(ytr))
for k, (a, b) in enumerate(gkf.split(Xtr, ytr, groups), 1):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(jf[SEL].values[tr][a], ytr[a])
    oof[b] = m.predict_proba(jf[SEL].values[tr][b])[:, 1]
def logit(p): p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))
cal = LogisticRegression(C=1e6).fit(logit(oof).reshape(-1, 1), ytr)
a1, b1 = float(cal.intercept_[0]), float(cal.coef_[0][0])
def ece10(y, p):
    q = pd.qcut(pd.Series(p), 10, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return round(float((g.size() / len(y) * np.abs(g.y.mean() - g.p.mean())).sum()), 4)
yte1 = jf.ge1.values[te]
pte1 = res["ge1"]["internal_test_auroc"]
pte_raw = mod.predict_proba(jf[SEL].values[te])[:, 1]  # placeholder replaced below

mod1 = lgb.LGBMClassifier(**PARAMS)
mod1.fit(jf[SEL].values[tr], ytr)
p1te = mod1.predict_proba(jf[SEL].values[te])[:, 1]
pr1te = sigmoid(a1 + b1 * logit(p1te))
def mets(y, p):
    lr_ = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return {"slope": round(float(lr_.coef_[0][0]), 3), "ece": ece10(y, p)}
calib = {"a": round(a1, 6), "b": round(b1, 6),
         "test_raw": mets(yte1, p1te), "test_recal": mets(yte1, pr1te)}
print("calibration:", calib)

eppp = {"params": nk,
        "ckpt_level_dev": round(float(jf.ge1.values[tr].sum()) / nk, 1),
        "patient_level_dev": round(float(jf.loc[tr & (jf.ge1 == 1), 'subject'].nunique()) / nk, 2)}

out = {"_meta": {"protocol": "standalone Jinhua-development pilot; identical machinery to the M4 pipeline (f1/f2)",
                 "date": "2026-10-02", "seed": SEED,
                 "note": "exploration only; separate from the M4/eICU manuscript"},
       "candidate_pool": len(CAND), "all_nan_dropped": len(all_nan),
       "stays": n_hadm, "subjects": n_subj,
       "split": {"train_ckpt": int(tr.sum()), "test_ckpt": int(te.sum()),
                 "train_subj": len(train_subj), "test_subj": int(len(subjects) - len(train_subj))},
       "curve_n_features": curve, "n_selected": nk,
       "results": res, "calibration_ge1": calib, "eppp": eppp,
       "train_events_ge1_ckpt": int(jf.ge1.values[tr].sum()),
       "train_event_subjects_ge1": int(jf.loc[tr & (jf.ge1 == 1), 'subject'].nunique())}
json.dump(out, open(J / "f10_jinhua_dev_results.json", "w"), indent=2)
print("\nSaved f10_jinhua_dev_results.json")
print(json.dumps({k: out[k] for k in ("candidate_pool", "n_selected", "curve_n_features", "eppp")}, indent=1))
