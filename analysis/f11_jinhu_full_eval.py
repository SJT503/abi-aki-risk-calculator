# -*- coding: utf-8 -*-
"""f11: Full evaluation suite for the Jinhua-development model (f10).
 (a) Jinhua internal test: AUROC/CI, calibration curve (10 bins), DCA
 (b) SHAP attribution (Tier-1 model, test sample) + beeswarm figure
 (c) EXTERNAL validation of the frozen Jinhua model in MIMIC-IV (M4) and eICU-CRD
     — same label machinery, features missing in either side left NaN (native handling)
Outputs: results/jinhu/f11_full_eval.json + results/jinhu/figs/*.png
"""
import json, warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

R = Path("E:/TBI subtype/09_tbi_aki/results")
J = R / "jinhu"
FIG = J / "figs"; FIG.mkdir(exist_ok=True)
SEED = 42

# ============ 1. rebuild the f10 pipeline deterministically ============
jf = pd.read_parquet(J / "jinhu_features.parquet").rename(columns={"stay_id": "hadm_id"})
jf["hadm_id"] = jf.hadm_id.astype(str)
all_nan = [c for c in jf.columns if c not in ("hadm_id", "t_hr") and jf[c].isna().all()]
CAND = [c for c in jf.columns if c not in ("hadm_id", "t_hr") and c not in all_nan]

con = duckdb.connect(); con.execute("SET threads=4")
DATA = "F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data"
sub = con.execute(f"SELECT hadm_id, subject_id FROM read_csv_auto('{DATA}/medical_record_front_page.csv', all_varchar=true)").df()
sub["hadm_id"] = sub.hadm_id.astype(str)
jf["subject"] = jf.hadm_id.map(dict(zip(sub.hadm_id, sub.subject_id)))

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
st[aki.fillna(False)] = 1; st[fold >= 2.0] = 2
st[(fold >= 3.0) | ((mg["f"] >= 4.0) & aki)] = 3
st[mg["f"].isna()] = 0
for t in [1, 2, 3]:
    mg[f"ge{t}"] = (st >= t).astype(int)
jf = jf.merge(mg[["hadm_id", "t_hr", "ge1", "ge2", "ge3"]], on=["hadm_id", "t_hr"], how="left")

subjects = np.array(sorted(jf.subject.unique()))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = jf.subject.isin(train_subj).values
te = ~tr

PARAMS_SEL = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
                  class_weight="balanced", random_state=SEED, verbose=-1, importance_type="gain")
Xtr = jf[CAND].values[tr]; ytr = jf.ge1.values[tr]; groups = jf.subject.values[tr]
acc = np.zeros(len(CAND))
for a, b in GroupKFold(5).split(Xtr, ytr, groups):
    m = lgb.LGBMClassifier(**PARAMS_SEL); m.fit(Xtr[a], ytr[a])
    imp = m.feature_importances_.astype(float); acc += imp / imp.sum()
imp = acc / 5
order = np.argsort(imp)[::-1]
cum = np.cumsum(imp[order]) / imp.sum()
SEL = [CAND[i] for i in order[:int(np.searchsorted(cum, 0.99) + 1)]]
print(f"rebuilt: pool {len(CAND)} -> selected {len(SEL)}")

PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight="balanced", random_state=SEED, verbose=-1)
models = {}
for t in [1, 2, 3]:
    models[t] = lgb.LGBMClassifier(**PARAMS)
    models[t].fit(jf[SEL].values[tr], jf[f"ge{t}"].values[tr])

# Tier-1 OOF recal layer
oof = np.zeros(len(ytr))
for a, b in GroupKFold(5).split(Xtr, ytr, groups):
    m = lgb.LGBMClassifier(**PARAMS); m.fit(jf[SEL].values[tr][a], ytr[a])
    oof[b] = m.predict_proba(jf[SEL].values[tr][b])[:, 1]
def logit(p): p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))
cal = LogisticRegression(C=1e6).fit(logit(oof).reshape(-1, 1), ytr)
A1, B1 = float(cal.intercept_[0]), float(cal.coef_[0][0])

# ============ 2. internal test: AUROC + calibration curve + DCA ============
def stay_boot(df, n=500, seed=SEED):
    r = np.random.default_rng(seed)
    uniq = df.s.unique(); idx = {s_: np.where(df.s.values == s_)[0] for s_ in uniq}
    yv, pv = df.y.values, df.p.values; aucs = []
    for _ in range(n):
        sel = r.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[s_] for s_ in sel])
        if yv[ii].sum() > 0 and (1 - yv[ii]).sum() > 0:
            aucs.append(roc_auc_score(yv[ii], pv[ii]))
    return np.percentile(aucs, [2.5, 97.5]).round(4).tolist()

out = {"selected_n": len(SEL), "selected_features_top20": SEL[:20]}

pte1 = models[1].predict_proba(jf[SEL].values[te])[:, 1]
yte1 = jf.ge1.values[te]
dfi = pd.DataFrame({"s": jf.hadm_id.values[te], "y": yte1, "p": pte1})
out["internal_ge1"] = {"auroc": round(float(roc_auc_score(yte1, pte1)), 4),
                       "ci": stay_boot(dfi), "auprc": round(float(average_precision_score(yte1, pte1)), 4)}
p_rec_i = sigmoid(A1 + B1 * logit(pte1))

def cal_curve(y, p, bins=10):
    q = pd.qcut(pd.Series(p), bins, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return [round(float(v), 4) for v in g.p.mean()], [round(float(v), 4) for v in g.y.mean()]

def dca(y, p, thrs):
    n = len(y); prev = y.mean(); rows = []
    for t in thrs:
        al = p >= t
        tp = (al & (y == 1)).sum() / n; fp = (al & (y == 0)).sum() / n
        rows.append({"thr": round(t, 3), "nb_model": round(tp - fp * (t / (1 - t)), 5),
                     "nb_treat_all": round(prev - (1 - prev) * (t / (1 - t)), 5)})
    return rows

out["internal_cal_curve"] = cal_curve(yte1, p_rec_i)
out["internal_dca"] = dca(yte1, p_rec_i, np.arange(0.05, 0.505, 0.05))

# ============ 3. SHAP (Tier-1) ============
import shap
ridx = np.random.default_rng(SEED).choice(np.where(te)[0], size=min(2000, te.sum()), replace=False)
Xs = jf.iloc[ridx][SEL]
sv = shap.TreeExplainer(models[1]).shap_values(Xs.values, check_additivity=False)
if isinstance(sv, list): sv = sv[1]
imp_shap = np.abs(sv).mean(0)
o2 = np.argsort(imp_shap)[::-1]
out["shap_top15"] = [{"feature": SEL[i], "mean_abs": round(float(imp_shap[i]), 4)} for i in o2[:15]]

# ============ 4. EXTERNAL: M4 + eICU ============
def label_machinery(cr_df, feat_df, key):
    cr_df = cr_df.sort_values([key, "t_hr"])
    mgx = feat_df[[key, "t_hr", "cr_base"]].drop_duplicates().merge(
        cr_df.groupby([key, "t_hr"]).cr_val.max().reset_index(), on=[key, "t_hr"], how="left").sort_values([key, "t_hr"])
    mgx["f"] = np.nan
    for s in range(0, 9):
        sh = mgx.groupby(key).cr_val.shift(-s)
        mgx["f"] = pd.DataFrame({"c": mgx["f"], "n": sh}).max(axis=1)
    fo = mgx["f"] / mgx.cr_base.clip(lower=0.1); ri = mgx["f"] - mgx.cr_base
    ak = (fo >= 1.5) | (ri >= 0.3)
    stx = pd.Series(0, index=mgx.index)
    stx[ak.fillna(False)] = 1; stx[fo >= 2.0] = 2
    stx[(fo >= 3.0) | ((mgx["f"] >= 4.0) & ak)] = 3
    stx[mgx["f"].isna()] = 0
    for t in [1, 2, 3]:
        mgx[f"ge{t}"] = (stx >= t).astype(int)
    return mgx[[key, "t_hr", "ge1", "ge2", "ge3"]]

m4 = pd.read_parquet(R / "n5_abi_rolling_features.parquet")
mlab = pd.read_parquet(R / "n2_m4_abi_cr_serial.parquet").rename(columns={"valuenum": "cr_val"})
mlab = mlab[mlab.cr_val.between(0.1, 30)].copy(); mlab["t_hr"] = (mlab.hr // 6) * 6
m4f = m4[["stay_id", "t_hr"] + [c for c in SEL if c in m4.columns]].merge(
    label_machinery(mlab[["stay_id", "t_hr", "cr_val"]], m4, "stay_id"), on=["stay_id", "t_hr"], how="left")

eicu = pd.read_parquet(R / "p1_expanded_features.parquet")
elab = pd.read_parquet(R / "n8_2_labs_series.parquet")
ecr = elab[(elab.labname == "creatinine") & (elab.labtypeid == 1)].rename(columns={"sid": "stay_id", "labresult": "cr_val"})
ecr = ecr[ecr.cr_val.between(0.1, 30)].copy(); ecr["t_hr"] = (ecr.hr // 6) * 6
eif = eicu[["stay_id", "t_hr"] + [c for c in SEL if c in eicu.columns]].merge(
    label_machinery(ecr[["stay_id", "t_hr", "cr_val"]], eicu, "stay_id"), on=["stay_id", "t_hr"], how="left")

miss_m4 = [c for c in SEL if c not in m4.columns]
miss_ei = [c for c in SEL if c not in eicu.columns]
print("features missing in M4:", miss_m4[:8], f"({len(miss_m4)})")
print("features missing in eICU:", miss_ei[:8], f"({len(miss_ei)})")
out["missing_in_m4"] = miss_m4; out["missing_in_eicu"] = miss_ei

for nm, df in [("M4_ext", m4f), ("eICU_ext", eif)]:
    X = df[SEL].copy()
    for c in miss_m4 if nm == "M4_ext" else miss_ei:
        X[c] = np.nan
    res_e = {}
    for t in [1, 2, 3]:
        p = models[t].predict_proba(X.values)[:, 1]
        y = df[f"ge{t}"].values
        dfe = pd.DataFrame({"s": df.stay_id.values, "y": y, "p": p})
        res_e[f"ge{t}"] = {"auroc": round(float(roc_auc_score(y, p)), 4),
                           "ci": stay_boot(dfe),
                           "auprc": round(float(average_precision_score(y, p)), 4),
                           "pos_ckpt": int(y.sum()),
                           "event_stays": int(df.loc[y == 1, "stay_id"].nunique())}
        print(nm, "ge%d:" % t, res_e["ge%d" % t]["auroc"], res_e["ge%d" % t]["ci"], "ev_stays=", res_e["ge%d" % t]["event_stays"], flush=True)
    # ge1 external calibration with frozen Jinhua layer
    p1 = models[1].predict_proba(X.values)[:, 1]
    p1r = sigmoid(A1 + B1 * logit(p1))
    y1 = df.ge1.values
    lr_ = LogisticRegression(C=1e6).fit(logit(p1r).reshape(-1, 1), y1)
    res_e["calib_ge1_frozen_layer"] = {"slope": round(float(lr_.coef_[0][0]), 3)}
    res_e["cal_curve_ge1"] = cal_curve(y1, p1r)
    res_e["dca_ge1"] = dca(y1, p1r, np.arange(0.05, 0.505, 0.05))
    out[nm] = res_e

# ============ 5. figures ============
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
# (a) calibration curves
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
for ax, (nm, lo, hi) in zip(axes, [("Jinhua internal", *out["internal_cal_curve"]),
                                    ("M4 external", *out["M4_ext"]["cal_curve_ge1"]),
                                    ("eICU external", *out["eICU_ext"]["cal_curve_ge1"])]):
    ax.plot([0, 1], [0, 1], "--", color="#b8860b", lw=1)
    ax.plot(lo, hi, "o-", color="#0f6e8c", ms=4, lw=1.2)
    ax.set_xlabel("predicted"); ax.set_ylabel("observed"); ax.set_title(nm, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
fig.suptitle("Tier-1 calibration (recalibrated probabilities, frozen layer)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "f11_calibration.png", dpi=200); plt.close(fig)
# (b) DCA
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, nm, d in zip(axes, ["Jinhua internal", "eICU external"],
                     [out["internal_dca"], out["eICU_ext"]["dca_ge1"]]):
    th = [r["thr"] for r in d]
    ax.plot(th, [r["nb_model"] for r in d], "-", color="#0f6e8c", lw=1.3, label="model")
    ax.plot(th, [r["nb_treat_all"] for r in d], "--", color="#7a7a7a", lw=1, label="treat all")
    ax.axhline(0, color="#bdbdbd", lw=0.8)
    ax.set_xlabel("threshold"); ax.set_ylabel("net benefit"); ax.set_title(nm, fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "f11_dca.png", dpi=200); plt.close(fig)
# (c) SHAP beeswarm top 12
top12 = o2[:12]
shap.summary_plot(sv[:, top12], features=Xs.values[:, top12], feature_names=[SEL[i] for i in top12],
                  show=False, plot_size=(7, 4.2))
plt.tight_layout(); plt.savefig(FIG / "f11_shap_beeswarm.png", dpi=200, bbox_inches="tight"); plt.close("all")

out["_meta"] = {"date": "2026-10-02", "protocol": "f11 full eval of the Jinhua-dev model (f10); external = frozen application to M4/eICU full grids"}
json.dump(out, open(J / "f11_full_eval.json", "w"), indent=2)
print("\nSaved f11_full_eval.json + 3 figures in", FIG)
