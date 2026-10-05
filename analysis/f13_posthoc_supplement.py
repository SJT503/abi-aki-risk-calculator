# -*- coding: utf-8 -*-
"""f13: post hoc supplement for the external-validation study (Jinhua development, MIMIC-IV + eICU-CRD external).
Uses ONLY the frozen checkpoint-level predictions saved by f12 (f12_preds_{int,M4,eICU}_ge{1,2,3}.parquet) and the f12 JSON.
No model is refitted. Two-parameter logistic recalibration layers are fitted on frozen scores only (see section 4).

Sections
  0. Reproduction checks of f12 numbers from the parquet files (AUROC, event counts, calibration, deployment)
  1. Stay-cluster bootstrap CI of calibration slope / ECE / Brier (Tier 1, raw and recalibrated)
  2. Proximity standardisation (four-stratum checkpoint-proximity method of the companion analysis, same logic) for M4 and eICU
  3. Equivalence test (margin +/-0.05) for the >=Stage 2 minus >=Stage 1 AUROC difference (TOST / CI inclusion)
  4. Tier-2 recalibration views on frozen scores (cross-cohort frozen layers; cohort-internal cross-fitted layers) + threshold views
  5. Utility curve (eICU and M4, thresholds 0.03-0.30; deployment definition v2 identical to f12)
  6. False-alarm review that needs only predictions (lead > 48 h share)
  7. Grid-level descriptors from the prediction files + optional stay-membership check against a companion file
  8. QC of the supplied hospital / SHAP / exemplar files against the f12 JSON
Usage: python f13_posthoc_supplement.py DATA_DIR OUT_DIR [P1_INT_PREDS_PARQUET]
  DATA_DIR must contain f12_full_suite.json and the 13 supplied files; OUT_DIR receives f13_posthoc_results.json and CSVs.
"""
import hashlib, itertools, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SEED = 42
B_CAL = int(os.environ.get("F13_B_CAL", 1000))    # calibration bootstrap
B_PROX = int(os.environ.get("F13_B_PROX", 1000))  # proximity bootstrap (companion analysis: 1000)
B_EQ = int(os.environ.get("F13_B_EQ", 2000))      # equivalence bootstrap
MARGIN = 0.05
COH = ("int", "M4", "eICU")


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ece10(y, p):                       # identical to f12
    q = pd.qcut(pd.Series(p), 10, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return float((g.size() / len(y) * np.abs(g.y.mean() - g.p.mean())).sum())


def slope_of(y, p):                    # identical to f12 cal_mets
    return float(LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y).coef_[0][0])


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def load_preds(D):
    P = {}
    for nm in COH:
        for g in (1, 2, 3):
            d = pd.read_parquet(os.path.join(D, f"f12_preds_{nm}_ge{g}.parquet"))
            d["stay_id"] = d.stay_id.astype(str)
            P[(nm, g)] = d.rename(columns={"stay_id": "s"}).reset_index(drop=True)
    for nm in COH:                                                      # the three endpoints share rows
        for g in (2, 3):
            assert (P[(nm, 1)].s.values == P[(nm, g)].s.values).all() and (P[(nm, 1)].t_hr.values == P[(nm, g)].t_hr.values).all(), (nm, g)
    return P


def stay_groups(s):
    u, inv = np.unique(s, return_inverse=True)
    idx_by = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    return u, idx_by


def deploy_v2(df, thr, pcol):
    """Deployment definition v2 -- same logic as f12 deploy_v2, with the score column made explicit."""
    d = df.copy()
    d["alert"] = d[pcol] >= thr
    lastpos = d[d.y == 1].groupby("s").t_hr.max()
    firstpos = d[d.y == 1].groupby("s").t_hr.min()
    capt, leads = [], []
    for sid, g in d[d.s.isin(lastpos.index)].groupby("s"):
        al = g[g["alert"] & (g.t_hr < lastpos[sid])]
        if len(al):
            capt.append(sid)
            leads.append(lastpos[sid] - al.t_hr.min())
    alerts = set(d.loc[d["alert"], "s"])
    nc = len(capt)
    nev = d[~d.s.isin(lastpos.index)].groupby("s").t_hr.max().clip(lower=6).div(24).sum()
    evf = firstpos.clip(lower=6).div(24).sum()
    return {"thr": float(thr), "capture": round(nc / len(lastpos), 3),
            "lead_median_h": round(float(np.median(leads)), 1) if leads else None,
            "lead_iqr": [round(float(np.percentile(leads, 25)), 1), round(float(np.percentile(leads, 75)), 1)] if leads else None,
            "NNE": round(len(alerts) / nc, 2) if nc else None,
            "FA_per_100ptd": round(100 * len(alerts - set(lastpos.index)) / (nev + evf), 2),
            "pct_stays_alerted": round(100 * len(alerts) / d.s.nunique(), 1),
            "n_event_stays": int(len(lastpos)), "n_captured": int(nc), "n_alert_stays": int(len(alerts)),
            "n_false_alert_stays": int(len(alerts - set(lastpos.index))),
            "lead_gt48h_n": int(np.sum(np.array(leads) > 48)) if leads else 0,
            "lead_gt48h_frac": round(float(np.mean(np.array(leads) > 48)), 3) if leads else None}


# ------------------------------------------------------------------ 1. calibration bootstrap
def calib_boot(nm, d, A1, B1, B=B_CAL):
    y, p = d.y.values, d.p.values
    prec = sigmoid(A1 + B1 * logit(p))
    keys = ["raw_slope", "recal_slope", "raw_ece", "recal_ece", "brier_raw", "brier_recal"]

    def stats(yy, pp, pr):
        return {"raw_slope": slope_of(yy, pp), "recal_slope": slope_of(yy, pr), "raw_ece": ece10(yy, pp), "recal_ece": ece10(yy, pr),
                "brier_raw": brier(yy, pp), "brier_recal": brier(yy, pr)}
    pt = stats(y, p, prec)
    prev = float(y.mean()); b0 = prev * (1 - prev)
    pt.update({"brier_null": b0, "bss_raw": 1 - pt["brier_raw"] / b0, "bss_recal": 1 - pt["brier_recal"] / b0, "prevalence": prev})
    u, idx_by = stay_groups(d.s.values)
    rng = np.random.default_rng(SEED)
    reps = {k: [] for k in keys + ["bss_raw", "bss_recal"]}
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        ii = np.concatenate([idx_by[j] for j in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        st = stats(yy, p[ii], prec[ii])
        pv = yy.mean(); bb = pv * (1 - pv)
        st["bss_raw"] = 1 - st["brier_raw"] / bb; st["bss_recal"] = 1 - st["brier_recal"] / bb
        for k in reps:
            reps[k].append(st[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in reps.items()}
    return nm, {"point": pt, "ci": ci, "B": B, "n_valid": len(reps["raw_slope"]), "n_stays": int(len(u)), "n_event_stays": int(d[d.y == 1].s.nunique()),
                "n_pos_ckpt": int(y.sum()), "n_ckpt": int(len(y))}


# ------------------------------------------------------------------ 2. proximity standardisation (companion-analysis logic)
BINS = [(0, 0), (6, 6), (12, 18), (24, 10 ** 6)]
BIN_LABELS = ["0", "6", "12-18", ">=24"]


def hload(d):
    d = d[["s", "t_hr", "y", "p"]].copy()
    d["dend"] = d.groupby("s").t_hr.transform("max") - d.t_hr
    d["hb"] = -1
    for k, (lo, hi) in enumerate(BINS):
        d.loc[(d.dend >= lo) & (d.dend <= hi), "hb"] = k
    assert (d.hb >= 0).all()
    return d.reset_index(drop=True)          # row order = original (rows of the 3 endpoints stay aligned)


def hweights(d, target_share):
    w = np.ones(len(d))
    pos = d.y.values == 1
    share = pd.Series(d.hb.values[pos]).value_counts(normalize=True)
    for k in range(len(BINS)):
        m = pos & (d.hb.values == k)
        if m.any() and share.get(k, 0) > 0:
            w[m] = target_share.get(k, 0) / share[k]
    return w


def proximity(nm, P, B=B_PROX):
    D = {g: hload(P[(nm, g)]) for g in (1, 2, 3)}
    assert (D[1][["s", "t_hr"]].values == D[3][["s", "t_hr"]].values).all()
    tgt = pd.Series(D[3].hb.values[D[3].y.values == 1]).value_counts(normalize=True).to_dict()
    r = {"bins_h": BIN_LABELS, "B": B, "seed": SEED, "ge3_pos_share_by_bin": {BIN_LABELS[k]: float(tgt.get(k, 0)) for k in range(4)}}
    for g in (1, 2, 3):
        pos = D[g].y.values == 1; neg = ~pos
        r[f"ge{g}_auc_by_bin"] = {}
        for k in range(4):
            m = pos & (D[g].hb.values == k)
            sel = m | neg
            r[f"ge{g}_auc_by_bin"][BIN_LABELS[k]] = {"n_pos": int(m.sum()),
                "auroc": (float(roc_auc_score(D[g].y.values[sel], D[g].p.values[sel])) if m.sum() >= 10 else None)}
    W = {g: hweights(D[g], tgt) for g in (1, 2)}
    pt = {"ge3": roc_auc_score(D[3].y, D[3].p)}
    for g in (1, 2):
        pt[f"ge{g}_std"] = roc_auc_score(D[g].y, D[g].p, sample_weight=W[g])
        pt[f"ge{g}_raw"] = roc_auc_score(D[g].y, D[g].p)
    r["point"] = {k: float(v) for k, v in pt.items()}
    u, idx_by = stay_groups(D[1].s.values)
    rng = np.random.default_rng(SEED)
    diffs = {"ge3_minus_ge1std": [], "ge3_minus_ge2std": [], "ge1std": [], "ge2std": []}
    for _b in range(B):
        pick = rng.integers(0, len(u), len(u))
        ii = np.concatenate([idx_by[j] for j in pick])
        y3, p3 = D[3].y.values[ii], D[3].p.values[ii]
        if y3.sum() == 0 or y3.sum() == len(y3):
            continue
        a3b = roc_auc_score(y3, p3)
        t3 = pd.Series(D[3].hb.values[ii][y3 == 1]).value_counts(normalize=True).to_dict()
        for g in (1, 2):
            yg = D[g].y.values[ii]
            if yg.sum() == 0:
                break
            sub_ = D[g].iloc[ii].reset_index(drop=True)
            ag = roc_auc_score(yg, D[g].p.values[ii], sample_weight=hweights(sub_, t3))
            diffs[f"ge{g}std"].append(ag)
            diffs[f"ge3_minus_ge{g}std"].append(a3b - ag)
    for k, v in diffs.items():
        v = np.array(v)
        if k.startswith("ge3_minus"):
            p = 2 * min((v <= 0).mean(), (v >= 0).mean())
            key = k.split("_minus_")[1].replace("std", "_std")
            r[k] = {"diff": float(pt["ge3"] - pt[key]), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                    "p": float(p), "n_valid": int(len(v))}
        else:
            r[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return nm, r


# ------------------------------------------------------------------ 3. equivalence (ge2 - ge1)
def eq_reps(nm, P, B=B_EQ):
    d1, d2 = P[(nm, 1)], P[(nm, 2)]
    obs = float(roc_auc_score(d2.y, d2.p) - roc_auc_score(d1.y, d1.p))
    u, idx_by = stay_groups(d1.s.values)
    rng = np.random.default_rng(SEED)
    reps = np.full(B, np.nan)
    for b in range(B):
        pick = rng.integers(0, len(u), len(u))
        ii = np.concatenate([idx_by[j] for j in pick])
        y1, y2 = d1.y.values[ii], d2.y.values[ii]
        if y1.sum() and y2.sum():
            reps[b] = roc_auc_score(y2, d2.p.values[ii]) - roc_auc_score(y1, d1.p.values[ii])
    return nm, obs, reps


def equiv_summary(obs, reps, B):
    ok = ~np.isnan(reps)
    r_ = reps[ok]
    lo90, hi90 = np.percentile(r_, [5, 95]); lo95, hi95 = np.percentile(r_, [2.5, 97.5])
    p_low = float(np.mean(r_ <= -MARGIN)); p_up = float(np.mean(r_ >= MARGIN))
    return {"delta_obs": float(obs), "se": float(np.std(r_, ddof=1)), "n_valid": int(ok.sum()),
            "ci90": [float(lo90), float(hi90)], "ci95": [float(lo95), float(hi95)],
            "p_tost": max(p_low, p_up), "p_lower": p_low, "p_upper": p_up,
            "equivalent_ci90_within_margin": bool(lo90 > -MARGIN and hi90 < MARGIN),
            "min_symmetric_margin_ci90": float(max(abs(lo90), abs(hi90)))}


def pooled(cs, obs, reps, B):
    w = np.array([1 / np.var(reps[c][~np.isnan(reps[c])], ddof=1) for c in cs])
    wn = w / w.sum()
    dobs = np.array([obs[c] for c in cs])
    pooled_obs = float((wn * dobs).sum())
    ok = np.all([~np.isnan(reps[c]) for c in cs], axis=0)
    pr = sum(wn[i] * reps[c][ok] for i, c in enumerate(cs))
    out = equiv_summary(pooled_obs, pr, B)
    out["weights"] = {c: float(wn[i]) for i, c in enumerate(cs)}
    Q = float((w * (dobs - pooled_obs) ** 2).sum()); df = len(cs) - 1
    out["Q"] = Q; out["Q_df"] = df; out["Q_p"] = float(chi2.sf(Q, df)); out["I2"] = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    out["se_fixed_effect"] = float(np.sqrt(1 / w.sum()))
    return out


# ------------------------------------------------------------------ 4. Tier-2 recalibration views
def fit_layer(y, p):
    m = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def tier2_views(P):
    out = {"layers": {}, "eval": [], "local_crossfit": [], "thresholds": []}
    L = {}
    for src in ("M4", "eICU"):
        d = P[(src, 3)]
        a, b = fit_layer(d.y.values, d.p.values)
        L[src] = (a, b); out["layers"][src] = {"a": a, "b": b, "n_pos_ckpt": int(d.y.sum()), "n_event_stays": int(d[d.y == 1].s.nunique())}
    for tgt in COH:
        d = P[(tgt, 3)]; y, p = d.y.values, d.p.values
        base = {"slope": slope_of(y, p), "ece": ece10(y, p)}
        for src in ("M4", "eICU"):
            if src == tgt:
                continue
            pr = sigmoid(L[src][0] + L[src][1] * logit(p))
            assert abs(roc_auc_score(y, pr) - roc_auc_score(y, p)) < 1e-9          # monotone transform
            out["eval"].append({"target": tgt, "layer_from": src, "before": base, "after": {"slope": slope_of(y, pr), "ece": ece10(y, pr)},
                                "n_pos_ckpt": int(y.sum()), "n_event_stays": int(d[d.y == 1].s.nunique())})
    for tgt in ("M4", "eICU"):                                                  # cohort-internal cross-fitted layer (GroupKFold by stay)
        d = P[(tgt, 3)]; y, p = d.y.values, d.p.values
        z = logit(p); oof = np.zeros(len(y))
        for a_, b_ in GroupKFold(5).split(z.reshape(-1, 1), y, d.s.values):
            m = LogisticRegression(C=1e6).fit(z[a_].reshape(-1, 1), y[a_])
            oof[b_] = m.predict_proba(z[b_].reshape(-1, 1))[:, 1]
        out["local_crossfit"].append({"target": tgt, "folds": 5, "grouping": "stay_id",
                                      "before": {"slope": slope_of(y, p), "ece": ece10(y, p)},
                                      "after": {"slope": slope_of(y, oof), "ece": ece10(y, oof)}})
    for tgt in ("M4", "eICU"):
        src = "eICU" if tgt == "M4" else "M4"
        d = P[(tgt, 3)].copy()
        d["p_rec"] = sigmoid(L[src][0] + L[src][1] * logit(d.p.values))
        for thr in (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20):   # 0.05-0.20 = pre-specified Tier-1 grid; 0.01-0.03 added (Tier-2 event rate ~0.3%)
            v = deploy_v2(d, thr, "p_rec"); v.update({"target": tgt, "layer_from": src})
            out["thresholds"].append(v)
    return out


# ------------------------------------------------------------------ main
def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(D, OUT, P1_INT=None):
    os.makedirs(OUT, exist_ok=True)
    F12 = json.load(open(os.path.join(D, "f12_full_suite.json"), encoding="utf-8"))
    A1, B1 = F12["calibration"]["a"], F12["calibration"]["b"]
    P = load_preds(D)
    R = {"_meta": {"script": "f13_posthoc_supplement.py", "seed": SEED, "B_calibration": B_CAL, "B_proximity": B_PROX, "B_equivalence": B_EQ,
                   "equivalence_margin": MARGIN, "tier1_layer": {"a": A1, "b": B1},
                   "note": "frozen f12 predictions only; no model refit"}}
    # ---- 0. reproduction
    chk = []
    for nm in COH:
        for g in (1, 2, 3):
            d = P[(nm, g)]; J = F12["discrimination"][f"{nm}_ge{g}"]
            ok = (round(roc_auc_score(d.y, d.p), 4) == J["auroc"] and int(d.y.sum()) == J["pos_ckpt"] and int(d[d.y == 1].s.nunique()) == J["event_stays"])
            chk.append({"check": f"auroc_counts_{nm}_ge{g}", "ok": bool(ok)})
            assert ok, (nm, g)
    for nm in COH:
        d = P[(nm, 1)]; J = F12["calibration"][nm]
        rec = sigmoid(A1 + B1 * logit(d.p.values))
        ok = (round(slope_of(d.y.values, d.p.values), 3) == J["raw"]["slope"] and round(ece10(d.y.values, d.p.values), 4) == J["raw"]["ece"]
              and round(slope_of(d.y.values, rec), 3) == J["recal"]["slope"] and round(ece10(d.y.values, rec), 4) == J["recal"]["ece"])
        chk.append({"check": f"calibration_{nm}", "ok": bool(ok)}); assert ok, nm
        dd = d.copy(); dd["p_rec"] = rec
        for t in (0.05, 0.08, 0.10, 0.15, 0.20):
            v = deploy_v2(dd, t, "p_rec"); J2 = F12["deployment"][nm][f"thr{t:g}"]
            ok = all(v[k] == J2[k] for k in ("capture", "lead_median_h", "lead_iqr", "NNE", "FA_per_100ptd", "pct_stays_alerted", "n_event_stays", "n_captured", "n_alert_stays"))
            chk.append({"check": f"deployment_{nm}_thr{t:g}", "ok": bool(ok)}); assert ok, (nm, t, v, J2)
    R["repro"] = {"n_checks": len(chk), "all_ok": all(c["ok"] for c in chk), "checks": chk}
    print("0. reproduction:", len(chk), "checks, all OK", flush=True)

    # ---- 1/2/3 in parallel
    jobs = [delayed(calib_boot)(nm, P[(nm, 1)], A1, B1) for nm in COH]
    jobs += [delayed(proximity)(nm, P) for nm in ("M4", "eICU")]
    jobs += [delayed(eq_reps)(nm, P) for nm in COH]
    res = Parallel(n_jobs=7)(jobs)
    R["calibration_bootstrap"] = {nm: v for nm, v in res[:3]}
    R["proximity"] = {nm: v for nm, v in res[3:5]}
    obs = {nm: o for nm, o, _ in res[5:]}; reps = {nm: rp for nm, _, rp in res[5:]}
    print("1-3 bootstraps done", flush=True)
    eq = {"margin": MARGIN, "B": B_EQ, "seed": SEED, "per_cohort": {nm: equiv_summary(obs[nm], reps[nm], B_EQ) for nm in COH}}
    eq["pooled_external"] = pooled(["M4", "eICU"], obs, reps, B_EQ)
    eq["pooled_all_three"] = pooled(["int", "M4", "eICU"], obs, reps, B_EQ)
    R["equivalence"] = eq

    # ---- 4. Tier-2 views
    R["tier2_recal"] = tier2_views(P)

    # ---- 5. utility curves (thresholds exactly as f12: np.arange(0.03, 0.301, 0.01))
    for nm in ("eICU", "M4"):
        d = P[(nm, 1)].copy(); d["p_rec"] = sigmoid(A1 + B1 * logit(d.p.values))
        rows = [deploy_v2(d, t, "p_rec") for t in np.arange(0.03, 0.301, 0.01)]
        for r_ in rows:
            r_["thr"] = round(r_["thr"], 2)
        R[f"utility_{nm}"] = rows
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"f13_utility_curve_{nm}.csv"), index=False)
    # utility curve must agree with the f12 deployment block at the five stored thresholds
    for nm in ("eICU", "M4"):
        for r_ in R[f"utility_{nm}"]:
            key = f"thr{r_['thr']:g}"
            if key in F12["deployment"][nm]:
                J2 = F12["deployment"][nm][key]
                assert all(r_[k] == J2[k] for k in ("capture", "lead_median_h", "NNE", "FA_per_100ptd", "pct_stays_alerted", "n_captured", "n_alert_stays")), (nm, key)

    # ---- 6. FA review that needs predictions only (thr 0.15)
    R["fa_review"] = {}
    for nm in ("eICU", "M4"):
        u15 = [r_ for r_ in R[f"utility_{nm}"] if abs(r_["thr"] - 0.15) < 1e-9][0]
        R["fa_review"][nm] = {k: u15[k] for k in ("thr", "n_event_stays", "n_captured", "n_alert_stays", "n_false_alert_stays", "lead_gt48h_n", "lead_gt48h_frac")}
        R["fa_review"][nm]["n_event_free_stays"] = int(P[(nm, 1)].s.nunique() - u15["n_event_stays"])
        R["fa_review"][nm]["near_criteria_creatinine_share"] = None            # needs the eICU creatinine series: not in the supplied files

    # ---- 7. grid-level descriptors from predictions
    desc = {}
    for nm in COH:
        d = P[(nm, 1)]
        per = d.groupby("s").size(); last = d.groupby("s").t_hr.max(); first = d.groupby("s").t_hr.min()
        desc[nm] = {"stays": int(d.s.nunique()), "ckpt": int(len(d)),
                    "ckpt_per_stay_median": float(per.median()), "ckpt_per_stay_q1": float(per.quantile(.25)), "ckpt_per_stay_q3": float(per.quantile(.75)),
                    "first_ckpt_hr_min": float(first.min()), "first_ckpt_hr_max": float(first.max()),
                    "last_ckpt_hr_median": float(last.median()), "last_ckpt_hr_q1": float(last.quantile(.25)), "last_ckpt_hr_q3": float(last.quantile(.75)),
                    "event_stays": {f"ge{g}": int(P[(nm, g)][P[(nm, g)].y == 1].s.nunique()) for g in (1, 2, 3)},
                    "event_stay_pct_ge1": float(100 * P[(nm, 1)][P[(nm, 1)].y == 1].s.nunique() / d.s.nunique())}
    R["descriptors"] = desc
    if P1_INT and os.path.exists(P1_INT):
        t = pd.read_parquet(P1_INT)
        t_ids = set(t.stay_id.astype(str).unique()); m4_ids = set(P[("M4", 1)].s.unique())
        R["m4_membership_check"] = {"companion_test_stays": len(t_ids), "in_M4_prediction_file": len(t_ids & m4_ids), "M4_stays": len(m4_ids),
                                    "all_contained": bool(t_ids <= m4_ids), "complement_stays": len(m4_ids - t_ids)}
        strata = {}
        for lab, sel in (("test_subset", t_ids), ("complement_subset", m4_ids - t_ids)):
            e = {"stays": len(sel), "ckpt": int(P[("M4", 1)].s.isin(sel).sum())}
            for g in (1, 2, 3):
                dg = P[("M4", g)]; dg = dg[dg.s.isin(sel)]
                e[f"event_stays_ge{g}"] = int(dg[dg.y == 1].s.nunique()); e[f"pos_ckpt_ge{g}"] = int(dg.y.sum())
            strata[lab] = e
        R["m4_membership_check"]["strata_from_predictions"] = strata

    # ---- 8. QC of supplied hospital / SHAP / exemplar files
    qc = []
    h = pd.read_csv(os.path.join(D, "f12_hospital_aucs.csv"))
    HJ = F12["hospital"]
    hq = {"n": int(len(h)), "median": round(float(h.auroc.median()), 3), "min": round(float(h.auroc.min()), 3), "max": round(float(h.auroc.max()), 3),
          "n_below_060": int((h.auroc < 0.60).sum()), "n_ge_070": int((h.auroc >= 0.70).sum()), "n_ge_080": int((h.auroc >= 0.80).sum())}
    hq["matches_f12_json"] = bool(hq["n"] == HJ["n_ge20"] and hq["median"] == HJ["median"] and hq["min"] == HJ["min"] and hq["max"] == HJ["max"] and hq["n_below_060"] == HJ["n_below_060"])
    R["hospital_file"] = hq
    sv = pd.read_parquet(os.path.join(D, "f12_shap_values.parquet")); sx = pd.read_parquet(os.path.join(D, "f12_shap_X.parquet"))
    feats = [c for c in sv.columns if c != "stay_id"]
    ma = sv[feats].abs().mean().sort_values(ascending=False)
    top = [{"feature": f, "mean_abs": round(float(v), 4)} for f, v in ma.head(15).items()]
    R["shap_file"] = {"rows": int(len(sv)), "n_features": len(feats), "X_rows": int(len(sx)), "same_stay_order": bool((sv.stay_id.values == sx.stay_id.values).all()),
                      "top15": top, "matches_f12_json_top15": bool(top == F12["shap_top15"])}
    ex = pd.read_csv(os.path.join(D, "f12_exemplars.csv"))
    R["exemplar_file"] = {"rows": int(len(ex)), "stays": int(ex.stay_id.nunique())}
    R["input_sha256"] = {f: sha256(os.path.join(D, f)) for f in sorted(os.listdir(D)) if f.startswith("f12_") and f != "f12_full_suite.json"}
    json.dump(R, open(os.path.join(OUT, "f13_posthoc_results.json"), "w"), indent=1)
    print("saved f13_posthoc_results.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
