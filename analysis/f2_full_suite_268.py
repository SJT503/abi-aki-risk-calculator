# -*- coding: utf-8 -*-
"""f2_full_suite_268: Complete downstream suite on the reselected 268-feature set.
Standard-operation definitions, all documented in the output JSON:
- Calibration: OOF layer via GroupKFold(5) by subject (stricter than the legacy plain
  5-fold; no subject crosses OOF folds).
- Deployment: anchor = the stay's FINAL POSITIVE checkpoint (the definition reverse-
  engineered by the round-8 audit to reproduce the authoritative capture/lead/NNE).
  capture: alert (recalibrated p >= thr) at t_hr < final positive checkpoint.
  lead: final positive checkpoint - first qualifying alert.
  NNE: alerted stays / captured stays.
  FA denominator: event-free patient-days = full duration of never-event stays +
  pre-first-positive-checkpoint duration of event stays (per stay floor 6 h).
- DeLong vs LR (age/charlson/cr_base), DCA 0.05-0.50, hospital AUROC (>=20 events),
  MK trend (right-closed 24-h strata), subgroups with stay-bootstrap CI.
- SHAP ge1 (8000-row sample).
- Proximity (horizon) standardisation: verbatim round-8 horizon.py logic.
- SM5 arms on 268: (1) fill0 missing-value arm x3 endpoints; (2) threshold-specific
  models trained on ALL development data (train+internal test) x3 endpoints;
  (3) false-alert autopsy at external thr 0.15.
- EPPP with 268 parameters.
Outputs: results/f2_full_results.json (+ p_rec columns added to f1_preds parquets,
+ f2_shap_values/X parquets, f2_dca_{int,ext}.csv, f2_horizon.json)
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

R = "results"
SEED = 42
FEATS = json.load(open(f"{R}/_reselected_features.json"))
assert len(FEATS) == 268
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=SEED, verbose=-1)

m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[['stay_id'] + FEATS]
eicu = pd.read_parquet(f"{R}/p1_expanded_features.parquet")[['stay_id'] + FEATS]

def labels_e02(labs_parquet, feat_df, db):
    labs = pd.read_parquet(labs_parquet)
    if 'sid' in labs.columns:
        cr = labs[labs['labname'] == 'creatinine'].rename(columns={'sid': 'stay_id', 'labresult': 'cr_val'})
    else:
        cr = labs.rename(columns={'valuenum': 'cr_val'})
    cr = cr[cr['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
    cr['t_hr'] = (cr['hr'] // 6) * 6
    cb = cr.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index().sort_values(['stay_id', 't_hr'])
    mg = feat_df[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
    mg['f'] = np.nan
    for s in range(0, 9):
        sh = mg.groupby('stay_id')['cr_val'].shift(-s)
        mg['f'] = pd.DataFrame({'c': mg['f'], 'n': sh}).max(axis=1)
    fold = mg['f'] / mg['cr_base'].clip(lower=0.1)
    rise = mg['f'] - mg['cr_base']
    aki = (fold >= 1.5) | (rise >= 0.3)
    st = pd.Series(0, index=mg.index)
    st[aki.fillna(False)] = 1
    st[fold >= 2.0] = 2
    st[(fold >= 3.0) | ((mg['f'] >= 4.0) & aki)] = 3
    st[mg['f'].isna()] = 0
    for t in [1, 2, 3]:
        mg[f'ge{t}'] = (st >= t).astype(int)
    print(f"  {db}: ge1={int(mg['ge1'].sum())} ge2={int(mg['ge2'].sum())} ge3={int(mg['ge3'].sum())}", flush=True)
    # near-criterion creatinine for FA autopsy: max future fold / rise short of criteria
    nf = fold.copy()
    return mg[['stay_id', 't_hr', 'ge1', 'ge2', 'ge3']]

m4_lab = labels_e02(f"{R}/n2_m4_abi_cr_serial.parquet", m4, "M4")
ei_lab = labels_e02(f"{R}/n8_2_labs_series.parquet", eicu, "eICU")
m4f = m4.merge(m4_lab, on=['stay_id', 't_hr'], how='left')
eif = eicu.merge(ei_lab, on=['stay_id', 't_hr'], how='left')

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr
ytr = m4f['ge1'].values[tr]

# ============ 1. ge1 model + OOF calibration (GroupKFold by subject) ============
print("=== 1. ge1 model + OOF calibration ===", flush=True)
mod1 = lgb.LGBMClassifier(**PARAMS)
mod1.fit(m4f[FEATS].values[tr], ytr)
pte = mod1.predict_proba(m4f[FEATS].values[te])[:, 1]
pee = mod1.predict_proba(eif[FEATS].values)[:, 1]
yte, yee = m4f['ge1'].values[te], eif['ge1'].values

Xtr = m4f[FEATS].values[tr]
gtr = m4f['stay_id'].map(s2sub).values[tr]
oof = np.zeros(len(ytr))
gkf = GroupKFold(n_splits=5)
for k, (itr, ite_) in enumerate(gkf.split(Xtr, ytr, gtr), 1):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xtr[itr], ytr[itr])
    oof[ite_] = m.predict_proba(Xtr[ite_])[:, 1]
    print(f"  OOF fold {k}/5", flush=True)

def logit(p): p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))
cal = LogisticRegression(C=1e6).fit(logit(oof).reshape(-1, 1), ytr)
a, b = float(cal.intercept_[0]), float(cal.coef_[0][0])
print(f"  recal layer: a={a:.6f} b={b:.6f}", flush=True)

def ece10(y, p):
    q = pd.qcut(pd.Series(p), 10, duplicates='drop')
    g = pd.DataFrame({'y': y, 'p': p, 'q': q}).groupby('q', observed=True)
    return float((g.size() / len(y) * np.abs(g['y'].mean() - g['p'].mean())).sum())
def metrics(y, p):
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return {'slope': round(float(lr.coef_[0][0]), 3), 'ece': round(ece10(y, p), 4),
            'brier': round(float(np.mean((p - y) ** 2)), 4)}
calib = {}
for nm, yy, pp in [('int_test', yte, pte), ('ext', yee, pee)]:
    pr = sigmoid(a + b * logit(pp))
    calib[nm] = {'raw': metrics(yy, pp), 'recal': metrics(yy, pr),
                 'brier_null': round(float(np.mean((np.mean(yy) - yy) ** 2)), 4)}
    print(f"  {nm}: raw {calib[nm]['raw']} -> recal {calib[nm]['recal']}", flush=True)

for c, df in [('int', pd.read_parquet(f"{R}/f1_preds_int_ge1.parquet")),
              ('ext', pd.read_parquet(f"{R}/f1_preds_ext_ge1.parquet"))]:
    df['p_rec'] = sigmoid(a + b * logit(df['p']))
    df.to_parquet(f"{R}/f1_preds_{c}_ge1.parquet", index=False)
pi = pd.read_parquet(f"{R}/f1_preds_int_ge1.parquet")
pe = pd.read_parquet(f"{R}/f1_preds_ext_ge1.parquet")

# ============ 2. Deployment (documented v2 definitions) ============
print("\n=== 2. Deployment ===", flush=True)
def deploy_v2(df, thr):
    d = df.copy()
    d['alert'] = d['p_rec'] >= thr
    lastpos = d[d.y == 1].groupby('stay_id')['t_hr'].max()
    firstpos = d[d.y == 1].groupby('stay_id')['t_hr'].min()
    capt, leads = [], []
    for sid, g in d[d.stay_id.isin(lastpos.index)].groupby('stay_id'):
        al = g[g['alert'] & (g['t_hr'] < lastpos[sid])]
        if len(al):
            capt.append(sid); leads.append(lastpos[sid] - al['t_hr'].min())
    alert_stays = set(d.loc[d['alert'], 'stay_id'])
    n_capt = len(capt)
    fa_stays = alert_stays - set(lastpos.index)
    ev_st = d[d.stay_id.isin(lastpos.index)].groupby('stay_id')['t_hr'].max()
    nev_days = d[~d.stay_id.isin(lastpos.index)].groupby('stay_id')['t_hr'].max().clip(lower=6).div(24).sum()
    evfree_days = firstpos.clip(lower=6).div(24).sum()
    return {'thr': thr,
            'capture': round(n_capt / len(lastpos), 3),
            'lead_median_h': round(float(np.median(leads)), 1) if leads else None,
            'lead_iqr_h': [round(float(np.percentile(leads, 25)), 1), round(float(np.percentile(leads, 75)), 1)] if leads else None,
            'NNE': round(len(alert_stays) / n_capt, 2) if n_capt else None,
            'FA_per_100ptd': round(100 * len(fa_stays) / (nev_days + evfree_days), 1),
            'n_event_stays': int(len(lastpos)), 'n_alert_stays': int(len(alert_stays)),
            'n_captured': n_capt,
            'pct_stays_alerted': round(100 * len(alert_stays) / d['stay_id'].nunique(), 1)}
deploy_int = {f"thr{t}": deploy_v2(pi, t) for t in [0.05, 0.10, 0.15, 0.20]}
deploy_ext = {f"thr{t}": deploy_v2(pe, t) for t in [0.05, 0.10, 0.15, 0.20]}
for t in [0.05, 0.10, 0.15, 0.20]:
    m = deploy_ext[f'thr{t}']
    print(f"  ext thr{t}: cap={m['capture']} lead={m['lead_median_h']}h NNE={m['NNE']} FA={m['FA_per_100ptd']} alert%={m['pct_stays_alerted']}", flush=True)

# ============ 3. DeLong vs LR baseline ============
print("\n=== 3. DeLong ===", flush=True)
CAND = [c for c in ['age', 'charlson', 'cr_base'] if c in FEATS]
lr = LogisticRegression(max_iter=2000, class_weight='balanced')
lr.fit(m4f[CAND].fillna(0).values[tr], ytr)
lr_pi = lr.predict_proba(m4f[CAND].fillna(0).values[te])[:, 1]
lr_pe = lr.predict_proba(eif[CAND].fillna(0).values)[:, 1]
def midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N); T2[J] = T; return T2
def delong_paired(y, p1, p2):
    a1, a2 = roc_auc_score(y, p1), roc_auc_score(y, p2)
    pos = y == 1; n1, n0 = int(pos.sum()), int((~pos).sum())
    x1, x0 = np.ascontiguousarray(p1[pos]), np.ascontiguousarray(p1[~pos])
    tx = midrank(np.r_[x1, x0]); tz = np.r_[midrank(x1), midrank(x0)]
    v10a = (tx[:n1] - tz[:n1]) / n0; v01a = (tx[n1:] - tz[n1:]) / n1
    x1b, x0b = np.ascontiguousarray(p2[pos]), np.ascontiguousarray(p2[~pos])
    tx2 = midrank(np.r_[x1b, x0b]); tz2 = np.r_[midrank(x1b), midrank(x0b)]
    v10b = (tx2[:n1] - tz2[:n1]) / n0; v01b = (tx2[n1:] - tz2[n1:]) / n1
    c1 = np.cov(np.stack([v10a, v10b]))[0, 1] / n1; c0 = np.cov(np.stack([v01a, v01b]))[0, 1] / n0
    vd = c1 + c0; d = a1 - a2; z = d / np.sqrt(vd)
    from scipy.stats import norm
    return {'auc_model': round(a1, 4), 'auc_lr': round(a2, 4), 'diff': round(d, 4),
            'z': round(z, 2), 'p': f'{2 * (1 - norm.cdf(abs(z))):.2e}'}
delong = {'int': delong_paired(yte, pte, lr_pi), 'ext': delong_paired(yee, pee, lr_pe)}
print(f"  ext: {delong['ext']}", flush=True)

# ============ 4. DCA ============
print("\n=== 4. DCA ===", flush=True)
def dca_calc(y, p, thrs):
    n = len(y); prev = y.mean(); rows = []
    for t in thrs:
        al = p >= t; tp = (al & (y == 1)).sum() / n; fp = (al & (y == 0)).sum() / n
        rows.append({'thr': round(t, 3), 'nb_model': round(tp - fp * (t / (1 - t)), 5),
                     'nb_treat_all': round(prev - (1 - prev) * (t / (1 - t)), 5)})
    return rows
ths = np.arange(0.05, 0.505, 0.025)
dca_int = dca_calc(yte, pi['p_rec'].values, ths)
dca_ext = dca_calc(yee, pe['p_rec'].values, ths)
pd.DataFrame(dca_int).to_csv(f"{R}/f2_dca_int.csv", index=False)
pd.DataFrame(dca_ext).to_csv(f"{R}/f2_dca_ext.csv", index=False)
print(f"  ext model>treat_all all: {all(r['nb_model'] > r['nb_treat_all'] for r in dca_ext)}", flush=True)

# ============ 5. Hospital + MK + subgroups (with CI) ============
print("\n=== 5. Hospital/MK/subgroups ===", flush=True)
hmap = pd.read_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet")[["stay_id", "hospitalid"]]
pe2 = pe.merge(hmap, on='stay_id', how='left')
hp = pe2.groupby('hospitalid').apply(lambda g: roc_auc_score(g.y, g.p) if g.y.sum() >= 20 else np.nan).dropna()
hospital = {'n_ge20': int(len(hp)), 'median': round(float(hp.median()), 3),
            'min': round(float(hp.min()), 3), 'max': round(float(hp.max()), 3),
            'n_below_060': int((hp < 0.60).sum())}
print(f"  hospitals: {hospital}", flush=True)
from scipy.stats import kendalltau
strata = pe2.groupby(pd.cut(pe2['t_hr'], bins=np.arange(24, 169, 24))).apply(
    lambda g: roc_auc_score(g.y, g.p) if g.y.sum() > 0 else np.nan).dropna()
tau, pv = kendalltau(range(len(strata)), strata.values)
mk = {'tau': round(float(tau), 3), 'p': round(float(pv), 3),
      'strata': [round(float(v), 3) for v in strata.values]}
print(f"  MK: tau={tau:.3f} p={pv:.3f}", flush=True)

def sub_boot(df, n=500, seed=SEED):
    r = np.random.default_rng(seed)
    stays = df['stay_id'].values; uniq = pd.unique(stays)
    idx_by = {s: np.where(stays == s)[0] for s in uniq}
    yv, pv = df['y'].values, df['p'].values
    aucs = []
    for _ in range(n):
        sel = r.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in sel])
        if yv[idx].sum() > 0 and (1 - yv[idx]).sum() > 0:
            aucs.append(roc_auc_score(yv[idx], pv[idx]))
    return np.percentile(aucs, [2.5, 97.5]).round(3).tolist()

sex = pd.read_parquet(f"{R}/p1_expanded_features.parquet", columns=['stay_id', 'male', 'age']).drop_duplicates('stay_id')
pe3 = pe.merge(sex, on='stay_id', how='left')
sub = {}
for nm, msk in [('age<65', pe3['age'] < 65), ('age>=65', pe3['age'] >= 65),
                ('male', pe3['male'] == 1), ('female', pe3['male'] == 0)]:
    g = pe3[msk]
    sub[nm] = {'auroc': round(float(roc_auc_score(g.y, g.p)), 3), 'ci': sub_boot(g)}
    print(f"  {nm}: {sub[nm]}", flush=True)

# ============ 6. SHAP ============
print("\n=== 6. SHAP ===", flush=True)
import shap
ridx = np.random.default_rng(SEED).choice(np.where(te)[0], size=8000, replace=False)
Xs = m4f.iloc[ridx][FEATS]
expl = shap.TreeExplainer(mod1)
sv = expl.shap_values(Xs.values, check_additivity=False)
if isinstance(sv, list): sv = sv[1]
imp = np.abs(sv).mean(0)
order = np.argsort(imp)[::-1][:15]
shap_top = [{'feature': FEATS[i], 'mean_abs': round(float(imp[i]), 4)} for i in order]
pd.DataFrame(sv, columns=FEATS).assign(stay_id=m4f.iloc[ridx]['stay_id'].values).to_parquet(f"{R}/f2_shap_values.parquet", index=False)
Xs.assign(stay_id=m4f.iloc[ridx]['stay_id'].values).to_parquet(f"{R}/f2_shap_X.parquet", index=False)
print(f"  top5: {[x['feature'] for x in shap_top[:5]]}", flush=True)

# ============ 7. Proximity/horizon standardisation (verbatim round-8 logic) ============
print("\n=== 7. Proximity standardisation ===", flush=True)
BINS = [(0, 0), (6, 6), (12, 18), (24, 10 ** 6)]
def hload(c, g):
    d = pd.read_parquet(f"{R}/f1_preds_{c}_ge{g}.parquet")[["stay_id", "t_hr", "y", "p"]]
    d["dend"] = d.groupby("stay_id").t_hr.transform("max") - d.t_hr
    d["hb"] = -1
    for k, (lo, hi) in enumerate(BINS):
        d.loc[(d.dend >= lo) & (d.dend <= hi), "hb"] = k
    return d.sort_values(["stay_id", "t_hr"]).reset_index(drop=True)
def hweights(d, target_share):
    w = np.ones(len(d))
    pos = d.y.values == 1
    share = pd.Series(d.hb.values[pos]).value_counts(normalize=True)
    for k in range(len(BINS)):
        m = pos & (d.hb.values == k)
        if m.any() and share.get(k, 0) > 0:
            w[m] = target_share.get(k, 0) / share[k]
    return w
horizon = {"bins_h": ["0", "6", "12-18", ">=24"], "B": 1000, "seed": 42}
for c in ("ext", "int"):
    D = {g: hload(c, g) for g in (1, 2, 3)}
    assert (D[1][["stay_id", "t_hr"]].values == D[3][["stay_id", "t_hr"]].values).all()
    tgt = pd.Series(D[3].hb.values[D[3].y.values == 1]).value_counts(normalize=True).to_dict()
    r = {"ge3_pos_share_by_bin": {horizon["bins_h"][k]: round(float(tgt.get(k, 0)), 4) for k in range(4)}}
    for g in (1, 2, 3):
        pos = D[g].y.values == 1
        r[f"ge{g}_auc_by_bin"] = {}
        neg = ~pos
        for k in range(4):
            m = pos & (D[g].hb.values == k)
            sel = m | neg
            r[f"ge{g}_auc_by_bin"][horizon["bins_h"][k]] = {
                "n_pos": int(m.sum()),
                "auroc": (round(float(roc_auc_score(D[g].y.values[sel], D[g].p.values[sel])), 4) if m.sum() >= 10 else None)}
    W = {g: hweights(D[g], tgt) for g in (1, 2)}
    pt = {"ge3": roc_auc_score(D[3].y, D[3].p)}
    for g in (1, 2):
        pt[f"ge{g}_std"] = roc_auc_score(D[g].y, D[g].p, sample_weight=W[g])
        pt[f"ge{g}_raw"] = roc_auc_score(D[g].y, D[g].p)
    r["point"] = {k: round(float(v), 4) for k, v in pt.items()}
    stays = D[1].stay_id.values
    u, inv = np.unique(stays, return_inverse=True)
    idx_by = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(42)
    diffs = {"ge3_minus_ge1std": [], "ge3_minus_ge2std": [], "ge1std": [], "ge2std": []}
    for _b in range(1000):
        pick = rng.integers(0, len(u), len(u))
        ii = np.concatenate([idx_by[j] for j in pick])
        y3, p3 = D[3].y.values[ii], D[3].p.values[ii]
        if y3.sum() == 0 or y3.sum() == len(y3):
            continue
        a3b = roc_auc_score(y3, p3)
        for g in (1, 2):
            yg = D[g].y.values[ii]
            if yg.sum() == 0:
                break
            sub_ = D[g].iloc[ii].reset_index(drop=True)
            t3 = pd.Series(D[3].hb.values[ii][y3 == 1]).value_counts(normalize=True).to_dict()
            ag = roc_auc_score(yg, D[g].p.values[ii], sample_weight=hweights(sub_, t3))
            diffs[f"ge{g}std"].append(ag)
            diffs[f"ge3_minus_ge{g}std"].append(a3b - ag)
    for k, v in diffs.items():
        v = np.array(v)
        if k.startswith("ge3_minus"):
            p = 2 * min((v <= 0).mean(), (v >= 0).mean())
            key = k.split("_minus_")[1].replace("std", "_std")
            r[k] = {"diff": round(float(pt["ge3"] - pt[key]), 4),
                    "ci": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
                    "p_boot": max(round(float(p), 3), 0.001), "p_is_floor": bool(p < 0.001), "n_valid": int(len(v))}
        else:
            r[k + "_ci"] = [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    horizon[c] = r
    print(f"  {c}: point={r['point']} ge3-ge1std={r['ge3_minus_ge1std']}", flush=True)
json.dump(horizon, open(f"{R}/f2_horizon.json", "w"), indent=1)

# ============ 8. SM5 arms ============
print("\n=== 8. SM5 arms ===", flush=True)
# 8a. fill0 arm
fill0 = {}
Xtr0 = np.nan_to_num(m4f[FEATS].values[tr])
Xte0 = np.nan_to_num(m4f[FEATS].values[te])
Xe0 = np.nan_to_num(eif[FEATS].values)
for t in [1, 2, 3]:
    m0 = lgb.LGBMClassifier(**PARAMS)
    m0.fit(Xtr0, m4f[f'ge{t}'].values[tr])
    yi0 = m4f[f'ge{t}'].values[te]
    fill0[f'ge{t}'] = {'internal': round(float(roc_auc_score(yi0, m0.predict_proba(Xte0)[:, 1])), 4),
                       'external': round(float(roc_auc_score(eif[f'ge{t}'].values, m0.predict_proba(Xe0)[:, 1])), 4)}
    print(f"  fill0 ge{t}: {fill0[f'ge{t}']}", flush=True)
# 8b. threshold-specific models (all development data = train + internal test)
thr_spec = {}
Xall = m4f[FEATS].values
for t in [1, 2, 3]:
    ma = lgb.LGBMClassifier(**PARAMS)
    ma.fit(Xall, m4f[f'ge{t}'].values)
    thr_spec[f'ge{t}'] = {'external': round(float(roc_auc_score(eif[f'ge{t}'].values, ma.predict_proba(eif[FEATS].values)[:, 1])), 4)}
    print(f"  thr-spec ge{t}: {thr_spec[f'ge{t}']}", flush=True)
# 8c. FA autopsy at ext thr 0.15
d = pe.copy(); d['alert'] = d['p_rec'] >= 0.15
lastpos = d[d.y == 1].groupby('stay_id')['t_hr'].max()
fa_stays = set(d.loc[d['alert'], 'stay_id']) - set(lastpos.index)
# near-criterion: never-event stay whose max measured cr (any bin) reaches fold>=1.3 or rise>=0.2
ei_cr = pd.read_parquet(f"{R}/n8_2_labs_series.parquet")
crc = ei_cr[ei_cr['labname'] == 'creatinine'][['sid', 'hr', 'labresult']].rename(columns={'sid': 'stay_id', 'labresult': 'cr_val'})
crc = crc[crc['cr_val'].between(0.1, 30)]
crc['t_hr'] = (crc['hr'] // 6) * 6
cbmax = crc.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index()
eibase = eif[['stay_id', 't_hr', 'cr_base']].drop_duplicates()
mgx = eibase.merge(cbmax, on=['stay_id', 't_hr'], how='inner')
mgx['fold'] = mgx['cr_val'] / mgx['cr_base'].clip(lower=0.1)
mgx['rise'] = mgx['cr_val'] - mgx['cr_base']
near = mgx[(mgx['fold'] >= 1.3) | (mgx['rise'] >= 0.2)]['stay_id'].unique()
fa_autopsy = {'thr': 0.15, 'n_fa_stays': len(fa_stays),
              'fa_with_near_criteria_cr': round(float(np.mean([s in set(near) for s in fa_stays])), 3)}
cap_stays = [sid for sid, g in d[d.stay_id.isin(lastpos.index)].groupby('stay_id')
             if (g['alert'] & (g['t_hr'] < lastpos[sid])).any()]
lead_all = []
for sid in cap_stays:
    g = d[d.stay_id == sid]
    al = g[g['alert'] & (g['t_hr'] < lastpos[sid])]
    lead_all.append(lastpos[sid] - al['t_hr'].min())
fa_autopsy['captured_lead_gt48h_frac'] = round(float(np.mean(np.array(lead_all) > 48)), 3)
print(f"  FA autopsy: {fa_autopsy}", flush=True)

# ============ 9. EPPP (268 parameters) ============
dev_pos_ckpt = int(m4f['ge1'].values.sum())
dev_stays_event = int(m4f.loc[m4f['ge1'] == 1, 'stay_id'].nunique())
eppp = {'params': len(FEATS),
        'ckpt_level': round(dev_pos_ckpt / len(FEATS), 1),
        'patient_level': round(dev_stays_event / len(FEATS), 1),
        'dev_pos_checkpoints': dev_pos_ckpt, 'dev_event_stays': dev_stays_event}

# ============ SAVE ============
out = {'_meta': {'protocol': 'full downstream suite on reselected 268-feature set (standard operation, PI 2026-09-30)',
                 'date': '2026-09-30',
                 'selection': 'f1: 350 candidates (M4 AND eICU computable) -> 268 at cumulative 99% 5-fold GroupKFold CV gain, E02-corrected ge1 labels, train subjects only',
                 'deployment_defs': 'anchor = final positive checkpoint; capture = alert at t_hr < final positive; lead = final positive - first qualifying alert; NNE = alerted/captured stays; FA denominator = event-free patient-days (never-event full stay + event-stay pre-first-positive duration, floor 6h)',
                 'oof_calibration': 'GroupKFold(5) by subject (stricter than legacy plain 5-fold)'},
       'calibration_ge1': {'recal_a': round(a, 6), 'recal_b': round(b, 6), **calib},
       'deployment_ge1': {'internal_test': deploy_int, 'external': deploy_ext},
       'delong': delong,
       'dca': {'internal': dca_int, 'external': dca_ext,
               'wins': {'int': all(x['nb_model'] > x['nb_treat_all'] for x in dca_int),
                        'ext': all(x['nb_model'] > x['nb_treat_all'] for x in dca_ext)}},
       'hospital': hospital, 'mk': mk, 'subgroups_with_ci': sub,
       'shap_top15': shap_top, 'horizon': horizon,
       'sm5_arms': {'fill0': fill0, 'threshold_specific': thr_spec, 'fa_autopsy': fa_autopsy},
       'eppp_268': eppp}
json.dump(out, open(f"{R}/f2_full_results.json", "w"), indent=2)
print("\nALL SAVED: f2_full_results.json")
