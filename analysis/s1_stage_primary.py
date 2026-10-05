# -*- coding: utf-8 -*-
"""
S1: KDIGO stage-stratified PRIMARY analysis (v2 framing, 2026-09-28).

New primary protocol (replaces the pre-registered temporal-split union analysis):
  - Endpoints: creatinine-based KDIGO >=Stage 1 (primary), >=Stage 2, >=Stage 3
  - Split: SUBJECT-level random 80/20 (seed 42), zero patient overlap
  - External: frozen model applied zero-touch to full eICU-CRD grid
  - Model: LightGBM, 291 frozen features, primary hyperparameters, native NaN

Computes everything the new manuscript needs that the old (union-endpoint)
chain cannot supply:
  1. AUROC + AUPRC, internal test + external, all three thresholds
  2. Stay-cluster bootstrap 95% CIs (1000x) for all six AUROCs
     + external two-level (hospital->stay) bootstrap if hospitalid available
  3. Calibration for >=Stage 1: logistic recalibration fitted on the TRAIN
     split, applied to test + external (slope/ECE/Brier before/after)
  4. Deployment views for >=Stage 1 (recalibrated probs, thr 0.10/0.20/0.30):
     capture, median lead time (onset = first criteria-met bin >=24h),
     NNE (alerted stays / captured event stays), false alarms per 100
     event-free patient-days (internal test + external)
  5. Static logistic-regression baseline (admission variables) under >=Stage 1
  6. Per-hospital external AUROC distribution for >=Stage 1
  7. Mann-Kendall trend of AUROC across checkpoint-hour strata (external)
  8. Age/sex subgroup AUROCs (external, >=Stage 1)
  9. SHAP top-15 features (TreeExplainer, 8k random test-split checkpoints)
 10. Saves per-checkpoint predictions parquet for figure building.

Output: results/s1_stage_primary.json + results/s1_preds_external.parquet
        + results/s1_preds_internal.parquet
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

R = "results"
SEED = 42
NBOOT = 1000

# ============ 1. data ============
print("loading features...", flush=True)
FEATS = json.load(open(f"{R}/_frozen291_features.json"))
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=SEED, verbose=-1)
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[['stay_id'] + FEATS]
eicu = pd.read_parquet(f"{R}/p1_expanded_features.parquet")[['stay_id'] + FEATS]
print(f"M4 {m4.shape}, eICU {eicu.shape}", flush=True)


def stage_labels(labs_parquet, feat_df, db):
    labs = pd.read_parquet(labs_parquet)
    if 'sid' in labs.columns:
        cr = labs[labs['labname'] == 'creatinine'].rename(
            columns={'sid': 'stay_id', 'labresult': 'cr_val'})
    else:
        cr = labs.rename(columns={'valuenum': 'cr_val'})
    cr = cr[cr['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
    cr['t_hr'] = (cr['hr'] // 6) * 6
    cb = cr.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index()
    cb = cb.sort_values(['stay_id', 't_hr'])
    mg = feat_df[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(
        cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
    mg['f'] = np.nan
    for s in range(9):
        sh = mg.groupby('stay_id')['cr_val'].shift(-s)
        mg['f'] = pd.DataFrame({'c': mg['f'], 'n': sh}).max(axis=1)
    b = mg['cr_base'].clip(lower=0.1)
    fold = mg['f'] / b
    st = pd.Series(0, index=mg.index)
    st[(fold >= 1.5) | ((mg['f'] - mg['cr_base']) >= 0.3)] = 1
    st[fold >= 2.0] = 2
    st[(fold >= 3.0) | (mg['f'] >= 4.0)] = 3
    st[mg['f'].isna()] = 0
    for t in [1, 2, 3]:
        mg[f'ge{t}'] = (st >= t).astype(int)
    # onset bin for >=Stage 1: first bin >=24h where running criteria met
    crit = (fold >= 1.5) | ((mg['f'] - mg['cr_base']) >= 0.3)
    crit = crit.fillna(False) & (mg['t_hr'] >= 24)
    onset = crit[crit].groupby(level=0).first() if False else None
    mg['crit'] = crit
    onsets = {}
    for sid, g in mg[mg['crit']].groupby('stay_id'):
        onsets[sid] = g['t_hr'].min()
    print(f"  {db}: ge1={mg['ge1'].sum()} ge2={mg['ge2'].sum()} ge3={mg['ge3'].sum()}; "
          f"event stays ge1={len(onsets)}", flush=True)
    keep = ['stay_id', 't_hr', 'ge1', 'ge2', 'ge3']
    return mg[keep], onsets


m4_lab, m4_onset = stage_labels(f"{R}/n2_m4_abi_cr_serial.parquet", m4, "M4")
ei_lab, _ = stage_labels(f"{R}/n8_2_labs_series.parquet", eicu, "eICU")

m4f = m4.merge(m4_lab, on=['stay_id', 't_hr'], how='left')
eif = eicu.merge(ei_lab, on=['stay_id', 't_hr'], how='left')

# ============ 2. subject-level split ============
n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
# NB: MUST match p2b_stage_291feat.py split exactly (np.random.seed(42) permutation,
# not default_rng) so every number reproduces the authoritative 0.849/0.801 set.
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr
assert len(set(m4f.loc[tr, 'stay_id'].map(s2sub)) & set(m4f.loc[te, 'stay_id'].map(s2sub))) == 0
print(f"split: {tr.sum()}/{te.sum()} ckpts (subject-level 80/20)", flush=True)

# hospital ids for external (if available)
hos = None
try:
    import pyarrow.parquet as pq
    allcols = pq.read_schema(f"{R}/p1_expanded_features.parquet").names
    hc = [c for c in allcols if 'hospital' in c.lower()]
    if hc:
        hmap = pd.read_parquet(f"{R}/p1_expanded_features.parquet",
                               columns=['stay_id', hc[0]])
        hos = dict(zip(hmap['stay_id'], hmap[hc[0]]))
        print(f"hospital col: {hc[0]}", flush=True)
except Exception as e:
    print("no hospital col:", e, flush=True)

# ============ 3. train 3 stage models + evaluate ============
out = {'_meta': {
    'protocol': 'KDIGO creatinine staging >=1/>=2/>=3 as endpoints; subject-level random 80/20 seed42; '
                'LightGBM 291 frozen features (800/0.02/127/balanced) native NaN; zero-touch eICU external',
    'date': '2026-09-28', 'nboot': NBOOT}}
preds_int, preds_ext = {}, {}
for t in [1, 2, 3]:
    y = m4f[f'ge{t}'].values
    mod = lgb.LGBMClassifier(**PARAMS)
    mod.fit(m4f[FEATS].values[tr], y[tr])
    pi = mod.predict_proba(m4f[FEATS].values[te])[:, 1]
    pe = mod.predict_proba(eif[FEATS].values)[:, 1]
    yi, ye = y[te], eif[f'ge{t}'].values
    preds_int[t] = pd.DataFrame({'stay_id': m4f.loc[te, 'stay_id'].values,
                                 'y': yi, 'p': pi, 't_hr': m4f.loc[te, 't_hr'].values})
    preds_ext[t] = pd.DataFrame({'stay_id': eif['stay_id'].values, 'y': ye, 'p': pe,
                                 't_hr': eif['t_hr'].values})
    print(f"ge{t}: int={roc_auc_score(yi, pi):.4f} ext={roc_auc_score(ye, pe):.4f} "
          f"AUPRC int={average_precision_score(yi, pi):.4f} ext={average_precision_score(ye, pe):.4f}",
          flush=True)

# dedicated models (train all M4 per threshold) with external preds saved
for t in [2, 3]:
    mod = lgb.LGBMClassifier(**PARAMS)
    mod.fit(m4f[FEATS].values, m4f[f'ge{t}'].values)
    pd_ = mod.predict_proba(eif[FEATS].values)[:, 1]
    preds_ext[10 + t] = pd.DataFrame({'stay_id': eif['stay_id'].values,
                                      'y': eif[f'ge{t}'].values, 'p': pd_,
                                      't_hr': eif['t_hr'].values})

# ============ 4. bootstrap CIs ============
def stay_boot(df, n=NBOOT, seed=SEED):
    r = np.random.default_rng(seed)
    stays = df['stay_id'].values
    uniq = pd.unique(stays)
    idx_by_stay = {s: np.where(stays == s)[0] for s in uniq}
    yv, pv = df['y'].values, df['p'].values
    aucs = []
    for _ in range(n):
        sel = r.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_stay[s] for s in sel])
        yy, pp = yv[idx], pv[idx]
        if yy.sum() > 0 and (1 - yy).sum() > 0:
            aucs.append(roc_auc_score(yy, pp))
    return np.percentile(aucs, [2.5, 97.5])


print("bootstrap CIs...", flush=True)
ci = {}
for t in [1, 2, 3]:
    ci[f'ge{t}_int'] = stay_boot(preds_int[t]).round(4).tolist()
    ci[f'ge{t}_ext'] = stay_boot(preds_ext[t]).round(4).tolist()
    print(f"  ge{t} int CI {ci[f'ge{t}_int']} ext CI {ci[f'ge{t}_ext']}", flush=True)
# two-level hospital bootstrap external ge1 (if hospital available)
if hos:
    df = preds_ext[1].copy()
    df['h'] = df['stay_id'].map(hos)
    r = np.random.default_rng(SEED)
    hs = pd.unique(df['h'].values)
    idx_by_h = {h: np.where(df['h'].values == h)[0] for h in hs}
    yv, pv = df['y'].values, df['p'].values
    aucs = []
    for _ in range(NBOOT):
        sel = r.choice(hs, size=len(hs), replace=True)
        idx = np.concatenate([idx_by_h[h] for h in sel])
        yy, pp = yv[idx], pv[idx]
        if yy.sum() > 0 and (1 - yy).sum() > 0:
            aucs.append(roc_auc_score(yy, pp))
    ci['ge1_ext_hospital2level'] = np.percentile(aucs, [2.5, 97.5]).round(4).tolist()
    print(f"  ge1 ext 2-level CI {ci['ge1_ext_hospital2level']}", flush=True)

# ============ 5. calibration ge1 ============
print("calibration ge1...", flush=True)
mod1 = lgb.LGBMClassifier(**PARAMS)
mod1.fit(m4f[FEATS].values[tr], m4f['ge1'].values[tr])
ptr = mod1.predict_proba(m4f[FEATS].values[tr])[:, 1]
pte = mod1.predict_proba(m4f[FEATS].values[te])[:, 1]
pee = mod1.predict_proba(eif[FEATS].values)[:, 1]
ytr, yte, yee = m4f['ge1'].values[tr], m4f['ge1'].values[te], eif['ge1'].values


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


from sklearn.linear_model import LogisticRegression
cal = LogisticRegression(C=1e6)
cal.fit(logit(ptr).reshape(-1, 1), ytr)
a, b0 = float(cal.intercept_[0]), float(cal.coef_[0][0])


def ece10(y, p):
    q = pd.qcut(pd.Series(p), 10, duplicates='drop')
    g = pd.DataFrame({'y': y, 'p': p, 'q': q}).groupby('q', observed=True)
    return float((g.size() / len(y) * np.abs(g['y'].mean() - g['p'].mean())).sum())


def slope_ece(y, p):
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), ece10(y, p), float(np.mean((p - y) ** 2))


def brier_null(y):
    return float(np.mean((np.mean(y) - y) ** 2))


calib = {'recal': {'a': round(a, 4), 'b': round(b0, 4)}}
for nm, yy, pp in [('int_test', yte, pte), ('ext', yee, pee)]:
    pr = sigmoid(a + b0 * logit(pp))
    s0, e0, br0 = slope_ece(yy, pp)
    s1_, e1_, br1 = slope_ece(yy, pr)
    calib[nm] = {'raw': {'slope': round(s0, 3), 'ece': round(e0, 4), 'brier': round(br0, 4)},
                 'recal': {'slope': round(s1_, 3), 'ece': round(e1_, 4), 'brier': round(br1, 4)},
                 'brier_null': round(brier_null(yy), 4)}
    print(f"  {nm}: slope {s0:.2f}->{s1_:.2f} ECE {e0:.4f}->{e1_:.4f}", flush=True)
# store recalibrated ge1 probs for deployment
preds_int[1]['p_rec'] = sigmoid(a + b0 * logit(preds_int[1]['p']))
preds_ext[1]['p_rec'] = sigmoid(a + b0 * logit(preds_ext[1]['p']))

# ============ 6. deployment views ge1 ============
print("deployment ge1...", flush=True)
# external onset via stage_labels onsets for eICU
_, ei_onset = stage_labels(f"{R}/n8_2_labs_series.parquet", eicu, "eICU(onset)")


def deploy(df, onsets, thr):
    d = df.copy()
    d['alert'] = d['p_rec'] >= thr
    ev = d[d['y'] == 1].copy()
    nev = d[d['y'] == 0].copy()
    ev['onset'] = ev['stay_id'].map(onsets)
    capt, leads = [], []
    for sid, g in ev.groupby('stay_id'):
        al = g[g['alert'] & (g['t_hr'] < g['onset'])]
        if len(al):
            capt.append(sid)
            leads.append(g['onset'].min() - al['t_hr'].min())
    # alert episodes: stays with any alert
    alert_stays = set(d.loc[d['alert'], 'stay_id'])
    n_alert = len(alert_stays)
    n_capt = len(capt)
    nne = n_alert / n_capt if n_capt else float('nan')
    # false alarms per 100 event-free patient-days (event-free = y==0 stays)
    fa_stays = alert_stays & set(nev['stay_id'])
    days = nev.groupby('stay_id')['t_hr'].max().clip(lower=6) / 24.0
    days = days.sum()
    fa = 100 * len(fa_stays) / days if days else float('nan')
    return {'thr': thr, 'capture': round(n_capt / ev['stay_id'].nunique(), 3),
            'lead_median_h': round(float(np.median(leads)), 1) if leads else None,
            'lead_iqr_h': [round(float(np.percentile(leads, 25)), 1),
                           round(float(np.percentile(leads, 75)), 1)] if leads else None,
            'NNE': round(nne, 2), 'FA_per_100ptd': round(fa, 1)}


deploy_int = {f"thr{t}": deploy(preds_int[1], m4_onset, t) for t in [0.10, 0.20, 0.30]}
deploy_ext = {f"thr{t}": deploy(preds_ext[1], ei_onset, t) for t in [0.10, 0.20, 0.30]}
print(json.dumps(deploy_ext['thr0.2'], ensure_ascii=False), flush=True)

# ============ 7. static LR baseline ge1 ============
cand = [c for c in ['age', 'sex', 'gender', 'sex_female', 'charlson', 'charlson_index',
                    'cr_base'] if c in FEATS]
Xb_tr = m4f[cand].fillna(0).values[tr]
Xb_te = m4f[cand].fillna(0).values[te]
Xb_ee = eif[cand].fillna(0).values
from sklearn.linear_model import LogisticRegression as LR
lrb = LR(max_iter=2000, class_weight='balanced')
lrb.fit(Xb_tr, ytr)
lr_int = roc_auc_score(yte, lrb.predict_proba(Xb_te)[:, 1])
lr_ext = roc_auc_score(yee, lrb.predict_proba(Xb_ee)[:, 1])
print(f"LR baseline ({cand}): int={lr_int:.4f} ext={lr_ext:.4f}", flush=True)

# ============ 8. per-hospital + MK + subgroups (external ge1) ============
d1 = preds_ext[1].copy()
extra = {}
if hos:
    d1['h'] = d1['stay_id'].map(hos)
    hp = d1.groupby('h').apply(
        lambda g: roc_auc_score(g['y'], g['p']) if g['y'].sum() >= 20 else np.nan)
    hp = hp.dropna()
    extra['hospital_auroc'] = {'n_ge20': int(len(hp)), 'median': round(float(hp.median()), 3),
                               'min': round(float(hp.min()), 3), 'max': round(float(hp.max()), 3),
                               'n_below_060': int((hp < 0.60).sum())}
    print(f"per-hospital: {extra['hospital_auroc']}", flush=True)
# MK by checkpoint-hour stratum
try:
    from scipy.stats import kendalltau
    strata = d1.groupby(pd.cut(d1['t_hr'], bins=np.arange(24, 169, 24), right=True))\
        .apply(lambda g: roc_auc_score(g['y'], g['p']) if g['y'].sum() > 0 else np.nan).dropna()
    tau, pv = kendalltau(range(len(strata)), strata.values)
    extra['MK_external_ge1'] = {'tau': round(float(tau), 3), 'p': round(float(pv), 3),
                                'strata_auroc': [round(float(v), 3) for v in strata.values]}
    print(f"MK external: tau={tau:.3f} p={pv:.3f}", flush=True)
except Exception as e:
    print("MK fail:", e, flush=True)
# subgroups (age/sex from feature frame)
sub = {}
for nm, col in [('age_lt65', 'age'), ]:
    if col in FEATS:
        pass
if 'age' in FEATS:
    age_map = eif[['stay_id', 'age']].drop_duplicates('stay_id').set_index('stay_id')['age']
    d1['age'] = d1['stay_id'].map(age_map)
    for nm, m in [('age<65', d1['age'] < 65), ('age>=65', d1['age'] >= 65)]:
        g = d1[m]
        sub[nm] = round(float(roc_auc_score(g['y'], g['p'])), 3)
sexcands = [c for c in FEATS if 'sex' in c or 'gender' in c]
if sexcands:
    sc = sexcands[0]
    smap = eif[['stay_id', sc]].drop_duplicates('stay_id').set_index('stay_id')[sc]
    d1[sc] = d1['stay_id'].map(smap)
    vals = d1[sc].dropna().unique()
    if len(vals) == 2:
        for v in vals:
            g = d1[d1[sc] == v]
            sub[f'{sc}={v}'] = round(float(roc_auc_score(g['y'], g['p'])), 3)
extra['subgroups_ext_ge1'] = sub
print("subgroups:", sub, flush=True)

# ============ 9. SHAP top features (ge1 model) ============
try:
    import shap
    ridx = np.random.default_rng(SEED).choice(len(pte), size=8000, replace=False)
    Xs = m4f[FEATS].values[te][ridx]
    expl = shap.TreeExplainer(mod1)
    sv = expl.shap_values(Xs, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    imp = np.abs(sv).mean(0)
    order = np.argsort(imp)[::-1][:15]
    extra['shap_top15'] = [{'feature': FEATS[i], 'mean_abs': round(float(imp[i]), 4)}
                           for i in order]
    print("SHAP top5:", [x['feature'] for x in extra['shap_top15'][:5]], flush=True)
except Exception as e:
    print("SHAP fail:", e, flush=True)

# ============ 10. assemble + save ============
res = {}
for t in [1, 2, 3]:
    res[f"ge{t}"] = {
        'internal': round(float(roc_auc_score(preds_int[t]['y'], preds_int[t]['p'])), 4),
        'external': round(float(roc_auc_score(preds_ext[t]['y'], preds_ext[t]['p'])), 4),
        'auprc_int': round(float(average_precision_score(preds_int[t]['y'], preds_int[t]['p'])), 4),
        'auprc_ext': round(float(average_precision_score(preds_ext[t]['y'], preds_ext[t]['p'])), 4),
        'events_int': int(preds_int[t]['y'].sum()),
        'events_ext': int(preds_ext[t]['y'].sum())}
    res[f'ge{t}']['ci_int'] = ci[f'ge{t}_int']
    res[f'ge{t}']['ci_ext'] = ci[f'ge{t}_ext']
for t in [2, 3]:
    res[f'ge{t}_dedicated_ext'] = round(
        float(roc_auc_score(preds_ext[10 + t]['y'], preds_ext[10 + t]['p'])), 4)

out.update({'results': res, 'calibration_ge1': calib,
            'deployment_ge1': {'internal_test': deploy_int, 'external': deploy_ext},
            'lr_baseline_ge1': {'vars': cand, 'internal': round(float(lr_int), 4),
                                'external': round(float(lr_ext), 4)},
            'extra': extra})
if hos:
    out['results']['ge1']['ci_ext_hospital2level'] = ci.get('ge1_ext_hospital2level')
json.dump(out, open(f"{R}/s1_stage_primary.json", "w"), indent=2)
preds_ext[1].to_parquet(f"{R}/s1_preds_external.parquet", index=False)
preds_int[1].to_parquet(f"{R}/s1_preds_internal.parquet", index=False)
for t in [2, 3]:
    preds_ext[t].to_parquet(f"{R}/s1_preds_ext_ge{t}.parquet", index=False)
    preds_ext[10 + t].to_parquet(f"{R}/s1_preds_ext_ge{t}_ded.parquet", index=False)
print("\nSAVED: s1_stage_primary.json + prediction parquets", flush=True)
print(json.dumps(res, ensure_ascii=False, indent=1)[:1500], flush=True)
