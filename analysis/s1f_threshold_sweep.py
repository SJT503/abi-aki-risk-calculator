# -*- coding: utf-8 -*-
"""s1f: deployment sweep over a calibrated-threshold grid for the >=Stage 1 model,
to select informative working points on the recalibrated scale (the class-weighted
raw scale shrinks after recalibration, so v1-style 0.10/0.20/0.30 are too strict
against a 4.7% base rate). Updates s1_stage_primary.json deployment block."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

R = "results"
pi = pd.read_parquet(f"{R}/s1_preds_internal.parquet")
pe = pd.read_parquet(f"{R}/s1_preds_external.parquet")


def onsets_from(labs_parquet):
    labs = pd.read_parquet(labs_parquet)
    cr = labs.rename(columns={'valuenum': 'cr_val'}) if 'sid' not in labs.columns else \
        labs[labs['labname'] == 'creatinine'].rename(columns={'sid': 'stay_id', 'labresult': 'cr_val'})
    cr = cr[cr['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
    cr['t_hr'] = (cr['hr'] // 6) * 6
    mg = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet", columns=['stay_id', 't_hr', 'cr_base']) \
        if 'valuenum' in labs.columns else \
        pd.read_parquet(f"{R}/p1_expanded_features.parquet", columns=['stay_id', 't_hr', 'cr_base'])
    cb = cr.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index()
    mg = mg.drop_duplicates(['stay_id', 't_hr']).merge(cb, on=['stay_id', 't_hr'], how='left')
    fold = mg['cr_val'] / mg['cr_base'].clip(lower=0.1)
    crit = ((fold >= 1.5) | ((mg['cr_val'] - mg['cr_base']) >= 0.3)).fillna(False) & (mg['t_hr'] >= 24)
    mg['crit'] = crit
    return {sid: g['t_hr'].min() for sid, g in mg[mg['crit']].groupby('stay_id')}


def deploy(df, onsets, thr):
    d = df.copy()
    d['alert'] = d['p_rec'] >= thr
    ev = d[d['y'] == 1].copy(); nev = d[d['y'] == 0].copy()
    ev['onset'] = ev['stay_id'].map(onsets)
    capt, leads = [], []
    for sid, g in ev.groupby('stay_id'):
        al = g[g['alert'] & (g['t_hr'] < g['onset'])]
        if len(al):
            capt.append(sid); leads.append(g['onset'].min() - al['t_hr'].min())
    alert_stays = set(d.loc[d['alert'], 'stay_id'])
    n_capt = len(capt)
    fa_stays = alert_stays & set(nev['stay_id'])
    days = nev.groupby('stay_id')['t_hr'].max().clip(lower=6).div(24).sum()
    return {'thr': thr, 'n_alert_stays': len(alert_stays),
            'capture': round(n_capt / ev['stay_id'].nunique(), 3),
            'lead_median_h': round(float(np.median(leads)), 1) if leads else None,
            'lead_iqr_h': [round(float(np.percentile(leads, 25)), 1),
                           round(float(np.percentile(leads, 75)), 1)] if leads else None,
            'NNE': round(len(alert_stays) / n_capt, 2) if n_capt else None,
            'FA_per_100ptd': round(100 * len(fa_stays) / days, 1)}


grid = [0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30]
mo = onsets_from(f"{R}/n2_m4_abi_cr_serial.parquet")
eo = onsets_from(f"{R}/n8_2_labs_series.parquet")
print(f"{'thr':>6} | {'int capt':>8} {'int NNE':>8} {'int lead':>8} | {'ext capt':>8} {'ext NNE':>8} {'ext lead':>8} {'ext FA':>7}")
for t in grid:
    di = deploy(pi, mo, t); de = deploy(pe, eo, t)
    print(f"{t:>6.3f} | {di['capture']:>8} {str(di['NNE']):>8} {str(di['lead_median_h']):>8} | "
          f"{de['capture']:>8} {str(de['NNE']):>8} {str(de['lead_median_h']):>8} {de['FA_per_100ptd']:>7}")
    for store, dd in [('int', di), ('ext', de)]:
        pass

# save full grid into json
s1 = json.load(open(f"{R}/s1_stage_primary.json"))
s1['deployment_ge1'] = {
    'internal_test': {f"thr{t}": deploy(pi, mo, t) for t in grid},
    'external': {f"thr{t}": deploy(pe, eo, t) for t in grid}}
json.dump(s1, open(f"{R}/s1_stage_primary.json", "w"), indent=2)
print("deployment grid saved")
