#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ANALYSIS 4 - Quantify the label circularity (root cause).

Analysis 2b showed a single feature, cr_ratio_base, reproduces almost all of the
model's discrimination (0.766 vs 0.769 in MIMIC-IV; 0.748 vs 0.777 in eICU-CRD).

Why: the label is the maximum KDIGO stage in [t, t + 54 h), and that window
INCLUDES the current checkpoint t. So whenever the current-to-baseline ratio has
already crossed the Stage 1 threshold at t, the checkpoint is positive by
definition. A feature equal to the current ratio therefore contains the label
outright for those checkpoints.

This script measures, honestly:
  Q1 share of positive checkpoints whose label was ALREADY satisfied at t
     (ratio >= 1.5, or absolute rise >= 0.3 mg/dL where the rise is available)
  Q2 model AUROC restricted to negatives + only the NON-definitional positives
     (positives whose ratio at t is still below threshold) - i.e. the genuine
     forecasting task
  Q3 the same restricted to >= 24 h before onset
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import sys, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('E:/TBI subtype/09_tbi_aki')
OUT = ROOT / '_analysis_r13' / 'out'

COH = {
    'MIMIC-IV': dict(
        grid=ROOT / 'results/n3_m4_abi_grid.parquet',
        pred=ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet',
        feat=ROOT / 'results/n5_abi_rolling_features.parquet'),
    'eICU-CRD': dict(
        grid=ROOT / 'results/n8_3_eicu_abi_grid.parquet',
        pred=ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet',
        feat=ROOT / 'results/p1_expanded_features.parquet'),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def auroc(y, s):
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or y.min() == y.max():
        return np.nan, 0
    return float(roc_auc_score(y, s)), int(m.sum())


def run(name, cfg):
    print()
    print('=' * 70)
    print(name)
    print('=' * 70)
    g = pd.read_parquet(cfg['grid'])
    p = pd.read_parquet(cfg['pred'])
    f = pd.read_parquet(cfg['feat'])

    cols = ['stay_id', 't_hr', 'cr_ratio_base']
    for c in ('cr_delta_since_adm', 'cr_base', 'cr_last'):
        if c in f.columns:
            cols.append(c)
    d = g[['stay_id', 't_hr', 'onset_hr']].merge(p[['stay_id', 't_hr', 'y', 'p']],
                                                 on=['stay_id', 't_hr'], how='inner')
    d = d.merge(f[cols], on=['stay_id', 't_hr'], how='left')
    print('  rows %d | stays %d | positives %d' % (len(d), d['stay_id'].nunique(), int(d['y'].sum())))

    ratio = pd.to_numeric(d['cr_ratio_base'], errors='coerce')
    # ---- Q1: how many positives already meet the criterion at t ----
    defn_ratio = ratio >= 1.5
    if 'cr_delta_since_adm' in d.columns:
        rise = pd.to_numeric(d['cr_delta_since_adm'], errors='coerce')
        defn_rise = rise >= 0.3
    else:
        defn_rise = pd.Series(False, index=d.index)
    defn = defn_ratio | defn_rise

    pos = d['y'] == 1
    pos_known = pos & ratio.notna()
    n_pos = int(pos.sum())
    n_pos_known = int(pos_known.sum())
    n_defn = int((pos_known & defn).sum())
    print()
    print('  --- Q1: positives whose criterion was ALREADY met at t ---')
    print('    positive checkpoints                       : %d' % n_pos)
    print('    with cr_ratio_base observed                : %d' % n_pos_known)
    print('    of those, ratio >= 1.5 or rise >= 0.3 at t : %d  (%.1f%% of observable positives)'
          % (n_defn, 100 * n_defn / n_pos_known if n_pos_known else np.nan))
    print('    (definitional: the label window [t, t+54h) contains t itself)')

    # ---- Q2: forecasting task = negatives + only non-definitional positives ----
    keep = (~pos) | (pos & ratio.notna() & ~defn)
    sub = d[keep]
    a_model, n_used = auroc(sub['y'], sub['p'])
    a_ratio, _ = auroc(sub['y'], pd.to_numeric(sub['cr_ratio_base'], errors='coerce'))
    print()
    print('  --- Q2: forecasting-only task (definitional positives removed) ---')
    print('    rows kept %d | positives %d' % (len(sub), int(sub['y'].sum())))
    print('    model AUROC            : %.4f' % a_model)
    print('    cr_ratio_base alone    : %.4f' % a_ratio)

    # ---- Q3: same, additionally restricted to >= 24 h before onset ----
    keep24 = keep & ((d['y'] == 0) | (d['onset_hr'].notna() & ((d['onset_hr'] - d['t_hr']) >= 24)))
    sub2 = d[keep24]
    a_model24, _ = auroc(sub2['y'], sub2['p'])
    a_ratio24, _ = auroc(sub2['y'], pd.to_numeric(sub2['cr_ratio_base'], errors='coerce'))
    print()
    print('  --- Q3: forecasting-only AND >= 24 h before onset ---')
    print('    rows kept %d | positives %d' % (len(sub2), int(sub2['y'].sum())))
    print('    model AUROC            : %.4f' % a_model24)
    print('    cr_ratio_base alone    : %.4f' % a_ratio24)

    return {
        'n_rows': len(d), 'n_pos': n_pos, 'n_pos_observed': n_pos_known,
        'n_pos_definitional': n_defn,
        'share_pos_definitional': round(n_defn / n_pos_known, 6) if n_pos_known else None,
        'forecast_only': {'model_auroc': round(a_model, 6),
                          'ratio_auroc': round(a_ratio, 6),
                          'n_pos': int(sub['y'].sum()), 'n_rows': int(len(sub))},
        'forecast_only_24h': {'model_auroc': round(a_model24, 6),
                              'ratio_auroc': round(a_ratio24, 6),
                              'n_pos': int(sub2['y'].sum()), 'n_rows': int(len(sub2))},
        'sha': {k: sha256(v)[:16] for k, v in cfg.items()},
    }


def mini_task():
    res = {}
    for nm, cfg in COH.items():
        res[nm] = run(nm, cfg)

    print()
    print('=' * 70)
    print('HEADLINE')
    print('=' * 70)
    for nm, r in res.items():
        print('  %-9s %.0f%% of observable positives already met the criterion at t;'
              % (nm, 100 * (r['share_pos_definitional'] or 0)))
        print('             on the forecasting-only task the model reaches %.4f (vs %.4f headline)'
              % (r['forecast_only']['model_auroc'], r.get('headline_base', float('nan'))))
    (OUT / 'a4_circularity_quantified.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print()
    print('  written ->', OUT / 'a4_circularity_quantified.json')


if __name__ == '__main__':
    print('=' * 70)
    print('ANALYSIS 4: quantify label circularity')
    print('=' * 70)
    mini_task()
