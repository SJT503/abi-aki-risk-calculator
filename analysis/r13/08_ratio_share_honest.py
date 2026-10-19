#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AUDIT 8 - the ratio's share of excess discrimination under honest forecasting.

Audit 7 fixed the row-set mismatch for the headline comparison. The remaining
question is the one that decides whether the novelty survives: when the task is
restricted to genuine prediction (>= 24 h before onset) on a MATCHED row set,
how much of the model's excess discrimination does cr_ratio_base still carry?
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import sys, json
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


def auroc(y, s):
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() == 0 or y[m].min() == y[m].max():
        return np.nan, 0
    return float(roc_auc_score(y[m], s[m])), int(m.sum())


def share(m, r):
    return (r - 0.5) / (m - 0.5)


print('=' * 78)
print('AUDIT 8  ratio share under honest forecasting (matched rows)')
print('=' * 78)
res = {}

for nm, cfg in COH.items():
    print()
    print('=' * 78)
    print(nm)
    print('=' * 78)
    g = pd.read_parquet(cfg['grid'], columns=['stay_id', 't_hr', 'onset_hr'])
    p = pd.read_parquet(cfg['pred'])
    f = pd.read_parquet(cfg['feat'], columns=['stay_id', 't_hr', 'cr_ratio_base'])
    d = g.merge(p[['stay_id', 't_hr', 'y', 'p']], on=['stay_id', 't_hr'], how='inner')
    d = d.merge(f, on=['stay_id', 't_hr'], how='left')
    r = pd.to_numeric(d['cr_ratio_base'], errors='coerce')

    # definitions
    is_pos = d['y'] == 1
    defn = (r >= 1.5) | (pd.to_numeric(d.get('cr_delta_since_adm', pd.Series(np.nan, index=d.index)),
                                        errors='coerce') >= 0.3)
    # cr_delta_since_adm not loaded above; recompute definitional on ratio alone is unsafe,
    # so reload that column explicitly
    f2 = pd.read_parquet(cfg['feat'], columns=['stay_id', 't_hr', 'cr_delta_since_adm'])
    d = d.drop(columns=[c for c in ('cr_delta_since_adm',) if c in d.columns]).merge(
        f2, on=['stay_id', 't_hr'], how='left')
    rise = pd.to_numeric(d['cr_delta_since_adm'], errors='coerce')
    defn = (r >= 1.5) | (rise >= 0.3)

    # forecasting-only AND >= 24h before onset, AND ratio observed (matched rows)
    keep = r.notna() & (
        (~is_pos) | ((is_pos) & (~defn) & d['onset_hr'].notna() & ((d['onset_hr'] - d['t_hr']) >= 24))
    )
    sub = d[keep]
    am, n = auroc(sub['y'], sub['p'])
    ar, _ = auroc(sub['y'], pd.to_numeric(sub['cr_ratio_base'], errors='coerce'))
    npos = int(sub['y'].sum())
    print('  honest-forecast matched subset : rows %d | positives %d' % (n, npos))
    print('    model AUROC                  : %.4f' % am)
    print('    cr_ratio_base alone          : %.4f' % ar)
    print('    ratio share of excess        : %.1f%%' % (100 * share(am, ar)))

    # for contrast, the unrestricted matched comparison from audit 7
    ok = r.notna()
    am0, _ = auroc(d.loc[ok, 'y'], d.loc[ok, 'p'])
    ar0, _ = auroc(d.loc[ok, 'y'], r[ok])
    print('  contrast, all matched rows     : model %.4f | ratio %.4f | share %.1f%%'
          % (am0, ar0, 100 * share(am0, ar0)))

    res[nm] = {'n_rows': n, 'n_pos': npos,
               'model': round(am, 6), 'ratio': round(ar, 6),
               'share_pct': round(100 * share(am, ar), 2),
               'share_all_matched_pct': round(100 * share(am0, ar0), 2)}

print()
print('=' * 78)
print('SUMMARY  how much of the model is the one ratio?')
print('=' * 78)
for nm, v in res.items():
    print('  %-9s all matched rows %5.1f%%  ->  honest >=24h forecast %5.1f%%  (model %.4f vs ratio %.4f)'
          % (nm, v['share_all_matched_pct'], v['share_pct'], v['model'], v['ratio']))

(OUT / 'a8_ratio_share_honest.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
print()
print('  written ->', OUT / 'a8_ratio_share_honest.json')
