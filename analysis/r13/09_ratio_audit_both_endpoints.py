#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AUDIT 9 - extend the cr_ratio_base audit to the SEVERE endpoint (ge3).

The r13 / audit-7 findings were measured on the any-AKI endpoint ONLY (ge1).
The paper reports TWO endpoints and the frozen pipeline trains a SEPARATE
LightGBM per endpoint (f12_jinhua_full_suite.py:95-99). So "one ratio carries
the model" has been shown for any-AKI and never checked on severe AKI.

This script measures the same quantities on both endpoints side by side.
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

GRID = {
    'MIMIC-IV': dict(
        ge1=ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet',
        ge3=ROOT / 'results/jinhu/f12_preds_M4_ge3.parquet',
        feat=ROOT / 'results/n5_abi_rolling_features.parquet'),
    'eICU-CRD': dict(
        ge1=ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet',
        ge3=ROOT / 'results/jinhu/f12_preds_eICU_ge3.parquet',
        feat=ROOT / 'results/p1_expanded_features.parquet'),
}

STAGE_THR = {1: 1.5, 3: 3.0}   # KDIGO ratio threshold behind each endpoint label


def auroc(y, s):
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() == 0 or y[m].min() == y[m].max():
        return np.nan, 0
    return float(roc_auc_score(y[m], s[m])), int(m.sum())


def share(model_auc, single_auc):
    return (single_auc - 0.5) / (model_auc - 0.5)


print('=' * 82)
print('AUDIT 9  ratio dominance on BOTH endpoints (any-AKI ge1 vs severe-AKI ge3)')
print('=' * 82)
res = {}

for nm, cfg in GRID.items():
    print()
    print('=' * 82)
    print(nm)
    print('=' * 82)
    f = pd.read_parquet(cfg['feat'], columns=['stay_id', 't_hr', 'cr_ratio_base'])
    res[nm] = {}
    for ep in (1, 3):
        p = pd.read_parquet(cfg['ge%d' % ep])
        d = p.merge(f, on=['stay_id', 't_hr'], how='left')
        r = pd.to_numeric(d['cr_ratio_base'], errors='coerce')
        ok = r.notna()

        auc_all, _ = auroc(d['y'], d['p'])
        auc_m, n_m = auroc(d.loc[ok, 'y'], d.loc[ok, 'p'])
        auc_r, _ = auroc(d.loc[ok, 'y'], r[ok])
        sh = share(auc_m, auc_r)
        pos_r = float(r[d['y'] == 1].mean())
        neg_r = float(r[d['y'] == 0].mean())
        thr = STAGE_THR[ep]
        n_pos = int(d['y'].sum())
        n_def = int(((d['y'] == 1) & ok & (r >= thr)).sum())

        print()
        print('  --- ge%d : %s ---' % (ep, 'any AKI (>=stage 1)' if ep == 1 else 'SEVERE AKI (>=stage 3)'))
        print('    positives %d | rows %d | ratio missing %d (%.2f%%)'
              % (n_pos, len(d), int(r.isna().sum()), 100 * r.isna().mean()))
        print('    model AUROC, all rows      : %.4f' % auc_all)
        print('    model AUROC, matched n=%-6d: %.4f' % (n_m, auc_m))
        print('    ratio alone,  matched      : %.4f' % auc_r)
        print('    ratio share of excess      : %.1f%%' % (100 * sh))
        print('    ratio already >= %.1f at t : %d of %d positives = %.1f%% (definitional)'
              % (thr, n_def, n_pos, 100 * n_def / n_pos if n_pos else float('nan')))
        print('    mean ratio  pos %.4f | neg %.4f' % (pos_r, neg_r))

        res[nm]['ge%d' % ep] = {
            'n_pos': n_pos, 'n_rows': int(len(d)),
            'auc_model_all': round(auc_all, 6),
            'auc_model_matched': round(auc_m, 6),
            'auc_ratio_matched': round(auc_r, 6),
            'share_pct': round(100 * sh, 2),
            'definitional_pct': round(100 * n_def / n_pos, 2) if n_pos else None,
            'pos_mean_ratio': round(pos_r, 4), 'neg_mean_ratio': round(neg_r, 4),
        }

print()
print('=' * 82)
print('SUMMARY  ratio share of excess discrimination, both endpoints')
print('=' * 82)
for nm, v in res.items():
    for ep in ('ge1', 'ge3'):
        x = v[ep]
        print('  %-9s %s  model %.4f | ratio %.4f | share %5.1f%% | definitional %4.1f%% | pos %d'
              % (nm, ep, x['auc_model_matched'], x['auc_ratio_matched'],
                 x['share_pct'], x['definitional_pct'] or 0.0, x['n_pos']))

(OUT / 'a9_ratio_audit_both_endpoints.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
print()
print('  written ->', OUT / 'a9_ratio_audit_both_endpoints.json')
