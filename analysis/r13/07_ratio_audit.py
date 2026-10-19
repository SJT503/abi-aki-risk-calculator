#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AUDIT 7 - Interrogate the "cr_ratio_base alone = 99% of excess discrimination" claim.

PI challenge: which model produced this? The frozen one or a new one? Is it an error?

Four questions, answered from the frozen artifacts only (no retraining):
  A  Is cr_ratio_base actually one of the frozen model's 166 features?
     -> read f12_shap_X.parquet, the literal input matrix the frozen model scored.
  B  Reproduce the model AUROC on the exact same row set the univariate AUROC uses.
     The univariate route drops rows where cr_ratio_base is NaN; the model AUROC
     does not. If those row sets differ, the "99%" is comparing different populations.
  C  Recompute the excess-discrimination share both ways (matched vs unmatched rows).
  D  Direction check: confirm higher ratio means higher label, not the reverse.
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
        pred=ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet',
        feat=ROOT / 'results/n5_abi_rolling_features.parquet'),
    'eICU-CRD': dict(
        pred=ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet',
        feat=ROOT / 'results/p1_expanded_features.parquet'),
}


def auroc(y, s):
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() == 0 or y[m].min() == y[m].max():
        return np.nan, 0
    return float(roc_auc_score(y[m], s[m])), int(m.sum())


def share(model_auc, single_auc):
    """Fraction of excess-over-chance discrimination captured by the single feature."""
    return (single_auc - 0.5) / (model_auc - 0.5)


print('=' * 76)
print('AUDIT 7  the cr_ratio_base / model comparison, re-derived')
print('=' * 76)

result = {}

# ---------------------------------------------------------------- A: is it a model feature?
print()
print('A  Is cr_ratio_base one of the frozen model features?')
sx = pd.read_parquet(ROOT / 'results/jinhu/f12_shap_X.parquet')
print('   f12_shap_X.parquet  shape = %d rows x %d cols' % sx.shape)
has = 'cr_ratio_base' in sx.columns
print('   cr_ratio_base present in the frozen model INPUT matrix : %s' % has)
if has:
    print('   -> it is one of the %d features the frozen model scored. Same model, no retraining.' % sx.shape[1])
result['in_model_input'] = bool(has)
result['model_input_n_features'] = int(sx.shape[1])

# ---------------------------------------------------------------- B/C/D per cohort
for nm, cfg in COH.items():
    print()
    print('=' * 76)
    print(nm)
    print('=' * 76)
    p = pd.read_parquet(cfg['pred'])
    f = pd.read_parquet(cfg['feat'], columns=['stay_id', 't_hr', 'cr_ratio_base'])

    # the canonical model number, on ALL predicted rows
    auc_all, n_all = auroc(p['y'], p['p'])
    print()
    print('B  row sets')
    print('   frozen pred file            : %7d rows | %5d stays | %5d positives'
          % (len(p), p['stay_id'].nunique(), int(p['y'].sum())))
    print('   model AUROC on ALL rows     : %.4f  (n=%d)' % (auc_all, n_all))

    d = p.merge(f, on=['stay_id', 't_hr'], how='left')
    r = pd.to_numeric(d['cr_ratio_base'], errors='coerce')
    n_nan = int(r.isna().sum())
    print('   cr_ratio_base missing       : %7d rows (%.2f%%)' % (n_nan, 100 * n_nan / len(d)))

    ok = r.notna()
    auc_m_ok, n_ok = auroc(d.loc[ok, 'y'], d.loc[ok, 'p'])
    auc_r_ok, _ = auroc(d.loc[ok, 'y'], r[ok])
    print('   --> MATCHED subset (ratio observed) n=%d' % n_ok)
    print('        model AUROC on matched   : %.4f' % auc_m_ok)
    print('        ratio AUROC on matched   : %.4f' % auc_r_ok)

    auc_r_all, n_ro = auroc(d['y'], r)
    print('   --> ratio alone, NaN-dropped : %.4f  (n=%d)  <- the r13 report number' % (auc_r_all, n_ro))

    print()
    print('C  excess-discrimination share, computed both ways')
    s_matched = share(auc_m_ok, auc_r_ok)
    s_unmatched = share(auc_all, auc_r_all)
    print('   like-for-like (same rows)    : %.4f / %.4f  -> %.1f%%'
          % (auc_r_ok, auc_m_ok, 100 * s_matched))
    print('   as first reported            : %.4f / %.4f  -> %.1f%%'
          % (auc_r_all, auc_all, 100 * s_unmatched))
    print('   (the two differ by %.1f percentage points)' % (100 * abs(s_matched - s_unmatched)))

    # D: direction
    pos_r = float(r[d['y'] == 1].mean()); neg_r = float(r[d['y'] == 0].mean())
    print()
    print('D  direction')
    print('   mean cr_ratio_base  positives %.4f | negatives %.4f  -> higher=%s'
          % (pos_r, neg_r, 'positive (correct)' if pos_r > neg_r else 'NEGATIVE (sign flipped!)'))

    result[nm] = {
        'n_rows': int(len(p)), 'n_nan_ratio': n_nan,
        'auc_model_all_rows': round(auc_all, 6),
        'auc_model_matched': round(auc_m_ok, 6),
        'auc_ratio_matched': round(auc_r_ok, 6),
        'auc_ratio_nan_dropped': round(auc_r_all, 6),
        'share_matched': round(s_matched, 6),
        'share_as_reported': round(s_unmatched, 6),
        'pos_mean_ratio': round(pos_r, 4), 'neg_mean_ratio': round(neg_r, 4),
    }

print()
print('=' * 76)
print('SUMMARY')
print('=' * 76)
for nm, v in result.items():
    if not isinstance(v, dict):
        continue
    print('  %-9s model(all)=%.4f  model(matched)=%.4f  ratio=%.4f  share matched=%.1f%%  as-reported=%.1f%%'
          % (nm, v['auc_model_all_rows'], v['auc_model_matched'],
             v['auc_ratio_matched'], 100 * v['share_matched'], 100 * v['share_as_reported']))

(OUT / 'a7_ratio_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print()
print('  written ->', OUT / 'a7_ratio_audit.json')
