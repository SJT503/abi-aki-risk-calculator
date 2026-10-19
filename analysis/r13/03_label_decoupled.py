#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ANALYSIS 2b - Label-decoupled evaluation (T3, corrected).

The first attempt used the earliest POSITIVE checkpoint of a stay as the anchor.
That is wrong: for a positive checkpoint the anchor equals the checkpoint itself,
so `anchor - t_hr >= lead` is never satisfied and every positive was dropped
(0 positives kept). This version uses the stay's true AKI onset hour, which the
frozen grid files carry as `onset_hr`.

Restriction: keep all negatives, and keep only positives at least `lead` hours
BEFORE the stay's onset. That removes the checkpoints nearest to the label
boundary, which are the ones most likely to be autocorrelated with it.

Inputs (frozen):
  MIMIC-IV grid : results/n3_m4_abi_grid.parquet      (stay_id, t_hr, label, onset_hr)
  eICU-CRD grid : results/n8_3_eicu_abi_grid.parquet  (stay_id, t_hr, label, onset_hr)
  MIMIC-IV preds: results/jinhu/f12_preds_M4_ge1.parquet
  eICU-CRD preds: results/jinhu/f12_preds_eICU_ge1.parquet
  MIMIC-IV feats: results/n5_abi_rolling_features.parquet   (cr_ratio_base)
  eICU-CRD feats: results/p1_expanded_features.parquet      (cr_ratio_base)
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
OUT.mkdir(parents=True, exist_ok=True)

COHORTS = {
    'MIMIC-IV': {
        'grid':  ROOT / 'results/n3_m4_abi_grid.parquet',
        'pred':  ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet',
        'feat':  ROOT / 'results/n5_abi_rolling_features.parquet',
    },
    'eICU-CRD': {
        'grid':  ROOT / 'results/n8_3_eicu_abi_grid.parquet',
        'pred':  ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet',
        'feat':  ROOT / 'results/p1_expanded_features.parquet',
    },
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def auroc(y, s):
    """AUROC on complete pairs; NaN scores are dropped pairwise."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s))


def run(name, cfg):
    print()
    print('=' * 68)
    print(name)
    print('=' * 68)
    g = pd.read_parquet(cfg['grid'])
    p = pd.read_parquet(cfg['pred'])
    f = pd.read_parquet(cfg['feat'])
    for lbl, path in [('grid', cfg['grid']), ('pred', cfg['pred']), ('feat', cfg['feat'])]:
        print('  %-5s %8d rows  sha %s  %s' % (lbl, len(pd.read_parquet(path, columns=None)
              if False else pd.read_parquet(path)), sha256(path)[:12], Path(path).name))

    d = g[['stay_id', 't_hr', 'label', 'onset_hr']].merge(
        p[['stay_id', 't_hr', 'y', 'p']], on=['stay_id', 't_hr'], how='inner')
    d = d.merge(f[['stay_id', 't_hr', 'cr_ratio_base']], on=['stay_id', 't_hr'], how='left')
    print('  merged %d rows | %d stays | positives %d' % (len(d), d['stay_id'].nunique(), int(d['y'].sum())))

    # sanity: label from grid must equal y from preds (same definition)
    agree = float((d['label'] == d['y']).mean())
    print('  grid label == pred y : %.4f' % agree)

    base = auroc(d['y'], d['p'])
    base_r = auroc(d['y'], pd.to_numeric(d['cr_ratio_base'], errors='coerce'))
    print()
    print('  base (all positives)                model AUROC %.4f | ratio alone %.4f  (n+=%d)'
          % (base, base_r, int(d['y'].sum())))

    out = {'n_rows': int(len(d)), 'n_stays': int(d['stay_id'].nunique()),
           'base_model_auroc': round(base, 6), 'base_ratio_auroc': round(base_r, 6),
           'base_pos': int(d['y'].sum()), 'label_pred_agreement': round(agree, 6)}

    for lead in (12, 24, 48):
        keep_pos = (d['y'] == 1) & d['onset_hr'].notna() & ((d['onset_hr'] - d['t_hr']) >= lead)
        keep = (d['y'] == 0) | keep_pos
        sub = d[keep]
        if sub['y'].sum() == 0:
            print('  lead %2d h: no positives survive' % lead)
            out['lead_%dh' % lead] = None
            continue
        a = auroc(sub['y'], sub['p'])
        ar = auroc(sub['y'], pd.to_numeric(sub['cr_ratio_base'], errors='coerce'))
        npos = int(sub['y'].sum())
        out['lead_%dh' % lead] = {'model_auroc': round(a, 6),
                                  'ratio_auroc': round(ar, 6) if np.isfinite(ar) else None,
                                  'n_pos': npos, 'n_rows': int(len(sub))}
        print('  positives >= %2d h before onset       model AUROC %.4f | ratio alone %.4f  (n+=%d, rows=%d)'
              % (lead, a, ar, npos, len(sub)))
    return out


def mini_task():
    res = {}
    for nm, cfg in COHORTS.items():
        res[nm] = run(nm, cfg)

    print()
    print('=' * 68)
    print('READING')
    print('=' * 68)
    for nm, r in res.items():
        b = r['base_model_auroc']
        br = r['base_ratio_auroc']
        print('  %-9s model %.4f vs ratio-alone %.4f  -> ratio explains %.0f%% of the model\'s '
              'discrimination margin over chance' % (nm, b, br, 100 * max(0, (br - 0.5) / (b - 0.5))))
        for lead in (12, 24, 48):
            k = r.get('lead_%dh' % lead)
            if k:
                print('      >= %2d h before onset: %.4f (drop %.4f vs base)' % (lead, k['model_auroc'],
                                                                               b - k['model_auroc']))

    (OUT / 'a2b_label_decoupled.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print()
    print('  written ->', OUT / 'a2b_label_decoupled.json')
    return res


if __name__ == '__main__':
    print('=' * 68)
    print('ANALYSIS 2b: label-decoupled evaluation (onset-anchored)')
    print('=' * 68)
    mini_task()
