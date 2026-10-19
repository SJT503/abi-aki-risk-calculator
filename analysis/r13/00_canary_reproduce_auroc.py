#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CANARY (1% workload): reproduce the frozen MIMIC-IV any-AKI AUROC = 0.769
from the frozen prediction file only.

Passing this canary is the precondition for the rest of the analysis round:
it proves the prediction file, the label column and the AUROC definition are
aligned with the frozen manuscript before any new quantity is computed.

If this does not reproduce 0.769, STOP - do not run downstream analyses.
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

PRED = ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mini_task():
    df = pd.read_parquet(PRED)
    print('  file sha256 :', sha256(PRED)[:16])
    print('  rows        : %d' % len(df))
    print('  cols        :', list(df.columns))
    print('  unique stays: %d' % df['stay_id'].nunique())
    print('  positives   : %d checkpoints (%.2f%%)' % (df['y'].sum(), 100 * df['y'].mean()))

    point = roc_auc_score(df['y'], df['p'])
    print()
    print('  checkpoint-level AUROC (p)      : %.4f' % point)
    print('  expected (manuscript)           : 0.769')
    print('  |diff|                          : %.4f' % abs(point - 0.769))

    # also report the recalibrated column, if present
    extra = {}
    for col in ('p_rec', 'p_intercept_updated'):
        if col in df.columns:
            v = roc_auc_score(df['y'], df[col])
            extra[col] = round(float(v), 6)
            print('  AUROC (%s)%s: %.4f' % (col, ' ' * max(0, 12 - len(col)), v))

    res = {
        'file': str(PRED),
        'sha256': sha256(PRED),
        'rows': int(len(df)),
        'stays': int(df['stay_id'].nunique()),
        'pos_checkpoints': int(df['y'].sum()),
        'auroc_p': round(float(point), 6),
        'auroc_extra': extra,
        'target': 0.769,
        'abs_diff': round(abs(float(point) - 0.769), 6),
        'pass': abs(float(point) - 0.769) < 0.0005,
    }
    (OUT / 'canary_m4_auroc.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    return res


if __name__ == '__main__':
    print('=' * 66)
    print('CANARY: reproduce frozen MIMIC-IV any-AKI AUROC')
    print('=' * 66)
    res = mini_task()
    print()
    print('  RESULT: %s' % ('PASS - canary reproduces the frozen value' if res['pass']
                            else '** FAIL - do not run downstream analyses **'))
    print('  written ->', OUT / 'canary_m4_auroc.json')
    sys.exit(0 if res['pass'] else 1)
