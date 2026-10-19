#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ANALYSIS 6 - Disclosures the reviewer asked for.

D1  RRT-censored stays: how many stays were removed from the grid because they
    started renal replacement therapy between 24 and 168 h. Those are, by
    definition, the sickest tail of the severe endpoint, so the severe-AKI
    estimate is computed on a population from which they were removed.
    The bias runs UPWARD (the hardest events are gone), the opposite direction
    from the manuscript's disclosed misclassification bias.

D2  Checkpoints with no evaluable creatinine in the label window, i.e. the
    magnitude of the "treated as event-free" convention whose direction the
    manuscript already discloses but never quantifies.

Inputs: raw JinhuaNSICU RRT table if present; otherwise the frozen grids and
the raw databases for MIMIC-IV and eICU.
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import sys, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('E:/TBI subtype/09_tbi_aki')
DATA = Path('E:/TBI subtype/data')
OUT = ROOT / '_analysis_r13' / 'out'
OUT.mkdir(parents=True, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


print('=' * 72)
print('ANALYSIS 6: RRT censoring and unmeasured-creatinine disclosure')
print('=' * 72)

# ---------------------------------------------------------------- D1: MIMIC-IV
# The manuscript's rule: RRT between 24 and 168 h ends the grid.
print()
print('--- D1  MIMIC-IV: RRT-censored stays ---')
try:
    pe = pd.read_csv(DATA / 'mimic-iv-3.1/icu/procedureevents.csv.gz',
                     usecols=['stay_id', 'starttime', 'itemid'])
    di = pd.read_csv(DATA / 'mimic-iv-3.1/icu/d_items.csv.gz',
                     usecols=['itemid', 'label'])
    print('  procedureevents %d rows | d_items %d' % (len(pe), len(di)))

    # find dialysis-like itemids by label
    lab = di.set_index('itemid')['label'].astype(str)
    dl = lab[lab.str.contains('dialy|CRRT|renal replacement|hemofiltration',
                              case=False, na=False)]
    print('  dialysis-related itemids found: %d' % len(dl))
    for i, v in dl.head(12).items():
        print('     %-8s %s' % (i, v))

    ours = pd.read_parquet(ROOT / 'results/n3_m4_abi_grid.parquet',
                           columns=['stay_id'])['stay_id'].unique()
    pe_ours = pe[pe['stay_id'].isin(ours) & pe['itemid'].isin(dl.index)].copy()
    print('  RRT procedures among our stays: %d rows, %d unique stays'
          % (len(pe_ours), pe_ours['stay_id'].nunique() if len(pe_ours) else 0))

    if len(pe_ours):
        icu = pd.read_csv(DATA / 'mimic-iv-3.1/icu/icustays.csv.gz',
                          usecols=['stay_id', 'intime'])
        pe_ours = pe_ours.merge(icu, on='stay_id', how='left')
        pe_ours['hr'] = (pd.to_datetime(pe_ours['starttime'])
                         - pd.to_datetime(pe_ours['intime'])).dt.total_seconds() / 3600
        in_win = pe_ours[(pe_ours['hr'] >= 24) & (pe_ours['hr'] <= 168)]
        print('  of those, starting in the 24-168 h window (grid-ending): %d stays'
              % in_win['stay_id'].nunique())
        print('  (these are removed from the evaluated grid)')
except Exception as e:
    print('  ERR:', e)

# ---------------------------------------------------------------- D2: all cohorts
print()
print('--- D2  checkpoints with no evaluable creatinine in the label window ---')
for nm, fp, gp in [
    ('MIMIC-IV', ROOT / 'results/n5_abi_rolling_features.parquet',
     ROOT / 'results/n3_m4_abi_grid.parquet'),
    ('eICU-CRD', ROOT / 'results/p1_expanded_features.parquet',
     ROOT / 'results/n8_3_eicu_abi_grid.parquet'),
]:
    f = pd.read_parquet(fp, columns=['stay_id', 't_hr', 'cr_last'])
    g = pd.read_parquet(gp, columns=['stay_id', 't_hr', 'label', 'onset_hr'])
    d = g.merge(f, on=['stay_id', 't_hr'], how='left')
    miss_cr = d['cr_last'].isna()
    n = len(d)
    print('  %-9s checkpoints %6d | cr_last missing %5d (%.2f%%) | of which labelled positive: %d'
          % (nm, n, int(miss_cr.sum()), 100 * miss_cr.mean(), int((miss_cr & (d['label'] == 1)).sum())))
    # stays whose entire grid has no cr
    per = d.groupby('stay_id')['cr_last'].apply(lambda s: s.isna().all())
    print('             stays whose whole grid lacks cr_last: %d' % int(per.sum()))

# ---------------------------------------------------------------- Jinhua RRT
print()
print('--- D1b  JinhuaNSICU RRT (development cohort) ---')
cand = list(ROOT.glob('results/jinhu/jinhu_rrt.parquet')) + \
       list(ROOT.glob('**/jinhu_rrt.parquet'))
if cand:
    r = pd.read_parquet(cand[0])
    print('  file:', cand[0].name, '| rows', len(r), '| cols', list(r.columns))
    print(r.head(3).to_string())
else:
    print('  jinhu_rrt.parquet not found under results/')

print()
print('  NOTE: figures above are counts of the frozen/raw inputs, not new model output.')
