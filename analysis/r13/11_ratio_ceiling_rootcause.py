#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AUDIT 11 - is the cr_ratio_base ceiling real, and is cr_last the current value?

Observed: cr_ratio_base max == 2.0000 exactly, only 8/62,961 rows >= 1.5,
p75 == 1.0000 exactly. Per-stay max >= 1.5 for only 1 stay out of 7,442.
That is not what an AKI cohort should look like if cr_last is the current value.

Decisive checks:
  A  grid length / t_hr range, positive vs negative stays
  B  does the grid stop at AKI onset?
  C  for one high-ratio stay and one positive stay: raw cr series vs cr_base/cr_last
  D  how many checkpoints have cr_last == cr_base exactly
"""
import pandas as pd
import numpy as np

R = 'E:/TBI subtype/09_tbi_aki/results'
g = pd.read_parquet(R + '/n3_m4_abi_grid.parquet')
f = pd.read_parquet(R + '/n5_abi_rolling_features.parquet',
                    columns=['stay_id', 't_hr', 'cr_ratio_base', 'cr_base', 'cr_last', 'label'])
cr = pd.read_parquet(R + '/n2_m4_abi_cr_serial.parquet')

print('=' * 74)
print('A  grid structure')
print('=' * 74)
print('grid columns :', list(g.columns))
print('grid rows %d | stays %d' % (len(g), g.stay_id.nunique()))
per = g.groupby('stay_id')['t_hr'].agg(['min', 'max', 'count'])
print('checkpoints per stay : min %d | median %.1f | max %d'
      % (per['count'].min(), per['count'].median(), per['count'].max()))
print('t_hr max across grid : min %.0f | median %.0f | max %.0f'
      % (per['max'].min(), per['max'].median(), per['max'].max()))
print()
if 'onset_hr' in g.columns:
    on = g.groupby('stay_id')['onset_hr'].first().dropna()
    print('stays with a defined onset_hr :', len(on))
    print('onset_hr : min %.0f | median %.0f | max %.0f' % (on.min(), on.median(), on.max()))
    last = g.groupby('stay_id')['t_hr'].max()
    cmp = pd.DataFrame({'last_ckpt': last, 'onset': on}).dropna()
    print('  of those, last checkpoint <= onset : %d / %d'
          % (int((cmp.last_ckpt <= cmp.onset).sum()), len(cmp)))
    print('  last checkpoint >  onset           : %d'
          % int((cmp.last_ckpt > cmp.onset).sum()))

print()
print('=' * 74)
print('B  do positives appear AFTER onset? (tests a grid-ending rule)')
print('=' * 74)
pos = g[g.label == 1]
print('positive checkpoints: %d across %d stays' % (len(pos), pos.stay_id.nunique()))
if 'onset_hr' in g.columns:
    p2 = pos.dropna(subset=['onset_hr'])
    after = int((p2.t_hr > p2.onset_hr).sum())
    print('  positives with t_hr >  onset_hr : %d (%.1f%%)'
          % (after, 100 * after / len(p2) if len(p2) else 0))
    print('  positives with t_hr <= onset_hr : %d'
          % int((p2.t_hr <= p2.onset_hr).sum()))

print()
print('=' * 74)
print('C  raw creatinine vs cr_base/cr_last, two stays')
print('=' * 74)
for sid in [39518898, 30059997]:
    s = cr[cr.stay_id == sid].sort_values('hr')
    ff = f[f.stay_id == sid].sort_values('t_hr')
    if not len(s) or not len(ff):
        print('stay %s: raw rows %d | feature rows %d' % (sid, len(s), len(ff)))
        continue
    print('stay %s  cr_base=%.3f' % (sid, ff.cr_base.iloc[0]))
    print('  raw cr series (hr, value): %s'
          % ', '.join('(%.0f, %.1f)' % (a, b) for a, b in zip(s.hr, s.valuenum))[:200])
    print('  features (t_hr, cr_last, ratio): %s'
          % ', '.join('(%.0f, %.2f, %.3f)' % (a, b, c)
                      for a, b, c in zip(ff.t_hr, ff.cr_last, ff.cr_ratio_base))[:220])
    print()

print('=' * 74)
print('D  how often is cr_last exactly cr_base?')
print('=' * 74)
d = f.dropna(subset=['cr_last', 'cr_base'])
eq = np.isclose(d.cr_last, d.cr_base)
print('rows with cr_last == cr_base : %d / %d = %.2f%%' % (int(eq.sum()), len(d), 100 * eq.mean()))
print('ratio exactly 1.0            : %d = %.2f%%'
      % (int((np.isclose(d.cr_ratio_base, 1.0)).sum()),
         100 * float(np.isclose(d.cr_ratio_base, 1.0).mean())))
