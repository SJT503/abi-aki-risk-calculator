#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AUDIT 10 - why does cr_ratio_base top out at exactly 2.0000?

Read the frozen feature table only. No retraining.
Questions:
  Q1  what is the empirical distribution of cr_ratio_base, especially the tail
  Q2  for AKI-positive stays, does the ratio ever approach the label threshold
  Q3  is cr_last itself bounded, or is the ratio bounded
"""
import pandas as pd
import numpy as np

R = 'E:/TBI subtype/09_tbi_aki/results'
f = pd.read_parquet(R + '/n5_abi_rolling_features.parquet',
                    columns=['stay_id', 't_hr', 'cr_ratio_base', 'cr_base', 'cr_last',
                             'cr_delta_since_adm'])
r = pd.to_numeric(f['cr_ratio_base'], errors='coerce')

print('=' * 70)
print('Q1  cr_ratio_base distribution')
print('=' * 70)
print('count %d | min %.4f | max %.4f | mean %.4f | median %.4f'
      % (r.count(), r.min(), r.max(), r.mean(), r.median()))
for p in [50, 75, 90, 95, 99, 99.9, 99.99, 100]:
    print('  p%-6s %.4f' % (p, np.nanpercentile(r, p)))
print()
print('  rows >= 1.5 :', int((r >= 1.5).sum()), ' (%.6f%%)' % (100 * float((r >= 1.5).mean())))
print('  rows >= 1.8 :', int((r >= 1.8).sum()))
print('  rows == 2.0 :', int((r == 2.0).sum()))
print('  rows >  2.0 :', int((r > 2.0).sum()))

print()
print('=' * 70)
print('Q2  the extreme rows, in full')
print('=' * 70)
top = f.assign(_r=r).dropna(subset=['_r']).nlargest(10, '_r')
print(top[['stay_id', 't_hr', 'cr_base', 'cr_last', 'cr_ratio_base']].to_string(index=False))

print()
print('=' * 70)
print('Q3  is cr_last bounded, or the ratio?')
print('=' * 70)
print('cr_last : min %.3f  max %.3f  n=%d' % (f.cr_last.min(), f.cr_last.max(), f.cr_last.count()))
print('cr_base : min %.3f  max %.3f  n=%d' % (f.cr_base.min(), f.cr_base.max(), f.cr_base.count()))
ratio_from_parts = f['cr_last'] / f['cr_base'].clip(lower=0.1)
print('recomputed cr_last/max(cr_base,0.1): max %.4f' % ratio_from_parts.max())
print()
print('  -> if the two maxima differ, a cap sits on the stored ratio, not on its inputs')

print()
print('=' * 70)
print('Q4  per-stay: how high does the ratio actually go for AKI stays?')
print('=' * 70)
mx = f.groupby('stay_id')['cr_ratio_base'].max()
print('per-stay max ratio: min %.3f | median %.3f | p99 %.3f | max %.3f'
      % (mx.min(), mx.median(), np.nanpercentile(mx, 99), mx.max()))
print('stays whose max ratio >= 1.5 :', int((mx >= 1.5).sum()))
print('stays whose max ratio >= 2.0 :', int((mx >= 2.0).sum()))
