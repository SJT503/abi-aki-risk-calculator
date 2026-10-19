#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INDEPENDENT verification (read-only recomputation) of r16-1 LHO recalibration.

Own route: Newton-Raphson logistic fit (no sklearn for any fit), own
quantile/summary logic, recomputed straight from the two input parquets.
No file in out/ is touched; results printed to stdout only.
"""
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.stdout.reconfigure(encoding='utf-8')
R = 'E:/TBI subtype/09_tbi_aki'
A1, B1 = 0.005067, 0.268467


def fit_logistic(x, y, penalty=0.0, max_iter=300, tol=1e-11):
    """Unpenalized (or ridge-penalized, intercept exempt) logistic via
    Newton-Raphson with step halving. x: (n,p); returns (intercept, coefs)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, float)
    n, p = x.shape
    X = np.column_stack([np.ones(n), x])
    m = y.mean()
    beta = np.zeros(p + 1)
    if 0.0 < m < 1.0:
        beta[0] = np.log(m / (1.0 - m))
    lam = np.zeros(p + 1)
    lam[1:] = penalty

    def nll(b):
        e = X @ b
        return -np.sum(y * e - np.logaddexp(0.0, e)) + 0.5 * np.sum(lam * b * b)

    for _ in range(max_iter):
        e = X @ beta
        pr = expit(e)
        Wd = pr * (1.0 - pr)
        g = X.T @ (pr - y) + lam * beta
        H = (X * Wd[:, None]).T @ X + np.diag(lam)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        cur = nll(beta)
        t = 1.0
        while t > 1e-10 and nll(beta - t * step) > cur - 1e-14:
            t *= 0.5
        beta = beta - t * step
        if np.max(np.abs(g)) < tol * max(1.0, float(np.abs(beta).max())):
            break
    return beta[0], beta[1:]


def slope_of(y, p):
    """Calibration slope: logistic of y on logit(clip(p))."""
    x = logit(np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9))
    a, b = fit_logistic(x, y)
    return float(b[0])


pe = pd.read_parquet(R + '/results/jinhu/f12_preds_eICU_ge1.parquet')
hm = pd.read_parquet(R + '/FINAL_20260928/data_current/s1_hospital_map.parquet')
d = pe.merge(hm, on='stay_id', how='inner')
d['lg'] = logit(np.clip(d['p'], 1e-9, 1 - 1e-9))
d['p_tr'] = expit(A1 + B1 * d['lg'])
print('merged rows %d | hospitals %d | pos rate %.4f' %
      (len(d), d.hospitalid.nunique(), d['y'].mean()))

# ---- canary: transported slope over all of eICU ----
sl_all = slope_of(d['y'].values, d['p_tr'].values)
print('CANARY my all-eICU transported slope = %.4f  (claim 0.881)' % sl_all)

# ---- eligibility + leave-hospital-out ----
g = d.groupby('hospitalid')['y']
elig = [h for h, s in g.sum().items() if s >= 20]
print('hospitals with >=20 positive checkpoints: %d (claim 50)' % len(elig))

sl_tr, sl_lo = [], []
rows = []
for h in elig:
    te = d[d.hospitalid == h]
    tr = d[d.hospitalid != h]
    a_h, b_h = fit_logistic(tr['lg'].values, tr['y'].values)
    p_lo = expit(a_h + b_h[0] * te['lg'].values)
    a = slope_of(te['y'].values, te['p_tr'].values)
    b = slope_of(te['y'].values, p_lo)
    sl_tr.append(a)
    sl_lo.append(b)
    rows.append((h, len(te), int(te['y'].sum()), a, b, b_h[0]))

sl_tr = np.array(sl_tr)
sl_lo = np.array(sl_lo)
print()
print('MY transported : median %.3f  IQR %.3f-%.3f  (claim 0.811, 0.657-0.986)' %
      (np.median(sl_tr), np.quantile(sl_tr, .25), np.quantile(sl_tr, .75)))
print('MY local-refit : median %.3f  IQR %.3f-%.3f  (claim 0.919, 0.745-1.124)' %
      (np.median(sl_lo), np.quantile(sl_lo, .25), np.quantile(sl_lo, .75)))
in_tr = int(((sl_tr >= 0.8) & (sl_tr <= 1.25)).sum())
in_lo = int(((sl_lo >= 0.8) & (sl_lo <= 1.25)).sum())
print('MY in [0.8,1.25]: transported %d/%d (claim 23) | local %d/%d (claim 32)' %
      (in_tr, len(sl_tr), in_lo, len(sl_lo)))

# ---- per-hospital cross-check vs frozen csv ----
froz = pd.read_csv(R + '/_analysis_r16/out/lho_recalibration_per_hospital.csv')
mine = pd.DataFrame(rows, columns=['hospitalid', 'n', 'npos', 'my_tr', 'my_lo', 'my_b_h'])
chk = froz.merge(mine, on='hospitalid')
print()
print('per-hospital rows joined: %d (frozen %d, mine %d)' % (len(chk), len(froz), len(mine)))
print('max |my_tr - frozen slope_transported| = %.4f' %
      (chk.my_tr - chk.slope_transported).abs().max())
print('max |my_lo - frozen slope_local|       = %.4f' %
      (chk.my_lo - chk.slope_local).abs().max())
bad = chk[(chk.my_tr - chk.slope_transported).abs() > 0.02]
print('hospitals differing >0.02 in transported slope:', len(bad))
if len(bad):
    print(bad[['hospitalid', 'slope_transported', 'my_tr']].head(10).to_string())
