# -*- coding: utf-8 -*-
"""f2b: audit fix-ups for f2_full_results.json (2026-09-30 exhaustive gap-fill).
(1) EPPP recomputed on the DEVELOPMENT sample (= subject-split training set), with the
    full-grid values retained as secondary; round-8 convention = development sample.
(2) MK exact permutation p (720 enumerations over 6 strata), matching round-8 mk_exact.
(3) DeLong p strings '0.00e+00' -> floor notation.
(4) fa_autopsy near-criteria definition recorded in JSON.
"""
import json, itertools
import numpy as np, pandas as pd
import pyarrow.parquet as pq

R = "results"
d = json.load(open(f"{R}/f2_full_results.json"))

# ---- (1) EPPP on development (training) sample ----
m4cols = pq.ParquetFile(f"{R}/n5_abi_rolling_features.parquet").schema.names
need = ['stay_id', 't_hr', 'cr_base'] + [c for c in ('age', 'male', 'charlson') if c in m4cols]
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet", columns=need)
labs = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet").rename(columns={'valuenum': 'cr_val'})
labs = labs[labs['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
labs['t_hr'] = (labs['hr'] // 6) * 6
cb = labs.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index()
mg = m4[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
mg['f'] = np.nan
for s in range(0, 9):
    sh = mg.groupby('stay_id')['cr_val'].shift(-s)
    mg['f'] = pd.DataFrame({'c': mg['f'], 'n': sh}).max(axis=1)
fold = mg['f'] / mg['cr_base'].clip(lower=0.1)
rise = mg['f'] - mg['cr_base']
aki = (fold >= 1.5) | (rise >= 0.3)
st = pd.Series(0, index=mg.index)
st[aki.fillna(False)] = 1
st[fold >= 2.0] = 2
st[(fold >= 3.0) | ((mg['f'] >= 4.0) & aki)] = 3
st[mg['f'].isna()] = 0
mg['ge1'] = (st >= 1).astype(int)

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in mg['stay_id'].unique()}))
np.random.seed(42)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
trmask = mg['stay_id'].map(s2sub).isin(train_subj).values
dev_pos = int(mg['ge1'].values[trmask].sum())
dev_stays = int(mg.loc[trmask & (mg['ge1'] == 1), 'stay_id'].nunique())
P = 268
d['eppp_268'] = {
    'params': P,
    'dev_sample': 'subject-split TRAINING set (80% subjects); round-8 development-sample convention',
    'ckpt_level_dev': round(dev_pos / P, 1), 'patient_level_dev': round(dev_stays / P, 1),
    'dev_pos_checkpoints': dev_pos, 'dev_event_stays': dev_stays,
    'ckpt_level_fullgrid': round(int(mg['ge1'].sum()) / P, 1),
    'patient_level_fullgrid': round(int(mg.loc[mg['ge1'] == 1, 'stay_id'].nunique()) / P, 1),
    'fullgrid_note': 'full-grid values retained for reference only; primary = dev sample'}
print(f"EPPP dev: {dev_pos}/{P} = {dev_pos/P:.1f} ckpt | {dev_stays}/{P} = {dev_stays/P:.1f} patient")

# ---- (2) MK exact permutation test (round-8 mk_exact convention: S = concordant - discordant pairs) ----
strata = d['mk']['strata']
def S_stat(v):
    c = sum(1 for i in range(len(v)) for j in range(i + 1, len(v)) if v[j] > v[i])
    dd = sum(1 for i in range(len(v)) for j in range(i + 1, len(v)) if v[j] < v[i])
    return c - dd
Sobs = S_stat(strata)
n = len(strata)
cnt = 0
total = 0
for perm_ in itertools.permutations(range(n)):
    S = S_stat([strata[i] for i in perm_])
    total += 1
    if S >= Sobs:
        cnt += 1
p_exact = 2 * cnt / total
d['mk']['exact_p_two_sided'] = round(float(min(p_exact, 1.0)), 4)
d['mk']['S_obs'] = int(Sobs)
d['mk']['tau_from_S'] = round(Sobs / (n * (n - 1) // 2), 3)
d['mk']['note'] = 'asymptotic kendalltau unreliable at n=6 strata; exact permutation over 720 orderings, S=concordant-discordant pairs (round-8 mk_exact convention)'
print(f"MK: S={Sobs} tau={Sobs/15:.3f} exact two-sided p={min(p_exact,1.0):.4f} (asymptotic was {d['mk']['p']})")

# ---- (3) DeLong p floor ----
for k in ('int', 'ext'):
    if d['delong'][k]['p'] in ('0.00e+00', '0.0', 0.0):
        d['delong'][k]['p'] = '<1e-300'
        d['delong'][k]['p_note'] = 'z exceeds double precision; reported as floor'

# ---- (4) fa_autopsy definition ----
d['sm5_arms']['fa_autopsy']['near_criteria_def'] = (
    'never-event stay whose ANY measured creatinine bin reaches fold>=1.3 or rise>=0.2 vs baseline '
    '(i.e. within 20-27% of the AKI threshold); lead>48h fraction among captured event stays at thr 0.15')

json.dump(d, open(f"{R}/f2_full_results.json", "w"), indent=2)
print("patched f2_full_results.json")
