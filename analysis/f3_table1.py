# -*- coding: utf-8 -*-
"""f3: Table 1 with the SUBJECT-LEVEL 80/20 random split (round-8 audit item A4),
on the 268-feature basis. Training vs internal-test (M4, same subjects never cross)
vs external (eICU). Demographics + checkpoints + events per endpoint.
Output: results/f3_table1.json (+ printed markdown-ready rows)
"""
import json
import numpy as np, pandas as pd

R = "results"
SEED = 42

m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet", columns=['stay_id', 't_hr', 'cr_base', 'age', 'male', 'charlson'])
eicu = pd.read_parquet(f"{R}/p1_expanded_features.parquet", columns=['stay_id', 't_hr', 'cr_base', 'age', 'male', 'charlson'])
n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])

# labels (E02-corrected, verbatim machinery)
def labels_e02(labs_parquet, feat_df):
    labs = pd.read_parquet(labs_parquet)
    if 'sid' in labs.columns:
        cr = labs[labs['labname'] == 'creatinine'].rename(columns={'sid': 'stay_id', 'labresult': 'cr_val'})
    else:
        cr = labs.rename(columns={'valuenum': 'cr_val'})
    cr = cr[cr['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
    cr['t_hr'] = (cr['hr'] // 6) * 6
    cb = cr.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index().sort_values(['stay_id', 't_hr'])
    mg = feat_df[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
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
    for t in [1, 2, 3]:
        mg[f'ge{t}'] = (st >= t).astype(int)
    return mg[['stay_id', 't_hr', 'ge1', 'ge2', 'ge3']]

m4lab = labels_e02(f"{R}/n2_m4_abi_cr_serial.parquet", m4)
eilab = labels_e02(f"{R}/n8_2_labs_series.parquet", eicu)
m4 = m4.merge(m4lab, on=['stay_id', 't_hr'], how='left')
eicu = eicu.merge(eilab, on=['stay_id', 't_hr'], how='left')

s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4['stay_id'].unique()}))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
m4['split'] = np.where(m4['stay_id'].map(s2sub).isin(train_subj), 'train', 'int_test')
eicu['split'] = 'external'

stay_m4 = m4.drop_duplicates('stay_id')
stay_ei = eicu.drop_duplicates('stay_id')

def summ(df_stays, df_ckpt, name):
    ev = {}
    for t in [1, 2, 3]:
        pos = df_ckpt[df_ckpt[f'ge{t}'] == 1]
        ev[f'ge{t}_pos_ckpt'] = int(len(pos))
        ev[f'ge{t}_event_stays'] = int(pos['stay_id'].nunique())
    return {
        'cohort': name,
        'stays': int(df_stays['stay_id'].nunique()),
        'subjects': int(df_stays['stay_id'].map(s2sub).nunique()) if name != 'external' else None,
        'checkpoints': int(len(df_ckpt)),
        'age_median_iqr': [round(float(df_stays['age'].median()), 1),
                           round(float(df_stays['age'].quantile(.25)), 1), round(float(df_stays['age'].quantile(.75)), 1)],
        'male_pct': round(float(df_stays['male'].mean() * 100), 1),
        'charlson_median_iqr': [round(float(df_stays['charlson'].median()), 1),
                                round(float(df_stays['charlson'].quantile(.25)), 1), round(float(df_stays['charlson'].quantile(.75)), 1)],
        'cr_base_median_iqr': [round(float(df_stays['cr_base'].median()), 2),
                               round(float(df_stays['cr_base'].quantile(.25)), 2), round(float(df_stays['cr_base'].quantile(.75)), 2)],
        **ev}

rows = [
    summ(stay_m4[stay_m4['split'] == 'train'], m4[m4['split'] == 'train'], 'M4 training (80% subjects)'),
    summ(stay_m4[stay_m4['split'] == 'int_test'], m4[m4['split'] == 'int_test'], 'M4 internal test (20% subjects)'),
    summ(stay_ei, eicu, 'eICU external'),
]
# split integrity
tr_sub = set(n1[n1['stay_id'].isin(m4[m4['split'] == 'train']['stay_id'])]['subject_id'])
te_sub = set(n1[n1['stay_id'].isin(m4[m4['split'] == 'int_test']['stay_id'])]['subject_id'])
overlap = tr_sub & te_sub
print(f"subject overlap train∩test: {len(overlap)} (must be 0)")
assert len(overlap) == 0

json.dump({'_meta': {'split': 'subject-level 80/20 seed 42, zero-overlap asserted',
                     'date': '2026-09-30', 'labels': 'E02-corrected, window [t, t+54h)'},
           'rows': rows}, open(f"{R}/f3_table1.json", "w"), indent=2)

hdr = ['cohort', 'stays', 'subjects', 'checkpoints', 'age', 'male%', 'charlson', 'cr_base',
       'ge1+ckpt', 'ge1 stays', 'ge2+ckpt', 'ge2 stays', 'ge3+ckpt', 'ge3 stays']
print('\t'.join(hdr))
for r in rows:
    print('\t'.join(str(x) for x in [
        r['cohort'], r['stays'], r['subjects'], r['checkpoints'],
        f"{r['age_median_iqr'][0]} ({r['age_median_iqr'][1]}-{r['age_median_iqr'][2]})",
        r['male_pct'], r['charlson_median_iqr'][0],
        f"{r['cr_base_median_iqr'][0]} ({r['cr_base_median_iqr'][1]}-{r['cr_base_median_iqr'][2]})",
        r['ge1_pos_ckpt'], r['ge1_event_stays'], r['ge2_pos_ckpt'], r['ge2_event_stays'],
        r['ge3_pos_ckpt'], r['ge3_event_stays']]))
print("\nSaved f3_table1.json")
