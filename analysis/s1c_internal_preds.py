# -*- coding: utf-8 -*-
"""s1c: retrain ge2/ge3 (identical protocol/split) and save internal-test
prediction parquets so Fig2 can draw internal curves for all thresholds."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb

R = "results"
FEATS = json.load(open(f"{R}/_frozen291_features.json"))
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=42, verbose=-1)
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[['stay_id'] + FEATS]
labs = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet").rename(columns={'valuenum': 'cr_val'})
labs = labs[labs['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
labs['t_hr'] = (labs['hr'] // 6) * 6
cb = labs.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index().sort_values(['stay_id', 't_hr'])
mg = m4[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
mg['f'] = np.nan
for s in range(9):
    sh = mg.groupby('stay_id')['cr_val'].shift(-s)
    mg['f'] = pd.DataFrame({'c': mg['f'], 'n': sh}).max(axis=1)
b = mg['cr_base'].clip(lower=0.1); fold = mg['f'] / b
st = pd.Series(0, index=mg.index)
st[(fold >= 1.5) | ((mg['f'] - mg['cr_base']) >= 0.3)] = 1
st[fold >= 2.0] = 2
st[(fold >= 3.0) | (mg['f'] >= 4.0)] = 3
st[mg['f'].isna()] = 0
for t in [1, 2, 3]:
    mg[f'ge{t}'] = (st >= t).astype(int)
m4f = m4.merge(mg[['stay_id', 't_hr', 'ge1', 'ge2', 'ge3']], on=['stay_id', 't_hr'], how='left')

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(42)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr

from sklearn.metrics import roc_auc_score
for t in [2, 3]:
    mod = lgb.LGBMClassifier(**PARAMS)
    mod.fit(m4f[FEATS].values[tr], m4f[f'ge{t}'].values[tr])
    p = mod.predict_proba(m4f[FEATS].values[te])[:, 1]
    df = pd.DataFrame({'stay_id': m4f.loc[te, 'stay_id'].values, 'y': m4f.loc[te, f'ge{t}'].values,
                       'p': p, 't_hr': m4f.loc[te, 't_hr'].values})
    df.to_parquet(f"{R}/s1_preds_int_ge{t}.parquet", index=False)
    print(f"ge{t} internal-test AUROC = {roc_auc_score(df.y, df.p):.4f} (saved)")
# rename ge1 internal for uniform naming
import shutil
shutil.copy(f"{R}/s1_preds_internal.parquet", f"{R}/s1_preds_int_ge1.parquet")
print("done")
