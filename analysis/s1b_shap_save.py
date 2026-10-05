# -*- coding: utf-8 -*-
"""s1b: retrain the ge1 model (deterministic, same split) and save SHAP values
+ feature matrix for the supplementary beeswarm/dependence figure."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
import shap

R = "results"
FEATS = json.load(open(f"{R}/_frozen291_features.json"))
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=42, verbose=-1)
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[['stay_id'] + FEATS]
labs = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet")
cr = labs.rename(columns={'valuenum': 'cr_val'})
cr = cr[cr['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
cr['t_hr'] = (cr['hr'] // 6) * 6
cb = cr.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index().sort_values(['stay_id', 't_hr'])
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
mg['ge1'] = (st >= 1).astype(int)
m4f = m4.merge(mg[['stay_id', 't_hr', 'ge1']], on=['stay_id', 't_hr'], how='left')

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(42)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr

mod = lgb.LGBMClassifier(**PARAMS)
mod.fit(m4f[FEATS].values[tr], m4f['ge1'].values[tr])
ridx = np.random.default_rng(42).choice(np.where(te)[0], size=8000, replace=False)
Xs = m4f.iloc[ridx][FEATS]
expl = shap.TreeExplainer(mod)
sv = expl.shap_values(Xs.values, check_additivity=False)
if isinstance(sv, list):
    sv = sv[1]
out = pd.DataFrame(sv, columns=FEATS)
out['stay_id'] = m4f.iloc[ridx]['stay_id'].values
out['t_hr'] = m4f.iloc[ridx]['t_hr'].values
out.to_parquet(f"{R}/s1_shap_values.parquet", index=False)
Xs.assign(stay_id=m4f.iloc[ridx]['stay_id'].values).to_parquet(f"{R}/s1_shap_X.parquet", index=False)
imp = np.abs(sv).mean(0)
order = np.argsort(imp)[::-1][:15]
print("top10:", [FEATS[i] for i in order[:10]])
print("saved s1_shap_values/X parquets")
