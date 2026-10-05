# -*- coding: utf-8 -*-
"""s1_fixed: COMPLETE re-run with corrected labels.
E01 fix: range(1,9) — exclude current bin, strict future window
E02 fix: Stage 3 absolute ≥4.0 requires AKI precondition (fold≥1.5 or rise≥0.3)
Everything else identical to s1_stage_primary.py protocol."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

R = "results"
SEED = 42

FEATS = json.load(open(f"{R}/_frozen291_features.json"))
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=SEED, verbose=-1)

m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[['stay_id'] + FEATS]
eicu = pd.read_parquet(f"{R}/p1_expanded_features.parquet")[['stay_id'] + FEATS]
print(f"M4 {m4.shape}, eICU {eicu.shape}")


def fixed_labels(labs_parquet, feat_df, db):
    """Corrected labels: range(1,9) + AKI precondition for Stage 3 absolute branch."""
    labs = pd.read_parquet(labs_parquet)
    if 'sid' in labs.columns:
        cr = labs[labs['labname'] == 'creatinine'].rename(
            columns={'sid': 'stay_id', 'labresult': 'cr_val'})
    else:
        cr = labs.rename(columns={'valuenum': 'cr_val'})
    cr = cr[cr['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
    cr['t_hr'] = (cr['hr'] // 6) * 6
    cb = cr.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index()
    cb = cb.sort_values(['stay_id', 't_hr'])
    mg = feat_df[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(
        cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])

    # E01 FIX: range(1,9) — exclude current bin, strict future
    mg['f'] = np.nan
    for s in range(1, 9):
        sh = mg.groupby('stay_id')['cr_val'].shift(-s)
        mg['f'] = pd.DataFrame({'c': mg['f'], 'n': sh}).max(axis=1)

    fold = mg['f'] / mg['cr_base'].clip(lower=0.1)
    rise = mg['f'] - mg['cr_base']

    # AKI definition (acute change)
    aki = (fold >= 1.5) | (rise >= 0.3)

    st = pd.Series(0, index=mg.index)
    # Stage 1: AKI definition
    st[aki.fillna(False)] = 1
    # Stage 2: fold >= 2.0 (implies AKI since fold≥2>1.5)
    st[fold >= 2.0] = 2
    # E02 FIX: Stage 3 = fold≥3 OR (f≥4.0 AND AKI)
    st[(fold >= 3.0) | ((mg['f'] >= 4.0) & aki)] = 3
    st[mg['f'].isna()] = 0

    for t in [1, 2, 3]:
        mg[f'ge{t}'] = (st >= t).astype(int)

    n1, n2, n3 = int(mg['ge1'].sum()), int(mg['ge2'].sum()), int(mg['ge3'].sum())
    s1, s2, s3 = int(mg[mg.ge1==1]['stay_id'].nunique()), int(mg[mg.ge2==1]['stay_id'].nunique()), int(mg[mg.ge3==1]['stay_id'].nunique())
    print(f"  {db}: ge1={n1}({s1}st) ge2={n2}({s2}st) ge3={n3}({s3}st)")
    return mg[['stay_id', 't_hr', 'ge1', 'ge2', 'ge3']]


m4_lab = fixed_labels(f"{R}/n2_m4_abi_cr_serial.parquet", m4, "M4")
ei_lab = fixed_labels(f"{R}/n8_2_labs_series.parquet", eicu, "eICU")

m4f = m4.merge(m4_lab, on=['stay_id', 't_hr'], how='left')
eif = eicu.merge(ei_lab, on=['stay_id', 't_hr'], how='left')

# Subject-level split (identical to original)
n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr
assert len(set(m4f.loc[tr, 'stay_id'].map(s2sub)) & set(m4f.loc[te, 'stay_id'].map(s2sub))) == 0
print(f"split: {tr.sum()}/{te.sum()} ckpts")

# Train and evaluate
print("\n=== FIXED RESULTS ===")
out_results = {}
for t in [1, 2, 3]:
    mod = lgb.LGBMClassifier(**PARAMS)
    mod.fit(m4f[FEATS].values[tr], m4f[f'ge{t}'].values[tr])
    pi = mod.predict_proba(m4f[FEATS].values[te])[:, 1]
    pe = mod.predict_proba(eif[FEATS].values)[:, 1]
    yi, ye = m4f[f'ge{t}'].values[te], eif[f'ge{t}'].values
    auc_i = round(float(roc_auc_score(yi, pi)), 4)
    auc_e = round(float(roc_auc_score(ye, pe)), 4)
    ap_i = round(float(average_precision_score(yi, pi)), 4)
    ap_e = round(float(average_precision_score(ye, pe)), 4)
    ev_i, ev_e = int(yi.sum()), int(ye.sum())
    es_i = int(m4f.loc[te & (m4f[f'ge{t}']==1), 'stay_id'].nunique())
    es_e = int(eif.loc[eif[f'ge{t}']==1, 'stay_id'].nunique())
    out_results[f'ge{t}'] = {
        'internal': auc_i, 'external': auc_e,
        'auprc_int': ap_i, 'auprc_ext': ap_e,
        'events_int': ev_i, 'events_ext': ev_e,
        'event_stays_int': es_i, 'event_stays_ext': es_e}
    print(f"  ge{t}: int={auc_i} ext={auc_e} | AUPRC int={ap_i} ext={ap_e} | "
          f"events int={ev_i} ext={ev_e} | event_stays int={es_i} ext={es_e}")

# Save predictions for CI computation
preds_int = {1: pd.DataFrame({'stay_id': m4f.loc[te, 'stay_id'].values, 'y': m4f.loc[te, 'ge1'].values,
                               'p': mod.predict_proba(m4f[FEATS].values[te])[:, 1], 't_hr': m4f.loc[te, 't_hr'].values})}

# Save all predictions
for t in [1, 2, 3]:
    mod = lgb.LGBMClassifier(**PARAMS)
    mod.fit(m4f[FEATS].values[tr], m4f[f'ge{t}'].values[tr])
    pi = mod.predict_proba(m4f[FEATS].values[te])[:, 1]
    pe = mod.predict_proba(eif[FEATS].values)[:, 1]
    pd.DataFrame({'stay_id': m4f.loc[te, 'stay_id'].values, 'y': m4f.loc[te, f'ge{t}'].values,
                  'p': pi, 't_hr': m4f.loc[te, 't_hr'].values}).to_parquet(
        f"{R}/s1f_preds_int_ge{t}.parquet", index=False)
    pd.DataFrame({'stay_id': eif['stay_id'].values, 'y': eif[f'ge{t}'].values,
                  'p': pe, 't_hr': eif['t_hr'].values}).to_parquet(
        f"{R}/s1f_preds_ext_ge{t}.parquet", index=False)

# Save results
out = {'_meta': {'protocol': 'E01 fix range(1,9) + E02 fix AKI precondition for Stage 3 absolute branch',
                  'date': '2026-09-29', 'fixes': 'E01+E02'},
        'results': out_results}
json.dump(out, open(f"{R}/s1f_fixed_results.json", "w"), indent=2)
print("\nSaved s1f_fixed_results.json + prediction parquets")
