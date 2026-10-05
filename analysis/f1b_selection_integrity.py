# -*- coding: utf-8 -*-
"""f1b: Selection integrity checks for the 268-feature re-selection (gap-fill, 2026-09-30).
(1) Canonical recompute of the seed-42 5-fold ge1 CV gain importance with the full
    cumulative curve saved under correct labels (fixes f1's dict-key collision:
    int(0.995*100)==99 overwrote the 99% entry; true 99% -> 268 as used by the rule).
(2) Fold-count perturbation: 10-fold GroupKFold -> same 99% rule -> set overlap
    with the adopted 268 (the meaningful stability axis; LGBM training is
    deterministic at colsample/subsample=1, so seed alone perturbs nothing).
Outputs: results/f1b_selection_integrity.json, results/f1b_imp_ge1_5fold.parquet
"""
import json, warnings
import numpy as np, pandas as pd
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

R = "results"
SEED = 42
CUM_KEEP = 0.99

m4cols = pq.ParquetFile(f"{R}/n5_abi_rolling_features.parquet").schema.names
ecols = pq.ParquetFile(f"{R}/p1_expanded_features.parquet").schema.names
CAND = sorted(set(c for c in m4cols if c != "stay_id") &
              set(c for c in ecols if c not in ("stay_id", "label", "label_severe")))
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[["stay_id"] + CAND]
for c in CAND:
    m4[c] = pd.to_numeric(m4[c], errors="coerce")

# E02-corrected ge1 labels (verbatim)
labs = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet").rename(columns={'valuenum': 'cr_val'})
labs = labs[labs['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
labs['t_hr'] = (labs['hr'] // 6) * 6
cb = labs.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index().sort_values(['stay_id', 't_hr'])
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
m4f = m4.merge(mg[['stay_id', 't_hr', 'ge1']], on=['stay_id', 't_hr'], how='left')
assert m4f['ge1'].sum() == 3425, f"label mismatch: {m4f['ge1'].sum()}"

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
Xtr = m4f[CAND].values[tr]
y = m4f['ge1'].values[tr]
groups = m4f['stay_id'].map(s2sub).values[tr]

PARAMS_SEL = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
                  class_weight='balanced', random_state=SEED, verbose=-1,
                  importance_type='gain')

def cv_gain(nfolds):
    acc = np.zeros(len(CAND))
    gkf = GroupKFold(n_splits=nfolds)
    for k, (itr, _), in enumerate(gkf.split(Xtr, y, groups), 1):
        m = lgb.LGBMClassifier(**PARAMS_SEL)
        m.fit(Xtr[itr], y[itr])
        imp = m.feature_importances_.astype(float)
        acc += imp / imp.sum()
        print(f"  {nfolds}-fold: fold {k}/{nfolds}", flush=True)
    return acc / nfolds

def set_at(imp, cum_target):
    order = np.argsort(imp)[::-1]
    cum = np.cumsum(imp[order]) / imp.sum()
    nk = min(int(np.searchsorted(cum, cum_target) + 1), len(CAND))
    return [CAND[i] for i in order[:nk]], cum

print("[1/2] canonical 5-fold seed-42 importance (curve-label fix)...", flush=True)
imp5 = cv_gain(5)
pd.DataFrame({'feature': CAND, 'imp_gain_cv5': imp5}).to_parquet(f"{R}/f1b_imp_ge1_5fold.parquet", index=False)
sel5, cum5 = set_at(imp5, CUM_KEEP)
pts = {f"{p:.1%}": int(np.searchsorted(cum5, p) + 1) for p in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)}
assert sel5 == json.load(open(f"{R}/_reselected_features.json")) or set(sel5) == set(json.load(open(f"{R}/_reselected_features.json"))), \
    "canonical 5-fold recompute disagrees with adopted set"
print(f"  curve: {pts}", flush=True)
print(f"  99% set: {len(sel5)} (matches adopted)", flush=True)

print("[2/2] 10-fold perturbation...", flush=True)
imp10 = cv_gain(10)
sel10, _ = set_at(imp10, CUM_KEEP)
adopted = set(json.load(open(f"{R}/_reselected_features.json")))
s10 = set(sel10)
j = len(adopted & s10) / len(adopted | s10)
print(f"  10-fold set: {len(s10)} | overlap {len(adopted & s10)}/{len(adopted)} | jaccard {j:.3f}", flush=True)
from scipy.stats import spearmanr
rho, pv = spearmanr(imp5, imp10)

json.dump({'canonical_curve_n_features': pts,
           'n_selected_5fold_99pct': len(sel5),
           'fold_perturbation': {'n_selected_10fold': len(s10),
                                  'overlap_with_adopted': f"{len(adopted & s10)}/{len(adopted)}",
                                  'jaccard': round(float(j), 3),
                                  'spearman_imp5_vs_imp10': round(float(rho), 4)},
           'note': 'f1 curve key collision fixed: int(0.995*100)=99 had overwritten the 99% entry; '
                   'true 99% cutoff is 268 (as used by the rule), 316 belongs to 99.9%'},
          open(f"{R}/f1b_selection_integrity.json", "w"), indent=2)
print("Saved f1b_selection_integrity.json")
