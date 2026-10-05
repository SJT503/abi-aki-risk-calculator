# -*- coding: utf-8 -*-
"""f1_reselect_e02: Re-screen the feature set on E02-corrected labels (PI decision 2026-09-30).

Protocol
--------
- Candidate pool: every feature computable in BOTH M4 (n5) and eICU (p1_expanded),
  i.e. external-validatable candidates only (includes the 59 features the old
  density freeze had dropped, e.g. lab SD/CV, lactate/po2/base_excess families).
- Labels: E02-corrected (identical machinery to s1_e02fix_only.py; window range(0,9);
  Stage 3 absolute branch gated by AKI precondition).
- Split: subject-level 80/20 seed 42, identical to the primary analysis.
  Selection uses TRAIN subjects ONLY (no test/external leakage into selection).
- Selection: 5-fold GroupKFold(by subject) CV within train; LightGBM gain importance
  with the primary hyperparameters; per-fold normalized, averaged across folds.
- Rule (predefined): keep features up to cumulative 99% of mean CV gain.
- Rebuild: retrain ge1/ge2/ge3 on the reselected set (primary hyperparameters),
  internal test + external eICU AUROC/AUPRC + stay-cluster bootstrap 95% CI,
  side-by-side against the frozen-291 results.

Outputs
-------
- results/_reselected_features.json        (ranked list, importance order)
- results/f1_reselection_results.json      (selection + rebuild + comparison)
- results/f1_preds_{int,ext}_ge{1,2,3}.parquet
"""
import json, warnings
import numpy as np, pandas as pd
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

R = "results"
SEED = 42
CUM_KEEP = 0.99          # predefined selection rule
N_FOLDS = 5

# ============ 1. Candidate pool ============
m4cols = pq.ParquetFile(f"{R}/n5_abi_rolling_features.parquet").schema.names
ecols = pq.ParquetFile(f"{R}/p1_expanded_features.parquet").schema.names
CAND = sorted(set(c for c in m4cols if c != "stay_id") &
              set(c for c in ecols if c not in ("stay_id", "label", "label_severe")))
FROZEN = json.load(open(f"{R}/_frozen291_features.json"))
print(f"Candidate pool (M4 AND eICU computable): {len(CAND)} | frozen291: {len(FROZEN)}", flush=True)
assert "t_hr" in CAND and "cr_base" in CAND, "pool must contain t_hr and cr_base"

m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[["stay_id"] + CAND]
eicu = pd.read_parquet(f"{R}/p1_expanded_features.parquet")[["stay_id"] + CAND]
for c in CAND:
    m4[c] = pd.to_numeric(m4[c], errors="coerce")
    eicu[c] = pd.to_numeric(eicu[c], errors="coerce")

# ============ 2. E02-corrected labels (verbatim s1_e02fix machinery) ============
def labels_e02fix(labs_parquet, feat_df, db):
    labs = pd.read_parquet(labs_parquet)
    if 'sid' in labs.columns:
        cr = labs[labs['labname'] == 'creatinine'].rename(columns={'sid': 'stay_id', 'labresult': 'cr_val'})
    else:
        cr = labs.rename(columns={'valuenum': 'cr_val'})
    cr = cr[cr['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
    cr['t_hr'] = (cr['hr'] // 6) * 6
    cb = cr.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index().sort_values(['stay_id', 't_hr'])
    mg = feat_df[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(
        cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
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
    st[(fold >= 3.0) | ((mg['f'] >= 4.0) & aki)] = 3   # E02 fix
    st[mg['f'].isna()] = 0
    for t in [1, 2, 3]:
        mg[f'ge{t}'] = (st >= t).astype(int)
    print(f"  {db}: ge1={int(mg['ge1'].sum())} ge2={int(mg['ge2'].sum())} ge3={int(mg['ge3'].sum())}", flush=True)
    return mg[['stay_id', 't_hr', 'ge1', 'ge2', 'ge3']]

m4_lab = labels_e02fix(f"{R}/n2_m4_abi_cr_serial.parquet", m4, "M4")
ei_lab = labels_e02fix(f"{R}/n8_2_labs_series.parquet", eicu, "eICU")
m4f = m4.merge(m4_lab, on=['stay_id', 't_hr'], how='left')
eif = eicu.merge(ei_lab, on=['stay_id', 't_hr'], how='left')
assert m4f[['ge1', 'ge2', 'ge3']].notna().all().all() and eif[['ge1', 'ge2', 'ge3']].notna().all().all()

# ============ 3. Subject-level split (identical to primary) ============
n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr
print(f"Split: {tr.sum()} train / {te.sum()} test checkpoints "
      f"({len(train_subj)}/{len(subjects) - len(train_subj)} subjects)", flush=True)

# ============ 4. CV gain-importance selection on TRAIN ONLY ============
PARAMS_SEL = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
                  class_weight='balanced', random_state=SEED, verbose=-1,
                  importance_type='gain')
Xtr = m4f[CAND].values[tr]
groups = m4f['stay_id'].map(s2sub).values[tr]
gkf = GroupKFold(n_splits=N_FOLDS)

def cv_gain(ykey):
    y = m4f[ykey].values[tr]
    acc = np.zeros(len(CAND))
    for k, (itr, _ite) in enumerate(gkf.split(Xtr, y, groups), 1):
        m = lgb.LGBMClassifier(**PARAMS_SEL)
        m.fit(Xtr[itr], y[itr])
        imp = m.feature_importances_.astype(float)
        acc += imp / imp.sum()
        print(f"  [{ykey}] fold {k}/{N_FOLDS} done", flush=True)
    return acc / N_FOLDS

print("Selection CV (ge1, primary rule):", flush=True)
imp_ge1 = cv_gain('ge1')
order = np.argsort(imp_ge1)[::-1]
cum = np.cumsum(imp_ge1[order]) / imp_ge1.sum()
nk = min(int(np.searchsorted(cum, CUM_KEEP) + 1), len(CAND))
SELECTED = [CAND[i] for i in order[:nk]]
curve = {f"{int(p*100)}%": int(np.searchsorted(cum, p) + 1) for p in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)}
print(f"  cumulative-gain curve (n features): {curve}", flush=True)
print(f"  rule {int(CUM_KEEP*100)}% -> keep {len(SELECTED)} features", flush=True)

# ge3 diagnostic (informational: severe endpoint's top features coverage)
print("Selection CV (ge3, diagnostic only):", flush=True)
imp_ge3 = cv_gain('ge3')
top50_ge3 = {CAND[i] for i in np.argsort(imp_ge3)[::-1][:50]}
cov3 = len(top50_ge3 & set(SELECTED))

# sanity gates before rebuild
assert 50 <= len(SELECTED) <= len(CAND), f"degenerate selection size {len(SELECTED)}"
ovl = len(set(SELECTED) & set(FROZEN))
print(f"\nOverlap with frozen291: {ovl}/{len(FROZEN)} ({100*ovl/len(FROZEN):.0f}%)", flush=True)
print(f"ge3 top-50 covered by selected set: {cov3}/50", flush=True)
print(f"Selected-not-in-frozen ({len(set(SELECTED)-set(FROZEN))}): {sorted(set(SELECTED)-set(FROZEN))[:20]}", flush=True)
print(f"Frozen-not-selected ({len(set(FROZEN)-set(SELECTED))}): {sorted(set(FROZEN)-set(SELECTED))[:20]}", flush=True)

json.dump(SELECTED, open(f"{R}/_reselected_features.json", "w"), indent=1)

# ============ 5. Rebuild 3 endpoints on reselected set ============
PARAMS_FIT = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
                  class_weight='balanced', random_state=SEED, verbose=-1)
print(f"\n{'='*65}\nRebuild on reselected {len(SELECTED)} features:", flush=True)
print(f"{'Endpoint':<12} {'Internal':>10} {'External':>10} {'AUPRC-I':>8} {'AUPRC-E':>8} {'Ev-Int':>7} {'Ev-Ext':>7}")
res = {}
for t in [1, 2, 3]:
    mod = lgb.LGBMClassifier(**PARAMS_FIT)
    mod.fit(m4f[SELECTED].values[tr], m4f[f'ge{t}'].values[tr])
    pi = mod.predict_proba(m4f[SELECTED].values[te])[:, 1]
    pe = mod.predict_proba(eif[SELECTED].values)[:, 1]
    yi, ye = m4f[f'ge{t}'].values[te], eif[f'ge{t}'].values
    ai = round(float(roc_auc_score(yi, pi)), 4)
    ae = round(float(roc_auc_score(ye, pe)), 4)
    res[f'ge{t}'] = {'internal': ai, 'external': ae,
                     'auprc_int': round(float(average_precision_score(yi, pi)), 4),
                     'auprc_ext': round(float(average_precision_score(ye, pe)), 4),
                     'events_int': int(yi.sum()), 'events_ext': int(ye.sum())}
    print(f"  ge{t}: int={ai:>8} ext={ae:>8} "
          f"API={res[f'ge{t}']['auprc_int']:>6} APE={res[f'ge{t}']['auprc_ext']:>6} "
          f"ev_i={int(yi.sum()):>5} ev_e={int(ye.sum()):>5}", flush=True)
    pd.DataFrame({'stay_id': m4f.loc[te, 'stay_id'].values, 'y': yi, 'p': pi,
                  't_hr': m4f.loc[te, 't_hr'].values}).to_parquet(
        f"{R}/f1_preds_int_ge{t}.parquet", index=False)
    pd.DataFrame({'stay_id': eif['stay_id'].values, 'y': ye, 'p': pe,
                  't_hr': eif['t_hr'].values}).to_parquet(
        f"{R}/f1_preds_ext_ge{t}.parquet", index=False)

def stay_boot(df, n=500, seed=SEED):
    r = np.random.default_rng(seed)
    stays = df['stay_id'].values
    uniq = pd.unique(stays)
    idx_by = {s: np.where(stays == s)[0] for s in uniq}
    yv, pv = df['y'].values, df['p'].values
    aucs = []
    for _ in range(n):
        sel = r.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in sel])
        yy, pp = yv[idx], pv[idx]
        if yy.sum() > 0 and (1 - yy).sum() > 0:
            aucs.append(roc_auc_score(yy, pp))
    return np.percentile(aucs, [2.5, 97.5]).round(4).tolist()

print(f"\n95% CI (stay-cluster bootstrap, 500x):", flush=True)
for t in [1, 2, 3]:
    ci_i = stay_boot(pd.read_parquet(f"{R}/f1_preds_int_ge{t}.parquet"))
    ci_e = stay_boot(pd.read_parquet(f"{R}/f1_preds_ext_ge{t}.parquet"))
    res[f'ge{t}']['ci_int'] = ci_i
    res[f'ge{t}']['ci_ext'] = ci_e
    print(f"  ge{t}: int [{ci_i[0]:.3f}, {ci_i[1]:.3f}] ext [{ci_e[0]:.3f}, {ci_e[1]:.3f}]", flush=True)

# ============ 6. Comparison vs frozen-291 ============
old = json.load(open(f"{R}/s1e2_fixed_results.json"))['results']
print(f"\n{'='*65}\n{'Endpoint':<10} {'':>22} frozen291 -> reselected")
for t in [1, 2, 3]:
    o, n = old[f'ge{t}'], res[f'ge{t}']
    print(f"  ge{t}: internal {o['internal']} -> {n['internal']} | external {o['external']} -> {n['external']}")

json.dump({
    '_meta': {'protocol': 'feature re-selection on E02-corrected labels (PI decision 2026-09-30)',
              'date': '2026-09-30',
              'candidate_pool': len(CAND), 'rule': f'cumulative {CUM_KEEP:.0%} of 5-fold GroupKFold CV gain (train subjects only)',
              'folds': N_FOLDS, 'seed': SEED},
    'curve_n_features': curve,
    'n_selected': len(SELECTED),
    'overlap_with_frozen291': f"{ovl}/{len(FROZEN)}",
    'ge3_top50_covered': f"{cov3}/50",
    'selected_not_in_frozen': sorted(set(SELECTED) - set(FROZEN)),
    'frozen_not_selected': sorted(set(FROZEN) - set(SELECTED)),
    'results': res,
    'comparison_frozen291': old,
}, open(f"{R}/f1_reselection_results.json", "w"), indent=2)
print("\nSaved: _reselected_features.json, f1_reselection_results.json, f1_preds_*.parquet")
