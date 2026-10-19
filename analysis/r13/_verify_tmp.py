import numpy as np, pandas as pd, json, os
from sklearn.metrics import roc_auc_score

B = r'E:/TBI subtype/09_tbi_aki/results/'
OUT = []


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    m = np.isfinite(s) & np.isfinite(y)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return np.nan, int(m.sum())
    return float(roc_auc_score(y[m], s[m])), int(m.sum())


# ---------------- load ----------------
pred_m = pd.read_parquet(B + 'jinhu/f12_preds_M4_ge1.parquet')
pred_e = pd.read_parquet(B + 'jinhu/f12_preds_eICU_ge1.parquet')
feat_m = pd.read_parquet(B + 'n5_abi_rolling_features.parquet')
feat_e = pd.read_parquet(B + 'p1_expanded_features.parquet')
grid_m = pd.read_parquet(B + 'n3_m4_abi_grid.parquet')
grid_e = pd.read_parquet(B + 'n8_3_eicu_abi_grid.parquet')
shapX = pd.read_parquet(B + 'jinhu/f12_shap_X.parquet')

p('=' * 78); p('T1: MODEL AUROC ON ALL ROWS')
for nm, pr in (('MIMIC', pred_m), ('eICU', pred_e)):
    a = roc_auc_score(pr.y, pr.p)
    p(f'  {nm}: n={len(pr)} pos={int(pr.y.sum())} posrate={pr.y.mean():.5f} '
      f'stays={pr.stay_id.nunique()} AUROC={a:.4f}')

# ---------------- key uniqueness ----------------
p('=' * 78); p('KEY UNIQUENESS (stay_id,t_hr)')
for nm, df in (('pred_M', pred_m), ('feat_M', feat_m), ('pred_e', pred_e), ('feat_e', feat_e)):
    d = df.duplicated(['stay_id', 't_hr']).sum()
    p(f'  {nm}: rows={len(df)} dup_pairs={d}')

# ---------------- T2: merge ----------------
p('=' * 78); p('T2: MISSINGNESS OF cr_ratio_base AFTER LEFT MERGE')


def build(pred, feat, cols=('cr_ratio_base', 'cr_delta_since_adm', 'cr_last', 'cr_base')):
    return pred.merge(feat[['stay_id', 't_hr'] + list(cols)], on=['stay_id', 't_hr'],
                      how='left', validate='one_to_one', indicator=True)


M = build(pred_m, feat_m)
E = build(pred_e, feat_e)
for nm, d in (('MIMIC', M), ('eICU', E)):
    nmiss = int(d.cr_ratio_base.isna().sum())
    nomatch = int((d._merge == 'left_only').sum())
    inner_nan = int(d.loc[d._merge == 'both', 'cr_ratio_base'].isna().sum())
    p(f'  {nm}: n={len(d)} NaN_ratio={nmiss} ({nmiss/len(d)*100:.2f}%) '
      f'unmatched_rows={nomatch} NaN_but_matched={inner_nan}')
    p(f'        stay_id set diff pred\\feat={len(set(pred_m.stay_id)-set(feat_m.stay_id)) if nm=="MIMIC" else len(set(pred_e.stay_id)-set(feat_e.stay_id))} '
      f'feat\\pred={len(set(feat_m.stay_id)-set(pred_m.stay_id)) if nm=="MIMIC" else len(set(feat_e.stay_id)-set(pred_e.stay_id))}')

# ---------------- T3 ----------------
p('=' * 78); p('T3: MATCHED-SUBSET AUROCS')
res = {}
for nm, d in (('MIMIC', M), ('eICU', E)):
    s = d[d.cr_ratio_base.notna()]
    am = roc_auc_score(s.y, s.p)
    ar = roc_auc_score(s.y, s.cr_ratio_base)
    npos_ = int(s.y.sum())
    res[nm] = dict(n=len(s), pos=npos_, am=am, ar=ar)
    p(f'  {nm}: matched n={len(s)} pos={npos_} | model={am:.4f} ratio={ar:.4f} '
      f'| model_all={roc_auc_score(d.y,d.p):.4f}')

# ---------------- T4 ----------------
p('=' * 78); p('T4: EXCESS DISCRIMINATION SHARE')
for nm, d in (('MIMIC', M), ('eICU', E)):
    s = d[d.cr_ratio_base.notna()]
    share_matched = (roc_auc_score(s.y, s.cr_ratio_base) - .5) / (roc_auc_score(s.y, s.p) - .5)
    # ORIGINAL mismatched: ratio on NaN-dropped rows vs model on ALL rows
    am_all = roc_auc_score(d.y, d.p)
    share_orig = (roc_auc_score(s.y, s.cr_ratio_base) - .5) / (am_all - .5)
    p(f'  {nm}: matched-share={share_matched*100:.1f}%  ORIGINAL(mismatched)={share_orig*100:.1f}%')

# ---------------- T5 ----------------
p('=' * 78); p('T5: DIRECTION SANITY')
for nm, d in (('MIMIC', M), ('eICU', E)):
    s = d[d.cr_ratio_base.notna()]
    p(f'  {nm}: mean_ratio pos={s.loc[s.y==1,"cr_ratio_base"].mean():.4f} '
      f'neg={s.loc[s.y==0,"cr_ratio_base"].mean():.4f} '
      f'| pos_on_ALL_mean={d.loc[d.y==1,"cr_ratio_base"].mean():.4f}')

# ---------------- T6 ----------------
p('=' * 78); p('T6: HONEST FORECASTING SUBSET')


def t6(d, grid, label):
    g = d.merge(grid[['stay_id', 't_hr', 'onset_hr']], on=['stay_id', 't_hr'],
                how='left', validate='one_to_one')
    return g


G_M = t6(M, grid_m, 'MIMIC')
G_E = t6(E, grid_e, 'eICU')
for nm, d in (('MIMIC', G_M), ('eICU', G_E)):
    p(f'  --{nm}-- onset NaN: pos={int(d.loc[d.y==1,"onset_hr"].isna().sum())} '
      f'neg={int(d.loc[d.y==0,"onset_hr"].isna().sum())} '
      f'n_missing_grid_rows={int(d.onset_hr.isna().sum())}')
    defn = (d.cr_ratio_base >= 1.5) | (d.cr_delta_since_adm >= 0.3)
    # (a) without requiring ratio observed on negatives
    a = d[(d.y == 0) | ((d.y == 1) & (~defn) & ((d.onset_hr - d.t_hr) >= 24))]
    p(f'     (a) n={len(a)} pos={int(a.y.sum())} posrate={a.y.mean():.4f} AUROC_model={roc_auc_score(a.y,a.p):.4f}')
    # (b) matched
    b = a[a.cr_ratio_base.notna()]
    ab = roc_auc_score(b.y, b.p); ar = roc_auc_score(b.y, b.cr_ratio_base)
    p(f'     (b) n={len(b)} pos={int(b.y.sum())} AUROC_model={ab:.4f} ratio={ar:.4f} '
      f'share={(ar-.5)/(ab-.5)*100:.1f}%')
    p(f'     dropped by matching: {len(a)-len(b)} rows '
      f'(pos={int(a.y.sum()-b.y.sum())}, neg={int((a.y==0).sum()-(b.y==0).sum())})')

# ---------------- T7 ----------------
p('=' * 78); p('T7: IS cr_ratio_base A FROZEN MODEL FEATURE?')
p(f'  in f12_shap_X columns: {"cr_ratio_base" in shapX.columns}  (shap_X shape={shapX.shape})')
p(f'  shap_X is a subsample: rows={len(shapX)} stays={shapX.stay_id.nunique()}')
try:
    ev = json.load(open(B + 'jinhu/f11_full_eval.json'))
    p('  f11_full_eval.json top-level keys: ' + ', '.join(list(ev.keys())[:20]))
    for k, v in ev.items():
        if isinstance(v, list) and v and isinstance(v[0], (str, dict)):
            if isinstance(v[0], dict) and any('cr_' in str(x) for x in v[0].keys()):
                p(f'   list key "{k}" first 25: {v[:25]}')
except Exception as e:
    p('  f11 read err ' + str(e))
p('  --- f12_log.txt ---')
print(open(B + 'jinhu/f12_log.txt', encoding='utf-8', errors='replace').read())
p(OUT[-1] if False else '')

# ---------------- T8 ----------------
p('=' * 78); p('T8: MISSED-THINGS SCAN')
# univariate AUROC scan
for nm, d, feat in (('MIMIC', M, feat_m), ('eICU', E, feat_e)):
    num = [c for c in feat.select_dtypes(include=[np.number]).columns
           if c not in ('stay_id', 't_hr')]
    dd = d[['stay_id', 't_hr', 'y', 'p']].merge(
        feat[['stay_id', 't_hr'] + list(num)], on=['stay_id', 't_hr'], how='left',
        validate='one_to_one')
    rows = []
    for c in num:
        if c in ('y',) or dd[c].nunique(dropna=True) < 2:
            continue
        a, nn = auc(dd.y.values, dd[c].values)
        if np.isfinite(a):
            rows.append((c, a, nn, nn / len(dd)))
    rows.sort(key=lambda r: -r[1])
    p(f'  --- {nm} top 12 univariate AUROC ---')
    for c, a, nn, cov in rows[:12]:
        p(f'      {c:28s} AUROC={a:.4f} n_obs={nn} cov={cov*100:.1f}%')
    json.dump([dict(feature=c, auroc=a, n=nn) for c, a, nn, _ in rows],
              open(B + f'_verify_univ_{nm}.json', 'w'), indent=1)

open(r'E:/TBI subtype/09_tbi_aki/_analysis_r13/scripts/_verify_out.txt', 'w').write('\n'.join(OUT))
print('\nDONE')
