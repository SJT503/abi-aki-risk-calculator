import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B = r'E:/TBI subtype/09_tbi_aki/results/'

M = pd.read_parquet(B + 'jinhu/f12_preds_M4_ge1.parquet').merge(
    pd.read_parquet(B + 'n5_abi_rolling_features.parquet')[
        ['stay_id', 't_hr', 'cr_ratio_base', 'cr_delta_since_adm', 'cr_last', 'cr_base', 'label']],
    on=['stay_id', 't_hr'], how='left', validate='one_to_one')
G = pd.read_parquet(B + 'n3_m4_abi_grid.parquet')
M = M.merge(G[['stay_id', 't_hr', 'onset_hr']], on=['stay_id', 't_hr'], how='left',
            validate='one_to_one')

print('pred label == grid label?')
Gl = G.set_index(['stay_id', 't_hr']).label
chk = M.set_index(['stay_id', 't_hr']).label
j = Gl.align(chk, join='inner')
print('   identical:', bool((j[0].values == j[1].values).all()), 'n=', len(j[0]))
print('   n with label NaN in feat:', int(M.label.isna().sum()))

defn = (M.cr_ratio_base >= 1.5) | (M.cr_delta_since_adm >= 0.3)
lead_ok = (M.onset_hr - M.t_hr) >= 24

print()
print('=== T6 variants for MIMIC (n=%d) ===' % len(M))
variants = {
    'A: defn-drop, 24h, NO ratio-obligation (my reading)':
        (M.y == 0) | ((M.y == 1) & (~defn) & lead_ok),
    'A0: defn-drop only, NO 24h filter':
        (M.y == 0) | ((M.y == 1) & (~defn)),
    'A_ratioObsOnly: defn-drop, 24h, positives need onset & ratio':
        (M.y == 0) | ((M.y == 1) & (~defn) & lead_ok & M.cr_ratio_base.notna()),
}
for k, mask in variants.items():
    s = M[mask]
    print(f'  {k}\n     n={len(s)} pos={int(s.y.sum())} posrate={s.y.mean():.4f} AUROC={roc_auc_score(s.y,s.p):.4f}')

# variant: onset required observable for POSITIVES only, 24h
m2 = (M.y == 0) | ((M.y == 1) & (~defn) & M.onset_hr.notna() & ((M.onset_hr - M.t_hr) >= 24))
s = M[m2]
print(f'  B: positives need onset observable + >=24h (negatives unrestricted)\n'
      f'     n={len(s)} pos={int(s.y.sum())} AUROC={roc_auc_score(s.y,s.p):.4f}')
s = M[m2 & M.cr_ratio_base.notna()]
print(f'  B+ratio-observed-all\n     n={len(s)} pos={int(s.y.sum())} AUROC={roc_auc_score(s.y,s.p):.4f} '
      f'ratio={roc_auc_score(s.y,s.cr_ratio_base):.4f}')

# what does onset_hr NaN mean for positives? check defn status of those
pos = M[M.y == 1]
print()
print('positives with onset_hr NaN: n=%d' % int(pos.onset_hr.isna().sum()))
print('  of those, defn-positive:', int(defn[pos.onset_hr.isna()].sum()),
      ' non-defn:', int((~defn[pos.onset_hr.isna()]).sum()))
print('  of those, cr_ratio_base observed:', int(pos.loc[pos.onset_hr.isna(), 'cr_ratio_base'].notna().sum()))
print('negatives with onset_hr NaN: n=%d' % int(M.loc[M.y == 0, 'onset_hr'].isna().sum()))
print('negatives with onset observed: n=%d' % int(M.loc[M.y == 0, 'onset_hr'].notna().sum()))
print()
print('onset_hr unique values sample:', np.sort(M.onset_hr.dropna().unique())[:12], '... max', M.onset_hr.max())
print('t_hr range:', M.t_hr.min(), M.t_hr.max())
