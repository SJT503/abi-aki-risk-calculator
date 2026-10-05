# -*- coding: utf-8 -*-
"""s1e2_full: Complete analysis suite on E02-corrected predictions.
Computes calibration, deployment, DeLong, DCA, hospital, MK, subgroups, SHAP."""
import json, warnings, os
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

R = "results"
SEED = 42
FEATS = json.load(open(f"{R}/_frozen291_features.json"))
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=SEED, verbose=-1)

# Load corrected predictions
pi = pd.read_parquet(f"{R}/s1e2_preds_int_ge1.parquet")
pe = pd.read_parquet(f"{R}/s1e2_preds_ext_ge1.parquet")

# Load features for LR baseline and calibration retraining
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[['stay_id'] + FEATS]
eicu = pd.read_parquet(f"{R}/p1_expanded_features.parquet")[['stay_id'] + FEATS]

# Recompute labels for the full pipeline
def labels_e02(labs_parquet, feat_df):
    labs = pd.read_parquet(labs_parquet)
    if 'sid' in labs.columns:
        cr = labs[labs['labname']=='creatinine'].rename(columns={'sid':'stay_id','labresult':'cr_val'})
    else:
        cr = labs.rename(columns={'valuenum':'cr_val'})
    cr = cr[cr['cr_val'].between(0.1,30)].sort_values(['stay_id','hr'])
    cr['t_hr'] = (cr['hr']//6)*6
    cb = cr.groupby(['stay_id','t_hr'])['cr_val'].max().reset_index().sort_values(['stay_id','t_hr'])
    mg = feat_df[['stay_id','t_hr','cr_base']].drop_duplicates().merge(cb,on=['stay_id','t_hr'],how='left').sort_values(['stay_id','t_hr'])
    mg['f'] = np.nan
    for s in range(0,9):
        sh = mg.groupby('stay_id')['cr_val'].shift(-s)
        mg['f'] = pd.DataFrame({'c':mg['f'],'n':sh}).max(axis=1)
    fold = mg['f']/mg['cr_base'].clip(lower=0.1)
    rise = mg['f']-mg['cr_base']
    aki = (fold>=1.5)|(rise>=0.3)
    st = pd.Series(0,index=mg.index)
    st[aki.fillna(False)] = 1
    st[fold>=2.0] = 2
    st[(fold>=3.0)|((mg['f']>=4.0)&aki)] = 3
    st[mg['f'].isna()] = 0
    mg['ge1'] = (st>=1).astype(int)
    # onset for deployment (first criteria-met bin >=24h)
    crit = ((fold>=1.5)|(rise>=0.3)).fillna(False)&(mg['t_hr']>=24)
    mg['crit'] = crit
    onsets = {sid: g['t_hr'].min() for sid,g in mg[mg['crit']].groupby('stay_id')}
    return mg[['stay_id','t_hr','ge1']], onsets

m4_lab, m4_onset = labels_e02(f"{R}/n2_m4_abi_cr_serial.parquet", m4)
ei_lab, ei_onset = labels_e02(f"{R}/n8_2_labs_series.parquet", eicu)
m4f = m4.merge(m4_lab, on=['stay_id','t_hr'], how='left')
eif = eicu.merge(ei_lab, on=['stay_id','t_hr'], how='left')

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id','subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8*len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr

# Retrain ge1 model for calibration
mod1 = lgb.LGBMClassifier(**PARAMS)
mod1.fit(m4f[FEATS].values[tr], m4f['ge1'].values[tr])
ptr = mod1.predict_proba(m4f[FEATS].values[tr])[:,1]
pte = mod1.predict_proba(m4f[FEATS].values[te])[:,1]
pee = mod1.predict_proba(eif[FEATS].values)[:,1]
ytr, yte, yee = m4f['ge1'].values[tr], m4f['ge1'].values[te], eif['ge1'].values

# ============ 1. CALIBRATION (OOF) ============
print("=== Calibration (OOF) ===")
oof = cross_val_predict(lgb.LGBMClassifier(**PARAMS), m4f[FEATS].values[tr], ytr, cv=5, method='predict_proba')[:,1]
def logit(p): p=np.clip(p,1e-12,1-1e-12); return np.log(p/(1-p))
def sigmoid(z): return 1/(1+np.exp(-z))
cal = LogisticRegression(C=1e6).fit(logit(oof).reshape(-1,1), ytr)
a,b = float(cal.intercept_[0]), float(cal.coef_[0][0])
print(f"  recal layer: a={a:.6f} b={b:.6f}")
def ece10(y,p):
    q = pd.qcut(pd.Series(p),10,duplicates='drop')
    g = pd.DataFrame({'y':y,'p':p,'q':q}).groupby('q',observed=True)
    return float((g.size()/len(y)*np.abs(g['y'].mean()-g['p'].mean())).sum())
def metrics(y,p):
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1,1),y)
    return {'slope':round(float(lr.coef_[0][0]),3),'ece':round(ece10(y,p),4),'brier':round(float(np.mean((p-y)**2)),4)}
calib = {}
for nm,yy,pp in [('int_test',yte,pte),('ext',yee,pee)]:
    pr = sigmoid(a+b*logit(pp))
    calib[nm] = {'raw':metrics(yy,pp),'recal':metrics(yy,pr),
                  'brier_null':round(float(np.mean((np.mean(yy)-yy)**2)),4)}
    print(f"  {nm}: raw {calib[nm]['raw']} -> recal {calib[nm]['recal']}")
pi['p_rec'] = sigmoid(a+b*logit(pi['p']))
pe['p_rec'] = sigmoid(a+b*logit(pe['p']))
pi.to_parquet(f"{R}/s1e2_preds_int_ge1.parquet",index=False)
pe.to_parquet(f"{R}/s1e2_preds_ext_ge1.parquet",index=False)

# ============ 2. DEPLOYMENT ============
print("\n=== Deployment ===")
def deploy(df, onsets, thr):
    d = df.copy()
    d['alert'] = d['p_rec'] >= thr
    ev = d[d['y']==1].copy(); nev = d[d['y']==0].copy()
    ev['onset'] = ev['stay_id'].map(onsets)
    capt, leads = [], []
    for sid,g in ev.groupby('stay_id'):
        al = g[g['alert'] & (g['t_hr'] < g['onset'])]
        if len(al):
            capt.append(sid); leads.append(g['onset'].min()-al['t_hr'].min())
    alert_stays = set(d.loc[d['alert'],'stay_id'])
    n_capt = len(capt)
    fa_stays = alert_stays & set(nev['stay_id'])
    days = nev.groupby('stay_id')['t_hr'].max().clip(lower=6).div(24).sum()
    return {'thr':thr,'capture':round(n_capt/ev['stay_id'].nunique(),3),
            'lead_median_h':round(float(np.median(leads)),1) if leads else None,
            'lead_iqr_h':[round(float(np.percentile(leads,25)),1),round(float(np.percentile(leads,75)),1)] if leads else None,
            'NNE':round(len(alert_stays)/n_capt,2) if n_capt else None,
            'FA_per_100ptd':round(100*len(fa_stays)/days,1)}
deploy_int = {f"thr{t}":deploy(pi,m4_onset,t) for t in [0.05,0.10,0.15,0.20]}
deploy_ext = {f"thr{t}":deploy(pe,ei_onset,t) for t in [0.05,0.10,0.15,0.20]}
for t in [0.05,0.10,0.15,0.20]:
    print(f"  ext thr{t}: cap={deploy_ext[f'thr{t}']['capture']} lead={deploy_ext[f'thr{t}']['lead_median_h']}h NNE={deploy_ext[f'thr{t}']['NNE']} FA={deploy_ext[f'thr{t}']['FA_per_100ptd']}")

# ============ 3. DeLONG vs LR ============
print("\n=== DeLong vs LR baseline ===")
CAND = [c for c in ['age','charlson','cr_base'] if c in FEATS]
lr = LogisticRegression(max_iter=2000,class_weight='balanced')
lr.fit(m4f[CAND].fillna(0).values[tr], ytr)
lr_pi = lr.predict_proba(m4f[CAND].fillna(0).values[te])[:,1]
lr_pe = lr.predict_proba(eif[CAND].fillna(0).values)[:,1]
def midrank(x):
    J=np.argsort(x);Z=x[J];N=len(x);T=np.zeros(N);i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1); i=j
    T2=np.empty(N);T2[J]=T;return T2
def delong_paired(y,p1,p2):
    a1,a2=roc_auc_score(y,p1),roc_auc_score(y,p2)
    pos=y==1;n1,n0=int(pos.sum()),int((~pos).sum())
    x1,x0=np.ascontiguousarray(p1[pos]),np.ascontiguousarray(p1[~pos])
    tx=midrank(np.r_[x1,x0]);tz=np.r_[midrank(x1),midrank(x0)]
    v10a=(tx[:n1]-tz[:n1])/n0;v01a=(tx[n1:]-tz[n1:])/n1
    x1b,x0b=np.ascontiguousarray(p2[pos]),np.ascontiguousarray(p2[~pos])
    tx2=midrank(np.r_[x1b,x0b]);tz2=np.r_[midrank(x1b),midrank(x0b)]
    v10b=(tx2[:n1]-tz2[:n1])/n0;v01b=(tx2[n1:]-tz2[n1:])/n1
    c1=np.cov(np.stack([v10a,v10b]))[0,1]/n1;c0=np.cov(np.stack([v01a,v01b]))[0,1]/n0
    vd=c1+c0;d=a1-a2;z=d/np.sqrt(vd)
    from scipy.stats import norm
    return {'auc1':round(a1,4),'auc2':round(a2,4),'diff':round(d,4),'z':round(z,2),
            'p':f'{2*(1-norm.cdf(abs(z))):.2e}'}
delong = {'int':delong_paired(yte,pte,lr_pi),'ext':delong_paired(yee,pee,lr_pe)}
print(f"  ext: diff={delong['ext']['diff']} z={delong['ext']['z']}")

# ============ 4. DCA ============
print("\n=== DCA ===")
def dca_calc(y,p,thrs):
    n=len(y);prev=y.mean();rows=[]
    for t in thrs:
        al=p>=t;tp=(al&(y==1)).sum()/n;fp=(al&(y==0)).sum()/n
        rows.append({'thr':round(t,3),'nb_model':round(tp-fp*(t/(1-t)),5),
                      'nb_treat_all':round(prev-(1-prev)*(t/(1-t)),5)})
    return rows
ths=np.arange(0.05,0.505,0.025)
dca_int=dca_calc(yte,pi['p_rec'].values,ths)
dca_ext=dca_calc(yee,pe['p_rec'].values,ths)
print(f"  ext model>treat_all all: {all(r['nb_model']>r['nb_treat_all'] for r in dca_ext)}")

# ============ 5. HOSPITAL + MK + SUBGROUPS ============
print("\n=== Hospital/MK/Subgroups ===")
hmap = pd.read_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet")[["stay_id","hospitalid"]]
pe2 = pe.merge(hmap,on='stay_id',how='left')
hp = pe2.groupby('hospitalid').apply(lambda g: roc_auc_score(g.y,g.p) if g.y.sum()>=20 else np.nan).dropna()
hospital = {'n_ge20':int(len(hp)),'median':round(float(hp.median()),3),
             'min':round(float(hp.min()),3),'max':round(float(hp.max()),3),
             'n_below_060':int((hp<0.60).sum())}
print(f"  hospitals: {hospital}")
from scipy.stats import kendalltau
strata = pe2.groupby(pd.cut(pe2['t_hr'],bins=np.arange(24,169,24))).apply(
    lambda g: roc_auc_score(g.y,g.p) if g.y.sum()>0 else np.nan).dropna()
tau,pv = kendalltau(range(len(strata)),strata.values)
mk = {'tau':round(float(tau),3),'p':round(float(pv),3),
       'strata':[round(float(v),3) for v in strata.values]}
print(f"  MK: tau={tau:.3f} p={pv:.3f}")
sex = pd.read_parquet(f"{R}/p1_expanded_features.parquet",columns=['stay_id','male','age']).drop_duplicates('stay_id')
pe3 = pe.merge(sex,on='stay_id',how='left')
sub = {}
for nm,m in [('age<65',pe3['age']<65),('age>=65',pe3['age']>=65),
             ('male',pe3['male']==1),('female',pe3['male']==0)]:
    g=pe3[m];sub[nm]=round(float(roc_auc_score(g.y,g.p)),3)
print(f"  subgroups: {sub}")

# ============ 6. SHAP ============
print("\n=== SHAP ===")
import shap
ridx = np.random.default_rng(SEED).choice(np.where(te)[0],size=8000,replace=False)
Xs = m4f.iloc[ridx][FEATS]
expl = shap.TreeExplainer(mod1)
sv = expl.shap_values(Xs.values,check_additivity=False)
if isinstance(sv,list): sv=sv[1]
imp = np.abs(sv).mean(0)
order = np.argsort(imp)[::-1][:15]
shap_top = [{'feature':FEATS[i],'mean_abs':round(float(imp[i]),4)} for i in order]
pd.DataFrame(sv,columns=FEATS).assign(stay_id=m4f.iloc[ridx]['stay_id'].values).to_parquet(f"{R}/s1e2_shap_values.parquet",index=False)
Xs.assign(stay_id=m4f.iloc[ridx]['stay_id'].values).to_parquet(f"{R}/s1e2_shap_X.parquet",index=False)
print(f"  top5: {[x['feature'] for x in shap_top[:5]]}")

# ============ SAVE ============
r = json.load(open(f"{R}/s1e2_fixed_results.json"))
r['calibration_ge1'] = calib
r['deployment_ge1'] = {'internal_test':deploy_int,'external':deploy_ext}
r['delong'] = delong
r['dca'] = {'internal':dca_int,'external':dca_ext,
             'wins':{'int':all(x['nb_model']>x['nb_treat_all'] for x in dca_int),
                      'ext':all(x['nb_model']>x['nb_treat_all'] for x in dca_ext)}}
r['extra'] = {'hospital':hospital,'mk':mk,'subgroups':sub,'shap_top15':shap_top}
json.dump(r,open(f"{R}/s1e2_fixed_results.json","w"),indent=2)
print(f"\n{'='*60}")
print("ALL SAVED: s1e2_fixed_results.json (complete)")
