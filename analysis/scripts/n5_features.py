# -*- coding: utf-8 -*-
"""N5: ABI 滚动特征构建器（高效版：前缀和+searchsorted，~75k 检查点×~200 特征）"""
import os, sys, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate
gate(__file__)

R = "E:/TBI subtype/09_tbi_aki/results"
grid = pd.read_parquet(f"{R}/n3_m4_abi_grid.parquet")
coh = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet")
ch = pd.read_parquet(f"{R}/n2_m4_abi_charlson.parquet").set_index("stay_id")
wt1 = pd.read_parquet(f"{R}/n2_m4_abi_weights.parquet")
_PRI = {226512:0, 224639:1, 226531:2}
wt1["_p"] = wt1.itemid.map(_PRI).fillna(9)
wtk = (wt1.sort_values(["stay_id","_p","hr_from_icu"],kind="mergesort")
         .groupby("stay_id",as_index=False).first().set_index("stay_id").value)
uo = pd.read_parquet(f"{R}/n2_m4_abi_uo_rows.parquet")
vit = pd.read_parquet(f"{R}/n4_abi_vitals.parquet")
labs = pd.read_parquet(f"{R}/n4_abi_labs.parquet").dropna(subset=["hr"])
meds = pd.read_parquet(f"{R}/n4_abi_meds.parquet")
fl = pd.read_parquet(f"{R}/n4_abi_fluids.parquet")
vent = pd.read_parquet(f"{R}/n4_abi_vent.parquet")
cr = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet")

VIT_MAP = {220045:"hr_rate",220179:"sbp",220052:"mbp",220210:"rr",220277:"spo2",
           226329:"temp_c1",223762:"temp_c2",220739:"gcs_eye",223900:"gcs_verbal",223901:"gcs_motor"}
LAB_MAP = {50912:"cr",50983:"na",50971:"k",50931:"glu",51006:"bun",50902:"cl",
           50882:"hco3",50868:"ag",51301:"wbc",51221:"hct",
           50813:"lactate",50821:"po2",50802:"base_excess",50960:"mg",50970:"phos",
           51277:"rdw",51237:"inr",51222:"hb"}
MED_MAP = {221906:"norepi",221289:"epi",221662:"dopa",221668:"dobo",222315:"vaso",
           222168:"propofol",229420:"dex",225150:"mido",227871:"furosemide",221794:"furosemide2",
           227531:"mannitol_ie",225161:"hts_ie"}

# 序列字典（排序后入 dict）
ser = {}
for iid, nm in VIT_MAP.items():
    g = vit[vit.itemid==iid]
    if nm == "temp_c2": g = g.assign(valuenum=(g.valuenum-32)/1.8)
    for sid, gg in g.groupby("stay_id"):
        gg = gg.sort_values("hr",kind="mergesort")
        ser.setdefault((sid,nm), (gg.hr.to_numpy(float), gg.valuenum.to_numpy(float)))
for iid, nm in LAB_MAP.items():
    for sid, gg in labs[labs.itemid==iid].groupby("stay_id"):
        gg = gg.sort_values("hr",kind="mergesort")
        ser.setdefault((sid,nm), (gg.hr.to_numpy(float), gg.valuenum.to_numpy(float)))
for sid, gg in uo.groupby("stay_id"):
    gg = gg.sort_values("hr_from_icu",kind="mergesort")
    ser.setdefault((sid,"uo"), (gg.hr_from_icu.to_numpy(float), gg.uo_signed.to_numpy(float)))
for sid, gg in cr.groupby("stay_id"):
    gg = gg.sort_values("hr",kind="mergesort")
    ser.setdefault((sid,"cr_full"), (gg.hr.to_numpy(float), gg.valuenum.to_numpy(float)))

# 药物/液体/通气
med_ev = {}
for iid, nm in MED_MAP.items():
    for sid, gg in meds[meds.itemid==iid].groupby("stay_id"):
        med_ev.setdefault((sid,nm), (gg.start_hr.to_numpy(float), gg.end_hr.to_numpy(float)))
fl_g = {sid:(g.start_hr.to_numpy(float), g.amount.to_numpy(float)) for sid,g in fl.groupby("stay_id")}
vent_g = {sid: np.sort(g.hr.to_numpy(float)) for sid,g in vent.groupby("stay_id")}

# 静态
ch_k = ch.charlson if "charlson" in ch.columns else ch.charlson
cr_base = {}
for sid, gg in cr.groupby("stay_id"):
    gg = gg.sort_values("hr",kind="mergesort")
    w24 = gg[(gg.hr>=0)&(gg.hr<=24)]
    pre = gg[gg.hr<0]
    if len(w24): cr_base[sid] = float(w24.valuenum.min())
    elif len(pre): cr_base[sid] = float(pre.valuenum.min())

def wf(hrs, vals, t, w=24.0):
    """窗口特征（排序安全）；D33 变异度族：sd/cv（n>=2 才算；cv 分母 |mean|<1e-9→NaN）"""
    m = (hrs > t-w) & (hrs <= t)
    hh, vv = hrs[m], vals[m]
    if len(vv)==0: return None
    o = np.argsort(hh,kind="mergesort"); hh,vv=hh[o],vv[o]
    slope = np.polyfit(hh,vv,1)[0] if len(vv)>=2 and np.ptp(hh)>0 else 0.0
    sd = float(np.std(vv, ddof=1)) if len(vv)>=2 else np.nan
    mu = float(np.mean(vv))
    cv = float(np.std(vv, ddof=1)/abs(mu)) if len(vv)>=2 and abs(mu)>1e-9 else np.nan
    return {"last":vv[-1],"min":vv.min(),"max":vv.max(),"delta":vv[-1]-vv[0],
            "slope":slope,"n":len(vv),"h_last":t-hh[-1],"sd":sd,"cv":cv}

print("Building features for", grid.stay_id.nunique(), "patients /", len(grid), "checkpoints...")
rows = []
grid_grouped = list(grid.groupby("stay_id"))
for pi, (sid, g) in enumerate(grid_grouped):
    if pi % 500 == 0: print(f"  {pi}/{len(grid_grouped)}...", flush=True)
    wk = float(wtk.get(sid, np.nan))
    cvd = int(ch_k.get(sid, 0)) if sid in ch_k.index else 0
    base = cr_base.get(sid, np.nan)
    vh = vent_g.get(sid)
    for t, label in zip(g.t_hr, g.label):
        t = float(t)
        r = {"stay_id":sid, "t_hr":t, "label":label,
             "age":coh.set_index("stay_id").anchor_age.get(sid,np.nan),
             "male":int(coh.set_index("stay_id").gender.get(sid,"F")=="M"),
             "charlson":cvd, "cr_base":base}
        for nm in set(n for (s,n) in ser if s==sid):
            hrs_s, vals_s = ser[(sid,nm)]
            f = wf(hrs_s, vals_s, t)
            if f:
                for k,v in f.items(): r[f"{nm}_{k}"]=v
            j = np.searchsorted(hrs_s,t,side="right")-1
            if j>=0:
                r[f"{nm}_locf"]=float(vals_s[j]); r[f"{nm}_locf_age"]=float(t-hrs_s[j])
        if "cr_last" in r and pd.notna(base):
            r["cr_ratio_base"]=r["cr_last"]/base; r["cr_delta_since_adm"]=r["cr_last"]-base
        # 48h 窗
        for nm in ("cr","bun","hr_rate","sbp"):
            if (sid,nm) in ser:
                h,v = ser[(sid,nm)]
                m48=(h>t-48)&(h<=t)
                if m48.any():
                    hh,vv=h[m48],v[m48]
                    o=np.argsort(hh,kind="mergesort"); hh,vv=hh[o],vv[o]
                    r[f"{nm}_last48"]=float(vv[-1]); r[f"{nm}_delta48"]=float(vv[-1]-vv[0])
                    r[f"{nm}_slope48"]=float(np.polyfit(hh,vv,1)[0]) if len(vv)>=2 and np.ptp(hh)>0 else 0.0
                    r[f"{nm}_sd48"]=float(np.std(vv,ddof=1)) if len(vv)>=2 else np.nan
        # GCS 总分
        lv,mv,vv_ = [r.get(f"gcs_{x}_locf",np.nan) for x in ("eye","motor","verbal")]
        if all(np.isfinite(x) for x in (lv,mv,vv_)): r["gcs_total_locf"]=lv+mv+vv_
        # 药物
        for (s2,mnm),(sa,ea) in med_ev.items():
            if s2!=sid: continue
            r[f"{mnm}_on"]=int(np.any((sa<=t)&(ea>t)))
            r[f"{mnm}_n24"]=int(np.sum((sa>t-24)&(sa<=t)))
        for mnm in ("norepi","epi","dopa","dobo","vaso","propofol","dex","mido","furosemide"):
            r.setdefault(f"{mnm}_on",0); r.setdefault(f"{mnm}_n24",0)
        r["mannitol_g_cum"]=r.get("mannitol_ie_n24",0); r["hts_meq_cum"]=r.get("hts_ie_n24",0)
        r["mannitol_g_24h"]=r.get("mannitol_ie_n24",0); r["hts_meq_24h"]=r.get("hts_ie_n24",0)
        r["vent_on_24h"]=int(np.any((vh>t-24)&(vh<=t))) if vh is not None else 0
        # UO/液体
        infl=outl=np.nan
        if sid in fl_g:
            h,v=fl_g[sid]; m=h<=t
            infl=float(v[m].sum()) if m.any() else 0.0
            m24=(h>t-24)&(h<=t); r["intake_ml_24h"]=float(v[m24].sum()) if m24.any() else 0.0
        if (sid,"uo") in ser:
            h,v=ser[(sid,"uo")]
            m=h<=t; outl=float(v[m].sum()) if m.any() else 0.0
            for w_,tg in ((24,"24h"),(12,"12h"),(6,"6h")):
                mw=(h>t-w_)&(h<=t)
                tot=float(v[mw].sum()) if mw.any() else np.nan
                r[f"uo_ml_kg_h_{tg}"]=tot/(w_*wk) if np.isfinite(tot) and np.isfinite(wk) and wk>0 else np.nan
            m48=(h>t-48)&(h<=t); r["uo_tot48"]=float(v[m48].sum()) if m48.any() else np.nan
        if np.isfinite(infl) and np.isfinite(outl) and np.isfinite(wk) and wk>0:
            r["net_balance_ml_kg"]=(infl-outl)/wk
        rows.append(r)

F = pd.DataFrame(rows)
F.to_parquet(f"{R}/n5_abi_rolling_features.parquet", index=False)
print(f"ABI features: {F.shape} | pos: {int(F.label.sum())} | cols: {F.shape[1]}")
