# -*- coding: utf-8 -*-
"""N3: M4 ABI 事件轴（完整 KDIGO Cr∪UO）+ 滚动网格（复用 v2 全部冻结规则）"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate
gate(__file__)

R = "E:/TBI subtype/09_tbi_aki/results"
cr = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet").dropna(subset=["valuenum"])
uo = pd.read_parquet(f"{R}/n2_m4_abi_uo_rows.parquet")
coh = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet")
wt = pd.read_parquet(f"{R}/n2_m4_abi_weights.parquet")

# 体重（优先级 226512 > 224639 > 226531，同档取最早）
_PRI = {226512: 0, 224639: 1, 226531: 2}
wt["_p"] = wt.itemid.map(_PRI).fillna(9)
wt1 = (wt.sort_values(["stay_id","_p","hr_from_icu"], kind="mergesort")
         .groupby("stay_id", as_index=False).first()[["stay_id","value"]]
         .rename(columns={"value":"weight_kg"}))

# UO 小时回填桶（6h strict，同 P0-2 v2）
def bucketize_uo(uo, wt):
    uo = uo.sort_values(["stay_id","hr_from_icu"], kind="mergesort")
    rows = []
    for sid, g in uo.groupby("stay_id"):
        if len(g) < 2: continue
        hrs = g.hr_from_icu.to_numpy(float)
        vals = g.uo_signed.to_numpy(float)
        w = float(wt1.set_index("stay_id").weight_kg.get(sid, np.nan))
        if not np.isfinite(w) or w <= 0: continue
        # 小时级回填（>4h 间隔=未监测）
        hour_out = np.zeros(192)
        hour_cov = np.zeros(192, dtype=bool)
        for i in range(len(hrs)):
            h = int(np.clip(hrs[i], 0, 191))
            prev_h = int(np.clip(hrs[i-1], 0, 191)) if i > 0 and hrs[i]-hrs[i-1] <= 4 else max(0, h-1)
            dur = h - prev_h if h > prev_h else 1
            hour_out[h] = vals[i]
            hour_cov[h] = True
        # 6h 桶
        for b in range(32):  # 192h / 6 = 32 桶
            sl = slice(b*6, (b+1)*6)
            cov = hour_cov[sl].sum()
            uo_ml = hour_out[sl].sum()
            if cov >= 6:
                rate = uo_ml / (6 * w)
                rows.append((sid, b, b*6, (b+1)*6, uo_ml, cov, True, rate, rate < 0.5))
    return pd.DataFrame(rows, columns=["stay_id","bucket_idx","t0_hr","t1_hr","uo_ml","cov_hours",
                                        "evaluable_strict","ml_kg_h_strict","below_05_strict"])
uo_bk = bucketize_uo(uo, wt1)
uo_bk.to_parquet(f"{R}/n3_m4_abi_uo_buckets.parquet", index=False)
print(f"UO 桶: {len(uo_bk)} / {uo_bk.stay_id.nunique()} stays")

# ---- 事件轴（完整 KDIGO Cr∪UO + ffill V2 + RRT） ----
FFILL_H, KDIGO_RATIO, W_START, W_END = 96.0, 1.5, 24.0, 168.0

def has_aki_locked(w_vals, w_hrs, base):
    if len(w_vals) == 0 or pd.isna(base): return np.nan
    v, h = w_vals, w_hrs
    r48, j0 = 0.0, 0
    for i in range(len(v)):
        while h[i]-h[j0] > 48: j0 += 1
        r48 = max(r48, v[i]-v[j0:i+1].min())
    return int((v.max()/base >= KDIGO_RATIO) or (r48 >= 0.3))

def ffill_label(g, base, cutoff=None):
    end = cutoff if cutoff is not None else W_END
    if base != base or end <= W_START: return False, np.nan, np.nan
    hr = g.hr.to_numpy(float); val = g.valuenum.to_numpy(float)
    thresh = KDIGO_RATIO * base
    viw = (hr + FFILL_H > W_START) & (hr < end)
    if not viw.any(): return False, np.nan, np.nan
    t_eff = np.maximum(hr, W_START)
    times = [float(t_eff[k]) for k in np.where(viw & (val >= thresh))[0]]
    j0 = 0
    for i in range(len(hr)):
        while hr[i]-hr[j0] > 48: j0 += 1
        if val[i]-val[j0:i+1].min() >= 0.3:
            t = float(max(hr[i], W_START))
            if W_START < t < end: times.append(t)
    if not times: return True, 0, np.nan
    inwin = [t for t in times if t > W_START]
    v2 = int(bool(inwin)) if inwin else np.nan
    v2hr = float(min(inwin)) if inwin else np.nan
    return True, v2, v2hr

# RRT
import duckdb
con = duckdb.connect()
rrt = con.execute(f"""
SELECT pe.stay_id, (epoch(pe.starttime)-epoch(c.intime))/3600.0 AS hr
FROM read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/icu/procedureevents.csv.gz') pe
JOIN read_parquet('{R}/n1_m4_abi_cohort.parquet') c USING(stay_id)
WHERE pe.itemid IN (225441, 740477, 772042) AND pe.stay_id IN (SELECT stay_id FROM read_parquet('{R}/n1_m4_abi_cohort.parquet'))
""").df()
rrt_first = rrt.groupby("stay_id").hr.min()

uo_onset = (uo_bk[(uo_bk.bucket_idx >= 4) & uo_bk.evaluable_strict & uo_bk.below_05_strict]
            .groupby("stay_id").t0_hr.min().rename("uo_onset_hr"))
uo_prev24 = set(uo_bk[(uo_bk.bucket_idx <= 3) & uo_bk.evaluable_strict & uo_bk.below_05_strict].stay_id)

rows_ax, rows_fin = [], []
gmap = dict(tuple(cr.groupby("stay_id")))
for sid in coh.stay_id:
    g = gmap.get(sid)
    if g is None:
        rows_ax.append((sid, np.nan, "no_cr", np.nan)); rows_fin.append((sid, np.nan, np.nan, "no_cr"))
        continue
    g = g.sort_values(["hr","valuenum"], kind="mergesort")
    pre = g[g.hr < 0]; win24 = g[(g.hr >= 0) & (g.hr <= 24)]; win7 = g[(g.hr > 24) & (g.hr <= 168)]
    if len(win24): base, br = float(win24.valuenum.min()), "min24"
    elif len(pre): base, br = float(pre.valuenum.min()), "preicu"
    else: base, br = np.nan, "none"
    aki24 = has_aki_locked(win24.valuenum.values, win24.hr.values.astype(float), base)
    rrt_hr = float(rrt_first[sid]) if sid in rrt_first.index else np.nan
    rows_ax.append((sid, base, br, aki24))
    if base != base or pd.isna(aki24) or aki24 == 1:
        rows_fin.append((sid, np.nan, np.nan, "prevalent_or_no_base")); continue
    uo_on = float(uo_onset[sid]) if sid in uo_onset.index else np.nan
    onset = min([x for x in [float('inf') if pd.isna(x) else x for x in [rows_fin[-1][2] if False else np.nan, uo_on]] if pd.notna(x)], default=np.nan)
    # 简化：先跑 Cr-only 终点，UO 并集后续加
    ev, v2, v2hr = ffill_label(g, base)
    if not ev: rows_fin.append((sid, np.nan, np.nan, "no_eval"))
    elif pd.isna(v2) and ev: rows_fin.append((sid, np.nan, np.nan, "carryin"))
    elif rrt_hr == rrt_hr and rrt_hr <= W_START: rows_fin.append((sid, np.nan, np.nan, "rrt_pre"))
    else:
        cutoff = rrt_hr if (rrt_hr == rrt_hr and rrt_hr <= W_END) else None
        _, v2c, vhrc = ffill_label(g, base, cutoff)
        if pd.isna(v2c): rows_fin.append((sid, np.nan, np.nan, "rrt_censor"))
        else: rows_fin.append((sid, float(v2c), float(vhrc), "ok" if v2c == 1 else "ok0"))

AX = pd.DataFrame(rows_ax, columns=["stay_id","cr_base","base_branch","aki24"])
FIN = pd.DataFrame(rows_fin, columns=["stay_id","label","event_hr","censor"])
FIN.loc[FIN.label != 1, "event_hr"] = np.nan

# UO 并集 onset
onset_combined = FIN.merge(uo_onset.rename("uo_hr"), on="stay_id", how="left")
onset_combined["onset_hr"] = onset_combined[["event_hr","uo_hr"]].min(axis=1)

# UO prev24 排除 + 合并 onset
FIN["uo_prev"] = FIN.stay_id.isin(uo_prev24)
# onset_hr = min(Cr事件时刻, UO首桶时刻)
FIN = FIN.merge(uo_onset.rename("uo_hr"), on="stay_id", how="left")
FIN["onset_hr"] = FIN[["event_hr","uo_hr"]].min(axis=1)

# 网格
coh_los = (coh.set_index("stay_id").outtime - coh.set_index("stay_id").intime).dt.total_seconds() / 3600.0
rows_grid = []
kept = FIN[(FIN.censor.isin(["ok","ok0"])) & (~FIN.uo_prev)]
for _, r in kept.iterrows():
    onset = r.onset_hr if pd.notna(r.onset_hr) else np.inf
    tmax = min(coh_los.get(r.stay_id, 168) - 6.0, 168.0)
    for t in np.arange(24.0, tmax + 1e-9, 6.0):
        if t >= onset: break
        rows_grid.append((r.stay_id, t, int(0 < onset - t <= 48.0), onset if np.isfinite(onset) else np.nan))
grid = pd.DataFrame(rows_grid, columns=["stay_id","t_hr","label","onset_hr"])
grid.to_parquet(f"{R}/n3_m4_abi_grid.parquet", index=False)

uo_only = int(((kept.label != 1) & kept.onset_hr.notna()).sum())
rep = {"n_stays": len(coh), "at_risk": int(len(kept)), "onset_cr": int((kept.label==1).sum()),
       "onset_uo_only": uo_only,
       "grid": {"checkpoints": len(grid), "positive": int(grid.label.sum()), "patients": int(grid.stay_id.nunique())}}
with open(f"{R}/n3_m4_abi_grid.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
