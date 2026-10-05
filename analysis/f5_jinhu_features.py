# -*- coding: utf-8 -*-
"""f5: JinhuaNSICU event axis + rolling features (n5 semantics port, 2026-09-30).
- ICU-anchored hours: t = (time_base_min - icu_start)/60
- Event axis (eICU n8_3 template, Cr-only): base = min raw Cr in [0,24] ICU-h else pre-ICU min;
  bin-max series (6h bins, -24..168) with <=96h carry; prevalent (crossing <=24h) excluded;
  RRT <=24h excluded, RRT in (24,168] truncates grid; onset = first carried bin >24h crossing
  fold>=1.5 or 48h-rise>=0.3.
- Grid: t in {24,...,min(los-6,168)} step 6, stop before onset / RRT truncation.
- Features: n5 wf() verbatim semantics (24h half-open window: last/min/max/delta/slope/n/
  h_last/sd/cv + locf/locf_age; 48h family for cr/bun/hr_rate/sbp; med n24 counts;
  vent_on_24h; derived cr_ratio_base/cr_delta_since_adm). GCS/UO/intake/net left as NaN.
Output: results/jinhu/jinhu_features.parquet (+ f5_report.json)
"""
import json
import numpy as np, pandas as pd

J = "results/jinhu"
coh = pd.read_parquet(f"{J}/jinhu_cohort.parquet")
labs = pd.read_parquet(f"{J}/jinhu_labs.parquet")
vits = pd.read_parquet(f"{J}/jinhu_vitals.parquet")
meds = pd.read_parquet(f"{J}/jinhu_meds.parquet")
vent = pd.read_parquet(f"{J}/jinhu_vent.parquet")
rrt = pd.read_parquet(f"{J}/jinhu_rrt.parquet")

idx = coh.set_index("hadm_id")
t0 = idx.icu_start.to_dict()
los = idx.los_icu_h.to_dict()

def to_icu_h(df, col):
    return (df[col] - df.hadm_id.map(t0)) / 60.0

labs["hr"] = to_icu_h(labs, "tmin")
vits["hr"] = to_icu_h(vits, "tmin")
meds["sh"] = to_icu_h(meds, "stmin")
vent["sh"] = to_icu_h(vent, "stmin")
rrt["sh"] = to_icu_h(rrt, "stmin")

# ============ event axis (Cr-only, eICU n8_3 template) ============
cr_all = labs[labs.fam == "cr"]
cr_ax = cr_all[(cr_all.hr > -24) & (cr_all.hr <= 168) & cr_all.val.between(0.1, 30)]
FFILL_H, W_START, W_END = 96.0, 24.0, 168.0
rrt_first = rrt.groupby("hadm_id").sh.min()

axis = {}
for hadm, g in cr_ax.groupby("hadm_id"):
    g = g.sort_values(["hr", "val"], kind="mergesort")
    pre = g[g.hr < 0]; win24 = g[(g.hr >= 0) & (g.hr <= 24)]
    if len(win24): base = float(win24.val.min())
    elif len(pre): base = float(pre.val.min())
    else: base = np.nan
    if base != base or base < 0.1:
        axis[hadm] = ("no_base", np.nan, np.nan); continue
    # prevalent: criteria met WITHIN the first-24h window itself (n8_3 aki24_prevalent semantics)
    wv, wh = win24.val.to_numpy(float), win24.hr.to_numpy(float)
    if len(wv) == 0:
        pass  # base from pre-ICU only; no in-window evidence to test
    else:
        r48 = 0.0; j0 = 0
        for i in range(len(wv)):
            while wh[i] - wh[j0] > 48: j0 += 1
            r48 = max(r48, wv[i] - wv[j0:i + 1].min())
        if (wv.max() / base >= 1.5) or (len(wv) >= 2 and r48 >= 0.3):
            axis[hadm] = ("prevalent_24h", base, np.nan); continue
    # onset on RAW bin-max crossing (self-consistent with the v2 label machinery;
    # verified against eICU: grid ends AT the first crossing bin, 281/300)
    hr = g.hr.to_numpy(float); val = g.val.to_numpy(float)
    bins = np.arange(0, 174, 6.0)
    tb = bins[:-1]
    onset = np.nan
    for i in range(len(tb)):
        m = (hr >= bins[i]) & (hr < bins[i + 1])
        if not m.any(): continue
        bmax = float(val[m].max())
        tr_min = np.nan
        for k in range(max(0, i - 8), i + 1):
            mk = (hr >= bins[k]) & (hr < bins[k + 1])
            if mk.any():
                v = float(val[mk].max())
                tr_min = v if np.isnan(tr_min) else min(tr_min, v)
        rise48 = bmax - tr_min if np.isfinite(tr_min) else np.nan
        if tb[i] < W_START: continue
        if (bmax / base >= 1.5) or (np.isfinite(rise48) and rise48 >= 0.3):
            onset = float(tb[i]); break
    axis[hadm] = ("ok", base, onset)

AX = pd.DataFrame([(k,) + v for k, v in axis.items()], columns=["hadm_id", "ax_status", "cr_base_ax", "onset_hr"])
rrt_map = rrt_first.to_dict()
keep = AX[AX.ax_status == "ok"].copy()
keep["rrt_hr"] = keep.hadm_id.map(rrt_map)
n_excl_rrt = int((keep.rrt_hr <= 24).sum())
keep = keep[~(keep.rrt_hr <= 24)]
print(f"axis: {len(AX)} with Cr | ok {len(keep)} | {dict(AX.ax_status.value_counts())} | rrt<=24h excluded {n_excl_rrt}")

# ============ grid (v2 semantics: ladder to min(los-6,168); INCLUDE onset bin then stop; stop before RRT) ============
rows_grid = []
for _, r in keep.iterrows():
    onset = r.onset_hr if np.isfinite(r.onset_hr) else np.inf
    tmax = min(los[r.hadm_id] - 6.0, 168.0)
    rrt_lim = r.rrt_hr if np.isfinite(r.rrt_hr) else np.inf
    for t in np.arange(24.0, tmax + 1e-9, 6.0):
        if t > onset: break          # include the onset bin itself (eICU-verified rule)
        if t >= rrt_lim: break       # stop before RRT delivery
        rows_grid.append((r.hadm_id, float(t)))
grid = pd.DataFrame(rows_grid, columns=["hadm_id", "t_hr"])
print(f"grid: {len(grid)} checkpoints / {grid.hadm_id.nunique()} stays")

# ============ features (n5 wf semantics) ============
def wf(hrs, vals, t, w=24.0):
    m = (hrs > t - w) & (hrs <= t)
    hh, vv = hrs[m], vals[m]
    if len(vv) == 0: return None
    o = np.argsort(hh, kind="mergesort"); hh, vv = hh[o], vv[o]
    slope = np.polyfit(hh, vv, 1)[0] if len(vv) >= 2 and np.ptp(hh) > 0 else 0.0
    sd = float(np.std(vv, ddof=1)) if len(vv) >= 2 else np.nan
    mu = float(np.mean(vv))
    cv = float(np.std(vv, ddof=1) / abs(mu)) if len(vv) >= 2 and abs(mu) > 1e-9 else np.nan
    return {"last": vv[-1], "min": vv.min(), "max": vv.max(), "delta": vv[-1] - vv[0],
            "slope": slope, "n": len(vv), "h_last": t - hh[-1], "sd": sd, "cv": cv}

ser = {}
for fam, g in pd.concat([labs, vits]).groupby("fam"):
    for hadm, gg in g.groupby("hadm_id"):
        gg = gg.sort_values("hr", kind="mergesort")
        ser[(hadm, fam)] = (gg.hr.to_numpy(float), gg.val.to_numpy(float))
cr_full = cr_ax.copy()
for hadm, gg in cr_full.groupby("hadm_id"):
    gg = gg.sort_values("hr", kind="mergesort")
    ser[(hadm, "cr_full")] = (gg.hr.to_numpy(float), gg.val.to_numpy(float))

med_ev = {}
for fam, g in meds.dropna(subset=["sh"]).groupby("fam"):
    for hadm, gg in g.groupby("hadm_id"):
        med_ev[(hadm, fam)] = gg.sh.to_numpy(float)
vent_g = {hadm: np.sort(g.sh.to_numpy(float)) for hadm, g in vent.dropna(subset=["sh"]).groupby("hadm_id")}

grid_grouped = list(grid.groupby("hadm_id"))
out = []
for pi, (hadm, g) in enumerate(grid_grouped):
    if pi % 200 == 0: print(f"  {pi}/{len(grid_grouped)}", flush=True)
    cvd = float(idx.charlson.get(hadm, 0))
    base = float(AX.set_index("hadm_id").cr_base_ax.get(hadm, np.nan))
    vh = vent_g.get(hadm)
    for t in g.t_hr:
        t = float(t)
        r = {"stay_id": hadm, "t_hr": t,
             "age": float(idx.age.get(hadm, np.nan)),
             "male": int(idx.male.get(hadm, 0)),
             "charlson": cvd, "cr_base": base}
        for nm in sorted({n for (s, n) in ser if s == hadm}):
            hrs_s, vals_s = ser[(hadm, nm)]
            f = wf(hrs_s, vals_s, t)
            if f:
                for k, v in f.items(): r[f"{nm}_{k}"] = v
            j = np.searchsorted(hrs_s, t, side="right") - 1
            if j >= 0:
                r[f"{nm}_locf"] = float(vals_s[j]); r[f"{nm}_locf_age"] = float(t - hrs_s[j])
        if "cr_last" in r and pd.notna(base):
            r["cr_ratio_base"] = r["cr_last"] / base; r["cr_delta_since_adm"] = r["cr_last"] - base
        for nm in ("cr", "bun", "hr_rate", "sbp"):
            if (hadm, nm) in ser:
                h, v = ser[(hadm, nm)]
                m48 = (h > t - 48) & (h <= t)
                if m48.any():
                    hh, vv = h[m48], v[m48]
                    o = np.argsort(hh, kind="mergesort"); hh, vv = hh[o], vv[o]
                    r[f"{nm}_last48"] = float(vv[-1]); r[f"{nm}_delta48"] = float(vv[-1] - vv[0])
                    r[f"{nm}_slope48"] = float(np.polyfit(hh, vv, 1)[0]) if len(vv) >= 2 and np.ptp(hh) > 0 else 0.0
                    r[f"{nm}_sd48"] = float(np.std(vv, ddof=1)) if len(vv) >= 2 else np.nan
        for mnm in ("norepi", "propofol", "dex", "mido", "hts"):
            sa = med_ev.get((hadm, mnm))
            r[f"{mnm}_n24"] = int(np.sum((sa > t - 24) & (sa <= t))) if sa is not None else 0
        r["hts_meq_24h"] = r.get("hts_n24", 0)
        r["vent_on_24h"] = int(np.any((vh > t - 24) & (vh <= t))) if vh is not None else 0
        out.append(r)

F = pd.DataFrame(out)
# add the 43 Jinhua-unavailable features as explicit NaN columns
import json as _json
SEL = _json.load(open("results/_reselected_features.json"))
cov = {r_["feature"]: r_["status"] for r_ in _json.load(open("results/jinhu_268_coverage.json"))}
for f in SEL:
    if cov[f] == "missing" and f not in F.columns:
        F[f] = np.nan
missing_cols = [f for f in SEL if f not in F.columns]
assert not missing_cols, f"features still absent: {missing_cols}"
F.to_parquet(f"{J}/jinhu_features.parquet", index=False)
nn = F[SEL].isna().mean()
rep = {"axis_status": AX.ax_status.value_counts().to_dict(), "rrt_excluded": n_excl_rrt,
       "grid_checkpoints": len(F), "grid_stays": int(F.stay_id.nunique()),
       "mean_nan_frac_all268": round(float(nn.mean()), 3),
       "mean_nan_frac_225computable": round(float(nn[[f for f in SEL if cov[f] == 'ok']].mean()), 3),
       "top10_nan": nn.sort_values(ascending=False).head(10).round(3).to_dict()}
json.dump(rep, open(f"{J}/f5_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
