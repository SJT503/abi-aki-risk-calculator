# -*- coding: utf-8 -*-
"""N8-3: eICU ABI 事件轴 + 滚动网格
事件轴 = p1_3 冻结规则逐字移植（min24 基线→pre-ICU 回退 + 完整 KDIGO Cr 判据 +
ffill≤96h 仅窗内证据 + carry-in-only 判 prevalent + RRT 剔除/删失，hr∈(-24,168]）
UO 口径 = r4_2 / SAP D26（eICU 班次级记录：行率= val/6h/体重，桶有≥1条尿液行即可评估；
         M4 为严格 6/6h 覆盖——粒度差异披露）
网格 = 与 M4 r1_2/n3 同构：6h 步长，t∈[24, min(los-6,168)]，label=未来48h内 onset
输出：results/n8_3_eicu_abi_event_axis.parquet + n8_3_eicu_abi_grid.parquet + .json
"""
import json
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

R = "E:/TBI subtype/09_tbi_aki/results"
E = "E:/TBI subtype/data/eicu-crd-2.0"
FFILL_H, KDIGO_RATIO, W_START, W_END = 96.0, 1.5, 24.0, 168.0

labs = pd.read_parquet(f"{R}/n8_2_labs_series.parquet")
cr = labs[(labs.labname == "creatinine") & (labs.labtypeid == 1)
          & (labs.hr > -24) & (labs.hr <= 168)].dropna(subset=["labresult"])
cr = cr[["sid", "hr", "labresult"]].rename(columns={"labresult": "valuenum"})
print(f"Cr rows: {len(cr)} / {cr.sid.nunique()} stays")

rrt_raw = pd.read_parquet(f"{R}/n8_2_rrt_treatments.parquet")
rrt_delivery = rrt_raw[~rrt_raw.s.str.contains("insertion")]
rrt_first = rrt_delivery.groupby("sid").hr.min()
print(f"RRT delivery: {rrt_delivery.sid.nunique()} stays")


def has_aki_locked(w_vals, w_hrs, base):
    if len(w_vals) == 0 or pd.isna(base):
        return np.nan
    v, h = w_vals, w_hrs
    r48 = 0.0
    j0 = 0
    for i in range(len(v)):
        while h[i] - h[j0] > 48:
            j0 += 1
        r48 = max(r48, v[i] - v[j0:i + 1].min())
    return int((v.max() / base >= KDIGO_RATIO) or (r48 >= 0.3))


def ffill_label(g, base, rrt_cutoff_hr=None):
    end = rrt_cutoff_hr if rrt_cutoff_hr is not None else W_END
    if base != base or end <= W_START:
        return False, np.nan, np.nan, False, np.nan, np.nan
    hr = g.hr.to_numpy(dtype=float)
    val = g.valuenum.to_numpy(dtype=float)
    thresh = KDIGO_RATIO * base
    valid_into_win = (hr + FFILL_H > W_START) & (hr < end)
    if not valid_into_win.any():
        return False, np.nan, np.nan, False, np.nan, np.nan
    t_eff = np.maximum(hr, W_START)
    times = [float(t_eff[k]) for k in np.where(valid_into_win & (val >= thresh))[0]]
    o = np.argsort(hr, kind="stable")
    hr_s, val_s = hr[o], val[o]
    j0 = 0
    for i in range(len(hr_s)):
        while hr_s[i] - hr_s[j0] > 48:
            j0 += 1
        if val_s[i] - val_s[j0:i + 1].min() >= 0.3:
            t = float(max(hr_s[i], W_START))
            if W_START < t < end:
                times.append(t)
    if not times:
        return True, 0, np.nan, False, 0, np.nan
    t_min = float(min(times))
    inwin = [t for t in times if t > W_START]
    v2_lab = int(bool(inwin)) if inwin else np.nan
    v2_hr = float(min(inwin)) if inwin else np.nan
    return True, 1, t_min, bool(t_min == W_START), v2_lab, v2_hr


rows_ax, rows_fin = [], []
gmap = dict(tuple(cr.groupby("sid")))
for sid, g in gmap.items():
    g = g.sort_values(["hr", "valuenum"], kind="mergesort")
    pre = g[g.hr < 0]
    win24 = g[(g.hr >= 0) & (g.hr <= 24)]
    if len(win24):
        base, branch = float(win24.valuenum.min()), "min24"
    elif len(pre):
        base, branch = float(pre.valuenum.min()), "preicu_fallback"
    else:
        base, branch = np.nan, "none"
    aki24 = has_aki_locked(win24.valuenum.values, win24.hr.values.astype(float), base)
    rrt_hr = float(rrt_first.get(sid, np.nan))
    rows_ax.append((sid, base, branch, aki24, rrt_hr))
    if base != base or pd.isna(aki24) or aki24 == 1:
        rows_fin.append((sid, np.nan, np.nan, "prevalent_or_no_base"))
        continue
    ev, v1, v1hr, cin, v2n, _ = ffill_label(g, base)
    if not ev:
        rows_fin.append((sid, np.nan, np.nan, "no_evaluable_evidence"))
        continue
    if pd.isna(v2n) and v1 == 1:
        rows_fin.append((sid, np.nan, np.nan, "carryin_prevalent"))
        continue
    if rrt_hr == rrt_hr and rrt_hr <= W_START:
        rows_fin.append((sid, np.nan, np.nan, "rrt_before_landmark"))
        continue
    cutoff = rrt_hr if (rrt_hr == rrt_hr and rrt_hr <= W_END) else None
    _, _, _, _, v2, v2hr = ffill_label(g, base, rrt_cutoff_hr=cutoff)
    if pd.isna(v2):
        rows_fin.append((sid, np.nan, np.nan, "rrt_censor"))
    else:
        reason = "rrt_truncated_followup" if (cutoff is not None and v2 == 0) else "ok"
        rows_fin.append((sid, float(v2), float(v2hr), reason))

AX = pd.DataFrame(rows_ax, columns=["stay_id", "cr_base", "base_branch", "aki24_prevalent", "rrt_first_hr"])
FIN = pd.DataFrame(rows_fin, columns=["stay_id", "cr_label_final", "event_hr_final", "censor_reason"])

# ---- UO（M4 n3 小时覆盖桶化规则逐字移植，修正 D26 行级÷6 偏差） ----
# 实证诊断（2026-09-21）：eICU 尿行=增量制，行值覆盖间隔=与前一行的时间差
# （1h 间隔→中位 65mL，5-7h 间隔→中位 300mL），行级 val/6/wt 对 ~90% 医院系统性低估 6×
# → 假性少尿（旧规则 5,770 误排除）。修复=镜像 M4 冻结规则：
#   行落小时桶（同小时 last-wins，多行桶仅 1%）→ 6h 桶 cov>=6 才可评估 → 桶和/(6×wt)
uof = pd.read_parquet(f"{R}/n8_2_uo_fluid_series.parquet")
con = duckdb.connect()
wt = con.execute(f"""
SELECT p.patientunitstayid sid, p.admissionweight FROM read_csv_auto('{E}/patient.csv.gz') p
WHERE p.admissionweight IS NOT NULL AND p.admissionweight BETWEEN 30 AND 300
""").df().set_index("sid").admissionweight.to_dict()
uo = uof[uof.cellpath.str.contains("urine") | uof.celllabel.str.contains("urine")].copy()
uo = uo[(uo.hr >= 0) & (uo.hr < 192)].sort_values(["sid", "hr"], kind="mergesort")
uo_rows = []
n_evalu = 0
for sid, g in uo.groupby("sid"):
    w = float(wt.get(sid, np.nan))
    if not np.isfinite(w) or w <= 0:
        continue
    hrs = g.hr.to_numpy(float)
    vals = g.val.to_numpy(float)
    hour_out = np.zeros(192)
    hour_cov = np.zeros(192, dtype=bool)
    for i in range(len(hrs)):
        h = int(np.clip(hrs[i], 0, 191))
        hour_out[h] = vals[i]
        hour_cov[h] = True
    for b in range(32):
        sl = slice(b * 6, (b + 1) * 6)
        cov = int(hour_cov[sl].sum())
        uo_ml = float(hour_out[sl].sum())
        if cov >= 6:
            rate = uo_ml / (6 * w)
            uo_rows.append((sid, b, rate, rate < 0.5))
            n_evalu += 1
UB = pd.DataFrame(uo_rows, columns=["sid", "bucket_idx", "rate", "below"])
print(f"UO 可评估桶: {n_evalu} / {UB.sid.nunique()} stays（cov>=6 严格规则，稀疏记录医院不可评估）")
uo_onset = UB[UB.below & (UB.bucket_idx >= 4)].groupby("sid").bucket_idx.min() * 6
uo_prev24 = set(UB[UB.below & (UB.bucket_idx <= 3)].sid)

coh = pd.read_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet")
d = AX.merge(FIN, on="stay_id").merge(coh[["stay_id", "los_h"]], on="stay_id", how="inner")
d = d.merge(uo_onset.rename("uo_onset_hr"), left_on="stay_id", right_index=True, how="left")
d["uo_prev24"] = d.stay_id.isin(uo_prev24)
excl_cr = d.censor_reason.isin(["prevalent_or_no_base", "carryin_prevalent",
                                "rrt_before_landmark", "no_evaluable_evidence"])
kept = d[~excl_cr & ~d.uo_prev24].copy()
kept["onset_hr"] = kept[["event_hr_final", "uo_onset_hr"]].min(axis=1)

rows = []
for _, r in kept.iterrows():
    onset = r.onset_hr if pd.notna(r.onset_hr) else np.inf
    tmax = min(r.los_h - 6.0, 168.0)
    for t in np.arange(24.0, tmax + 1e-9, 6.0):
        if t >= onset:
            break
        rows.append((r.stay_id, t, int(0 < onset - t <= 48.0), onset if np.isfinite(onset) else np.nan))
grid = pd.DataFrame(rows, columns=["stay_id", "t_hr", "label", "onset_hr"])
grid.to_parquet(f"{R}/n8_3_eicu_abi_grid.parquet", index=False)
d.to_parquet(f"{R}/n8_3_eicu_abi_event_axis.parquet", index=False)

rep = {"probe": "N8-3 eICU ABI event axis + grid", "date": "2026-09-21",
       "n_with_cr": int(len(AX)), "aki24_prevalent": int(AX.aki24_prevalent.sum()),
       "censor_reasons": d.censor_reason.value_counts().to_dict(),
       "excl_uo_prev24": int((~excl_cr & d.uo_prev24).sum()),
       "at_risk": int(len(kept)),
       "onset_cr": int((kept.cr_label_final == 1).sum()),
       "onset_uo_only": int((kept.event_hr_final.isna() & kept.uo_onset_hr.notna()).sum()),
       "grid": {"checkpoints": len(grid), "positive": int(grid.label.sum()),
                "pos_rate": round(float(grid.label.mean()), 4),
                "patients": int(grid.stay_id.nunique())},
       "uo_rule": "M4 n3 小时覆盖桶化逐字移植（cov>=6 严格；行级÷6 旧 D26 规则已废弃——实证：行值覆盖=与前一行间隔，÷6 低估 6×致假性少尿）",
       "uo_evaluable_buckets": int(n_evalu)}
with open(f"{R}/n8_3_eicu_abi_grid.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
