# -*- coding: utf-8 -*-
"""N8-4: eICU ABI 滚动特征（对齐 n5/n6 的 271 列 schema；应用时 reindex 到冻结 feats）
来源映射（labtypeid 枚举见 preflight_n8）：
  type1: creatinine→cr sodium→na potassium→k glucose→glu BUN→bun chloride→cl
         bicarbonate→hco3 'anion gap'→ag 'WBC x 1000'→wbc Hematocrit→hct
         magnesium→mg phosphate→phos lactate→lactate
  type3: Hgb→hb RDW→rdw 'PT - INR'→inr
  type7: paO2→po2 'Base Excess'→base_excess
  vitals: heartrate→hr_rate systemicsystolic→sbp respiration→rr sao2→spo2
  GCS: nurseCharting motor/eye/verbal；UO/入量: intakeOutput（hr≥0 截断=M4 对齐）
已知口径差异（披露进 JSON）：med _on=曾启动≤t（eICU 只有起始时刻；M4 为 t 时点在输）；
  mannitol/hts 剂量列=0（输注速率 VARCHAR 不解析）；temp 不在 271 feats（M4 筛选已剔除）
输出：results/n8_4_eicu_abi_features.parquet
"""
import json
import os
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

R = "E:/TBI subtype/09_tbi_aki/results"
E = "E:/TBI subtype/data/eicu-crd-2.0"
grid = pd.read_parquet(f"{R}/n8_3_eicu_abi_grid.parquet")
coh = pd.read_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet").set_index("stay_id")
ax = pd.read_parquet(f"{R}/n8_3_eicu_abi_event_axis.parquet").set_index("stay_id")
labs = pd.read_parquet(f"{R}/n8_2_labs_series.parquet")
vit = pd.read_parquet(f"{R}/n8_2_vitals_series.parquet")
gcs = pd.read_parquet(f"{R}/n8_2_gcs_temp_series.parquet")
meds = pd.read_parquet(f"{R}/n8_2_meds_series.parquet")
uof = pd.read_parquet(f"{R}/n8_2_uo_fluid_series.parquet")
vent = pd.read_parquet(f"{R}/n8_2_vent_series.parquet")

con = duckdb.connect()
wt = con.execute(f"""
SELECT p.patientunitstayid sid, p.admissionweight FROM read_csv_auto('{E}/patient.csv.gz') p
WHERE p.admissionweight IS NOT NULL AND p.admissionweight BETWEEN 30 AND 300
""").df().set_index("sid").admissionweight

LAB_MAP = {("creatinine", 1): "cr", ("sodium", 1): "na", ("potassium", 1): "k",
           ("glucose", 1): "glu", ("BUN", 1): "bun", ("chloride", 1): "cl",
           ("bicarbonate", 1): "hco3", ("anion gap", 1): "ag",
           ("magnesium", 1): "mg", ("phosphate", 1): "phos", ("lactate", 1): "lactate",
           ("Hgb", 3): "hb", ("RDW", 3): "rdw", ("PT - INR", 3): "inr",
           ("Hct", 3): "hct", ("WBC x 1000", 3): "wbc",
           ("paO2", 7): "po2", ("Base Excess", 7): "base_excess"}

ser = {}
for (nm_raw, lt), nm in LAB_MAP.items():
    sub = labs[(labs.labname == nm_raw) & (labs.labtypeid == lt)]
    for sid, gg in sub.groupby("sid"):
        gg = gg.sort_values("hr", kind="mergesort")
        ser.setdefault((sid, nm), (gg.hr.to_numpy(float), gg.labresult.to_numpy(float)))
for col, nm in [("heartrate", "hr_rate"), ("systemicsystolic", "sbp"),
                ("respiration", "rr"), ("sao2", "spo2")]:
    sub = vit[vit[col].notna()]
    for sid, gg in sub.groupby("sid"):
        gg = gg.sort_values("hr", kind="mergesort")
        ser.setdefault((sid, nm), (gg.hr.to_numpy(float), gg[col].to_numpy(float)))
# sbp 合并无创（vitalAperiodic，99% 覆盖）——M4 220179=NBP 语义对齐（2026-09-21 补）
nbp = pd.read_parquet(f"{R}/n8_2_aperiodic_sbp.parquet")
for sid, gg in nbp.groupby("sid"):
    gg = gg.sort_values("hr", kind="mergesort")
    if (sid, "sbp") in ser:
        h0, v0 = ser[(sid, "sbp")]
        h1 = np.concatenate([h0, gg.hr.to_numpy(float)])
        v1 = np.concatenate([v0, gg.val.to_numpy(float)])
        o = np.argsort(h1, kind="mergesort")
        ser[(sid, "sbp")] = (h1[o], v1[o])
    else:
        ser[(sid, "sbp")] = (gg.hr.to_numpy(float), gg.val.to_numpy(float))
GCS_MAP = {"best motor response": "gcs_motor", "motor response": "gcs_motor",
           "best eye response": "gcs_eye", "eye opening": "gcs_eye",
           "best verbal response": "gcs_verbal", "verbal response": "gcs_verbal"}
# eICU 本副本 GCS 为文本值（2026-09-21 全值域枚举实测）；映射表覆盖枚举到的每个值
GCS_TXT = {
    "obeys simple commands": 6, "localizes to noxious stimuli": 5, "withdraws": 4,
    "abnormal flexion": 3, "abnormal extension": 2, "flaccid": 1,
    "oriented": 5, "clearly oriented/can indicate needs": 5,
    "confused": 4, "inappropriate words": 3, "incomprehensible sounds": 2,
    "none": 1, "clearly unresponsive": 1,
    "trached or intubated": np.nan,  # NT → 缺失
    "orientation/ability to communicate questionable": np.nan,
}
_gcs_unmapped = {}


def gcs_parse(v):
    s = str(v).strip()
    if s in GCS_TXT:
        return GCS_TXT[s]
    m = re.match(r"^([1-6])-->", s)
    if m:
        return float(m.group(1))
    try:
        f = float(s)
        return f if 1 <= f <= 6 else np.nan
    except ValueError:
        _gcs_unmapped[s] = _gcs_unmapped.get(s, 0) + 1
        return np.nan


for lab_raw, nm in GCS_MAP.items():
    sub = gcs[gcs.lab.str.lower() == lab_raw].copy()
    sub["val"] = sub.val.map(gcs_parse)
    sub = sub.dropna(subset=["val"])
    for sid, gg in sub.groupby("sid"):
        gg = gg.sort_values("hr", kind="mergesort")
        # 同 stay 同变量已有 Best 变体则不覆盖（Best 优先，后写者跳过）
        if (sid, nm) in ser and not lab_raw.startswith("best"):
            continue
        ser[(sid, nm)] = (gg.hr.to_numpy(float), gg.val.to_numpy(float))
if _gcs_unmapped:
    print("GCS 未映射值:", _gcs_unmapped)

uo = uof[uof.cellpath.str.contains("urine") | uof.celllabel.str.contains("urine")].copy()
uo = uo[uo.hr >= 0]  # M4 对齐：UO 从入科起算
for sid, gg in uo.groupby("sid"):
    gg = gg.sort_values("hr", kind="mergesort")
    ser.setdefault((sid, "uo"), (gg.hr.to_numpy(float), gg.val.to_numpy(float)))
cr_full = labs[(labs.labname == "creatinine") & (labs.labtypeid == 1)]
for sid, gg in cr_full.groupby("sid"):
    gg = gg.sort_values("hr", kind="mergesort")
    ser.setdefault((sid, "cr_full"), (gg.hr.to_numpy(float), gg.labresult.to_numpy(float)))

intake = uof[uof.cellpath.str.contains("intake") & (uof.hr >= 0)]  # 实测路径 'flowsheet|...|i&o|intake (ml)|...'
int_g = {sid: (g.hr.to_numpy(float), g.val.to_numpy(float)) for sid, g in intake.groupby("sid")}
vent_g = {sid: np.sort(g.hr.to_numpy(float)) for sid, g in vent.groupby("sid")}

MED_CLASS = [("norepinephrine", "norepi"), ("epinephrine", "epi"), ("dopamine", "dopa"),
             ("dobutamine", "dobo"), ("vasopressin", "vaso"), ("propofol", "propofol"),
             ("dexmedetomidine", "dex"), ("midazolam", "mido"), ("furosemide", "furosemide"),
             ("mannitol", "mannitol"), ("3% saline", "hts"), ("hypertonic", "hts2")]
med_ev = {}
for pat, nm in MED_CLASS:
    sub = meds[meds.drug.str.contains(pat, na=False)]
    for sid, gg in sub.groupby("sid"):
        med_ev.setdefault((sid, nm), gg.start_hr.to_numpy(float))

cov_n = {}


def win_feats(hrs, vals, t, w=24.0):
    """D33 变异度族：sd/cv（与 M4 n5 同构；n>=2 才算；|mean|<1e-9→cv=NaN）"""
    m = (hrs > t - w) & (hrs <= t)
    hh, vv = hrs[m], vals[m]
    if len(vv) == 0:
        return None
    o = np.argsort(hh, kind="mergesort")
    hh, vv = hh[o], vv[o]
    slope = np.polyfit(hh, vv, 1)[0] if len(vv) >= 2 and np.ptp(hh) > 0 else 0.0
    sd = float(np.std(vv, ddof=1)) if len(vv) >= 2 else np.nan
    mu = float(np.mean(vv))
    cv = float(np.std(vv, ddof=1) / abs(mu)) if len(vv) >= 2 and abs(mu) > 1e-9 else np.nan
    return {"last": vv[-1], "min": vv.min(), "max": vv.max(), "delta": vv[-1] - vv[0],
            "slope": slope, "n": len(vv), "h_last": t - hh[-1], "sd": sd, "cv": cv}


print("Building features for", grid.stay_id.nunique(), "patients /", len(grid), "checkpoints...")
# 预建 sid→变量名索引（避免逐患者全字典扫描）
by_sid = {}
for (s, n) in ser:
    by_sid.setdefault(s, []).append(n)
med_by_sid = {}
for (s2, mnm) in med_ev:
    med_by_sid.setdefault(s2, []).append(mnm)
t0 = time.time()
rows = []
for pi, (sid, g) in enumerate(grid.groupby("stay_id")):
    if pi % 1000 == 0:
        print(f"  {pi}/{grid.stay_id.nunique()} ({time.time()-t0:.0f}s)", flush=True)
    st = coh.loc[sid]
    wk = float(wt.get(sid, np.nan))
    base = float(ax.cr_base.get(sid, np.nan)) if sid in ax.index else np.nan
    vh = vent_g.get(sid)
    for t, label in zip(g.t_hr, g.label):
        t = float(t)
        r = {"stay_id": sid, "t_hr": t, "label": label,
             "age": st.age, "male": st.male, "charlson": st.charlson, "cr_base": base}
        for nm in by_sid.get(sid, []):
            hrs_s, vals_s = ser[(sid, nm)]
            f = win_feats(hrs_s, vals_s, t)
            if f:
                for k, v in f.items():
                    r[f"{nm}_{k}"] = v
            j = np.searchsorted(hrs_s, t, side="right") - 1
            if j >= 0:
                r[f"{nm}_locf"] = float(vals_s[j])
                r[f"{nm}_locf_age"] = float(t - hrs_s[j])
        if "cr_last" in r and pd.notna(base):
            r["cr_ratio_base"] = r["cr_last"] / base
            r["cr_delta_since_adm"] = r["cr_last"] - base
        for nm in ("cr", "bun", "hr_rate", "sbp"):  # 48h 窗（n5 对齐）
            if (sid, nm) in ser:
                h, v = ser[(sid, nm)]
                m48 = (h > t - 48) & (h <= t)
                if m48.any():
                    hh, vv = h[m48], v[m48]
                    o = np.argsort(hh, kind="mergesort")
                    hh, vv = hh[o], vv[o]
                    r[f"{nm}_last48"] = float(vv[-1])
                    r[f"{nm}_delta48"] = float(vv[-1] - vv[0])
                    r[f"{nm}_slope48"] = float(np.polyfit(hh, vv, 1)[0]) if len(vv) >= 2 and np.ptp(hh) > 0 else 0.0
                    r[f"{nm}_sd48"] = float(np.std(vv, ddof=1)) if len(vv) >= 2 else np.nan
        lv, mv, vv_ = [r.get(f"gcs_{x}_locf", np.nan) for x in ("eye", "motor", "verbal")]
        if all(np.isfinite(x) for x in (lv, mv, vv_)):
            r["gcs_total_locf"] = lv + mv + vv_
        for mnm in med_by_sid.get(sid, []):
            ev_hrs = med_ev[(sid, mnm)]
            r[f"{mnm}_on"] = int(np.any(ev_hrs <= t))
            r[f"{mnm}_n24"] = int(np.sum((ev_hrs > t - 24) & (ev_hrs <= t)))
        for mnm in ("norepi", "epi", "dopa", "dobo", "vaso", "propofol", "dex", "mido", "furosemide"):
            r.setdefault(f"{mnm}_on", 0)
            r.setdefault(f"{mnm}_n24", 0)
        r["mannitol_g_cum"] = r["mannitol_g_24h"] = r["hts_meq_cum"] = r["hts_meq_24h"] = 0.0
        r["vent_on_24h"] = int(np.any((vh > t - 24) & (vh <= t))) if vh is not None else 0
        infl = outl = np.nan
        if sid in int_g:
            h, v = int_g[sid]
            m = h <= t
            infl = float(v[m].sum()) if m.any() else 0.0
            m24 = (h > t - 24) & (h <= t)
            r["intake_ml_24h"] = float(v[m24].sum()) if m24.any() else 0.0
        if (sid, "uo") in ser:
            h, v = ser[(sid, "uo")]
            m = h <= t
            outl = float(v[m].sum()) if m.any() else 0.0
            for w_, tag_ in ((24, "24h"), (12, "12h"), (6, "6h")):
                mw = (h > t - w_) & (h <= t)
                tot = float(v[mw].sum()) if mw.any() else np.nan
                r[f"uo_ml_kg_h_{tag_}"] = tot / (w_ * wk) if np.isfinite(tot) and np.isfinite(wk) and wk > 0 else np.nan
            m48 = (h > t - 48) & (h <= t)
            r["uo_tot48"] = float(v[m48].sum()) if m48.any() else np.nan
        if np.isfinite(infl) and np.isfinite(outl) and np.isfinite(wk) and wk > 0:
            r["net_balance_ml_kg"] = (infl - outl) / wk
        rows.append(r)

F = pd.DataFrame(rows)
F.to_parquet(f"{R}/n8_4_eicu_abi_features.parquet", index=False)

import joblib
feats = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")["feats"]
missing = [c for c in feats if c not in F.columns]
cov = {c: round(float(F[c].notna().mean()), 3) for c in feats if c in F.columns}
rep = {"probe": "N8-4 eICU ABI features", "date": "2026-09-21",
       "shape": list(F.shape), "pos": int(F.label.sum()),
       "n_feats_frozen": len(feats), "missing_from_frozen": missing,
       "low_cov_feats": {k: v for k, v in cov.items() if v < 0.5},
       "caveats": ["med _on=曾启动≤t(M4=t时点在输)", "mannitol/hts 剂量=0",
                   "UO/入量 hr≥0 截断(M4 对齐)", "GCS 缺 eICU 无记录→LOCF NaN"]}
with open(f"{R}/n8_4_eicu_abi_features.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1)
print("eICU ABI features:", F.shape, "| pos:", int(F.label.sum()), "| frozen缺失列:", len(missing))
print(json.dumps({k: rep[k] for k in ("missing_from_frozen",)}, ensure_ascii=False))
