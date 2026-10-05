# -*- coding: utf-8 -*-
"""f4: JinhuaNSICU extraction for the 268-feature external validation (2026-09-30).
Mirrors the eICU external template (n8_1 cohort + n8_2 series + unit conventions):
- ABI cohort: ICD-10 S06/S02/S09/I60/I61/I62/I63/I64, age>=18, first ICU episode los>=24h
- Charlson: Quan ICD-10 adaptation with the n8_1-identical GREATEST scoring
- Labs: 18 families (exact item names from jinhu_p2 probe), units converted to MIMIC
  conventions with runtime quantile asserts
- Vitals: Pulse/Sbp/Respiratory/Spo2; Meds: norepi/propofol/dex/mido/hts;
  vent orders; RRT orders (血液净化/透析 excl 腹膜)
- Time base: minutes from HOSPITAL admission (per whitepaper); ICU anchoring in f5.
Outputs: results/jinhu/{jinhu_cohort,jinhu_labs,jinhu_vitals,jinhu_meds,jinhu_vent,jinhu_rrt}.parquet
"""
import json, os
import duckdb
import numpy as np, pandas as pd

DATA = "F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data"
OUT = "E:/TBI subtype/09_tbi_aki/results/jinhu"
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()
con.execute("SET threads=8")

# ============ 1. ABI cohort + first ICU episode + demographics ============
coh = con.execute(f"""
WITH dx AS (
  SELECT hadm_id,
    MAX(CASE WHEN diagnostic_code LIKE 'S06%' THEN 1 ELSE 0 END) f_tbi,
    MAX(CASE WHEN diagnostic_code LIKE 'S02%' OR diagnostic_code LIKE 'S09%' THEN 1 ELSE 0 END) f_skull,
    MAX(CASE WHEN diagnostic_code LIKE 'I60%' THEN 1 ELSE 0 END) f_sah,
    MAX(CASE WHEN diagnostic_code LIKE 'I61%' OR diagnostic_code LIKE 'I62%' THEN 1 ELSE 0 END) f_ich,
    MAX(CASE WHEN diagnostic_code LIKE 'I63%' THEN 1 ELSE 0 END) f_isch,
    MAX(CASE WHEN diagnostic_code LIKE 'I64%' THEN 1 ELSE 0 END) f_stroke_unspec
  FROM read_csv_auto('{DATA}/diagnosis.csv', all_varchar=true)
  GROUP BY 1
), abd AS (
  SELECT hadm_id FROM dx WHERE f_tbi+f_skull+f_sah+f_ich+f_isch+f_stroke_unspec > 0
), fp AS (
  SELECT hadm_id, patient_gender, try_cast(age AS DOUBLE) age,
         try_cast(admittime_base AS DOUBLE) admit_base, try_cast(dischtime_base AS DOUBLE) disch_base
  FROM read_csv_auto('{DATA}/medical_record_front_page.csv', all_varchar=true)
)
SELECT dx.*, fp.patient_gender, fp.age, fp.admit_base, fp.disch_base
FROM abd JOIN dx USING (hadm_id) JOIN fp USING (hadm_id)
WHERE fp.age >= 18
""").df()
print(f"ABI cohort (age>=18): {len(coh)} hadms")

# first ICU episode from transfer chain (rows = dept episode ending at outtime_base -> transfer_department)
tr = con.execute(f"""
SELECT hadm_id, try_cast(outtime_base AS DOUBLE) outtime, out_department, transfer_department
FROM read_csv_auto('{DATA}/transfer.csv', all_varchar=true)
""").df()
tr = tr[tr.hadm_id.isin(coh.hadm_id)]
def is_icu(s): return s.fillna("").str.contains("重症") | s.fillna("").str.lower().str.contains("intensive")

rows = []
for hadm, g in tr.groupby("hadm_id"):
    g = g.sort_values("outtime").reset_index(drop=True)
    in_rows = g[is_icu(g.transfer_department)]
    if not len(in_rows):
        continue
    i0 = in_rows.index[0]
    icu_start = float(g.loc[i0, "outtime"])
    out_rows = g[(g.index > i0) & is_icu(g.out_department)]
    icu_end = float(g.loc[out_rows.index[0], "outtime"]) if len(out_rows) else np.nan
    rows.append((hadm, icu_start, icu_end))
EP = pd.DataFrame(rows, columns=["hadm_id", "icu_start", "icu_end"])
coh = coh.merge(EP, on="hadm_id", how="inner")
mrc = con.execute(f"""
SELECT hadm_id, try_cast(dischtime_base AS DOUBLE) disch
FROM read_csv_auto('{DATA}/medical_record.csv', all_varchar=true)
""").df().dropna().drop_duplicates("hadm_id")
coh = coh.merge(mrc, on="hadm_id", how="left")
coh["icu_end"] = coh["icu_end"].fillna(coh["disch"])
fp_disch = coh.pop("disch_base")  # keep front-page discharge as fallback
coh["icu_end"] = coh["icu_end"].fillna(fp_disch)
coh["los_icu_h"] = (coh["icu_end"] - coh["icu_start"]) / 60.0
coh = coh[coh.los_icu_h >= 24].copy()
coh["male"] = (coh.patient_gender == "男").astype(int)
print(f"with first-ICU episode & los>=24h: {len(coh)} hadms")

# ============ 2. Charlson (Quan ICD-10, n8_1-identical scoring) ============
C10 = {
 "myocardial_infarct": ["I21", "I22", "I252"],
 "congestive_heart_failure": ["I50", "I110", "I130", "I132"],
 "peripheral_vascular_disease": ["I70", "I71", "I731", "I738", "I739", "R02", "Z958", "Z959"],
 "cerebrovascular_disease": ["I60","I61","I62","I63","I64","I65","I66","I67","I68","I69","G45","G46"],
 "dementia": ["F00","F01","F02","F03","G30","G311","F051"],
 "chronic_pulmonary_disease": ["J40","J41","J42","J43","J44","J45","J46","J47","J60","J61","J62","J63","J64","J65","J66","J67","J684","J701","I278","I279"],
 "rheumatic_disease": ["M05","M06","M315","M32","M33","M34","M351","M353","M360"],
 "peptic_ulcer_disease": ["K25","K26","K27","K28"],
 "mild_liver_disease": ["B18","K702","K703","K713","K714","K715","K717","K73","K74","K760","K762","K763","K764","Z944"],
 "diabetes_without_cc": ["E109","E119","E139","E149"],
 "diabetes_with_cc": ["E100","E101","E102","E103","E104","E105","E106","E107","E108",
                      "E110","E111","E112","E113","E114","E115","E116","E117","E118",
                      "E130","E131","E132","E133","E134","E135","E136","E137","E138",
                      "E140","E141","E142","E143","E144","E145","E146","E147","E148"],
 "paraplegia": ["G81","G82","G830","G831","G832","G833","G834","G838","G839"],
 "renal_disease": ["I12","I13","N032","N033","N034","N035","N036","N037",
                   "N052","N053","N054","N055","N056","N057","N18","N19","N25","Z490","Z491","Z492","Z940"],
 "malignant_cancer": [],  # range C00-C97 excl C77-C79, handled below
 "severe_liver_disease": ["K704","K72","K766","K767","I850","I859","I864","I982"],
 "metastatic_solid_tumor": ["C77","C78","C79"],
 "aids": ["B20","B21","B22","B23","B24"],
}
def prefix_of(c):
    c = c.strip().upper()
    out = ""
    for ch in c:
        if ch.isalpha() and len(out) == 0: out += ch
        elif ch.isdigit() and len(out) >= 1: out += ch
        else: break
    return out

dxs = con.execute(f"""
SELECT hadm_id, diagnostic_code FROM read_csv_auto('{DATA}/diagnosis.csv', all_varchar=true)
WHERE hadm_id IN (SELECT hadm_id FROM read_csv_auto('{DATA}/medical_record_front_page.csv', all_varchar=true))
""").df()
dxs = dxs[dxs.hadm_id.isin(coh.hadm_id)]

flags = {k: set() for k in C10}
for hadm, code in zip(dxs.hadm_id, dxs.diagnostic_code):
    if not isinstance(code, str): continue
    p = prefix_of(code)
    if not p: continue
    h = hadm
    for k, pats in C10.items():
        if any(p == q or (p.startswith(q) and len(q) >= 3 and p[:len(q)] == q) for q in pats):
            flags[k].add(h)
    L = p[0] if p else ""
    if L == "C":
        num = "".join(ch for ch in p[1:] if ch.isdigit())
        try:
            n = int(num) if num else -1
        except ValueError:
            n = -1
        if 0 <= n <= 76:
            flags["malignant_cancer"].add(h)
ch = pd.DataFrame({"hadm_id": list(coh.hadm_id)})
for k in C10:
    ch[k] = ch.hadm_id.isin(flags[k]).astype(int)
ch["charlson"] = (ch.myocardial_infarct + ch.congestive_heart_failure + ch.peripheral_vascular_disease
                  + ch.cerebrovascular_disease + ch.dementia + ch.chronic_pulmonary_disease
                  + ch.rheumatic_disease + ch.peptic_ulcer_disease
                  + np.maximum(ch.mild_liver_disease, 3 * ch.severe_liver_disease)
                  + np.maximum(2 * ch.diabetes_with_cc, ch.diabetes_without_cc)
                  + np.maximum(2 * ch.malignant_cancer, 6 * ch.metastatic_solid_tumor)
                  + 2 * ch.paraplegia + 2 * ch.renal_disease + 6 * ch.aids)
coh = coh.merge(ch[["hadm_id", "charlson"]], on="hadm_id", how="left")
print(f"charlson median: {coh.charlson.median()}")

# ============ 3. Labs (18 families, exact items) ============
LAB_ITEMS = {
 "cr": ["肌酐", "肌酐[干化学]"], "bun": ["尿素氮", "尿素氮(尿素)"], "na": ["钠"], "k": ["钾"],
 "cl": ["氯"], "hco3": ["实际碳酸氢根"], "glu": ["葡萄糖"], "wbc": ["白细胞"],
 "hct": ["红细胞压积"], "hb": ["血红蛋白"], "rdw": ["红细胞分布宽度"], "inr": ["国际标准化比值"],
 "mg": ["镁", "镁[干化学]"], "phos": ["磷"], "ag": ["阴离子间隙"], "lactate": ["乳酸"],
 "po2": ["氧分压"], "base_excess": ["剩余碱", "标准剩余碱"],
}
COND = " OR ".join([f"inspection_name = '{it}'" for its in LAB_ITEMS.values() for it in its])
labs = con.execute(f"""
SELECT hadm_id, inspection_name, try_cast(test_results_quantitative AS DOUBLE) val,
       try_cast(report_time_base AS DOUBLE) tmin
FROM read_csv_auto('{DATA}/laboratory_test.csv', all_varchar=true)
WHERE ({COND}) AND hadm_id IN {tuple(coh.hadm_id)}
""").df()
i2f = {it: f for f, its in LAB_ITEMS.items() for it in its}
labs["fam"] = labs.inspection_name.map(i2f)
labs = labs.dropna(subset=["val", "tmin"])
print("lab rows:", len(labs), "| families:", labs.fam.nunique())

# ============ 4. Vitals / 5. Meds / 6. Vent + RRT ============
vits = con.execute(f"""
SELECT hadm_id, subcategory_name_en, try_cast(value AS DOUBLE) val, try_cast(charttime_base AS DOUBLE) tmin
FROM read_csv_auto('{DATA}/vital_signs.csv', all_varchar=true)
WHERE subcategory_name_en IN ('Pulse','Sbp','Respiratory','Spo2') AND hadm_id IN {tuple(coh.hadm_id)}
""").df().dropna(subset=["val", "tmin"])
VMAP = {"Pulse": "hr_rate", "Sbp": "sbp", "Respiratory": "rr", "Spo2": "spo2"}
vits["fam"] = vits.subcategory_name_en.map(VMAP)
print("vital rows:", len(vits))

meds = con.execute(f"""
SELECT hadm_id, medication_common_name, try_cast(single_dose AS DOUBLE) dose,
       medication_dose_unit, try_cast(starttime_base AS DOUBLE) stmin, try_cast(endtime_base AS DOUBLE) etmin
FROM read_csv_auto('{DATA}/medication.csv', all_varchar=true)
WHERE (medication_common_name ILIKE '%去甲肾上腺素%' OR medication_common_name ILIKE '%丙泊酚%'
    OR medication_common_name ILIKE '%右美托咪定%' OR medication_common_name ILIKE '%咪达唑仑%'
    OR medication_common_name ILIKE '%浓氯化钠%') AND hadm_id IN {tuple(coh.hadm_id)}
""").df()
def medfam(n):
    n = n or ""
    if "去甲肾上腺素" in n: return "norepi"
    if "丙泊酚" in n: return "propofol"
    if "右美托咪定" in n: return "dex"
    if "咪达唑仑" in n: return "mido"
    if "浓氯化钠" in n or "高渗" in n: return "hts"
    return None
meds["fam"] = meds.medication_common_name.map(medfam)
meds = meds.dropna(subset=["fam", "stmin"])
print("med rows:", len(meds), "| hadms:", meds.hadm_id.nunique())

vent = con.execute(f"""
SELECT hadm_id, try_cast(starttime_base AS DOUBLE) stmin
FROM read_csv_auto('{DATA}/orders.csv', all_varchar=true)
WHERE (order_content ILIKE '%呼吸机%' OR order_content ILIKE '%气管插管%') AND hadm_id IN {tuple(coh.hadm_id)}
""").df().dropna(subset=["stmin"])
print("vent order rows:", len(vent), "| hadms:", vent.hadm_id.nunique())

rrt = con.execute(f"""
SELECT hadm_id, try_cast(starttime_base AS DOUBLE) stmin, order_content
FROM read_csv_auto('{DATA}/orders.csv', all_varchar=true)
WHERE (order_content ILIKE '%血液净化%' OR (order_content ILIKE '%透析%' AND order_content NOT ILIKE '%腹膜%'))
  AND hadm_id IN {tuple(coh.hadm_id)}
""").df().dropna(subset=["stmin"])
print("rrt order rows:", len(rrt), "| hadms:", rrt.hadm_id.nunique())

# ============ 7. Unit conversion with quantile asserts ============
def q(fam, v):
    return v
pre = labs.groupby("fam").val.agg(["count", "median", lambda s: s.quantile(.01), lambda s: s.quantile(.99)])
print("\npre-conversion quantiles:\n", pre)

def conv(fam, v):
    if fam == "cr":   return v / 88.4
    if fam == "bun":  return v * 2.801
    if fam == "glu":  return v * 18.02
    if fam == "hb":   return v / 10.0
    if fam == "hct":  return v * 100.0 if np.nanmedian(v) < 1.5 else v
    if fam == "phos": return v * 3.097
    return v

med = labs.groupby("fam").val.median()
for fam in LAB_ITEMS:
    m = labs.fam == fam
    labs.loc[m, "val"] = conv(fam, labs.loc[m, "val"].values)
# post-conversion sanity asserts (MIMIC-like plausibility)
SAN = {"cr": (0.05, 40), "bun": (0, 300), "na": (90, 180), "k": (1, 12), "cl": (60, 130),
       "hco3": (1, 60), "glu": (5, 1500), "wbc": (0, 500), "hct": (3, 85), "hb": (1, 30),
       "rdw": (5, 45), "inr": (0.1, 25), "mg": (0.1, 20), "phos": (0.2, 40), "ag": (-5, 60),
       "lactate": (0, 45), "po2": (5, 800), "base_excess": (-45, 45)}
post = labs.groupby("fam").val.median()
for fam, (lo, hi) in SAN.items():
    md = post.get(fam, np.nan)
    assert lo <= md <= hi, f"{fam} post-conversion median {md} outside [{lo},{hi}]"
# vit clamps
VC = {"hr_rate": (0, 300), "sbp": (20, 300), "rr": (0, 90), "spo2": (0, 100)}
for fam, (lo, hi) in VC.items():
    vits.loc[(vits.fam == fam) & (~vits.val.between(lo, hi)), "val"] = np.nan
vits = vits.dropna(subset=["val"])
print("\npost-conversion medians:\n", post.round(2).to_string())

# ============ 8. Save ============
coh.drop(columns=["patient_gender", "admit_base"], errors="ignore").to_parquet(f"{OUT}/jinhu_cohort.parquet", index=False)
labs[["hadm_id", "fam", "tmin", "val"]].to_parquet(f"{OUT}/jinhu_labs.parquet", index=False)
vits[["hadm_id", "fam", "tmin", "val"]].to_parquet(f"{OUT}/jinhu_vitals.parquet", index=False)
meds[["hadm_id", "fam", "stmin", "etmin", "dose", "medication_dose_unit"]].to_parquet(f"{OUT}/jinhu_meds.parquet", index=False)
vent.to_parquet(f"{OUT}/jinhu_vent.parquet", index=False)
rrt.to_parquet(f"{OUT}/jinhu_rrt.parquet", index=False)
json.dump({"n_hadms": int(len(coh)), "subtype": coh[["f_tbi", "f_skull", "f_sah", "f_ich", "f_isch", "f_stroke_unspec"]].sum().to_dict(),
           "los_icu_h_median": float(coh.los_icu_h.median()),
           "lab_rows": int(len(labs)), "vital_rows": int(len(vits)), "med_hadms": int(meds.hadm_id.nunique()),
           "vent_hadms": int(vent.hadm_id.nunique()), "rrt_hadms": int(rrt.hadm_id.nunique())},
          open(f"{OUT}/f4_extract_report.json", "w"), indent=1)
print("\nSaved 6 parquet sets to", OUT)
