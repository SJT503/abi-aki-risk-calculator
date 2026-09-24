# -*- coding: utf-8 -*-
"""N8-1: eICU ABI 队列（crosswalk 冻结版）
==========================================
入口算法等价原则（对齐 M4 ABI 锁定定义 S06+S02+S09+I60-I64+G931+G04）：
  - dx 源 = eICU diagnosis 表（住院期间任意时点诊断，最接近 M4 any-position ICD-10 语义）
  - admissionDx 不用（入院时点语义 + 粒度粗，无法分卒中亚型）
  - 入组条件与 M4 ABI 同构：成人(≥18) + LOS≥24h + ABI dx；不加 first-stay 限制（M4 ABI 多 stay 保留）
  - 不加 unittype 限制（与 07 期 eICU TBI 外验机制一致；分布入审计 JSON）
crosswalk（词表全量枚举锚定 2026-09-21，preflight_n8 验证）：
  TBI(S06)          %trauma - CNS|intracranial injury%     [4450]
  颅骨骨折(S02)      %trauma - CNS|fracture of skull%       [607]
  SAH(I60)          %stroke|hemorrhagic stroke|subarachnoid hemorrhage% [972]
  ICH(I61/I62)      %stroke|hemorrhagic stroke% + %cerebral subdural hematoma% [3620/1635]
  IS(I63)           %stroke|ischemic stroke%                [4117]
  未特指卒中(I64)    精确 'neurologic|disorders of vasculature|stroke' [3897]
  缺氧(G931)        %encephalopathy|post-anoxic%            [626]
  脑炎(G04)         %infectious disease of nervous system|encephalitis% [110]
已知映射缺口（披露）：S09 头颈其他损伤在 eICU 无细类（被 intracranial injury 宽树部分覆盖）
Charlson：07f Quan ICD-9 码表逐字移植（eICU icd9code 列；含 ICD-10 混入 → 低估，披露）
输出：results/n8_1_eicu_abi_cohort.parquet + n8_1_eicu_abi_crosswalk.json（含 M4 vs eICU 亚型构成审计）
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

E = "E:/TBI subtype/data/eicu-crd-2.0"
R = "E:/TBI subtype/09_tbi_aki/results"
con = duckdb.connect()

DX = """
  MAX(CASE WHEN diagnosisstring LIKE '%trauma - CNS|intracranial injury%' THEN 1 ELSE 0 END) f_tbi,
  MAX(CASE WHEN diagnosisstring LIKE '%trauma - CNS|fracture of skull%' THEN 1 ELSE 0 END) f_skull,
  MAX(CASE WHEN diagnosisstring LIKE '%stroke|hemorrhagic stroke|subarachnoid hemorrhage%' THEN 1 ELSE 0 END) f_sah,
  MAX(CASE WHEN (diagnosisstring LIKE '%stroke|hemorrhagic stroke%'
              OR diagnosisstring LIKE '%cerebral subdural hematoma%') THEN 1 ELSE 0 END) f_ich,
  MAX(CASE WHEN diagnosisstring LIKE '%stroke|ischemic stroke%' THEN 1 ELSE 0 END) f_isch,
  MAX(CASE WHEN diagnosisstring = 'neurologic|disorders of vasculature|stroke' THEN 1 ELSE 0 END) f_stroke_unspec,
  MAX(CASE WHEN diagnosisstring LIKE '%encephalopathy|post-anoxic%' THEN 1 ELSE 0 END) f_anoxic,
  MAX(CASE WHEN diagnosisstring LIKE '%infectious disease of nervous system|encephalitis%' THEN 1 ELSE 0 END) f_enceph
"""

coh = con.execute(f"""
WITH dx AS (
  SELECT patientunitstayid sid, {DX}
  FROM read_csv_auto('{E}/diagnosis.csv.gz') GROUP BY 1
),
flagged AS (
  SELECT d.*, p.gender, p.age, p.ethnicity, p.unittype, p.hospitalid,
         p.patienthealthsystemstayid, p.unitdischargeoffset/60.0 los_h
  FROM dx d JOIN read_csv_auto('{E}/patient.csv.gz') p ON d.sid = p.patientunitstayid
  WHERE d.f_tbi+d.f_skull+d.f_sah+d.f_ich+d.f_isch+d.f_stroke_unspec+d.f_anoxic+d.f_enceph > 0
),
aged AS (SELECT *, TRY_CAST(replace(age,'> 89','90') AS DOUBLE) age_num FROM flagged)
SELECT sid, gender, age_num age, ethnicity, unittype, hospitalid,
       patienthealthsystemstayid, los_h,
       f_tbi, f_skull, f_sah, f_ich, f_isch, f_stroke_unspec, f_anoxic, f_enceph
FROM aged WHERE age_num >= 18 AND los_h >= 24
""").df()
print(f"eICU ABI raw flagged+adult+LOS24: {len(coh)} stays / {coh.patienthealthsystemstayid.nunique()} hosp-visits")

PRI = [("f_tbi", "TBI"), ("f_sah", "SAH"), ("f_ich", "ICH"), ("f_isch", "IS"),
       ("f_stroke_unspec", "stroke_unspec"), ("f_anoxic", "anoxic"), ("f_enceph", "encephalitis")]


def tag_subtype(df):
    sub = pd.Series("other", index=df.index)
    for col, nm in reversed(PRI):  # 后写覆盖 → 优先级高的最后写
        sub[df[col] == 1] = nm
    return sub


coh["subtype"] = tag_subtype(coh)

# ---- Charlson（07f Quan ICD-9 码表逐字移植，范围=本队列） ----
CASES = {
    "myocardial_infarct": "SUBSTR(c,1,3) IN('410','412')",
    "congestive_heart_failure": "SUBSTR(c,1,3)='428' OR SUBSTR(c,1,5) IN('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') OR SUBSTR(c,1,4) BETWEEN '4254' AND '4259'",
    "peripheral_vascular_disease": "SUBSTR(c,1,3) IN('440','441') OR SUBSTR(c,1,4) IN('0930','4373','4471','5571','5579','V434') OR SUBSTR(c,1,4) BETWEEN '4431' AND '4439'",
    "cerebrovascular_disease": "SUBSTR(c,1,3) BETWEEN '430' AND '438' OR SUBSTR(c,1,5)='36234'",
    "dementia": "SUBSTR(c,1,3)='290' OR SUBSTR(c,1,4) IN('2941','3312')",
    "chronic_pulmonary_disease": "SUBSTR(c,1,3) BETWEEN '490' AND '505' OR SUBSTR(c,1,4) IN('4168','4169','5064','5081','5088')",
    "rheumatic_disease": "SUBSTR(c,1,3)='725' OR SUBSTR(c,1,4) IN('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148')",
    "peptic_ulcer_disease": "SUBSTR(c,1,3) IN('531','532','533','534')",
    "mild_liver_disease": "SUBSTR(c,1,3) IN('570','571') OR SUBSTR(c,1,4) IN('0706','0709','5733','5734','5738','5739','V427') OR SUBSTR(c,1,5) IN('07022','07023','07032','07033','07044','07054')",
    "diabetes_without_cc": "SUBSTR(c,1,4) IN('2500','2501','2502','2503','2508','2509')",
    "diabetes_with_cc": "SUBSTR(c,1,4) IN('2504','2505','2506','2507')",
    "paraplegia": "SUBSTR(c,1,3) IN('342','343') OR SUBSTR(c,1,4) IN('3341','3440','3441','3442','3443','3444','3445','3446','3449')",
    "renal_disease": "SUBSTR(c,1,3) IN('582','585','586','V56') OR SUBSTR(c,1,4) IN('5880','V420','V451') OR SUBSTR(c,1,4) BETWEEN '5830' AND '5837' OR SUBSTR(c,1,5) IN('40301','40311','40391','40402','40403','40412','40413','40492','40493')",
    "malignant_cancer": "SUBSTR(c,1,3) BETWEEN '140' AND '172' OR SUBSTR(c,1,4) BETWEEN '1740' AND '1958' OR SUBSTR(c,1,3) BETWEEN '200' AND '208' OR SUBSTR(c,1,4)='2386'",
    "severe_liver_disease": "SUBSTR(c,1,4) IN('4560','4561','4562') OR SUBSTR(c,1,4) BETWEEN '5722' AND '5728'",
    "metastatic_solid_tumor": "SUBSTR(c,1,3) IN('196','197','198','199')",
    "aids": "SUBSTR(c,1,3) IN('042','043','044')",
}
FLAGS = list(CASES)
SCORE = ("COALESCE(myocardial_infarct,0)+COALESCE(congestive_heart_failure,0)+COALESCE(peripheral_vascular_disease,0)"
         "+COALESCE(cerebrovascular_disease,0)+COALESCE(dementia,0)+COALESCE(chronic_pulmonary_disease,0)"
         "+COALESCE(rheumatic_disease,0)+COALESCE(peptic_ulcer_disease,0)"
         "+GREATEST(COALESCE(mild_liver_disease,0), 3*COALESCE(severe_liver_disease,0))"
         "+GREATEST(2*COALESCE(diabetes_with_cc,0), COALESCE(diabetes_without_cc,0))"
         "+GREATEST(2*COALESCE(malignant_cancer,0), 6*COALESCE(metastatic_solid_tumor,0))"
         "+2*COALESCE(paraplegia,0)+2*COALESCE(renal_disease,0)+6*COALESCE(aids,0)")
flags_sql = ",\n  ".join(f"MAX(CASE WHEN {cond} THEN 1 ELSE 0 END) {k}" for k, cond in CASES.items())
coals = ",\n  ".join(f"COALESCE({k},0) {k}" for k in FLAGS)
# sids 临时表先落盘，Charlson 查询再引用
coh[["sid"]].to_parquet(f"{R}/_n8_tmp_sids.parquet", index=False)
ch = con.execute(f"""
WITH s AS (SELECT sid FROM read_parquet('{R}/_n8_tmp_sids.parquet')),
dx AS (
  SELECT patientunitstayid, replace(trim(t.u),'.','') c
  FROM read_csv_auto('{E}/diagnosis.csv.gz') d,
       UNNEST(string_split(COALESCE(d.icd9code,''), ',')) AS t(u)
  WHERE patientunitstayid IN (SELECT sid FROM s)),
com AS (SELECT patientunitstayid, {flags_sql} FROM dx GROUP BY 1)
SELECT patientunitstayid sid, {coals}, {SCORE} charlson FROM com
""").df()

coh = coh.merge(ch, on="sid", how="left")
for c in FLAGS + ["charlson"]:
    coh[c] = coh[c].fillna(0).astype(int)
coh["male"] = (coh.gender == "Male").astype(int)
coh = coh.rename(columns={"sid": "stay_id"})
coh.to_parquet(f"{R}/n8_1_eicu_abi_cohort.parquet", index=False)

# ---- M4 侧亚型构成（同优先级打标，供 case-mix 审计） ----
m4 = con.execute(f"""
WITH c AS (SELECT stay_id, hadm_id FROM read_parquet('{R}/n1_m4_abi_cohort.parquet')),
d AS (
  SELECT c.stay_id,
    MAX(CASE WHEN icd_code LIKE 'S06%' THEN 1 ELSE 0 END) f_tbi,
    MAX(CASE WHEN icd_code LIKE 'S02%' OR icd_code LIKE 'S09%' THEN 1 ELSE 0 END) f_skull,
    MAX(CASE WHEN icd_code LIKE 'I60%' THEN 1 ELSE 0 END) f_sah,
    MAX(CASE WHEN icd_code LIKE 'I61%' OR icd_code LIKE 'I62%' THEN 1 ELSE 0 END) f_ich,
    MAX(CASE WHEN icd_code LIKE 'I63%' THEN 1 ELSE 0 END) f_isch,
    MAX(CASE WHEN icd_code = 'I64' THEN 1 ELSE 0 END) f_stroke_unspec,
    MAX(CASE WHEN icd_code = 'G931' THEN 1 ELSE 0 END) f_anoxic,
    MAX(CASE WHEN icd_code LIKE 'G04%' THEN 1 ELSE 0 END) f_enceph
  FROM c JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz') x
    ON c.hadm_id = x.hadm_id
  WHERE icd_version = 10
  GROUP BY 1)
SELECT * FROM d
""").df()
m4["subtype"] = tag_subtype(m4)

eu = coh.subtype.value_counts(normalize=True)
mu = m4.subtype.value_counts(normalize=True)
audit = pd.DataFrame({"m4_pct": (mu * 100).round(1), "eicu_pct": (eu * 100).round(1)}).fillna(0)

rep = {"probe": "N8-1 eICU ABI cohort (crosswalk frozen)", "date": "2026-09-21",
       "n_stays": int(len(coh)), "n_hosp_visits": int(coh.patienthealthsystemstayid.nunique()),
       "n_hospitals": int(coh.hospitalid.nunique()),
       "excl": "成人≥18 + LOS≥24h；无 first-stay/unittype 限制（镜像 M4 ABI 机制）",
       "subtype_eicu": coh.subtype.value_counts().to_dict(),
       "subtype_mix_audit": audit.to_dict(),
       "age_median": float(coh.age.median()), "male_pct": round(100 * coh.male.mean(), 1),
       "charlson_median": float(coh.charlson.median()),
       "unittype_top": coh.unittype.value_counts().head(8).to_dict(),
       "known_gaps": ["S09 无 eICU 细类（宽树部分覆盖）", "admissionDx 未用（入院时点语义）",
                      "Charlson 走 icd9code 列（ICD-10 混入→低估，同 07f 先例）",
                      "meningitis/TIA/seizure/脊髓/肿瘤 按锁定定义排除"]}
with open(f"{R}/n8_1_eicu_abi_crosswalk.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: rep[k] for k in ("n_stays", "n_hospitals", "age_median", "male_pct", "charlson_median")}, ensure_ascii=False))
print("\n=== 亚型构成审计（M4 vs eICU, %）===")
print(audit.to_string())
print("\nSAVED:", f"{R}/n8_1_eicu_abi_cohort.parquet")
