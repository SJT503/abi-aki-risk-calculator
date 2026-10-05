# -*- coding: utf-8 -*-
"""N8-2: eICU ABI 全量时序提取（r4_1 机制扩展到 ABI 队列 + 补齐 271 特征所需化验）
扩展（对照 n6 feats 全清单枚举，preflight_n8 实测 labtypeid）：
  type1: creatinine/sodium/potassium/glucose/BUN/chloride/bicarbonate/anion gap/
         WBC x 1000/Hematocrit/magnesium/phosphate/lactate
  type3: Hgb(hb)/RDW(rdw)/PT - INR(inr)
  type7: paO2(po2)/Base Excess(base_excess)
其余与 r4_1 完全同构（UO+液体 intakeOutput / 生命体征 vitalPeriodic / GCS+体温 nurseCharting /
用药 infusionDrug / 通气 respiratoryCharting）+ RRT 治疗（p1_3 判定规则）。
输出：results/n8_2_*.parquet × 7
"""
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

E = "E:/TBI subtype/data/eicu-crd-2.0"
R = "E:/TBI subtype/09_tbi_aki/results"
con = duckdb.connect()
C = f"WITH c AS (SELECT stay_id sid FROM read_parquet('{R}/n8_1_eicu_abi_cohort.parquet'))"


def cache(name, q):
    df = con.execute(q).df()
    df.to_parquet(f"{R}/n8_2_{name}.parquet", index=False)
    print(name, len(df), "rows /", df.iloc[:, 0].nunique(), "stays", flush=True)


cache("labs_series", f"""
{C} SELECT l.patientunitstayid sid, l.labname, l.labtypeid, l.labresultoffset/60.0 hr, l.labresult
FROM read_csv_auto('{E}/lab.csv.gz') l JOIN c ON l.patientunitstayid=c.sid
WHERE l.labresult IS NOT NULL
  AND ((l.labtypeid=1 AND l.labname IN ('creatinine','sodium','potassium','glucose','BUN','chloride',
        'bicarbonate','anion gap','magnesium','phosphate','lactate'))
    OR (l.labtypeid=3 AND l.labname IN ('Hgb','RDW','PT - INR','Hct','WBC x 1000'))
    OR (l.labtypeid=7 AND l.labname IN ('paO2','Base Excess')))
""")
cache("uo_fluid_series", f"""
{C} SELECT i.patientunitstayid sid, i.intakeoutputoffset/60.0 hr,
       lower(i.cellpath) cellpath, lower(i.celllabel) celllabel, i.cellvaluenumeric val
FROM read_csv_auto('{E}/intakeOutput.csv.gz') i JOIN c ON i.patientunitstayid=c.sid
WHERE i.cellvaluenumeric IS NOT NULL AND i.intakeoutputoffset >= -48
""")
cache("vitals_series", f"""
{C} SELECT v.patientunitstayid sid, v.observationoffset/60.0 hr,
       v.heartrate, v.systemicsystolic, v.systemicmean, v.respiration, v.sao2
FROM read_csv_auto('{E}/vitalPeriodic.csv.gz') v JOIN c ON v.patientunitstayid=c.sid
WHERE v.observationoffset >= -48
  AND (v.heartrate IS NOT NULL OR v.systemicsystolic IS NOT NULL OR v.sao2 IS NOT NULL)
""")
cache("gcs_temp_series", f"""
{C} SELECT n.patientunitstayid sid, n.nursingchartoffset/60.0 hr,
       n.nursingchartcelltypevallabel lab, n.nursingchartvalue val
FROM read_csv_auto('{E}/nurseCharting.csv.gz') n JOIN c ON n.patientunitstayid=c.sid
WHERE n.nursingchartcelltypevallabel IN
    ('Best Motor Response','Motor Response','Best Eye Response','Eye Opening',
     'Best Verbal Response','Verbal Response','Temperature')
  AND n.nursingchartoffset >= -48
""")
cache("meds_series", f"""
{C} SELECT f.patientunitstayid sid, f.infusionoffset/60.0 start_hr, lower(f.drugname) drug
FROM read_csv_auto('{E}/infusionDrug.csv.gz') f JOIN c ON f.patientunitstayid=c.sid
WHERE lower(f.drugname) LIKE '%norepinephrine%' OR lower(f.drugname) LIKE '%epinephrine%'
   OR lower(f.drugname) LIKE '%dopamine%' OR lower(f.drugname) LIKE '%dobutamine%'
   OR lower(f.drugname) LIKE '%vasopressin%' OR lower(f.drugname) LIKE '%propofol%'
   OR lower(f.drugname) LIKE '%dexmedetomidine%' OR lower(f.drugname) LIKE '%midazolam%'
   OR lower(f.drugname) LIKE '%furosemide%' OR lower(f.drugname) LIKE '%mannitol%'
   OR lower(f.drugname) LIKE '%3% saline%' OR lower(f.drugname) LIKE '%hypertonic%'
""")
cache("vent_series", f"""
{C} SELECT r.patientunitstayid sid, r.respirationoffset/60.0 hr
FROM read_csv_auto('{E}/respiratoryCharting.csv.gz') r JOIN c ON r.patientunitstayid=c.sid
WHERE r.respirationoffset >= -48
GROUP BY 1,2
""")
cache("rrt_treatments", f"""
{C} SELECT t.patientunitstayid sid, t.treatmentoffset/60.0 hr, lower(t.treatmentstring) s
FROM read_csv_auto('{E}/treatment.csv.gz') t JOIN c ON t.patientunitstayid=c.sid
WHERE lower(t.treatmentstring) LIKE 'renal|dialysis|%' OR lower(t.treatmentstring) LIKE '%crrt%'
""")
# 无创收缩压（vitalAperiodic）——M4 220179=NBP 语义对齐（2026-09-21 补档；
# 有创周期 systemicsystolic 仅覆盖 28% stay，NBP 99%；核验员独立重提 3 stay diff=0 验真）
cache("aperiodic_sbp", f"""
{C} SELECT v.patientunitstayid sid, v.observationoffset/60.0 hr, v.noninvasivesystolic val
FROM read_csv_auto('{E}/vitalAperiodic.csv.gz') v JOIN c ON v.patientunitstayid=c.sid
WHERE v.noninvasivesystolic IS NOT NULL AND v.observationoffset >= -48
""")
print("DONE n8_2")
