# -*- coding: utf-8 -*-
"""G7: Table 1 基线特征表（M4 train/select/intval 分割 + eICU 并排）
核验员修正：原 g7_table1.json 的 "M4_intval" 键实为全网格（与 M4_all 重复）——本脚本按真分割重出。
输出：results/g7_table1.json（覆盖，含 4 块：M4_train/M4_select/M4_intval/eICU）
"""
import json
import os
import sys
import warnings

import duckdb
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

R = "E:/TBI subtype/09_tbi_aki/results"
F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
con = duckdb.connect()
adm = con.execute(f"""
SELECT c.stay_id, c.subject_id, a.admittime, p.anchor_year
FROM read_parquet('{R}/n1_m4_abi_cohort.parquet') c
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/admissions.csv.gz') a ON c.hadm_id = a.hadm_id
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz') p ON c.subject_id = p.subject_id
""").df()
pat2 = pd.read_csv("E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz", usecols=["subject_id", "anchor_year_group"])
_g = pat2["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
pat2["group_mid"] = (_g[0].astype(int) + _g[1].astype(int)) / 2.0
coh = adm.merge(pat2[["subject_id", "group_mid"]], on="subject_id", how="left")
coh["year"] = (coh.group_mid + (coh.admittime.dt.year - coh.anchor_year)).round().astype(int)
F = F.merge(coh[["stay_id", "year"]], on="stay_id", how="left")
E = pd.read_parquet(f"{R}/n8_4_eicu_abi_features.parquet")


def summ(df):
    pat = df.groupby("stay_id").agg(age=("age", "first"), male=("male", "first"),
                                    charlson=("charlson", "first"), cr_base=("cr_base", "first"))
    n_stays = int(df.stay_id.nunique())
    return {"n_ckpt": len(df), "n_patients": n_stays,
            # caliber note (2026-09-24): the key "n_patients" is legacy naming and holds the
            # count of distinct ICU stays, NOT distinct patients. The manuscript's Table 1
            # column is labelled "ICU stays, n" accordingly. n_stays is the unambiguous alias.
            "n_stays": n_stays,
            "caliber_note": "n_patients/n_stays = distinct ICU stays (stay_id.nunique()); "
                            "distinct patients are not counted in this file",
            "pos_ckpt": int(df.label.sum()), "pos_rate": round(float(df.label.mean()), 4),
            "age_median": round(float(pat.age.median()), 1),
            "age_iqr": [round(float(pat.age.quantile(.25)), 1), round(float(pat.age.quantile(.75)), 1)],
            "male_pct": round(float(pat.male.mean() * 100), 1),
            "charlson_median": float(pat.charlson.median()),
            "cr_base_median": round(float(pat.cr_base.median()), 2)}


t1 = {"M4_train": summ(F[F.year <= 2017]),
      "M4_select": summ(F[(F.year >= 2018) & (F.year <= 2019)]),
      "M4_intval": summ(F[(F.year >= 2020) & (F.year <= 2022)]),
      "eICU_external": summ(E)}
# 自洽断言（核验员锚）
assert t1["M4_intval"]["n_ckpt"] == 28885 and t1["M4_intval"]["n_patients"] == 2563, "intval 锚不匹配"
assert t1["M4_train"]["n_ckpt"] == 16180 and t1["eICU_external"]["n_patients"] == 9071, "锚不匹配"
json.dump(t1, open(f"{R}/g7_table1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(t1, ensure_ascii=False, indent=1))
