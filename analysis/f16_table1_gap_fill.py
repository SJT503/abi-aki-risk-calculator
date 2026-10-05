# -*- coding: utf-8 -*-
"""f16: fill the remaining Table 1b demographic gaps from local authoritative sources.
  1. Pooled M4 demographics (age/male/charlson/cr_base) from the grid feature table
     used for the external predictions (first checkpoint per stay, 7,442 stays),
     with consistency assertions against the companion strata (Table 1d).
  2. Jinhua dev/test Charlson + baseline creatinine via the deterministic split
     (same machinery as f14; stays 825/208 asserted).
  3. Records the corrected pooled Jinhua cr_base IQR (0.95, reproducible from both
     the features table and the labs axis restricted to grid stays).
Output: results/jinhu/f16_table1_gap_fill.json  (also shipped in package data/)
"""
import json
import numpy as np
import pandas as pd
import duckdb

R = "E:/TBI subtype/09_tbi_aki/results"
J = f"{R}/jinhu"
out = {"_meta": {"date": "2026-10-03", "purpose": "Table 1b demographic gap fill (post hoc, no model involvement)"}}

# ---------- 1. M4 pooled ----------
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
st = m4.drop_duplicates("stay_id")
assert len(st) == 7442
m4d = {
    "stays": int(len(st)),
    "age": [float(st.age.median()), float(st.age.quantile(.25)), float(st.age.quantile(.75))],
    "male_pct": round(100 * float(st.male.mean()), 1),
    "charlson": [float(st.charlson.median()), float(st.charlson.quantile(.25)), float(st.charlson.quantile(.75))],
    "cr_base": [float(st.cr_base.median()), float(st.cr_base.quantile(.25)), float(st.cr_base.quantile(.75))],
    "source": "n5_abi_rolling_features.parquet (grid used for external predictions), first checkpoint per stay",
}
# consistency vs companion strata (Table 1d): pooled male% must equal stay-weighted strata mean within 0.1
wa = 5969 / 7442
assert abs(m4d["male_pct"] - (56.1 * wa + 55.1 * (1 - wa))) < 0.15, "male% inconsistent with strata"
assert 66 <= m4d["age"][0] <= 67 and 54 <= m4d["age"][1] and m4d["age"][2] <= 78, "age outside strata bounds"
out["m4_pooled"] = m4d
print("M4 pooled:", m4d["age"], m4d["male_pct"], m4d["charlson"], m4d["cr_base"])

# ---------- 2. Jinhua dev/test ----------
jf = pd.read_parquet(f"{J}/jinhu_features.parquet")
jf["hadm_id"] = jf.stay_id.astype(str)
con = duckdb.connect(); con.execute("SET threads=4")
DATA = "F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data"
sub = con.execute(f"SELECT hadm_id, subject_id FROM read_csv_auto('{DATA}/medical_record_front_page.csv', all_varchar=true)").df()
sub["hadm_id"] = sub.hadm_id.astype(str)
jf["subject"] = jf.hadm_id.map(dict(zip(sub.hadm_id, sub.subject_id)))
subjects = np.array(sorted(jf.subject.unique()))
np.random.seed(42); perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
sts = jf.drop_duplicates("hadm_id")
tr = sts[sts.subject.isin(train_subj)]; te = sts[~sts.subject.isin(train_subj)]
assert (len(tr), len(te)) == (825, 208), "split mismatch"
def summ(d, c):
    return [round(float(d[c].median()), 2), round(float(d[c].quantile(.25)), 2), round(float(d[c].quantile(.75)), 2)]
out["jinhu"] = {
    "dev_stays": 825, "test_stays": 208,
    "dev_charlson": [int(tr.charlson.median()), int(tr.charlson.quantile(.25)), int(tr.charlson.quantile(.75))],
    "test_charlson": [int(te.charlson.median()), int(te.charlson.quantile(.25)), int(te.charlson.quantile(.75))],
    "dev_cr_base": summ(tr, "cr_base"), "test_cr_base": summ(te, "cr_base"),
    "pooled_cr_base": summ(sts, "cr_base"),
    "pooled_cr_base_note": "0.76 (0.61-0.95); reproducible from the features table and from the labs axis restricted to the 1,033 grid stays; supersedes the 0.96 upper quartile in the earlier summary snapshot, which no current pipeline artifact reproduces",
    "eicu_patients": "not counted: eICU-CRD patient identifiers (uniquepid) were not retained by the extraction; patienthealthsystemstayid (8,607) is a hospital-admission identifier, not a patient identifier",
    "source": "jinhu_features.parquet + deterministic subject split (seed 42, f14 machinery)",
}
print("Jinhua dev:", out["jinhu"]["dev_charlson"], out["jinhu"]["dev_cr_base"],
      "| test:", out["jinhu"]["test_charlson"], out["jinhu"]["test_cr_base"])

# ---------- 3. eICU-CRD unique patients (patient table of the database) ----------
pt = pd.read_csv("E:/TBI subtype/data/eicu-crd-2.0/patient.csv.gz",
                 usecols=["patientunitstayid", "patienthealthsystemstayid", "uniquepid"])
grid_ids = pd.read_parquet(f"{R}/p1_expanded_features.parquet").drop_duplicates("stay_id")
m = grid_ids.merge(pt.rename(columns={"patientunitstayid": "stay_id"}), on="stay_id", how="left")
assert m.uniquepid.notna().all(), "unmatched eICU stays"
n_adm = int(m.patienthealthsystemstayid.nunique())
n_pat = int(m.uniquepid.nunique())
assert n_adm == 8607, "admission count inconsistent with n8_1 cohort"
out["eicu_patients"] = {
    "icu_stays": int(len(m)), "hospital_admissions": n_adm, "unique_patients": n_pat,
    "patients_with_multiple_abi_stays": int((m.groupby("uniquepid").stay_id.nunique() > 1).sum()),
    "max_stays_per_patient": int(m.groupby("uniquepid").stay_id.nunique().max()),
    "source": "eicu-crd-2.0 patient.csv.gz (official database file), joined on patientunitstayid",
}
print("eICU:", len(m), "stays =", n_adm, "admissions =", n_pat, "patients")

json.dump(out, open(f"{J}/f16_table1_gap_fill.json", "w"), indent=2)
print("Saved f16_table1_gap_fill.json")
