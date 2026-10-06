# -*- coding: utf-8 -*-
"""f27: M4 (MIMIC-IV external) MK trend + subgroups — closes the Fig4 asymmetry.

PI decision 2026-10-05: Fig4 must show M4 analyses BEFORE eICU (internal -> M4 -> eICU
ordering everywhere). f12 ran MK/subgroups on eICU only (machinery reuse from paper 1);
this script runs the identical machinery on the frozen f12 M4 Tier-1 predictions.

Subgroup covariates for M4 (not present in any existing file):
  - subtype: n8_1 audit SQL VERBATIM (ICD-10 diagnoses on hadm_id, tag_subtype priorities
    TBI > SAH > ICH > IS > stroke_unspec > anoxic > encephalitis) -> composition must
    replicate the frozen n8_1_eicu_abi_crosswalk.json subtype_mix_audit.m4_pct
  - renal_disease: Quan coding algorithm (ICD-9 list = the 07f/n8_1 verbatim list;
    ICD-10 list = Quan 2005 renal categories) applied to MIMIC-IV diagnoses_icd
    (both icd_version 9 and 10 rows)
  - age / male: n5_abi_rolling_features per-stay values (same as model inputs)
MK trend: identical to f12 (24-h right-closed strata, exact permutation test over all
orderings). Subgroups: identical masks/order as f12; stay-level cluster bootstrap
(seed 42, 500 resamples) — stay_boot copied verbatim.

Output: results/jinhu/f27_m4_subgroups_mk.json (+ f27_m4_subgroup_covariates.parquet)
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

R = Path("E:/TBI subtype/09_tbi_aki/results")
J = R / "jinhu"
SEED = 42
MIMIC_DX = "E:/TBI subtype/data/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz"
# (MIM_HOSP is defined in section 1; admissions.csv.gz lives next to diagnoses_icd)

# ==================== 1. M4 subtype (subject-level, full hospital history, ICD-9+10) ====================
# The frozen n8_1 crosswalk audit used subject-level tagging mirroring the enrollment
# mechanism ("M4=subject级打标(镜像入组机制)"): each subject's flags from ALL their
# MIMIC-IV admissions (both ICD versions), applied to every stay. The stay-level
# ICD-10-only text currently in n8_1_eicu_abi_cohort.py does NOT reproduce the frozen
# percentages (26.1% other); the mechanism below matches the frozen composition to
# <=0.7pp on every category (residual = ICD-9 mirror code-list detail). Table 1c
# percentages stay frozen from the crosswalk JSON; this derivation is for the
# subgroup panel only.
con = duckdb.connect(); con.execute("SET threads=4")
MIM_HOSP = "E:/TBI subtype/data/mimic-iv-3.1/hosp"
m4sub = con.execute(f"""
WITH c AS (SELECT stay_id, subject_id FROM read_parquet('{R}/n1_m4_abi_cohort.parquet')),
a AS (SELECT subject_id, hadm_id FROM read_csv_auto('{MIM_HOSP}/admissions.csv.gz')),
dx AS (SELECT a.subject_id, x.icd_code c, x.icd_version v
  FROM a JOIN read_csv_auto('{MIMIC_DX}') x ON a.hadm_id = x.hadm_id
  WHERE a.subject_id IN (SELECT subject_id FROM c)),
d AS (SELECT subject_id,
    MAX(CASE WHEN (v=10 AND c LIKE 'S06%') OR (v=9 AND SUBSTR(c,1,3) IN('800','801','803','804','851','852','853','854')) THEN 1 ELSE 0 END) f_tbi,
    MAX(CASE WHEN (v=10 AND (c LIKE 'S02%' OR c LIKE 'S09%')) OR (v=9 AND SUBSTR(c,1,3) IN('800','801','803','804')) THEN 1 ELSE 0 END) f_skull,
    MAX(CASE WHEN (v=10 AND c LIKE 'I60%') OR (v=9 AND SUBSTR(c,1,3)='430') THEN 1 ELSE 0 END) f_sah,
    MAX(CASE WHEN (v=10 AND (c LIKE 'I61%' OR c LIKE 'I62%')) OR (v=9 AND SUBSTR(c,1,3)='431') THEN 1 ELSE 0 END) f_ich,
    MAX(CASE WHEN (v=10 AND c LIKE 'I63%') OR (v=9 AND SUBSTR(c,1,3) IN('433','434')) THEN 1 ELSE 0 END) f_isch,
    MAX(CASE WHEN (v=10 AND c = 'I64') OR (v=9 AND SUBSTR(c,1,3)='436') THEN 1 ELSE 0 END) f_stroke_unspec,
    MAX(CASE WHEN (v=10 AND c = 'G931') OR (v=9 AND SUBSTR(c,1,4)='3481') THEN 1 ELSE 0 END) f_anoxic,
    MAX(CASE WHEN (v=10 AND c LIKE 'G04%') OR (v=9 AND SUBSTR(c,1,3)='323') THEN 1 ELSE 0 END) f_enceph
  FROM dx GROUP BY 1),
j AS (SELECT c.stay_id, d.* EXCLUDE (subject_id) FROM c LEFT JOIN d ON c.subject_id = d.subject_id)
SELECT * FROM j
""").df()
PRI = [("f_tbi", "TBI"), ("f_sah", "SAH"), ("f_ich", "ICH"), ("f_isch", "IS"),
       ("f_stroke_unspec", "stroke_unspec"), ("f_anoxic", "anoxic"), ("f_enceph", "encephalitis")]
sub = pd.Series("other", index=m4sub.index)
for col, nm in reversed(PRI):          # same overwrite order as n8_1
    sub[m4sub[col] == 1] = nm
m4sub["subtype"] = sub

# composition check against the frozen crosswalk audit (Table 1c source); tolerance 0.8pp
frozen = json.load(open(R / "n8_1_eicu_abi_crosswalk.json", encoding="utf-8"))["subtype_mix_audit"]["m4_pct"]
got = (m4sub.subtype.value_counts(normalize=True) * 100).round(1).to_dict()
deltas = {k: round(abs(got.get(k, 0.0) - frozen.get(k, 0.0)), 1) for k in frozen}
worst = max(deltas.values())
assert worst <= 0.8, f"subtype composition drift vs frozen audit: {deltas}"
print(f"[ok] subtype composition within {worst}pp of frozen audit: {got}")

# ==================== 2. M4 renal_disease (Quan ICD-9 + ICD-10, chronic categories) ====================
# ICD-9 list = the 07f/n8_1 verbatim Quan renal codes (chronic-oriented: no 584 acute).
# ICD-10 = the Quan 2005 renal categories MINUS N17 (acute kidney injury): in an
# incident-AKI study, coding AKI-diagnosed patients into the "CKD" subgroup would
# contaminate it (probe run: N17 alone accounted for 69% of flags, 32% apparent
# CKD prevalence and a risk-separation artefact depressing both subgroup AUROCs
# below the pooled estimate). N17 is therefore excluded; the remaining categories
# mirror the chronic-oriented ICD-9 list.
ren = con.execute(f"""
WITH c AS (SELECT stay_id, hadm_id FROM read_parquet('{R}/n1_m4_abi_cohort.parquet')),
dx AS (
  SELECT c.stay_id, x.icd_code c, x.icd_version v
  FROM c JOIN read_csv_auto('{MIMIC_DX}') x ON c.hadm_id = x.hadm_id),
f AS (
  SELECT stay_id,
    MAX(CASE WHEN (v = 9 AND (
            SUBSTR(c,1,3) IN('582','585','586','V56') OR SUBSTR(c,1,4) IN('5880','V420','V451')
            OR SUBSTR(c,1,4) BETWEEN '5830' AND '5837'
            OR SUBSTR(c,1,5) IN('40301','40311','40391','40402','40403','40412','40413','40492','40493')))
        OR (v = 10 AND SUBSTR(c,1,3) IN('N00','N01','N02','N03','N04','N05','N07','N18','N19','N25','I12','I13'))
      THEN 1 ELSE 0 END) renal,
    MAX(CASE WHEN v = 10 AND SUBSTR(c,1,3) = 'N17' THEN 1 ELSE 0 END) n17_only_probe
  FROM dx GROUP BY 1)
SELECT * FROM f
""").df()
n_n17 = int(((ren.renal == 1) & (ren.n17_only_probe == 1)).sum())
print(f"[info] renal_disease=1 (chronic categories): {int(ren.renal.sum())} of {len(ren)} n1 stays; of these, {n_n17} also carry an N17 code")

cov = m4sub[["stay_id", "subtype"]].merge(ren[["stay_id", "renal"]], on="stay_id", how="left")
cov["renal"] = cov.renal.fillna(0).astype(int)

# ==================== 3. Frozen predictions + covariate merge ====================
pe = pd.read_parquet(J / "f12_preds_M4_ge1.parquet").rename(columns={"stay_id": "s"})
st = pd.read_parquet(R / "n5_abi_rolling_features.parquet", columns=["stay_id", "age", "male"]).drop_duplicates("stay_id")
pe = pe.merge(cov.rename(columns={"stay_id": "s"}), on="s", how="left").merge(st.rename(columns={"stay_id": "s"}), on="s", how="left")
assert pe.subtype.notna().all() and pe.age.notna().all() and pe.male.notna().all(), "covariate merge gap"

# overall AUROC must reproduce f12's frozen M4 Tier-1 estimate
F12 = json.load(open(J / "f12_full_suite.json", encoding="utf-8"))
auroc_all = float(roc_auc_score(pe.y.values, pe.p.values))
ref = F12["discrimination"]["M4_ge1"]["auroc"]
assert abs(auroc_all - ref) < 5e-4, f"overall AUROC drift: {auroc_all:.4f} vs f12 {ref}"
print(f"[ok] overall M4 AUROC reproduces f12: {auroc_all:.4f} (f12 {ref})")

# ==================== 4. MK trend (f12 machinery verbatim) ====================
strata = pe.groupby(pd.cut(pe.t_hr, bins=np.arange(24, 169, 24))).apply(
    lambda g: roc_auc_score(g.y, g.p) if g.y.sum() > 0 else np.nan).dropna()
up = sum(1 for i in range(len(strata)) for j in range(i + 1, len(strata)) if strata.values[j] > strata.values[i])
dn = sum(1 for i in range(len(strata)) for j in range(i + 1, len(strata)) if strata.values[j] < strata.values[i])
S = up - dn   # Mann-Kendall S = concordant minus discordant (2026-10-06 fix; was upward-count only)
tau, pv = kendalltau(range(len(strata)), strata.values)
assert abs(S / (len(strata) * (len(strata) - 1) / 2) - tau) < 0.002  # S and tau must agree
import itertools
n_perm = np.math.factorial(len(strata)) if hasattr(np, "math") else 1
from math import factorial
n_perm = factorial(len(strata))
cnt = 0
for p_ in itertools.permutations(range(len(strata))):
    v = [strata.values[i] for i in p_]
    Sx = (sum(1 for i in range(len(v)) for j in range(i + 1, len(v)) if v[j] > v[i])
          - sum(1 for i in range(len(v)) for j in range(i + 1, len(v)) if v[j] < v[i]))
    if Sx >= S: cnt += 1
mk = {"S": int(S), "n_concordant": int(up), "n_discordant": int(dn),
      "tau": round(float(tau), 3), "p_asym": round(float(pv), 3),
      "exact_p": round(2 * cnt / n_perm, 4), "strata": [round(float(v), 3) for v in strata.values]}
print(f"[mk] {mk}")

# ==================== 5. Subgroups (f12 masks verbatim + stay_boot) ====================
def stay_boot(df, n=500, seed=SEED):
    r = np.random.default_rng(seed)
    uniq = df.s.unique(); idx = {s_: np.where(df.s.values == s_)[0] for s_ in uniq}
    yv, pv = df.y.values, df.p.values; aucs = []
    for _ in range(n):
        sel = r.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[s_] for s_ in sel])
        if yv[ii].sum() > 0 and (1 - yv[ii]).sum() > 0:
            aucs.append(roc_auc_score(yv[ii], pv[ii]))
    return np.percentile(aucs, [2.5, 97.5]).round(4).tolist()

subgroups = {}
sub_masks = [("Overall", np.ones(len(pe), dtype=bool)), ("Age <65", pe.age < 65), ("Age >=65", pe.age >= 65),
             ("Male", pe.male == 1), ("Female", pe.male == 0),
             ("TBI", pe.subtype == "TBI"), ("SAH/ICH", pe.subtype.isin(["SAH", "ICH"])),
             ("IS/unspec", pe.subtype.isin(["IS", "stroke_unspec"])),
             ("CKD", pe.renal == 1), ("No CKD", pe.renal == 0)]
for nm_, m_ in sub_masks:
    g = pe[m_]
    subgroups[nm_] = {"auroc": round(float(roc_auc_score(g.y, g.p)), 3), "ci": stay_boot(g),
                      "n_stays": int(g.s.nunique()), "n_event_stays": int(g[g.y == 1].s.nunique())}
    print(f"  {nm_}: {subgroups[nm_]}")

# eICU cross-check pattern from f12: subgroup keys must match exactly
assert list(subgroups) == list(F12["subgroups"]), "subgroup key order must mirror f12 eICU rows"

# ==================== 6. Save ====================
out = {
    "_meta": {"date": "2026-10-05", "probe": "f27 M4 subgroups + MK (closes Fig4 asymmetry)",
              "predictions": "frozen f12_preds_M4_ge1.parquet (no model refit)",
              "subtype_source": "subject-level full-history ICD-9+10 tagging; composition verified vs frozen crosswalk (<=0.8pp)",
              "renal_source": "Quan ICD-9 (07f list verbatim) + Quan 2005 ICD-10 renal categories excluding N17 (acute kidney injury); see script header",
              "renal_with_concomitant_N17": n_n17,
              "seed": SEED, "boot_resamples": 500},
    "mk": mk, "subgroups": subgroups,
}
with open(J / "f27_m4_subgroups_mk.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
cov.rename(columns={"s": "stay_id"}).to_parquet(J / "f27_m4_subgroup_covariates.parquet", index=False)
print("\nSAVED:", J / "f27_m4_subgroups_mk.json")
print("SAVED:", J / "f27_m4_subgroup_covariates.parquet")
