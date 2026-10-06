# -*- coding: utf-8 -*-
"""f28: M4-first Fig4 text cascade (PI order 2026-10-05).
Patches the round-11 package text sources + resolved files so every Fig4-related
statement covers MIMIC-IV before eICU-CRD, matching the redesigned 6-panel Fig4.
Files touched (exact-match replacements, every edit asserted):
  scripts/text/manuscript_body_r11.md   (source, {N:} placeholders)
  scripts/text/captions_r11.md          (source)
  scripts/text/SM5_sensitivity_r11.md   (source)
  manuscript/MANUSCRIPT_npjDM_round11.md (resolved, literals)
  supplementary/Supplementary_Information_round11.md (resolved SI)
"""
import io, sys
from pathlib import Path

FIN = Path(r"E:/TBI subtype/09_tbi_aki/_biomni_r112/final")
EN = "\u2013"

def patch(path, pairs, must=True):
    p = Path(path)
    t = io.open(p, encoding="utf-8").read()
    n_applied = 0
    for old, new in pairs:
        if old not in t:
            if new in t:
                print(f"[already] {p.name}: {old[:60]!r}")
                continue
            if must:
                print(f"[FAIL] anchor not found in {p.name}: {old[:80]!r}")
                sys.exit(1)
            print(f"[skip] {p.name}: {old[:60]!r}")
            continue
        if t.count(old) != 1:
            print(f"[FAIL] anchor not unique ({t.count(old)}x) in {p.name}: {old[:60]!r}")
            sys.exit(1)
        t = t.replace(old, new)
        n_applied += 1
    io.open(p, "w", encoding="utf-8", newline="").write(t)
    print(f"[ok] {p.name}: {n_applied} edits applied")

# ============ 1. SOURCE manuscript body ============
body = FIN / "scripts/text/manuscript_body_r11.md"

s2_old = (
    "Discrimination varied more between hospitals than across subgroups and did not deteriorate over the first week. "
    "Among the {N:hosp_n} eICU-CRD hospitals with at least 20 positive checkpoints, within-hospital Tier-1 AUROC had a median of {N:hosp_med} "
    "(range {N:hosp_rng}); {N:hq_ge70} hospitals ({N:hq_ge70_pct}%) reached at least 0.70 and {N:hq_ge80} ({N:hq_ge80_pct}%) at least 0.80, "
    "whereas {N:hosp_lt60} fell below 0.60 (Supplementary Fig. 3). "
    "Across 24-hour checkpoint-hour strata, eICU-CRD AUROC was {N:mk_s0}, {N:mk_s1}, {N:mk_s2}, {N:mk_s3}, {N:mk_s4} and {N:mk_s5}, "
    "with an upward trend (exact Mann\u2013Kendall S = {N:mk_S}, \u03c4 = {N:mk_tau}, {N:mk_p}; Fig. 4b). "
    "Across the overall estimate and {N:sg_count} eICU-CRD subgroups (Fig. 4c) no point estimate fell below {N:sg_min}; "
    "discrimination was {N:sg_ckd_auc} with and {N:sg_nockd_auc} without chronic kidney disease, and {N:sg_tbi_auc}, {N:sg_sahich_auc} "
    "and {N:sg_isunspec_auc} in traumatic, haemorrhagic and ischaemic or unspecified stroke subtypes (all subgroups in Supplementary Material 5). "
    "The remaining {N:sg_other_stays} stays with other diagnoses are not a subgroup, and the TBI subgroup comprised {N:sg_tbi_pct}% of stays."
)
s2_new = (
    "Discrimination varied more between hospitals than across subgroups and did not deteriorate over the first week in either external cohort. "
    "Among the {N:hosp_n} eICU-CRD hospitals with at least 20 positive checkpoints, within-hospital Tier-1 AUROC had a median of {N:hosp_med} "
    "(range {N:hosp_rng}); {N:hq_ge70} hospitals ({N:hq_ge70_pct}%) reached at least 0.70 and {N:hq_ge80} ({N:hq_ge80_pct}%) at least 0.80, "
    "whereas {N:hosp_lt60} fell below 0.60 (Supplementary Fig. 3); no within-hospital view is possible in MIMIC-IV, a single-centre database. "
    "Across 24-hour checkpoint-hour strata, MIMIC-IV AUROC was {N:mk_m4_s0}, {N:mk_m4_s1}, {N:mk_m4_s2}, {N:mk_m4_s3}, {N:mk_m4_s4} and {N:mk_m4_s5} "
    "and eICU-CRD AUROC was {N:mk_s0}, {N:mk_s1}, {N:mk_s2}, {N:mk_s3}, {N:mk_s4} and {N:mk_s5}, with upward trends in both cohorts "
    "(exact Mann\u2013Kendall S = {N:mk_m4_S}, \u03c4 = {N:mk_m4_tau}, {N:mk_m4_p} in MIMIC-IV and S = {N:mk_S}, \u03c4 = {N:mk_tau}, {N:mk_p} in eICU-CRD; Fig. 4b, e). "
    "Across the overall estimate and {N:sg_count} subgroups in each external cohort (Fig. 4c, f) no point estimate fell below {N:sg_m4_min}; "
    "discrimination was {N:sg_m4_ckd_auc} with and {N:sg_m4_nockd_auc} without chronic kidney disease in MIMIC-IV and {N:sg_ckd_auc} and {N:sg_nockd_auc} "
    "in eICU-CRD, and {N:sg_m4_tbi_auc}, {N:sg_m4_sahich_auc} and {N:sg_m4_isunspec_auc} in traumatic, haemorrhagic and ischaemic or unspecified stroke "
    "subtypes in MIMIC-IV versus {N:sg_tbi_auc}, {N:sg_sahich_auc} and {N:sg_isunspec_auc} in eICU-CRD (all subgroups in Supplementary Material 5). "
    "The remaining {N:sg_m4_other_stays} MIMIC-IV and {N:sg_other_stays} eICU-CRD stays with other diagnoses are not subgroups, "
    "and the TBI subgroup comprised {N:sg_tbi_pct}% of stays in both external cohorts."
)

s3_old = (
    "Tier-1 discrimination was examined across nine subgroups in eICU-CRD (age, sex, diagnostic subtype and chronic kidney disease) "
    "with stay-level bootstrap intervals, across 24-hour checkpoint-hour strata ((24, 48] to (144, 168] h) with an exact Mann\u2013Kendall "
    "trend test over all 720 orderings {R:mann}, and across eICU-CRD hospitals with at least 20 positive checkpoints "
    "(within-hospital AUROC, shown in rank order without hospital identifiers)."
)
s3_new = (
    "Tier-1 discrimination was examined across nine subgroups in each external cohort (age, sex, diagnostic subtype and chronic kidney disease) "
    "with stay-level bootstrap intervals, across 24-hour checkpoint-hour strata ((24, 48] to (144, 168] h) with an exact Mann\u2013Kendall "
    "trend test over all 720 orderings {R:mann}, and across eICU-CRD hospitals with at least 20 positive checkpoints "
    "(within-hospital AUROC, shown in rank order without hospital identifiers; MIMIC-IV is a single-centre database). "
    "Subgroup covariates came from ICD-coded diagnoses: diagnostic subtype was assigned per patient from all coded MIMIC-IV admissions "
    "(priority order traumatic brain injury, subarachnoid haemorrhage, intracerebral haemorrhage, ischaemic or unspecified stroke), "
    "and renal disease followed the Quan coding algorithm on ICD-9 and, in MIMIC-IV, ICD-10 codes, excluding the acute kidney injury category."
)

s4_old = (
    "Eighth, the per-hospital estimates come without case-mix descriptors or event counts, so low-performing hospitals cannot be characterised; "
    "subgroup analyses were run in eICU-CRD only, and the logistic-regression comparison was run in MIMIC-IV only and its AUROC was not stored."
)
s4_new = (
    "Eighth, the per-hospital estimates come without case-mix descriptors or event counts, so low-performing hospitals cannot be characterised, "
    "and within-hospital analysis was possible only in eICU-CRD because MIMIC-IV is a single-centre database; "
    "the logistic-regression comparison was run in MIMIC-IV only and its AUROC was not stored; "
    "and subgroup covariates were derived from ICD-coded diagnoses in both cohorts, "
    "so misclassification of chronic kidney disease and diagnostic subtype cannot be excluded."
)

patch(body, [
    ("(Fig. 3d\u2013f; Fig. 4a; Table 4)", "(Fig. 3d\u2013f; Fig. 4a, d; Table 4)"),
    (s2_old, s2_new),
    (s3_old, s3_new),
    (s4_old, s4_new),
])

# ============ 2. SOURCE captions ============
cap = FIN / "scripts/text/captions_r11.md"
c_old = io.open(cap, encoding="utf-8").read()
start = c_old.find("**Fig. 4.")
end = c_old.find("\n", start)
old_legend = c_old[start:end]
new_legend = (
    "**Fig. 4. Warning lead time, stability over the first week and subgroups in the two external cohorts.** "
    "**a\u2013c** MIMIC-IV; **d\u2013f** eICU-CRD. "
    "**a**, **d** Lead time (median, symbols; IQR, bars) from the first qualifying alert to the stay's final positive checkpoint at five thresholds; "
    "percentages above the bars are the share of event stays captured. "
    "**b**, **e** Tier-1 AUROC in six 24-hour checkpoint-hour strata; dashed line, pooled cohort AUROC (MIMIC-IV {N:auc1_m4}; eICU-CRD {N:auc1_ei}); "
    "exact Mann\u2013Kendall S = {N:mk_m4_S}, \u03c4 = {N:mk_m4_tau}, {N:mk_m4_p} in MIMIC-IV and S = {N:mk_S}, \u03c4 = {N:mk_tau}, {N:mk_p} in eICU-CRD. "
    "**c**, **f** Tier-1 AUROC (95% CI) for the overall cohort and {N:sg_count} subgroups, with event stays/stays; dashed line, overall AUROC. "
    "TBI, traumatic brain injury; SAH, subarachnoid haemorrhage; ICH, intracerebral haemorrhage; IS, ischaemic stroke; CKD, chronic kidney disease."
)
if "two external cohorts" in old_legend:
    print("[already] captions_r11.md: Fig4 legend")
else:
    assert "Subgroup analyses were not run in MIMIC-IV" in old_legend
    patch(cap, [(old_legend, new_legend)])

# ============ 3. RESOLVED manuscript ============
res = FIN / "manuscript/MANUSCRIPT_npjDM_round11.md"
r2_old = (
    "Discrimination varied more between hospitals than across subgroups and did not deteriorate over the first week. "
    "Among the 50 eICU-CRD hospitals with at least 20 positive checkpoints, within-hospital Tier-1 AUROC had a median of 0.754 "
    "(range 0.518\u20130.877); 38 hospitals (76.0%) reached at least 0.70 and 10 (20.0%) at least 0.80, whereas 2 fell below 0.60 "
    "(Supplementary Fig. 3). Across 24-hour checkpoint-hour strata, eICU-CRD AUROC was 0.750, 0.762, 0.769, 0.802, 0.845 and 0.915, "
    "with an upward trend (exact Mann\u2013Kendall S = 15, \u03c4 = 1.000, P = 0.0028; Fig. 4b). "
    "Across the overall estimate and 9 eICU-CRD subgroups (Fig. 4c) no point estimate fell below 0.752; "
    "discrimination was 0.759 with and 0.767 without chronic kidney disease, and 0.757, 0.778 and 0.757 in traumatic, hemorrhagic and "
    "ischaemic or unspecified stroke subtypes (all subgroups in Supplementary Material 5). "
    "The remaining 397 stays with other diagnoses are not a subgroup, and the TBI subgroup comprised 25.0% of stays."
)
r2_new = (
    "Discrimination varied more between hospitals than across subgroups and did not deteriorate over the first week in either external cohort. "
    "Among the 50 eICU-CRD hospitals with at least 20 positive checkpoints, within-hospital Tier-1 AUROC had a median of 0.754 "
    "(range 0.518\u20130.877); 38 hospitals (76.0%) reached at least 0.70 and 10 (20.0%) at least 0.80, whereas 2 fell below 0.60 "
    "(Supplementary Fig. 3); no within-hospital view is possible in MIMIC-IV, a single-center database. "
    "Across 24-hour checkpoint-hour strata, MIMIC-IV AUROC was 0.757, 0.764, 0.761, 0.754, 0.800 and 0.833 and eICU-CRD AUROC was "
    "0.750, 0.762, 0.769, 0.802, 0.845 and 0.915, with upward trends in both cohorts "
    "(exact Mann\u2013Kendall S = 11, \u03c4 = 0.467, P = 0.2722 in MIMIC-IV and S = 15, \u03c4 = 1.000, P = 0.0028 in eICU-CRD; Fig. 4b, e). "
    "Across the overall estimate and nine subgroups in each external cohort (Fig. 4c, f) no point estimate fell below 0.739; "
    "discrimination was 0.751 with and 0.739 without chronic kidney disease in MIMIC-IV and 0.759 and 0.767 in eICU-CRD, "
    "and 0.762, 0.762 and 0.775 in traumatic, hemorrhagic and ischemic or unspecified stroke subtypes in MIMIC-IV "
    "versus 0.757, 0.778 and 0.757 in eICU-CRD (all subgroups in Supplementary Material 5). "
    "The remaining 592 MIMIC-IV and 397 eICU-CRD stays with other diagnoses are not subgroups, "
    "and the TBI subgroup comprised 25.0% of stays in both external cohorts."
)
r3_old = (
    "Tier-1 discrimination was examined across nine subgroups in eICU-CRD (age, sex, diagnostic subtype and chronic kidney disease) "
    "with stay-level bootstrap intervals, across 24-hour checkpoint-hour strata ((24, 48] to (144, 168] h) with an exact Mann\u2013Kendall "
    "trend test over all 720 orderings [23], and across eICU-CRD hospitals with at least 20 positive checkpoints "
    "(within-hospital AUROC, shown in rank order without hospital identifiers)."
)
r3_new = (
    "Tier-1 discrimination was examined across nine subgroups in each external cohort (age, sex, diagnostic subtype and chronic kidney disease) "
    "with stay-level bootstrap intervals, across 24-hour checkpoint-hour strata ((24, 48] to (144, 168] h) with an exact Mann\u2013Kendall "
    "trend test over all 720 orderings [23], and across eICU-CRD hospitals with at least 20 positive checkpoints "
    "(within-hospital AUROC, shown in rank order without hospital identifiers; MIMIC-IV is a single-center database). "
    "Subgroup covariates came from ICD-coded diagnoses: diagnostic subtype was assigned per patient from all coded MIMIC-IV admissions "
    "(priority order traumatic brain injury, subarachnoid hemorrhage, intracerebral hemorrhage, ischemic or unspecified stroke), "
    "and renal disease followed the Quan coding algorithm on ICD-9 and, in MIMIC-IV, ICD-10 codes, excluding the acute kidney injury category."
)
r4_old = (
    "Eighth, the per-hospital estimates come without case-mix descriptors or event counts, so low-performing hospitals cannot be characterized; "
    "subgroup analyzes were run in eICU-CRD only, and the logistic-regression comparison was run in MIMIC-IV only and its AUROC was not stored."
)
r4_new = (
    "Eighth, the per-hospital estimates come without case-mix descriptors or event counts, so low-performing hospitals cannot be characterized, "
    "and within-hospital analysis was possible only in eICU-CRD because MIMIC-IV is a single-center database; "
    "the logistic-regression comparison was run in MIMIC-IV only and its AUROC was not stored; "
    "and subgroup covariates were derived from ICD-coded diagnoses in both cohorts, "
    "so misclassification of chronic kidney disease and diagnostic subtype cannot be excluded."
)
# resolved legend: whole-line replace by anchor
res_t = io.open(res, encoding="utf-8").read()
ls = res_t.find("**Fig. 4.")
le = res_t.find("\n", ls)
r5_old = res_t[ls:le]
r5_new = (
    "**Fig. 4. Warning lead time, stability over the first week and subgroups in the two external cohorts.** "
    "**a\u2013c** MIMIC-IV; **d\u2013f** eICU-CRD. "
    "**a**, **d** Lead time (median, symbols; IQR, bars) from the first qualifying alert to the stay's final positive checkpoint at five thresholds; "
    "percentages above the bars are the share of event stays captured. "
    "**b**, **e** Tier-1 AUROC in six 24-hour checkpoint-hour strata; dashed line, pooled cohort AUROC (MIMIC-IV 0.769; eICU-CRD 0.777); "
    "exact Mann\u2013Kendall S = 11, \u03c4 = 0.467, P = 0.2722 in MIMIC-IV and S = 15, \u03c4 = 1.000, P = 0.0028 in eICU-CRD. "
    "**c**, **f** Tier-1 AUROC (95% CI) for the overall cohort and nine subgroups, with event stays/stays; dashed line, overall AUROC. "
    "TBI, traumatic brain injury; SAH, subarachnoid hemorrhage; ICH, intracerebral hemorrhage; IS, ischemic stroke; CKD, chronic kidney disease."
)
assert "Warning lead time" in r5_old
# residual v1 noun error: "analyzes" (misconverted) -> "analyses" (all remaining instances are nouns)
n_analyzes = res_t.count("analyzes")
patch(res, [
    ("(Fig. 3d\u2013f; Fig. 4a; Table 4)", "(Fig. 3d\u2013f; Fig. 4a, d; Table 4)"),
    (r2_old, r2_new),
    (r3_old, r3_new),
    (r4_old, r4_new),
    (r5_old, r5_new),
])
if n_analyzes:
    import re as _re
    t2 = io.open(res, encoding="utf-8").read()
    for m in _re.finditer(r"\banalyzes\b", t2):
        print(f"  [ctx] ...{t2[max(0, m.start()-40):m.end()+30]}...")
    t2 = t2.replace("analyzes", "analyses")
    io.open(res, "w", encoding="utf-8", newline="").write(t2)
    print(f"[ok] MANUSCRIPT_npjDM_round11.md: {n_analyzes}x analyzes->analyses (v1 noun-fix, aligns with submission docx)")

# ============ 4. RESOLVED SI ============
si = FIN / "supplementary/Supplementary_Information_round11.md"
si_t = io.open(si, encoding="utf-8").read()
sub_h = "## Subgroups (eICU-CRD, Tier 1; overall and nine subgroups)"
foot_old = ("*The overall row and the nine subgroups are shown in Fig. 4c. Subgroup intervals are stay-level bootstrap intervals; "
            "no interaction tests were performed. The 397 stays with other diagnoses belong to no diagnostic-subtype subgroup.*")
foot_new = ("*The overall row and the nine subgroups of each cohort are shown in Fig. 4c and f. Subgroup intervals are stay-level "
            "bootstrap intervals; no interaction tests were performed. The 397 eICU-CRD stays and 592 MIMIC-IV stays with other "
            "diagnoses belong to no diagnostic-subtype subgroup. MIMIC-IV subgroup covariates were derived from ICD-coded diagnoses "
            "(renal disease by the Quan coding algorithm, excluding the acute kidney injury category).*")
m4_table = (
    "## Subgroups (external cohorts, Tier 1; overall and nine subgroups each)\n\n"
    "**MIMIC-IV**\n\n"
    "| Subgroup                                  | AUROC (95% CI)      | Stays / event stays   |\n"
    "|:------------------------------------------|:--------------------|:----------------------|\n"
    "| Overall                                   | 0.769 (0.748\u20130.791) | 7,442 / 731           |\n"
    "| Age <65 years                             | 0.773 (0.739\u20130.803) | 3,420 / 343           |\n"
    "| Age \u226565 years                             | 0.767 (0.739\u20130.794) | 4,022 / 388           |\n"
    "| Male                                      | 0.777 (0.751\u20130.800) | 4,160 / 437           |\n"
    "| Female                                    | 0.758 (0.725\u20130.792) | 3,282 / 294           |\n"
    "| Traumatic brain injury                    | 0.762 (0.712\u20130.808) | 1,860 / 142           |\n"
    "| Subarachnoid or intracerebral hemorrhage | 0.762 (0.729\u20130.798) | 2,745 / 270           |\n"
    "| Ischemic or unspecified stroke            | 0.775 (0.741\u20130.808) | 2,245 / 244           |\n"
    "| Chronic kidney disease                    | 0.751 (0.713\u20130.782) | 1,381 / 305           |\n"
    "| No chronic kidney disease                 | 0.739 (0.715\u20130.762) | 6,061 / 426           |\n\n"
    "**eICU-CRD**\n\n"
)
tl_old = ("External eICU-CRD Tier-1 AUROC by 24-hour checkpoint-hour stratum ((24, 48] to (144, 168] h) was 0.750, 0.762, 0.769, 0.802, 0.845 and 0.915. "
          "Exact two-sided Mann\u2013Kendall test over all 720 orderings: S = 15, \u03c4 = 1.000, P = 0.0028 (asymptotic P = 0.003). "
          "The trend is upward; there is no evidence of deterioration over the first week.")
tl_new = ("External MIMIC-IV Tier-1 AUROC by 24-hour checkpoint-hour stratum ((24, 48] to (144, 168] h) was 0.757, 0.764, 0.761, 0.754, 0.800 and 0.833. "
          "Exact two-sided Mann\u2013Kendall test over all 720 orderings: S = 11, \u03c4 = 0.467, P = 0.2722 (asymptotic P = 0.272). "
          "External eICU-CRD Tier-1 AUROC over the same strata was 0.750, 0.762, 0.769, 0.802, 0.845 and 0.915 "
          "(exact test S = 15, \u03c4 = 1.000, P = 0.0028; asymptotic P = 0.003). "
          "The trend is upward in both cohorts, statistically significant only in eICU-CRD; there is no evidence of deterioration over the first week.")
patch(si, [
    (sub_h + "\n\n", m4_table),
    (foot_old, foot_new),
    (tl_old, tl_new),
    ("| Ischaemic or unspecified stroke", "| Ischemic or unspecified stroke"),
])

# ============ 5. SOURCE SM5 ============
sm5 = FIN / "scripts/text/SM5_sensitivity_r11.md"
sm5_h = "## Subgroups (eICU-CRD, Tier 1; overall and nine subgroups)"
sm5_m4 = (
    "## Subgroups (external cohorts, Tier 1; overall and nine subgroups each)\n\n"
    "**MIMIC-IV** (f27_m4_subgroups_mk.json; covariates from ICD-coded diagnoses, renal disease by the Quan algorithm excluding the acute kidney injury category)\n\n"
    "| Subgroup | AUROC (95% CI) | Stays / event stays |\n"
    "|:---|:---|:---|\n"
    "| Overall | {N:sg_m4_overall} | {N:sg_m4_overall_n} |\n"
    "| Age <65 years | {N:sg_m4_age_lt65} | {N:sg_m4_age_lt65_n} |\n"
    "| Age \u226565 years | {N:sg_m4_age_ge65} | {N:sg_m4_age_ge65_n} |\n"
    "| Male | {N:sg_m4_male} | {N:sg_m4_male_n} |\n"
    "| Female | {N:sg_m4_female} | {N:sg_m4_female_n} |\n"
    "| Traumatic brain injury | {N:sg_m4_tbi} | {N:sg_m4_tbi_n} |\n"
    "| Subarachnoid or intracerebral haemorrhage | {N:sg_m4_sahich} | {N:sg_m4_sahich_n} |\n"
    "| Ischaemic or unspecified stroke | {N:sg_m4_isunspec} | {N:sg_m4_isunspec_n} |\n"
    "| Chronic kidney disease | {N:sg_m4_ckd} | {N:sg_m4_ckd_n} |\n"
    "| No chronic kidney disease | {N:sg_m4_nockd} | {N:sg_m4_nockd_n} |\n\n"
    "**eICU-CRD**\n\n"
)
sm5_foot_old = ("*The overall row and the nine subgroups are shown in Fig. 4c. Subgroup intervals are stay-level bootstrap intervals; "
                "no interaction tests were performed. The {N:sg_other_stays} stays with other diagnoses belong to no diagnostic-subtype subgroup.*")
sm5_foot_new = ("*The overall row and the nine subgroups of each cohort are shown in Fig. 4c and f. Subgroup intervals are stay-level "
                "bootstrap intervals; no interaction tests were performed. The {N:sg_other_stays} eICU-CRD stays and {N:sg_m4_other_stays} "
                "MIMIC-IV stays with other diagnoses belong to no diagnostic-subtype subgroup.*")
sm5_tl_old = ("External eICU-CRD Tier-1 AUROC by 24-hour checkpoint-hour stratum ((24, 48] to (144, 168] h) was {N:mk_s0}, {N:mk_s1}, {N:mk_s2}, "
              "{N:mk_s3}, {N:mk_s4} and {N:mk_s5}. Exact two-sided Mann\u2013Kendall test over all 720 orderings: S = {N:mk_S}, \u03c4 = {N:mk_tau}, "
              "{N:mk_p} (asymptotic {N:mk_p_asym}). The trend is upward; there is no evidence of deterioration over the first week.")
sm5_tl_new = ("External MIMIC-IV Tier-1 AUROC by 24-hour checkpoint-hour stratum ((24, 48] to (144, 168] h) was {N:mk_m4_s0}, {N:mk_m4_s1}, "
              "{N:mk_m4_s2}, {N:mk_m4_s3}, {N:mk_m4_s4} and {N:mk_m4_s5}. Exact two-sided Mann\u2013Kendall test over all 720 orderings: "
              "S = {N:mk_m4_S}, \u03c4 = {N:mk_m4_tau}, {N:mk_m4_p} (asymptotic {N:mk_m4_p_asym}). "
              "External eICU-CRD Tier-1 AUROC over the same strata was {N:mk_s0}, {N:mk_s1}, {N:mk_s2}, {N:mk_s3}, {N:mk_s4} and {N:mk_s5} "
              "(exact test S = {N:mk_S}, \u03c4 = {N:mk_tau}, {N:mk_p}; asymptotic {N:mk_p_asym}). "
              "The trend is upward in both cohorts, statistically significant only in eICU-CRD; there is no evidence of deterioration over the first week.")
patch(sm5, [
    (sm5_h + "\n\n", sm5_m4),
    (sm5_foot_old, sm5_foot_new),
    (sm5_tl_old, sm5_tl_new),
])

print("\nALL TEXT PATCHES APPLIED")
