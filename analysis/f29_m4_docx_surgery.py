# -*- coding: utf-8 -*-
"""f29: submission MANUSCRIPT.docx surgery for the M4-first Fig4 redesign (PI order 2026-10-05).
Five edits, all anchored and asserted against the docx's own text:
  para 33  (Fig. 4a -> Fig. 4a, d citation)
  para 38  trend/subgroup paragraph rewritten (M4 before eICU)
  para 50  limitations eighth clause rewritten
  para 68  Methods subgroup sentence rewritten (+covariate derivation)
  para 104 Fig4 legend rebuilt as 6-panel two-cohort legend with bold run structure
"""
from docx import Document

P = r"E:/TBI subtype/09_tbi_aki/_submission_npjDM/MANUSCRIPT.docx"
d = Document(P)
EN = "\u2013"; TAU = "\u03c4"; AP = "\u2019"

def find_para(anchor):
    for i, par in enumerate(d.paragraphs):
        if anchor in par.text:
            return i, par
    raise SystemExit(f"anchor not found: {anchor[:60]}")

def rewrite_single_run(par, old_substr, new_text, full_old=None):
    t = par.text
    if full_old is not None:
        assert t == full_old, f"paragraph text drift:\n{t[:200]}"
    else:
        assert old_substr in t, f"substr missing: {old_substr[:60]}"
        new_text = t.replace(old_substr, new_text) if isinstance(new_text, str) else None
    assert len(par.runs) >= 1
    par.runs[0].text = new_text if full_old is not None else new_text
    for r in par.runs[1:]:
        r.text = ""

# ---- 1. para: deployment citation ----
i, par = find_para("(Fig. 3d\u2013f; Fig. 4a; Table 4)")
rewrite_single_run(par, "(Fig. 3d\u2013f; Fig. 4a; Table 4)", "(Fig. 3d\u2013f; Fig. 4a, d; Table 4)")
print(f"[1 ok] para {i}: Fig. 4a, d")

# ---- 2. trend/subgroup paragraph ----
i, par = find_para("Across the overall estimate and 9 eICU-CRD subgroups")
new38 = (
    "Discrimination varied more between hospitals than across subgroups and did not deteriorate over the first week in either external cohort. "
    "Among the 50 eICU-CRD hospitals with at least 20 positive checkpoints, within-hospital Tier-1 AUROC had a median of 0.754 "
    f"(range 0.518{EN}0.877); 38 hospitals (76.0%) reached at least 0.70 and 10 (20.0%) at least 0.80, whereas 2 fell below 0.60 "
    "(Supplementary Fig. 3); no within-hospital view is possible in MIMIC-IV, a single-center database. "
    "Across 24-hour checkpoint-hour strata, MIMIC-IV AUROC was 0.757, 0.764, 0.761, 0.754, 0.800 and 0.833 and eICU-CRD AUROC was "
    f"0.750, 0.762, 0.769, 0.802, 0.845 and 0.915, with upward trends in both cohorts "
    f"(exact Mann{EN}Kendall S = 11, {TAU} = 0.467, P = 0.2722 in MIMIC-IV and S = 15, {TAU} = 1.000, P = 0.0028 in eICU-CRD; Fig. 4b, e). "
    "Across the overall estimate and nine subgroups in each external cohort (Fig. 4c, f) no point estimate fell below 0.739; "
    "discrimination was 0.751 with and 0.739 without chronic kidney disease in MIMIC-IV and 0.759 and 0.767 in eICU-CRD, "
    "and 0.762, 0.762 and 0.775 in traumatic, hemorrhagic and ischemic or unspecified stroke subtypes in MIMIC-IV "
    "versus 0.757, 0.778 and 0.757 in eICU-CRD (all subgroups in Supplementary Material 5). "
    "The remaining 592 MIMIC-IV and 397 eICU-CRD stays with other diagnoses are not subgroups, "
    "and the TBI subgroup comprised 25.0% of stays in both external cohorts."
)
assert "did not deteriorate over the first week." in par.text  # old topic sentence
par.runs[0].text = new38
for r in par.runs[1:]: r.text = ""
print(f"[2 ok] para {i}: trend/subgroup paragraph")

# ---- 3. limitations eighth ----
i, par = find_para("Eighth, the per-hospital estimates")
t = par.text
s = t.find("Eighth,"); e = t.find("Ninth,")
assert s > 0 and e > s
old8 = t[s:e]
assert "subgroup analyses were run in eICU-CRD only" in old8
new8 = (
    "Eighth, the per-hospital estimates come without case-mix descriptors or event counts, so low-performing hospitals cannot be characterized, "
    "and within-hospital analysis was possible only in eICU-CRD because MIMIC-IV is a single-center database; "
    "the logistic-regression comparison was run in MIMIC-IV only and its AUROC was not stored; "
    "and subgroup covariates were derived from ICD-coded diagnoses in both cohorts, "
    "so misclassification of chronic kidney disease and diagnostic subtype cannot be excluded. "
)
par.runs[0].text = t[:s] + new8 + t[e:]
for r in par.runs[1:]: r.text = ""
print(f"[3 ok] para {i}: limitations eighth")

# ---- 4. methods subgroup sentence ----
i, par = find_para("Tier-1 discrimination was examined across nine subgroups in eICU-CRD")
old_m = (
    "Tier-1 discrimination was examined across nine subgroups in eICU-CRD (age, sex, diagnostic subtype and chronic kidney disease) "
    f"with stay-level bootstrap intervals, across 24-hour checkpoint-hour strata ((24, 48] to (144, 168] h) with an exact Mann{EN}Kendall "
    "trend test over all 720 orderings [23], and across eICU-CRD hospitals with at least 20 positive checkpoints "
    "(within-hospital AUROC, shown in rank order without hospital identifiers)."
)
new_m = (
    "Tier-1 discrimination was examined across nine subgroups in each external cohort (age, sex, diagnostic subtype and chronic kidney disease) "
    f"with stay-level bootstrap intervals, across 24-hour checkpoint-hour strata ((24, 48] to (144, 168] h) with an exact Mann{EN}Kendall "
    "trend test over all 720 orderings [23], and across eICU-CRD hospitals with at least 20 positive checkpoints "
    "(within-hospital AUROC, shown in rank order without hospital identifiers; MIMIC-IV is a single-center database). "
    "Subgroup covariates came from ICD-coded diagnoses: diagnostic subtype was assigned per patient from all coded MIMIC-IV admissions "
    "(priority order traumatic brain injury, subarachnoid hemorrhage, intracerebral hemorrhage, ischemic or unspecified stroke), "
    "and renal disease followed the Quan coding algorithm on ICD-9 and, in MIMIC-IV, ICD-10 codes, excluding the acute kidney injury category."
)
assert old_m in par.text, "methods anchor mismatch"
par.runs[0].text = par.text.replace(old_m, new_m)
for r in par.runs[1:]: r.text = ""
print(f"[4 ok] para {i}: methods sentence")

# ---- 5. Fig4 legend rebuild ----
i, par = find_para("Warning lead time")
segs = [
    ("Fig. 4. Warning lead time, stability over the first week and subgroups in the two external cohorts.", True),
    (" ", False),
    (f"a{EN}c", True),
    (" MIMIC-IV; ", False),
    ("d" + EN + "f", True),
    (" eICU-CRD. ", False),
    ("a", True), (", ", False), ("d", True),
    (f" Lead time (median, symbols; IQR, bars) from the first qualifying alert to the stay{AP}s final positive checkpoint at five thresholds; "
     "percentages above the bars are the share of event stays captured. ", False),
    ("b", True), (", ", False), ("e", True),
    (f" Tier-1 AUROC in six 24-hour checkpoint-hour strata; dashed line, pooled cohort AUROC (MIMIC-IV 0.769; eICU-CRD 0.777); "
     f"exact Mann{EN}Kendall S = 11, {TAU} = 0.467, P = 0.2722 in MIMIC-IV and S = 15, {TAU} = 1.000, P = 0.0028 in eICU-CRD. ", False),
    ("c", True), (", ", False), ("f", True),
    (" Tier-1 AUROC (95% CI) for the overall cohort and nine subgroups, with event stays/stays; dashed line, overall AUROC. "
     "TBI, traumatic brain injury; SAH, subarachnoid hemorrhage; ICH, intracerebral hemorrhage; IS, ischemic stroke; "
     "CKD, chronic kidney disease.", False),
]
for r in list(par.runs):
    r._element.getparent().remove(r._element)
for text, bold in segs:
    r = par.add_run(text)
    r.bold = bold
print(f"[5 ok] para {i}: Fig4 legend ({len(segs)} runs)")

d.save(P)
print("\nDOCX SAVED:", P)
