# -*- coding: utf-8 -*-
"""JMIR progress completion: apply the full accumulated backlog (npjDM post-pause fixes, adapted)."""
import re

P = "MANUSCRIPT_JMIR.md"
t = open(P, encoding="utf-8").read()

# ---- step 1: shift Chen [33]->[34] BEFORE inserting Cao as [33] ----
body, rest = t.split("## References", 1)
body = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]",
              lambda m: "[" + ",".join(str(34 if int(x.strip()) == 33 else int(x.strip())) for x in m.group(1).split(",")) + "]",
              body)

edits = [
    # title 171-of-180
    ("full-KDIGO endpoint, zero-touch validation across 180 hospitals",
     "full-KDIGO endpoint, zero-touch validation across 171 of 180 hospitals"),
    # Intro incidence rewrite (evidence-grounded) + AKI/ICU expansions
    ("In this population AKI is common and consequential—it complicates 8 to 23% of ICU stays after traumatic brain injury and up to 30% after non-traumatic acute brain injury [2,3], lengthens mechanical ventilation, increases length of stay, and independently raises mortality, with mortality remaining elevated in contemporary traumatic-brain-injury AKI series [4] and AKI staying associated with death after adjustment for injury severity in recent cohort analyses [5].",
     "In this population acute kidney injury (AKI) is common and consequential—it complicates roughly one in ten intensive care unit (ICU) stays after traumatic brain injury (10.6% and 10.9% in two contemporary cohorts [2,3]), lengthens mechanical ventilation, increases length of stay, and independently raises mortality (adjusted odds ratio 2.2 in a multicentre ICU registry [2]; mortality remaining elevated in contemporary traumatic-brain-injury AKI series [4]; and adverse-outcome associations documented in recent cohort analyses [5])."),
    ("after aneurysmal subarachnoid haemorrhage, AKI affects a substantial minority of patients",
     "after aneurysmal subarachnoid haemorrhage, AKI affects more than one in ten patients"),
    ("after intracerebral haemorrhage it is similarly prevalent and consequential [7]",
     "after intracerebral haemorrhage it develops in about 15% of hospitalizations nationally, with higher in-hospital mortality when it occurs [7]"),
    # KDIGO expansion
    ("the urine-output arm of the KDIGO classification",
     "the urine-output arm of the Kidney Disease: Improving Global Outcomes (KDIGO) classification"),
    # eICU-CRD expansion
    ("zero-touch external validation in 171 eICU-CRD hospitals",
     "zero-touch external validation in 171 hospitals of the eICU Collaborative Research Database (eICU-CRD)"),
    # IQR expansion
    ("(IQR 54–77)", "(interquartile range [IQR] 54–77)"),
    # RRT expansion
    ("Patients on renal replacement therapy before the first checkpoint were excluded",
     "Patients on renal replacement therapy (RRT) before the first checkpoint were excluded"),
    # ECE expansion
    ("reduced expected calibration error from 0.059 to 0.013",
     "reduced expected calibration error (ECE) from 0.059 to 0.013"),
    # EHR rephrase
    ("increasingly the standard against which EHR-only models are judged",
     "increasingly the standard against which models restricted to electronic health record data are judged"),
    # lower-AUROC: 0-hour precision + First re-attribution
    ("Our AUROCs are nonetheless lower than some published values in this space—including the deep-learning AKI system recently validated under simulated continuous monitoring conditions in npj Digital Medicine (external AUROC 0.956–0.963) [29] and a multitask perioperative model that includes AKI among its outcomes (0.789–0.863 externally) [30]—and we think the reasons are informative rather than discouraging.",
     "Our AUROCs are nonetheless lower than some published values in this space—including the deep-learning AKI system recently validated under simulated continuous monitoring conditions in npj Digital Medicine, whose headline external AUROCs of 0.956–0.963 belong to its 0-hour detection models—nowcasts of AKI as it unfolds—while its 48-hour-horizon external trajectory ranges down to 0.749 [29], and a multitask perioperative model that includes AKI among its outcomes (0.789–0.863 externally) [30]—and we think the reasons are informative rather than discouraging."),
    ("First, the complete KDIGO endpoint is harder: adding the urine-output arm identifies 358 events",
     "First, against the creatinine-only tools cited above, the complete KDIGO endpoint is harder: adding the urine-output arm identifies 358 events"),
    # MOD-03 variability mechanism (Zhang = [10] in JMIR)
    ("supporting the inclusion of physiological-variability signals alongside level features.",
     "supporting the inclusion of physiological-variability signals alongside level features. The pattern—variability, not just level, carrying prognostic weight—fits the broader observation that loss of physiological complexity precedes clinically manifest deterioration: in a sepsis trajectory model built on the same public databases, reduced heart-rate variability independently predicted mortality [10], and systolic-blood-pressure variability appears to play the analogous role for the renal endpoint this system watches."),
    # deployment para: faithfulness + Cao equity
    ("are exactly the quantities a hospital governance committee must weigh when approving or declining an alert. At matched sensitivity",
     "are exactly the quantities a hospital governance committee must weigh when approving or declining an alert. The system's flat performance across the monitoring timeline (Mann-Kendall P = 0.24) means its risk does not need to mature toward onset—unlike detection-style models, whose confidence is designed to rise as the event approaches [29]—so a first-day alert is as trustworthy as a fifth-day one. Nor is deployment a luxury of large centres: hospitals with too few renal events to train a local model are precisely those that benefit most from a shared one [33], and the zero-touch form of this system—frozen model, frozen recalibration layer, no local labels required—is the form in which prediction reaches them. At matched sensitivity"),
    # 180 unification (2 body spots)
    ("Does performance transport without refitting to 180 external hospitals?",
     "Does performance transport without refitting to 171 of the 180 crosswalk hospitals?"),
    ("and a zero-touch 180-hospital external test introduces cross-system variation that random-split validations do not.",
     "and a zero-touch external test across 171 of the 180 crosswalk hospitals introduces cross-system variation that random-split validations do not."),
    # Limitations direction clauses (5)
    ("First, both databases are retrospective and North American; the prospective value of earlier risk identification requires silent-mode evaluation",
     "First, both databases are retrospective and North American; whether earlier risk identification helps, harms, or is simply ignored cannot be known from these data—the direction of the outcome effect is itself unknown—and the prospective value of earlier identification therefore requires silent-mode evaluation"),
    ("a limitation we address through the learning-curve analysis and disclose in the Discussion.",
     "a limitation we address through the learning-curve analysis and disclose in the Discussion; any cost of the shortfall would run toward underestimating discrimination, and the flat learning curve is the direct evidence that it is not binding."),
    ("a cross-generational validation (for example, MIMIC-III CareVue) was precluded by a project-level data-use restriction.",
     "a cross-generational validation (for example, MIMIC-III CareVue) was precluded by a project-level data-use restriction; a second external system could raise or lower the point estimate, and the direction cannot be anticipated."),
    ("but the difference is a source of feature-scale shift that we disclose.",
     "but the difference is a source of feature-scale shift whose induced bias we cannot sign a priori, and we disclose it."),
    ("Finally, SHAP interaction analysis was computationally infeasible at 291 features and was not performed.",
     "Finally, SHAP interaction analysis was computationally infeasible at 291 features and was not performed, so interaction effects that could qualify the single-feature attributions remain unquantified."),
    # colloquialisms
    ("that a creatinine-only model would never see coming", "that a creatinine-only model cannot anticipate"),
    ("falls short of the rule of thumb", "falls short of the conventional threshold"),
    # LLM disclosure
    ("No patient or public involvement contributed to the design, conduct, or reporting of this study.",
     "In preparing this manuscript, the authors used a large language model to assist with drafting and editing under the direction and supervision of the authors; all model-assisted text was reviewed, revised and approved by the authors, who take full responsibility for the content of the manuscript. No patient or public involvement contributed to the design, conduct, or reporting of this study."),
    # Vickers + Moons PMIDs
]
for a, b in edits:
    c = body.count(a)
    assert c == 1, f"anchor x{c}: {a[:60]}"
    body = body.replace(a, b)
print(f"[text] {len(edits)} edits applied")

for a, b in [("PMID: 17099594", "PMID: 17099194"),
            ("prediction models using regression or artificial intelligence methods. BMJ. 2025;388:e082505.",
             "prediction models using regression or artificial intelligence methods. BMJ. 2025;388:e082505. PMID: 40127903.")]:
    c = rest.count(a); assert c == 1, f"rest anchor x{c}: {a[:50]}"
    rest = rest.replace(a, b)
print("[PMIDs] Vickers 17099194 + Moons 40127903 fixed")

# ---- step 2: uppercase panels in body + legends ----
body, n1 = re.subn(r"Figure (\d)a-b\b", r"Figure \1A,B", body)
body, n2 = re.subn(r"Figure (\d)([a-c])\b", lambda m: "Figure " + m.group(1) + m.group(2).upper(), body)
body, l0 = re.subn(r"\(a, b\)", "(A, B)", body)
body, l1 = re.subn(r"\(a\)", "(A)", body)
body, l2 = re.subn(r"\(b\)", "(B)", body)
body, l3 = re.subn(r"\(c\)", "(C)", body)
body, l4 = re.subn(r"\(a; slope 0.33 raw → 0.91 recalibrated\)", "(A; slope 0.33 raw → 0.91 recalibrated)", body)
body, l5 = re.subn(r"\(b; slope 0.40 → 1.11\)", "(B; slope 0.40 → 1.11)", body)
body, l6 = re.subn(r"In b, a post-hoc", "In B, a post-hoc", body)
print(f"[panels] body {n1}+{n2}, legends {l0}+{l1}+{l2}+{l3}+{l4}+{l5}+{l6}")

# ---- step 3: legend abbreviation keys (10; Fig7 none - no abbreviations used) ----
AU = "AUROC, area under the receiver operating characteristic curve"
keys = {
    "red marker the AKI onset. ": f"Abbreviations: ABI, acute brain injury; AKI, acute kidney injury; ICU, intensive care unit; KDIGO, Kidney Disease: Improving Global Outcomes.",
    "paired bootstrap over 1,000 resamples). ": f"Abbreviations: {AU}; CI, confidence interval.",
    "threshold range in both databases. ": "Abbreviations: ECE, expected calibration error.",
    "a monitoring-intensity proxy disclosed as a care-bias limitation). ": "Abbreviations: SHAP, SHapley Additive exPlanations.",
    "performance is usable from the first checkpoint. ": f"Abbreviations: AKI, acute kidney injury; {AU}; NNE, number needed to evaluate; PPV, positive predictive value.",
    "sex differences were ≤ 0.018 in both databases. ": f"Abbreviations: {AU}.",
    "reference dashboard numbers correspond to Table 2. ": f"Abbreviations: {AU}.",
    "which preserves the original sample space. ": f"Abbreviations: {AU}; SMOTE, synthetic minority oversampling technique.",
    "trained under the fixed-budget protocol disclosed in the text. ": f"Abbreviations: {AU}; CNN, convolutional neural network; LSTM, long short-term memory; SD, standard deviation.",
    "reported separately throughout (Table 4 Panels A and B). ": "Abbreviations: AKI, acute kidney injury; PPV, positive predictive value.",
}
_leg_i = body.index("## Figure Legends")
body_head, body_legs = body[:_leg_i], body[_leg_i:]
for anchor, key in keys.items():
    a = anchor.rstrip()   # paragraph-end anchors carry no trailing space
    c = body_legs.count(a)
    assert c == 1, f"key anchor x{c}: {a[:50]}"
    body_legs = body_legs.replace(a, a + " " + key)
body = body_head + body_legs
print(f"[keys] {len(keys)} legend keys appended")

# ---- step 4: ref list rebuild: insert Cao at 33, Chen -> 34 ----
entries, cur = {}, None
tail_split = re.search(r"\n## ", rest[1:])
tail = rest[1:][tail_split.start():].lstrip("\n") if tail_split else ""
for line in rest.split("\n")[1:]:
    if line.startswith("## "):
        break
    mm = re.match(r"^(\d+)\.\s", line)
    if mm:
        cur = int(mm.group(1)); entries[cur] = [line]
    elif cur is not None and line.strip():
        entries[cur].append(line)
assert sorted(entries) == list(range(1, 34)), sorted(entries)[-3:]
CAO = ("33. Cao J, et al. Exploring the limits of localization: federated model stacking improves hospital-level prediction in a "
       "national research network. npj Digit Med. 2026;9(1):492. PMID: 42032114.")
out = []
for n in range(1, 33):
    blk = list(entries[n]); blk[0] = re.sub(r"^\d+\.", str(n) + ".", blk[0], count=1); out += blk + [""]
out += [CAO, ""]
blk = list(entries[33]); blk[0] = re.sub(r"^\d+\.", "34.", blk[0], count=1); out += blk + [""]
new_refs = "## References\n\n" + "\n".join(out).rstrip() + "\n\n"
open(P, "w", encoding="utf-8").write(body.rstrip() + "\n\n" + new_refs + tail)
print("[refs] Cao=33 inserted, Chen->34; list now 34")
