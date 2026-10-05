# -*- coding: utf-8 -*-
"""journal-benchmark-review Step 8: execute MOD-01..06 on the npjDM package."""
import json, re
from datetime import datetime

SF = "submission_npjDM/REVISION_STATUS.json"
status = {"phase": "journal_benchmark_review", "target_journal": "npj Digital Medicine",
          "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "total": 6, "completed": [], "skipped": [], "failed": [], "current": "MOD-01",
          "items": {f"MOD-0{i}": {"status": "pending", "priority": p, "description": d}
                    for i, (p, d) in enumerate([
              ("P0", "cover letter Lee fact correction (full-KDIGO + 0-h detection caliber + 48-h 0.749)"),
              ("P0", "manuscript lower-AUROC paragraph horizon precision + First-argument re-attribution"),
              ("P1", "SHAP variability-mechanism sentences (existing Zhang ref)"),
              ("P1", "deployment paragraph: small-hospital equity line + clinical-faithfulness dialogue (new Cao ref, 33->34)"),
              ("P2", "timeline-stability dialogue sentence (folded into MOD-04 paragraph)"),
              ("P0", "uppercase panel letters: figures (done, v1/v2 staged) + text + legends + TABLES/SM6"),
          ], 1)}}
json.dump(status, open(SF, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("[ledger] created", SF)

# ---------------- MOD-01 cover letter ----------------
p = "submission_npjDM/cover_letter_npjDM.md"
t = open(p, encoding="utf-8").read()
old = ("**On comparators.** The deep-learning AKI external-validation study recently published in npj Digital Medicine "
       "(Lee et al., 2026) reports external AUROCs of 0.956–0.963 under simulated continuous monitoring, with isotonic "
       "recalibration fitted per external site. Our lower external AUROC (0.709) answers a deliberately harder question—incident "
       "AKI under the complete KDIGO definition (creatinine ∪ urine output) in an acute-brain-injury-specific "
       "population—transported zero-touch across 171 hospitals of a different electronic health record system rather than two "
       "external centres, with a recalibration layer that ships frozen from development data alone so that a deploying site needs "
       "no labelled outcomes.")
new = ("**On comparators.** The deep-learning AKI external-validation study recently published in npj Digital Medicine "
       "(Lee et al., 2026) reports external AUROCs of 0.956–0.963 for its 0-hour detection models under simulated continuous "
       "monitoring, with isotonic recalibration fitted per external site; at the 48-hour prediction horizon its external AUROC "
       "trajectory ranges down to 0.749. Our external AUROC of 0.709 answers a different question—incident AKI in the following "
       "48 hours, prevalent cases excluded, in an acute-brain-injury-specific population under the same complete KDIGO "
       "definition—transported zero-touch across 171 hospitals of a different electronic health record system rather than two "
       "external centres, with a recalibration layer that ships frozen from development data alone so that a deploying site needs "
       "no labelled outcomes.")
assert t.count(old) == 1, "MOD-01 anchor"
open(p, "w", encoding="utf-8").write(t.replace(old, new))
print("[MOD-01] cover letter corrected")

# ---------------- manuscript: shift 22..33 -> 23..34, then text mods ----------------
p = "submission_npjDM/MANUSCRIPT_npjDM.md"
t = open(p, encoding="utf-8").read()
body, rest = t.split("## References", 1)
shift = {n: n + 1 for n in range(22, 34)}

def rm(m):
    return "[" + ",".join(str(shift.get(int(x.strip()), int(x.strip()))) for x in m.group(1).split(",")) + "]"

body = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", rm, body)

# MOD-02 opening + First re-scope
o1 = ("Our AUROCs are nonetheless lower than some published values in this space—including the deep-learning AKI system recently "
      "validated under simulated continuous monitoring conditions in this journal (external AUROC 0.956–0.963) [18] and a "
      "multitask perioperative model that includes AKI among its outcomes (0.789–0.863 externally) [19]—and we think the reasons "
      "are informative rather than discouraging.")
n1 = ("Our AUROCs are nonetheless lower than some published values in this space—including the deep-learning AKI system recently "
      "validated under simulated continuous monitoring conditions in this journal, whose headline external AUROCs of 0.956–0.963 "
      "belong to its 0-hour detection models—nowcasts of AKI as it unfolds—while its 48-hour-horizon external trajectory ranges "
      "down to 0.749 [18], and a multitask perioperative model that includes AKI among its outcomes (0.789–0.863 externally) "
      "[19]—and we think the reasons are informative rather than discouraging.")
assert body.count(o1) == 1, "MOD-02 opening anchor"
body = body.replace(o1, n1)

o2 = "First, the complete KDIGO endpoint is harder: adding the urine-output arm identifies 358 events"
n2 = "First, against the creatinine-only tools cited above, the complete KDIGO endpoint is harder: adding the urine-output arm identifies 358 events"
assert body.count(o2) == 1
body = body.replace(o2, n2)
print("[MOD-02] lower-AUROC paragraph rewritten")

# MOD-03 SHAP mechanism
o3 = "supporting the inclusion of physiological-variability signals alongside level features."
n3 = (o3 + " The pattern—variability, not just level, carrying prognostic weight—fits the broader observation that loss of "
      "physiological complexity precedes clinically manifest deterioration: in a sepsis trajectory model built on the same public "
      "databases, reduced heart-rate variability independently predicted mortality [10], and systolic-blood-pressure variability "
      "appears to play the analogous role for the renal endpoint this system watches.")
assert body.count(o3) == 1
body = body.replace(o3, n3)
print("[MOD-03] SHAP mechanism sentences added")

# MOD-04/05 deployment paragraph additions
o4 = "are exactly the quantities a hospital governance committee must weigh when approving or declining an alert. At matched sensitivity"
n4 = ("are exactly the quantities a hospital governance committee must weigh when approving or declining an alert. The system's "
      "flat performance across the monitoring timeline (Mann-Kendall P = 0.24) means its risk does not need to mature toward "
      "onset—unlike detection-style models, whose confidence is designed to rise as the event approaches [18]—so a first-day "
      "alert is as trustworthy as a fifth-day one. Nor is deployment a luxury of large centres: hospitals with too few renal "
      "events to train a local model are precisely those that benefit most from a shared one [22], and the zero-touch form of "
      "this system—frozen model, frozen recalibration layer, no local labels required—is the form in which prediction reaches "
      "them. At matched sensitivity")
assert body.count(o4) == 1
body = body.replace(o4, n4)
print("[MOD-04/05] deployment paragraph extended")

# MOD-06 text side
body, n_rng = re.subn(r"Fig\. (\d)a-b\b", r"Fig. \1A,B", body)
body, n_one = re.subn(r"Fig\. (\d)([a-c])\b", lambda m: "Fig. " + m.group(1) + m.group(2).upper(), body)
print("[MOD-06] in-text panels uppercased:", n_rng, "ranges +", n_one, "singles")

# legends (post-References part)
rest, l1 = re.subn(r"\(a, b\)", "(A, B)", rest)
rest, l2 = re.subn(r"\(a\)", "(A)", rest)
rest, l3 = re.subn(r"\(b\)", "(B)", rest)
rest, l4 = re.subn(r"\(c\)", "(C)", rest)
print("[MOD-06] legends uppercased:", l1, l2, l3, l4)

# rebuild ref list with Cao at 22
entries, cur = {}, None
for line in rest.split("\n")[1:]:
    if line.startswith("## "):
        break
    mm = re.match(r"^(\d+)\.\s", line)
    if mm:
        cur = int(mm.group(1)); entries[cur] = [line]
    elif cur is not None and line.strip():
        entries[cur].append(line)
assert sorted(entries) == list(range(1, 34)), f"parsed {len(entries)} refs"
mm2 = re.search(r"\n## ", rest[1:])
tail = rest[1:][mm2.start():].lstrip("\n") if mm2 else ""
CAO = ("22. Cao J, et al. Exploring the limits of localization: federated model stacking improves hospital-level prediction in a "
       "national research network. npj Digit Med. 2026;9(1):492. PMID: 42032114.")
out = []
for n in range(1, 22):
    blk = list(entries[n]); blk[0] = re.sub(r"^\d+\.", str(n) + ".", blk[0], count=1); out += blk + [""]
out += [CAO, ""]
for n in range(23, 35):
    blk = list(entries[n - 1]); blk[0] = re.sub(r"^\d+\.", str(n) + ".", blk[0], count=1); out += blk + [""]
new_refs = "## References\n\n" + "\n".join(out).rstrip() + "\n\n"
open(p, "w", encoding="utf-8").write(body.rstrip() + "\n\n" + new_refs + tail)
print("[MOD-04] Cao inserted as ref 22; list now 34")

# MOD-06 side files
for f, subs in [("submission_npjDM/TABLES.md", [("(see Fig. 5c)", "(see Fig. 5C)")]),
                ("TABLES.md", [("(see Fig. 5c)", "(see Fig. 5C)")]),
                ("submission_npjDM/supplementary_materials/SM6_TRIPOD_AI_checklist.md",
                 [(r"\bFig\. 1a\b", "Fig. 1A"), (r"\bFig\. 5c\b", "Fig. 5C")])]:
    tt = open(f, encoding="utf-8").read(); c = 0
    for a, b in subs:
        tt, k = re.subn(a, b, tt); c += k
    open(f, "w", encoding="utf-8").write(tt)
    print("[MOD-06]", f, ":", c)

print("ALL MOD EXECUTED")
