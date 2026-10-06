# -*- coding: utf-8 -*-
"""f30: purge 'Tier 1/Tier 2/tier' paper-wide -> clinical names (PI order 2026-10-06).
Targets: submission MANUSCRIPT.docx (paragraphs + table cells), package docx copy.
Also: Fig3 legend rewrite (9-panel calibration/DCA/deployment), SuppFig renumber
(2=DCA dissolves into Fig3d-f; within-hospital 3->2; beeswarm 4->3).
Run-local replacement preserves bold panel-letter runs in legends; cross-run stragglers
are caught by the final zero-residue assertion.
"""
import re
import sys
from docx import Document

P = r"E:/TBI subtype/09_tbi_aki/_submission_npjDM/MANUSCRIPT.docx"
PKG = r"E:/TBI subtype/09_tbi_aki/_biomni_r112/final/manuscript/MANUSCRIPT_npjDM_round11.docx"

# ---- ordered replacement rules (longest / most specific first) ----
RULES = [
    # SuppFig renumber via placeholders (applied first, resolved last)
    ("Supplementary Fig. 4", "Supplementary Fig. @S3@"),
    ("Supplementary Fig. 3", "Supplementary Fig. @S2@"),
    ("Supplementary Fig. 2", "@FIG3DF@"),
    # definitions / targeted
    ("Tier 1 (primary outcome) was any AKI (\u2265Stage 1, cumulative); Tier 2 (key secondary outcome) was severe AKI (\u2265Stage 3), predicted by a separate model.",
     "The primary outcome was any AKI (\u2265Stage 1, cumulative); the key secondary outcome was severe AKI (\u2265Stage 3), predicted by a separate model."),
    ("Tier 1 (any AKI, \u2265Stage 1; Stages 1 and 2 merged) is the primary outcome and Tier 2 (severe AKI, \u2265Stage 3) is a separate model without a recalibration layer.",
     "Any AKI (\u2265Stage 1; Stages 1 and 2 merged) is the primary outcome and severe AKI (\u2265Stage 3) a separate model without a recalibration layer."),
    ("the outcome was structured as two tiers: Tier 1 (any AKI, primary outcome) and Tier 2 (severe AKI, \u2265Stage 3, key secondary outcome)",
     "the outcome was structured as two endpoints: any AKI (primary outcome) and severe AKI (\u2265Stage 3, key secondary outcome)"),
    ("Any-AKI discrimination (Tier 1) was", "Any-AKI discrimination was"),
    ("Severe AKI (Tier 2, \u2265Stage 3) was discriminated better", "Severe AKI (\u2265Stage 3) was discriminated better"),
    ("Positive checkpoints for any AKI (Tier 1) were", "Positive checkpoints for any AKI were"),
    ("severe-AKI (Tier 2) positives", "severe-AKI positives"),
    ("T1, Tier 1; T2, Tier 2; GCS, Glasgow Coma Scale.", "GCS, Glasgow Coma Scale."),
    ("(Tier 1 819; Tier 2 68)", "(any AKI 819; severe AKI 68)"),
    ("Discrimination for the two tiers in the development", "Discrimination for the two endpoints in the development"),
    ("for Tier 1 (any AKI, \u2265Stage 1) and Tier 2 (severe AKI, \u2265Stage 3) in the JinhuaNSICU",
     "for any AKI (\u2265Stage 1) and severe AKI (\u2265Stage 3) in the JinhuaNSICU"),
    ("between Tier 2 and Tier 1 (paired", "between severe AKI and any AKI (paired"),
    ("Outcome and tiers", "Outcomes"),
    ("the two-tier output maps onto an action ladder: intensified renal surveillance for Tier 1 and nephrology review and renal-replacement readiness for Tier 2",
     "the two-endpoint output maps onto an action ladder: intensified renal surveillance for any AKI and nephrology review and renal-replacement readiness for severe AKI"),
    ("an ordered risk tier and not as an absolute risk", "an ordered risk score and not an absolute risk"),
    ("whether tiered alerts change management", "whether risk-stratified alerts change management"),
    ("the Tier-1 output passes through a logistic recalibration layer", "the any-AKI output passes through a logistic recalibration layer"),
    # compound nouns
    ("the Tier-1 model", "the any-AKI model"), ("the Tier-2 model", "the severe-AKI model"),
    ("Tier-2 model", "severe-AKI model"), ("Tier-1 model", "any-AKI model"),
    ("Tier-2 AUROC", "severe-AKI AUROC"), ("Tier-1 AUROCs", "any-AKI AUROCs"), ("Tier-1 AUROC", "any-AKI AUROC"),
    ("Tier-1 output", "any-AKI output"), ("Tier-2 output", "severe-AKI output"),
    ("Tier-2 scores", "severe-AKI scores"), ("Tier-2 slope", "severe-AKI slope"),
    ("Tier-2 positives", "severe-AKI positives"), ("Tier-2 advantage", "severe-AKI advantage"),
    ("Tier-2 event stays", "severe-AKI event stays"), ("Tier-2 endpoints", "severe-AKI endpoints"),
    ("Tier-2 deployment metrics", "severe-AKI deployment metrics"),
    ("Tier-2 positive checkpoints", "severe-AKI positive checkpoints"),
    ("Tier-2 recalibration view", "severe-AKI recalibration view"), ("Tier-2 view", "severe-AKI view"),
    ("Tier-2 predictions", "severe-AKI predictions"),
    ("the Tier-2 and \u2265Stage 2 models", "the severe-AKI and \u2265Stage 2 models"),
    ("Tier-1 discrimination", "any-AKI discrimination"), ("Tier-1 labels", "any-AKI labels"),
    ("Tier-1 thresholds", "any-AKI thresholds"), ("Tier-1 predictions", "any-AKI predictions"),
    ("Tier-1 probabilities", "any-AKI probabilities"), ("Tier-1-negative", "any-AKI-negative"),
    ("Tier-1 and \u2265Stage 2 models", "any-AKI and \u2265Stage 2 models"),
    ("Tier-1 layer", "any-AKI layer"), ("Tier-1 threshold sweep", "any-AKI threshold sweep"),
    ("Tier-1 recalibration layer", "any-AKI recalibration layer"),
    ("the internal Tier-2 estimate", "the internal severe-AKI estimate"),
    ("within-hospital Tier-1", "within-hospital any-AKI"),
    ("severe-tier advantage", "severe-AKI advantage"), ("the severe-tier result", "the severe-AKI result"),
    ("the severe tier was", "the severe endpoint was"), ("the internal severe tier", "the internal severe endpoint"),
    ("deployment metrics for the severe tier", "deployment metrics for the severe endpoint"),
    ("discriminated the severe tier better", "discriminated severe AKI better"),
    # structural words
    ("two-tier", "two-endpoint"), ("two outcome tiers", "two outcome endpoints"), ("two tiers", "two endpoints"),
    ("outcome tiers", "outcome endpoints"),
    # prepositions
    ("for Tier 1", "for any AKI"), ("for Tier 2", "for severe AKI"),
    ("from Tier 1", "from any AKI"), ("against Tier 1", "against any AKI"), ("over Tier 1", "over any AKI"),
    # table difference rows (en-dash forms come from both hyphen and minus variants)
    ("Tier 2 \u2212 Tier 1", "severe AKI \u2212 any AKI"), ("Tier 2 \u2013 Tier 1", "severe AKI \u2013 any AKI"),
    ("Tier 2 - Tier 1", "severe AKI - any AKI"),
    ("Tier 2 \u2212 \u2265Stage 2", "severe AKI \u2212 \u2265Stage 2"), ("Tier 2 \u2013 \u2265Stage 2", "severe AKI \u2013 \u2265Stage 2"),
    ("\u2265Stage 2 \u2212 Tier 1", "\u2265Stage 2 \u2212 any AKI"), ("\u2265Stage 2 \u2013 Tier 1", "\u2265Stage 2 \u2013 any AKI"),
    # table endpoint rows
    ("Tier 1: any AKI (\u2265Stage 1), primary", "Any AKI (\u2265Stage 1), primary"),
    ("Tier 2: severe AKI (\u2265Stage 3), key secondary", "Severe AKI (\u2265Stage 3), key secondary"),
    # parentheticals, then catch-alls
    (" (Tier 1)", ""), ("(Tier 1)", ""), (" (Tier 2)", ""), ("(Tier 2)", ""),
    ("Tier-1", "any-AKI"), ("Tier-2", "severe-AKI"),
    ("Tier 1", "any AKI"), ("Tier 2", "severe AKI"),
]

CLEANUP = [("  ", " "), (" ,", ","), (" ;", ";"), (" .", ".")]

def apply_rules(text):
    for old, new in RULES:
        text = text.replace(old, new)
    # resolve placeholders
    text = text.replace("@S3@", "3").replace("@S2@", "2").replace("@FIG3DF@", "Fig. 3d\u2013f")
    for _ in range(2):
        for old, new in CLEANUP:
            text = text.replace(old, new)
    return text

def iter_all_paragraphs(doc):
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    yield p

def main(path):
    d = Document(path)
    changed = 0
    for p in iter_all_paragraphs(d):
        if not re.search(r"(?i)tier|@S\d@|@FIG3DF@", p.text):
            continue
        if len(p.runs) <= 1:
            new = apply_rules(p.text)
            if new != p.text:
                if p.runs:
                    p.runs[0].text = new
                    for r in p.runs[1:]:
                        r.text = ""
                else:
                    p.add_run(new)
                changed += 1
        else:
            # run-local replace first (preserves bold panel letters)
            before = p.text
            for r in p.runs:
                if re.search(r"(?i)tier|@S\d@|@FIG3DF@", r.text):
                    r.text = apply_rules(r.text)
            if p.text != before:
                changed += 1
                continue
            # no run contained a hit but paragraph does -> cross-run; collapse
            new = apply_rules(before)
            if new != before:
                p.runs[0].text = new
                for r in p.runs[1:]:
                    r.text = ""
                changed += 1
    d.save(path)
    # verification pass
    d2 = Document(path)
    residue = [(i, p.text[:120]) for i, p in enumerate(iter_all_paragraphs(d2)) if re.search(r"(?i)tier", p.text)]
    print(f"{path.split('/')[-1]}: {changed} paragraphs touched; residue: {len(residue)}")
    for i, s in residue[:8]:
        print(f"  RESIDUE [{i}] {s}")
    return len(residue)

if __name__ == "__main__":
    r1 = main(P)
    r2 = main(PKG)
    sys.exit(0 if (r1 == 0 and r2 == 0) else 1)
