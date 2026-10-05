# -*- coding: utf-8 -*-
"""Round-9.2 patch (on Biomni's round9_revision_91_final):
#23 resolve the 5 stale 'not recalibrated' sentences; #24 SM4 items cover Tier-2 layer;
#30 strip internal identifiers from journal-facing text; #32/#33 threshold-table n/N +
neutral interpretation. md -> ledger -> issues -> docx surgery -> SI PDF -> manifest.
"""
import copy, json, csv
from pathlib import Path
import docx

FIN = Path("E:/TBI subtype/09_tbi_aki/_biomni_r9_1/round9_1_review/round9_revision_91_final")
RES = Path("E:/TBI subtype/09_tbi_aki/results")

def patch_md(path, pairs):
    t = path.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in t, f"ANCHOR MISSING {path.name}: {old[:70]!r}"
        assert t.count(old) == 1, f"NOT UNIQUE {path.name}: {old[:70]!r}"
        t = t.replace(old, new)
    path.write_text(t, encoding="utf-8")
    print(f"md patched: {path.name} ({len(pairs)})")

# ================= md edits =================
MANU = [
 # #23a Results
 ("; the Tier-2 output was not recalibrated.",
  "; the Tier-2 output received an identically fitted recalibration layer (Supplementary Material 5)."),
 # #23b Discussion action ladder
 ("The second step is a proposal for prospective testing: the Tier-2 output is not recalibrated, its operating threshold has not been set, and its evidence comes from one external database.",
  "The second step is a proposal for prospective testing: the Tier-2 output now carries a development-fitted recalibration layer with development-side threshold views (Supplementary Material 5), but its action threshold is not prospectively validated and its evidence comes from one external database."),
 # #23c Methods
 ("Tier-2 probabilities were not recalibrated.",
  "Tier-2 probabilities received an identically fitted layer (a = −0.301766, b = 0.393865; Supplementary Material 5)."),
 # #23d Table 3 footnote
 ("The Tier-2 output was not recalibrated.*",
  "The Tier-2 output received an identically fitted layer (Supplementary Material 5).*"),
 # #30 Table 2b footnote
 ("Internal paired differences use the same resampling scheme (analysis file f9_fillins.json, patch 9.1).",
  "Internal paired differences use the same resampling scheme as the external comparisons."),
 # #33 Discussion numbers refinement
 ("at which a 0.05 severe-tier threshold captured 43.4% of external severe events a median 6 h ahead while alerting 2.0% of stays (Supplementary Material 5)",
  "at which the lowest examined severe-tier threshold (0.05) captured 36 of 83 external severe-event stays (43.4%) a median 6 h—one checkpoint—ahead while alerting 179 of 9,071 stays (2.0%) (Supplementary Material 5)"),
]
patch_md(FIN / "manuscript/MANUSCRIPT_npjDM_round9.md", MANU)

TAB = [
 ("Internal paired differences use the same resampling scheme (analysis file f9_fillins.json, patch 9.1).",
  "Internal paired differences use the same resampling scheme as the external comparisons."),
 ("The Tier-2 output was not recalibrated.*",
  "The Tier-2 output received an identically fitted layer (Supplementary Material 5).*"),
]
patch_md(FIN / "manuscript/TABLES_npjDM_round9.md", TAB)

# SM4 (#24)
patch_md(FIN / "supplementary/SM4_TRIPOD_AI_checklist_round9.md", [
 ("| | 16b | MS Results, Calibration (slope/ECE/Brier); Table 3; Fig. 3a, b |",
  "| | 16b | MS Results, Calibration (slope/ECE/Brier for Tier 1); Table 3; Fig. 3a, b; Tier-2 layer in Supplementary Material 5 |"),
 ("| **Model updating** | 17 | MS Methods, Calibration (recalibration layer fitted on patient-grouped out-of-fold development predictions; shipped frozen) |",
  "| **Model updating** | 17 | MS Methods, Calibration (recalibration layers for both tiers fitted on patient-grouped out-of-fold development predictions; shipped frozen; Tier-2 layer in Supplementary Material 5) |"),
])

# SM5 + merged SI (#30, #32, #33)
SM5 = [
 ("*Raw internal paired differences involving ≥Stage 3 were added in patch 9.1 (analysis file f9_fillins.json): Tier 2 − Tier 1",
  "*Raw internal paired differences involving ≥Stage 3, estimated with the same resampling scheme as the external comparisons: Tier 2 − Tier 1"),
 ("(logistic recalibration on out-of-fold predictions from five-fold GroupKFold by patient on the training split; analysis file f9_tier2_calibration.json, patch 9.1): intercept",
  "(logistic recalibration on out-of-fold predictions from five-fold GroupKFold by patient on the training split, identical to the Tier-1 protocol): intercept"),
 ("| 0.05 | 0.434 | 6.0 (6.0–18.0) | 4.97 | 0.46 | 2.0% |",
  "| 0.05 | 0.434 (36/83) | 6.0 (6.0–18.0) | 4.97 | 0.46 | 2.0% (179/9,071) |"),
 ("| 0.10 | 0.253 | 12.0 (6.0–24.0) | 5.52 | 0.30 | 1.3% |",
  "| 0.10 | 0.253 (21/83) | 12.0 (6.0–24.0) | 5.52 | 0.30 | 1.3% (116/9,071) |"),
 ("| 0.15 | 0.193 | 12.0 (6.0–25.5) | 4.25 | 0.16 | 0.7% |",
  "| 0.15 | 0.193 (16/83) | 12.0 (6.0–25.5) | 4.25 | 0.16 | 0.7% (68/9,071) |"),
 ("| 0.20 | 0.133 | 12.0 (6.0–18.0) | 4.09 | 0.10 | 0.5% |",
  "| 0.20 | 0.133 (11/83) | 12.0 (6.0–18.0) | 4.09 | 0.10 | 0.5% (45/9,071) |"),
 ("A severe-tier threshold near 0.05 offers the best trade-off in this external view: 43.4% of severe events captured with alerts on 2.0% of stays.",
  "The 0.05 threshold was the lowest examined and captured the largest share of severe event stays (36/83, 43.4%) with alerts on 179 of 9,071 stays (2.0%); its median lead of 6 h equals the checkpoint spacing, so most captures preceded the anchor by a single six-hour checkpoint."),
]
patch_md(FIN / "supplementary/SM5_sensitivity_round9.md", SM5)
patch_md(FIN / "supplementary/Supplementary_Information_round9.md", SM5)

# ================= ledger =================
LED = FIN / "audit/number_ledger_round9.csv"
rows = [
 ("tier2_thr005_ncaptured", "36", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.05.n_captured", "False", "True"),
 ("tier2_thr005_nalerts", "179", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.05.n_alert_stays", "False", "True"),
 ("tier2_thr010_ncaptured", "21", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.1.n_captured", "False", "True"),
 ("tier2_thr010_nalerts", "116", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.1.n_alert_stays", "False", "True"),
 ("tier2_thr015_ncaptured", "16", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.15.n_captured", "False", "True"),
 ("tier2_thr015_nalerts", "68", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.15.n_alert_stays", "False", "True"),
 ("tier2_thr020_ncaptured", "11", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.2.n_captured", "False", "True"),
 ("tier2_thr020_nalerts", "45", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.2.n_alert_stays", "False", "True"),
]
with open(LED, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for r in rows:
        w.writerow(r)
print(f"ledger +{len(rows)} rows")

# ================= issues append =================
ISS = FIN / "audit/issues_and_residual_risks_round9.md"
ISS.write_text(ISS.read_text(encoding="utf-8") + """

## 五、Patch 9.2（PI 侧，2026-10-01）
- **#23 已解决**：5 句 "not recalibrated / threshold not set" 全部改写为与新 Tier-2 校准内容一致的表述（Results 校准段、Discussion 行动阶梯段、Methods 校准段、Table 3 脚注 ×2）。
- **#24 已解决**：SM4 item 16b/17 增补 Tier-2 层指向；Methods 校准段已含 Tier-2 层参数。
- **#30 已解决**：正文与 SM5 的内部标识（patch 9.1、f9_fillins.json、f9_tier2_calibration.json）全部清除，改为中性表述或 SM5 指向。
- **#32/#33 已解决**：阈值表全部加 n/N（36/83、179/9,071 等 8 格）；"best trade-off" 改为中性陈述（0.05 为所测最低阈值）；6 h 中位提前量注明=检查点间距；43.4% 分母明确为 83 个重度事件 stays；Discussion 同步改为 36 of 83 (43.4%) / 179 of 9,071。计数已存 f9_tier2_calibration.json（n_captured/n_alert_stays 字段，patch 9.2 补算）。
- #27 维持：词表命中 8 个整数确在 data/ 无源（probe 字段为标题字符串），SM3 保留 carry-over 标注。
- 开放 A 级仅剩 #1（重度档需外部重复——科学层面，已披露）。
""", encoding="utf-8")
print("issues appended")

# ================= docx surgery =================
def replace_in_runs(para, old, new):
    full = "".join(r.text for r in para.runs)
    if old not in full:
        return False
    s, e = full.index(old), full.index(old) + len(old)
    pos, first = 0, True
    for r in para.runs:
        rs, re_ = pos, pos + len(r.text)
        if re_ <= s or rs >= e:
            pos = re_
            continue
        ls, le = max(s - rs, 0), min(e - rs, len(r.text))
        r.text = (r.text[:ls] + new + r.text[le:]) if first else (r.text[:ls] + r.text[le:])
        first = False
        pos = re_
    return True

def replace_in_doc(doc, old, new, expect=1):
    n = sum(replace_in_runs(p, old, new) for p in doc.paragraphs)
    assert n == expect, f"docx replace expected {expect} got {n}: {old[:60]!r}"

def replace_cell(doc, old, new, expect=1):
    n = 0
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                if c.text.strip() == old:
                    c.text = new
                    n += 1
    assert n == expect, f"cell replace expected {expect} got {n}: {old!r}"

# MANUSCRIPT docx
f = FIN / "manuscript/MANUSCRIPT_npjDM_round9.docx"
doc = docx.Document(f)
for old, new in MANU:
    replace_in_doc(doc, old, new)
doc.save(f)
print("MANUSCRIPT docx patched")

# TABLES docx
f = FIN / "manuscript/TABLES_npjDM_round9.docx"
doc = docx.Document(f)
for old, new in TAB:
    replace_in_doc(doc, old, new)
doc.save(f)
print("TABLES docx patched")

# SI docx (SM5 text + table cells + SM4 rows)
f = FIN / "supplementary/Supplementary_Information_round9.docx"
doc = docx.Document(f)
replace_in_doc(doc, SM5[0][0], SM5[0][1])
replace_in_doc(doc, SM5[1][0], SM5[1][1])
replace_in_doc(doc, SM5[6][0], SM5[6][1])
CELLS = [("0.434", "0.434 (36/83)"), ("0.253", "0.253 (21/83)"), ("0.193", "0.193 (16/83)"), ("0.133", "0.133 (11/83)"),
         ("2.0%", "2.0% (179/9,071)"), ("1.3%", "1.3% (116/9,071)"), ("0.7%", "0.7% (68/9,071)"), ("0.5%", "0.5% (45/9,071)")]
for old, new in CELLS:
    replace_cell(doc, old, new)
replace_in_doc(doc, "| | 16b | MS Results, Calibration (slope/ECE/Brier); Table 3; Fig. 3a, b |",
               "| | 16b | MS Results, Calibration (slope/ECE/Brier for Tier 1); Table 3; Fig. 3a, b; Tier-2 layer in Supplementary Material 5 |", expect=0) if False else None
# SM4 rows live in a table in the SI docx -> cell-level replace
def replace_cell_contains(doc, frag, new):
    n = 0
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                if frag in c.text and c.text.strip() != new:
                    c.text = new if c.text.strip().startswith(("MS Results, Calibration (slope/ECE/Brier);", "MS Methods, Calibration (recalibration layer")) else c.text
                    if c.text.strip() == new: n += 1
    return n
for t in doc.tables:
    for row in t.rows:
        for c in row.cells:
            txt = c.text.strip()
            if txt == "MS Results, Calibration (slope/ECE/Brier); Table 3; Fig. 3a, b":
                c.text = "MS Results, Calibration (slope/ECE/Brier for Tier 1); Table 3; Fig. 3a, b; Tier-2 layer in Supplementary Material 5"
            elif txt == "MS Methods, Calibration (recalibration layer fitted on patient-grouped out-of-fold development predictions; shipped frozen)":
                c.text = "MS Methods, Calibration (recalibration layers for both tiers fitted on patient-grouped out-of-fold development predictions; shipped frozen; Tier-2 layer in Supplementary Material 5)"
doc.save(f)
print("SI docx patched")
print("\nPATCH 9.2 COMPLETE")
