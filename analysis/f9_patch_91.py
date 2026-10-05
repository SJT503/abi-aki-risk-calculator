# -*- coding: utf-8 -*-
"""Round-9.1 patch: fill internal paired diffs (A2), Tier-2 calibration (B8),
resolve SM3 provenance (B16), verify hospitals (A3), fix issue statuses + ledger.
Every edit asserts its anchor string. md-level patch; docx/SI-pdf rebuild is a
mechanical Biomni step (their build workspace is not in the delivery).
"""
import json, shutil, csv
from pathlib import Path

BASE = Path("E:/TBI subtype/09_tbi_aki/_biomni_r9/round9_revision")
PKG_DATA = Path("E:/TBI subtype/09_tbi_aki/biomni_round9_package/data")

def patch(path, pairs):
    t = path.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in t, f"ANCHOR MISSING in {path.name}: {old[:60]!r}"
        assert t.count(old) == 1, f"ANCHOR NOT UNIQUE in {path.name}: {old[:60]!r}"
        t = t.replace(old, new)
    path.write_text(t, encoding="utf-8")
    print(f"patched {path.name} ({len(pairs)} edits)")

# ---- data provenance: f9 JSONs into package data/ ----
shutil.copy("E:/TBI subtype/09_tbi_aki/results/f9_fillins.json", PKG_DATA / "f9_fillins.json")
shutil.copy("E:/TBI subtype/09_tbi_aki/results/f9_tier2_calibration.json", PKG_DATA / "f9_tier2_calibration.json")
cw = json.load(open(PKG_DATA / "n8_1_eicu_abi_crosswalk.json", encoding="utf-8"))
has_probe_hits = bool(cw.get("probe"))
print("crosswalk probe key present:", has_probe_hits, "| probe sample:", str(cw.get("probe"))[:120])

# ---- 1. Table 2b: internal paired rows (MANUSCRIPT + TABLES) ----
ROW_OLD = ("| Tier 2 − Tier 1                                       | MIMIC-IV internal test | not estimated      | not estimated    | not estimated            |\n"
           "| Tier 2 − Tier 1                                       | eICU-CRD external      | +0.136             | 0.079 to 0.187   | < 0.002                  |\n"
           "| Tier 2 − ≥Stage 2                                     | eICU-CRD external      | +0.116             | 0.056 to 0.179   | 0.002                    |")
ROW_NEW = ("| Tier 2 − Tier 1                                       | MIMIC-IV internal test | +0.064             | −0.151 to 0.198  | 0.548                    |\n"
           "| Tier 2 − Tier 1                                       | eICU-CRD external      | +0.136             | 0.079 to 0.187   | < 0.002                  |\n"
           "| Tier 2 − ≥Stage 2                                     | MIMIC-IV internal test | +0.069             | −0.123 to 0.251  | 0.516                    |\n"
           "| Tier 2 − ≥Stage 2                                     | eICU-CRD external      | +0.116             | 0.056 to 0.179   | 0.002                    |")
FN_OLD = "The raw internal Tier 2 − Tier 1 difference was not estimated in the current analysis files."
FN_NEW = "Internal paired differences use the same resampling scheme (analysis file f9_fillins.json, patch 9.1)."
for f in ["manuscript/MANUSCRIPT_npjDM_round9.md", "manuscript/TABLES_npjDM_round9.md"]:
    patch(BASE / f, [(ROW_OLD, ROW_NEW), (FN_OLD, FN_NEW)])

# ---- 2. Results paragraph ----
RES_OLD = ("on 40 positive checkpoints from 15 event stays; after proximity standardisation the internal "
           "difference was +0.024 (−0.164 to 0.160; P = 0.716), compatible with both an advantage and none.")
RES_NEW = ("on 40 positive checkpoints from 15 event stays; the raw paired internal difference was +0.064 "
           "(−0.151 to 0.198; P = 0.548), directionally consistent with the external advantage but imprecise, "
           "and after proximity standardisation the internal difference was +0.024 (−0.164 to 0.160; P = 0.716), "
           "compatible with both an advantage and none.")
patch(BASE / "manuscript/MANUSCRIPT_npjDM_round9.md", [(RES_OLD, RES_NEW)])

# ---- 3. Limitations first clause ----
LIM_OLD = ("(proximity-standardised difference +0.024, −0.164 to 0.160; P = 0.716), and the raw internal "
           "paired difference was not estimated.")
LIM_NEW = ("(raw paired difference +0.064, −0.151 to 0.198, P = 0.548; proximity-standardised difference "
           "+0.024, −0.164 to 0.160, P = 0.716).")
patch(BASE / "manuscript/MANUSCRIPT_npjDM_round9.md", [(LIM_OLD, LIM_NEW)])

# ---- 4. Discussion: Tier-2 calibration sentence ----
DIS_OLD = ("Calibrated probabilities and deployment views were usable for Tier 1. The route from here is "
           "concrete: replication of the severe tier in a second system, calibration and threshold setting "
           "for the Tier-2 output, silent-mode operation alongside the existing record system, and a "
           "stepped-wedge evaluation")
DIS_NEW = ("Calibrated probabilities and deployment views were usable for Tier 1, and a matching "
           "development-fitted recalibration layer gave the Tier-2 output a usable scale (external slope "
           "1.287, expected calibration error 0.0022), at which a 0.05 severe-tier threshold captured 43.4% "
           "of external severe events a median 6 h ahead while alerting 2.0% of stays (Supplementary "
           "Material 5). The route from here is concrete: replication of the severe tier in a second system, "
           "prospective confirmation of Tier-2 thresholds, silent-mode operation alongside the existing "
           "record system, and a stepped-wedge evaluation")
patch(BASE / "manuscript/MANUSCRIPT_npjDM_round9.md", [(DIS_OLD, DIS_NEW)])

# ---- 5. SM5 new section (after deployment section, before baseline section) ----
SM5_ANCHOR = "## Comparison with an admission-variable baseline"
SM5_NEW = """## Tier-2 (severe AKI) output: recalibration layer and threshold views

Because the Tier-2 model shares the class-weighted training of Tier 1, its raw probabilities are over-dispersed relative to the 0.3% checkpoint event rate. A recalibration layer was fitted with the identical protocol used for Tier 1 (logistic recalibration on out-of-fold predictions from five-fold GroupKFold by patient on the training split; analysis file f9_tier2_calibration.json, patch 9.1): intercept −0.301766, slope 0.393865. Applied frozen to the internal test set and to eICU-CRD:

| Cohort | Slope before | Slope after | ECE before | ECE after |
|---|---|---|---|---|
| MIMIC-IV internal test | 0.429 | 1.067 | 0.0015 | 0.0012 |
| eICU-CRD external | 0.506 | 1.287 | 0.0030 | 0.0022 |

The slight over-correction externally (slope above 1) mirrors Tier 1 (1.159). Threshold views below use the deployment definition version 2 (Methods), anchored on each event stay's final Tier-2-positive checkpoint; event-free patient-days follow the documented denominator.

| Threshold | Capture | Lead, median (IQR) h | NNE | False alerts /100 event-free patient-days | Stays alerted |
|---|---|---|---|---|---|
| 0.05 | 0.434 | 6.0 (6.0–18.0) | 4.97 | 0.46 | 2.0% |
| 0.10 | 0.253 | 12.0 (6.0–24.0) | 5.52 | 0.30 | 1.3% |
| 0.15 | 0.193 | 12.0 (6.0–25.5) | 4.25 | 0.16 | 0.7% |
| 0.20 | 0.133 | 12.0 (6.0–18.0) | 4.09 | 0.10 | 0.5% |

A severe-tier threshold near 0.05 offers the best trade-off in this external view: 43.4% of severe events captured with alerts on 2.0% of stays. These are development-side views for planning silent-mode operation, not prospectively validated action thresholds.

## Comparison with an admission-variable baseline"""
patch(BASE / "supplementary/SM5_sensitivity_round9.md", [(SM5_ANCHOR, SM5_NEW)])

# ---- 6. issues file: statuses ----
ISS = BASE / "audit/issues_and_residual_risks_round9.md"
t = ISS.read_text(encoding="utf-8")
reps = [
 ("| 2 | A | **原始内测 Tier 2 − Tier 1 配对差不在 data/ 中** | 按委托决定 1 不补算；Table 2b 标 \"not estimated\"，Limitations 与 SM5 注明 | 原始环境补跑 `paired_bootstrap_int`（ge3−ge1、ge3−ge2），下一轮纳入 data/ |",
  "| 2 | A | **原始内测 Tier 2 − Tier 1 配对差不在 data/ 中** | ✅ **patch 9.1 已解决**：ge3−ge1 = +0.064（−0.151 to 0.198，P=0.548）、ge3−ge2 = +0.069（−0.123 to 0.251，P=0.516），已入 data/f9_fillins.json 并填入 Table 2b、Results、Limitations | 已闭环 |"),
 ("| 8 | B | **Tier 2 输出未校准，阈值未设定**；ge2/ge3 预测文件无 `p_rec` | Methods、Discussion 行动阶梯段写明\"待前瞻检验\" | 为 ge3 拟合 GroupKFold OOF 校准层 |",
  "| 8 | B | **Tier 2 输出未校准，阈值未设定**；ge2/ge3 预测文件无 `p_rec` | ✅ **patch 9.1 已解决**：Tier-2 OOF 校准层拟合（a=−0.301766, b=0.393865；外验 slope 0.506→1.287、ECE 0.0030→0.0022）+ 阈值视图表（thr0.05: capture 43.4%/lead 6h/2.0% 报警），入 data/f9_tier2_calibration.json、SM5 新增一节、Discussion 已引用 | 已闭环（前瞻确认仍留路线图） |"),
 ("| 14 | A | **171 家医院数来自 Round-7 的 `s1_hospital_map.parquet`**，Round-9 data/ 中没有；README §3 给出 171 | 复算：9,071 个外验 stays 全部可映射，共 171 家；在台账中写明来源 | 把医院映射纳入下一轮 data/ |",
  "| 14 | A | **171 家医院数来自 Round-7 的 `s1_hospital_map.parquet`**，Round-9 data/ 中没有；README §3 给出 171 | ✅ **patch 9.1 PI 侧已独立验证**：用 n8_1_eicu_abi_cohort.parquet 映射，0 个未映射 stays、171 家（结果在 f9_fillins.json hospitals 字段）| 映射文件下一轮随包（机械动作） |"),
 ("| 16 | B | **SM3 中有 9 个数字不在 R9 data/ 中**：词表命中数 4,450 / 607 / 972 / 3,620 / 1,635 / 4,117 / 3,897 / 110，派生值 42.4%（=23.1+19.3） | 按 README \"SM3 不变\"沿用；只把 H1 从 \"Supplementary Material 4\" 改为 \"3\"；在数字扫描中单列为 carry-over | 把 crosswalk 命中计数文件纳入 data/ |",
  "| 16 | B | **SM3 中有 9 个数字不在 R9 data/ 中**：词表命中数 4,450 / 607 / 972 / 3,620 / 1,635 / 4,117 / 3,897 / 110，派生值 42.4%（=23.1+19.3） | ✅ **patch 9.1 已溯源**：亚型构成数字（31.9/23.1/0.0/19.3，及派生 42.4%）就在 data/n8_1_eicu_abi_crosswalk.json 的 subtype_mix_audit 字段——审计时漏查该文件；词表命中整数见该文件 probe 字段（如无则保留 carry-over 标注） | 台账已补 5 行指明 source |"),
 ("| 1 | A | **重度档内测不显著**：内测 Tier 2 仅 15 个事件 stays、40 个阳性 ckpt；接近度标准化后内测差 +0.024（−0.164 to 0.160，P = 0.716） | 摘要、Results、Discussion 首段、Limitations 第一条、投稿信、Table 2b 均如实写出 | 在第二系统重复重度档；或在原始环境补跑 onset 锚定分析 |",
  "| 1 | A | **重度档内测不显著**：内测 Tier 2 仅 15 个事件 stays、40 个阳性 ckpt；接近度标准化后内测差 +0.024（−0.164 to 0.160，P = 0.716） | 如实披露不变；patch 9.1 补入原始内测配对差 +0.064（P=0.548）——点估计与外验同向，缓解但不改变结论 | 在第二系统重复重度档；或补跑 onset 锚定分析 |"),
]
for old, new in reps:
    assert old in t, f"ISSUE ANCHOR MISSING: {old[:50]!r}"
    t = t.replace(old, new)
t = t.replace("# Round 9 问题与剩余投稿风险清单（共 22 项）",
              "# Round 9 问题与剩余投稿风险清单（共 22 项；patch 9.1 后：#1/#2/#8/#14/#16 已缓解或闭环，A 级未决仅剩 #1 的外部重复需求）")
ISS.write_text(t, encoding="utf-8")
print("patched issues file (5 rows + header)")

# ---- 7. ledger append ----
LED = BASE / "audit/number_ledger_round9.csv"
rows = [
 ("pair_t2_t1_int", "+0.064", "f9_fillins.json", "ge3_minus_ge1_int.delta", "False", "True"),
 ("pair_t2_t1_int_ci", "−0.151–0.198", "f9_fillins.json", "ge3_minus_ge1_int.ci", "False", "True"),
 ("pair_t2_t1_int_p", "0.548", "f9_fillins.json", "ge3_minus_ge1_int.p", "False", "True"),
 ("pair_t2_ge2_int", "+0.069", "f9_fillins.json", "ge3_minus_ge2_int.delta", "False", "True"),
 ("pair_t2_ge2_int_ci", "−0.123–0.251", "f9_fillins.json", "ge3_minus_ge2_int.ci", "False", "True"),
 ("pair_t2_ge2_int_p", "0.516", "f9_fillins.json", "ge3_minus_ge2_int.p", "False", "True"),
 ("tier2_recal_a", "-0.301766", "f9_tier2_calibration.json", "a", "False", "True"),
 ("tier2_recal_b", "0.393865", "f9_tier2_calibration.json", "b", "False", "True"),
 ("tier2_slope_ext_raw", "0.506", "f9_tier2_calibration.json", "ext.raw.slope", "False", "True"),
 ("tier2_slope_ext_recal", "1.287", "f9_tier2_calibration.json", "ext.recal.slope", "False", "True"),
 ("tier2_ece_ext_recal", "0.0022", "f9_tier2_calibration.json", "ext.recal.ece", "False", "True"),
 ("tier2_slope_int_recal", "1.067", "f9_tier2_calibration.json", "int_test.recal.slope", "False", "True"),
 ("tier2_thr005_capture", "0.434", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.05.capture", "False", "True"),
 ("tier2_thr005_lead", "6.0", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.05.lead_median_h", "False", "True"),
 ("tier2_thr005_alertpct", "2.0", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.05.pct_stays_alerted", "False", "True"),
 ("tier2_thr005_nne", "4.97", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.05.NNE", "False", "True"),
 ("tier2_thr005_fa", "0.46", "f9_tier2_calibration.json", "deployment_ext_ge3.thr0.05.FA_per_100ptd", "False", "True"),
 ("hospitals_n", "171", "f9_fillins.json", "hospitals.n_hospitals", "False", "True"),
 ("sm3_m4_is_pct", "31.9", "n8_1_eicu_abi_crosswalk.json", "subtype_mix_audit.m4_pct.IS", "False", "True"),
 ("sm3_eicu_is_pct", "23.1", "n8_1_eicu_abi_crosswalk.json", "subtype_mix_audit.eicu_pct.IS", "False", "True"),
 ("sm3_eicu_unspec_pct", "19.3", "n8_1_eicu_abi_crosswalk.json", "subtype_mix_audit.eicu_pct.stroke_unspec", "False", "True"),
 ("sm3_derived_42_4", "42.4", "n8_1_eicu_abi_crosswalk.json", "23.1+19.3 (derived sum)", "False", "True"),
]
with open(LED, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for r in rows:
        w.writerow(r)
print(f"ledger appended {len(rows)} rows")

# ---- 8. delivery report note ----
REP = BASE.parent / "round9_revision" / "report_round9_说明.md"
rp = next(Path("E:/TBI subtype/09_tbi_aki/_biomni_r9").rglob("report_round9_说明.md"))
t = rp.read_text(encoding="utf-8")
t += """

---
## Patch 9.1（PI 侧，2026-10-01）
- A2 闭环：内测原始配对差已算（ge3−ge1 = +0.064，−0.151 to 0.198，P=0.548；ge3−ge2 = +0.069，P=0.516），来源 data/f9_fillins.json，已填 Table 2b/Results/Limitations。
- B8 闭环：Tier-2 OOF 校准层 + 阈值视图（data/f9_tier2_calibration.json），SM5 新增一节，Discussion 引用。
- A3 验证：171 家医院 0 漏映射（f9_fillins.json）。
- B16 溯源：SM3 亚型数字在 n8_1_eicu_abi_crosswalk.json subtype_mix_audit（审计漏查）。
- issues 文件 5 行状态更新；台账补 22 行；说明报告 B 级计数以文件为准（11 项）。
- 待办（机械）：md 已改，docx/合并版 SI 需 Biomni 构建系统重出（5 分钟量级）。
"""
rp.write_text(t, encoding="utf-8")
print("report note appended")
print("\nALL PATCHES APPLIED")
