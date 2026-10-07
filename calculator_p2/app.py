# -*- coding: utf-8 -*-
"""ABI-AKI Rolling Risk Calculator — Paper 2 (JinhuaNSICU-developed model)
Bilingual (English default / 中文), journal-style UI built on the Paper-1 final app (v2026.09.22).

Model : LightGBM, class-weighted, 166 of 280 candidate features (frozen, inference only)
        any AKI (>= KDIGO creatinine Stage 1, primary) + frozen logistic recalibration
        severe AKI (>= Stage 3, secondary) reported as a raw ranking score
Evidence (frozen suite f12): internal AUROC 0.815 (0.754-0.872); external MIMIC-IV 0.769,
        eICU-CRD 0.777; recalibrated slope 0.909 / 0.839 / 0.881
Run   : streamlit run app.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

import riskcalc as rc

from matplotlib import font_manager
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples_local"
VERSION = "v2026.10.07 · JinhuaNSICU development"
BAND_COLOR = ["#2e7d32", "#b8860b", "#c62828"]
BAND_TINT = ["#edf7ee", "#fff8e1", "#fdecea"]
BAND_EDGE = ["#2e7d32", "#f9a825", "#c62828"]

st.set_page_config(page_title="ABI-AKI Risk Calculator · Paper 2", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ style (Paper-1 final CSS + additions)
st.markdown("""
<style>
  :root { --ink:#1a2b3c; --muted:#5b7083; --accent:#0f6e8c; --card:#ffffff; --bg:#f6f8fa; --navy:#0f2f4a; }
  .stApp { background: var(--bg); font-family: 'Source Sans Pro','Source Sans 3','Segoe UI',system-ui,sans-serif; color: var(--ink); }
  .block-container { padding-top: 2.2rem; max-width: 1480px; }
  section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e3e9ef; }
  section[data-testid="stSidebar"] * { font-size:.92rem; }
  .banner { background:#fdecea; border:1px solid #f1b8b4; border-left:5px solid #c62828; color:#8e1b1b;
            border-radius:10px; padding:9px 14px; margin-bottom:12px; font-size:.86rem; }
  .hdr { display:flex; align-items:center; gap:14px; padding:18px 22px; margin-bottom:14px;
         background:linear-gradient(90deg,#0f2f4a 0%,#0f6e8c 100%); border-radius:12px; color:#fff; }
  .hdr .badge { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35);
                padding:3px 10px; border-radius:999px; font-size:.78rem; letter-spacing:.4px; white-space:nowrap; }
  .card { background:var(--card); border:1px solid #e3e9ef; border-radius:12px; padding:18px 20px; margin-bottom:12px; }
  div[data-testid="stVerticalBlockBorderWrapper"] { background:#ffffff; border-color:#e3e9ef !important; border-radius:12px; }
  .risk-num { font-size:3.1rem; font-weight:700; line-height:1; margin-top:4px; }
  .lbl { font-size:.78rem; text-transform:uppercase; letter-spacing:.8px; color:var(--muted); }
  .sec { font-size:.95rem; font-weight:700; color:var(--navy); margin:2px 0 6px; }
  .gwrap { position:relative; margin:22px 0 2px; }
  .gauge { position:relative; height:14px; border-radius:7px;
           background:linear-gradient(90deg,#2e7d32 0%,#2e7d32 19.6%,#f9a825 20.4%,#f9a825 29.6%,#c62828 30.4%,#c62828 100%); }
  .gauge .div { position:absolute; top:0; width:2px; height:14px; background:rgba(255,255,255,.85); transform:translateX(-50%); }
  .tri { position:absolute; top:-15px; width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
         border-top:12px solid var(--navy); transform:translateX(-50%); }
  .pin { position:absolute; top:-3px; width:3px; height:20px; background:var(--navy); border-radius:2px;
         box-shadow:0 0 0 2px #fff; transform:translateX(-50%); }
  .gticks { position:relative; height:18px; font-size:.72rem; color:var(--muted); margin-top:4px; }
  .gticks span { position:absolute; transform:translateX(-50%); white-space:nowrap; }
  .gticks span.wk { color:#c62828; font-weight:700; }
  .kpi { padding:10px 10px; background:#f2f6f8; border-radius:10px; }
  .kpi .v { font-size:1.15rem; font-weight:700; color:var(--ink); }
  .evi { font-size:.85rem; color:var(--muted); line-height:1.55; }
  .evi b { color:var(--ink); }
  .act { border-radius:12px; padding:14px 18px; margin-bottom:12px; border:1px solid #e3e9ef; }
  .act .t { font-size:1.02rem; font-weight:600; color:var(--ink); margin-top:5px; }
  .note { background:#eef5f9; border:1px solid #d4e5ee; border-radius:10px; padding:9px 12px; font-size:.83rem; color:#24445c; margin:4px 0 10px; }
  .chip { display:inline-block; background:#fff3cd; border:1px solid #eadfa8; color:#7a5b00; border-radius:8px;
          padding:5px 10px; font-size:.8rem; margin-top:8px; }
  .lgwrap { position:relative; margin:18px 0 2px; }
  .lg { position:relative; height:10px; border-radius:5px; background:linear-gradient(90deg,#dfe7ee 0%,#9fb4c5 100%); }
  .lg .ref { position:absolute; top:-5px; width:2px; height:20px; background:#c62828; transform:translateX(-50%); }
  .sevnum { font-size:1.7rem; font-weight:700; color:var(--ink); line-height:1.1; }
  .foot { font-size:.78rem; color:var(--muted); margin-top:6px; }
  .keyres { background:#fff8e1; border:1px solid #f3d27a; border-left:5px solid #e0a100; border-radius:10px;
            padding:10px 14px; margin-top:8px; }
  .keytag { display:inline-block; background:#e0a100; color:#fff; font-weight:700; font-size:.72rem; letter-spacing:.8px;
            padding:2px 9px; border-radius:999px; margin-right:8px; vertical-align:middle; }
  .keyres .v { font-size:1.9rem; font-weight:800; color:var(--ink); line-height:1.15; }
  .info { background:#eef5f9; border:1px solid #c9dfeb; border-left:5px solid #0f6e8c; color:#18405a;
          border-radius:10px; padding:9px 14px; margin-bottom:12px; font-size:.86rem; }
  div[data-testid="stTabs"] button p { font-size:.98rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ strings (EN primary / 中文)
S = {
    "en": dict(
        banner="<b>Externally validated on MIMIC-IV (7,442 stays) and eICU-CRD (9,071 stays, 171 hospitals).</b> "
               "Designed for clinical decision support; run silent-mode evaluation, confirm thresholds and recalibrate "
               "locally before clinical use.",
        subtitle="Rolling 48-h incident AKI prediction at 6-h checkpoints · KDIGO creatinine criteria · "
                 "any AKI (≥Stage 1) + severe AKI (≥Stage 3)",
        tab1="① Single-point calculator", tab2="② Batch prediction (pipeline CSV)",
        side_lang="Language / 语言",
        side_cal="Frozen recalibration layer",
        side_cal_src="any-AKI only · read at runtime from model/calibration.json",
        side_ref="Reference lines",
        side_ref_any="any-AKI (calibrated): 0.10 · <b>0.15 working</b> · 0.20",
        side_ref_sev="severe-AKI (raw ranking): median-input reference {ref}",
        side_tpl="Batch template",
        side_tpl_cap="stay_id, t_hr + 166 model features (one median row)",
        side_tpl_btn="⬇ Download 166-column template",
        pf="🧾 Patient features",
        pf_note="<b>{n} of 166</b> model features are derived from these bedside inputs; the other <b>{m}</b> are held at "
                "the JinhuaNSICU development-cohort median.",
        g_demo="Demographics", g_cr="Creatinine kinetics", g_vit="Vital signs · labs · vasopressor",
        age="Age (yr)", charlson="Charlson comorbidity index", male="Male (context only)", male_help="Sex is not among the 166 model features; recorded for context only.",
        cr_base="Baseline creatinine, mg/dL", cr_cur="Current creatinine, mg/dL",
        cr_low24="Lowest creatinine in past 24 h, mg/dL",
        cr_first48="First creatinine in 48-h window (≈48 h ago), mg/dL",
        t_hr="Checkpoint (ICU hour)",
        t_hr_help="Sets the time span of the 48-h creatinine trend (slope). Not a model feature itself.",
        sbp="Systolic BP, mmHg", hr="Heart rate, bpm", spo2="SpO₂, %", rr="Respiratory rate, /min",
        bun="Blood urea nitrogen (BUN), mg/dL",
        norepi="Norepinephrine administrations in past 24 h (count)",
        kdigo="Entered creatinine already meets a KDIGO Stage 1 criterion ({why}); the model predicts "
              "<i>incident</i> AKI at checkpoints before onset.",
        kdigo_ratio="≥1.5× baseline", kdigo_rise="≥0.3 mg/dL rise within 48 h",
        risk_lbl="Calibrated 48-h AKI risk",
        bands=["LOW", "INTERMEDIATE", "HIGH"],
        raw_cal="raw {a} → calibrated {b}",
        ticks=["0%", "10%", "15% working", "20%", "50%+"],
        action_lbl="Suggested action (demo)",
        action=["Routine monitoring — renal function per unit protocol.",
                "Intensified renal surveillance — repeat creatinine, review nephrotoxic drugs and fluid targets.",
                "Nephrology consultation and renal-replacement therapy (RRT) readiness."],
        workpoint="At the 0.15 working threshold (internal test): {cap} of AKI stays captured (15/33) · median lead time "
                  "18 h · NNE 4.47 · 6.07 false alarms per 100 patient-days. External capture: MIMIC-IV 32.7% · eICU-CRD 34.8%.",
        sev_lbl="Severe AKI (≥Stage 3) · raw score",
        sev_note="Ranking score — no calibrated probability scale",
        sev_fold="{fold} × the median-input reference",
        sev_ref="median-input ref.",
        sev_ref_note="Reference = raw score with every feature at its development-cohort median; not a cohort percentile.",
        evi_lbl="Model evidence",
        key=("KEY RESULT", "Severe AKI (≥Stage 3) · external AUROC", "0.922 · 0.878",
             "MIMIC-IV · eICU-CRD · 95% CI 0.882–0.954 · 0.791–0.953 · 87 / 83 event stays",
             "Severe AKI, the endpoint that calls for nephrology review, was the best-discriminated endpoint in both "
             "external cohorts. Internal estimate 0.995 rests on 3 event stays (underpowered)."),
        kpi=[("Internal · any AKI", "0.815", "JinhuaNSICU test · 208 stays", "95% CI 0.754–0.872"),
             ("External · any AKI", "0.769 · 0.777", "MIMIC-IV · eICU-CRD", "95% CI 0.748–0.791 · 0.759–0.795"),
             ("Calibration slope", "0.909 · 0.839 · 0.881", "internal · MIMIC-IV · eICU-CRD", "recalibrated (raw 0.244 · 0.225 · 0.236)")],
        evi_body=("<b>Model</b>: LightGBM, class-weighted; 166 of 280 candidate features (static · 24-h rolling windows · "
                  "LOCF · 48-h trends · variability · medications), NaN-native<br>"
                  "<b>Endpoint</b>: incident AKI within 48 h by KDIGO creatinine criteria — any AKI ≥Stage 1 (primary) · "
                  "severe AKI ≥Stage 3 (secondary)<br>"
                  "<b>Cohorts</b>: n = 1,033 / 7,442 / 9,071 ICU stays — JinhuaNSICU development (825 training · 208 internal test) / "
                  "MIMIC-IV external / eICU-CRD external; models applied externally without refitting<br>"
                  "<b>Explainability</b>: SHAP — current-to-baseline creatinine ratio leads (mean |SHAP| 1.476, ≈2× age 0.776), "
                  "then creatinine last in 48 h · Charlson · systolic BP<br>"
                  "<b>Risk bands</b>: operating points 0.05 / 0.08 / 0.10 / <b>0.15 working</b> / 0.20 on the calibrated scale<br>"
                  "<b>Inputs excluded by design</b>: no urine output, no GCS"),
        methods="📐 How single-point inputs map to the 166 features",
        methods_items=[
            "Creatinine: current value → most recent / last available / last in 48 h; current ÷ baseline → ratio; "
            "current − baseline → change since admission; lowest in 24 h → 24-h minimum.",
            "48-h creatinine trend: points at (checkpoint − min(checkpoint, 48 h), first value), (checkpoint − 12 h, lowest 24-h value) "
            "and (checkpoint, current); slope = least-squares mg/dL per hour, SD with ddof = 1, change = current − first — "
            "the same definitions as the development feature pipeline.",
            "BUN → most recent / last available / last in 48 h; SBP → last available / last in 48 h; HR, RR → most recent, with "
            "the 24-h maximum/minimum widened to include the entered value; SpO₂ → most recent / last available.",
            "Norepinephrine enters as the number of administrations in the past 24 h (development feature norepi_n24).",
            "Batch mode scores pipeline-computed features directly and is the reference mode for research use."],
        b_desc="Upload a pipeline feature CSV — <b>one row per 6-h checkpoint</b>; columns = <code>stay_id</code>, "
               "<code>t_hr</code> (ICU hour) + the 166 model features (template in the sidebar; blank = missing, handled natively). "
               "Or open one of the three bundled synthetic trajectories, each scored live by the frozen model.",
        b_src="Input source", b_up="Upload CSV", b_file="Feature CSV",
        b_ex=["Example A · stable creatinine", "Example B · gradual creatinine rise",
              "Example C · rapid creatinine rise + norepinephrine"],
        b_peak="{band} · peak {p}",
        b_stay="Stay", b_missing="{k} of 166 feature columns absent → treated as missing (NaN): {first}",
        b_kpi=["Checkpoints", "Peak calibrated any-AKI", "First ≥0.15 (working)", "Peak severe raw score"],
        b_never="not reached",
        b_title_any="Any AKI within 48 h · calibrated probability", b_title_sev="Severe AKI within 48 h · raw ranking score",
        b_x="ICU hour (checkpoint)", b_y_any="Calibrated probability", b_y_sev="Raw score (log scale)",
        b_working="0.15 working", b_tab="Scored checkpoints",
        b_cols={"t_hr": "ICU hour", "any_p_cal": "Any-AKI calibrated", "band_lbl": "Band", "any_p_raw": "Any-AKI raw",
                "severe_p_raw": "Severe-AKI raw (ranking)", "alert": "≥0.15 working"},
        b_dl="⬇ Download scored checkpoints (all stays)",
        foot="Computed locally; uploaded files are not stored or transmitted.",
    ),
    "zh": dict(
        banner="<b>已在 MIMIC-IV（7,442 住院次）和 eICU-CRD（9,071 住院次，171 家医院）完成外部验证。</b>"
               "面向临床决策支持设计；临床使用前须在本地进行静默运行评估、确认阈值并重新校准。",
        subtitle="每 6 小时检查点滚动预测未来 48 h 新发 AKI · KDIGO 肌酐标准 · any AKI（≥1 期）+ severe AKI（≥3 期）",
        tab1="① 单点计算器", tab2="② 批量预测（管线 CSV）",
        side_lang="Language / 语言",
        side_cal="冻结重校准层",
        side_cal_src="仅用于 any-AKI · 运行时读取 model/calibration.json",
        side_ref="参考线",
        side_ref_any="any-AKI（校准）：0.10 · <b>0.15 工作阈值</b> · 0.20",
        side_ref_sev="severe-AKI（raw 排名分）：中位输入参考值 {ref}",
        side_tpl="批量模板",
        side_tpl_cap="stay_id、t_hr + 166 个模型特征（一行中位数）",
        side_tpl_btn="⬇ 下载 166 列模板",
        pf="🧾 患者特征",
        pf_note="以下床旁输入派生 <b>166 个中的 {n} 个</b>模型特征；其余 <b>{m} 个</b>取 JinhuaNSICU 开发队列中位数。",
        g_demo="人口学", g_cr="肌酐动力学", g_vit="生命体征 · 化验 · 血管活性药",
        age="年龄（岁）", charlson="Charlson 合并症指数", male="男性（仅记录）", male_help="性别不在 166 个模型特征中，仅作记录。",
        cr_base="基线肌酐 mg/dL", cr_cur="当前肌酐 mg/dL",
        cr_low24="过去 24 h 最低肌酐 mg/dL",
        cr_first48="48 h 窗内首次肌酐（约 48 h 前）mg/dL",
        t_hr="检查点（ICU 小时）",
        t_hr_help="决定 48 h 肌酐趋势（斜率）的时间跨度；本身不是模型特征。",
        sbp="收缩压 mmHg", hr="心率 bpm", spo2="SpO₂ %", rr="呼吸频率 /min",
        bun="血尿素氮（BUN）mg/dL",
        norepi="过去 24 h 去甲肾上腺素给药次数",
        kdigo="所输入肌酐已满足 KDIGO 1 期标准（{why}）；本模型预测的是发病前检查点的<i>新发</i> AKI。",
        kdigo_ratio="≥1.5 倍基线", kdigo_rise="48 h 内升高 ≥0.3 mg/dL",
        risk_lbl="校准后 48 h AKI 风险",
        bands=["低风险", "中风险", "高风险"],
        raw_cal="原始 {a} → 校准 {b}",
        ticks=["0%", "10%", "15% 工作阈值", "20%", "50%+"],
        action_lbl="建议行动（演示）",
        action=["常规监测——按科室流程复查肾功能。",
                "强化肾功能监测——复查肌酐，复核肾毒性药物与液体目标。",
                "肾内科会诊，并做好肾脏替代治疗（RRT）准备。"],
        workpoint="0.15 工作阈值（内部测试集）：捕获 {cap} 的 AKI 住院次（15/33）· 中位提前 18 h · NNE 4.47 · "
                  "每 100 病人日假警报 6.07 次。外部捕获率：MIMIC-IV 32.7% · eICU-CRD 34.8%。",
        sev_lbl="Severe AKI（≥3 期）· raw 分",
        sev_note="排名分——无校准概率刻度",
        sev_fold="为中位输入参考值的 {fold} 倍",
        sev_ref="中位输入参考",
        sev_ref_note="参考值 = 所有特征取开发队列中位数时的 raw 分；不是队列分位数。",
        evi_lbl="模型证据",
        key=("关键结果", "Severe AKI（≥3 期）· 外部 AUROC", "0.922 · 0.878",
             "MIMIC-IV · eICU-CRD · 95% CI 0.882–0.954 · 0.791–0.953 · 事件住院次 87 / 83",
             "Severe AKI（需肾内科会诊的终点）在两个外部队列中均为区分度最高的终点。内部估计 0.995 仅基于 3 个事件住院次（检验效能不足）。"),
        kpi=[("内部验证 · any AKI", "0.815", "JinhuaNSICU 测试集 · 208 住院次", "95% CI 0.754–0.872"),
             ("外部验证 · any AKI", "0.769 · 0.777", "MIMIC-IV · eICU-CRD", "95% CI 0.748–0.791 · 0.759–0.795"),
             ("校准斜率", "0.909 · 0.839 · 0.881", "内部 · MIMIC-IV · eICU-CRD", "重校准后（原始 0.244 · 0.225 · 0.236）")],
        evi_body=("<b>模型</b>：LightGBM（类加权），280 个候选特征中选定 166 个（静态 · 24 h 滚动窗 · LOCF · 48 h 趋势 · 变异度 · 用药），NaN 原生<br>"
                  "<b>终点</b>：未来 48 h 新发 AKI（KDIGO 肌酐标准）——any AKI ≥1 期（主要）· severe AKI ≥3 期（次要）<br>"
                  "<b>队列</b>：n = 1,033 / 7,442 / 9,071 ICU 住院次——JinhuaNSICU 开发（训练 825 · 内部测试 208）/ "
                  "MIMIC-IV 外部 / eICU-CRD 外部；外部应用不重新拟合<br>"
                  "<b>可解释性</b>：SHAP——当前/基线肌酐比居首（平均 |SHAP| 1.476，约为年龄 0.776 的 2 倍），"
                  "其次为 48 h 窗内末次肌酐 · Charlson · 收缩压<br>"
                  "<b>风险分带</b>：校准刻度上的操作点 0.05 / 0.08 / 0.10 / <b>0.15 工作阈值</b> / 0.20<br>"
                  "<b>设计上排除的输入</b>：无尿量，无 GCS"),
        methods="📐 单点输入如何映射为 166 个特征",
        methods_items=[
            "肌酐：当前值 → 最近值 / 末次可得值 / 48 h 窗内末次值；当前 ÷ 基线 → 比值；当前 − 基线 → 入院以来变化；24 h 最低 → 24 h 最小值。",
            "48 h 肌酐趋势：取（检查点 − min(检查点, 48 h)，首次值）、（检查点 − 12 h，24 h 最低值）、（检查点，当前值）三点；"
            "斜率 = 最小二乘 mg/dL/小时，SD 取 ddof = 1，变化 = 当前 − 首次——与开发特征管线定义一致。",
            "BUN → 最近值 / 末次可得值 / 48 h 窗内末次值；收缩压 → 末次可得值 / 48 h 窗内末次值；心率、呼吸 → 最近值，"
            "24 h 最大/最小值扩展至包含输入值；SpO₂ → 最近值 / 末次可得值。",
            "去甲肾上腺素以过去 24 h 给药次数进入模型（开发特征 norepi_n24）。",
            "批量模式直接对管线计算的特征打分，是研究用途的参考模式。"],
        b_desc="上传管线特征 CSV——<b>每行一个 6 小时检查点</b>；列 = <code>stay_id</code>、<code>t_hr</code>（ICU 小时）"
               "+ 166 个模型特征（模板见侧栏；空白 = 缺失，模型原生处理）。也可打开下方三个合成示例轨迹，均由冻结模型实时打分。",
        b_src="输入来源", b_up="上传 CSV", b_file="特征 CSV",
        b_ex=["示例 A · 肌酐平稳", "示例 B · 肌酐缓慢上升", "示例 C · 肌酐快速上升 + 去甲肾上腺素"],
        b_peak="{band} · 峰值 {p}",
        b_stay="住院次", b_missing="166 个特征列中缺少 {k} 个 → 按缺失（NaN）处理：{first}",
        b_kpi=["检查点数", "any-AKI 校准峰值", "首次 ≥0.15（工作阈值）", "severe raw 峰值"],
        b_never="未达到",
        b_title_any="未来 48 h any AKI · 校准概率", b_title_sev="未来 48 h severe AKI · raw 排名分",
        b_x="ICU 小时（检查点）", b_y_any="校准概率", b_y_sev="raw 分（对数刻度）",
        b_working="0.15 工作阈值", b_tab="检查点打分表",
        b_cols={"t_hr": "ICU 小时", "any_p_cal": "any-AKI 校准", "band_lbl": "风险带", "any_p_raw": "any-AKI raw",
                "severe_p_raw": "severe-AKI raw（排名）", "alert": "≥0.15 工作阈值"},
        b_dl="⬇ 下载检查点打分（全部住院次）",
        foot="本地计算；上传文件不存储、不传输。",
    ),
}

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_p(p: float) -> str:
    """Percent for >= 0.1 %; otherwise scientific notation (raw scores are often ~1e-6)."""
    if p >= 0.001:
        return f"{p * 100:.1f}%"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10{str(int(e)).translate(SUP)}"


@st.cache_resource
def load():
    return rc.load_models()


M = load()
SEV_REF = rc.severe_reference(M)   # median-input profile, raw severe score (contextual line only)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"**{S['en']['side_lang']}**")
    LANG = st.radio("Language / 语言", ["English", "中文"], index=0, label_visibility="collapsed", key="lang")
T = S["en" if LANG == "English" else "zh"]
ZH = LANG == "中文"

with st.sidebar:
    st.divider()
    st.markdown(f"**{T['side_cal']}**")
    st.markdown(f"<div class='evi'>p_cal = sigmoid(a + b · logit p_raw)<br>"
                f"a = <b>{M['a']:.6f}</b> &nbsp;·&nbsp; b = <b>{M['b']:.6f}</b><br>{T['side_cal_src']}</div>",
                unsafe_allow_html=True)
    st.divider()
    st.markdown(f"**{T['side_ref']}**")
    st.markdown(f"<div class='evi'>{T['side_ref_any']}<br>{T['side_ref_sev'].format(ref=fmt_p(SEV_REF))}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"**{T['side_tpl']}**")
    st.caption(T["side_tpl_cap"])
    st.download_button(T["side_tpl_btn"], rc.template_frame(M).to_csv(index=False).encode(),
                       "feature_template.csv", "text/csv", width="stretch")
    st.divider()
    st.caption(VERSION)

# ------------------------------------------------------------------ banner + header
st.markdown(f'<div class="info">ℹ️ {T["banner"]}</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="hdr">
  <div style="font-size:1.9rem">🧠</div>
  <div>
    <div style="font-size:1.28rem;font-weight:700">Acute Brain Injury · Rolling AKI Risk</div>
    <div style="font-size:.82rem;opacity:.9">{T['subtitle']}</div>
  </div>
  <div class="badge" style="margin-left:auto">{VERSION}</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs([T["tab1"], T["tab2"]])

# ================================================================== ① single point
with tab1:
    cL, cR = st.columns([1, 1.22], gap="medium")
    D = rc.BEDSIDE_DEFAULTS
    with cL:
        with st.container(border=True):
            st.markdown(f"<div class='sec' style='font-size:1.05rem'>{T['pf']}</div>", unsafe_allow_html=True)
            note_slot = st.empty()
            a1, a2 = st.columns(2, gap="medium")
            with a1:
                st.markdown(f"<div class='lbl'>{T['g_demo']}</div>", unsafe_allow_html=True)
                age = st.slider(T["age"], 18, 95, D["age"], key="age")
                charlson = st.slider(T["charlson"], 0, 15, D["charlson"], key="charlson")
                male = st.checkbox(T["male"], value=D["male"], key="male", help=T["male_help"])
                st.markdown(f"<div class='lbl' style='margin-top:10px'>{T['g_cr']}</div>", unsafe_allow_html=True)
                cr_base = st.number_input(T["cr_base"], 0.20, 8.00, D["cr_base"], 0.01, format="%.2f", key="cr_base")
                cr_cur = st.number_input(T["cr_cur"], 0.20, 10.00, D["cr_cur"], 0.01, format="%.2f", key="cr_cur")
                cr_low24 = st.number_input(T["cr_low24"], 0.20, 10.00, D["cr_low24"], 0.01, format="%.2f", key="cr_low24")
                cr_first48 = st.number_input(T["cr_first48"], 0.20, 10.00, D["cr_first48"], 0.01, format="%.2f", key="cr_first48")
            with a2:
                st.markdown(f"<div class='lbl'>{T['g_vit']}</div>", unsafe_allow_html=True)
                t_hr = st.slider(T["t_hr"], 24, 168, D["t_hr"], 6, help=T["t_hr_help"], key="t_hr")
                sbp = st.slider(T["sbp"], 60, 220, D["sbp"], key="sbp")
                hr = st.slider(T["hr"], 30, 180, D["hr"], key="hr")
                spo2 = st.slider(T["spo2"], 70, 100, D["spo2"], key="spo2")
                rr = st.slider(T["rr"], 6, 45, D["rr"], key="rr")
                bun = st.number_input(T["bun"], 2.0, 150.0, D["bun"], 0.1, format="%.1f", key="bun")
                norepi = st.number_input(T["norepi"], 0, 4, D["norepi_n24"], 1, key="norepi")

    b = dict(age=age, charlson=charlson, male=male, cr_base=cr_base, cr_cur=cr_cur, cr_low24=cr_low24,
             cr_first48=cr_first48, t_hr=t_hr, sbp=sbp, hr=hr, spo2=spo2, rr=rr, bun=bun, norepi_n24=norepi)
    R = rc.predict_bedside(b, M)
    note_slot.markdown(f"<div class='note'>ℹ️ {T['pf_note'].format(n=R['n_set'], m=166 - R['n_set'])}</div>",
                       unsafe_allow_html=True)
    p, praw, psev, k = R["any_p_cal"], R["any_p_raw"], R["severe_p_raw"], R["band"]

    why = []
    if cr_cur >= 1.5 * cr_base:
        why.append(T["kdigo_ratio"])
    if cr_cur - min(cr_first48, cr_low24) >= 0.3 - 1e-9:
        why.append(T["kdigo_rise"])

    with cR:
        # --- risk card (one HTML block: label, number, gauge, band, raw->cal)
        pos = min(max(p / 0.5, 0.0), 1.0) * 100
        ticks = T["ticks"]
        kd = f"<div class='chip'>⚑ {T['kdigo'].format(why=' · '.join(why))}</div>" if why else ""
        st.markdown(f"""
        <div class="card">
          <div class="lbl">{T['risk_lbl']}</div>
          <div class="risk-num" style="color:{BAND_COLOR[k]}">{p * 100:.1f}%</div>
          <div class="gwrap">
            <div class="tri" style="left:{pos:.1f}%"></div>
            <div class="gauge"><div class="div" style="left:40%"></div><div class="pin" style="left:{pos:.1f}%"></div></div>
          </div>
          <div class="gticks">
            <span style="left:0;transform:none">{ticks[0]}</span><span style="left:20%">{ticks[1]}</span>
            <span class="wk" style="left:30%">{ticks[2]}</span><span style="left:40%">{ticks[3]}</span>
            <span style="left:100%;transform:translateX(-100%)">{ticks[4]}</span>
          </div>
          <div style="margin-top:8px"><span style="color:{BAND_COLOR[k]};font-weight:700;font-size:1.05rem">{T['bands'][k]}</span>
          &nbsp;·&nbsp;<span style="color:#5b7083;font-size:.88rem">{T['raw_cal'].format(a=fmt_p(praw), b=f"{p * 100:.1f}%")}</span></div>
          {kd}
        </div>
        """, unsafe_allow_html=True)
        # --- suggested action
        st.markdown(f"""
        <div class="act" style="background:{BAND_TINT[k]};border-left:5px solid {BAND_EDGE[k]}">
          <div class="lbl">{T['action_lbl']}</div>
          <div class="t">{T['action'][k]}</div>
          <div class="evi" style="margin-top:6px">{T['workpoint'].format(cap='45.5%')}</div>
        </div>
        """, unsafe_allow_html=True)
        # --- severe AKI (log-scale strip 1e-8 .. 1)
        lo, hi = -8.0, 0.0
        sev_pos = (min(max(math.log10(max(psev, 1e-12)), lo), hi) - lo) / (hi - lo) * 100
        ref_pos = (math.log10(SEV_REF) - lo) / (hi - lo) * 100
        fold = psev / SEV_REF
        fold_s = f"{fold:.1f}" if fold < 100 else f"{fold:,.0f}"
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap">
            <div><div class="lbl">{T['sev_lbl']}</div><div class="sevnum">{fmt_p(psev)}</div></div>
            <div class="evi"><b>{T['sev_fold'].format(fold=fold_s)}</b> · <i>{T['sev_note']}</i></div>
          </div>
          <div class="lgwrap">
            <div class="tri" style="left:{sev_pos:.1f}%;top:-14px;border-top-width:10px;border-left-width:7px;border-right-width:7px"></div>
            <div class="lg"><div class="ref" style="left:{ref_pos:.1f}%"></div></div>
          </div>
          <div class="gticks">
            <span style="left:0;transform:none">10⁻⁸</span><span style="left:25%">10⁻⁶</span><span style="left:50%">10⁻⁴</span>
            <span style="left:75%">10⁻²</span><span class="wk" style="left:{ref_pos:.1f}%;top:-34px">{T['sev_ref']}</span>
            <span style="left:100%;transform:translateX(-100%)">1</span>
          </div>
          <div class="evi" style="margin-top:6px;font-size:.78rem">{T['sev_ref_note']}</div>
        </div>
        """, unsafe_allow_html=True)
        # --- model evidence
        k1, k2, k3 = T["kpi"]
        kp = lambda x: (f'<div class="kpi" style="flex:1"><div class="lbl">{x[0]}</div><div class="v">{x[1]}</div>'
                        f'<div class="evi">{x[2]}<br>{x[3]}</div></div>')
        st.markdown(f"""
        <div class="card">
          <div class="lbl">{T['evi_lbl']}</div>
          <div class="keyres"><span class="keytag">{T['key'][0]}</span><span class="lbl">{T['key'][1]}</span>
            <div class="v">{T['key'][2]}</div><div class="evi">{T['key'][3]}</div>
            <div class="evi" style="margin-top:4px"><b>{T['key'][4]}</b></div></div>
          <div style="display:flex;gap:10px;margin-top:8px">{kp(k1)}{kp(k2)}{kp(k3)}</div>
          <div class="evi" style="margin-top:12px">{T['evi_body']}</div>
        </div>
        """, unsafe_allow_html=True)
    with cL:
        with st.expander(T["methods"]):
            st.markdown("\n".join(f"- {x}" for x in T["methods_items"]))
    st.markdown(f"<div class='foot'>{T['foot']} · {VERSION}</div>", unsafe_allow_html=True)


# ================================================================== ② batch
@st.cache_data
def score_examples():
    out = {}
    for i, f in enumerate(sorted(EXAMPLES.glob("demo_*.csv"), key=lambda x: ["low_rise", "intermediate", "rapid_rise"].index(x.stem[5:]))):
        d = pd.read_csv(f)
        s, _ = rc.score_frame(d, M)
        out[i] = (f.name, d, s)
    return out


def traj_figure(r: pd.DataFrame):
    plt.rcParams.update({"font.family": ["Liberation Sans", "Noto Sans CJK JP", "DejaVu Sans"],
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.9), dpi=150)
    x = r["t_hr"].to_numpy()
    ymax = max(0.30, float(r["any_p_cal"].max()) * 1.18)
    ax1.axhspan(0, 0.10, color="#2e7d32", alpha=.07, lw=0)
    ax1.axhspan(0.10, 0.15, color="#f9a825", alpha=.12, lw=0)
    ax1.axhspan(0.15, ymax, color="#c62828", alpha=.06, lw=0)
    for y, lw, lab in ((0.10, 1.0, "0.10"), (0.15, 1.6, T["b_working"]), (0.20, 1.0, "0.20")):
        ax1.axhline(y, ls="--", lw=lw, color="#c62828" if y == 0.15 else "#8a9aa8")
        ax1.text(1.01, y, lab, va="center", fontsize=8, color="#c62828" if y == 0.15 else "#5b7083",
                 transform=ax1.get_yaxis_transform(), clip_on=False)
    ax1.plot(x, r["any_p_cal"], "-o", color="#0f6e8c", lw=2, ms=4.5)
    ax1.set_ylim(0, ymax)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax1.set_title(T["b_title_any"], fontsize=11, loc="left", color="#0f2f4a", fontweight="bold")
    ax1.set_ylabel(T["b_y_any"])
    ax2.plot(x, r["severe_p_raw"], "-o", color="#5b7083", lw=2, ms=4.5)
    ax2.axhline(SEV_REF, ls="--", lw=1.4, color="#c62828")
    ax2.text(1.01, SEV_REF, T["sev_ref"], va="center", fontsize=8, color="#c62828",
             transform=ax2.get_yaxis_transform(), clip_on=False)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-8, 1)
    ax2.set_title(T["b_title_sev"], fontsize=11, loc="left", color="#0f2f4a", fontweight="bold")
    ax2.set_ylabel(T["b_y_sev"])
    for ax in (ax1, ax2):
        ax.set_xlabel(T["b_x"])
        ax.set_xlim(x.min() - 4, x.max() + 4)
        ax.set_xticks(range(int(x.min()), int(x.max()) + 1, 24))
        ax.grid(axis="y", color="#e3e9ef", lw=.8)
        ax.tick_params(labelsize=9)
    fig.tight_layout(w_pad=3)
    return fig


with tab2:
    EX = score_examples()
    st.markdown(f"<div class='card' style='padding:14px 18px'><div class='evi' style='color:#1a2b3c'>{T['b_desc']}</div></div>",
                unsafe_allow_html=True)
    opts = [T["b_up"]]
    for i in range(3):
        s = EX[i][2]
        pk = float(s["any_p_cal"].max())
        opts.append(f"{T['b_ex'][i]} — {T['b_peak'].format(band=T['bands'][rc.band(pk)], p=f'{pk * 100:.1f}%')}")
    cA, cB = st.columns([1, 1.6], gap="medium")
    with cA:
        with st.container(border=True):
            src_i = st.radio(T["b_src"], list(range(4)), index=3, format_func=lambda j: opts[j], key="src")
            df = scored = None
            if src_i == 0:
                up = st.file_uploader(T["b_file"], type=["csv"], key="up")
                if up is not None:
                    df = pd.read_csv(up)
                    scored, miss = rc.score_frame(df, M)
                    if miss:
                        st.warning(T["b_missing"].format(k=len(miss), first=", ".join(miss[:6]) + (" …" if len(miss) > 6 else "")))
            else:
                i = src_i - 1
                _, df, scored = EX[i]
            if scored is not None:
                stays = list(pd.unique(scored["stay_id"]))
                sid = st.selectbox(T["b_stay"], stays, key="stay")
    if scored is not None:
        r = scored[scored["stay_id"] == sid].copy()
        if "t_hr" not in r.columns:
            r["t_hr"] = range(24, 24 + 6 * len(r), 6)
        r = r.sort_values("t_hr")
        first = r.loc[r["any_p_cal"] >= rc.ANY_WORKING, "t_hr"]
        kv = [str(len(r)), f"{r['any_p_cal'].max() * 100:.1f}%",
              (f"{int(first.iloc[0])} h" if len(first) else T["b_never"]), fmt_p(float(r["severe_p_raw"].max()))]
        with cB:
            pk = rc.band(float(r["any_p_cal"].max()))
            cells = "".join(f'<div class="kpi" style="flex:1"><div class="lbl">{T["b_kpi"][j]}</div>'
                            f'<div class="v" style="color:{BAND_COLOR[pk] if j == 1 else "#1a2b3c"}">{kv[j]}</div></div>'
                            for j in range(4))
            st.markdown(f"<div class='card'><div class='lbl'>{sid}</div><div style='display:flex;gap:10px;margin-top:8px'>{cells}</div></div>",
                        unsafe_allow_html=True)
        with st.container(border=True):
            fig = traj_figure(r)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        show = r.copy()
        show["band_lbl"] = [T["bands"][x] for x in show["band"]]
        show["alert"] = show["any_p_cal"] >= rc.ANY_WORKING
        show["any_p_cal"] = show["any_p_cal"] * 100
        show = show[["t_hr", "any_p_cal", "band_lbl", "any_p_raw", "severe_p_raw", "alert"]]
        st.markdown(f"<div class='sec'>{T['b_tab']}</div>", unsafe_allow_html=True)
        C = T["b_cols"]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "t_hr": st.column_config.NumberColumn(C["t_hr"], format="%d"),
            "any_p_cal": st.column_config.NumberColumn(C["any_p_cal"], format="%.1f%%"),
            "band_lbl": st.column_config.TextColumn(C["band_lbl"]),
            "any_p_raw": st.column_config.NumberColumn(C["any_p_raw"], format="%.2e"),
            "severe_p_raw": st.column_config.NumberColumn(C["severe_p_raw"], format="%.2e"),
            "alert": st.column_config.CheckboxColumn(C["alert"])})
        st.download_button(T["b_dl"], scored.to_csv(index=False).encode(), "scored_checkpoints_p2.csv", "text/csv")
    st.markdown(f"<div class='foot'>{T['foot']} · {VERSION}</div>", unsafe_allow_html=True)
