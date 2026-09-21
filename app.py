# -*- coding: utf-8 -*-
"""ABI-AKI Rolling Risk Calculator — bilingual (EN default / 中文), journal-style UI
Model: LightGBM 291-feature frozen (n6, D33) + select-fit recalibration layer (g2)
Validation: M4 internal 0.7369 / eICU external 0.7094 (zero-touch, 180 hospitals)
Run: streamlit run app.py
⚠️ Research use only — not for clinical decision-making
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
# 云部署包结构优先（./model/），本地开发回退（../results/）
MODEL_DIR = os.path.join(BASE, "model") if os.path.exists(os.path.join(BASE, "model")) else os.path.join(BASE, "..", "results")

st.set_page_config(page_title="ABI-AKI Risk Calculator", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  :root { --ink:#1a2b3c; --muted:#5b7083; --accent:#0f6e8c; --card:#ffffff; --bg:#f6f8fa; }
  .stApp { background: var(--bg); font-family: 'Source Sans Pro','Segoe UI',system-ui,sans-serif; color: var(--ink); }
  section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e3e9ef; }
  section[data-testid="stSidebar"] * { font-size:.92rem; }
  .hdr { display:flex; align-items:center; gap:14px; padding:18px 22px; margin-bottom:14px;
         background:linear-gradient(90deg,#0f2f4a 0%,#0f6e8c 100%); border-radius:12px; color:#fff; }
  .hdr .badge { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35);
                padding:3px 10px; border-radius:999px; font-size:.78rem; letter-spacing:.4px; }
  .card { background:var(--card); border:1px solid #e3e9ef; border-radius:12px; padding:18px 20px; }
  .risk-num { font-size:3.1rem; font-weight:700; line-height:1; }
  .lbl { font-size:.78rem; text-transform:uppercase; letter-spacing:.8px; color:var(--muted); }
  .gauge { position:relative; height:14px; border-radius:7px; margin:10px 0 4px;
           background:linear-gradient(90deg,#2e7d32 0%,#2e7d32 30%,#f9a825 45%,#f9a825 60%,#c62828 78%,#c62828 100%); }
  .gauge .pin { position:absolute; top:-5px; width:4px; height:24px; background:#1a2b3c; border-radius:2px;
                box-shadow:0 0 0 2px #fff; transform:translateX(-50%); }
  .gauge-ticks { display:flex; justify-content:space-between; font-size:.7rem; color:var(--muted); }
  .kpi { padding:10px 6px; background:#f2f6f8; border-radius:10px; }
  .kpi .v { font-size:1.15rem; font-weight:700; }
  .evi { font-size:.85rem; color:var(--muted); line-height:1.55; }
  .evi b { color:var(--ink); }
  .warn { background:#fff8e6; border:1px solid #eadfa8; border-radius:10px; padding:10px 14px; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

STRINGS = {
    "en": dict(
        lang_label="Language",
        sidebar_title="🧾 Patient features",
        sidebar_caption="Base = training-set typical patient (median profile of 291 features); adjust key items",
        age="Age (yr)", charlson="Charlson index", cr_base="Baseline creatinine mg/dL (min of first 24 h)",
        cr_last="Current creatinine mg/dL", uo="24-h urine output (mL/kg/h)",
        net="Cumulative net balance (mL/kg; neg = net output)", uo_n="UO chart entries in 24 h",
        t_hr="Checkpoint (ICU hour)", sbp="Systolic BP mmHg", hr="Heart rate bpm", bun="Current BUN mg/dL",
        norepi="On/prior norepinephrine",
        risk_lbl="Calibrated 48-h AKI risk (deployment)",
        ticks=["0%", "10% low/int", "20% int/high", "50%+"],
        raw_cal="raw {a}% → select-fit calibrated {b}%",
        action_lbl="Suggested action (demo)",
        action=["Routine monitoring; renal function per unit protocol",
                "Suggested review: assess volume status and nephrotoxic exposure; intensify Cr/UO monitoring",
                "Suggested clinician review: volume status, nephrotoxic agents; nephrology consult if indicated"],
        bands=["LOW", "INTERMEDIATE", "HIGH"],
        workpoint="Band thresholds 0.10/0.20 match deployment working point; at thr=0.20 (eICU): median lead time 12.6 h · NNE 2.2 · false alarms 1.9/100 patient-days",
        evi_lbl="Model evidence",
        kpi=[("Internal", "M4 temporal 20-22", "95%CI 0.727-0.747"),
             ("External", "eICU 180 hospitals", "95%CI 0.703-0.716"),
             ("Calibrated ECE", "select-fit layer", "internal 0.013")],
        evi_body=("<b>Model</b>: LightGBM, 291 features (static / rolling windows / LOCF / 48-h trends / "
                  "variability SD·CV / medications / UO-fluids / ventilation), NaN-native<br>"
                  "<b>Endpoint</b>: incident AKI within 48 h, full KDIGO (Cr∪UO), incident-only<br>"
                  "<b>Cohort</b>: M4 acute brain injury 10,723 ICU stays (TBI/stroke/SAH/ICH/anoxic/encephalitis); "
                  "development 2008-2019, internal validation 2020-2022<br>"
                  "<b>Explainability</b>: SHAP top = net fluid balance · 24-h urine output · baseline creatinine (fluid-kidney axis)"),
        methods="Methodology & limitations",
        methods_items=["Single-point mode keeps UO-family raw-value features at the median = demonstrative approximation; use batch mode for research",
                       "Cross-DB sampling granularity of variability features (M4 q1h vs eICU q5min) disclosed",
                       "Riley patient-level sample size not met (checkpoint EPPP 23.4 met); SMOTE sensitivity Δ+0.010",
                       "SHAP interactions not performed (infeasible at 291 features); age ≥80 subgroup AUROC 0.67 (internal)"],
        warn=("⚠️ <b>Disclaimer</b>: developed on retrospective public databases (MIMIC-IV v3.1 / eICU-CRD v2.0). "
              "For research and external-validation reproduction only; not validated prospectively; "
              "not for clinical decision-making. Batch mode is processed locally and uploads nothing."),
        tab1="① Single-point calculator", tab2="② Batch prediction (pipeline CSV)",
        batch_desc="Upload a pipeline feature CSV (any feature-column subset) and download per-checkpoint calibrated risk. **Primary mode for research/reproduction.**",
        upload="Feature CSV", download="⬇ Download predictions", warn2="🧭 Decision-support research tool · supports, not replaces, clinical judgment.",
        disclaimer='🧭 <b>A decision-support research tool — supports, not replaces, clinical judgment · prospective validation pending.</b>\n<details style="margin-top:6px"><summary style="cursor:pointer;color:#0f6e8c">Intended use, boundaries &amp; pathway</summary>\n<div style="font-size:.82rem;color:#5b7083;line-height:1.6;padding-top:6px">\n<b>Intended use</b> — This tool translates a validated machine-learning model into a usable interface for researchers and clinicians. It is designed to <b>support — not replace — clinical judgment</b>: to help research and quality-improvement teams identify acute-brain-injury ICU patients who may benefit from closer renal monitoring, to provide calibrated 48-h AKI risk estimates for cohort enrichment and trial screening, and to let other groups reproduce and locally validate the model on their own data.<br>\n<b>Boundaries</b> — The model was developed and validated retrospectively (MIMIC-IV v3.1; external eICU-CRD v2.0, 180 hospitals) and has not yet undergone prospective or point-of-care evaluation. It is therefore distributed as a research and education tool; it has not been cleared or approved as a medical device by any regulatory authority (FDA/CE/NMPA) and should not be used as the sole basis for diagnosis, triage, or treatment decisions.<br>\n<b>Pathway to clinical use</b> — Implementation is the intended next step. The model\'s alerting profile (median 12.6-h lead time, NNE 2.2, 1.9 false alarms per 100 patient-days at the 0.20 working point) and positive net benefit across decision thresholds support its evaluation in silent-mode prospective deployments and stepped-wedge trials — the design used by recent AKI early-warning systems. The frozen pipeline is provided openly to enable such local, zero-touch evaluation.<br>\n<b>Privacy</b> — computation runs within your browser session; no entered data are stored or transmitted to the authors.<br>\n<b>Data &amp; code</b> — model artifacts are derived from MIMIC-IV/eICU-CRD under the PhysioNet Data Use Agreement (source data not included); code and frozen model are open at github.com/SJT503/abi-aki-risk-calculator.\n</div></details>'
    ),
    "zh": dict(
        lang_label="语言",
        sidebar_title="🧾 患者特征",
        sidebar_caption="底座=训练集典型患者（291 特征中位画像），调整核心项",
        age="年龄（岁）", charlson="Charlson 指数", cr_base="基线肌酐 mg/dL（首 24h 最小）",
        cr_last="当前肌酐 mg/dL", uo="24h 尿量（mL/kg/h）",
        net="累计净平衡（mL/kg，负=净排出）", uo_n="24h 尿量记录次数",
        t_hr="检查点（ICU 小时）", sbp="收缩压 mmHg", hr="心率 bpm", bun="当前 BUN mg/dL",
        norepi="正在/曾用去甲肾上腺素",
        risk_lbl="校准后 48h AKI 风险（部署口径）",
        ticks=["0%", "10% 低/中界", "20% 中/高界", "50%+"],
        raw_cal="原始概率 {a}% → select-fit 校准 {b}%",
        action_lbl="建议行动（演示）",
        action=["常规监测；按科室流程复查肾功能",
                "建议复核：评估容量状态与肾毒性暴露，加密 Cr/UO 监测",
                "建议临床团队复核：评估容量状态、肾毒性药物，必要时肾脏科会诊"],
        bands=["低风险", "中风险", "高风险"],
        workpoint="分带阈值 0.10/0.20 与部署工作点一致；thr=0.20 实测（eICU）：中位提前预警 12.6h · NNE 2.2 · 假警报 1.9/100 病人日",
        evi_lbl="模型证据",
        kpi=[("内部验证", "M4 时间分割 20-22", "95%CI 0.727-0.747"),
             ("外部验证", "eICU 180 医院", "95%CI 0.703-0.716"),
             ("校准后 ECE", "select-fit 层", "内验 0.013")],
        evi_body=("<b>模型</b>：LightGBM，291 特征（静态/动态窗/LOCF/48h 趋势/变异度 SD·CV/用药/UO-液体/通气），NaN 原生<br>"
                  "<b>终点</b>：未来 48h incident AKI，完整 KDIGO（Cr∪UO），incident-only<br>"
                  "<b>队列</b>：M4 急性脑损伤 10,723 stays（TBI/卒中/SAH/ICH/缺氧/脑炎）；开发 2008-2019，内验 2020-2022<br>"
                  "<b>可解释性</b>：SHAP top = 净液体平衡 · 24h 尿量 · 基线肌酐（液体-肾脏轴）"),
        methods="方法学与局限",
        methods_items=["单点模式尿量同族原始值保持中位=演示性近似；研究请用批量模式",
                       "变异度特征跨库采样粒度差（M4 q1h vs eICU q5min）已披露",
                       "Riley 患者级严格样本量未达标（检查点 EPPP 23.4 达标）；SMOTE 敏感性 Δ+0.010",
                       "SHAP 交互未执行（291 特征计算不可行）；≥80 岁亚组 AUROC 0.67（内验）"],
        warn=("⚠️ <b>免责声明</b>：本工具基于回顾性公开数据库（MIMIC-IV v3.1 / eICU-CRD v2.0）开发，"
              "仅供研究与外部验证复现使用，未经前瞻性验证，不得用于临床决策。批量模式本地处理，不上传任何服务器。"),
        tab1="① 单点计算器", tab2="② 批量预测（管线 CSV）",
        batch_desc="上传管线输出的特征 CSV（任意特征列子集），下载逐检查点校准风险。**研究/复现主模式。**",
        upload="特征 CSV", download="⬇ 下载预测结果", warn2="🧭 临床决策支持研究工具 · 辅助而非替代临床判断。",
        disclaimer='🧭 <b>临床决策支持研究工具——辅助而非替代临床判断 · 前瞻验证待开展。</b>\n<details style="margin-top:6px"><summary style="cursor:pointer;color:#0f6e8c">用途、边界与临床化路径</summary>\n<div style="font-size:.82rem;color:#5b7083;line-height:1.6;padding-top:6px">\n<b>预期用途</b>——本工具把经过验证的机器学习模型转化为研究者和临床医生可用的界面。其定位是<b>辅助而非替代临床判断</b>：帮助研究与质量改进团队识别可能受益于强化肾脏监测的急性脑损伤 ICU 患者；为队列富集和临床试验筛查提供校准的 48h AKI 风险估计；支持其他中心在自有数据上复现与本地验证本模型。<br>\n<b>边界</b>——模型基于回顾性数据开发与验证（MIMIC-IV v3.1 开发；eICU-CRD v2.0 外部验证，180 家医院），尚未经过前瞻性或床旁评估。因此以研究与教育工具形式发布；未经任何监管机构（FDA/CE/NMPA）作为医疗器械注册或批准，不应作为诊断、分诊或治疗决策的唯一依据。<br>\n<b>临床化路径</b>——临床落地是明确的下一步。模型的预警特性（0.20 工作点：中位提前 12.6 小时，NNE 2.2，每 100 病人日 1.9 次假警报）与全阈值范围的正净效益，支持其进入静默模式前瞻部署与 stepped-wedge 随机试验——这正是近年 AKI 预警系统采用的评估设计。冻结管线完全开放，以支持各中心开展本地零触摸评估。<br>\n<b>隐私</b>——全部计算在浏览器会话内完成；不存储、不向作者传输任何输入数据。<br>\n<b>数据与代码</b>——模型工件在 PhysioNet 数据使用协议下由 MIMIC-IV/eICU-CRD 衍生（不含源数据）；代码与冻结模型开源于 github.com/SJT503/abi-aki-risk-calculator。\n</div></details>'
    ),
}
LANG = st.sidebar.radio("Language / 语言", ["English", "中文"], index=0, label_visibility="collapsed")
T = STRINGS["en" if LANG == "English" else "zh"]

st.markdown(f"""
<div class="hdr">
  <div style="font-size:1.9rem">🧠</div>
  <div>
    <div style="font-size:1.28rem;font-weight:700">Acute Brain Injury · Rolling AKI Risk</div>
    <div style="font-size:.82rem;opacity:.9">{'急性脑损伤 ICU 滚动 AKI 风险预测 · 6h 检查点 × 未来 48h（完整 KDIGO Cr∪UO）' if LANG=='中文' else 'Rolling 48-h incident AKI prediction at 6-h checkpoints · full KDIGO (Cr∪UO)'}</div>
  </div>
  <div class="badge" style="margin-left:auto">v2026.09.22 · external eICU n=9,071</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chain():
    frozen = joblib.load(os.path.join(MODEL_DIR, "n6_abi_gbm_frozen.joblib"))
    layer = json.load(open(os.path.join(MODEL_DIR, "g2_recalibration_dca.json"), encoding="utf-8"))["recal_layer_frozen"]
    defaults = json.load(open(os.path.join(BASE, "defaults.json"), encoding="utf-8"))
    return frozen["feats"], frozen["model"], float(layer["a_intercept"]), float(layer["b_slope"]), defaults


FEATS, MODEL, A, B, DEFAULTS = load_chain()


def predict_risk(df: pd.DataFrame):
    X = df.reindex(columns=FEATS)
    p_raw = MODEL.predict_proba(X)[:, 1]
    z = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_raw, 1e-6, 1 - 1e-6)))
    p_cal = 1 / (1 + np.exp(-(A + B * z)))
    return p_raw, p_cal


def band(p):
    idx = 0 if p < 0.10 else (1 if p < 0.20 else 2)
    return T["bands"][idx], ["#2e7d32", "#b8860b", "#c62828"][idx], T["action"][idx]


tab1, tab2 = st.tabs([T["tab1"], T["tab2"]])

with tab1:
    with st.sidebar:
        st.markdown(f"### {T['sidebar_title']}")
        st.caption(T["sidebar_caption"])
        age = st.slider(T["age"], 18, 95, int(DEFAULTS["age"]))
        charlson = st.slider(T["charlson"], 0, 15, int(DEFAULTS["charlson"]))
        cr_base = st.number_input(T["cr_base"], 0.2, 8.0, float(DEFAULTS["cr_base"]), 0.1)
        cr_last = st.number_input(T["cr_last"], 0.2, 10.0, float(DEFAULTS["cr_last"]), 0.1)
        st.divider()
        uo_24 = st.slider(T["uo"], 0.0, 4.0, round(float(DEFAULTS["uo_ml_kg_h_24h"]), 2), 0.05)
        net_bal = st.slider(T["net"], -120.0, 460.0, round(float(DEFAULTS["net_balance_ml_kg"]), 1), 5.0)
        uo_n = st.slider(T["uo_n"], 0, 30, int(DEFAULTS["uo_n"]))
        t_hr = st.slider(T["t_hr"], 24, 168, int(DEFAULTS["t_hr"]), 6)
        st.divider()
        sbp_last = st.slider(T["sbp"], 60, 220, int(DEFAULTS["sbp_last"]))
        hr_last = st.slider(T["hr"], 30, 180, int(DEFAULTS["hr_rate_last"]))
        bun_last = st.number_input(T["bun"], 2.0, 150.0, float(DEFAULTS["bun_last"]), 1.0)
        norepi_on = st.checkbox(T["norepi"], value=bool(DEFAULTS["norepi_on"]))

    profile = dict(DEFAULTS)
    profile.update({"age": age, "charlson": charlson, "cr_base": cr_base, "cr_last": cr_last,
                    "uo_ml_kg_h_24h": uo_24, "net_balance_ml_kg": net_bal, "uo_n": uo_n,
                    "t_hr": t_hr, "sbp_last": sbp_last, "hr_rate_last": hr_last,
                    "bun_last": bun_last, "norepi_on": int(norepi_on)})
    p_raw, p_cal = predict_risk(pd.DataFrame([profile]))
    p = float(p_cal[0])
    nm, color, action = band(p)

    cL, cR = st.columns([1.15, 1])
    with cL:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="lbl">{T["risk_lbl"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-num" style="color:{color}">{p*100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="gauge"><div class="pin" style="left:{min(max(p,0),1)*100:.1f}%"></div></div>
        <div class="gauge-ticks"><span>{T['ticks'][0]}</span><span>{T['ticks'][1]}</span><span>{T['ticks'][2]}</span><span>{T['ticks'][3]}</span></div>
        <div style="margin-top:10px"><span style="color:{color};font-weight:700;font-size:1.05rem">{nm}</span>
        &nbsp;·&nbsp;<span style="color:#5b7083;font-size:.85rem">{T['raw_cal'].format(a=f"{p_raw[0]*100:.1f}", b=f"{p*100:.1f}")}</span></div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="margin-top:12px;border-left:4px solid {color}">
        <div class="lbl">{T['action_lbl']}</div>
        <div style="margin-top:6px">{action}</div>
        <div class="evi" style="margin-top:8px">{T['workpoint']}</div>
        </div>
        """, unsafe_allow_html=True)
    with cR:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="lbl">{T["evi_lbl"]}</div>', unsafe_allow_html=True)
        k1, k2, k3 = T["kpi"]
        st.markdown(f"""
        <div style="display:flex;gap:10px;margin-top:8px">
          <div class="kpi" style="flex:1"><div class="lbl">{k1[0]}</div><div class="v">0.737</div><div class="evi">{k1[1]}<br>{k1[2]}</div></div>
          <div class="kpi" style="flex:1"><div class="lbl">{k2[0]}</div><div class="v">0.709</div><div class="evi">{k2[1]}<br>{k2[2]}</div></div>
          <div class="kpi" style="flex:1"><div class="lbl">{k3[0]}</div><div class="v">0.019</div><div class="evi">{k3[1]}<br>{k3[2]}</div></div>
        </div>
        <div class="evi" style="margin-top:12px">{T['evi_body']}</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander(f"📐 {T['methods']}"):
            st.markdown("\n".join(f"- {x}" for x in T["methods_items"]))
    st.markdown(f'<div class="warn">{T["disclaimer"]}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(T["batch_desc"])
    up = st.file_uploader(T["upload"], type=["csv"])
    if up is not None:
        d = pd.read_csv(up)
        p_raw, p_cal = predict_risk(d)
        d["p_raw"] = p_raw
        d["p_calibrated"] = p_cal
        st.dataframe(d.head(20), use_container_width=True)
        st.download_button(T["download"], d.to_csv(index=False).encode(), "predictions.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="warn">{T["warn2"]}</div>', unsafe_allow_html=True)
