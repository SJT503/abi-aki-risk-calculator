# -*- coding: utf-8 -*-
"""ABI-AKI Rolling Risk Calculator — Paper 2 (JinhuaNSICU-developed model)
v3: upload-first single page (no tabs), plotly charts (CJK native), cached scoring.

Run: streamlit run app.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
import riskcalc as rc

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples_local"
VERSION = "v2026.10.09 · JinhuaNSICU development"
BAND_COLOR = ["#2e7d32", "#b8860b", "#c62828"]

st.set_page_config(page_title="ABI-AKI Risk Calculator", page_icon="🧠", layout="wide")

# ---- CSS (kept from Paper-1 final, minus gauge/tri/pin which are removed) ----
st.markdown("""
<style>
  :root { --ink:#1a2b3c; --muted:#5b7083; --accent:#0f6e8c; --card:#ffffff; --bg:#f6f8fa; --navy:#0f2f4a; }
  .stApp { background: var(--bg); font-family:'Source Sans Pro','Source Sans 3','Segoe UI','Noto Sans SC',system-ui,sans-serif; color:var(--ink); }
  .block-container { padding-top:3.5rem; max-width:1380px; overflow:visible; }
  section[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e3e9ef; }
  .banner { background:#fdecea; border:1px solid #f1b8b4; border-left:5px solid #c62828; color:#8e1b1b;
            border-radius:10px; padding:9px 14px; margin-bottom:10px; font-size:.86rem; }
  .hdr { display:flex; align-items:center; gap:14px; padding:16px 22px; margin-bottom:14px;
         background:linear-gradient(90deg,#0f2f4a 0%,#0f6e8c 100%); border-radius:12px; color:#fff; }
  .hdr .badge { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35);
                padding:3px 10px; border-radius:999px; font-size:.78rem; white-space:nowrap; }
  .card { background:var(--card); border:1px solid #e3e9ef; border-radius:12px; padding:16px 18px; margin-bottom:10px; }
  .risk-num { font-size:2.6rem; font-weight:700; line-height:1; margin-top:4px; }
  .lbl { font-size:.78rem; text-transform:uppercase; letter-spacing:.8px; color:var(--muted); }
  .sec { font-size:.95rem; font-weight:700; color:var(--navy); margin:2px 0 6px; }
  .kpi { padding:10px 12px; background:#f2f6f8; border-radius:10px; flex:1; }
  .kpi .v { font-size:1.15rem; font-weight:700; color:var(--ink); }
  .evi { font-size:.85rem; color:var(--muted); line-height:1.55; }
  .evi b { color:var(--ink); }
  .info { background:#eef5f9; border:1px solid #c9dfeb; border-left:5px solid #0f6e8c; color:#18405a;
          border-radius:10px; padding:10px 14px; margin-bottom:10px; font-size:.86rem;
          line-height:1.5; overflow:visible; white-space:normal; }
  .keyres { background:#fff8e1; border:1px solid #f3d27a; border-left:5px solid #e0a100; border-radius:10px;
            padding:10px 14px; margin-top:8px; }
  .keytag { display:inline-block; background:#e0a100; color:#fff; font-weight:700; font-size:.72rem;
            letter-spacing:.8px; padding:2px 9px; border-radius:999px; margin-right:8px; vertical-align:middle; }
  .keyres .v { font-size:1.9rem; font-weight:800; color:var(--ink); line-height:1.15; }
  .sevnum { font-size:1.8rem; font-weight:700; color:var(--ink); line-height:1.1; }
  .foot { font-size:.78rem; color:var(--muted); margin-top:8px; }
  .upzone { border:2px dashed #b0bec5; border-radius:14px; padding:20px 24px; text-align:center;
            background:#fafcfd; margin:8px 0 14px; }
  .upzone .big { font-size:1.1rem; font-weight:600; color:var(--navy); margin-bottom:6px; }
  .upzone .sub { font-size:.82rem; color:var(--muted); }
  .exbtn { display:inline-block; margin:6px 8px 0 0; }
</style>
""", unsafe_allow_html=True)

# ---- i18n ----
S = {
"en": dict(
    banner="<b>Externally validated on MIMIC-IV (7,442 stays) and eICU-CRD (9,071 stays, 171 hospitals).</b> "
           "Designed for clinical decision support; run silent-mode evaluation, confirm thresholds and recalibrate "
           "locally before clinical use.",
    subtitle="Rolling 48-h incident AKI prediction at 6-h checkpoints · any AKI (≥Stage 1) + severe AKI (≥Stage 3)",
    up_title="Upload Patient Data",
    up_hint="Drag & drop a CSV file, or click to browse. Columns: <code>stay_id</code>, <code>t_hr</code> + 166 model features "
            "(blank = missing, handled natively). Download the template in the sidebar.",
    up_btn="Choose CSV file",
    or_lbl="— or load an example —",
    ex=["Example A · stable, low risk", "Example B · rising any-AKI risk",
        "Example C · severe-AKI high risk"],
    results="Results",
    kpi_lbl=["Checkpoints", "Peak any-AKI (calibrated)", "First ≥0.15 (working)", "Peak severe-AKI (raw)"],
    never="not reached",
    chart_any="Any-AKI · calibrated probability",
    chart_sev="Severe-AKI · raw probability (log scale)",
    x_label="ICU hour (checkpoint)",
    y_any="Calibrated probability",
    y_sev="Raw probability (log scale)",
    working="0.15 working threshold",
    sev_ref="median-input reference",
    table_lbl="Scored checkpoints",
    dl_btn="⬇ Download scored checkpoints (all stays)",
    stay_lbl="Stay",
    evi_lbl="Model evidence",
    key=("KEY RESULT", "Severe AKI (≥Stage 3) · external AUROC", "0.922 · 0.878",
         "MIMIC-IV · eICU-CRD · 95% CI 0.882–0.954 · 0.791–0.953 · 87 / 83 event stays",
         "Severe AKI, the endpoint that calls for nephrology review and RRT readiness, "
         "was the best-discriminated endpoint in both external cohorts."),
    kpi=[("Internal · any AKI", "0.815", "JinhuaNSICU test · 208 stays", "95% CI 0.754–0.872"),
         ("External · any AKI", "0.769 · 0.777", "MIMIC-IV · eICU-CRD", "95% CI 0.748–0.791 · 0.759–0.795"),
         ("Calibration slope", "0.909 · 0.839 · 0.881", "int · MIMIC-IV · eICU-CRD", "recalibrated (raw 0.244 · 0.225 · 0.236)")],
    evi_body=("<b>Model</b>: LightGBM, class-weighted; 166 of 280 candidate features (static · rolling · LOCF · trends · medications)<br>"
              "<b>Endpoint</b>: incident AKI within 48 h by KDIGO creatinine — any AKI ≥Stage 1 (primary) · severe AKI ≥Stage 3 (secondary)<br>"
              "<b>Cohorts</b>: n = 1,033 / 7,442 / 9,071 — JinhuaNSICU dev / MIMIC-IV ext / eICU-CRD ext; no refitting externally<br>"
              "<b>Explainability</b>: SHAP — creatinine ratio leads (1.476, ≈2× age 0.776)<br>"
              "<b>Inputs excluded by design</b>: no urine output, no GCS"),
    cols={"t_hr":"ICU hour","any_p_cal":"Any-AKI calibrated","band_lbl":"Band","severe_p_raw":"Severe-AKI raw","alert":"≥0.15"},
    foot="Computed locally; uploaded files are not stored or transmitted.",
    bands=["LOW","INTERMEDIATE","HIGH"],
    sev_note="ranking score — no calibrated probability scale",
    side_cal="Frozen recalibration layer",
    side_cal_src="any-AKI only · read from model/calibration.json",
    side_tpl="Batch template",
    side_tpl_btn="⬇ Download 166-column CSV template",
    side_tpl_cap="stay_id, t_hr + 166 model features (one median row)",
),
"zh": dict(
    banner="<b>已在 MIMIC-IV（7,442 住院次）和 eICU-CRD（9,071 住院次，171 家医院）完成外部验证。</b>"
           "面向临床决策支持设计；临床使用前须在本地进行静默运行评估、确认阈值并重新校准。",
    subtitle="每 6 小时检查点滚动预测未来 48 h 新发 AKI · any AKI（≥1 期）+ severe AKI（≥3 期）",
    up_title="上传患者数据",
    up_hint="拖拽或点击上传 CSV 文件。列：<code>stay_id</code>、<code>t_hr</code> + 166 个模型特征（空白 = 缺失，模型原生处理）。模板可在侧栏下载。",
    up_btn="选择 CSV 文件",
    or_lbl="— 或加载示例 —",
    ex=["示例 A · 平稳低风险", "示例 B · any-AKI 风险缓慢上升", "示例 C · severe-AKI 高风险"],
    results="结果",
    kpi_lbl=["检查点数", "any-AKI 校准峰值", "首次 ≥0.15（工作阈值）", "severe-AKI raw 峰值"],
    never="未达到",
    chart_any="any-AKI · 校准概率",
    chart_sev="severe-AKI · raw 概率（对数刻度）",
    x_label="ICU 小时（检查点）",
    y_any="校准概率",
    y_sev="raw 概率（对数刻度）",
    working="0.15 工作阈值",
    sev_ref="中位输入参考",
    table_lbl="检查点打分表",
    dl_btn="⬇ 下载检查点打分（全部住院次）",
    stay_lbl="住院次",
    evi_lbl="模型证据",
    key=("关键结果", "Severe AKI（≥3 期）· 外部 AUROC", "0.922 · 0.878",
         "MIMIC-IV · eICU-CRD · 95% CI 0.882–0.954 · 0.791–0.953 · 事件住院次 87 / 83",
         "Severe AKI（需肾内科会诊和 RRT 准备的终点）在两个外部队列中均为区分度最高的终点。"),
    kpi=[("内部验证 · any AKI", "0.815", "JinhuaNSICU 测试集 · 208 住院次", "95% CI 0.754–0.872"),
         ("外部验证 · any AKI", "0.769 · 0.777", "MIMIC-IV · eICU-CRD", "95% CI 0.748–0.791 · 0.759–0.795"),
         ("校准斜率", "0.909 · 0.839 · 0.881", "内部 · MIMIC-IV · eICU-CRD", "重校准后（原始 0.244 · 0.225 · 0.236）")],
    evi_body=("<b>模型</b>：LightGBM（类加权），280 个候选特征中选定 166 个（静态 · 滚动窗 · LOCF · 趋势 · 用药）<br>"
              "<b>终点</b>：未来 48 h 新发 AKI（KDIGO 肌酐标准）——any AKI ≥1 期（主要）· severe AKI ≥3 期（次要）<br>"
              "<b>队列</b>：n = 1,033 / 7,442 / 9,071——JinhuaNSICU 开发 / MIMIC-IV 外部 / eICU-CRD 外部；外部不重拟合<br>"
              "<b>可解释性</b>：SHAP——当前/基线肌酐比居首（1.476，约为年龄 0.776 的 2 倍）<br>"
              "<b>设计上排除的输入</b>：无尿量，无 GCS"),
    cols={"t_hr":"ICU 小时","any_p_cal":"any-AKI 校准","band_lbl":"风险带","severe_p_raw":"severe-AKI raw","alert":"≥0.15"},
    foot="本地计算；上传文件不存储、不传输。",
    bands=["低风险","中风险","高风险"],
    sev_note="排名分——无校准概率刻度",
    side_cal="冻结重校准层",
    side_cal_src="仅 any-AKI · 运行时读取 model/calibration.json",
    side_tpl="批量模板",
    side_tpl_btn="⬇ 下载 166 列 CSV 模板",
    side_tpl_cap="stay_id、t_hr + 166 个模型特征（一行中位数）",
),
}

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
def fmt_p(p):
    """Severe-AKI raw-score display: >=1% in the same percent style as any-AKI, <1% in scientific percent."""
    pct = p * 100
    if pct >= 1: return f"{pct:.1f}%"
    m, e = f"{pct:.1e}".split("e")
    return f"{m} × 10{str(int(e)).translate(SUP)}%"

def sev_axis_ticks():
    """Decade ticks (fraction 1e-8..1) as percent labels: >=1% integer percent like the
    any-AKI axis, <1% scientific percent (10^-6% ... 10^-1%)."""
    vals, txt = [], []
    for e in range(-8, 1):
        vals.append(10.0 ** e)
        k = e + 2                       # fraction 10^e = 10^(e+2) percent
        txt.append(f"{10.0 ** k:.0f}%" if k >= 0 else f"10{str(k).translate(SUP)}%")
    return vals, txt

@st.cache_resource
def load():
    return rc.load_models()
M = load()
SEV_REF = rc.severe_reference(M)

# ---- sidebar ----
with st.sidebar:
    LANG = st.radio("Language / 语言", ["English", "中文"], index=0, label_visibility="collapsed", key="lang")
    T = S["en" if LANG == "English" else "zh"]
    st.divider()
    st.markdown(f"**{T['side_cal']}**")
    st.markdown(f"<div class='evi' style='font-size:.82rem'>p_cal = sigmoid(a + b · logit p)<br>"
                f"a = <b>{M['a']:.6f}</b> · b = <b>{M['b']:.6f}</b><br>{T['side_cal_src']}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"**{T['side_tpl']}**")
    st.caption(T["side_tpl_cap"])
    st.download_button(T["side_tpl_btn"], rc.template_frame(M).to_csv(index=False).encode(),
                       "feature_template.csv", "text/csv")
    st.divider()
    st.caption(VERSION)

# ---- header ----
st.markdown(f'<div class="info">ℹ️ {T["banner"]}</div>', unsafe_allow_html=True)
st.markdown("""
<div class="hdr">
  <div style="font-size:1.9rem">🧠</div>
  <div>
    <div style="font-size:1.28rem;font-weight:700">Acute Brain Injury · Rolling AKI Risk</div>
    <div style="font-size:.82rem;opacity:.9">{}</div>
  </div>
  <div class="badge" style="margin-left:auto">{}</div>
</div>
""".format(T["subtitle"], VERSION), unsafe_allow_html=True)

# ---- data input (upload + examples, same card) ----
@st.cache_data
def score_csv_bytes(raw: bytes):
    import io as _io
    df = pd.read_csv(_io.BytesIO(raw))
    s, miss = rc.score_frame(df, M)
    return df, s, miss

@st.cache_data
def score_example(path_str: str):
    df = pd.read_csv(path_str)
    s, miss = rc.score_frame(df, M)
    return df, s, miss

st.markdown(f"### {T['up_title']}")

up = st.file_uploader(T["up_btn"], type=["csv"], key="csv_up",
                      label_visibility="collapsed",
                      help=None)

# example buttons (always visible, not behind a tab)
c_ex1, c_ex2, c_ex3 = st.columns(3)
ex_clicked = None
with c_ex1:
    if st.button(T["ex"][0], key="ex0", use_container_width=True): ex_clicked = 0
with c_ex2:
    if st.button(T["ex"][1], key="ex1", use_container_width=True): ex_clicked = 1
with c_ex3:
    if st.button(T["ex"][2], key="ex2", use_container_width=True): ex_clicked = 2

st.caption(T["up_hint"])

# ---- score ----
df = scored = miss = None
if up is not None:
    raw = up.getvalue()
    df, scored, miss = score_csv_bytes(raw)
elif ex_clicked is not None:
    ex_names = ["demo_low_rise.csv", "demo_intermediate.csv", "demo_rapid_rise.csv"]
    ex_path = str(EXAMPLES / ex_names[ex_clicked])
    df, scored, miss = score_example(ex_path)
else:
    # always show a default example so results never disappear
    df, scored, miss = score_example(str(EXAMPLES / "demo_low_rise.csv"))

if miss:
    st.warning(f"{len(miss)} of 166 feature columns absent → treated as missing (NaN): {', '.join(miss[:6])}")

# ---- results (same page, below input) ----
if scored is not None:
    st.markdown(f"### {T['results']}")
    stays = list(pd.unique(scored["stay_id"]))
    if len(stays) > 1:
        sid = st.selectbox(T["stay_lbl"], stays, key="stay_pick")
    else:
        sid = stays[0]
        st.markdown(f"**{T['stay_lbl']}:** {sid}")

    r = scored[scored["stay_id"] == sid].copy().sort_values("t_hr")
    if "t_hr" not in r.columns:
        r["t_hr"] = range(24, 24 + 6 * len(r), 6)
    x = r["t_hr"].to_numpy()
    pk_val = float(r["any_p_cal"].max())
    pk_band = rc.band(pk_val)
    first_alert = r.loc[r["any_p_cal"] >= rc.ANY_WORKING, "t_hr"]
    kv = [str(len(r)), f"{pk_val*100:.1f}%",
          (f"{int(first_alert.iloc[0])} h" if len(first_alert) else T["never"]),
          fmt_p(float(r["severe_p_raw"].max()))]

    # KPI row
    c_k = st.columns(4)
    for j in range(4):
        with c_k[j]:
            color = BAND_COLOR[pk_band] if j == 1 else "#1a2b3c"
            st.markdown(f'<div class="kpi"><div class="lbl">{T["kpi_lbl"][j]}</div>'
                        f'<div class="v" style="color:{color};font-size:1.3rem">{kv[j]}</div></div>',
                        unsafe_allow_html=True)

    # plotly charts (side by side)
    cA, cB = st.columns(2)
    with cA:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x, y=r["any_p_cal"], mode="lines+markers",
                                   line=dict(color="#0f6e8c", width=2.5), marker=dict(size=6),
                                   name=T["chart_any"], hovertemplate="h %{x}: %{y:.1%}<extra></extra>"))
        fig1.add_hline(y=0.15, line_dash="dash", line_color="#c62828", line_width=2,
                       annotation_text=T["working"], annotation_position="top right")
        fig1.add_hline(y=0.10, line_dash="dot", line_color="#8a9aa8", line_width=1)
        fig1.add_hline(y=0.20, line_dash="dot", line_color="#8a9aa8", line_width=1)
        ymax = max(0.30, pk_val * 1.2)
        fig1.update_layout(title=T["chart_any"], xaxis_title=T["x_label"], yaxis_title=T["y_any"],
                           yaxis_range=[0, ymax], yaxis_tickformat=".0%",
                           height=320, margin=dict(l=50, r=20, t=45, b=40),
                           plot_bgcolor="white", paper_bgcolor="white",
                           font=dict(family="Source Sans Pro, Noto Sans SC, sans-serif", size=12))
        fig1.update_xaxes(gridcolor="#e8edf2")
        fig1.update_yaxes(gridcolor="#e8edf2")
        st.plotly_chart(fig1, use_container_width=True, key="chart_any_fig")

    with cB:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=r["severe_p_raw"], mode="lines+markers",
                                   line=dict(color="#5b7083", width=2.5), marker=dict(size=6),
                                   name=T["chart_sev"], customdata=[fmt_p(v) for v in r["severe_p_raw"]],
                                   hovertemplate="h %{x}: %{customdata}<extra></extra>"))
        fig2.add_hline(y=SEV_REF, line_dash="dash", line_color="#c62828", line_width=1.5,
                       annotation_text=T["sev_ref"], annotation_position="top right")
        fig2.update_layout(title=T["chart_sev"], xaxis_title=T["x_label"], yaxis_title=T["y_sev"],
                           yaxis_type="log", yaxis_range=[-8, 0],
                           height=320, margin=dict(l=60, r=20, t=45, b=40),
                           plot_bgcolor="white", paper_bgcolor="white",
                           font=dict(family="Source Sans Pro, Noto Sans SC, sans-serif", size=12))
        sv_vals, sv_txt = sev_axis_ticks()
        fig2.update_yaxes(tickvals=sv_vals, ticktext=sv_txt)
        fig2.update_xaxes(gridcolor="#e8edf2")
        fig2.update_yaxes(gridcolor="#e8edf2")
        st.plotly_chart(fig2, use_container_width=True, key="chart_sev_fig")

    # KEY RESULT badge + severe raw
    st.markdown(f"""
    <div class="keyres">
      <span class="keytag">{T['key'][0]}</span><span class="lbl">{T['key'][1]}</span>
      <div class="v">{T['key'][2]}</div>
      <div class="evi">{T['key'][3]}</div>
      <div class="evi" style="margin-top:4px"><b>{T['key'][4]}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # table
    show = r.copy()
    show["band_lbl"] = [T["bands"][b] for b in show["band"]]
    show["alert"] = show["any_p_cal"] >= rc.ANY_WORKING
    show["any_p_cal"] = (show["any_p_cal"] * 100).round(1)
    show["severe_p_raw"] = [fmt_p(v) for v in show["severe_p_raw"]]
    C = T["cols"]
    st.markdown(f"**{T['table_lbl']}**")
    st.dataframe(show[["t_hr", "any_p_cal", "band_lbl", "severe_p_raw", "alert"]],
                 hide_index=True, use_container_width=True, column_config={
        "t_hr": st.column_config.NumberColumn(C["t_hr"], format="%d"),
        "any_p_cal": st.column_config.NumberColumn(C["any_p_cal"], format="%.1f%%"),
        "severe_p_raw": st.column_config.TextColumn(C["severe_p_raw"]),
        "alert": st.column_config.CheckboxColumn(C["alert"]),
    })
    st.download_button(T["dl_btn"], scored.to_csv(index=False).encode(),
                       "scored_checkpoints.csv", "text/csv")

# ---- model evidence (collapsible, bottom) ----
with st.expander(f"📋 {T['evi_lbl']}", expanded=False):
    k1, k2, k3 = T["kpi"]
    kp = lambda x: (f'<div class="kpi" style="flex:1"><div class="lbl">{x[0]}</div><div class="v">{x[1]}</div>'
                    f'<div class="evi">{x[2]}<br>{x[3]}</div></div>')
    st.markdown(f'<div style="display:flex;gap:10px;margin:8px 0">{kp(k1)}{kp(k2)}{kp(k3)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="evi">{T["evi_body"]}</div>', unsafe_allow_html=True)

st.markdown(f'<div class="foot">{T["foot"]} · {VERSION}</div>', unsafe_allow_html=True)
