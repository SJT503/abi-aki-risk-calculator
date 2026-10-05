"""Streamlit research demonstration of the rolling two-tier ABI-AKI model.

Run:  pip install -r requirements.txt && streamlit run app.py
All computation is local; uploaded files are not stored or transmitted.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import riskcalc

TIER1_REF, TIER2_REF = 0.15, 0.05  # reference lines only (development-side views; confirm locally)
ROOT = Path(__file__).resolve().parent
LOCAL_EX = ROOT / "examples_local"

st.set_page_config(page_title="ABI-AKI two-tier risk (research demo)", layout="wide")
st.error("**Research demonstration, not a clinical decision tool; confirm thresholds locally.** "
         "Outputs have not been prospectively validated and must not guide patient care.")
st.title("Rolling two-tier AKI risk after acute brain injury")
st.caption("Tier 1: any AKI (KDIGO creatinine ≥Stage 1). Tier 2: severe AKI (≥Stage 3). "
           "Each checkpoint row → frozen 268-feature LightGBM → frozen recalibration p_rec = logistic(a + b·logit(p)).")


@st.cache_resource
def _models():
    return riskcalc.load_models()


models = _models()
feats = models["tier1"]["features"]
with st.sidebar:
    st.subheader("Frozen recalibration layers")
    st.write(f"Tier 1: a = {models['tier1']['a']}, b = {models['tier1']['b']}")
    st.write(f"Tier 2: a = {models['tier2']['a']}, b = {models['tier2']['b']}")
    st.caption("Read at runtime from model/calibration_and_manifest.json.")
    st.subheader("Reference lines")
    st.write(f"Tier 1: {TIER1_REF}  ·  Tier 2: {TIER2_REF}")
    st.download_button("Download 268-column template", (ROOT / "examples/feature_template.csv").read_bytes(),
                       "feature_template.csv", "text/csv")

sources = {}
if LOCAL_EX.is_dir():
    for f in sorted(LOCAL_EX.glob("*.csv")):
        sources[f"Local example: {f.stem}"] = f
choice = st.radio("Input", ["Upload a checkpoint-feature CSV"] + list(sources), horizontal=True)
df = None
if choice.startswith("Upload"):
    up = st.file_uploader("One row per six-hour checkpoint; columns stay_id + the 268 model features "
                          "(see template; blanks = missing, handled natively).", type="csv")
    if up is not None:
        df = pd.read_csv(up)
    if not sources:
        st.info("No bundled patient examples: MIMIC-IV / eICU-CRD rows cannot be redistributed under the PhysioNet "
                "data use agreement. Credentialed users can create local examples with tools/make_local_examples.py.")
else:
    df = pd.read_csv(sources[choice])

if df is not None:
    miss = riskcalc.check_columns(df, feats)
    if miss:
        st.error(f"{len(miss)} of {len(feats)} required feature columns are missing (first: {', '.join(miss[:8])}).")
        st.stop()
    res = riskcalc.score(df, models)
    if "stay_id" not in res:
        res["stay_id"] = "uploaded"
    stays = list(pd.unique(res["stay_id"]))
    sid = st.selectbox("Stay", stays)
    r = res[res["stay_id"] == sid]
    if "t_hr" in r:
        r = r.sort_values("t_hr")
    x = r["t_hr"] if "t_hr" in r else list(range(len(r)))
    c1, c2 = st.columns(2)
    for col, tier, ref, lab in ((c1, "tier1", TIER1_REF, "Tier 1 (any AKI)"), (c2, "tier2", TIER2_REF, "Tier 2 (severe AKI)")):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, r[f"{tier}_p_rec"], marker="o", color="#0F6E8C")
        ax.axhline(ref, ls="--", color="#C62828", lw=1, label=f"reference {ref}")
        ax.set_xlabel("Hours since ICU admission (checkpoint)")
        ax.set_ylabel("Recalibrated probability")
        ax.set_title(lab)
        ax.set_ylim(0, max(0.3, float(r[f"{tier}_p_rec"].max()) * 1.1))
        ax.legend(frameon=False)
        col.pyplot(fig)
        plt.close(fig)
    show = r.rename(columns={"tier1_p_rec": "Tier-1 p_rec", "tier2_p_rec": "Tier-2 p_rec",
                             "tier1_p_raw": "Tier-1 p_raw", "tier2_p_raw": "Tier-2 p_raw"})
    show["Tier-1 ≥ ref"] = show["Tier-1 p_rec"] >= TIER1_REF
    show["Tier-2 ≥ ref"] = show["Tier-2 p_rec"] >= TIER2_REF
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download scored checkpoints", res.to_csv(index=False).encode(), "scored_checkpoints.csv", "text/csv")
