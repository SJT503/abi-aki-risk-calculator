# -*- coding: utf-8 -*-
"""s2: enrichment analyses from the five-paper style study (Tomašev/Cao/Yoon devices):
 1. False-positive autopsy (external, ge1 model @ working threshold 0.15):
    among alerted event-FREE stays, decompose into (a) early alerts - an event
    occurred later in the stay beyond the 48h window; (b) near-criteria - peak
    fold-change 1.3-1.5 or rise 0.2-0.3 (below staging threshold); (c) neither.
 2. Workload translation: % of stays alerted / % of event-free patient-days
    under review at each threshold.
 3. Progression-anchored capture (Tomašev dialysis-analogue): among ge1 event
    stays, capture rate of the ge1 model for those that later progressed to
    >=Stage 2 / >=Stage 3 within the stay.
Writes results/s2_enrichment.json."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

R = "results"
THR = 0.15
pe = pd.read_parquet(f"{R}/s1_preds_external.parquet")          # ge1, p + p_rec
pe2 = pd.read_parquet(f"{R}/s1_preds_ext_ge2.parquet")[["stay_id", "t_hr", "y"]] \
    .rename(columns={"y": "ge2"})
pe3 = pd.read_parquet(f"{R}/s1_preds_ext_ge3.parquet")[["stay_id", "t_hr", "y"]] \
    .rename(columns={"y": "ge3"})
d = pe.merge(pe2, on=["stay_id", "t_hr"]).merge(pe3, on=["stay_id", "t_hr"])

# ---- stage trajectory per stay (any ge2/ge3 event anywhere in stay) ----
prog2 = d.groupby("stay_id")["ge2"].max()
prog3 = d.groupby("stay_id")["ge3"].max()

# ---- cr trajectory stats per stay for FP autopsy (near-criteria) ----
labs = pd.read_parquet(f"{R}/n8_2_labs_series.parquet")
cr = labs[labs["labname"] == "creatinine"].rename(columns={"sid": "stay_id", "labresult": "cr_val"})
cr = cr[cr["cr_val"].between(0.1, 30)]
base_map = pd.read_parquet(f"{R}/p1_expanded_features.parquet", columns=["stay_id", "cr_base"]) \
    .drop_duplicates("stay_id").set_index("stay_id")["cr_base"]
cr = cr[cr["stay_id"].isin(base_map.index)]
cr["base"] = cr["stay_id"].map(base_map)
cr = cr.dropna(subset=["base"])
cr["fold"] = cr["cr_val"] / cr["base"].clip(lower=0.1)
cr["rise"] = cr["cr_val"] - cr["base"]
peak_fold = cr.groupby("stay_id")["fold"].max()
peak_rise = cr.groupby("stay_id")["rise"].max()
# staging threshold reached anywhere (>=1.5 or >=0.3) — to define "never staged"
staged_any = ((peak_fold >= 1.5) | (peak_rise >= 0.3))

# ---- event (ge1) occurrence per stay from predictions ----
event_stay = d.groupby("stay_id")["y"].max()

# ---- alerts at working threshold ----
d["alert"] = d["p_rec"] >= THR
alert_stays = set(d.loc[d["alert"], "stay_id"])
event_stays = set(d.loc[d["y"] == 1, "stay_id"])
fp_stays = alert_stays - event_stays
n_fp = len(fp_stays)

# 1. FP autopsy
fp_idx = peak_fold.index.intersection(list(fp_stays))
early = sum(1 for s in fp_stays if event_stay.get(s, 0) == 1)   # event later in stay (any window)
near = sum(1 for s in fp_idx
           if (not staged_any.get(s, False)) and
              ((1.3 <= peak_fold[s] < 1.5) or (0.2 <= peak_rise[s] < 0.3)))
autopsy = {
    "threshold": THR, "alerted_event_free_stays": n_fp,
    "alert_preceded_event_later_in_stay": early,
    "pct_early": round(100 * early / n_fp, 1),
    "near_criteria_no_staging": int(near),
    "pct_near_criteria": round(100 * near / n_fp, 1),
    "neither": int(n_fp - early - near),
    "pct_neither": round(100 * (n_fp - early - near) / n_fp, 1)}
print("FP autopsy:", json.dumps(autopsy))

# 2. workload translation
work = {}
for t in [0.05, 0.10, 0.15, 0.20]:
    al = d["p_rec"] >= t
    al_st = set(d.loc[al, "stay_id"])
    days_all = d.groupby("stay_id")["t_hr"].max().clip(lower=6).div(24).sum()
    al_days = d[al].groupby("stay_id")["t_hr"].max().clip(lower=6).div(24).sum()
    work[f"thr{t}"] = {
        "pct_stays_alerted": round(100 * len(al_st) / d["stay_id"].nunique(), 1),
        "pct_patient_days_under_alert": round(100 * al_days / days_all, 1)}
print("workload:", json.dumps(work))

# 3. progression-anchored capture (onset = first ge1 bin of stay)
def onset_map(df, col):
    ev = df[df[col] == 1]
    return ev.groupby("stay_id")["t_hr"].min()

on1 = onset_map(d, "y")
cap = {}
for prog_col, prog_lab in [("ge2", "prog_ge2"), ("ge3", "prog_ge3")]:
    prog_stays = set(d.loc[d[d[prog_col] == 1]["prog_col"] if False else d[prog_col] == 1, "prog_col"].index) \
        if False else set(d.loc[d[prog_col] == 1, "stay_id"])
    ev_prog = event_stays & prog_stays
    caught = 0
    for s in ev_prog:
        g = d[d["stay_id"] == s]
        on = on1.get(s, 1e9)
        if (g["alert"] & (g["t_hr"] < on)).any():
            caught += 1
    cap[prog_lab] = {"n_event_stays": len(ev_prog), "captured": caught,
                     "capture_rate": round(caught / len(ev_prog), 3) if ev_prog else None}
# also plain ge1 capture for reference
caught_all = sum(1 for s in event_stays
                 if (d[(d.stay_id == s)]["alert"] & (d[(d.stay_id == s)]["t_hr"] < on1.get(s, 1e9))).any())
cap["all_ge1"] = {"n_event_stays": len(event_stays), "captured": caught_all,
                  "capture_rate": round(caught_all / len(event_stays), 3)}
print("progression capture:", json.dumps(cap))

json.dump({"fp_autopsy": autopsy, "workload": work, "progression_capture": cap},
          open(f"{R}/s2_enrichment.json", "w"), indent=2)
print("saved s2_enrichment.json")
