# -*- coding: utf-8 -*-
"""s2b: corrected enrichment (true criteria-onset, checkpoint-level FP autopsy)."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

R = "results"
THR = 0.15

# ---- true onset from cr series (same construction as s1d/s1f) ----
labs = pd.read_parquet(f"{R}/n8_2_labs_series.parquet")
cr = labs[labs["labname"] == "creatinine"].rename(columns={"sid": "stay_id", "labresult": "cr_val"})
cr = cr[cr["cr_val"].between(0.1, 30)].sort_values(["stay_id", "hr"])
cr["t_hr"] = (cr["hr"] // 6) * 6
cb = cr.groupby(["stay_id", "t_hr"])["cr_val"].max().reset_index().sort_values(["stay_id", "t_hr"])
mg = pd.read_parquet(f"{R}/p1_expanded_features.parquet", columns=["stay_id", "t_hr", "cr_base"]) \
    .drop_duplicates(["stay_id", "t_hr"]).merge(cb, on=["stay_id", "t_hr"], how="left")
fold = mg["cr_val"] / mg["cr_base"].clip(lower=0.1)
crit = ((fold >= 1.5) | ((mg["cr_val"] - mg["cr_base"]) >= 0.3)).fillna(False) & (mg["t_hr"] >= 24)
mg["crit"] = crit
mg["near"] = ((fold >= 1.3) & (fold < 1.5)) | ((mg["cr_val"] - mg["cr_base"] >= 0.2) & (mg["cr_val"] - mg["cr_base"] < 0.3))
mg["near"] = mg["near"].fillna(False)
onsets = {sid: g["t_hr"].min() for sid, g in mg[mg["crit"]].groupby("stay_id")}
# per-stay peak (for near-criteria among never-staged)
mg2 = mg.dropna(subset=["cr_val"]).copy()
mg2["fold2"] = mg2["cr_val"] / mg2["cr_base"].clip(lower=0.1)
mg2["rise2"] = mg2["cr_val"] - mg2["cr_base"]
peak_fold = mg2.groupby("stay_id")["fold2"].max()
peak_rise = mg2.groupby("stay_id")["rise2"].max()

pe = pd.read_parquet(f"{R}/s1_preds_external.parquet")
pe2 = pd.read_parquet(f"{R}/s1_preds_ext_ge2.parquet")[["stay_id", "t_hr", "y"]].rename(columns={"y": "ge2"})
pe3 = pd.read_parquet(f"{R}/s1_preds_ext_ge3.parquet")[["stay_id", "t_hr", "y"]].rename(columns={"y": "ge3"})
d = pe.merge(pe2, on=["stay_id", "t_hr"]).merge(pe3, on=["stay_id", "t_hr"])
d["alert"] = d["p_rec"] >= THR
d["onset"] = d["stay_id"].map(onsets)

event_stays = set(d.loc[d["y"] == 1, "stay_id"])
alert_stays = set(d.loc[d["alert"], "stay_id"])

# ---- 1. progression-anchored capture (corrected) ----
cap = {}
prog2_st = set(d.loc[d["ge2"] == 1, "stay_id"])
prog3_st = set(d.loc[d["ge3"] == 1, "stay_id"])
for lab, prog_st in [("prog_ge2", prog2_st), ("prog_ge3", prog3_st), ("all_ge1", event_stays)]:
    target = event_stays if lab != "all_ge1" else event_stays
    src = event_stays & prog_st if lab != "all_ge1" else event_stays
    caught = sum(1 for s in src
                 if (d.loc[d["stay_id"] == s, "alert"] & (d.loc[d["stay_id"] == s, "t_hr"] < d.loc[d["stay_id"] == s, "onset"].max() if d.loc[d['stay_id']==s,'onset'].notna().any() else -1)).any())
    # 简洁正确版：
    caught = 0
    for s in src:
        g = d[d["stay_id"] == s]
        on = g["onset"].dropna()
        if len(on) and (g["alert"] & (g["t_hr"] < on.min())).any():
            caught += 1
    cap[lab] = {"n_event_stays": len(src), "captured": caught,
                "capture_rate": round(caught / len(src), 3) if src else None}
print("progression capture:", json.dumps(cap))

# ---- 2. checkpoint-level FP autopsy among ALERTED NEGATIVE checkpoints ----
al = d[d["alert"] & (d["y"] == 0)].copy()          # alerted, window-negative ckpts
al["event_later"] = al.apply(
    lambda r: (r["onset"] == r["onset"]) and (r["t_hr"] < r["onset"]) and (r["onset"] - r["t_hr"] > 48),
    axis=1)
n_ck = len(al)
n_early = int(al["event_later"].sum())
# among alerted event-free STAYS: near-criteria fraction
fp_st = alert_stays - event_stays
pf = peak_fold.reindex(list(fp_st)).dropna()
pr = peak_rise.reindex(list(fp_st)).dropna()
near_st = sum(1 for s in pf.index
              if ((pf[s] >= 1.3) and (pf[s] < 1.5)) or ((pr[s] >= 0.2) and (pr[s] < 0.3)))
aut = {"threshold": THR,
       "alerted_negative_checkpoints": n_ck,
       "too_early_checkpoints_event_beyond_48h": n_early,
       "pct_too_early": round(100 * n_early / n_ck, 1),
       "alerted_event_free_stays": len(fp_st),
       "near_criteria_stays": int(near_st),
       "pct_near_criteria": round(100 * near_st / len(fp_st), 1)}
print("FP autopsy:", json.dumps(aut))

# ---- 3. workload (same as s2, keep) ----
work = {}
for t in [0.05, 0.10, 0.15, 0.20]:
    a = d["p_rec"] >= t
    al_st = set(d.loc[a, "stay_id"])
    days_all = d.groupby("stay_id")["t_hr"].max().clip(lower=6).div(24).sum()
    al_days = d[a].groupby("stay_id")["t_hr"].max().clip(lower=6).div(24).sum()
    work[f"thr{t}"] = {"pct_stays_alerted": round(100 * len(al_st) / d["stay_id"].nunique(), 1),
                       "pct_patient_days_under_alert": round(100 * al_days / days_all, 1)}

json.dump({"fp_autopsy": aut, "workload": work, "progression_capture": cap},
          open(f"{R}/s2_enrichment.json", "w"), indent=2)
print("saved (corrected) s2_enrichment.json")
