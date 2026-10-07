# -*- coding: utf-8 -*-
"""Backend for the Paper-2 (JinhuaNSICU-developed) rolling AKI risk calculator.

Inference only — no training, no fitting. Artifacts are read from ./model/ (byte-identical
copies of the frozen exports):
  model_anyaki.joblib     LightGBM (class-weighted), any AKI  >= KDIGO creatinine Stage 1
  model_severeaki.joblib  LightGBM (class-weighted), severe AKI >= KDIGO creatinine Stage 3
  calibration.json        frozen logistic recalibration, any-AKI only:
                          p_cal = sigmoid(a + b * logit(p_raw))
  features.json           166 ordered model features
  features_spec.json      clinical labels + development-cohort medians ("default")
The severe-AKI output has no deployable calibration layer and is reported as a raw
ranking score.

Feature semantics follow the development feature script (f5_jinhu_features.py):
  24-h window (t-24, t]: *_last / *_min / *_max / *_slope (per hour) / *_h_last ...
  48-h window (t-48, t] for cr/bun/hr_rate/sbp: *_last48 = LAST value in the window,
  *_delta48 = last - first, *_slope48 = np.polyfit slope per HOUR, *_sd48 (ddof=1)
  *_locf = last available value; cr_ratio_base = cr_last / cr_base;
  cr_delta_since_adm = cr_last - cr_base; <drug>_n24 = number of administrations in 24 h.
`male` and `t_hr` are NOT model features.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "model"
EPS = 1e-12                     # same logit clip as the JinhuaNSICU development code (f10-f12)

BAND_CUTS = (0.10, 0.15)        # LOW < 0.10 <= INTERMEDIATE < 0.15 (working) <= HIGH
ANY_WORKING = 0.15
ANY_REFS = (0.10, 0.15, 0.20)   # reference lines drawn on the any-AKI scale
# Severe-AKI contextual reference (severe round): raw severe score of the median-input profile, i.e. every feature at
# its development-cohort median (features_spec.json 'default'); computed at runtime by severe_reference(). It is NOT a
# cohort percentile (development predictions are not shipped). The severe output stays raw / uncalibrated (ranking only).
RISK_GRID = (0.05, 0.08, 0.10, 0.15, 0.20)   # deployment operating points in the frozen suite


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def recalibrate(p, a: float, b: float):
    return 1.0 / (1.0 + np.exp(-(a + b * logit(p))))


def load_models(art: Path = ART) -> dict:
    cal = json.loads((art / "calibration.json").read_text(encoding="utf-8"))
    feats = json.loads((art / "features.json").read_text(encoding="utf-8"))
    spec = {s["feature"]: s for s in json.loads((art / "features_spec.json").read_text(encoding="utf-8"))}
    if len(feats) != 166:
        raise ValueError(f"{len(feats)} features, expected 166")
    m_any = joblib.load(art / "model_anyaki.joblib")
    m_sev = joblib.load(art / "model_severeaki.joblib")
    for m in (m_any, m_sev):
        if int(m.n_features_in_) != 166:
            raise ValueError("model expects a different feature count")
    return {"any": m_any, "severe": m_sev, "a": float(cal["a"]), "b": float(cal["b"]),
            "transform": cal.get("transform", ""), "features": feats, "spec": spec}


def defaults_vector(models: dict) -> dict:
    """Development-cohort median for every model feature (features_spec.json 'default')."""
    return {f: float(models["spec"][f]["default"]) for f in models["features"]}


def band(p_cal: float) -> int:
    """0 = LOW (<0.10), 1 = INTERMEDIATE (0.10-<0.15), 2 = HIGH (>=0.15 working threshold)."""
    return 0 if p_cal < BAND_CUTS[0] else (1 if p_cal < BAND_CUTS[1] else 2)


def predict(X: pd.DataFrame, models: dict) -> pd.DataFrame:
    """X: rows with (a subset of) the 166 feature columns; absent columns -> NaN (native missing)."""
    M = X.reindex(columns=models["features"]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    p_any = models["any"].predict_proba(M)[:, 1]
    out = pd.DataFrame(index=X.index)
    out["any_p_raw"] = p_any
    out["any_p_cal"] = recalibrate(p_any, models["a"], models["b"])
    out["severe_p_raw"] = models["severe"].predict_proba(M)[:, 1]
    out["band"] = [band(p) for p in out["any_p_cal"]]
    return out


def score_frame(df: pd.DataFrame, models: dict) -> tuple[pd.DataFrame, list]:
    """Batch scoring. Returns (scored frame with stay_id/t_hr first, list of missing feature columns)."""
    missing = [f for f in models["features"] if f not in df.columns]
    res = predict(df, models)
    for c in ("t_hr", "stay_id"):
        if c in df.columns:
            res.insert(0, c, df[c].values)
    if "stay_id" not in res.columns:
        res.insert(0, "stay_id", "uploaded")
    return res, missing


# ---------------------------------------------------------------------------------------
# Single-point mode: 14 bedside inputs -> 166-feature vector
# ---------------------------------------------------------------------------------------
BEDSIDE_DEFAULTS = dict(age=64, charlson=1, male=True,
                        cr_base=0.77, cr_cur=0.77, cr_low24=0.77, cr_first48=0.77, t_hr=54,
                        sbp=129, hr=80, spo2=99, rr=16, bun=16.6, norepi_n24=0)


def bedside_to_features(b: dict, models: dict) -> tuple[dict, list]:
    """Map bedside inputs to the 166-feature vector.

    Returns (feature dict, list of features that were set from the inputs). Every feature
    not in that list keeps its development-cohort median.

    Creatinine timeline assumed for the 48-h trend: first value in the 48-h window at
    t - span (span = min(t_hr, 48) h), lowest 24-h value at t - 12 h, current value at t.
    male and t_hr are recorded for context only (not model features).
    """
    v = defaults_vector(models)
    set_ = []

    def put(k, x):
        if k not in v:
            raise KeyError(k)
        v[k] = float(x)
        set_.append(k)

    put("age", b["age"])
    put("charlson", b["charlson"])

    base = max(float(b["cr_base"]), 0.1)
    cur = max(float(b["cr_cur"]), 0.1)
    low24 = min(max(float(b["cr_low24"]), 0.1), cur)      # the 24-h minimum cannot exceed current
    first48 = max(float(b["cr_first48"]), 0.1)
    t = float(b["t_hr"])
    span = max(min(t, 48.0), 6.0)

    put("cr_base", base)
    for k in ("cr_last", "cr_locf", "cr_last48"):        # most recent / last available / last in 48 h
        put(k, cur)
    put("cr_ratio_base", cur / base)
    put("cr_delta_since_adm", cur - base)
    put("cr_min", low24)
    put("cr_max", cur)                                   # max over the 24-h window given low24 <= current
    put("cr_delta48", cur - first48)
    hh = [t - span, t]
    vv = [first48, cur]
    if span > 12.0:
        hh.insert(1, t - 12.0)
        vv.insert(1, low24)
    hh, vv = np.array(hh), np.array(vv)
    put("cr_slope48", float(np.polyfit(hh, vv, 1)[0]))    # mg/dL per hour (as in development)
    put("cr_sd48", float(np.std(vv, ddof=1)))

    bun = float(b["bun"])
    for k in ("bun_last", "bun_locf", "bun_last48"):
        put(k, bun)
    put("bun_max", max(v["bun_max"], bun))
    put("bun_min", min(v["bun_min"], bun))

    sbp = float(b["sbp"])
    put("sbp_locf", sbp)
    put("sbp_last48", sbp)

    hr = float(b["hr"])
    put("hr_rate_last", hr)
    put("hr_rate_max", max(v["hr_rate_max"], hr))
    put("hr_rate_min", min(v["hr_rate_min"], hr))

    spo2 = float(b["spo2"])
    put("spo2_locf", spo2)
    put("spo2_last", spo2)

    rr = float(b["rr"])
    put("rr_last", rr)
    put("rr_max", max(v["rr_max"], rr))
    put("rr_min", min(v["rr_min"], rr))

    put("norepi_n24", int(b["norepi_n24"]))
    return v, sorted(set(set_))


def predict_bedside(b: dict, models: dict) -> dict:
    fv, used = bedside_to_features(b, models)
    r = predict(pd.DataFrame([fv]), models).iloc[0]
    return {"any_p_raw": float(r["any_p_raw"]), "any_p_cal": float(r["any_p_cal"]),
            "severe_p_raw": float(r["severe_p_raw"]), "band": int(r["band"]),
            "n_set": len(used), "used": used, "features": fv}


def severe_reference(models: dict) -> float:
    """Raw severe-AKI score of the median-input profile (contextual reference line; not a percentile)."""
    return float(predict(pd.DataFrame([defaults_vector(models)]), models).iloc[0]["severe_p_raw"])


def template_frame(models: dict) -> pd.DataFrame:
    """166-column batch template: stay_id, t_hr, then the 166 features (one median row)."""
    row = {"stay_id": "example_stay", "t_hr": 24}
    row.update(defaults_vector(models))
    return pd.DataFrame([row])
