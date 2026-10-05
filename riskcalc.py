"""Core scoring functions for the two-tier ABI-AKI model (no training, no fitting).

Tier 1 = any AKI (>= KDIGO creatinine Stage 1); Tier 2 = severe AKI (>= Stage 3).
Each tier has a frozen LightGBM model (268 features) and a frozen logistic recalibration layer
    p_rec = logistic(a + b * logit(p))
whose coefficients are read at runtime from model/calibration_and_manifest.json (never hard-coded).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "model"
MANIFEST = MODEL_DIR / "calibration_and_manifest.json"
EPS = 1e-12  # same clipping as the analysis code (f2_full_suite_268.py / f9_tier2_calibration.py)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def recalibrate(p, a: float, b: float):
    """p_rec = logistic(a + b * logit(p))."""
    return 1.0 / (1.0 + np.exp(-(a + b * logit(p))))


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_models(model_dir: Path = MODEL_DIR) -> dict:
    """Return {'tier1': {...}, 'tier2': {...}} with model, feature list and recalibration coefficients."""
    man = load_manifest(Path(model_dir) / "calibration_and_manifest.json")
    out = {}
    for tier in ("tier1", "tier2"):
        art = joblib.load(Path(model_dir) / man[tier]["model"])
        feats = list(art["features"])
        if len(feats) != man["n_features"]:
            raise ValueError(f"{tier}: {len(feats)} features in artefact, manifest says {man['n_features']}")
        out[tier] = {"model": art["model"], "features": feats, "label": art.get("label", ""),
                     "a": float(man[tier]["recal"]["a"]), "b": float(man[tier]["recal"]["b"])}
    if out["tier1"]["features"] != out["tier2"]["features"]:
        raise ValueError("Tier-1 and Tier-2 feature lists differ")
    return out


def check_columns(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Return the list of required feature columns missing from df (empty list = OK)."""
    return [f for f in features if f not in df.columns]


def score(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Score one row per checkpoint. Missing values (NaN) are allowed and handled natively by LightGBM,
    as in development. Returns raw and recalibrated probabilities for both tiers."""
    feats = models["tier1"]["features"]
    miss = check_columns(df, feats)
    if miss:
        raise ValueError(f"{len(miss)} required feature columns missing, e.g. {miss[:5]}")
    # The boosters were fitted on arrays: column order MUST follow the artefact's feature list.
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    res = pd.DataFrame(index=df.index)
    for c in ("stay_id", "t_hr"):
        if c in df.columns:
            res[c] = df[c].values
    for tier in ("tier1", "tier2"):
        m = models[tier]
        p = m["model"].predict_proba(X)[:, 1]
        res[f"{tier}_p_raw"] = p
        res[f"{tier}_p_rec"] = recalibrate(p, m["a"], m["b"])
    return res
