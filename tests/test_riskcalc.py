"""Tests: artefact integrity, recalibration formula, scoring shape. No patient data required."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import riskcalc  # noqa: E402

SHA = {  # SHA-256 of the files shipped with the manuscript (see model/README section in README.md)
    "model_tier1_268.joblib": "f69678bf9f67f5e327e8943f72fde79c1066293c56559c37332d44287322af53",
    "model_tier3_268.joblib": "2ae6ea9b6cbbf1694700164c12a7f883dd04132fdbbbecbe7f6cd2280700e904",
    "calibration_and_manifest.json": "f8f541091d91ad13bdce2f86e606c40174cb632e6c2f5c505700d715a01f0350",
}


def test_artefact_hashes():
    for f, h in SHA.items():
        assert hashlib.sha256((ROOT / "model" / f).read_bytes()).hexdigest() == h, f


def test_manifest_and_models():
    man = json.loads((ROOT / "model/calibration_and_manifest.json").read_text())
    m = riskcalc.load_models()
    assert m["tier1"]["a"] == man["tier1"]["recal"]["a"] and m["tier1"]["b"] == man["tier1"]["recal"]["b"]
    assert m["tier2"]["a"] == man["tier2"]["recal"]["a"] and m["tier2"]["b"] == man["tier2"]["recal"]["b"]
    assert len(m["tier1"]["features"]) == man["n_features"] == m["tier1"]["model"].n_features_in_
    assert m["tier1"]["features"] == m["tier2"]["features"]
    # boosters were fitted on arrays (feature_name_ = Column_0..267): column ORDER is defined solely by the
    # artefact's 'features' list, which riskcalc.score() enforces via df[features].
    assert list(m["tier1"]["model"].feature_name_) == [f"Column_{i}" for i in range(man["n_features"])]


def test_recalibration_formula():
    a, b = -0.4, 0.5
    p = np.array([1e-6, 0.01, 0.2, 0.5, 0.9])
    exp = 1 / (1 + np.exp(-(a + b * np.log(p / (1 - p)))))
    assert np.allclose(riskcalc.recalibrate(p, a, b), exp)
    assert np.isclose(riskcalc.recalibrate(0.5, a, b), 1 / (1 + np.exp(-a)))  # logit(0.5) = 0
    assert np.all(np.diff(riskcalc.recalibrate(p, a, b)) > 0)  # monotone for b > 0


def test_score_all_missing_row_runs():
    m = riskcalc.load_models()
    df = pd.DataFrame([{f: np.nan for f in m["tier1"]["features"]}]).assign(stay_id=1)
    r = riskcalc.score(df, m)
    for c in ("tier1_p_rec", "tier2_p_rec"):
        assert 0 < r[c].iloc[0] < 1


def test_template_columns():
    cols = pd.read_csv(ROOT / "examples/feature_template.csv", nrows=0).columns.tolist()
    m = riskcalc.load_models()
    assert cols[0] == "stay_id" and set(cols[1:]) == set(m["tier1"]["features"]) and len(cols) == 269
