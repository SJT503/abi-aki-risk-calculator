# -*- coding: utf-8 -*-
"""Self-check tests for calculator_p2.  Run:  pytest -q tests/"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import riskcalc as rc  # noqa: E402

# SHA-256 of the frozen exports as delivered in 02_artifacts/ (biomni_calc_v2_package.zip)
FROZEN_SHA = {
    "calibration.json": "25259666e8b83ffadf33e138046e675563d56aa4a64330141a1aaebb83b2ef4a",
    "features.json": "c7ad0dedf950cb7412800de0e908a7893d8e2fb3d4a4d710c4be8dd312622387",
    "features_spec.json": "01324ea09d8d6bbc426e8a41fbd8bc3d22d49bfc94741786dd1e0502910ad6b1",
    "model_anyaki.joblib": "2e0d032d5989ed37e77c51603e39a9e9c5348496d0ee4a9f56a16ac14a1d1bdd",
    "model_severeaki.joblib": "6cf4467c44f776aaa394617216701f5e67a0863e323b6a41b08433f56a3f0b62",
}


@pytest.fixture(scope="module")
def M():
    return rc.load_models()


def test_artifacts_byte_identical():
    for name, sha in FROZEN_SHA.items():
        assert hashlib.sha256((ROOT / "model" / name).read_bytes()).hexdigest() == sha, name


def test_166_features_and_template_order(M):
    feats = json.loads((ROOT / "model" / "features.json").read_text())
    assert len(feats) == 166 == len(set(feats))
    assert M["features"] == feats
    assert M["any"].n_features_in_ == 166 and M["severe"].n_features_in_ == 166
    tpl = pd.read_csv(ROOT / "feature_template.csv")
    assert list(tpl.columns) == ["stay_id", "t_hr"] + feats
    assert "male" not in feats and "t_hr" not in feats


def test_recalibration_coefficients_and_formula(M):
    assert M["a"] == pytest.approx(0.005067, abs=5e-7)      # f12 calibration.a
    assert M["b"] == pytest.approx(0.268467, abs=5e-7)      # f12 calibration.b
    p = np.array([1e-6, 1e-4, 0.01, 0.3, 0.9])
    z = np.log(p / (1 - p))
    np.testing.assert_allclose(rc.recalibrate(p, M["a"], M["b"]), 1 / (1 + np.exp(-(M["a"] + M["b"] * z))))
    # monotone increasing
    assert np.all(np.diff(rc.recalibrate(np.linspace(1e-6, 0.99, 50), M["a"], M["b"])) > 0)


def test_median_vector_score(M):
    r = rc.predict(pd.DataFrame([rc.defaults_vector(M)]), M).iloc[0]
    assert r["any_p_cal"] == pytest.approx(0.01827, abs=5e-5)


def test_column_order_robust(M):
    d = pd.DataFrame([rc.defaults_vector(M)])
    a = rc.predict(d, M)["any_p_raw"].iloc[0]
    b = rc.predict(d[d.columns[::-1]], M)["any_p_raw"].iloc[0]
    assert a == b


def test_severe_is_raw_uncalibrated(M):
    d = pd.read_csv(ROOT / "examples_local" / "demo_rapid_rise.csv")
    res = rc.predict(d, M)
    raw = M["severe"].predict_proba(d[M["features"]].to_numpy(float))[:, 1]
    np.testing.assert_array_equal(res["severe_p_raw"].to_numpy(), raw)
    assert "severe_p_cal" not in res.columns


def test_bedside_derivation_semantics(M):
    b = dict(rc.BEDSIDE_DEFAULTS, cr_base=0.80, cr_cur=1.15, cr_low24=1.00, cr_first48=0.88, t_hr=54,
             bun=30.0, norepi_n24=1)
    v, used = rc.bedside_to_features(b, M)
    assert set(v) == set(M["features"])                       # exactly the 166 model features
    assert "male" not in v and "t_hr" not in v
    assert v["cr_ratio_base"] == pytest.approx(1.15 / 0.80)
    assert v["cr_delta_since_adm"] == pytest.approx(0.35)
    assert v["cr_delta48"] == pytest.approx(1.15 - 0.88)        # last - first in 48-h window
    assert v["cr_last48"] == v["cr_last"] == v["cr_locf"] == pytest.approx(1.15)
    assert v["cr_min"] == pytest.approx(1.00) and v["cr_max"] == pytest.approx(1.15)
    hh, vv = np.array([6.0, 42.0, 54.0]), np.array([0.88, 1.00, 1.15])
    assert v["cr_slope48"] == pytest.approx(np.polyfit(hh, vv, 1)[0])   # per HOUR
    assert 0 < v["cr_slope48"] < 0.02
    assert v["cr_sd48"] == pytest.approx(np.std(vv, ddof=1))
    assert v["norepi_n24"] == 1 and v["bun_last"] == 30.0
    assert v["cr_h_last"] == M["spec"]["cr_h_last"]["default"]   # not entered -> median
    assert len(used) == 29


def test_bedside_ratio_monotone_with_current_cr(M):
    rs = [rc.bedside_to_features(dict(rc.BEDSIDE_DEFAULTS, cr_base=0.8, cr_cur=c), M)[0]["cr_ratio_base"]
          for c in (0.8, 1.0, 1.2)]
    assert rs[0] < rs[1] < rs[2]


def test_low24_clamped_to_current(M):
    v, _ = rc.bedside_to_features(dict(rc.BEDSIDE_DEFAULTS, cr_cur=0.9, cr_low24=1.4), M)
    assert v["cr_min"] == pytest.approx(0.9)


def test_screenshot_states(M):
    r0 = rc.predict_bedside(dict(rc.BEDSIDE_DEFAULTS), M)
    r1 = rc.predict_bedside(dict(rc.BEDSIDE_DEFAULTS, cr_base=0.80, cr_cur=1.15, cr_low24=1.00,
                                 cr_first48=0.88, bun=30.0, norepi_n24=1), M)
    assert round(r0["any_p_cal"] * 100, 1) == pytest.approx(2.05, abs=0.1) and r0["band"] == 0
    assert round(r1["any_p_cal"] * 100, 1) == 17.8 and r1["band"] == 2


def test_examples_scores(M):
    peaks, sev_peaks = {}, {}
    for n in ("low_rise", "intermediate", "rapid_rise"):
        d = pd.read_csv(ROOT / "examples_local" / f"demo_{n}.csv")
        s, miss = rc.score_frame(d, M)
        assert miss == [] and len(s) == 24
        peaks[n] = s["any_p_cal"].max()
        sev_peaks[n] = float(s["severe_p_raw"].max())
    assert round(peaks["low_rise"] * 100, 1) == 3.0 and rc.band(peaks["low_rise"]) == 0
    assert round(peaks["intermediate"] * 100, 1) == 25.9 and rc.band(peaks["intermediate"]) == 2
    assert round(peaks["rapid_rise"] * 100, 1) == 64.2 and rc.band(peaks["rapid_rise"]) == 2
    # Example C must be a SUBSTANTIVE severe-AKI case: raw score >= 1e-2 (v1 was 3.8e-05)
    # and >= 2 orders of magnitude above Example A, mirroring the real severe phenotype
    assert sev_peaks["rapid_rise"] >= 1e-2
    assert sev_peaks["rapid_rise"] / sev_peaks["low_rise"] >= 100


def test_bands():
    assert [rc.band(x) for x in (0.0999, 0.10, 0.1499, 0.15, 0.6)] == [0, 1, 1, 2, 2]


SRC = (ROOT / "app.py").read_text(encoding="utf-8")
WIDGET = re.compile(r"st\.(slider|number_input|checkbox|selectbox|text_input|radio|toggle|select_slider)\(([^\n]*)")


def test_no_uo_gcs_input_widgets():
    # v3 upload-first UI: any widget count is fine; none may mention UO/GCS
    calls = [m.group(0) for m in WIDGET.finditer(SRC)]
    assert len(calls) >= 1
    for c in calls:
        assert not re.search(r"urine|\buo\b|gcs|glasgow|尿量", c, re.I), c
    keys = [l for l in SRC.splitlines() if re.search(r'^\s+(uo|gcs|urine)\w*\s*=\s*st\.', l, re.I)]
    assert keys == []


def test_no_banned_vocabulary():
    for f in ("app.py", "riskcalc.py"):
        assert not re.search(r"t[i]er", (ROOT / f).read_text(encoding="utf-8"), re.I), f


def test_evidence_numbers_present():
    for s in ("0.815", "0.754–0.872", "0.769 · 0.777", "0.909 · 0.839 · 0.881", "1,033 / 7,442 / 9,071",
              "166 of 280", "1.476", "0.15 working", "0.20", "no urine output, no GCS",
              "v2026.10.08 · JinhuaNSICU development"):
        assert s in SRC, s


def test_app_runs_headless_en_zh():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.radio(key="lang").options) == 2          # English / 中文 toggle present
    at.radio(key="lang").set_value("中文").run()
    assert not at.exception


# ---------------- severe round (B1-B3) ----------------
def test_severe_reference_is_median_input_profile(M):
    ref = rc.severe_reference(M)
    r = rc.predict(pd.DataFrame([rc.defaults_vector(M)]), M).iloc[0]
    assert ref == pytest.approx(float(r["severe_p_raw"]), rel=1e-12)
    assert ref == pytest.approx(8.6647e-08, rel=1e-3)
    assert not hasattr(rc, "SEVERE_REF") and "0.05 reference" not in SRC and "0.05 参考线" not in SRC


def test_disclaimer_b1():
    assert "Externally validated on MIMIC-IV (7,442 stays) and eICU-CRD (9,071 stays, 171 hospitals)." in SRC
    assert "Designed for clinical decision support; run silent-mode evaluation, confirm thresholds and recalibrate" in SRC
    assert "not a clinical decision tool" not in SRC and "非临床决策工具" not in SRC


def test_key_result_b2():
    for s in ("KEY RESULT", "关键结果", "0.922 · 0.878", "0.882–0.954 · 0.791–0.953", "87 / 83 event stays",
              "best-discriminated endpoint", "RRT readiness", "区分度最高的终点"):
        assert s in SRC, s
    assert "matches published" not in SRC and "exceeds" not in SRC
    assert "underpowered" not in SRC and "检验效能不足" not in SRC   # dropped per PI 2026-10-08
