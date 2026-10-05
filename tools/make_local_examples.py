"""Create LOCAL example stays for the demo app from credentialed PhysioNet-derived files.

Output goes to examples_local/ (git-ignored). Do NOT commit these files: MIMIC-IV rows are covered by the
PhysioNet data use agreement and may not be redistributed.

Inputs (from the analysis pipeline, see analysis/README.md):
  --features  checkpoint-feature parquet with stay_id + the 268 model features
              (e.g. results/n5_abi_rolling_features.parquet, or results/f2_shap_X.parquet)
  --preds     directory holding f1_preds_int_ge1.parquet and f1_preds_int_ge3.parquet (internal test set)

Selection (deterministic): among internal-test stays present in --features, take the stay with the most
checkpoints in each class — (1) >=1 Tier-2 (>=Stage 3) positive checkpoint, (2) Tier-1 positive but never
Tier-2 positive, (3) event-free. Each file also passes a reproduction check: the frozen models must return the
same raw probabilities as the stored f1 predictions (max abs difference < 1e-9).

Usage: python tools/make_local_examples.py --features PATH --preds DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import riskcalc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--preds", required=True)
ap.add_argument("--out", default=str(ROOT / "examples_local"))
a = ap.parse_args()

models = riskcalc.load_models()
feats = models["tier1"]["features"]
X = pd.read_parquet(a.features)
X = X[["stay_id"] + feats] if "stay_id" not in feats else X[feats]
p1 = pd.read_parquet(Path(a.preds) / "f1_preds_int_ge1.parquet")
p3 = pd.read_parquet(Path(a.preds) / "f1_preds_int_ge3.parquet")
lab = p1[["stay_id", "t_hr", "y", "p"]].rename(columns={"y": "y1", "p": "p1"}).merge(
    p3[["stay_id", "t_hr", "y", "p"]].rename(columns={"y": "y3", "p": "p3"}), on=["stay_id", "t_hr"])
M = X.merge(lab, on=["stay_id", "t_hr"], how="inner")
if M.empty:
    sys.exit("No internal-test checkpoints found in --features")
g = M.groupby("stay_id").agg(n=("t_hr", "size"), y1=("y1", "max"), y3=("y3", "max"))
cls = {"severe_event": g[g.y3 == 1], "any_aki_not_severe": g[(g.y1 == 1) & (g.y3 == 0)], "event_free": g[g.y1 == 0]}
out = Path(a.out)
out.mkdir(exist_ok=True)
for name, sub in cls.items():
    if sub.empty:
        print("no stay for", name)
        continue
    sid = sub.sort_values(["n", "stay_id"], ascending=[False, True]).index[0]
    rows = M[M.stay_id == sid].sort_values("t_hr")
    res = riskcalc.score(rows, models)
    d1 = np.abs(res["tier1_p_raw"].values - rows["p1"].values).max()
    d3 = np.abs(res["tier2_p_raw"].values - rows["p3"].values).max()
    assert d1 < 1e-9 and d3 < 1e-9, (name, d1, d3)
    rows[["stay_id"] + feats].to_csv(out / f"example_{name}.csv", index=False)
    print(f"{name}: stay {sid}, {len(rows)} checkpoints, reproduction max|diff| tier1={d1:.1e} tier2={d3:.1e}")
print("written to", out, "(git-ignored; do not commit)")
