# abi-aki-risk-calculator

## Paper 2: JinhuaNSICU development with MIMIC-IV and eICU-CRD external validation

`app.py`, `riskcalc.py`, `model/` and `tests/` in this repository belong to the companion analysis (model developed on MIMIC-IV) and are unchanged. Paper 2 reverses the roles: the model is developed and internally tested on JinhuaNSICU and applied without refitting to MIMIC-IV and eICU-CRD. Its scripts are in `analysis/` as byte-identical copies of the scripts that produced the reported values:

| Script | Purpose |
|---|---|
| `analysis/f4_jinhu_extract.py` | JinhuaNSICU cohort, laboratory, vital-sign and treatment extraction; unit conversions |
| `analysis/f5_jinhu_features.py` | creatinine event axis, six-hour grid (24 h to min(LOS - 6, 168) h), rolling features |
| `analysis/f10_jinhu_dev.py` | subject-level 80/20 split, labels, re-selection of 166 features from 280 candidates, LightGBM, recalibration layer |
| `analysis/f11_jinhu_full_eval.py` | full evaluation, calibration deciles, decision curves, external application |
| `analysis/f12_jinhua_full_suite.py` | two-tier suite: discrimination, paired differences, calibration, deployment (definition version 2), hospitals, subgroups, SHAP |
| `analysis/f13_posthoc_supplement.py` | post hoc supplement on the frozen f12 prediction files: bootstrap intervals for calibration and Brier score, proximity standardisation, equivalence test, Tier-2 recalibration views, extended threshold curves, false-alarm review, MIMIC-IV descriptors; reproduces 27 stored f12 values before computing anything new |

The prediction files read by `f13` are produced by `f12` and are not included in this repository because they hold patient-level identifiers of credentialed databases.

Reported development-cohort AUROC for any AKI (Tier 1): 0.815 internal test, 0.769 MIMIC-IV, 0.777 eICU-CRD. The internal Tier-2 value (0.995) rests on 3 event stays and is underpowered. The JinhuaNSICU-trained model objects are not saved by these scripts (they retrain with fixed seed 42). JinhuaNSICU is open access; MIMIC-IV and eICU-CRD require PhysioNet credentialed access. The scripts are shipped unedited, including two unused stub lines in `f12` (`lr_int`, `severity_grad`).


**Rolling two-tier AKI risk model after acute brain injury — 268-feature LightGBM with frozen dual recalibration, externally validated across 171 eICU hospitals**

> **Research demonstration, not a clinical decision tool; confirm thresholds locally.** The model has not been
> prospectively validated. Do not use it to guide patient care.

## What the model does

At every six-hour checkpoint of an ICU stay after acute brain injury (ABI), the model estimates two risks from
data charted from ICU admission up to that checkpoint:

| Tier | Endpoint (KDIGO creatinine criteria, label window [t, t + 54 h)) | Model file |
|---|---|---|
| Tier 1 (primary) | any AKI, ≥Stage 1 | `model/model_tier1_268.joblib` |
| Tier 2 (key secondary) | severe AKI, ≥Stage 3 | `model/model_tier3_268.joblib` |

Both are LightGBM classifiers on the same 268 features. Each raw probability passes through a frozen logistic
recalibration layer, p_rec = logistic(a + b·logit(p)), fitted on patient-grouped (GroupKFold) out-of-fold
development predictions and applied unchanged to all validation data:

| Tier | a | b |
|---|---|---|
| Tier 1 | −0.552414 | 0.434874 |
| Tier 2 | −0.301766 | 0.393865 |

The coefficients are read at runtime from `model/calibration_and_manifest.json`.

## Performance reported in the manuscript

Development: MIMIC-IV v3.1, split by unique patient (80/20, seed 42): training 5,969 stays, internal test 1,473 stays.
External validation (no refitting): eICU-CRD v2.0, 9,071 stays and 84,866 checkpoints from 171 hospitals.

| Endpoint | AUROC, internal test (95% CI) | AUROC, external (95% CI) |
|---|---|---|
| Tier 1, any AKI | 0.826 (0.787–0.863) | 0.778 (0.761–0.798) |
| Tier 2, severe AKI | 0.878 (0.676–0.998) | 0.914 (0.856–0.964) |

95% CIs are stay-level cluster bootstraps (1,000 resamples). The Tier-2 internal estimate rests on few events and
is imprecise. Calibration, threshold views, subgroups, hospital heterogeneity and a PROBAST+AI risk-of-bias
assessment are in the manuscript and its Supplementary Information.

## Demo app

```bash
pip install -r requirements.txt
pytest -q tests          # artefact hashes, manifest, recalibration formula, scoring
streamlit run app.py
```

The app scores a CSV with one row per checkpoint (`stay_id`, then the 268 model features; blanks are missing
values, handled natively by the trees as in development) and plots the Tier-1 and Tier-2 recalibrated
probabilities with reference lines at 0.15 (Tier 1) and 0.05 (Tier 2). These lines are development-side reference
points, not validated action thresholds. A column template is in `examples/feature_template.csv`.

**No patient examples are bundled.** MIMIC-IV and eICU-CRD rows may not be redistributed under the PhysioNet
data use agreement. Credentialed users can create local examples (written to the git-ignored `examples_local/`)
from the pipeline outputs:

```bash
python tools/make_local_examples.py --features results/f2_shap_X.parquet --preds results/
```

The script selects three internal-test stays (a severe-AKI stay, an any-AKI-only stay and an event-free stay) and
checks that the shipped models reproduce the stored predictions exactly before writing the files.

## How the model evolved (see Supplementary Material 1)

The study began under a statistical analysis plan frozen before any model was fitted (v1.0, 2026-09-20): a single
complete-KDIGO endpoint on a temporal split, reported in a prior submission and retired from the primary
presentation. In revision, KDIGO creatinine staging became the endpoint of record with a subject-level random 80/20
split (D44–D45). The **E02 correction** (D46) made the Stage 3 absolute-creatinine branch (≥4.0 mg/dL) require an
acute criterion (fold change ≥1.5 or rise ≥0.3 mg/dL), as KDIGO specifies; all endpoints were relabelled, all models
retrained and all downstream outputs regenerated. The feature set was then **re-selected** on the corrected labels
(D47): from 350 candidate features computable in both databases, features up to 99% cumulative gain importance in
patient-grouped cross-validation on training patients were retained, giving 268 features. Stage 1 and Stage 2 were
merged into Tier 1 after paired tests showed no discrimination difference (D48). All values from earlier versions,
including those in earlier versions of this repository, are superseded.

## Repository contents

```
app.py                         Streamlit demo (research use only)
riskcalc.py                    loading, scoring and recalibration functions
model/                         model_tier1_268.joblib, model_tier3_268.joblib, calibration_and_manifest.json
examples/feature_template.csv  268-feature column template (no data)
tools/make_local_examples.py   creates local examples from credentialed data (git-ignored output)
tests/                         pytest suite
analysis/                      original analysis scripts with a pipeline guide (analysis/README.md)
```

`analysis/` is a working record: scripts are kept as run, with hard-coded local paths, Chinese comments and
superseded branches. Only the pipeline listed in `analysis/README.md` produced the reported values.

## Data

MIMIC-IV v3.1 (https://doi.org/10.13026/kpb9-mt58) and eICU-CRD v2.0 (https://doi.org/10.13026/C2WM1R) are
available through PhysioNet after credentialing and a data use agreement. This repository contains no patient-level data.

## Citation

Sheng J, Liu X, Lin R, Sun Q, Chen X, Li K, Chen W. Externally validated rolling prediction of any and severe acute
kidney injury after acute brain injury. Manuscript submitted.

## Disclaimer

Research demonstration, not a clinical decision tool; confirm thresholds locally. Provided as is, without warranty.

Post hoc analysis scripts: f13_posthoc_supplement.py (round-11.1 post hoc suite), f14_gap_fill.py (funnel, near-miss creatinine, Tier-2 development-protocol recalibration layer), f16_table1_gap_fill.py (Table 1 demographic gap fill; eICU-CRD unique-patient count from the database patient table).
