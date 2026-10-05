# analysis/ — scripts behind the manuscript

## Paper 2: JinhuaNSICU development with MIMIC-IV and eICU-CRD external validation

`app.py`, `riskcalc.py`, `model/` and `tests/` in this repository belong to the companion analysis (model developed on MIMIC-IV) and are unchanged. Paper 2 reverses the roles: the model is developed and internally tested on JinhuaNSICU and applied without refitting to MIMIC-IV and eICU-CRD. Its scripts are in `analysis/` as byte-identical copies of the scripts that produced the reported values:

| Script | Purpose |
|---|---|
| `analysis/f4_jinhu_extract.py` | JinhuaNSICU cohort, laboratory, vital-sign and treatment extraction; unit conversions |
| `analysis/f5_jinhu_features.py` | creatinine event axis, six-hour grid (24 h to min(LOS - 6, 168) h), rolling features |
| `analysis/f10_jinhu_dev.py` | subject-level 80/20 split, labels, re-selection of 166 features from 280 candidates, LightGBM, recalibration layer |
| `analysis/f11_jinhu_full_eval.py` | full evaluation, calibration deciles, decision curves, external application |
| `analysis/f12_jinhua_full_suite.py` | two-tier suite: discrimination, paired differences, calibration, deployment (definition version 2), hospitals, subgroups, SHAP |
| `analysis/f13_posthoc_supplement.py, f14_gap_fill.py (post hoc gap fill: funnel / near-miss creatinine / Tier-2 development-protocol layer), f16_table1_gap_fill.py (Table 1 demographic gap fill incl. eICU unique-patient count)` | post hoc supplement on the frozen f12 prediction files: bootstrap intervals for calibration and Brier score, proximity standardisation, equivalence test, Tier-2 recalibration views, extended threshold curves, false-alarm review, MIMIC-IV descriptors; reproduces 27 stored f12 values before computing anything new |

The prediction files read by `f13` are produced by `f12` and are not included in this repository because they hold patient-level identifiers of credentialed databases.

Reported development-cohort AUROC for any AKI (Tier 1): 0.815 internal test, 0.769 MIMIC-IV, 0.777 eICU-CRD. The internal Tier-2 value (0.995) rests on 3 event stays and is underpowered. The JinhuaNSICU-trained model objects are not saved by these scripts (they retrain with fixed seed 42). JinhuaNSICU is open access; MIMIC-IV and eICU-CRD require PhysioNet credentialed access. The scripts are shipped unedited, including two unused stub lines in `f12` (`lr_int`, `severity_grad`).


These are the original analysis scripts, unedited (58 files: `n*`, `s*`, `f*`, `g*` series). They were written
to run against locally held, credentialed copies of MIMIC-IV v3.1 and eICU-CRD v2.0 and use hard-coded local
paths (`E:/TBI subtype/...`, `results/`). Change those paths before running. No patient data is included in this repository.

The repository is a working record: script names, comments (partly in Chinese) and superseded branches are kept
as they were run. **Values quoted in the manuscript come only from the current pipeline listed under "Current
pipeline" below.** Earlier outputs (previous feature set, pre-correction labels, temporal split) are superseded; see
Supplementary Material 1 (deviations D44–D49) for the history.

## Current pipeline (the version reported in the manuscript)

| Step | Script(s) | Role | Main outputs |
|---|---|---|---|
| 1. Development cohort, event axis and grid (MIMIC-IV) | `n3_event_grid.py` | KDIGO event axis + six-hour rolling checkpoint grid | `n3_m4_abi_grid.parquet/.json` |
| 2. Development features (MIMIC-IV) | `n5_features.py` | admission-to-now rolling features for every checkpoint | `n5_abi_rolling_features.parquet` |
| 3. External cohort, extraction, grid, features (eICU-CRD) | `n8_1_eicu_abi_cohort.py` → `n8_2_eicu_abi_extract.py` → `n8_3_eicu_abi_grid.py` → `n8_4_eicu_abi_features.py` | frozen diagnosis crosswalk, time series, identical grid rules, feature schema | `n8_*` parquets/JSON |
| 4. Label correction (E02) | `s1_e02fix_only.py` | Stage 3 absolute-creatinine branch requires an acute criterion (KDIGO); label window = current + 8 following six-hour checkpoints | `s1e2_*` predictions |
| 5. Feature re-selection (268 features) | `f1_reselect_e02.py`, `f1b_selection_integrity.py` | 99% cumulative-gain rule in patient-grouped CV on training patients; integrity checks | `_reselected_features.json`, `f1_preds_{int,ext}_ge{1,2,3}.parquet`, `f1_reselection_results.json`, `f1b_selection_integrity.json` |
| 6. Full downstream suite | `f2_full_suite_268.py`, `f2b_fixups.py` | discrimination, Tier-1 recalibration layer, deployment metrics, DCA, subgroups, hospitals, proximity standardisation, SHAP, EPP | `f2_full_results.json`, `f2_*` |
| 7. Table 1 | `f3_table1.py` | subject-level split characteristics | `f3_table1.json` |
| 8. Patch analyses | `f9_tier2_calibration.py` | Tier-2 recalibration layer and threshold views | `f9_tier2_calibration.json` |
|  | `f9_patch_91.py`, `f9_patch_92.py`, `f9_docx_surgery.py`, `f9_docx_surgery_92.py` | text/ledger patches applied to manuscript files during revision (no statistics) | — |

The frozen artefacts in `../model/` are the Tier-1 (≥Stage 1) and Tier-2 (≥Stage 3) models of steps 5–8 with
their recalibration coefficients (`calibration_and_manifest.json`).

### Dependencies not contained in this folder

- The upstream MIMIC-IV cohort / extraction files read by steps 1–2 (`n1_m4_abi_cohort.parquet`, `n2_m4_abi_*.parquet`,
  `n4_abi_*.parquet`) were produced by scripts that are not included in this release.
- Steps 4–8 read the eICU-CRD feature table `p1_expanded_features.parquet`, which extends the `n8_4` output; its
  producer script (`p1_*` series) is not included in this release.
- `gate.py` (imported by 31 of the scripts here, marked G below) is not included.

## Gate mechanism

Scripts marked **G** call `from gate import gate; gate(__file__)` at start-up. The gate refuses to run unless a
preflight check has written a `.gate_passed` marker within the previous 30 minutes (for example, the eICU-CRD
preflight verified crosswalk vocabulary anchors and laboratory-name enumerations before any external extraction;
the preflight scripts are not included). To run a gated
script, run the preflight (or provide an equivalent `gate.py` that performs your own checks), then run the script
within the time window. `.gate_passed` is git-ignored.

G = `n3_event_grid`, `n5_features`, `n6_gbm`, `n7_deep_models*` (3), `n8_1`–`n8_5` (5), `g1`, `g2`, `g3`, `g4`,
`g4b`, `g5`, `g5b`, `g6`, `g6b`, `g7`, `g9`–`g18`.

## Superseded and historical scripts (kept for provenance, not used for reported values)

- `n6_gbm.py`, `n7_deep_models*.py`, `n8_5_eicu_abi_apply.py`: earlier single-endpoint model (previous feature set, temporal split) and deep-learning comparators.
- `s1_stage_primary.py`, `s1_fixed_rerun.py`, `s1b`–`s1f`, `s1e2_full_suite.py`, `s1_fill_manuscript.py`, `s2_enrichment.py`, `s2b_fix.py`, `s3_stats.py`: stage-stratified analysis before the feature re-selection.
- `g1`–`g21`: historical audit and revision line of the earlier model version (recalibration/DCA, deployment metrics, SHAP, fairness, robustness, baselines, bootstrap variants, reference renumbering and manuscript-editing utilities).

Not included: analyses of a non-PhysioNet cohort that is not part of the manuscript.
