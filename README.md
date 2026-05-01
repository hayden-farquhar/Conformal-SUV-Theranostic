# Cross-Cohort Calibration of Conformal Prediction for Lesion-Level SUV Quantification in Theranostic PET

Reproducibility code for the accompanying manuscript:

> **Cross-Cohort Calibration of Conformal Prediction for Lesion-Level SUV Quantification in Theranostic PET: A Three-Cohort External-Validation Study with Per-Cohort Recalibration.** Hayden Farquhar (Independent researcher; ORCID [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X)).

- **Pre-registration:** OSF [10.17605/OSF.IO/4KAZN](https://doi.org/10.17605/OSF.IO/4KAZN) (twelve amendments on component j5ry4)
- **Preprint:** to be posted

## Overview

This study presents a pre-registered three-cohort evaluation of split conformalised quantile regression (CQR) for lesion-level SUV<sub>max</sub> quantification in PET, spanning two tracers (FDG, PSMA), four scanner vendors, and seven independent centres (n=14,621 lesions across 1,470 patients from AutoPET-I, AutoPET-III, and HECKTOR 2025). The repository contains the conformal prediction infrastructure, the Phase 2 within-replicate coefficient of variation (wCV) reference pipeline, the Phase 3 four-mode coverage comparison driver, and the locked input parquets that allow bit-identical reproduction of all reported numbers, tables, and figures.

Four covariate-shift correction methods are compared: (i) naive transfer of CQR thresholds; (ii) weighted conformal prediction (WCP) with image-only radiomics features; (iii) WCP with extended cohort-defining categorical features alongside a pre-registered support-overlap diagnostic; and (iv) per-cohort split conformal recalibration with 20% patient-level holdout. The Calibration Shift Index (CSI) is introduced as a small-sample deployment diagnostic.

## Data sources

| Source | DOI | Access | Licence |
|---|---|---|---|
| AutoPET-I (FDG, whole-body) | [10.57754/FDAT.wf9fy-txq84](https://doi.org/10.57754/FDAT.wf9fy-txq84) | Free; FDAT registration | CC BY-NC 4.0 |
| AutoPET-III (PSMA, whole-body) | [10.7937/R7EP-3X37](https://doi.org/10.7937/R7EP-3X37) | Free; TCIA REST API | CC BY 4.0 |
| HECKTOR 2025 (FDG, head-and-neck) | (challenge release) | Challenge participants; HECKTOR DUA | CC BY 4.0 components |

The locked per-lesion feature parquets (~1.5 MB total) are bundled in
`data/interim/lesion_tables/` and `data/processed/`. Their SHA-256 hashes
match the OSF j5ry4 freeze and are verified by `scripts/verify_input_shas.py`.
The raw DICOM and segmentation NIfTI archives (~330 GB total) are not
redistributed; see `data_dictionary.md` for end-to-end rebuilding
instructions.

## Requirements

Python 3.14 (any 3.10+ should work; the locked Phase 3 freeze used 3.14.3).
Install the curated runtime dependencies via:

```bash
pip install -r requirements.txt
```

For exact-bit reproducibility including transitive dependencies, install
the full pip-freeze instead:

```bash
pip install -r requirements-frozen.txt
```

Approximate disk requirements: 2.5 MB for the repository, plus ~5 GB if you
materialise the raw DICOM archives on disk for full preprocessing.

## Reproduction

The locked Phase 3 freeze can be reproduced bit-identically from the bundled
parquets without re-running preprocessing:

```bash
# 1. Verify locked input artefacts
python scripts/verify_input_shas.py

# 2. Phase 3 four-mode coverage comparison (locked driver)
python scripts/phase3_evaluate_amendment_11.py --target suvmax --dry-run

# 3. Post-freeze sensitivity analyses
python scripts/sensitivity_percohort_seed.py
python scripts/sensitivity_back_transformed_widths.py

# 4. Regenerate manuscript figures
python scripts/build_manuscript_figures.py
```

To rebuild the lesion parquets from raw imaging:

```bash
# 5. Acquire raw data per data_dictionary.md instructions
# 6. AutoPET-I preprocessing (Kaggle/Colab notebook)
jupyter run kaggle_notebooks/process_autopet_i.ipynb

# 7. AutoPET-III preprocessing + nnU-Net inference
jupyter run kaggle_notebooks/preprocess_autopet_iii.ipynb
jupyter run kaggle_notebooks/process_autopet_iii.ipynb
jupyter run kaggle_notebooks/infer_autopet_iii.ipynb

# 8. HECKTOR lesion extraction (run cells in hecktor_lesion_extraction_cells.txt)

# 9. Phase 2 wCV reference (Poisson-noise injection on a 50-series subsample)
python scripts/sample_autopet_iii_phase2.py
python scripts/run_phase2_poisson_reference.py
```

Estimated runtime on a single workstation (no GPU required for Phase 3):
Phase 3 driver ~2 minutes; sensitivity analyses ~1 minute each; figure
regeneration ~30 seconds. The nnU-Net inference (step 7) requires a GPU and
runs in approximately 15 T4-hours on Kaggle's free tier.

## Repository structure

```
repository/
├── README.md                       This file
├── LICENSE                         MIT (code) + CC BY 4.0 (docs/data tables)
├── data_dictionary.md              Variable definitions and data sources
├── requirements.txt                Curated runtime dependencies (pinned)
├── requirements-frozen.txt         Full pip-freeze (184 packages)
├── .zenodo.json                    Zenodo deposit metadata
├── configs/
│   └── default.yaml                Project configuration
├── data/
│   ├── interim/
│   │   └── lesion_tables/          Per-cohort lesion parquets
│   │       ├── autopet_i_lesions.parquet
│   │       ├── autopet_iii_lesions_reviewed.parquet
│   │       └── hecktor_lesions_reviewed.parquet
│   └── processed/
│       ├── autopet_i_splits.parquet
│       └── phase2_autopet_iii_primary_wcv.parquet
├── kaggle_notebooks/               Data acquisition + preprocessing notebooks
├── scripts/                        Analysis driver scripts
├── src/
│   ├── conformal/                  CQR, WCP, Mondrian, coverage modules
│   ├── preprocess/                 SUV conversion, lesion extraction
│   ├── features/                   Lesion feature extractors
│   ├── segment/                    DICOM I/O
│   ├── testretest/                 Phase 2 Poisson-noise pipeline
│   ├── evaluation/                 Bootstrap, seed-stability, figures
│   └── clinical/                   VISION / PERCIST overlay, indeterminacy
├── tests/                          Unit tests (pytest)
└── outputs/
    ├── figures/                    Regenerated figures (after running step 4)
    └── tables/                     Regenerated tables (after running step 4)
```

## Script descriptions

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `verify_input_shas.py` | Verify locked artefact SHA-256 hashes match OSF freeze | `data/interim/lesion_tables/*.parquet`, `data/processed/*.parquet`, `src/conformal/*.py` | stdout PASS/FAIL ledger |
| `phase3_evaluate.py` | Phase 3 baseline evaluation (cohort loaders + naive transfer) | All locked parquets | `results/phase3/*.csv` |
| `phase3_evaluate_amendment_11.py` | Phase 3 four-mode comparison (naive / WCP-image / WCP-extended / per-cohort recal) + CSI | All locked parquets | `results/phase3/amendment_11/*.parquet` |
| `phase3_diagnose.py` | Read-only Phase 3 diagnostic | Locked freeze | stdout coverage table |
| `cqr_smoke_test_autopet_i.py` | CQR base-model smoke test on AutoPET-I cal/test | AutoPET-I splits + lesions | stdout |
| `run_freeze_gate_1.py` | Patient-level split allocator (seed=42) | AutoPET-I lesions | `data/processed/autopet_i_splits.parquet` |
| `phase2_preflight.py` | Phase 2 pipeline preflight checks | AutoPET-III lesions | stdout |
| `sample_autopet_iii_phase2.py` | Stratified 50-series subsample for Phase 2 | AutoPET-III lesions | sampling manifest |
| `run_phase2_poisson_reference.py` | Phase 2 Poisson-noise reference (within-lesion CV) | AutoPET-III subsample | `data/processed/phase2_autopet_iii_primary_wcv.parquet` |
| `run_phase2_validation.py` | Phase 2 §3.5 validation (9/9 pipeline PASS check) | AutoPET-III lesions | stdout |
| `enumerate_autopet_iii_serial_pairs.py` | Serial-pair enumeration for test-retest analysis | AutoPET-III lesions | enumeration manifest |
| `section_3_9_*.py` (6 files) | §3.9 outlier triage (Amendment 5 INSUFFICIENT_AGREEMENT branch) | Cohort lesions, review decisions | reviewed parquets |
| `sensitivity_percohort_seed.py` | Post-freeze sensitivity: per-cohort recalibration across seeds | Locked freeze | `results/phase3/amendment_11/sensitivity_seed.csv` |
| `sensitivity_back_transformed_widths.py` | Post-freeze sensitivity: median interval widths in natural-SUV units | Locked freeze | `results/phase3/amendment_11/widths_natural_suv.csv` |
| `build_manuscript_figures.py` | Regenerate publication figures + tables from locked freeze | Locked freeze | `outputs/figures/*.png + .pdf`, `outputs/tables/*.md` |

## Outputs

After running the scripts in `Reproduction` order, the following outputs
correspond to the manuscript figures and tables (manuscript section
references are in the bundled supplementary):

| Output | Manuscript reference |
|---|---|
| `outputs/figures/fig01_coverage_four_modes.{png,pdf}` | Figure 1: four-mode marginal coverage comparison |
| `outputs/figures/fig02_importance_weight_distributions.{png,pdf}` | Figure 2: WCP-image vs WCP-extended weights |
| `outputs/figures/fig03_per_feature_classifier_coefficients.{png,pdf}` | Figure 3: per-feature classifier coefficients |
| `outputs/figures/fig04_csi_vs_coverage_miss.{png,pdf}` | Figure 4: CSI vs naive-transfer coverage miss |
| `outputs/figures/fig05_residuals_per_cohort.png` | Figure 5: residual scatter |
| `outputs/figures/fig06_coverage_by_volume_decile.{png,pdf}` | Figure 6: per-volume-decile coverage |
| `outputs/figures/fig07_hecktor_centre_vendor_coverage.{png,pdf}` | Figure 7: HECKTOR per-centre coverage |
| `outputs/figures/fig08_gtvp_gtvn_coverage.{png,pdf}` | Figure 8: GTVp vs GTVn coverage |
| `outputs/figures/fig09_pipeline_overview.{png,pdf}` | Figure 9: end-to-end pipeline schematic |
| `outputs/figures/fig10_clinical_indeterminacy.{png,pdf}` | Figure 10: clinical indeterminacy at decision thresholds |
| `outputs/tables/table01_cohort_characteristics.md` | Table 1: cohort characteristics |
| `outputs/tables/table02_phase2_wcv_summary.md` | Table 2: Phase 2 wCV summary |
| `outputs/tables/table03_four_mode_coverage.md` | Table 3: four-mode marginal coverage |
| `outputs/tables/table04_support_overlap_diagnostic.md` | Table 4: WCP support-overlap diagnostic |
| `outputs/tables/table05_csi.md` | Table 5: Calibration Shift Index |
| `outputs/tables/table06_per_centre_coverage.md` | Table 6: HECKTOR per-centre coverage |
| `outputs/tables/table07_per_class_coverage.md` | Table 7: HECKTOR GTVp/GTVn coverage |
| `outputs/tables/table08_per_quartile_coverage.md` | Table 8: per-volume-quartile coverage |
| `outputs/tables/table09_hypothesis_verdicts.md` | Table 9: hypothesis verdicts |

## Tests

The `tests/` directory contains 13 pytest modules covering the conformal
modules, the SUV pipeline, the lesion-feature extractors, the Phase 2
Poisson-noise injection, and the clinical-overlay logic. Run all tests
from the repository root:

```bash
pytest tests/
```

The locked Phase 3 freeze was produced on a workstation passing 230 unit
tests across these modules.

## Citation

If you use this code, please cite the accompanying manuscript (citation
to be added once a preprint DOI is minted) and the OSF pre-registration:

```
Farquhar, H. Cross-Cohort Calibration of Conformal Prediction for
Lesion-Level SUV Quantification in Theranostic PET: A Three-Cohort
External-Validation Study with Per-Cohort Recalibration.
Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN
```

This repository will receive a Zenodo DOI on first tagged release.

## License

Code is released under the **MIT License**. Documentation and the derived
data tables under `data/` are released under **CC BY 4.0**, with the
upstream cohort licences (FDAT CC BY-NC 4.0 for AutoPET-I; TCIA CC BY 4.0
for AutoPET-III; HECKTOR 2025 CC BY 4.0 components under the HECKTOR DUA)
inheriting onto cohort-specific derivative content. See `LICENSE` and
`data_dictionary.md` for full licence inheritance.

## Acknowledgements

This work uses publicly available datasets from the AutoPET (FDG and PSMA)
and HECKTOR challenge organising committees, and the FDAT institutional
repository at the University of Tübingen. Pre-registration and amendment
trail at OSF/4KAZN.
