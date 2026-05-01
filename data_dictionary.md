# Data dictionary

This file documents every variable in the locked Phase 3 input parquets and
the structure of the upstream cohorts. Variable names match those used by the
analysis scripts in `scripts/`.

## Upstream cohorts

| Cohort | Source | DOI | Licence | Access |
|---|---|---|---|---|
| AutoPET-I (FDG, whole-body) | FDAT (Forschungs-Datenrepositorium Tübingen) | 10.57754/FDAT.wf9fy-txq84 | CC BY-NC 4.0 | Free; institutional repository registration |
| AutoPET-III (PSMA, whole-body) | TCIA collection PSMA-PET-CT-Lesions | 10.7937/R7EP-3X37 | CC BY 4.0 | Free; TCIA REST API |
| HECKTOR 2025 (FDG, head-and-neck) | HECKTOR challenge organising committee | (per challenge release) | CC BY 4.0 components, HECKTOR DUA | Challenge participants |

The original AutoPET-I TCIA distribution migrated to dbGaP credentialing in
April 2025; the FDAT NIfTI distribution is the canonical free-access path used
in this work.

## Locked Phase 3 input parquets (`data/interim/lesion_tables/` and `data/processed/`)

The parquets are identical to those locked on OSF j5ry4 with bit-identical
SHA-256 verification (see `scripts/verify_input_shas.py`). Sizes are small
enough to ship in this repository (~1.5 MB total for the lesion tables).

### `autopet_i_lesions.parquet`
Per-lesion feature table for AutoPET-I (FDG whole-body, single-vendor Siemens,
single centre Tübingen 2014–2018).

| Column | Type | Description | Units |
|---|---|---|---|
| `case_id` | str | Patient identifier from FDAT | — |
| `study_date` | str | Acquisition date (YYYY-MM-DD) | date |
| `lesion_id` | int | Per-case connected-component label after ≥1 mL filter | — |
| `cancer_type` | str | NEGATIVE / MELANOMA / LUNG_CANCER / LYMPHOMA | — |
| `volume_ml` | float | Lesion volume | mL |
| `surface_area_cm2` | float | Lesion surface area via voxel-face counting | cm² |
| `sphericity` | float | IBSI sphericity = π^(1/3) (6V)^(2/3) / SA | unitless [0, 1] |
| `voxel_volume_ml` | float | Per-voxel volume in mL | mL |
| `voxel_spacing_0/1/2` | float | DICOM voxel spacing per axis | mm |
| `suvmax`, `suvmean`, `suvpeak` | float | Per-lesion SUV statistics (SUVpeak via 1 cm³ sphere) | g/mL |
| `tlg` | float | Total lesion glycolysis (volume × SUVmean) | mL · g/mL |
| `softmax_mean`, `softmax_entropy` | float | Sentinel values 1.0 / 0.0 for ground-truth-segmented cohorts (Amendment 9 §9b) | unitless |
| `section_3_9_excluded` | bool | True if excluded by §3.9 outlier review | — |

### `autopet_iii_lesions_reviewed.parquet`
Per-lesion feature table for AutoPET-III (PSMA whole-body, two vendors at
LMU Munich). Same schema as AutoPET-I, with the following additions:

| Column | Type | Description |
|---|---|---|
| `tracer` | str | "18F-PSMA" or "68Ga-PSMA-11" |
| `vendor` | str | "Siemens" or "GE" |
| `scanner_model` | str | "Biograph_mCT_Flow_20", "Discovery_690", or "Biograph_64-4R_TruePoint" |
| `softmax_mean` | float | nnU-Net mean predicted lesion-class probability |
| `softmax_entropy` | float | Mean Shannon entropy of voxel softmax |

### `hecktor_lesions_reviewed.parquet`
Per-lesion feature table for HECKTOR 2025 (FDG head-and-neck, three vendors
across seven centres CHUM/CHUS/CHUP/CHUV/MDA/USZ/HMR).

| Column | Type | Description |
|---|---|---|
| `centre_id` | int | HECKTOR-encoded centre ID (1, 2, 3, 5, 6, 7, 8; gap at 4 per challenge) |
| `centre_name` | str | "CHUM" / "CHUS" / "CHUP" / "MDA" / "CHUV" / "USZ" / "HMR" |
| `vendor` | str | "Siemens" / "GE" / "Philips" |
| `lesion_class` | str | "GTVp" (primary) / "GTVn" (nodal) / "mixed" (none observed) |

### `autopet_i_splits.parquet`
Patient-level allocation manifest for AutoPET-I.

| Column | Type | Description |
|---|---|---|
| `case_id` | str | Patient identifier |
| `split` | str | "train" (60% / 276 patients) / "cal" (20% / 92) / "test" (20% / 93) / "serial" (held out, unused in Phase 3) |
| `seed` | int | RNG seed used for the allocation (locked = 42) |

Locked at Freeze Gate 1 with seed=42 and SHA-256
`0c4e345c3519bc854df6d2fc3ce8bf83bf8330d1669adbe90b8b99fe84eb6197`.

### `phase2_autopet_iii_primary_wcv.parquet`
Phase 2 within-replicate coefficient of variation (wCV) reference. Per-lesion
× dose-fraction × replicate Poisson-noise-injection results.

| Column | Type | Description | Units |
|---|---|---|---|
| `case_id`, `lesion_id` | str / int | Lesion identifiers (joining to `autopet_iii_lesions_reviewed`) | — |
| `dose_fraction` | float | 0.50 / 0.25 / 0.10 (50% / 25% / 10% dose) | — |
| `replicate` | int | 1..K (K=10 per dose level, locked) | — |
| `suvmax`, `suvpeak`, `suvmean` | float | Per-replicate SUV statistics | g/mL |
| `wcv_suvmax`, `wcv_suvpeak`, `wcv_suvmean` | float | Within-lesion CV across replicates at this dose | unitless [0, 1] |

NEMA NU-2 calibration anchors: Siemens Biograph mCT Flow 20 (9.7 cps/kBq;
Jakoby 2011), Siemens Biograph 64-4R TruePoint (7.0 cps/kBq; Jakoby 2009),
GE Discovery 690 (7.5 cps/kBq; Bettinardi 2011).

## Pre-registered feature schema (CQR base model and WCP classifier)

| Feature | Used by | Notes |
|---|---|---|
| `volume_ml` | CQR base, WCP-image, WCP-extended | Per-lesion volume |
| `surface_area_cm2` | CQR base, WCP-image, WCP-extended | Voxel-face counting |
| `sphericity` | CQR base, WCP-image, WCP-extended | IBSI formula |
| `softmax_mean` | CQR base | Sentinel 1.0 for AutoPET-I and HECKTOR (ground-truth) |
| `softmax_entropy` | CQR base | Sentinel 0.0 for AutoPET-I and HECKTOR (ground-truth) |
| `voxel_volume_ml` | WCP-extended | Acquisition continuous |
| `lesions_per_patient_in_cohort` | WCP-extended | Per-case lesion count |
| `tracer_is_psma` | WCP-extended | Categorical (cohort indicator) |
| `vendor_is_siemens / _ge / _philips` | WCP-extended | One-hot vendor |
| `centre_1` … `centre_8` | WCP-extended | One-hot centre (HECKTOR only) |

The 16-feature WCP-extended classifier is constructed by
`src.conformal.weighted.build_extended_wcp_features`.

## Raw imaging data (not redistributed)

The raw DICOM and segmentation NIfTI archives total ~330 GB and cannot be
redistributed in this repository. To rebuild the lesion parquets from
scratch:

1. **AutoPET-I FDAT.** Register and download from
   <https://doi.org/10.57754/FDAT.wf9fy-txq84>. The 282.9 GB zip contains
   pre-computed SUV NIfTIs and ground-truth segmentation NIfTIs per study.
   Then run `kaggle_notebooks/process_autopet_i.ipynb`.

2. **AutoPET-III TCIA.** Pull DICOM via TCIA REST API for collection
   `PSMA-PET-CT-Lesions` (DOI 10.7937/R7EP-3X37). Then run
   `kaggle_notebooks/preprocess_autopet_iii.ipynb` followed by
   `kaggle_notebooks/process_autopet_iii.ipynb` and
   `kaggle_notebooks/infer_autopet_iii.ipynb` (nnU-Net inference using
   the AutoPET LesionTracer checkpoint, SHA-256
   `29a2b99097666f418b4fb7c50908eb2416158dcc54e7c8fb38d110f0135f49d4`).

3. **HECKTOR 2025.** Obtain the challenge zip
   (SHA-256 `1abcf1d96d38bb3d7b1eaf1889fa8ddd688f14b70876a1c7cf0cd7482d076df2`)
   under the HECKTOR data-use agreement. Then run the lesion-extraction
   cells in `kaggle_notebooks/hecktor_lesion_extraction_cells.txt`.

The locked lesion parquets shipped in this repository under
`data/interim/lesion_tables/` and `data/processed/` allow reproduction of
all Phase 3 results without re-running the full preprocessing chain.
