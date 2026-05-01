"""Literature-typical acquisition parameter defaults for the Phase 2 Poisson-noise
reference. Amendment 6 (2026-04-29) locked the AutoPET-I cohort defaults; Amendment 7
(2026-04-29) adds per-scanner-model `calibration_factor` lookup (replaces Amendment 6
§6a's incorrect `calibration_factor = 1.0` claim).

These values are LOCKED in writing in `osf/amendment_log.md` BEFORE any data flows
through the Phase 2 driver.

Sources:
    - Boellaard R et al. FDG PET/CT: EANM procedure guidelines for tumour imaging:
      version 2.0. EJNMMI 2015; 42(2):328-354. (uptake time, dose-per-weight)
    - Gatidis S et al. A whole-body FDG-PET/CT dataset with manually annotated tumor
      lesions. Sci Data 2022; 9:601. (AutoPET-I cohort: UKT Tubingen Siemens Biograph
      mCT family, ~2 min/bed continuous-bed-motion, 300-400 MBq typical activity)
    - Jakoby BW et al. Physical and clinical performance of the mCT time-of-flight
      PET/CT scanner. Phys Med Biol 2011; 56(8):2375-2389. (Biograph mCT NEMA sens)
    - Jakoby BW et al. Performance characteristics of a new LSO PET/CT scanner with
      extended axial field-of-view and PSF reconstruction. EJNMMI 2009; 36(11):1638-1647.
      (Biograph 64-4R TruePoint NEMA sens)
    - Bettinardi V et al. Physical performance of the new hybrid PET/CT Discovery-690.
      Med Phys 2011; 38(10):5394-5411. (GE Discovery 690 NEMA sens)
    - Lodge MA. Repeatability of SUV in oncologic 18F-FDG PET. J Nucl Med 2017;
      58(4):523-532. (test-retest CV anchor for the calibration target)
    - NIST radionuclide tables (F-18 half-life)

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN sec 3.5
Amendment 6: 2026-04-29 (initial Phase 2 specification)
Amendment 7: 2026-04-29 (calibration_factor correction)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionParams:
    """Per-series acquisition parameters needed by the Poisson-noise pipeline."""
    injected_dose_bq: float
    patient_weight_kg: float
    decay_factor: float
    frame_duration_sec: float
    calibration_factor: float = 1.0


# AutoPET-I cohort defaults (Amendment 6 locked values; calibration_factor updated by Amendment 7)
AUTOPET_I_FDG_DEFAULTS = {
    "dose_per_weight_mbq_per_kg": 4.0,        # Boellaard 2015 EANM v2.0
    "dose_mid_point_mbq_fallback": 350.0,     # Gatidis 2022 cohort-typical for 75-kg patient
    "patient_weight_kg_fallback": 75.0,       # EANM adult oncology working assumption
    "uptake_time_sec": 3600.0,                # 60 min (EANM/SNMMI standard)
    "half_life_sec_f18": 6588.0,              # NIST
    "frame_duration_sec": 120.0,              # 2 min/bed Biograph mCT continuous-bed-motion
    "calibration_factor": 5.0e-4,             # Amendment 7 anchor: Siemens Biograph mCT
                                              # (NEMA sens 9.7 cps/kBq, Jakoby 2011);
                                              # supersedes Amendment 6's incorrect 1.0
}


# Per-scanner-model calibration factors (Amendment 7 locked values).
# Anchor: Siemens Biograph mCT Flow 20 at 5.0e-4 (calibrated so SUV=10 voxel produces
# ~1000 effective counts at 100% dose, giving SUV-domain Poisson SD ~3% per voxel,
# consistent with Lodge 2017 lesion-level test-retest CV ~5-8% after lesion aggregation).
# Other scanners scale by NEMA NU2 sensitivity ratio.
SCANNER_CALIBRATION = {
    "Biograph mCT Flow 20":      5.0e-4,                       # Jakoby 2011: 9.7 cps/kBq (anchor)
    "Biograph 64-4R TruePoint":  5.0e-4 * (7.0 / 9.7),         # Jakoby 2009: 7.0 cps/kBq -> 3.61e-4
    "Discovery 690":             5.0e-4 * (7.5 / 9.7),         # Bettinardi 2011: 7.5 cps/kBq -> 3.87e-4
}
SCANNER_CALIBRATION_DEFAULT = 4.5e-4  # cohort-typical fallback for unrecognised models


def lookup_scanner_calibration(manufacturer_model_name: str | None) -> float:
    """Map DICOM ManufacturerModelName -> Amendment 7 calibration factor.

    Case-insensitive substring match against `SCANNER_CALIBRATION` keys. Unrecognised
    models return `SCANNER_CALIBRATION_DEFAULT` (cohort-typical fallback). Vendor
    coverage in AutoPET-III (per data_snapshot_log.md): Siemens Biograph mCT Flow 20
    (~251 studies), GE Discovery 690 (~230 studies), Siemens Biograph 64-4R TruePoint
    (~116 studies). AutoPET-I (Gatidis 2022): single Siemens Biograph mCT scanner.
    """
    if not isinstance(manufacturer_model_name, str):
        return SCANNER_CALIBRATION_DEFAULT
    needle = manufacturer_model_name.strip().lower()
    if not needle:
        return SCANNER_CALIBRATION_DEFAULT
    for key, value in SCANNER_CALIBRATION.items():
        if key.lower() in needle:
            return value
    return SCANNER_CALIBRATION_DEFAULT


def autopet_i_defaults_for_patient(patient_weight_kg: float | None) -> AcquisitionParams:
    """Materialise AutoPET-I literature defaults for one patient.

    Per-patient weight is preserved when available (from the AutoPET-I clinical
    metadata CSV); only patients with missing weight fall back to the cohort
    average. Dose is computed from weight using the EANM dose-per-weight rule
    (4 MBq/kg) when weight is known; otherwise the cohort-typical mid-point
    (350 MBq) is used. Calibration factor uses the Amendment 7 Siemens Biograph
    mCT anchor (5e-4) -- AutoPET-I is single-scanner per Gatidis 2022.

    Returns
    -------
    AcquisitionParams
        Frozen acquisition parameters, decay_factor pre-computed for F-18 + 60min uptake.
    """
    cfg = AUTOPET_I_FDG_DEFAULTS
    weight_kg = patient_weight_kg if (patient_weight_kg and patient_weight_kg > 0) else cfg["patient_weight_kg_fallback"]
    if patient_weight_kg and patient_weight_kg > 0:
        dose_mbq = cfg["dose_per_weight_mbq_per_kg"] * patient_weight_kg
    else:
        dose_mbq = cfg["dose_mid_point_mbq_fallback"]
    decay_factor = 2.0 ** (-cfg["uptake_time_sec"] / cfg["half_life_sec_f18"])
    return AcquisitionParams(
        injected_dose_bq=dose_mbq * 1e6,
        patient_weight_kg=weight_kg,
        decay_factor=decay_factor,
        frame_duration_sec=cfg["frame_duration_sec"],
        calibration_factor=cfg["calibration_factor"],
    )
