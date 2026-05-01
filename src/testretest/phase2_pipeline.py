"""Per-case Poisson-noise Phase 2 pipeline (Amendment 6 implementation).

Given a SUV volume + segmentation mask + acquisition parameters, generates
K replicates of noisy SUV at each pre-registered dose level (0.50, 0.25, 0.10),
re-extracts per-lesion features inside each fixed lesion mask, and returns
a tidy long-form table with one row per (lesion, dose, replicate).

The wCV reduction (across replicates per lesion per dose) is left to the caller
so the same pipeline can produce both the raw replicate trace (for sensitivity
analyses) and the wCV summary.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN sec 3.5
Amendment 6: 2026-04-29
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from src.features.suvpeak import compute_suvpeak
from src.testretest.defaults import AcquisitionParams
from src.testretest.poisson_noise import (
    DOSE_REDUCTION_LEVELS,
    N_REPLICATES,
    counts_to_suv,
    inject_poisson_noise,
    suv_to_counts,
)

# Pre-registered min lesion volume (sec 3.2)
MIN_LESION_VOLUME_ML = 1.0


@dataclass
class CaseInputs:
    """Per-case inputs to the Phase 2 pipeline."""
    case_id: str
    series_uid: str
    cohort: str  # 'autopet_i' or 'autopet_iii'
    params_source: str  # 'dicom' or 'defaults'
    suv_volume: np.ndarray  # 3D float SUV (z, y, x)
    seg_mask: np.ndarray  # 3D int/bool segmentation mask, same shape as suv_volume
    voxel_spacing_mm: tuple[float, float, float]  # (z, y, x) in mm
    params: AcquisitionParams


def _label_lesions_above_threshold(
    seg_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    min_volume_ml: float = MIN_LESION_VOLUME_ML,
) -> tuple[np.ndarray, list[int]]:
    """Connected-component label the seg mask; keep only lesions >= min_volume_ml.

    Returns
    -------
    labelled : np.ndarray
        Same shape as seg_mask. Each retained lesion has a unique integer label;
        excluded components and background are 0.
    kept_labels : list[int]
        Sorted list of retained lesion labels (matches lesion_id in the existing parquet).
    """
    binary = (seg_mask > 0).astype(np.int32)
    labelled, n_labels = ndimage.label(binary)
    voxel_volume_ml = float(np.prod(voxel_spacing_mm)) / 1000.0

    out = np.zeros_like(labelled, dtype=np.int32)
    kept = []
    for lbl in range(1, n_labels + 1):
        comp = labelled == lbl
        n_voxels = int(comp.sum())
        if n_voxels * voxel_volume_ml >= min_volume_ml:
            out[comp] = lbl
            kept.append(lbl)
    return out, kept


def _extract_suv_stats_for_lesion(
    suv_volume: np.ndarray,
    lesion_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
) -> dict:
    """SUVmax / SUVpeak / SUVmean inside a fixed lesion mask, on a (possibly noisy) SUV volume."""
    inside = suv_volume[lesion_mask]
    suvmax = float(inside.max())
    suvmean = float(inside.mean())
    suvpeak, _is_fallback = compute_suvpeak(suv_volume, lesion_mask, voxel_spacing_mm)
    return {"suvmax": suvmax, "suvmean": suvmean, "suvpeak": float(suvpeak)}


def run_case(
    inputs: CaseInputs,
    dose_fractions: list[float] = None,
    n_replicates: int = N_REPLICATES,
    seed: int = 42,
    min_volume_ml: float = MIN_LESION_VOLUME_ML,
) -> pd.DataFrame:
    """Run Phase 2 noise injection on one case; return long-form replicate trace.

    Per-case workflow:
      1. Label lesions in the segmentation mask (>= min_volume_ml).
      2. Extract baseline SUV stats per lesion (no noise) -- recorded as replicate=0.
      3. For each dose fraction, generate n_replicates noisy SUV volumes.
      4. For each replicate, re-extract SUV stats inside every fixed lesion mask.

    Lesion masks are FIXED across replicates (frozen segmentation). The Poisson-noise
    fallback simulates measurement noise in the SUV intensities, not segmentation
    drift. This is the pre-reg sec 3.5 design intent: the wCV reference captures
    intensity-quantification variability, not boundary variability.

    Returns
    -------
    pd.DataFrame
        Columns: case_id, series_uid, cohort, params_source, lesion_id,
                 dose_fraction, replicate, suvmax, suvpeak, suvmean.
        replicate=0 is the noise-free baseline (dose_fraction=1.0).
    """
    if dose_fractions is None:
        dose_fractions = list(DOSE_REDUCTION_LEVELS)

    labelled, kept_labels = _label_lesions_above_threshold(
        inputs.seg_mask, inputs.voxel_spacing_mm, min_volume_ml=min_volume_ml
    )
    if not kept_labels:
        return pd.DataFrame(
            columns=[
                "case_id", "series_uid", "cohort", "params_source",
                "lesion_id", "dose_fraction", "replicate",
                "suvmax", "suvpeak", "suvmean",
            ]
        )

    rng = np.random.RandomState(seed)
    rows: list[dict] = []

    # Baseline (replicate=0, dose_fraction=1.0): the noise-free reference SUV stats.
    for lbl in kept_labels:
        lesion_mask = labelled == lbl
        stats = _extract_suv_stats_for_lesion(inputs.suv_volume, lesion_mask, inputs.voxel_spacing_mm)
        rows.append({
            "case_id": inputs.case_id, "series_uid": inputs.series_uid,
            "cohort": inputs.cohort, "params_source": inputs.params_source,
            "lesion_id": lbl, "dose_fraction": 1.0, "replicate": 0,
            **stats,
        })

    counts_full_dose = suv_to_counts(
        inputs.suv_volume,
        injected_dose_bq=inputs.params.injected_dose_bq,
        patient_weight_kg=inputs.params.patient_weight_kg,
        decay_factor=inputs.params.decay_factor,
        frame_duration_sec=inputs.params.frame_duration_sec,
        calibration_factor=inputs.params.calibration_factor,
    )

    for frac in dose_fractions:
        for rep in range(1, n_replicates + 1):
            noisy_counts = inject_poisson_noise(counts_full_dose, frac, rng=rng)
            noisy_suv = counts_to_suv(
                noisy_counts,
                injected_dose_bq=inputs.params.injected_dose_bq,
                patient_weight_kg=inputs.params.patient_weight_kg,
                decay_factor=inputs.params.decay_factor,
                frame_duration_sec=inputs.params.frame_duration_sec,
                dose_fraction=frac,
                calibration_factor=inputs.params.calibration_factor,
            )
            for lbl in kept_labels:
                lesion_mask = labelled == lbl
                stats = _extract_suv_stats_for_lesion(noisy_suv, lesion_mask, inputs.voxel_spacing_mm)
                rows.append({
                    "case_id": inputs.case_id, "series_uid": inputs.series_uid,
                    "cohort": inputs.cohort, "params_source": inputs.params_source,
                    "lesion_id": lbl, "dose_fraction": frac, "replicate": rep,
                    **stats,
                })

    return pd.DataFrame(rows)


def reduce_to_wcv(replicate_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a long-form replicate trace to per-lesion per-dose wCV summaries.

    For each (case_id, lesion_id, dose_fraction), compute the within-lesion
    coefficient of variation across replicates for SUVmax / SUVpeak / SUVmean.
    The replicate=0 baseline (dose_fraction=1.0) is excluded from the variance
    calculation but its values are joined back as `*_baseline` for reference.

    Returns
    -------
    pd.DataFrame
        One row per (case_id, lesion_id, dose_fraction) with columns:
        case_id, series_uid, cohort, params_source, lesion_id, dose_fraction,
        n_replicates, wcv_suvmax_pct, wcv_suvpeak_pct, wcv_suvmean_pct,
        suvmax_baseline, suvpeak_baseline, suvmean_baseline,
        suvmax_mean_replicates, suvpeak_mean_replicates, suvmean_mean_replicates.
    """
    if replicate_df.empty:
        return replicate_df.copy()

    baseline_mask = replicate_df["replicate"] == 0
    baseline = (
        replicate_df.loc[baseline_mask, ["case_id", "lesion_id", "suvmax", "suvpeak", "suvmean"]]
        .rename(columns={
            "suvmax": "suvmax_baseline",
            "suvpeak": "suvpeak_baseline",
            "suvmean": "suvmean_baseline",
        })
    )

    reps = replicate_df.loc[~baseline_mask].copy()

    def _wcv(values: pd.Series) -> float:
        arr = values.to_numpy(dtype=float)
        m = arr.mean()
        if m <= 0:
            return float("nan")
        return float(arr.std(ddof=1) / m * 100.0)

    summary = (
        reps.groupby(
            ["case_id", "series_uid", "cohort", "params_source", "lesion_id", "dose_fraction"],
            as_index=False,
        )
        .agg(
            n_replicates=("replicate", "count"),
            wcv_suvmax_pct=("suvmax", _wcv),
            wcv_suvpeak_pct=("suvpeak", _wcv),
            wcv_suvmean_pct=("suvmean", _wcv),
            suvmax_mean_replicates=("suvmax", "mean"),
            suvpeak_mean_replicates=("suvpeak", "mean"),
            suvmean_mean_replicates=("suvmean", "mean"),
        )
    )
    return summary.merge(baseline, on=["case_id", "lesion_id"], how="left")
