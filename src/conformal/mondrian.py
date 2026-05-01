"""Mondrian conformal prediction for group-conditional coverage.

Extends CQR by computing separate conformal thresholds per stratum,
guaranteeing coverage within each group (not just marginally).

Pre-registered stratification (§4.3):
    tracer × vendor × lesion_volume_quartile

Minimum stratum size: 30 (pre-registered §4.3).
Merging rules are pre-specified in the registration.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§4.3, §5.2.3)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.conformal.cqr import (
    CQRCalibration,
    calibrate_cqr,
    predict_intervals_array,
)

# Pre-registered minimum stratum size
MIN_STRATUM_SIZE = 30


@dataclass
class MondrianCalibration:
    """Stores per-stratum calibration results."""

    alpha: float
    strata: dict[str, CQRCalibration]  # stratum_name -> calibration
    marginal: CQRCalibration  # marginal (unstratified) for comparison
    merged_strata: dict[str, str]  # original -> merged stratum name


def assign_volume_quartiles(volumes: np.ndarray | pd.Series) -> np.ndarray:
    """Assign lesion volumes to quartiles Q1-Q4.

    Parameters
    ----------
    volumes : array-like
        Lesion volumes in mL.

    Returns
    -------
    np.ndarray of str
        Quartile labels: 'Q1', 'Q2', 'Q3', 'Q4'.
    """
    volumes = np.asarray(volumes)
    quartiles = np.digitize(
        volumes,
        bins=np.percentile(volumes, [25, 50, 75]),
        right=True,
    )
    labels = np.array(["Q1", "Q2", "Q3", "Q4"])
    return labels[quartiles]


def build_mondrian_strata(
    df: pd.DataFrame,
    tracer_col: str = "tracer_category",
    vendor_col: str = "vendor",
    volume_col: str = "volume_ml",
) -> pd.Series:
    """Build Mondrian stratum labels from lesion features.

    Creates: tracer × vendor × volume_quartile
    e.g. "FDG_Siemens_Q1", "PSMA_GE_Q3"

    Parameters
    ----------
    df : pd.DataFrame
        Lesion feature table with tracer, vendor, and volume columns.

    Returns
    -------
    pd.Series
        Stratum labels, same index as df.
    """
    vq = assign_volume_quartiles(df[volume_col])
    tracer = df[tracer_col].fillna("Unknown").astype(str)
    vendor = df[vendor_col].fillna("Unknown").astype(str)
    return tracer + "_" + vendor + "_" + pd.Series(vq, index=df.index)


def merge_small_strata(
    strata: pd.Series,
    min_size: int = MIN_STRATUM_SIZE,
) -> tuple[pd.Series, dict[str, str]]:
    """Merge strata below minimum size per pre-registered rules (§4.3).

    Rules:
    1. Volume quartiles: merge adjacent (Q1+Q2 or Q3+Q4)
    2. Vendor: merge non-dominant into "Other"
    3. Tracer: never merged

    Parameters
    ----------
    strata : pd.Series
        Original stratum labels.
    min_size : int
        Minimum stratum size.

    Returns
    -------
    tuple[pd.Series, dict]
        (merged_strata, merge_map) where merge_map shows original -> merged.
    """
    counts = strata.value_counts()
    small = counts[counts < min_size].index.tolist()

    if not small:
        return strata, {}

    merge_map = {}
    merged = strata.copy()

    for stratum in small:
        parts = stratum.split("_")
        if len(parts) != 3:
            continue

        tracer, vendor, quartile = parts

        # Rule 1: merge adjacent volume quartiles
        if quartile in ("Q1", "Q2"):
            partner_q = "Q2" if quartile == "Q1" else "Q1"
            partner = f"{tracer}_{vendor}_{partner_q}"
            merged_name = f"{tracer}_{vendor}_Q1Q2"
        elif quartile in ("Q3", "Q4"):
            partner_q = "Q4" if quartile == "Q3" else "Q3"
            partner = f"{tracer}_{vendor}_{partner_q}"
            merged_name = f"{tracer}_{vendor}_Q3Q4"
        else:
            # Already merged quartile label
            continue

        merge_map[stratum] = merged_name
        if partner in merge_map:
            pass  # partner already being merged
        elif partner in small:
            merge_map[partner] = merged_name

        merged = merged.replace(stratum, merged_name)
        if partner in small:
            merged = merged.replace(partner, merged_name)

    # Rule 2: check vendor merging for still-small strata
    counts_after = merged.value_counts()
    still_small = counts_after[counts_after < min_size].index.tolist()

    for stratum in still_small:
        parts = stratum.split("_")
        if len(parts) < 2:
            continue
        tracer = parts[0]
        rest = "_".join(parts[1:])
        # Merge vendor into "Other" while keeping tracer and quartile
        quartile_part = parts[-1] if parts[-1].startswith("Q") else ""
        merged_name = f"{tracer}_Other_{quartile_part}" if quartile_part else f"{tracer}_Other"
        merge_map[stratum] = merged_name
        merged = merged.replace(stratum, merged_name)

    return merged, merge_map


def calibrate_mondrian(
    y_cal: np.ndarray,
    lower_pred_cal: np.ndarray,
    upper_pred_cal: np.ndarray,
    strata_cal: np.ndarray | pd.Series,
    alpha: float,
) -> MondrianCalibration:
    """Calibrate CQR per Mondrian stratum.

    Parameters
    ----------
    y_cal : array, shape (n,)
        True values on calibration set.
    lower_pred_cal : array, shape (n,)
        Lower quantile predictions.
    upper_pred_cal : array, shape (n,)
        Upper quantile predictions.
    strata_cal : array-like, shape (n,)
        Stratum labels for each calibration point.
    alpha : float
        Miscoverage level.

    Returns
    -------
    MondrianCalibration
    """
    strata_cal = np.asarray(strata_cal)

    # Marginal calibration (for comparison)
    marginal = calibrate_cqr(y_cal, lower_pred_cal, upper_pred_cal, alpha, "marginal")

    # Per-stratum calibration
    unique_strata = np.unique(strata_cal)
    strata_calibrations = {}

    for stratum in unique_strata:
        mask = strata_cal == stratum
        if mask.sum() < MIN_STRATUM_SIZE:
            # Use marginal q_hat for small strata (pre-registered §4.3 rule 4)
            strata_calibrations[stratum] = CQRCalibration(
                alpha=alpha,
                stratum=stratum,
                q_hat=marginal.q_hat,
                n_calibration=int(mask.sum()),
                scores=np.array([]),
            )
        else:
            strata_calibrations[stratum] = calibrate_cqr(
                y_cal[mask],
                lower_pred_cal[mask],
                upper_pred_cal[mask],
                alpha,
                stratum,
            )

    return MondrianCalibration(
        alpha=alpha,
        strata=strata_calibrations,
        marginal=marginal,
        merged_strata={},
    )


def predict_mondrian_intervals(
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    strata_test: np.ndarray | pd.Series,
    calibration: MondrianCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Mondrian-calibrated intervals to test data.

    Each test point gets the q̂ from its stratum.

    Parameters
    ----------
    lower_pred : array, shape (n,)
        Lower quantile predictions on test data.
    upper_pred : array, shape (n,)
        Upper quantile predictions on test data.
    strata_test : array-like, shape (n,)
        Stratum labels for test data.
    calibration : MondrianCalibration
        From calibrate_mondrian().

    Returns
    -------
    tuple[lower_bounds, upper_bounds, widths]
    """
    strata_test = np.asarray(strata_test)
    n = len(strata_test)

    lower_bounds = np.zeros(n)
    upper_bounds = np.zeros(n)

    for i, stratum in enumerate(strata_test):
        if stratum in calibration.strata:
            q_hat = calibration.strata[stratum].q_hat
        else:
            # Unseen stratum at test time — fall back to marginal
            q_hat = calibration.marginal.q_hat

        lower_bounds[i] = lower_pred[i] - q_hat
        upper_bounds[i] = upper_pred[i] + q_hat

    widths = upper_bounds - lower_bounds
    return lower_bounds, upper_bounds, widths
