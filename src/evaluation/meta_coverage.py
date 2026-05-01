"""Nested conformal validation (meta-coverage).

Empirically validates the finite-sample coverage guarantee by repeating
the full conformal pipeline across 200 random resplits.

If conformal is valid under exchangeability, at least 90% of resplits
should achieve ≥90% coverage (the "meta-coverage").

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.5.13)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.conformal.cqr import (
    compute_nonconformity_scores,
    calibrate_conformal_threshold,
    predict_intervals_array,
)

# Pre-registered parameter
N_RESPLITS = 200


@dataclass
class MetaCoverageResult:
    """Result of the nested conformal validation."""

    n_resplits: int
    target_coverage: float  # 1 - alpha
    achieved_coverages: np.ndarray  # one per resplit
    mean_coverage: float
    sd_coverage: float
    meta_coverage_90: float  # fraction of resplits achieving ≥90%
    meta_coverage_85: float  # fraction achieving ≥85% (5pp tolerance)


def run_meta_coverage(
    y: np.ndarray,
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    patient_ids: np.ndarray,
    alpha: float = 0.10,
    n_resplits: int = N_RESPLITS,
    cal_fraction: float = 0.5,
    seed: int = 42,
) -> MetaCoverageResult:
    """Run nested conformal validation across random resplits.

    For each resplit:
    1. Split data into calibration and test (patient-level)
    2. Calibrate conformal threshold on calibration set
    3. Evaluate coverage on test set

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        True values for all data points.
    lower_pred : np.ndarray, shape (n,)
        Lower quantile predictions.
    upper_pred : np.ndarray, shape (n,)
        Upper quantile predictions.
    patient_ids : np.ndarray, shape (n,)
        Patient IDs for patient-level splitting.
    alpha : float
        Miscoverage level.
    n_resplits : int
        Number of random resplits.
    cal_fraction : float
        Fraction of patients allocated to calibration per resplit.
    seed : int
        Random seed.

    Returns
    -------
    MetaCoverageResult
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    n_cal_patients = int(n_patients * cal_fraction)

    target = 1.0 - alpha
    coverages = np.zeros(n_resplits)

    for i in range(n_resplits):
        # Resplit at patient level
        perm = rng.permutation(n_patients)
        cal_patients = set(unique_patients[perm[:n_cal_patients]])

        cal_mask = np.array([p in cal_patients for p in patient_ids])
        test_mask = ~cal_mask

        if cal_mask.sum() == 0 or test_mask.sum() == 0:
            coverages[i] = np.nan
            continue

        # Calibrate
        scores = compute_nonconformity_scores(
            y[cal_mask], lower_pred[cal_mask], upper_pred[cal_mask]
        )
        q_hat = calibrate_conformal_threshold(scores, alpha)

        # Test
        lb, ub, _ = predict_intervals_array(
            lower_pred[test_mask], upper_pred[test_mask], q_hat
        )
        covered = (y[test_mask] >= lb) & (y[test_mask] <= ub)
        coverages[i] = covered.mean()

    valid = coverages[~np.isnan(coverages)]

    return MetaCoverageResult(
        n_resplits=len(valid),
        target_coverage=target,
        achieved_coverages=valid,
        mean_coverage=float(np.mean(valid)),
        sd_coverage=float(np.std(valid)),
        meta_coverage_90=float((valid >= target).mean()),
        meta_coverage_85=float((valid >= target - 0.05).mean()),
    )
