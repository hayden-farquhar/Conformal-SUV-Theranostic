"""Bootstrap confidence intervals for conformal prediction metrics.

Patient-level resampling to preserve within-patient correlation.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.5.10)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pre-registered parameters
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_resamples: int
    se: float  # bootstrap standard error


def patient_level_bootstrap(
    patient_ids: np.ndarray,
    metric_fn: callable,
    n_resamples: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap CI with patient-level resampling.

    Resamples patients (not individual lesions) to respect within-patient
    correlation. All lesions from a resampled patient are included.

    Parameters
    ----------
    patient_ids : np.ndarray, shape (n_lesions,)
        Patient ID for each data point (lesion).
    metric_fn : callable
        Function that takes a boolean index mask (shape n_lesions) and
        returns a scalar metric value. Called once per bootstrap resample.
    n_resamples : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (e.g. 0.95).
    seed : int
        Random seed.

    Returns
    -------
    BootstrapCI
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    # Point estimate (full data)
    all_mask = np.ones(len(patient_ids), dtype=bool)
    point_estimate = metric_fn(all_mask)

    # Bootstrap resamples
    boot_values = np.zeros(n_resamples)
    for b in range(n_resamples):
        # Resample patients with replacement
        resampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)

        # Build mask: include all lesions from resampled patients
        # Handle duplicate patient selections by counting occurrences
        mask = np.zeros(len(patient_ids), dtype=bool)
        for patient in resampled_patients:
            mask |= (patient_ids == patient)

        boot_values[b] = metric_fn(mask)

    # Percentile CI
    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_values, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_values, (1 - alpha / 2) * 100))
    se = float(np.std(boot_values))

    return BootstrapCI(
        metric_name="",
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_resamples=n_resamples,
        se=se,
    )


def bootstrap_coverage(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap CI for empirical coverage."""

    def coverage_fn(mask):
        y = y_true[mask]
        lb = lower_bounds[mask]
        ub = upper_bounds[mask]
        covered = (y >= lb) & (y <= ub)
        return covered.mean() if len(y) > 0 else 0.0

    result = patient_level_bootstrap(
        patient_ids, coverage_fn, n_resamples, seed=seed
    )
    result.metric_name = "coverage"
    return result


def bootstrap_median_width(
    patient_ids: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap CI for median interval width."""
    widths = upper_bounds - lower_bounds

    def width_fn(mask):
        return float(np.median(widths[mask])) if mask.sum() > 0 else 0.0

    result = patient_level_bootstrap(
        patient_ids, width_fn, n_resamples, seed=seed
    )
    result.metric_name = "median_width"
    return result


def bootstrap_indeterminacy_rate(
    patient_ids: np.ndarray,
    decisions: np.ndarray,
    indeterminate_label: str = "indeterminate",
    n_resamples: int = N_BOOTSTRAP,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap CI for indeterminacy rate."""

    def indet_fn(mask):
        d = decisions[mask]
        return (d == indeterminate_label).mean() if len(d) > 0 else 0.0

    result = patient_level_bootstrap(
        patient_ids, indet_fn, n_resamples, seed=seed
    )
    result.metric_name = "indeterminacy_rate"
    return result
