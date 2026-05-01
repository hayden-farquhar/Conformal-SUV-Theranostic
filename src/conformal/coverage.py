"""Coverage diagnostics for conformal prediction intervals.

Computes marginal and conditional coverage, interval efficiency,
and Clopper-Pearson confidence intervals.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.2.4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CoverageResult:
    """Coverage statistics for a set of prediction intervals."""

    label: str  # e.g., "marginal", "Q1", "Siemens_FDG_Q3"
    n: int
    n_covered: int
    coverage: float  # empirical coverage
    nominal: float  # target coverage (1 - alpha)
    coverage_gap: float  # coverage - nominal (positive = overcovers)
    ci_lower: float  # 95% Clopper-Pearson lower
    ci_upper: float  # 95% Clopper-Pearson upper
    median_width: float
    mean_width: float
    width_iqr: tuple[float, float]  # (25th, 75th percentile)
    median_relative_width: float  # width / true value, as percentage


def compute_coverage(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    nominal: float,
    label: str = "marginal",
) -> CoverageResult:
    """Compute coverage and interval efficiency metrics.

    Parameters
    ----------
    y_true : array, shape (n,)
        True values.
    lower_bounds : array, shape (n,)
        Lower prediction bounds.
    upper_bounds : array, shape (n,)
        Upper prediction bounds.
    nominal : float
        Target coverage level (1 - alpha).
    label : str
        Label for this coverage group.

    Returns
    -------
    CoverageResult
    """
    n = len(y_true)
    covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    n_covered = int(covered.sum())
    coverage = n_covered / n if n > 0 else 0.0

    # 95% Clopper-Pearson exact binomial CI
    ci_lower, ci_upper = clopper_pearson_ci(n_covered, n, 0.95)

    # Interval widths
    widths = upper_bounds - lower_bounds

    # Relative widths (as percentage of true value)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_widths = np.where(y_true > 0, widths / y_true * 100, np.nan)
    relative_widths = relative_widths[~np.isnan(relative_widths)]

    return CoverageResult(
        label=label,
        n=n,
        n_covered=n_covered,
        coverage=coverage,
        nominal=nominal,
        coverage_gap=coverage - nominal,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        median_width=float(np.median(widths)) if n > 0 else 0.0,
        mean_width=float(np.mean(widths)) if n > 0 else 0.0,
        width_iqr=(
            float(np.percentile(widths, 25)),
            float(np.percentile(widths, 75)),
        ) if n > 0 else (0.0, 0.0),
        median_relative_width=float(np.median(relative_widths)) if len(relative_widths) > 0 else 0.0,
    )


def compute_conditional_coverage(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    strata: np.ndarray,
    nominal: float,
) -> list[CoverageResult]:
    """Compute per-stratum conditional coverage.

    Parameters
    ----------
    y_true, lower_bounds, upper_bounds : arrays, shape (n,)
    strata : array, shape (n,)
        Stratum labels.
    nominal : float
        Target coverage level.

    Returns
    -------
    list[CoverageResult]
        One per unique stratum, sorted by label.
    """
    results = []
    for stratum in sorted(np.unique(strata)):
        mask = strata == stratum
        results.append(compute_coverage(
            y_true[mask], lower_bounds[mask], upper_bounds[mask],
            nominal, label=str(stratum),
        ))
    return results


def compute_coverage_disparity(results: list[CoverageResult]) -> float:
    """Max coverage - min coverage across strata."""
    coverages = [r.coverage for r in results]
    return max(coverages) - min(coverages) if coverages else 0.0


def clopper_pearson_ci(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval.

    Parameters
    ----------
    k : int
        Number of successes (covered).
    n : int
        Total number of trials.
    confidence : float
        Confidence level.

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds.
    """
    alpha = 1.0 - confidence

    if n == 0:
        return (0.0, 1.0)

    if k == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

    return (float(lower), float(upper))


def coverage_summary_table(
    marginal: CoverageResult,
    conditional: list[CoverageResult],
) -> "pd.DataFrame":
    """Create a summary DataFrame of coverage results.

    Returns
    -------
    pd.DataFrame
        One row per stratum + one row for marginal.
    """
    import pandas as pd

    rows = []
    for r in [marginal] + conditional:
        rows.append({
            "stratum": r.label,
            "n": r.n,
            "coverage": r.coverage,
            "nominal": r.nominal,
            "gap_pp": r.coverage_gap * 100,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "median_width": r.median_width,
            "median_rel_width_pct": r.median_relative_width,
            "within_tolerance": abs(r.coverage_gap) <= 0.02,  # ±2pp for H1
        })

    return pd.DataFrame(rows)
