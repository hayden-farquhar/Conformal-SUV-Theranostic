"""PERCIST response overlay with conformal uncertainty propagation.

PERCIST 1.0 defines partial metabolic response as ≥30% decrease in
SULpeak of the hottest lesion, with absolute decrease ≥0.8 SUL units.

With conformal intervals for SULpeak at baseline [L_b, U_b] and
follow-up [L_f, U_f], the percentage change interval is:

    Conservative (worst-case, assuming independence):
        Lower bound of Δ%: (L_f - U_b) / U_b × 100
        Upper bound of Δ%: (U_f - L_b) / L_b × 100

Each serial pair is classified as:
    - Response (confident):   upper_delta < -30% AND |delta_point| >= 0.8
    - Stable (confident):     lower_delta > -30% AND upper_delta < +30%
    - Progression (confident): lower_delta > +30% AND |delta_point| >= 0.8
    - Indeterminate:          interval straddles -30% or +30%

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.3.2)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


# Pre-registered thresholds (PERCIST 1.0)
RESPONSE_THRESHOLD_PCT = -30.0
PROGRESSION_THRESHOLD_PCT = 30.0
MIN_ABSOLUTE_CHANGE_SUL = 0.8


class PercistDecision(str, Enum):
    RESPONSE = "response"
    STABLE = "stable"
    PROGRESSION = "progression"
    INDETERMINATE = "indeterminate"


@dataclass
class PercistResult:
    """Result of PERCIST response classification for a serial pair."""

    patient_id: str
    baseline_sulpeak: float
    followup_sulpeak: float
    delta_pct_point: float  # point estimate of % change
    delta_pct_lower: float  # lower bound of % change (conservative)
    delta_pct_upper: float  # upper bound of % change (conservative)
    absolute_change: float  # |followup - baseline|
    decision: PercistDecision


def compute_delta_pct_conservative(
    baseline_lower: float,
    baseline_upper: float,
    followup_lower: float,
    followup_upper: float,
) -> tuple[float, float]:
    """Compute worst-case percentage change bounds assuming independence.

    This is conservative: it assumes baseline and follow-up measurement
    errors are uncorrelated (worst case for the delta).

    Parameters
    ----------
    baseline_lower, baseline_upper : float
        Conformal interval for baseline SULpeak.
    followup_lower, followup_upper : float
        Conformal interval for follow-up SULpeak.

    Returns
    -------
    tuple[float, float]
        (delta_pct_lower, delta_pct_upper) — the widest possible
        percentage change interval.
    """
    if baseline_upper <= 0 or baseline_lower <= 0:
        return (-100.0, 100.0)  # degenerate case

    # Most negative delta: smallest follow-up relative to largest baseline
    delta_lower = (followup_lower - baseline_upper) / baseline_upper * 100.0
    # Most positive delta: largest follow-up relative to smallest baseline
    delta_upper = (followup_upper - baseline_lower) / baseline_lower * 100.0

    return (delta_lower, delta_upper)


def compute_delta_pct_correlated(
    baseline_point: float,
    followup_point: float,
    baseline_lower: float,
    baseline_upper: float,
    followup_lower: float,
    followup_upper: float,
    correlation: float,
) -> tuple[float, float]:
    """Compute correlation-adjusted percentage change bounds.

    Uses the delta method to produce tighter intervals when baseline
    and follow-up measurement errors are correlated.

    Pre-registration: §5.5.8

    Parameters
    ----------
    baseline_point, followup_point : float
        Point estimates.
    baseline_lower, baseline_upper, followup_lower, followup_upper : float
        Conformal intervals.
    correlation : float
        Estimated Pearson correlation between measurement errors (from
        test-retest data). Range [0, 1].

    Returns
    -------
    tuple[float, float]
        (delta_pct_lower, delta_pct_upper) — tighter than conservative.
    """
    if baseline_point <= 0:
        return (-100.0, 100.0)

    # Estimate variances from CI widths (CI ≈ point ± 1.645 * sigma for 90%)
    # Using half-width as a proxy for sigma
    sigma_b = (baseline_upper - baseline_lower) / (2 * 1.645)
    sigma_f = (followup_upper - followup_lower) / (2 * 1.645)

    # Delta method variance for ratio f/b:
    # Var(f/b) ≈ (1/b²)[Var(f) + (f/b)² Var(b) - 2(f/b) Cov(f,b)]
    ratio = followup_point / baseline_point
    cov = correlation * sigma_f * sigma_b
    var_ratio = (1.0 / baseline_point ** 2) * (
        sigma_f ** 2 + ratio ** 2 * sigma_b ** 2 - 2 * ratio * cov
    )

    if var_ratio < 0:
        var_ratio = 0.0

    sigma_ratio = np.sqrt(var_ratio)

    # Percentage change = (ratio - 1) * 100
    delta_point = (ratio - 1.0) * 100.0
    delta_sigma = sigma_ratio * 100.0

    # 90% CI using normal approximation
    delta_lower = delta_point - 1.645 * delta_sigma
    delta_upper = delta_point + 1.645 * delta_sigma

    return (delta_lower, delta_upper)


def classify_percist_response(
    delta_pct_lower: float,
    delta_pct_upper: float,
    delta_pct_point: float,
    absolute_change: float,
) -> PercistDecision:
    """Classify a serial pair's PERCIST response.

    Parameters
    ----------
    delta_pct_lower : float
        Lower bound of percentage change interval.
    delta_pct_upper : float
        Upper bound of percentage change interval.
    delta_pct_point : float
        Point estimate of percentage change.
    absolute_change : float
        |follow-up SULpeak - baseline SULpeak|.

    Returns
    -------
    PercistDecision
    """
    # Check if interval is entirely below response threshold
    if delta_pct_upper < RESPONSE_THRESHOLD_PCT and absolute_change >= MIN_ABSOLUTE_CHANGE_SUL:
        return PercistDecision.RESPONSE

    # Check if interval is entirely above progression threshold
    if delta_pct_lower > PROGRESSION_THRESHOLD_PCT and absolute_change >= MIN_ABSOLUTE_CHANGE_SUL:
        return PercistDecision.PROGRESSION

    # Check if interval is entirely within stable range
    if delta_pct_lower > RESPONSE_THRESHOLD_PCT and delta_pct_upper < PROGRESSION_THRESHOLD_PCT:
        return PercistDecision.STABLE

    # Interval straddles at least one threshold
    return PercistDecision.INDETERMINATE


def apply_percist_overlay(
    serial_pairs_df: pd.DataFrame,
    method: str = "conservative",
    correlation: float = 0.0,
) -> pd.DataFrame:
    """Apply PERCIST response classification to serial scan pairs.

    Parameters
    ----------
    serial_pairs_df : pd.DataFrame
        Must have columns: patient_id, baseline_sulpeak, followup_sulpeak,
        baseline_ci_lower, baseline_ci_upper,
        followup_ci_lower, followup_ci_upper.
    method : str
        'conservative' (default) or 'correlated'.
    correlation : float
        Only used if method='correlated'. Estimated measurement error
        correlation from test-retest data.

    Returns
    -------
    pd.DataFrame
        With added columns: delta_pct_point, delta_pct_lower,
        delta_pct_upper, absolute_change, percist_decision.
    """
    df = serial_pairs_df.copy()

    results = []
    for _, row in df.iterrows():
        bl = row["baseline_sulpeak"]
        fu = row["followup_sulpeak"]

        delta_point = (fu - bl) / bl * 100.0 if bl > 0 else 0.0
        abs_change = abs(fu - bl)

        if method == "correlated" and correlation > 0:
            delta_lower, delta_upper = compute_delta_pct_correlated(
                baseline_point=bl,
                followup_point=fu,
                baseline_lower=row["baseline_ci_lower"],
                baseline_upper=row["baseline_ci_upper"],
                followup_lower=row["followup_ci_lower"],
                followup_upper=row["followup_ci_upper"],
                correlation=correlation,
            )
        else:
            delta_lower, delta_upper = compute_delta_pct_conservative(
                baseline_lower=row["baseline_ci_lower"],
                baseline_upper=row["baseline_ci_upper"],
                followup_lower=row["followup_ci_lower"],
                followup_upper=row["followup_ci_upper"],
            )

        decision = classify_percist_response(
            delta_pct_lower=delta_lower,
            delta_pct_upper=delta_upper,
            delta_pct_point=delta_point,
            absolute_change=abs_change,
        )

        results.append({
            "delta_pct_point": delta_point,
            "delta_pct_lower": delta_lower,
            "delta_pct_upper": delta_upper,
            "absolute_change": abs_change,
            "percist_decision": decision.value,
        })

    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)


def compute_percist_indeterminacy_rate(
    df: pd.DataFrame,
    decision_col: str = "percist_decision",
) -> dict:
    """Compute the PERCIST indeterminacy rate with 95% Clopper-Pearson CI."""
    from scipy import stats

    valid = df[df[decision_col].notna()]
    n = len(valid)
    n_resp = (valid[decision_col] == PercistDecision.RESPONSE.value).sum()
    n_stable = (valid[decision_col] == PercistDecision.STABLE.value).sum()
    n_prog = (valid[decision_col] == PercistDecision.PROGRESSION.value).sum()
    n_indet = (valid[decision_col] == PercistDecision.INDETERMINATE.value).sum()

    rate = n_indet / n if n > 0 else 0.0

    if n > 0:
        ci_lower = stats.beta.ppf(0.025, n_indet, n - n_indet + 1) if n_indet > 0 else 0.0
        ci_upper = stats.beta.ppf(0.975, n_indet + 1, n - n_indet) if n_indet < n else 1.0
    else:
        ci_lower = ci_upper = 0.0

    return {
        "n_total": n,
        "n_response": int(n_resp),
        "n_stable": int(n_stable),
        "n_progression": int(n_prog),
        "n_indeterminate": int(n_indet),
        "indeterminacy_rate": rate,
        "indeterminacy_pct": rate * 100,
        "indeterminacy_ci_lower": ci_lower,
        "indeterminacy_ci_upper": ci_upper,
    }
