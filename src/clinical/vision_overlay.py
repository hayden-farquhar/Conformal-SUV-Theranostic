"""VISION eligibility overlay with conformal uncertainty.

The VISION trial defined PSMA-positive lesions as those with uptake
visually greater than liver parenchyma. We operationalise this as
a quantitative threshold: lesion SUVmax > liver SUVmean.

With conformal prediction intervals [L, U] for lesion SUVmax, each
lesion is classified as:
    - Eligible (confident):    L > liver_suvmean  (entire CI above threshold)
    - Ineligible (confident):  U < liver_suvmean  (entire CI below threshold)
    - Indeterminate:           L <= liver_suvmean <= U  (CI straddles threshold)

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.3.1)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class VisionDecision(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    INDETERMINATE = "indeterminate"


@dataclass
class VisionResult:
    """Result of VISION eligibility classification for a single lesion."""

    patient_id: str
    lesion_id: int
    suvmax_point: float
    suvmax_lower: float
    suvmax_upper: float
    liver_suvmean: float
    decision: VisionDecision
    margin: float  # distance from threshold (positive = above, negative = below)


def classify_vision_eligibility(
    suvmax_point: float,
    suvmax_lower: float,
    suvmax_upper: float,
    liver_suvmean: float,
) -> VisionDecision:
    """Classify a single lesion's VISION eligibility.

    Parameters
    ----------
    suvmax_point : float
        Point estimate of lesion SUVmax.
    suvmax_lower : float
        Lower bound of conformal prediction interval.
    suvmax_upper : float
        Upper bound of conformal prediction interval.
    liver_suvmean : float
        Reference liver SUVmean (threshold).

    Returns
    -------
    VisionDecision
    """
    if suvmax_lower > liver_suvmean:
        return VisionDecision.ELIGIBLE
    elif suvmax_upper < liver_suvmean:
        return VisionDecision.INELIGIBLE
    else:
        return VisionDecision.INDETERMINATE


def apply_vision_overlay(
    lesion_df: pd.DataFrame,
    liver_suvmean_per_study: dict[str, float],
    ci_lower_col: str = "suvmax_ci_lower",
    ci_upper_col: str = "suvmax_ci_upper",
) -> pd.DataFrame:
    """Apply VISION eligibility classification to all lesions.

    Parameters
    ----------
    lesion_df : pd.DataFrame
        Must have columns: patient_id, lesion_id, study_uid, suvmax,
        and the CI columns.
    liver_suvmean_per_study : dict
        Mapping of study_uid -> liver SUVmean reference value.
    ci_lower_col : str
        Column name for CI lower bound.
    ci_upper_col : str
        Column name for CI upper bound.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: liver_suvmean, vision_decision,
        vision_margin.
    """
    df = lesion_df.copy()

    df["liver_suvmean"] = df["study_uid"].map(liver_suvmean_per_study)

    decisions = []
    margins = []

    for _, row in df.iterrows():
        liver = row["liver_suvmean"]
        if pd.isna(liver):
            decisions.append(None)
            margins.append(None)
            continue

        decision = classify_vision_eligibility(
            suvmax_point=row["suvmax"],
            suvmax_lower=row[ci_lower_col],
            suvmax_upper=row[ci_upper_col],
            liver_suvmean=liver,
        )
        margin = row["suvmax"] - liver
        decisions.append(decision.value)
        margins.append(margin)

    df["vision_decision"] = decisions
    df["vision_margin"] = margins

    return df


def compute_vision_indeterminacy_rate(
    df: pd.DataFrame,
    decision_col: str = "vision_decision",
) -> dict:
    """Compute the VISION indeterminacy rate and 95% CI.

    Parameters
    ----------
    df : pd.DataFrame
        With vision_decision column.

    Returns
    -------
    dict
        Keys: n_total, n_eligible, n_ineligible, n_indeterminate,
        indeterminacy_rate, indeterminacy_ci_lower, indeterminacy_ci_upper
    """
    valid = df[df[decision_col].notna()]
    n_total = len(valid)
    n_eligible = (valid[decision_col] == VisionDecision.ELIGIBLE.value).sum()
    n_ineligible = (valid[decision_col] == VisionDecision.INELIGIBLE.value).sum()
    n_indeterminate = (valid[decision_col] == VisionDecision.INDETERMINATE.value).sum()

    rate = n_indeterminate / n_total if n_total > 0 else 0.0

    # Clopper-Pearson 95% CI
    from scipy import stats
    if n_total > 0:
        ci_lower = stats.beta.ppf(0.025, n_indeterminate, n_total - n_indeterminate + 1)
        ci_upper = stats.beta.ppf(0.975, n_indeterminate + 1, n_total - n_indeterminate)
        if n_indeterminate == 0:
            ci_lower = 0.0
        if n_indeterminate == n_total:
            ci_upper = 1.0
    else:
        ci_lower = ci_upper = 0.0

    return {
        "n_total": n_total,
        "n_eligible": int(n_eligible),
        "n_ineligible": int(n_ineligible),
        "n_indeterminate": int(n_indeterminate),
        "eligible_pct": n_eligible / n_total * 100 if n_total else 0,
        "ineligible_pct": n_ineligible / n_total * 100 if n_total else 0,
        "indeterminacy_rate": rate,
        "indeterminacy_pct": rate * 100,
        "indeterminacy_ci_lower": ci_lower,
        "indeterminacy_ci_upper": ci_upper,
    }
