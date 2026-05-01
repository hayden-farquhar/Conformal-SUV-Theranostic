"""Generate the theranostic-eligibility indeterminacy table.

This is the primary clinical deliverable of the study (§5.3.4).
It summarises the proportion of decisions rendered indeterminate by
conformal prediction intervals at each confidence level.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.3.4)
"""

from __future__ import annotations

import pandas as pd

from src.clinical.vision_overlay import (
    apply_vision_overlay,
    compute_vision_indeterminacy_rate,
)
from src.clinical.percist_overlay import (
    apply_percist_overlay,
    compute_percist_indeterminacy_rate,
)


def build_indeterminacy_table(
    vision_results: list[dict],
    percist_results: list[dict] | None = None,
) -> pd.DataFrame:
    """Build the pre-registered indeterminacy table (§5.3.4).

    Parameters
    ----------
    vision_results : list[dict]
        One entry per CI level, each the output of
        compute_vision_indeterminacy_rate().
        Must include a 'ci_level' key (e.g. 0.80, 0.90, 0.95).
    percist_results : list[dict], optional
        Same format for PERCIST.

    Returns
    -------
    pd.DataFrame
        The indeterminacy table with columns: threshold, ci_level,
        n_decisions, eligible_pct, ineligible_pct, indeterminate_pct,
        indeterminacy_ci_lower, indeterminacy_ci_upper.
    """
    rows = []

    for r in vision_results:
        rows.append({
            "threshold": "VISION (PSMA)",
            "ci_level": f"{r['ci_level']*100:.0f}%",
            "ci_level_numeric": r["ci_level"],
            "n_decisions": r["n_total"],
            "eligible_pct": r.get("eligible_pct", 0),
            "ineligible_pct": r.get("ineligible_pct", 0),
            "indeterminate_pct": r["indeterminacy_pct"],
            "indeterminacy_ci_lower": r["indeterminacy_ci_lower"] * 100,
            "indeterminacy_ci_upper": r["indeterminacy_ci_upper"] * 100,
        })

    if percist_results:
        for r in percist_results:
            rows.append({
                "threshold": f"PERCIST {r.get('boundary', '-30%')} (FDG)",
                "ci_level": f"{r['ci_level']*100:.0f}%",
                "ci_level_numeric": r["ci_level"],
                "n_decisions": r["n_total"],
                "eligible_pct": r.get("n_response", 0) / r["n_total"] * 100 if r["n_total"] else 0,
                "ineligible_pct": r.get("n_stable", 0) / r["n_total"] * 100 if r["n_total"] else 0,
                "indeterminate_pct": r["indeterminacy_pct"],
                "indeterminacy_ci_lower": r["indeterminacy_ci_lower"] * 100,
                "indeterminacy_ci_upper": r["indeterminacy_ci_upper"] * 100,
            })

    table = pd.DataFrame(rows)
    table = table.sort_values(["threshold", "ci_level_numeric"]).reset_index(drop=True)

    return table


def format_indeterminacy_table(table: pd.DataFrame) -> str:
    """Format the indeterminacy table for display/manuscript.

    Returns
    -------
    str
        Formatted table string.
    """
    display = table[[
        "threshold", "ci_level", "n_decisions",
        "eligible_pct", "ineligible_pct", "indeterminate_pct",
        "indeterminacy_ci_lower", "indeterminacy_ci_upper",
    ]].copy()

    display.columns = [
        "Threshold", "CI Level", "N Decisions",
        "Eligible (%)", "Ineligible (%)", "Indeterminate (%)",
        "95% CI Lower (%)", "95% CI Upper (%)",
    ]

    # Format percentages
    for col in ["Eligible (%)", "Ineligible (%)", "Indeterminate (%)",
                 "95% CI Lower (%)", "95% CI Upper (%)"]:
        display[col] = display[col].apply(lambda x: f"{x:.1f}")

    return display.to_string(index=False)
