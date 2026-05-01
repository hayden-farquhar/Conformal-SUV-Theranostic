"""Seed stability analysis.

Repeats the primary pipeline across multiple random seeds to assess
robustness of results to random seed choice.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.5.12)
Acceptance criteria: SD(coverage) < 0.02, SD(indeterminacy_rate) < 0.03
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pre-registered seeds (§5.5.12)
SEEDS = [42, 123, 456, 789, 1024]

# Pre-registered acceptance criteria
MAX_SD_COVERAGE = 0.02
MAX_SD_INDETERMINACY = 0.03


@dataclass
class SeedStabilityResult:
    """Result of seed stability analysis."""

    seeds: list[int]
    metric_name: str
    values: list[float]  # one per seed
    mean: float
    sd: float
    passes_criterion: bool
    criterion_threshold: float


def assess_seed_stability(
    metric_values: dict[int, float],
    metric_name: str,
    max_sd: float,
) -> SeedStabilityResult:
    """Assess stability of a metric across seeds.

    Parameters
    ----------
    metric_values : dict[int, float]
        Mapping seed -> metric value.
    metric_name : str
        Name of the metric (e.g. 'coverage', 'indeterminacy_rate').
    max_sd : float
        Maximum acceptable SD (pre-registered criterion).

    Returns
    -------
    SeedStabilityResult
    """
    seeds = sorted(metric_values.keys())
    values = [metric_values[s] for s in seeds]
    arr = np.array(values)

    return SeedStabilityResult(
        seeds=seeds,
        metric_name=metric_name,
        values=values,
        mean=float(arr.mean()),
        sd=float(arr.std()),
        passes_criterion=float(arr.std()) < max_sd,
        criterion_threshold=max_sd,
    )


def run_seed_stability_summary(
    coverage_per_seed: dict[int, float],
    width_per_seed: dict[int, float],
    indeterminacy_per_seed: dict[int, float],
) -> list[SeedStabilityResult]:
    """Run the full pre-registered seed stability assessment.

    Parameters
    ----------
    coverage_per_seed : dict[int, float]
        Coverage at each seed.
    width_per_seed : dict[int, float]
        Median interval width at each seed.
    indeterminacy_per_seed : dict[int, float]
        Indeterminacy rate at each seed.

    Returns
    -------
    list[SeedStabilityResult]
    """
    return [
        assess_seed_stability(coverage_per_seed, "coverage", MAX_SD_COVERAGE),
        assess_seed_stability(width_per_seed, "median_width", np.inf),  # no pre-registered threshold
        assess_seed_stability(indeterminacy_per_seed, "indeterminacy_rate", MAX_SD_INDETERMINACY),
    ]
