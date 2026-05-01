"""Tests for bootstrap CIs, meta-coverage, and seed stability."""

import numpy as np
import pytest

from src.evaluation.bootstrap import (
    patient_level_bootstrap,
    bootstrap_coverage,
    bootstrap_median_width,
    bootstrap_indeterminacy_rate,
)
from src.evaluation.meta_coverage import (
    run_meta_coverage,
)
from src.evaluation.seed_stability import (
    assess_seed_stability,
    run_seed_stability_summary,
    MAX_SD_COVERAGE,
    MAX_SD_INDETERMINACY,
)


# === Bootstrap tests ===


class TestPatientLevelBootstrap:
    def test_deterministic_with_same_seed(self):
        patients = np.array(["P1"] * 5 + ["P2"] * 5 + ["P3"] * 5)
        values = np.random.rand(15)

        def mean_fn(mask):
            return values[mask].mean()

        r1 = patient_level_bootstrap(patients, mean_fn, n_resamples=100, seed=42)
        r2 = patient_level_bootstrap(patients, mean_fn, n_resamples=100, seed=42)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_ci_contains_point_estimate(self):
        patients = np.array(["P1"] * 10 + ["P2"] * 10)
        values = np.ones(20) * 5.0

        def mean_fn(mask):
            return values[mask].mean()

        result = patient_level_bootstrap(patients, mean_fn, n_resamples=500, seed=42)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_wider_ci_with_more_variance(self):
        patients = np.array(["P1"] * 5 + ["P2"] * 5 + ["P3"] * 5)

        # Low variance data
        low_var = np.ones(15) * 10.0
        def low_fn(mask): return low_var[mask].mean()
        r_low = patient_level_bootstrap(patients, low_fn, n_resamples=500, seed=42)

        # High variance data
        high_var = np.array([1.0] * 5 + [10.0] * 5 + [20.0] * 5)
        def high_fn(mask): return high_var[mask].mean()
        r_high = patient_level_bootstrap(patients, high_fn, n_resamples=500, seed=42)

        assert (r_high.ci_upper - r_high.ci_lower) > (r_low.ci_upper - r_low.ci_lower)


class TestBootstrapCoverage:
    def test_perfect_coverage(self):
        patients = np.array(["P1"] * 5 + ["P2"] * 5)
        y = np.ones(10) * 5.0
        lb = np.ones(10) * 3.0
        ub = np.ones(10) * 7.0

        result = bootstrap_coverage(patients, y, lb, ub, n_resamples=100, seed=42)
        assert result.point_estimate == 1.0
        assert result.ci_lower >= 0.95  # should be near 1.0

    def test_zero_coverage(self):
        patients = np.array(["P1"] * 5 + ["P2"] * 5)
        y = np.ones(10) * 50.0  # all outside
        lb = np.ones(10) * 3.0
        ub = np.ones(10) * 7.0

        result = bootstrap_coverage(patients, y, lb, ub, n_resamples=100, seed=42)
        assert result.point_estimate == 0.0


class TestBootstrapWidth:
    def test_known_width(self):
        patients = np.array(["P1"] * 10)
        lb = np.zeros(10)
        ub = np.ones(10) * 4.0

        result = bootstrap_median_width(patients, lb, ub, n_resamples=100, seed=42)
        assert abs(result.point_estimate - 4.0) < 1e-10


class TestBootstrapIndeterminacy:
    def test_known_rate(self):
        patients = np.array(["P1"] * 10)
        decisions = np.array(["eligible"] * 7 + ["indeterminate"] * 3)

        result = bootstrap_indeterminacy_rate(
            patients, decisions, n_resamples=100, seed=42
        )
        assert abs(result.point_estimate - 0.3) < 1e-10


# === Meta-coverage tests ===


class TestMetaCoverage:
    def test_valid_conformal_achieves_meta_coverage(self):
        """Under exchangeability, meta-coverage should be ≥0.90."""
        rng = np.random.RandomState(42)
        n = 600

        # Generate exchangeable data
        patient_ids = np.array([f"P{i}" for i in range(n)])
        y = rng.normal(10.0, 2.0, n)
        lo = y - 1.5 + rng.normal(0, 0.3, n)
        hi = y + 1.5 + rng.normal(0, 0.3, n)

        result = run_meta_coverage(
            y, lo, hi, patient_ids,
            alpha=0.10, n_resplits=50, seed=42,  # fewer resplits for speed
        )

        assert result.mean_coverage >= 0.85
        assert result.meta_coverage_85 >= 0.80  # relaxed for small n_resplits

    def test_result_fields(self):
        rng = np.random.RandomState(42)
        n = 100
        y = rng.normal(5.0, 1.0, n)
        lo = y - 1.0
        hi = y + 1.0
        pids = np.array([f"P{i}" for i in range(n)])

        result = run_meta_coverage(y, lo, hi, pids, n_resplits=10, seed=42)
        assert result.n_resplits == 10
        assert result.target_coverage == 0.90
        assert len(result.achieved_coverages) == 10
        assert 0 <= result.meta_coverage_90 <= 1.0


# === Seed stability tests ===


class TestSeedStability:
    def test_stable_metric_passes(self):
        values = {42: 0.90, 123: 0.91, 456: 0.89, 789: 0.90, 1024: 0.91}
        result = assess_seed_stability(values, "coverage", MAX_SD_COVERAGE)
        assert result.passes_criterion  # SD < 0.02
        assert result.sd < 0.02

    def test_unstable_metric_fails(self):
        values = {42: 0.70, 123: 0.95, 456: 0.80, 789: 0.60, 1024: 0.99}
        result = assess_seed_stability(values, "coverage", MAX_SD_COVERAGE)
        assert not result.passes_criterion

    def test_summary_returns_three_results(self):
        cov = {42: 0.90, 123: 0.91}
        width = {42: 3.0, 123: 3.1}
        indet = {42: 0.15, 123: 0.16}
        results = run_seed_stability_summary(cov, width, indet)
        assert len(results) == 3
        assert results[0].metric_name == "coverage"
        assert results[1].metric_name == "median_width"
        assert results[2].metric_name == "indeterminacy_rate"
