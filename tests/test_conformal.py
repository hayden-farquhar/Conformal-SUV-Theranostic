"""Tests for CQR, Mondrian stratification, and coverage diagnostics.

Uses synthetic data to verify the conformal coverage guarantee holds
in finite samples under exchangeability.
"""

import numpy as np
import pandas as pd
import pytest

from src.conformal.cqr import (
    compute_nonconformity_scores,
    calibrate_conformal_threshold,
    calibrate_cqr,
    predict_intervals_array,
)
from src.conformal.mondrian import (
    assign_volume_quartiles,
    build_mondrian_strata,
    merge_small_strata,
    calibrate_mondrian,
    predict_mondrian_intervals,
    MIN_STRATUM_SIZE,
)
from src.conformal.coverage import (
    compute_coverage,
    compute_conditional_coverage,
    compute_coverage_disparity,
    clopper_pearson_ci,
    coverage_summary_table,
)


# === Helpers ===

def _make_synthetic_data(n: int, seed: int = 42):
    """Generate synthetic SUV-like data with heteroscedastic noise.

    Mimics the key property of real PET data: small lesions (low volume)
    have higher SUV measurement variance than large lesions.
    """
    rng = np.random.RandomState(seed)
    volumes = rng.exponential(scale=10.0, size=n) + 1.0  # mL, minimum 1
    true_suv = 5.0 + 0.5 * np.log(volumes) + rng.normal(0, 0.5, size=n)
    # Heteroscedastic noise: smaller volume -> more noise
    noise_scale = 2.0 / np.sqrt(volumes)
    noise = rng.normal(0, noise_scale, size=n)
    observed_suv = true_suv + noise
    return volumes, true_suv, observed_suv


# === CQR core tests ===

class TestNonconformityScores:
    def test_perfect_prediction(self):
        """If y is inside [lower, upper], score should be negative."""
        y = np.array([5.0])
        lo = np.array([3.0])
        hi = np.array([7.0])
        scores = compute_nonconformity_scores(y, lo, hi)
        assert scores[0] < 0  # max(3-5, 5-7) = max(-2, -2) = -2

    def test_undercoverage(self):
        """If y is outside [lower, upper], score should be positive."""
        y = np.array([10.0])
        lo = np.array([3.0])
        hi = np.array([7.0])
        scores = compute_nonconformity_scores(y, lo, hi)
        assert scores[0] > 0  # max(3-10, 10-7) = max(-7, 3) = 3

    def test_exact_boundary(self):
        """If y equals boundary, score should be zero."""
        y = np.array([7.0])
        lo = np.array([3.0])
        hi = np.array([7.0])
        scores = compute_nonconformity_scores(y, lo, hi)
        assert abs(scores[0]) < 1e-10

    def test_symmetric(self):
        """Scores should be symmetric for symmetric deviations."""
        y_above = np.array([8.0])
        y_below = np.array([2.0])
        lo = np.array([3.0])
        hi = np.array([7.0])
        s_above = compute_nonconformity_scores(y_above, lo, hi)
        s_below = compute_nonconformity_scores(y_below, lo, hi)
        assert abs(s_above[0] - s_below[0]) < 1e-10


class TestConformalThreshold:
    def test_finite_sample_correction(self):
        """Threshold should use ceil((n+1)(1-alpha))/n quantile."""
        scores = np.arange(10, dtype=float)  # [0, 1, ..., 9]
        alpha = 0.10
        q = calibrate_conformal_threshold(scores, alpha)
        # ceil((10+1)*0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0 quantile
        # 1.0 quantile of [0..9] = 9.0
        assert abs(q - 9.0) < 1e-10

    def test_alpha_05(self):
        """At alpha=0.05, should be close to the 95th percentile."""
        rng = np.random.RandomState(42)
        scores = rng.randn(1000)
        q = calibrate_conformal_threshold(scores, 0.05)
        # Should be near the 95.1th percentile (finite correction on n=1000)
        assert q > np.percentile(scores, 94)
        assert q < np.percentile(scores, 97)

    def test_small_calibration_set(self):
        """With very few calibration points, correction should be larger."""
        scores = np.array([1.0, 2.0, 3.0])
        q_10 = calibrate_conformal_threshold(scores, 0.10)
        # ceil((3+1)*0.9)/3 = ceil(3.6)/3 = 4/3 ≈ 1.33
        # Capped at 1.0 -> max of scores = 3.0
        assert abs(q_10 - 3.0) < 1e-10


class TestCQRCalibration:
    def test_calibration_stores_correct_fields(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        hi = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        cal = calibrate_cqr(y, lo, hi, alpha=0.10)
        assert cal.alpha == 0.10
        assert cal.stratum == "all"
        assert cal.n_calibration == 5
        assert len(cal.scores) == 5


class TestCQRCoverage:
    """The key test: verify that CQR achieves ≥(1-α) coverage on
    synthetic exchangeable data."""

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_marginal_coverage_guarantee(self, alpha):
        """CQR should achieve at least (1-α) coverage on exchangeable data.

        We use the raw quantile predictions as a stand-in for trained
        models, adding known noise to simulate imperfect quantile estimation.
        """
        n_cal = 500
        n_test = 500
        rng = np.random.RandomState(42)

        # Generate exchangeable data
        y = rng.normal(10.0, 3.0, size=n_cal + n_test)
        # Simulate quantile predictions with some error
        lower_pred = y - 3.0 + rng.normal(0, 0.5, n_cal + n_test)
        upper_pred = y + 3.0 + rng.normal(0, 0.5, n_cal + n_test)

        y_cal, y_test = y[:n_cal], y[n_cal:]
        lo_cal, lo_test = lower_pred[:n_cal], lower_pred[n_cal:]
        hi_cal, hi_test = upper_pred[:n_cal], upper_pred[n_cal:]

        # Calibrate
        cal = calibrate_cqr(y_cal, lo_cal, hi_cal, alpha)

        # Predict
        lb, ub, _ = predict_intervals_array(lo_test, hi_test, cal.q_hat)

        # Check coverage
        covered = (y_test >= lb) & (y_test <= ub)
        empirical_coverage = covered.mean()

        # Should achieve at least (1-α) coverage
        # Allow 2pp tolerance for finite-sample variance
        assert empirical_coverage >= (1 - alpha) - 0.02, (
            f"Coverage {empirical_coverage:.3f} < {1-alpha:.3f} - 0.02"
        )


# === Mondrian tests ===

class TestVolumeQuartiles:
    def test_quartile_assignment(self):
        volumes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        q = assign_volume_quartiles(volumes)
        assert set(q) == {"Q1", "Q2", "Q3", "Q4"}
        # Smallest volumes should be Q1
        assert q[0] == "Q1"
        # Largest should be Q4
        assert q[-1] == "Q4"

    def test_all_same_volume(self):
        """If all volumes are equal, all should be in the same quartile."""
        volumes = np.ones(20) * 5.0
        q = assign_volume_quartiles(volumes)
        assert len(set(q)) == 1  # all in one quartile


class TestMondrianStrata:
    def test_build_strata(self):
        df = pd.DataFrame({
            "tracer_category": ["FDG"] * 8 + ["PSMA"] * 8,
            "vendor": (["Siemens"] * 4 + ["GE"] * 4) * 2,
            "volume_ml": list(range(1, 17)),
        })
        strata = build_mondrian_strata(df)
        # Should have tracer_vendor_quartile format
        assert all("_" in s for s in strata)
        assert any("FDG" in s for s in strata)
        assert any("PSMA" in s for s in strata)

    def test_merge_small_strata(self):
        """Strata below MIN_STRATUM_SIZE should be merged."""
        # Create strata where some have very few members
        strata = pd.Series(
            ["FDG_Siemens_Q1"] * 50
            + ["FDG_Siemens_Q2"] * 50
            + ["FDG_GE_Q1"] * 10  # too small
            + ["FDG_GE_Q2"] * 10  # too small
        )
        merged, merge_map = merge_small_strata(strata, min_size=30)
        # The two small GE strata should be merged
        assert "FDG_GE_Q1" in merge_map or "FDG_GE_Q2" in merge_map
        # Large strata should be unchanged
        assert (merged == "FDG_Siemens_Q1").sum() == 50


class TestMondrianCalibration:
    def test_per_stratum_coverage(self):
        """Mondrian should achieve coverage within each stratum.

        Uses a large sample and well-specified quantile regressors
        to ensure the guarantee holds empirically.
        """
        rng = np.random.RandomState(42)
        n = 2000
        alpha = 0.10

        # Two strata with different noise levels
        strata = np.array(["small"] * (n // 2) + ["large"] * (n // 2))
        y = rng.normal(10.0, 1.0, n)
        noise_scale = np.where(strata == "small", 2.0, 0.5)
        noise = rng.normal(0, noise_scale, n)
        y_observed = y + noise

        # Simulate reasonable quantile predictions (intentionally imperfect)
        lo = y_observed - 1.5 * noise_scale + rng.normal(0, 0.2, n)
        hi = y_observed + 1.5 * noise_scale + rng.normal(0, 0.2, n)

        # Split cal (first half) / test (second half) within each stratum
        cal_mask = np.zeros(n, dtype=bool)
        for s in ["small", "large"]:
            s_idx = np.where(strata == s)[0]
            cal_mask[s_idx[:len(s_idx) // 2]] = True

        mondrian_cal = calibrate_mondrian(
            y[cal_mask], lo[cal_mask], hi[cal_mask],
            strata[cal_mask], alpha,
        )

        test_mask = ~cal_mask
        lb, ub, _ = predict_mondrian_intervals(
            lo[test_mask], hi[test_mask],
            strata[test_mask], mondrian_cal,
        )

        for stratum in ["small", "large"]:
            s_mask = strata[test_mask] == stratum
            covered = (y[test_mask][s_mask] >= lb[s_mask]) & (y[test_mask][s_mask] <= ub[s_mask])
            cov = covered.mean()
            assert cov >= (1 - alpha) - 0.05, (
                f"Stratum '{stratum}' coverage {cov:.3f} too low"
            )

    def test_unseen_stratum_fallback(self):
        """Test data with unseen stratum should use marginal q_hat."""
        rng = np.random.RandomState(42)
        n = 100
        y_cal = rng.normal(10.0, 3.0, n)
        # Noisy, narrow predictions — many points will fall outside [lo, hi]
        lo_cal = y_cal - 0.5 + rng.normal(0, 1.0, n)
        hi_cal = y_cal + 0.5 + rng.normal(0, 1.0, n)
        strata_cal = np.array(["A"] * n)

        cal = calibrate_mondrian(y_cal, lo_cal, hi_cal, strata_cal, 0.10)

        # q_hat should be positive (raw intervals too narrow for 90% coverage)
        assert cal.marginal.q_hat > 0, f"Expected positive q_hat, got {cal.marginal.q_hat}"

        # Test with unseen stratum "B" — should fall back to marginal
        lo_test = np.array([5.0])
        hi_test = np.array([7.0])
        strata_test = np.array(["B"])

        lb, ub, _ = predict_mondrian_intervals(lo_test, hi_test, strata_test, cal)
        # Interval should be wider than the raw [5, 7] due to positive q_hat
        assert lb[0] < 5.0
        assert ub[0] > 7.0


# === Coverage diagnostics tests ===

class TestCoverage:
    def test_perfect_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([2.0, 3.0, 4.0])
        result = compute_coverage(y, lo, hi, nominal=0.90)
        assert result.coverage == 1.0
        assert result.n_covered == 3

    def test_no_coverage(self):
        y = np.array([10.0, 20.0, 30.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 1.0, 1.0])
        result = compute_coverage(y, lo, hi, nominal=0.90)
        assert result.coverage == 0.0

    def test_partial_coverage(self):
        y = np.array([5.0, 15.0])
        lo = np.array([4.0, 0.0])
        hi = np.array([6.0, 1.0])
        result = compute_coverage(y, lo, hi, nominal=0.90)
        assert result.coverage == 0.5

    def test_width_statistics(self):
        y = np.array([5.0, 5.0, 5.0])
        lo = np.array([3.0, 4.0, 2.0])
        hi = np.array([7.0, 6.0, 8.0])
        result = compute_coverage(y, lo, hi, nominal=0.90)
        assert result.median_width == 4.0  # widths: 4, 2, 6 -> median = 4
        assert result.mean_width == 4.0

    def test_coverage_gap(self):
        y = np.ones(100)
        lo = np.zeros(100)
        hi = np.ones(100) * 2
        result = compute_coverage(y, lo, hi, nominal=0.90)
        # All covered -> coverage = 1.0, gap = 0.10
        assert abs(result.coverage_gap - 0.10) < 1e-10


class TestClopperPearson:
    def test_zero_successes(self):
        lo, hi = clopper_pearson_ci(0, 100)
        assert lo == 0.0
        assert hi > 0.0

    def test_all_successes(self):
        lo, hi = clopper_pearson_ci(100, 100)
        assert lo < 1.0
        assert hi == 1.0

    def test_interval_contains_point(self):
        lo, hi = clopper_pearson_ci(50, 100)
        assert lo < 0.50 < hi

    def test_wider_with_fewer_samples(self):
        _, hi_100 = clopper_pearson_ci(50, 100)
        _, hi_1000 = clopper_pearson_ci(500, 1000)
        lo_100, _ = clopper_pearson_ci(50, 100)
        lo_1000, _ = clopper_pearson_ci(500, 1000)
        width_100 = hi_100 - lo_100
        width_1000 = hi_1000 - lo_1000
        assert width_100 > width_1000


class TestConditionalCoverage:
    def test_per_stratum(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([0.0, 1.0, 0.0, 3.0])
        hi = np.array([2.0, 3.0, 1.0, 5.0])
        strata = np.array(["A", "A", "B", "B"])

        results = compute_conditional_coverage(y, lo, hi, strata, 0.90)
        assert len(results) == 2
        # A: both covered (1,2 in [0,2] and [1,3]) -> 1.0
        assert results[0].label == "A"
        assert results[0].coverage == 1.0
        # B: 3 not in [0,1], 4 in [3,5] -> 0.5
        assert results[1].label == "B"
        assert results[1].coverage == 0.5

    def test_disparity(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([0.0, 1.0, 0.0, 3.0])
        hi = np.array([2.0, 3.0, 1.0, 5.0])
        strata = np.array(["A", "A", "B", "B"])
        results = compute_conditional_coverage(y, lo, hi, strata, 0.90)
        disparity = compute_coverage_disparity(results)
        assert abs(disparity - 0.5) < 1e-10  # 1.0 - 0.5


class TestCoverageSummaryTable:
    def test_creates_dataframe(self):
        marginal = compute_coverage(
            np.ones(10), np.zeros(10), np.ones(10) * 2, 0.90
        )
        conditional = compute_conditional_coverage(
            np.ones(10), np.zeros(10), np.ones(10) * 2,
            np.array(["A"] * 5 + ["B"] * 5), 0.90
        )
        table = coverage_summary_table(marginal, conditional)
        assert len(table) == 3  # marginal + 2 strata
        assert "coverage" in table.columns
        assert "within_tolerance" in table.columns
