"""Tests for VISION and PERCIST clinical overlay modules."""

import numpy as np
import pandas as pd
import pytest

from src.clinical.vision_overlay import (
    VisionDecision,
    classify_vision_eligibility,
    apply_vision_overlay,
    compute_vision_indeterminacy_rate,
)
from src.clinical.percist_overlay import (
    PercistDecision,
    classify_percist_response,
    compute_delta_pct_conservative,
    compute_delta_pct_correlated,
    apply_percist_overlay,
    compute_percist_indeterminacy_rate,
)
from src.clinical.indeterminacy_table import (
    build_indeterminacy_table,
    format_indeterminacy_table,
)


# === VISION tests ===


class TestVisionClassification:
    def test_clearly_eligible(self):
        """Entire CI above liver threshold -> eligible."""
        result = classify_vision_eligibility(
            suvmax_point=15.0, suvmax_lower=12.0, suvmax_upper=18.0,
            liver_suvmean=8.0,
        )
        assert result == VisionDecision.ELIGIBLE

    def test_clearly_ineligible(self):
        """Entire CI below liver threshold -> ineligible."""
        result = classify_vision_eligibility(
            suvmax_point=3.0, suvmax_lower=2.0, suvmax_upper=4.0,
            liver_suvmean=8.0,
        )
        assert result == VisionDecision.INELIGIBLE

    def test_indeterminate_straddles_threshold(self):
        """CI straddles the liver threshold -> indeterminate."""
        result = classify_vision_eligibility(
            suvmax_point=8.5, suvmax_lower=6.0, suvmax_upper=11.0,
            liver_suvmean=8.0,
        )
        assert result == VisionDecision.INDETERMINATE

    def test_lower_exactly_at_threshold(self):
        """Lower bound exactly at threshold -> indeterminate."""
        result = classify_vision_eligibility(
            suvmax_point=10.0, suvmax_lower=8.0, suvmax_upper=12.0,
            liver_suvmean=8.0,
        )
        assert result == VisionDecision.INDETERMINATE

    def test_upper_exactly_at_threshold(self):
        """Upper bound exactly at threshold -> indeterminate."""
        result = classify_vision_eligibility(
            suvmax_point=6.0, suvmax_lower=4.0, suvmax_upper=8.0,
            liver_suvmean=8.0,
        )
        assert result == VisionDecision.INDETERMINATE

    def test_zero_width_ci_above(self):
        """Point estimate above threshold with zero-width CI."""
        result = classify_vision_eligibility(
            suvmax_point=10.0, suvmax_lower=10.0, suvmax_upper=10.0,
            liver_suvmean=8.0,
        )
        assert result == VisionDecision.ELIGIBLE

    def test_zero_width_ci_below(self):
        """Point estimate below threshold with zero-width CI."""
        result = classify_vision_eligibility(
            suvmax_point=6.0, suvmax_lower=6.0, suvmax_upper=6.0,
            liver_suvmean=8.0,
        )
        assert result == VisionDecision.INELIGIBLE

    def test_wider_ci_more_indeterminate(self):
        """Wider CI should produce indeterminate for borderline case."""
        # Narrow CI: eligible
        r_narrow = classify_vision_eligibility(
            suvmax_point=10.0, suvmax_lower=9.0, suvmax_upper=11.0,
            liver_suvmean=8.0,
        )
        # Wide CI: indeterminate
        r_wide = classify_vision_eligibility(
            suvmax_point=10.0, suvmax_lower=5.0, suvmax_upper=15.0,
            liver_suvmean=8.0,
        )
        assert r_narrow == VisionDecision.ELIGIBLE
        assert r_wide == VisionDecision.INDETERMINATE


class TestVisionOverlay:
    def test_apply_overlay(self):
        df = pd.DataFrame({
            "patient_id": ["P1", "P1", "P2"],
            "lesion_id": [1, 2, 1],
            "study_uid": ["S1", "S1", "S2"],
            "suvmax": [15.0, 5.0, 8.5],
            "suvmax_ci_lower": [12.0, 3.0, 6.0],
            "suvmax_ci_upper": [18.0, 7.0, 11.0],
        })
        liver = {"S1": 8.0, "S2": 8.0}

        result = apply_vision_overlay(df, liver)

        assert result.iloc[0]["vision_decision"] == "eligible"
        assert result.iloc[1]["vision_decision"] == "ineligible"
        assert result.iloc[2]["vision_decision"] == "indeterminate"

    def test_indeterminacy_rate(self):
        df = pd.DataFrame({
            "vision_decision": ["eligible"] * 7 + ["ineligible"] * 2 + ["indeterminate"] * 1,
        })
        stats = compute_vision_indeterminacy_rate(df)
        assert stats["n_total"] == 10
        assert stats["n_indeterminate"] == 1
        assert abs(stats["indeterminacy_rate"] - 0.1) < 1e-10
        assert stats["indeterminacy_ci_lower"] < stats["indeterminacy_rate"]
        assert stats["indeterminacy_ci_upper"] > stats["indeterminacy_rate"]


# === PERCIST tests ===


class TestPercistDelta:
    def test_conservative_decrease(self):
        """Follow-up lower than baseline -> negative delta."""
        lower, upper = compute_delta_pct_conservative(
            baseline_lower=9.0, baseline_upper=11.0,
            followup_lower=5.0, followup_upper=7.0,
        )
        # Most negative: (5 - 11) / 11 = -54.5%
        # Most positive: (7 - 9) / 9 = -22.2%
        assert lower < -50.0
        assert upper < 0.0
        assert lower < upper

    def test_conservative_no_change(self):
        """Same values -> delta near zero."""
        lower, upper = compute_delta_pct_conservative(
            baseline_lower=9.0, baseline_upper=11.0,
            followup_lower=9.0, followup_upper=11.0,
        )
        # Most negative: (9-11)/11 = -18.2%
        # Most positive: (11-9)/9 = 22.2%
        assert lower < 0
        assert upper > 0

    def test_conservative_increase(self):
        """Follow-up higher than baseline -> positive delta."""
        lower, upper = compute_delta_pct_conservative(
            baseline_lower=5.0, baseline_upper=7.0,
            followup_lower=9.0, followup_upper=11.0,
        )
        assert lower > 0
        assert upper > 0

    def test_correlated_tighter_than_conservative(self):
        """With positive correlation, correlated bounds should be tighter."""
        cons_lower, cons_upper = compute_delta_pct_conservative(
            baseline_lower=8.0, baseline_upper=12.0,
            followup_lower=5.0, followup_upper=9.0,
        )
        corr_lower, corr_upper = compute_delta_pct_correlated(
            baseline_point=10.0, followup_point=7.0,
            baseline_lower=8.0, baseline_upper=12.0,
            followup_lower=5.0, followup_upper=9.0,
            correlation=0.7,
        )
        # Correlated interval should be narrower
        cons_width = cons_upper - cons_lower
        corr_width = corr_upper - corr_lower
        assert corr_width < cons_width


class TestPercistClassification:
    def test_clear_response(self):
        """Both bounds below -30%, sufficient absolute change."""
        result = classify_percist_response(
            delta_pct_lower=-55.0, delta_pct_upper=-35.0,
            delta_pct_point=-45.0, absolute_change=3.0,
        )
        assert result == PercistDecision.RESPONSE

    def test_clear_stable(self):
        """Both bounds between -30% and +30%."""
        result = classify_percist_response(
            delta_pct_lower=-10.0, delta_pct_upper=10.0,
            delta_pct_point=0.0, absolute_change=0.5,
        )
        assert result == PercistDecision.STABLE

    def test_clear_progression(self):
        """Both bounds above +30%, sufficient absolute change."""
        result = classify_percist_response(
            delta_pct_lower=35.0, delta_pct_upper=55.0,
            delta_pct_point=45.0, absolute_change=4.0,
        )
        assert result == PercistDecision.PROGRESSION

    def test_straddles_response_threshold(self):
        """Interval crosses -30% -> indeterminate."""
        result = classify_percist_response(
            delta_pct_lower=-40.0, delta_pct_upper=-20.0,
            delta_pct_point=-30.0, absolute_change=2.0,
        )
        assert result == PercistDecision.INDETERMINATE

    def test_straddles_progression_threshold(self):
        """Interval crosses +30% -> indeterminate."""
        result = classify_percist_response(
            delta_pct_lower=20.0, delta_pct_upper=40.0,
            delta_pct_point=30.0, absolute_change=2.0,
        )
        assert result == PercistDecision.INDETERMINATE

    def test_response_but_small_absolute_change(self):
        """Below -30% but |Δ| < 0.8 SUL -> indeterminate (PERCIST requires both)."""
        result = classify_percist_response(
            delta_pct_lower=-50.0, delta_pct_upper=-35.0,
            delta_pct_point=-42.0, absolute_change=0.3,
        )
        # Both bounds below -30% but absolute change too small
        # This should NOT be classified as response
        assert result != PercistDecision.RESPONSE


class TestPercistOverlay:
    def test_apply_overlay_conservative(self):
        df = pd.DataFrame({
            "patient_id": ["P1", "P2"],
            "baseline_sulpeak": [10.0, 10.0],
            "followup_sulpeak": [5.0, 9.5],
            "baseline_ci_lower": [9.0, 9.0],
            "baseline_ci_upper": [11.0, 11.0],
            "followup_ci_lower": [4.0, 8.5],
            "followup_ci_upper": [6.0, 10.5],
        })

        result = apply_percist_overlay(df, method="conservative")

        assert "delta_pct_point" in result.columns
        assert "percist_decision" in result.columns
        assert len(result) == 2


# === Indeterminacy table tests ===


class TestIndeterminacyTable:
    def test_build_table(self):
        vision_results = [
            {"ci_level": 0.80, "n_total": 100, "eligible_pct": 60,
             "ineligible_pct": 30, "indeterminacy_pct": 10,
             "indeterminacy_ci_lower": 0.05, "indeterminacy_ci_upper": 0.17},
            {"ci_level": 0.90, "n_total": 100, "eligible_pct": 50,
             "ineligible_pct": 25, "indeterminacy_pct": 25,
             "indeterminacy_ci_lower": 0.17, "indeterminacy_ci_upper": 0.34},
        ]

        table = build_indeterminacy_table(vision_results)
        assert len(table) == 2
        assert "threshold" in table.columns
        assert table.iloc[0]["ci_level"] == "80%"
        assert table.iloc[1]["ci_level"] == "90%"
        # Higher CI level should have higher indeterminacy
        assert table.iloc[1]["indeterminate_pct"] > table.iloc[0]["indeterminate_pct"]

    def test_format_table(self):
        vision_results = [
            {"ci_level": 0.90, "n_total": 100, "eligible_pct": 50,
             "ineligible_pct": 30, "indeterminacy_pct": 20,
             "indeterminacy_ci_lower": 0.13, "indeterminacy_ci_upper": 0.29},
        ]
        table = build_indeterminacy_table(vision_results)
        formatted = format_indeterminacy_table(table)
        assert "VISION" in formatted
        assert "90%" in formatted
