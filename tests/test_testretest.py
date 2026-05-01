"""Tests for test-retest serial pair matching and Poisson noise injection."""

import numpy as np
import pandas as pd
import pytest

from src.testretest.serial_pairs import (
    find_serial_scan_pairs,
    match_lesions_across_scans,
    compute_within_lesion_cv,
    check_decision_gate,
    MAX_SCAN_INTERVAL_WEEKS,
    MAX_CENTROID_DISPLACEMENT_MM,
    MIN_PAIRS_FOR_PRIMARY,
)
from src.testretest.poisson_noise import (
    suv_to_counts,
    inject_poisson_noise,
    counts_to_suv,
    generate_noisy_replicates,
    compute_replicate_cv,
)


# === Serial pair matching ===


class TestSerialPairFinding:
    def test_finds_pair_within_interval(self):
        df = pd.DataFrame({
            "patient_id": ["P1", "P1"],
            "study_uid": ["S1", "S2"],
            "study_date": ["2020-01-01", "2020-02-01"],  # 31 days apart
        })
        pairs = find_serial_scan_pairs(df)
        assert len(pairs) == 1
        assert pairs.iloc[0]["interval_days"] == 31

    def test_excludes_pair_beyond_interval(self):
        df = pd.DataFrame({
            "patient_id": ["P1", "P1"],
            "study_uid": ["S1", "S2"],
            "study_date": ["2020-01-01", "2020-06-01"],  # ~150 days
        })
        pairs = find_serial_scan_pairs(df)
        assert len(pairs) == 0

    def test_different_patients_not_paired(self):
        df = pd.DataFrame({
            "patient_id": ["P1", "P2"],
            "study_uid": ["S1", "S2"],
            "study_date": ["2020-01-01", "2020-01-02"],
        })
        pairs = find_serial_scan_pairs(df)
        assert len(pairs) == 0

    def test_single_scan_no_pair(self):
        df = pd.DataFrame({
            "patient_id": ["P1"],
            "study_uid": ["S1"],
            "study_date": ["2020-01-01"],
        })
        pairs = find_serial_scan_pairs(df)
        assert len(pairs) == 0

    def test_three_scans_multiple_pairs(self):
        """Three scans within interval should produce up to 3 pairs."""
        df = pd.DataFrame({
            "patient_id": ["P1"] * 3,
            "study_uid": ["S1", "S2", "S3"],
            "study_date": ["2020-01-01", "2020-01-15", "2020-02-01"],
        })
        pairs = find_serial_scan_pairs(df)
        assert len(pairs) == 3  # (S1,S2), (S1,S3), (S2,S3)

    def test_boundary_exactly_8_weeks(self):
        """Exactly 8 weeks = 56 days should be included."""
        df = pd.DataFrame({
            "patient_id": ["P1", "P1"],
            "study_uid": ["S1", "S2"],
            "study_date": ["2020-01-01", "2020-02-26"],  # 56 days
        })
        pairs = find_serial_scan_pairs(df)
        assert len(pairs) == 1


class TestLesionMatching:
    def test_close_lesions_matched(self):
        scan1 = pd.DataFrame({
            "lesion_id": [1, 2],
            "centroid_z": [100.0, 200.0],
            "centroid_y": [50.0, 50.0],
            "centroid_x": [50.0, 50.0],
        })
        scan2 = pd.DataFrame({
            "lesion_id": [1, 2],
            "centroid_z": [102.0, 198.0],  # slight shift
            "centroid_y": [51.0, 49.0],
            "centroid_x": [50.0, 50.0],
        })
        matches = match_lesions_across_scans(scan1, scan2)
        assert len(matches) == 2

    def test_distant_lesions_not_matched(self):
        scan1 = pd.DataFrame({
            "lesion_id": [1],
            "centroid_z": [100.0],
            "centroid_y": [50.0],
            "centroid_x": [50.0],
        })
        scan2 = pd.DataFrame({
            "lesion_id": [1],
            "centroid_z": [300.0],  # >20mm away
            "centroid_y": [50.0],
            "centroid_x": [50.0],
        })
        matches = match_lesions_across_scans(scan1, scan2)
        assert len(matches) == 0

    def test_greedy_one_to_one(self):
        """Each lesion should match at most one partner."""
        scan1 = pd.DataFrame({
            "lesion_id": [1, 2],
            "centroid_z": [100.0, 105.0],
            "centroid_y": [50.0, 50.0],
            "centroid_x": [50.0, 50.0],
        })
        scan2 = pd.DataFrame({
            "lesion_id": [1],  # only one target
            "centroid_z": [102.0],
            "centroid_y": [50.0],
            "centroid_x": [50.0],
        })
        matches = match_lesions_across_scans(scan1, scan2)
        assert len(matches) == 1  # only one match possible

    def test_empty_scans(self):
        empty = pd.DataFrame(columns=["lesion_id", "centroid_z", "centroid_y", "centroid_x"])
        nonempty = pd.DataFrame({
            "lesion_id": [1],
            "centroid_z": [100.0],
            "centroid_y": [50.0],
            "centroid_x": [50.0],
        })
        assert match_lesions_across_scans(empty, nonempty) == []
        assert match_lesions_across_scans(nonempty, empty) == []


class TestWithinLesionCV:
    def test_identical_values_zero_cv(self):
        df = pd.DataFrame({"suv1": [10.0, 20.0], "suv2": [10.0, 20.0]})
        result = compute_within_lesion_cv(df, "suv1", "suv2")
        assert result["median_cv_pct"] == 0.0

    def test_known_cv(self):
        """Two measurements of 8 and 12 -> mean=10, SD=2√2/√2=2, CV=20%."""
        df = pd.DataFrame({"suv1": [8.0], "suv2": [12.0]})
        result = compute_within_lesion_cv(df, "suv1", "suv2")
        # |8-12| / sqrt(2) = 2.828, mean = 10, CV = 28.28%
        assert 25.0 < result["mean_cv_pct"] < 32.0


class TestDecisionGate:
    def test_above_threshold(self):
        assert check_decision_gate(50) is True
        assert check_decision_gate(100) is True

    def test_below_threshold(self):
        assert check_decision_gate(49) is False
        assert check_decision_gate(0) is False


# === Poisson noise injection ===


class TestPoissonNoise:
    def test_suv_to_counts_roundtrip(self):
        """Converting SUV -> counts -> SUV should be approximately identity."""
        rng = np.random.RandomState(42)
        suv = rng.uniform(2.0, 20.0, size=(10, 10, 10))

        counts = suv_to_counts(
            suv, injected_dose_bq=370e6, patient_weight_kg=75.0,
            decay_factor=0.68, frame_duration_sec=180.0,
        )
        recovered = counts_to_suv(
            counts, injected_dose_bq=370e6, patient_weight_kg=75.0,
            decay_factor=0.68, frame_duration_sec=180.0, dose_fraction=1.0,
        )

        np.testing.assert_allclose(recovered, suv, rtol=1e-10)

    def test_poisson_noise_preserves_mean(self):
        """Mean of many Poisson samples should approximate lambda."""
        rng = np.random.RandomState(42)
        counts = np.ones((5, 5, 5)) * 1000.0  # high counts for stable mean

        noisy_samples = [inject_poisson_noise(counts, 1.0, rng) for _ in range(100)]
        mean_noisy = np.mean(noisy_samples, axis=0)

        np.testing.assert_allclose(mean_noisy, counts, rtol=0.05)

    def test_dose_reduction_increases_noise(self):
        """Lower dose fraction should produce noisier images."""
        counts = np.ones((10, 10, 10)) * 10000.0

        rng1 = np.random.RandomState(42)
        noisy_full = inject_poisson_noise(counts, 1.0, rng1)

        rng2 = np.random.RandomState(42)
        noisy_half = inject_poisson_noise(counts, 0.5, rng2)

        # Half dose should have higher relative noise (CV)
        cv_full = np.std(noisy_full) / np.mean(noisy_full)
        cv_half = np.std(noisy_half) / np.mean(noisy_half)
        assert cv_half > cv_full

    def test_counts_non_negative(self):
        """Noisy counts should never be negative."""
        rng = np.random.RandomState(42)
        counts = np.ones((10, 10, 10)) * 5.0  # low counts
        noisy = inject_poisson_noise(counts, 0.10, rng)
        assert (noisy >= 0).all()

    def test_generate_replicates_shape(self):
        suv = np.ones((5, 5, 5)) * 10.0
        replicates = generate_noisy_replicates(
            suv, injected_dose_bq=370e6, patient_weight_kg=75.0,
            decay_factor=0.68, frame_duration_sec=180.0,
            dose_fractions=[0.5, 0.25], n_replicates=3, seed=42,
        )
        assert len(replicates) == 2  # two dose levels
        assert len(replicates[0.5]) == 3  # three replicates each
        assert replicates[0.5][0].shape == (5, 5, 5)


class TestReplicateCV:
    def test_identical_values(self):
        assert compute_replicate_cv([10.0, 10.0, 10.0]) == 0.0

    def test_known_cv(self):
        values = [10.0, 10.0, 10.0, 20.0]  # mean=12.5, sd=5.0, cv=40%
        cv = compute_replicate_cv(values)
        assert 30.0 < cv < 50.0

    def test_zero_mean(self):
        assert compute_replicate_cv([0.0, 0.0]) == 0.0
