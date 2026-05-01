"""Tests for the Phase 2 Poisson-noise per-case orchestrator (Amendment 6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.testretest.defaults import (
    AUTOPET_I_FDG_DEFAULTS,
    SCANNER_CALIBRATION,
    SCANNER_CALIBRATION_DEFAULT,
    autopet_i_defaults_for_patient,
    lookup_scanner_calibration,
)
from src.testretest.phase2_pipeline import (
    CaseInputs,
    reduce_to_wcv,
    run_case,
    MIN_LESION_VOLUME_ML,
)


def _make_synthetic_case(seed: int = 0) -> CaseInputs:
    """A 32x32x32 SUV volume with two spherical 'lesions' embedded.

    The two lesions sit at well-separated centres so connected-component
    labelling produces two stable components. Voxel spacing is 4mm isotropic
    so each voxel is 64 mm^3 and a 25-voxel cluster is ~1.6 mL (above the 1mL
    threshold). The smaller seed cluster (8 voxels = 0.5 mL) tests the
    min_volume_ml exclusion path.
    """
    rng = np.random.RandomState(seed)
    suv = rng.uniform(0.1, 0.5, size=(32, 32, 32)).astype(np.float64)
    seg = np.zeros_like(suv, dtype=np.int32)

    # Lesion 1: 3-voxel-radius sphere at (8, 8, 8); SUV ~ 12 inside
    z, y, x = np.ogrid[:32, :32, :32]
    l1 = (z - 8) ** 2 + (y - 8) ** 2 + (x - 8) ** 2 <= 9  # ~r=3 sphere
    suv[l1] = 12.0
    seg[l1] = 1

    # Lesion 2: 3-voxel-radius sphere at (24, 24, 24); SUV ~ 6 inside
    l2 = (z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2 <= 9
    suv[l2] = 6.0
    seg[l2] = 1

    # Sub-threshold blip: single voxel at (16, 16, 16) -- should be filtered
    seg[16, 16, 16] = 1
    suv[16, 16, 16] = 99.0  # high SUV but excluded by volume

    voxel_spacing_mm = (4.0, 4.0, 4.0)  # 64 mm^3 per voxel

    params = autopet_i_defaults_for_patient(patient_weight_kg=75.0)

    return CaseInputs(
        case_id="SYNTHETIC_001",
        series_uid="SERIES_001",
        cohort="autopet_i",
        params_source="defaults",
        suv_volume=suv,
        seg_mask=seg,
        voxel_spacing_mm=voxel_spacing_mm,
        params=params,
    )


class TestDefaults:
    def test_locked_values_match_amendments_6_and_7(self):
        """Amendments 6 + 7 lock specific cohort-typical values; ensure module agrees."""
        d = AUTOPET_I_FDG_DEFAULTS
        # Amendment 6 locked values
        assert d["dose_per_weight_mbq_per_kg"] == 4.0
        assert d["dose_mid_point_mbq_fallback"] == 350.0
        assert d["patient_weight_kg_fallback"] == 75.0
        assert d["uptake_time_sec"] == 3600.0
        assert d["half_life_sec_f18"] == 6588.0
        assert d["frame_duration_sec"] == 120.0
        # Amendment 7: calibration_factor updated from 1.0 to 5e-4 (Siemens Biograph mCT anchor)
        assert d["calibration_factor"] == pytest.approx(5.0e-4, rel=1e-9)

    def test_amendment_7_scanner_calibration_table(self):
        """Amendment 7 locks per-scanner calibration values."""
        # Anchor
        assert SCANNER_CALIBRATION["Biograph mCT Flow 20"] == pytest.approx(5.0e-4, rel=1e-9)
        # NEMA-ratio scaled
        assert SCANNER_CALIBRATION["Biograph 64-4R TruePoint"] == pytest.approx(
            5.0e-4 * (7.0 / 9.7), rel=1e-9
        )
        assert SCANNER_CALIBRATION["Discovery 690"] == pytest.approx(
            5.0e-4 * (7.5 / 9.7), rel=1e-9
        )
        # Cohort fallback
        assert SCANNER_CALIBRATION_DEFAULT == pytest.approx(4.5e-4, rel=1e-9)

    def test_lookup_scanner_calibration_matches_substrings(self):
        # Exact / case-insensitive
        assert lookup_scanner_calibration("Biograph mCT Flow 20") == pytest.approx(5.0e-4)
        assert lookup_scanner_calibration("biograph mct flow 20") == pytest.approx(5.0e-4)
        # Substring (real DICOM strings often include extra suffixes)
        assert lookup_scanner_calibration(
            "Biograph mCT Flow 20-mCT_Flow_2009"
        ) == pytest.approx(5.0e-4)
        assert lookup_scanner_calibration("Discovery 690") == pytest.approx(
            5.0e-4 * (7.5 / 9.7)
        )

    def test_lookup_scanner_calibration_falls_back_for_unknown(self):
        assert lookup_scanner_calibration("Unknown Scanner X") == pytest.approx(4.5e-4)
        assert lookup_scanner_calibration(None) == pytest.approx(4.5e-4)
        assert lookup_scanner_calibration("") == pytest.approx(4.5e-4)

    def test_dose_uses_per_weight_when_weight_known(self):
        params = autopet_i_defaults_for_patient(patient_weight_kg=80.0)
        # 4 MBq/kg * 80 kg = 320 MBq = 3.2e8 Bq
        assert params.injected_dose_bq == pytest.approx(320e6, rel=1e-9)
        assert params.patient_weight_kg == 80.0

    def test_dose_falls_back_to_mid_point_when_weight_missing(self):
        params = autopet_i_defaults_for_patient(patient_weight_kg=None)
        # 350 MBq fallback
        assert params.injected_dose_bq == pytest.approx(350e6, rel=1e-9)
        assert params.patient_weight_kg == 75.0  # cohort fallback

    def test_decay_factor_correct_for_60min_uptake(self):
        params = autopet_i_defaults_for_patient(patient_weight_kg=75.0)
        # F-18 t1/2 = 6588s, uptake = 3600s -> 2^(-3600/6588) ~= 0.685
        expected = 2.0 ** (-3600.0 / 6588.0)
        assert params.decay_factor == pytest.approx(expected, rel=1e-12)


class TestRunCase:
    def test_two_lesions_recovered_above_min_volume(self):
        case = _make_synthetic_case()
        df = run_case(case, n_replicates=3)
        # Two lesions x (1 baseline + 3 dose levels x 3 replicates) = 2 * 10 = 20 rows
        assert len(df) == 20
        assert df["lesion_id"].nunique() == 2  # sub-threshold blip excluded
        assert (df.loc[df["replicate"] == 0, "dose_fraction"] == 1.0).all()

    def test_baseline_suvmax_matches_input(self):
        """Replicate=0 rows should be the noise-free SUV stats."""
        case = _make_synthetic_case()
        df = run_case(case, n_replicates=2)
        baseline = df[df["replicate"] == 0]
        # Lesion 1 SUV was set to 12.0; lesion 2 to 6.0
        suvmaxes = sorted(baseline["suvmax"].tolist())
        assert suvmaxes[0] == pytest.approx(6.0, abs=1e-6)
        assert suvmaxes[1] == pytest.approx(12.0, abs=1e-6)

    def test_noise_increases_with_lower_dose(self):
        """Lower dose fraction -> higher Poisson noise -> higher CV."""
        case = _make_synthetic_case()
        df = run_case(case, n_replicates=20, seed=123)
        wcv = reduce_to_wcv(df)
        # For each lesion, wCV at 0.10 dose should exceed wCV at 0.50 dose
        for lid, sub in wcv.groupby("lesion_id"):
            cv_50 = sub.loc[sub["dose_fraction"] == 0.50, "wcv_suvmax_pct"].iloc[0]
            cv_10 = sub.loc[sub["dose_fraction"] == 0.10, "wcv_suvmax_pct"].iloc[0]
            assert cv_10 > cv_50, (
                f"lesion {lid}: expected wCV(10% dose)={cv_10:.2f} > wCV(50% dose)={cv_50:.2f}"
            )

    def test_seed_reproducibility(self):
        """Same seed -> same output; different seed -> different output."""
        case = _make_synthetic_case()
        df_a = run_case(case, n_replicates=3, seed=42)
        df_b = run_case(case, n_replicates=3, seed=42)
        df_c = run_case(case, n_replicates=3, seed=99)
        # Drop replicate=0 rows since they're noise-free and identical regardless
        a_noisy = df_a[df_a["replicate"] > 0]["suvmax"].to_numpy()
        b_noisy = df_b[df_b["replicate"] > 0]["suvmax"].to_numpy()
        c_noisy = df_c[df_c["replicate"] > 0]["suvmax"].to_numpy()
        assert np.allclose(a_noisy, b_noisy)
        assert not np.allclose(a_noisy, c_noisy)

    def test_empty_segmentation_returns_empty_frame(self):
        case = _make_synthetic_case()
        case.seg_mask[:] = 0
        df = run_case(case, n_replicates=3)
        assert df.empty
        assert "wcv_suvmax_pct" not in df.columns  # this is the replicate-trace schema


class TestReduceToWcv:
    def test_wcv_is_zero_when_replicates_identical(self):
        import pandas as pd
        rep = pd.DataFrame({
            "case_id": ["A"] * 4, "series_uid": ["S"] * 4,
            "cohort": ["x"] * 4, "params_source": ["d"] * 4,
            "lesion_id": [1, 1, 1, 1],
            "dose_fraction": [1.0, 0.5, 0.5, 0.5],
            "replicate": [0, 1, 2, 3],
            "suvmax": [10.0, 5.0, 5.0, 5.0],
            "suvpeak": [9.0, 4.0, 4.0, 4.0],
            "suvmean": [7.0, 3.0, 3.0, 3.0],
        })
        wcv = reduce_to_wcv(rep)
        assert len(wcv) == 1
        row = wcv.iloc[0]
        assert row["wcv_suvmax_pct"] == 0.0
        assert row["suvmax_baseline"] == 10.0
        assert row["n_replicates"] == 3
