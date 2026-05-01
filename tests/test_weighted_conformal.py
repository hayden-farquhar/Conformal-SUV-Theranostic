"""Tests for src/conformal/weighted.py (Amendment 11 WCP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.conformal.weighted import (
    WCP_FEATURE_COLS,
    WCP_WEIGHT_CLIP_LOW,
    WCP_WEIGHT_CLIP_HIGH,
    calibration_shift_index,
    fit_wcp_classifier,
    weighted_conformal_threshold,
)


def _make_source_target(n_source=400, n_target=400, shift=2.0, seed=0):
    """Synthetic source vs target with controlled covariate shift on volume_ml."""
    rng = np.random.RandomState(seed)
    # Source: volume ~ N(5, 2)
    src_vol = rng.normal(5, 2, n_source)
    src_sa = rng.normal(15, 5, n_source)
    src_sph = rng.uniform(0.4, 0.8, n_source)
    # Target: volume ~ N(5 + shift, 2) (covariate shift on volume)
    tgt_vol = rng.normal(5 + shift, 2, n_target)
    tgt_sa = rng.normal(15, 5, n_target)
    tgt_sph = rng.uniform(0.4, 0.8, n_target)
    src = pd.DataFrame({"volume_ml": src_vol, "surface_area_cm2": src_sa, "sphericity": src_sph})
    tgt = pd.DataFrame({"volume_ml": tgt_vol, "surface_area_cm2": tgt_sa, "sphericity": tgt_sph})
    return src, tgt


class TestFitWCPClassifier:
    def test_no_shift_classifier_at_chance(self):
        """When source == target, the classifier should be near 0.5 AUC."""
        src, tgt = _make_source_target(n_source=500, n_target=500, shift=0.0)
        model = fit_wcp_classifier(src, tgt)
        assert 0.45 <= model.classifier_auc <= 0.55, (
            f"AUC {model.classifier_auc} should be near 0.5 with no shift"
        )
        # Weights should cluster around 1.0
        assert 0.7 < float(np.median(model.source_weights)) < 1.4

    def test_strong_shift_classifier_above_chance(self):
        """With a 2.0-unit shift on volume_ml (1 SD), AUC should be clearly > 0.5."""
        src, tgt = _make_source_target(n_source=500, n_target=500, shift=2.0)
        model = fit_wcp_classifier(src, tgt)
        assert model.classifier_auc > 0.65, (
            f"AUC {model.classifier_auc} should detect the shift"
        )

    def test_shift_direction_reflected_in_weights(self):
        """Source points whose features are closer to the target distribution
        should have higher importance weights."""
        src, tgt = _make_source_target(n_source=500, n_target=500, shift=2.0)
        model = fit_wcp_classifier(src, tgt)
        # Sort source by volume_ml: high-volume source points are closer to
        # target (which is shifted to higher volume) and should have higher weight
        order = np.argsort(src["volume_ml"].values)
        weights_sorted = model.source_weights[order]
        # Spearman-like check: weights should increase with volume_ml
        # (use simple correlation; high-vol source has higher weight)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(np.arange(len(weights_sorted)), weights_sorted)
        assert rho > 0.3, f"weight should increase with volume_ml; rho={rho}"

    def test_weights_clipped_to_percentile_range(self):
        src, tgt = _make_source_target(n_source=200, n_target=200, shift=4.0)
        model = fit_wcp_classifier(src, tgt)
        assert model.source_weights.min() >= model.weight_clip_low - 1e-9
        assert model.source_weights.max() <= model.weight_clip_high + 1e-9

    def test_feature_cols_locked_to_amendment_11(self):
        assert WCP_FEATURE_COLS == ["volume_ml", "surface_area_cm2", "sphericity"]
        assert WCP_WEIGHT_CLIP_LOW == 1.0
        assert WCP_WEIGHT_CLIP_HIGH == 99.0

    def test_missing_feature_raises(self):
        src = pd.DataFrame({"volume_ml": [1.0, 2.0]})  # missing SA + sphericity
        tgt = pd.DataFrame({"volume_ml": [1.0, 2.0], "surface_area_cm2": [1, 2], "sphericity": [0.5, 0.6]})
        with pytest.raises(ValueError, match="source_features missing"):
            fit_wcp_classifier(src, tgt)

    def test_empty_cohort_raises(self):
        src = pd.DataFrame({c: [] for c in WCP_FEATURE_COLS})
        tgt = pd.DataFrame({c: [1.0] for c in WCP_FEATURE_COLS})
        with pytest.raises(ValueError, match="empty source"):
            fit_wcp_classifier(src, tgt)


class TestWeightedConformalThreshold:
    def test_uniform_weights_recovers_unweighted_quantile(self):
        """When all weights are equal, WCP threshold = unweighted (1-alpha) quantile."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        weights = np.ones_like(scores)
        # 90% quantile (alpha=0.10) should be ~0.9
        q = weighted_conformal_threshold(scores, weights, alpha=0.10)
        # searchsorted on cum_w=[0.1, 0.2, ..., 1.0] for target 0.9 -> idx=8 -> scores[8]=0.9
        assert q == pytest.approx(0.9)

    def test_unequal_weights_changes_quantile(self):
        """Putting heavy weight on small scores should pull the quantile DOWN."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # Weight small scores 10x, large scores 1x
        weights = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        q_weighted = weighted_conformal_threshold(scores, weights, alpha=0.10)
        q_uniform = weighted_conformal_threshold(scores, np.ones_like(scores), alpha=0.10)
        assert q_weighted < q_uniform, "heavier weight on small scores should reduce q"

    def test_extreme_weights_target_very_close_to_max(self):
        """Putting heavy weight on the largest score should push the quantile UP."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        q = weighted_conformal_threshold(scores, weights, alpha=0.10)
        assert q == 1.0

    def test_invalid_alpha_raises(self):
        scores = np.array([0.1, 0.5, 0.9])
        weights = np.ones(3)
        for a in (0.0, 1.0, -0.1, 1.1):
            with pytest.raises(ValueError, match="alpha"):
                weighted_conformal_threshold(scores, weights, alpha=a)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            weighted_conformal_threshold(np.array([0.1, 0.2]), np.array([1.0, 1.0, 1.0]))

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="weights must be > 0"):
            weighted_conformal_threshold(np.array([0.1, 0.2]), np.zeros(2))


class TestCalibrationShiftIndex:
    def test_csi_one_means_no_shift(self):
        assert calibration_shift_index(0.1, 0.1) == 1.0

    def test_csi_three_means_residuals_three_x_larger(self):
        assert calibration_shift_index(0.3, 0.1) == pytest.approx(3.0)

    def test_csi_below_one_means_target_residuals_smaller(self):
        assert calibration_shift_index(0.05, 0.1) == pytest.approx(0.5)

    def test_csi_zero_source_returns_inf(self):
        assert calibration_shift_index(0.1, 0.0) == float("inf")


class TestPatientLevelHoldout:
    def _make_cohort(self, n_patients=100, lesions_per=3, seed=0):
        from src.conformal.weighted import patient_level_holdout_split  # noqa
        rng = np.random.RandomState(seed)
        rows = []
        for pid in range(n_patients):
            for lid in range(rng.randint(1, lesions_per + 1)):
                rows.append({"case_id": f"P{pid:03d}", "lesion_id": lid, "suvmax": rng.uniform(2, 30)})
        return pd.DataFrame(rows)

    def test_split_has_no_patient_overlap(self):
        from src.conformal.weighted import patient_level_holdout_split
        df = self._make_cohort(n_patients=100)
        ho, ev = patient_level_holdout_split(df)
        ho_pats = set(ho["case_id"]); ev_pats = set(ev["case_id"])
        assert ho_pats.isdisjoint(ev_pats)
        assert ho_pats | ev_pats == set(df["case_id"])

    def test_split_holdout_ratio_close_to_20_percent(self):
        from src.conformal.weighted import patient_level_holdout_split
        df = self._make_cohort(n_patients=100)
        ho, ev = patient_level_holdout_split(df)
        ho_n = len(set(ho["case_id"])); ev_n = len(set(ev["case_id"]))
        assert ho_n == 20 and ev_n == 80

    def test_split_seed_reproducible(self):
        from src.conformal.weighted import patient_level_holdout_split
        df = self._make_cohort(n_patients=100)
        ho_a, _ = patient_level_holdout_split(df, seed=42)
        ho_b, _ = patient_level_holdout_split(df, seed=42)
        assert set(ho_a["case_id"]) == set(ho_b["case_id"])
        ho_c, _ = patient_level_holdout_split(df, seed=99)
        assert set(ho_a["case_id"]) != set(ho_c["case_id"])

    def test_lesions_for_holdout_patient_all_in_holdout(self):
        """Patient-level cluster sampling: ALL of a patient's lesions in same split."""
        from src.conformal.weighted import patient_level_holdout_split
        df = self._make_cohort(n_patients=50, lesions_per=5, seed=7)
        ho, ev = patient_level_holdout_split(df)
        for pid in ho["case_id"].unique():
            assert (df[df["case_id"] == pid]["lesion_id"].sort_values().tolist() ==
                    ho[ho["case_id"] == pid]["lesion_id"].sort_values().tolist())


class TestExtendedFeatureBuilder:
    """Amendment 12 §12a: build_extended_wcp_features."""

    def _autopet_iii_like(self, n=10):
        return pd.DataFrame({
            "volume_ml": np.linspace(2, 30, n),
            "surface_area_cm2": np.linspace(8, 80, n),
            "sphericity": np.linspace(0.5, 0.7, n),
            "voxel_spacing_0": [4.0] * n,
            "voxel_spacing_1": [4.0] * n,
            "voxel_spacing_2": [4.0] * n,
            "case_id": [f"PSMA_{i:03d}" for i in range(n)],
            "tracer": ["PSMA"] * n,
            "vendor": ["Siemens"] * n,
        })

    def _hecktor_like(self, n=10):
        return pd.DataFrame({
            "volume_ml": np.linspace(2, 30, n),
            "surface_area_cm2": np.linspace(8, 80, n),
            "sphericity": np.linspace(0.5, 0.7, n),
            "voxel_spacing_0": [3.27] * n,
            "voxel_spacing_1": [3.27] * n,
            "voxel_spacing_2": [3.27] * n,
            "case_id": [f"MDA-{i:03d}" for i in range(n)],
            "tracer": ["FDG"] * n,
            "vendor": ["GE"] * n,
            "centre_id": [5] * n,
        })

    def _autopet_i_like(self, n=10):
        return pd.DataFrame({
            "volume_ml": np.linspace(2, 30, n),
            "surface_area_cm2": np.linspace(8, 80, n),
            "sphericity": np.linspace(0.5, 0.7, n),
            "voxel_spacing_0": [2.04] * n,
            "voxel_spacing_1": [2.04] * n,
            "voxel_spacing_2": [3.0] * n,
            "case_id": [f"PETCT_{i:03d}" for i in range(n)],
            "tracer": ["FDG"] * n,
            "vendor": ["Siemens"] * n,
        })

    def test_returns_16_columns_in_locked_order(self):
        from src.conformal.weighted import build_extended_wcp_features, WCP_EXTENDED_FEATURE_COLS
        out = build_extended_wcp_features(self._autopet_i_like())
        assert list(out.columns) == WCP_EXTENDED_FEATURE_COLS
        assert len(out.columns) == 16

    def test_psma_indicator_set_for_autopet_iii(self):
        from src.conformal.weighted import build_extended_wcp_features
        out_iii = build_extended_wcp_features(self._autopet_iii_like())
        out_i = build_extended_wcp_features(self._autopet_i_like())
        out_h = build_extended_wcp_features(self._hecktor_like())
        assert (out_iii["tracer_is_psma"] == 1).all()
        assert (out_i["tracer_is_psma"] == 0).all()
        assert (out_h["tracer_is_psma"] == 0).all()

    def test_vendor_one_hot_consistent(self):
        from src.conformal.weighted import build_extended_wcp_features
        out = build_extended_wcp_features(self._autopet_i_like())
        assert (out["vendor_is_siemens"] == 1).all()
        assert (out["vendor_is_ge"] == 0).all()
        assert (out["vendor_is_philips"] == 0).all()

    def test_centre_indicators_zero_for_non_hecktor(self):
        from src.conformal.weighted import build_extended_wcp_features
        out_i = build_extended_wcp_features(self._autopet_i_like())
        for cid in (1, 2, 3, 5, 6, 7, 8):
            assert (out_i[f"centre_{cid}"] == 0).all()

    def test_centre_indicators_set_for_hecktor(self):
        from src.conformal.weighted import build_extended_wcp_features
        out_h = build_extended_wcp_features(self._hecktor_like())
        assert (out_h["centre_5"] == 1).all()
        for cid in (1, 2, 3, 6, 7, 8):
            assert (out_h[f"centre_{cid}"] == 0).all()

    def test_voxel_volume_computed(self):
        from src.conformal.weighted import build_extended_wcp_features
        out = build_extended_wcp_features(self._autopet_i_like())
        # 2.04 * 2.04 * 3.0 / 1000 ~ 0.01249
        assert np.allclose(out["voxel_volume_ml"], 2.04 * 2.04 * 3.0 / 1000.0)

    def test_lesions_per_patient_is_cohort_median(self):
        from src.conformal.weighted import build_extended_wcp_features
        df = pd.DataFrame({
            "volume_ml": [1, 2, 3, 4, 5],
            "surface_area_cm2": [1, 2, 3, 4, 5],
            "sphericity": [0.5] * 5,
            "case_id": ["A", "A", "B", "C", "C"],  # A has 2, B has 1, C has 2 -> median 2
            "tracer": ["FDG"] * 5,
            "vendor": ["Siemens"] * 5,
        })
        out = build_extended_wcp_features(df)
        assert (out["lesions_per_patient_in_cohort"] == 2.0).all()


class TestSupportOverlapDiagnostic:
    """Amendment 12 §12b: support-overlap verdict thresholds."""

    def test_green_when_well_overlapping(self):
        from src.conformal.weighted import diagnose_support_overlap
        weights = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 1.0, 0.95, 1.05, 1.1] * 50)
        d = diagnose_support_overlap(classifier_auc=0.55, weights=weights)
        assert d.verdict == "green"
        assert d.flagged_reasons == []

    def test_red_when_auc_above_0_99(self):
        from src.conformal.weighted import diagnose_support_overlap
        weights = np.array([0.001] * 50 + [1000.0] * 50)  # extreme separation
        d = diagnose_support_overlap(classifier_auc=0.999, weights=weights)
        assert d.verdict == "red"
        assert any("classifier_auc" in r for r in d.flagged_reasons)

    def test_amber_when_auc_above_0_95(self):
        from src.conformal.weighted import diagnose_support_overlap
        weights = np.array([0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5] * 50)
        d = diagnose_support_overlap(classifier_auc=0.96, weights=weights)
        assert d.verdict == "amber"

    def test_amber_when_dispersion_above_100(self):
        from src.conformal.weighted import diagnose_support_overlap
        weights = np.array([0.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 10)
        d = diagnose_support_overlap(classifier_auc=0.55, weights=weights)
        # dispersion = 1.0 / 0.001 = 1000 > 100
        assert d.verdict == "amber"
        assert any("weight_dispersion" in r for r in d.flagged_reasons)

    def test_amber_when_ess_ratio_below_0_30(self):
        from src.conformal.weighted import diagnose_support_overlap
        # very heavy-tailed weights -> ESS / n is small
        weights = np.array([100.0] + [0.01] * 99)
        d = diagnose_support_overlap(classifier_auc=0.5, weights=weights)
        assert d.verdict in ("amber", "red")
        assert any("ess_ratio" in r for r in d.flagged_reasons) or any("weight_dispersion" in r for r in d.flagged_reasons)
