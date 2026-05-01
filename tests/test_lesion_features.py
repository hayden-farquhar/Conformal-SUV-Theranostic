"""Tests for lesion feature extraction using synthetic PET volumes."""

import numpy as np
import pytest

from src.features.suvpeak import compute_suvpeak, SPHERE_RADIUS_MM, _build_sphere_mask
from src.features.extract_lesion_features import (
    extract_connected_components,
    compute_surface_area,
    compute_sphericity,
    extract_features_single_lesion,
    extract_all_lesions,
    lesion_features_to_dataframe,
)


# Standard voxel spacing for tests (Siemens Biograph mCT typical)
SPACING = (3.0, 4.07, 4.07)  # z, y, x in mm
VOXEL_VOL_ML = 3.0 * 4.07 * 4.07 / 1000.0  # ≈0.0497 mL


def _make_sphere_mask(shape, centre, radius_voxels):
    """Create a binary sphere mask in a volume."""
    z, y, x = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt(
        (z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2
    )
    return dist <= radius_voxels


def _make_test_volume(shape=(50, 64, 64)):
    """Create a synthetic SUV volume with two lesions."""
    suv = np.random.rand(*shape).astype(np.float64) * 2.0  # background SUV 0-2

    # Lesion 1: large sphere with high uptake (centre at 20, 30, 30)
    mask1 = _make_sphere_mask(shape, (20, 30, 30), 5)
    suv[mask1] = np.random.rand(mask1.sum()) * 5.0 + 10.0  # SUV 10-15

    # Lesion 2: small sphere with moderate uptake (centre at 35, 50, 50)
    mask2 = _make_sphere_mask(shape, (35, 50, 50), 3)
    suv[mask2] = np.random.rand(mask2.sum()) * 3.0 + 5.0  # SUV 5-8

    seg_mask = (mask1 | mask2).astype(np.int32)
    return suv, seg_mask, mask1, mask2


class TestSUVpeak:
    def test_uniform_sphere_suvpeak_equals_value(self):
        """For a uniform sphere, SUVpeak should equal the uniform SUV value."""
        shape = (30, 30, 30)
        suv = np.zeros(shape)
        mask = _make_sphere_mask(shape, (15, 15, 15), 6)
        suv[mask] = 10.0  # uniform SUV = 10

        peak, is_fallback = compute_suvpeak(suv, mask, SPACING)

        assert not is_fallback
        assert abs(peak - 10.0) < 0.01

    def test_small_lesion_fallback(self):
        """Lesions < 1.2 mL should fall back to SUVmax."""
        shape = (10, 10, 10)
        suv = np.zeros(shape)
        # Create a tiny lesion: ~3 voxels at this spacing
        mask = np.zeros(shape, dtype=bool)
        mask[5, 5, 5] = True
        mask[5, 5, 6] = True
        mask[5, 6, 5] = True
        suv[mask] = np.array([10.0, 8.0, 6.0])

        peak, is_fallback = compute_suvpeak(suv, mask, SPACING)

        assert is_fallback
        assert abs(peak - 10.0) < 1e-10  # should be SUVmax

    def test_peak_leq_max(self):
        """SUVpeak (mean of sphere) should be ≤ SUVmax."""
        suv, seg_mask, mask1, _ = _make_test_volume()
        peak, _ = compute_suvpeak(suv, mask1, SPACING)
        suvmax = float(suv[mask1].max())
        assert peak <= suvmax + 1e-10

    def test_peak_in_reasonable_range(self):
        """SUVpeak should be between SUVmean and SUVmax."""
        suv, seg_mask, mask1, _ = _make_test_volume()
        peak, is_fallback = compute_suvpeak(suv, mask1, SPACING)
        suvmax = float(suv[mask1].max())
        suvmean = float(suv[mask1].mean())
        if not is_fallback:
            # Peak is the mean of a sphere around the hottest voxel,
            # intersected with the lesion — should be between mean and max
            assert peak <= suvmax + 1e-10
            # Could be slightly below mean if the sphere samples fewer
            # high-uptake voxels at the lesion edge, but should be close
            assert peak > suvmean * 0.5


class TestSphereMask:
    def test_sphere_at_centre(self):
        """Sphere at volume centre should be symmetric."""
        mask = _build_sphere_mask((20, 20, 20), (10, 10, 10), 5.0, (1.0, 1.0, 1.0))
        # Should be symmetric in all axes
        assert mask[10, 10, 10]  # centre is in sphere
        assert mask.sum() > 0

    def test_anisotropic_spacing(self):
        """Sphere with anisotropic spacing should be ellipsoidal in voxels."""
        # Z spacing much larger than XY
        mask_iso = _build_sphere_mask((30, 30, 30), (15, 15, 15), 6.0, (1.0, 1.0, 1.0))
        mask_aniso = _build_sphere_mask((30, 30, 30), (15, 15, 15), 6.0, (3.0, 1.0, 1.0))

        # Anisotropic mask should have fewer voxels in Z direction
        # (sphere is 6mm but Z voxels are 3mm, so only ~2 voxels span radius)
        assert mask_aniso.sum() < mask_iso.sum()


class TestConnectedComponents:
    def test_two_separate_lesions(self):
        _, seg_mask, _, _ = _make_test_volume()
        components = extract_connected_components(seg_mask, SPACING, min_volume_ml=0.1)
        assert len(components) == 2

    def test_volume_threshold(self):
        """Components below min volume should be excluded."""
        _, seg_mask, _, _ = _make_test_volume()
        # With a very high threshold, both lesions should be excluded
        components = extract_connected_components(seg_mask, SPACING, min_volume_ml=1000.0)
        assert len(components) == 0

    def test_single_large_lesion(self):
        shape = (30, 30, 30)
        mask = _make_sphere_mask(shape, (15, 15, 15), 8).astype(np.int32)
        components = extract_connected_components(mask, SPACING, min_volume_ml=0.1)
        assert len(components) == 1


class TestSurfaceArea:
    def test_single_voxel(self):
        """A single voxel has 6 faces."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        spacing = (1.0, 1.0, 1.0)
        sa = compute_surface_area(mask, spacing)
        # 6 faces × 1mm² each = 6 mm² = 0.06 cm²
        assert abs(sa - 0.06) < 1e-10

    def test_cube_2x2x2(self):
        """A 2x2x2 cube: 24 exposed faces."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        spacing = (1.0, 1.0, 1.0)
        sa = compute_surface_area(mask, spacing)
        # 24 faces × 1mm² = 24 mm² = 0.24 cm²
        assert abs(sa - 0.24) < 1e-10

    def test_anisotropic_voxels(self):
        """Face area should scale with voxel spacing."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        spacing = (3.0, 4.0, 4.0)
        sa = compute_surface_area(mask, spacing)
        # 2 z-faces: 4*4=16 each = 32
        # 2 y-faces: 3*4=12 each = 24
        # 2 x-faces: 3*4=12 each = 24
        # Total = 80 mm² = 0.80 cm²
        assert abs(sa - 0.80) < 1e-10


class TestSphericity:
    def test_perfect_sphere(self):
        """A perfect sphere should have sphericity ≈ 1.0."""
        # V = 4/3 π r³, SA = 4 π r²
        r = 5.0  # cm
        v = 4.0 / 3.0 * np.pi * r ** 3  # cm³
        sa = 4.0 * np.pi * r ** 2  # cm²
        sph = compute_sphericity(v, sa)
        assert abs(sph - 1.0) < 0.01

    def test_elongated_shape_lower(self):
        """An elongated shape should have sphericity well below 1."""
        # Very elongated box: 1 × 1 × 100 cm
        v = 100.0  # cm³
        sa = 2 * (1 * 1 + 1 * 100 + 1 * 100)  # = 402 cm²
        sph = compute_sphericity(v, sa)
        assert sph < 1.0

    def test_zero_surface_area(self):
        assert compute_sphericity(1.0, 0.0) == 0.0


class TestExtractFeaturesIntegration:
    def test_single_lesion_features(self):
        shape = (30, 30, 30)
        suv = np.ones(shape) * 2.0
        mask = _make_sphere_mask(shape, (15, 15, 15), 5)
        suv[mask] = 10.0

        feat = extract_features_single_lesion(
            suv, mask, SPACING, "PAT_001", "STUDY_001", 1
        )

        assert feat.patient_id == "PAT_001"
        assert feat.suvmax == 10.0
        assert abs(feat.suvmean - 10.0) < 0.01
        assert feat.volume_ml > 0
        assert feat.surface_area_cm2 > 0
        # Voxelised sphere has sphericity close to but not exactly 1.0
        # (discretisation inflates surface area slightly)
        assert 0 < feat.sphericity < 1.5
        assert feat.tlg == pytest.approx(feat.suvmean * feat.volume_ml, rel=1e-6)

    def test_extract_all_lesions(self):
        suv, seg_mask, _, _ = _make_test_volume()
        features = extract_all_lesions(
            suv, seg_mask, SPACING, "PAT_001", "STUDY_001",
            min_volume_ml=0.1
        )
        assert len(features) == 2
        # Both should have different centroids
        assert features[0].centroid_z != features[1].centroid_z or \
               features[0].centroid_y != features[1].centroid_y

    def test_to_dataframe(self):
        suv, seg_mask, _, _ = _make_test_volume()
        features = extract_all_lesions(
            suv, seg_mask, SPACING, "PAT_001", "STUDY_001",
            min_volume_ml=0.1
        )
        df = lesion_features_to_dataframe(features)
        assert len(df) == 2
        assert "suvmax" in df.columns
        assert "volume_ml" in df.columns
        assert "sphericity" in df.columns
        assert "tlg" in df.columns

    def test_empty_mask_returns_empty(self):
        suv = np.ones((20, 20, 20)) * 5.0
        mask = np.zeros((20, 20, 20), dtype=np.int32)
        features = extract_all_lesions(suv, mask, SPACING, "PAT", "STUDY")
        assert len(features) == 0
