"""Tests for src/features/augment_lesion_features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.augment_lesion_features import augment_case


def _make_synthetic_case():
    """32^3 mask with two GT spheres, returns (lesion_df, seg, voxel_spacing).

    Lesion 1: sphere at (8,8,8), radius 3 -> ~123 voxels at 4mm iso = ~7.9 mL
    Lesion 2: sphere at (24,24,24), radius 4 -> ~257 voxels at 4mm iso = ~16.4 mL
    """
    seg = np.zeros((32, 32, 32), dtype=np.int32)
    z, y, x = np.ogrid[:32, :32, :32]
    l1 = (z - 8) ** 2 + (y - 8) ** 2 + (x - 8) ** 2 <= 9
    seg[l1] = 1
    l2 = (z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2 <= 16
    seg[l2] = 1

    spacing = (4.0, 4.0, 4.0)
    voxel_volume_ml = float(np.prod(spacing)) / 1000.0  # 0.064

    # Manual centroid + n_voxels per lesion (mimics what an existing parquet has)
    coords1 = np.argwhere(l1)
    coords2 = np.argwhere(l2)
    rows = pd.DataFrame([
        {
            "case_id": "TEST-001", "lesion_id": 1,
            "centroid_0": coords1[:, 0].mean(),
            "centroid_1": coords1[:, 1].mean(),
            "centroid_2": coords1[:, 2].mean(),
            "n_voxels": int(l1.sum()),
            "volume_ml": float(l1.sum()) * voxel_volume_ml,
            "suvmax": 18.0,
        },
        {
            "case_id": "TEST-001", "lesion_id": 2,
            "centroid_0": coords2[:, 0].mean(),
            "centroid_1": coords2[:, 1].mean(),
            "centroid_2": coords2[:, 2].mean(),
            "n_voxels": int(l2.sum()),
            "volume_ml": float(l2.sum()) * voxel_volume_ml,
            "suvmax": 9.0,
        },
    ])
    return rows, seg, spacing


class TestAugmentCase:
    def test_both_lesions_matched_and_filled(self):
        rows, seg, spacing = _make_synthetic_case()
        out_df, result = augment_case("TEST-001", rows, seg, spacing)
        assert result.n_lesions_in == 2
        assert result.n_lesions_matched == 2
        assert result.n_lesions_unmatched == 0
        assert out_df["surface_area_cm2"].notna().all()
        assert out_df["sphericity"].notna().all()
        # Both lesions are spheres -> sphericity should be reasonably close to 1
        for sph in out_df["sphericity"]:
            assert 0.5 < sph <= 1.0, f"sphere sphericity {sph} outside expected range"

    def test_existing_columns_preserved(self):
        rows, seg, spacing = _make_synthetic_case()
        out_df, _ = augment_case("TEST-001", rows, seg, spacing)
        # Existing columns must round-trip exactly
        for col in ("case_id", "lesion_id", "n_voxels", "volume_ml", "suvmax"):
            assert (out_df[col].values == rows[col].values).all()

    def test_centroid_far_from_any_component_unmatched(self):
        """If parquet centroid doesn't match any SEG component within tolerance,
        the lesion is left unmatched (NaN) rather than mismatched (silent corruption)."""
        rows, seg, spacing = _make_synthetic_case()
        # Move lesion 1's centroid far from any SEG component
        rows.loc[rows["lesion_id"] == 1, "centroid_0"] = 0.5
        rows.loc[rows["lesion_id"] == 1, "centroid_1"] = 0.5
        rows.loc[rows["lesion_id"] == 1, "centroid_2"] = 0.5
        out_df, result = augment_case("TEST-001", rows, seg, spacing)
        assert result.n_lesions_matched == 1
        assert result.n_lesions_unmatched == 1
        # The displaced lesion should have NaN
        l1_row = out_df[out_df["lesion_id"] == 1].iloc[0]
        assert np.isnan(l1_row["surface_area_cm2"])
        assert np.isnan(l1_row["sphericity"])
        # Lesion 2 unaffected
        l2_row = out_df[out_df["lesion_id"] == 2].iloc[0]
        assert np.isfinite(l2_row["surface_area_cm2"])

    def test_n_voxels_mismatch_unmatched(self):
        """If centroid matches but voxel-count check fails, leave NaN."""
        rows, seg, spacing = _make_synthetic_case()
        # Inflate parquet's n_voxels to mismatch the actual component
        rows.loc[rows["lesion_id"] == 1, "n_voxels"] = int(rows.loc[rows["lesion_id"] == 1, "n_voxels"].iloc[0]) * 2
        out_df, result = augment_case("TEST-001", rows, seg, spacing)
        assert result.n_lesions_unmatched == 1

    def test_empty_seg_all_unmatched(self):
        rows, _, spacing = _make_synthetic_case()
        empty_seg = np.zeros((32, 32, 32), dtype=np.int32)
        out_df, result = augment_case("TEST-001", rows, empty_seg, spacing)
        assert result.n_lesions_matched == 0
        assert result.n_lesions_unmatched == 2
        assert out_df["surface_area_cm2"].isna().all()

    def test_multi_class_seg_treated_as_binary(self):
        """SEG with class labels 1 (GTVp) and 2 (GTVn) should label both as one
        binary mask (HECKTOR-style multi-class behaviour)."""
        rows, seg, spacing = _make_synthetic_case()
        # Make lesion 2 a different class
        z, y, x = np.ogrid[:32, :32, :32]
        l2 = (z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2 <= 16
        seg[l2] = 2
        out_df, result = augment_case("TEST-001", rows, seg, spacing)
        # Both lesions still matched (binary handling treats both as foreground)
        assert result.n_lesions_matched == 2
