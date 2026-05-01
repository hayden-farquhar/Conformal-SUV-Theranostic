"""Tests for split allocation logic."""

import pandas as pd
import pytest

from src.preprocess.split_allocation import (
    allocate_autopet_i_splits,
    allocate_autopet_iii_splits,
    compute_manifest_hash,
    SEED,
)


def _make_patient_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic patient DataFrame for testing."""
    vendors = ["Siemens"] * (n // 2) + ["GE"] * (n - n // 2)
    tracers = ["FDG"] * (n * 3 // 5) + ["PSMA"] * (n - n * 3 // 5)
    return pd.DataFrame({
        "patient_id": [f"PAT_{i:04d}" for i in range(n)],
        "vendor": vendors,
        "tracer_category": tracers,
    })


class TestAutoPETISplits:
    def test_all_patients_assigned(self):
        df = _make_patient_df(200)
        result = allocate_autopet_i_splits(df)
        assert (result["split"] != "").all()
        assert len(result) == 200

    def test_split_proportions(self):
        """Splits should be approximately 40/20/20/20."""
        df = _make_patient_df(500)
        result = allocate_autopet_i_splits(df)
        counts = result["split"].value_counts()
        total = len(result)
        assert abs(counts["train"] / total - 0.40) < 0.05
        assert abs(counts["calibration"] / total - 0.20) < 0.05
        assert abs(counts["test"] / total - 0.20) < 0.05
        assert abs(counts["serial"] / total - 0.20) < 0.05

    def test_no_patient_in_multiple_splits(self):
        df = _make_patient_df(200)
        result = allocate_autopet_i_splits(df)
        # Each patient_id should appear exactly once
        assert result["patient_id"].is_unique

    def test_deterministic_with_same_seed(self):
        df = _make_patient_df(100)
        r1 = allocate_autopet_i_splits(df, seed=42)
        r2 = allocate_autopet_i_splits(df, seed=42)
        assert (r1["split"].values == r2["split"].values).all()

    def test_different_seed_different_splits(self):
        df = _make_patient_df(100)
        r1 = allocate_autopet_i_splits(df, seed=42)
        r2 = allocate_autopet_i_splits(df, seed=99)
        # Very unlikely to be identical with different seeds
        assert not (r1["split"].values == r2["split"].values).all()

    def test_stratification_balance(self):
        """Vendor proportions should be similar across splits."""
        df = _make_patient_df(500)
        result = allocate_autopet_i_splits(df)

        overall_siemens_pct = (df["vendor"] == "Siemens").mean()

        for split in ["train", "calibration", "test", "serial"]:
            split_df = result[result["split"] == split]
            split_siemens_pct = (split_df["vendor"] == "Siemens").mean()
            # Within 10pp of overall proportion
            assert abs(split_siemens_pct - overall_siemens_pct) < 0.10

    def test_works_without_stratification_columns(self):
        """Should work even without vendor/tracer columns."""
        df = pd.DataFrame({
            "patient_id": [f"PAT_{i}" for i in range(100)],
        })
        result = allocate_autopet_i_splits(df)
        assert len(result) == 100
        assert set(result["split"].unique()) == {"train", "calibration", "test", "serial"}


class TestAutoPETIIISplits:
    def test_all_external(self):
        df = pd.DataFrame({
            "patient_id": ["A", "B", "C"],
            "tracer_category": ["FDG", "PSMA", "FDG"],
        })
        result = allocate_autopet_iii_splits(df)
        assert (result["split"] == "external").all()

    def test_tracer_group_added(self):
        df = pd.DataFrame({
            "patient_id": ["A", "B"],
            "tracer_category": ["FDG", "PSMA"],
        })
        result = allocate_autopet_iii_splits(df)
        assert "tracer_group" in result.columns
        assert list(result["tracer_group"]) == ["FDG", "PSMA"]


class TestManifestHash:
    def test_deterministic(self):
        df = pd.DataFrame({
            "patient_id": ["A", "B", "C"],
            "split": ["train", "test", "calibration"],
        })
        h1 = compute_manifest_hash(df)
        h2 = compute_manifest_hash(df)
        assert h1 == h2

    def test_order_independent(self):
        """Hash should be the same regardless of row order."""
        df1 = pd.DataFrame({
            "patient_id": ["A", "B", "C"],
            "split": ["train", "test", "calibration"],
        })
        df2 = pd.DataFrame({
            "patient_id": ["C", "A", "B"],
            "split": ["calibration", "train", "test"],
        })
        assert compute_manifest_hash(df1) == compute_manifest_hash(df2)

    def test_different_splits_different_hash(self):
        df1 = pd.DataFrame({
            "patient_id": ["A", "B"],
            "split": ["train", "test"],
        })
        df2 = pd.DataFrame({
            "patient_id": ["A", "B"],
            "split": ["test", "train"],
        })
        assert compute_manifest_hash(df1) != compute_manifest_hash(df2)

    def test_hash_is_sha256(self):
        df = pd.DataFrame({"patient_id": ["A"], "split": ["train"]})
        h = compute_manifest_hash(df)
        assert len(h) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in h)
