"""Split allocation for conformal SUV study.

Pre-registered split design (§3.4):

AutoPET-I (FDG, single-site):
    train       40%  - CQR quantile regressor training
    calibration 20%  - conformal calibration (nonconformity score quantiles)
    test        20%  - primary internal evaluation
    serial      20%  - test-retest pairs; PERCIST serial evaluation

AutoPET-III (PSMA + FDG, multi-site):
    Full external validation — no splitting for calibration.
    Used entirely for cross-tracer, cross-vendor evaluation.

All splits are patient-level (no lesion leakage) and stratified by
scanner vendor and tracer.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§3.4)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# Pre-registered constants
SEED = 42
AUTOPET_I_FRACTIONS = {
    "train": 0.40,
    "calibration": 0.20,
    "test": 0.20,
    "serial": 0.20,
}


def allocate_autopet_i_splits(
    patient_df: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Allocate AutoPET-I patients to train/calibration/test/serial splits.

    Parameters
    ----------
    patient_df : pd.DataFrame
        One row per patient. Must have columns: 'patient_id'.
        Optional stratification columns: 'vendor', 'tracer_category'.
    seed : int
        Random seed (pre-registered: 42).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'split' column.
    """
    df = patient_df.copy()
    n = len(df)
    rng = np.random.RandomState(seed)

    # Build stratification key from available columns
    strat_cols = [c for c in ["vendor", "tracer_category"] if c in df.columns]
    if strat_cols:
        df["_strat"] = df[strat_cols].astype(str).agg("_".join, axis=1)
    else:
        df["_strat"] = "all"

    # Allocate in two passes:
    # Pass 1: 60% train+cal vs 40% test+serial
    # Pass 2: within each group, split further

    indices = np.arange(n)
    strat = df["_strat"].values

    # Split into train_cal (60%) and test_serial (40%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    train_cal_idx, test_serial_idx = next(sss1.split(indices, strat))

    # Within train_cal (60%), split into train (40/60 = 2/3) and cal (20/60 = 1/3)
    strat_tc = strat[train_cal_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 3, random_state=seed + 1)
    train_rel_idx, cal_rel_idx = next(sss2.split(train_cal_idx, strat_tc))
    train_idx = train_cal_idx[train_rel_idx]
    cal_idx = train_cal_idx[cal_rel_idx]

    # Within test_serial (40%), split into test (20/40 = 1/2) and serial (20/40 = 1/2)
    strat_ts = strat[test_serial_idx]
    sss3 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 2)
    test_rel_idx, serial_rel_idx = next(sss3.split(test_serial_idx, strat_ts))
    test_idx = test_serial_idx[test_rel_idx]
    serial_idx = test_serial_idx[serial_rel_idx]

    # Assign splits
    df["split"] = ""
    df.iloc[train_idx, df.columns.get_loc("split")] = "train"
    df.iloc[cal_idx, df.columns.get_loc("split")] = "calibration"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    df.iloc[serial_idx, df.columns.get_loc("split")] = "serial"

    # Drop helper column
    df = df.drop(columns=["_strat"])

    assert (df["split"] != "").all(), "Some patients not assigned to a split"

    return df


def allocate_autopet_iii_splits(
    study_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mark all AutoPET-III studies as external validation.

    AutoPET-III is used entirely for external validation (no internal splitting).
    A 'tracer_group' column is added for convenience in cross-tracer analyses.

    Parameters
    ----------
    study_df : pd.DataFrame
        AutoPET-III study metadata (from build_metadata).

    Returns
    -------
    pd.DataFrame
        With added 'split' and 'tracer_group' columns.
    """
    df = study_df.copy()
    df["split"] = "external"

    # Add tracer group for within-dataset cross-tracer analysis
    if "tracer_category" in df.columns:
        df["tracer_group"] = df["tracer_category"]
    elif "tracer" in df.columns:
        df["tracer_group"] = df["tracer"].map({"68Ga": "PSMA", "18F": "FDG"}).fillna("Other")

    return df


def compute_manifest_hash(split_df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of the split manifest.

    The hash covers patient_id + split assignment, ensuring the split
    allocation is frozen and tamper-evident.

    Pre-registration requires this hash to be recorded BEFORE any
    conformal calibration (§3.4).

    Parameters
    ----------
    split_df : pd.DataFrame
        Must have 'patient_id' and 'split' columns.

    Returns
    -------
    str
        Hex SHA-256 hash.
    """
    # Sort deterministically
    manifest = split_df[["patient_id", "split"]].sort_values("patient_id").reset_index(drop=True)
    manifest_json = manifest.to_json(orient="records", indent=None)
    return hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def save_split_manifest(
    split_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Save split manifest as parquet and return SHA-256 hash.

    Parameters
    ----------
    split_df : pd.DataFrame
        DataFrame with 'patient_id' and 'split' columns.
    output_path : str or Path
        Path to save the parquet file.

    Returns
    -------
    str
        SHA-256 hash of the manifest.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_df.to_parquet(output_path, index=False)

    manifest_hash = compute_manifest_hash(split_df)
    print(f"Split manifest saved to {output_path}")
    print(f"SHA-256: {manifest_hash}")

    return manifest_hash


def print_split_summary(df: pd.DataFrame, dataset_name: str = "") -> None:
    """Print split allocation summary with stratification balance."""
    prefix = f"[{dataset_name}] " if dataset_name else ""

    print(f"\n{prefix}Split allocation summary")
    print("=" * 50)
    print(f"Total: {len(df)} entries")
    print(f"\nSplit sizes:")
    counts = df["split"].value_counts()
    for split, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {split:15s}: {count:5d} ({pct:.1f}%)")

    # Stratification balance
    for col in ["vendor", "tracer_category", "tracer_group", "scanner_model"]:
        if col in df.columns:
            print(f"\n{col} x split:")
            ct = pd.crosstab(df["split"], df[col])
            # Show as percentages within each split
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            print(ct_pct.round(1).to_string())


if __name__ == "__main__":
    from pathlib import Path

    base = Path(__file__).resolve().parents[2]

    # Load AutoPET-III metadata
    meta_path = base / "data" / "interim" / "autopet_iii_study_metadata.parquet"
    if not meta_path.exists():
        print(f"Run build_metadata.py first: {meta_path} not found")
        raise SystemExit(1)

    study_df = pd.read_parquet(meta_path)

    # For AutoPET-III: all external
    iii_splits = allocate_autopet_iii_splits(study_df)
    print_split_summary(iii_splits, "AutoPET-III")

    # Demo: simulate AutoPET-I split allocation on the same data
    # (In production, this would use AutoPET-I patient list)
    # For now, show what the split would look like with AutoPET-III as a proxy
    patient_df = study_df.drop_duplicates("patient_id")[["patient_id", "vendor", "tracer_category"]].copy()
    i_splits = allocate_autopet_i_splits(patient_df)
    print_split_summary(i_splits, "AutoPET-I (simulated on III patients)")

    # Save manifests
    out_dir = base / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    hash_iii = save_split_manifest(iii_splits, out_dir / "autopet_iii_splits.parquet")
    hash_i = save_split_manifest(i_splits, out_dir / "autopet_i_splits_simulated.parquet")

    print(f"\nAutoPET-III manifest hash: {hash_iii}")
    print(f"AutoPET-I (simulated) manifest hash: {hash_i}")
