"""§3.9 validation sampling -- AutoPET-III (Amendment 5).

Draws a stratified random sample of lesions from the SUVmax > 50 flagged
set, for the index-test-vs-image-review agreement validation.

Outputs two CSVs in `results/tables/section_3_9_validation/`:
- `sample_blinded.csv` -- what the reviewer sees during review (only review_id
  and empty `reviewer_decision`/`reviewer_notes` columns)
- `sample_key.csv` -- ground truth for analysis (review_id ↔ case_id +
  lesion_id + category + ratio + index-test predicted_decision)

The reviewer must NOT open `sample_key.csv` until decisions are complete.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (Amendment 5)

Usage
-----
    python scripts/section_3_9_validation_sample.py [--parquet PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Pre-registered constants (Amendment 5)
RANDOM_SEED = 42
SAMPLE_PER_STRATUM = {"A_extreme_suv": 12, "B_small_high_suv": 18, "C_focal_hotspot": 20}
INDEX_TEST_RATIO_THRESHOLD = 0.4  # retain if SUVpeak/SUVmax >= this

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/tables/section_3_9_validation"

# Default lookup paths for the AutoPET-III lesion parquet; pass --parquet to
# override (e.g. when running against a custom working directory).
PARQUET_CANDIDATES = [
    str(PROJECT_ROOT / "data/interim/lesion_tables/autopet_iii_lesions_reviewed.parquet"),
    str(PROJECT_ROOT / "data/interim/lesion_tables/autopet_iii_lesions.parquet"),
]


def find_parquet(arg_path: str | None) -> Path:
    if arg_path:
        p = Path(arg_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"--parquet path does not exist: {arg_path}")
    for c in PARQUET_CANDIDATES:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "autopet_iii_lesions.parquet not found at any default location. "
        "Pass --parquet PATH explicitly."
    )


def categorise(row: pd.Series) -> str:
    """§3.9 / Step 9 of process_autopet_iii.ipynb -- recomputed for sampling."""
    if row["suvmax"] > 150:
        return "A_extreme_suv"
    if row["volume_ml"] < 2.0:
        return "B_small_high_suv"
    if row["suv_uniformity"] < 0.30:
        return "C_focal_hotspot"
    return "D_likely_real"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=None, help="Path to autopet_iii_lesions.parquet")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                        help="Output directory for sample_blinded.csv + sample_key.csv")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_blinded = out_dir / "sample_blinded.csv"
    out_key = out_dir / "sample_key.csv"

    parquet_path = find_parquet(args.parquet)
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} lesion rows; {df['case_id'].nunique()} unique cases")

    # Compute uniformity + categorise the flagged subset
    flagged = df[df["suvmax"] > 50].copy()
    flagged["suv_uniformity"] = flagged["suvmean"] / flagged["suvmax"]
    flagged["suv_peakratio"] = flagged["suvpeak"] / flagged["suvmax"]
    flagged["triage_category"] = flagged.apply(categorise, axis=1)
    print(f"  Flagged (SUVmax > 50): {len(flagged)} lesions in "
          f"{flagged['case_id'].nunique()} cases")
    print(f"  Triage breakdown: {flagged['triage_category'].value_counts().to_dict()}")

    # Sample per-stratum with one-lesion-per-case cap GLOBALLY
    rng = np.random.RandomState(RANDOM_SEED)
    sampled_rows = []
    used_cases: set[str] = set()

    for stratum, n_target in SAMPLE_PER_STRATUM.items():
        stratum_df = flagged[flagged["triage_category"] == stratum].copy()
        # Shuffle within stratum
        shuffled_idx = rng.permutation(stratum_df.index.values)
        n_taken = 0
        for idx in shuffled_idx:
            if n_taken >= n_target:
                break
            row = stratum_df.loc[idx]
            if row["case_id"] in used_cases:
                continue
            sampled_rows.append(row)
            used_cases.add(row["case_id"])
            n_taken += 1
        print(f"  Stratum {stratum}: requested {n_target}, sampled {n_taken} "
              f"(stratum has {len(stratum_df)} lesions across "
              f"{stratum_df['case_id'].nunique()} cases)")

    sample = pd.DataFrame(sampled_rows).reset_index(drop=True)

    # Randomise sample order (so reviewer doesn't see categories grouped)
    sample = sample.sample(frac=1, random_state=rng).reset_index(drop=True)
    sample["review_id"] = range(1, len(sample) + 1)

    # Index test (ratio-based) prediction
    sample["index_predicted_decision"] = np.where(
        sample["suv_peakratio"] >= INDEX_TEST_RATIO_THRESHOLD, "retain", "exclude"
    )

    # --- Write outputs ---
    out_dir.mkdir(parents=True, exist_ok=True)

    # Blinded CSV: ONLY review_id + empty decision columns
    blinded_cols = ["review_id"]
    blinded = sample[blinded_cols].copy()
    blinded["reviewer_decision"] = ""        # must be 'retain' or 'exclude'
    blinded["reviewer_confidence"] = ""      # optional: high/medium/low
    blinded["reviewer_notes"] = ""
    blinded.to_csv(out_blinded, index=False)
    print(f"\nBlinded sample written to: {out_blinded}")
    print(f"  ({len(blinded)} rows; reviewer fills in `reviewer_decision` column)")

    # Key CSV: full ground-truth mapping
    key_cols = [
        "review_id", "case_id", "lesion_id", "series_uid",
        "triage_category", "suvmax", "suvpeak", "suvmean",
        "suv_uniformity", "suv_peakratio", "volume_ml",
        "centroid_0", "centroid_1", "centroid_2",
        "voxel_spacing_0", "voxel_spacing_1", "voxel_spacing_2",
        "tracer", "vendor", "radionuclide",
        "index_predicted_decision",
    ]
    key = sample[key_cols].copy()
    key.to_csv(out_key, index=False)
    print(f"Key (DO NOT OPEN UNTIL REVIEW IS COMPLETE) written to: {out_key}")

    # Summary for the user
    print(f"\n--- Sample summary ---")
    print(f"Total sample size: {len(sample)}")
    print(f"Per stratum:       {sample['triage_category'].value_counts().to_dict()}")
    print(f"Unique cases:      {sample['case_id'].nunique()}  (one-lesion-per-case cap honoured)")
    print(f"Index test predicts:")
    print(f"  retain:  {(sample['index_predicted_decision']=='retain').sum()}")
    print(f"  exclude: {(sample['index_predicted_decision']=='exclude').sum()}")

    # Hash for reproducibility log
    import hashlib
    sample_hash = hashlib.sha256(
        sample.sort_values("review_id").to_csv(index=False).encode()
    ).hexdigest()
    print(f"\nSample SHA-256 (deterministic given seed=42): {sample_hash}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
