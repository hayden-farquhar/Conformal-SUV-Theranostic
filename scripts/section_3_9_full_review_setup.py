"""§3.9 full image review setup -- remaining lesions after Amendment 5 INSUFFICIENT_AGREEMENT.

Per Amendment 5's pre-specified action rule (kappa_lower_95CI < 0.60), the
ratio-based index test was insufficient to substitute for image review. We
therefore revert to full image review for the remaining flagged Cat A/B/C
lesions not already reviewed in the validation sample.

This script:
1. Loads the lesion parquet and computes §3.9 triage categories
2. Identifies all flagged review-needed lesions (Cat A/B/C, excluding Cat D auto-retain)
3. Excludes the 49 already in the validation sample (sample_key.csv)
4. Outputs the remaining as `full_review_blinded.csv` + `full_review_key.csv`
   with review_id numbering continuing from 50 onward
5. Random order (seed=43) to avoid category-grouped review

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (Amendment 5 INSUFFICIENT_AGREEMENT branch)

Usage
-----
    python scripts/section_3_9_full_review_setup.py [--parquet PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 43  # different from validation sample seed=42
INDEX_TEST_RATIO_THRESHOLD = 0.4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/tables/section_3_9_validation"
DEFAULT_VALIDATION_KEY = DEFAULT_OUT_DIR / "sample_key.csv"

PARQUET_CANDIDATES = [
    "/content/drive/MyDrive/P79 Data/autopet_iii/autopet_iii_lesions.parquet",
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
        "autopet_iii_lesions.parquet not found at default locations. "
        "Pass --parquet PATH explicitly."
    )


def categorise(row: pd.Series) -> str:
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
    parser.add_argument("--validation-key", default=str(DEFAULT_VALIDATION_KEY),
                        help="Path to the original sample_key.csv (49 already-reviewed)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_blinded = out_dir / "full_review_blinded.csv"
    out_key = out_dir / "full_review_key.csv"

    # Load lesion parquet
    parquet_path = find_parquet(args.parquet)
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} lesion rows, {df['case_id'].nunique()} unique cases")

    # Filter to AutoPET-III + flagged
    df = df[df["dataset"] == "autopet_iii"].copy()
    flagged = df[df["suvmax"] > 50].copy()
    flagged["suv_uniformity"] = flagged["suvmean"] / flagged["suvmax"]
    flagged["suv_peakratio"] = flagged["suvpeak"] / flagged["suvmax"]
    flagged["triage_category"] = flagged.apply(categorise, axis=1)
    review_needed = flagged[flagged["triage_category"] != "D_likely_real"].copy()
    print(f"  Total flagged (SUVmax > 50): {len(flagged)}")
    print(f"  Review-needed (Cat A/B/C):    {len(review_needed)}")
    print(f"  Triage breakdown: "
          f"{review_needed['triage_category'].value_counts().to_dict()}")

    # Load validation sample key to identify already-reviewed
    print(f"\nLoading validation sample key: {args.validation_key}")
    val_key = pd.read_csv(args.validation_key)
    already_reviewed_ids = set(zip(val_key["case_id"], val_key["lesion_id"]))
    print(f"  Validation sample: {len(val_key)} lesions already reviewed")

    # Filter remaining
    remaining = review_needed[
        ~review_needed.apply(
            lambda r: (r["case_id"], r["lesion_id"]) in already_reviewed_ids,
            axis=1,
        )
    ].copy()
    print(f"\nRemaining to review: {len(remaining)} lesions across "
          f"{remaining['case_id'].nunique()} cases")
    print(f"  Triage breakdown: {remaining['triage_category'].value_counts().to_dict()}")

    # Random order (seed=43, different from validation seed=42)
    rng = np.random.RandomState(RANDOM_SEED)
    remaining = remaining.sample(frac=1, random_state=rng).reset_index(drop=True)

    # Continue review_id numbering from 50 (so combined dataset has unique ids 1-N)
    start_id = int(val_key["review_id"].max()) + 1
    remaining["review_id"] = range(start_id, start_id + len(remaining))

    # Index test prediction (carried for analysis but NOT shown to reviewer)
    remaining["index_predicted_decision"] = np.where(
        remaining["suv_peakratio"] >= INDEX_TEST_RATIO_THRESHOLD, "retain", "exclude"
    )

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    blinded = remaining[["review_id"]].copy()
    blinded["reviewer_decision"] = ""
    blinded["reviewer_confidence"] = ""
    blinded["reviewer_notes"] = ""
    blinded.to_csv(out_blinded, index=False)
    print(f"\nBlinded CSV written to: {out_blinded}")

    key_cols = [
        "review_id", "case_id", "lesion_id", "series_uid",
        "triage_category", "suvmax", "suvpeak", "suvmean",
        "suv_uniformity", "suv_peakratio", "volume_ml",
        "centroid_0", "centroid_1", "centroid_2",
        "voxel_spacing_0", "voxel_spacing_1", "voxel_spacing_2",
        "tracer", "vendor", "radionuclide",
        "index_predicted_decision",
    ]
    key = remaining[key_cols].copy()
    key.to_csv(out_key, index=False)
    print(f"Key (DO NOT OPEN UNTIL REVIEW IS COMPLETE) written to: {out_key}")

    # Summary
    print(f"\n--- Full review batch summary ---")
    print(f"Total lesions: {len(remaining)}")
    print(f"Per stratum:   {remaining['triage_category'].value_counts().to_dict()}")
    print(f"Unique cases:  {remaining['case_id'].nunique()}")
    print(f"Review IDs:    {start_id} ... {start_id + len(remaining) - 1}")

    # Hash for reproducibility
    import hashlib
    sample_hash = hashlib.sha256(
        remaining.sort_values("review_id").to_csv(index=False).encode()
    ).hexdigest()
    print(f"\nFull-review sample SHA-256 (deterministic given seed=43): {sample_hash}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
