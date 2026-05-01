"""§3.9 finalisation -- combine HF review decisions and apply to lesion parquet.

Joins the combined reviewer decisions CSV (49 validation + 200 full review) to the
review keys (which map review_id → case_id + lesion_id), then applies exclusions
to the autopet_iii_lesions.parquet, producing a reviewed parquet.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§3.9, Amendment 5)

Usage
-----
    python scripts/section_3_9_finalize.py \\
        [--parquet PATH] \\
        [--validation-key PATH] \\
        [--full-key PATH] \\
        [--decisions PATH] \\
        [--out PATH]

By default the script looks for files in:
- Parquet: data/interim/lesion_tables/autopet_iii_lesions.parquet (or the Drive path)
- Validation key: results/tables/section_3_9_validation/sample_key.csv
- Full review key: results/tables/section_3_9_validation/full_review_key.csv
- Combined decisions: results/tables/section_3_9_validation/section_3_9_combined_decisions.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "results/tables/section_3_9_validation"

# Default lookup paths for the AutoPET-III lesion parquet; pass --parquet to
# override (e.g. when running against a custom working directory).
PARQUET_CANDIDATES = [
    str(PROJECT_ROOT / "data/interim/lesion_tables/autopet_iii_lesions_reviewed.parquet"),
    str(PROJECT_ROOT / "data/interim/lesion_tables/autopet_iii_lesions.parquet"),
]


def find_parquet(arg_path: str | None) -> Path:
    if arg_path and Path(arg_path).exists():
        return Path(arg_path)
    for c in PARQUET_CANDIDATES:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "autopet_iii_lesions.parquet not found at default locations. "
        "Pass --parquet PATH or download the parquet from Drive first."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--validation-key", default=str(DEFAULT_DIR / "sample_key.csv"))
    parser.add_argument("--full-key", default=str(DEFAULT_DIR / "full_review_key.csv"))
    parser.add_argument("--decisions", default=str(DEFAULT_DIR / "section_3_9_combined_decisions.csv"))
    parser.add_argument("--out", default=None,
                        help="Output reviewed parquet path. Default: alongside input with _reviewed suffix.")
    args = parser.parse_args()

    parquet_path = find_parquet(args.parquet)
    out_path = Path(args.out) if args.out else parquet_path.parent / (parquet_path.stem + "_reviewed.parquet")

    print(f"Inputs:")
    print(f"  parquet:        {parquet_path}")
    print(f"  validation key: {args.validation_key}")
    print(f"  full key:       {args.full_key}")
    print(f"  decisions:      {args.decisions}")

    df = pd.read_parquet(parquet_path)
    print(f"\nLoaded {len(df)} lesion rows ({df['case_id'].nunique()} unique case_ids)")

    val_key = pd.read_csv(args.validation_key)
    full_key = pd.read_csv(args.full_key)
    keys = pd.concat([val_key, full_key]).drop_duplicates(subset="review_id")
    print(f"Combined review keys: {len(keys)} entries (validation {len(val_key)} + full {len(full_key)})")

    decisions = pd.read_csv(args.decisions)
    decisions["reviewer_decision"] = decisions["reviewer_decision"].astype(str).str.strip().str.lower()
    if "reviewer_confidence" in decisions.columns:
        decisions["reviewer_confidence"] = (
            decisions["reviewer_confidence"].astype(str).str.strip().str.lower()
        )
    print(f"Decisions loaded: {decisions['reviewer_decision'].value_counts().to_dict()}")

    # Validate all expected review_ids are decided
    missing = set(keys["review_id"]) - set(decisions["review_id"])
    if missing:
        print(f"\nWARNING: {len(missing)} review_ids in keys have no decision: "
              f"{sorted(missing)[:10]}...")

    # Identify exclusions
    excludes = decisions[decisions["reviewer_decision"] == "exclude"]["review_id"].tolist()
    excluded_keys = keys[keys["review_id"].isin(excludes)]
    print(f"\nExclusions to apply ({len(excluded_keys)}):")
    if len(excluded_keys) > 0:
        print(excluded_keys[["review_id", "case_id", "lesion_id", "series_uid",
                             "triage_category", "suvmax", "volume_ml"]].to_string(index=False))

    # Apply exclusions to the parquet using (case_id, lesion_id, series_uid)
    # as the join key. Note: lesion_id is a per-series connected-component label
    # from scipy.ndimage.label, so (case_id, lesion_id) alone is NOT a unique
    # identifier across series for the same patient. The reviewer was shown a
    # PNG generated from ONE specific (case_id, lesion_id, series_uid) tuple,
    # so the decision applies only to that exact row.
    excluded_triples = set(zip(
        excluded_keys["case_id"],
        excluded_keys["lesion_id"],
        excluded_keys["series_uid"],
    ))
    df["section_3_9_excluded"] = df.apply(
        lambda r: (r["case_id"], r["lesion_id"], r["series_uid"]) in excluded_triples,
        axis=1,
    )

    # Annotate decisions back to the parquet for traceability
    keys_indexed = keys.set_index("review_id")
    decisions_indexed = decisions.set_index("review_id")
    decision_by_triple = {}
    for rid in keys["review_id"]:
        if rid in decisions_indexed.index:
            triple = (
                keys_indexed.loc[rid, "case_id"],
                keys_indexed.loc[rid, "lesion_id"],
                keys_indexed.loc[rid, "series_uid"],
            )
            decision_by_triple[triple] = decisions_indexed.loc[rid, "reviewer_decision"]

    df["section_3_9_review"] = df.apply(
        lambda r: decision_by_triple.get(
            (r["case_id"], r["lesion_id"], r["series_uid"]), "auto_retain"
        ),
        axis=1,
    )

    # Summary
    print(f"\n--- Reviewed parquet summary ---")
    print(f"Total rows:                {len(df)}")
    print(f"  section_3_9_excluded=True: {df['section_3_9_excluded'].sum()}")
    print(f"  section_3_9_excluded=False: {(~df['section_3_9_excluded']).sum()}")
    print(f"  By section_3_9_review: {df['section_3_9_review'].value_counts().to_dict()}")

    # The "retained" cohort = rows with section_3_9_excluded=False
    retained = df[~df["section_3_9_excluded"]].copy()

    # Write outputs
    df.to_parquet(out_path, index=False)
    print(f"\nReviewed parquet written to: {out_path}")
    print(f"  Includes section_3_9_excluded + section_3_9_review columns")
    print(f"  Filter on section_3_9_excluded=False to get the cohort going into Phase 2/3")

    # SHA-256 for OSF audit trail
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"\nSHA-256 of {out_path.name}: {sha}")
    print(f"\n=== FINAL §3.9 OUTCOME for AutoPET-III ===")
    print(f"  Total lesions: {len(df)}")
    print(f"  Retained:      {(~df['section_3_9_excluded']).sum()} ({(~df['section_3_9_excluded']).mean()*100:.3f}%)")
    print(f"  Excluded:      {df['section_3_9_excluded'].sum()} ({df['section_3_9_excluded'].mean()*100:.3f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
