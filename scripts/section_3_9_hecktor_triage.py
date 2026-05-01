"""HECKTOR §3.9 outlier triage + finalisation (Amendment 5 protocol applied to HECKTOR).

Applies the pre-reg §3.9 / Amendment 5 outlier triage to `hecktor_lesions.parquet`:

    Cat A: SUVmax > 150  -- physically extreme, almost always artifact
    Cat B: SUVmax > 50 AND volume_ml < 2  -- small high-SUV (artefact-prone)
    Cat C: SUVmax > 50 AND volume_ml >= 2 AND SUVpeak/SUVmax < 0.30  -- focal hotspot
    Cat D: SUVmax > 50 AND volume_ml >= 2 AND SUVpeak/SUVmax >= 0.30  -- auto-retain

Lesions with SUVmax <= 50 are auto-retained without triage (pre-reg §3.9).

If 0 lesions land in Cat A/B/C, no image review is needed -- the script writes
the reviewed parquet directly with all `section_3_9_review = "auto_retain"`.

If any lesions land in Cat A/B/C, the script writes a flag CSV and stops short
of finalisation; the reviewer (HF) then runs the standard render + decision
capture loop (`scripts/section_3_9_validation_render.py` adapted for HECKTOR)
before re-running this script with `--decisions PATH` to apply the decisions.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN sec 3.9
Amendment 5: AutoPET-III INSUFFICIENT_AGREEMENT branch (full-review protocol)
Amendment 8: HECKTOR external validation cohort (osf/amendment_log.md)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = PROJECT_ROOT / "data/interim/lesion_tables/hecktor_lesions.parquet"
DEFAULT_OUT = PROJECT_ROOT / "data/interim/lesion_tables/hecktor_lesions_reviewed.parquet"
DEFAULT_FLAG_CSV = PROJECT_ROOT / "results/tables/section_3_9_hecktor_flagged.csv"

# Pre-reg §3.9 thresholds (unchanged across cohorts)
SUVMAX_FLAG_THRESHOLD = 50.0
SUVMAX_EXTREME_THRESHOLD = 150.0
MIN_VOLUME_ML_THRESHOLD = 2.0
SUV_UNIFORMITY_THRESHOLD = 0.30


def categorise(row: pd.Series) -> str:
    """Pre-reg §3.9 Cat A/B/C/D, or 'auto_retain' if SUVmax<=50."""
    suvmax = float(row["suvmax"])
    if suvmax <= SUVMAX_FLAG_THRESHOLD:
        return "auto_retain"
    if suvmax > SUVMAX_EXTREME_THRESHOLD:
        return "A_extreme_suv"
    if float(row["volume_ml"]) < MIN_VOLUME_ML_THRESHOLD:
        return "B_small_high_suv"
    suvpeak = float(row["suvpeak"])
    if suvmax > 0 and (suvpeak / suvmax) < SUV_UNIFORMITY_THRESHOLD:
        return "C_focal_hotspot"
    return "D_likely_real"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                    help="Path to hecktor_lesions.parquet (pre-§3.9)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Path to write hecktor_lesions_reviewed.parquet")
    ap.add_argument("--flag-csv", type=Path, default=DEFAULT_FLAG_CSV,
                    help="Path to write Cat A/B/C/D flag table (if any flagged)")
    ap.add_argument("--decisions", type=Path, default=None,
                    help="Optional reviewer decisions CSV (case_id, lesion_id, decision in {retain,exclude}) to apply")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    n_total = len(df)
    print(f"Loaded {n_total} lesions from {args.parquet}")
    print(f"Source SHA-256: {_sha256(args.parquet)}")
    print()

    df = df.copy()
    df.loc[:, "suv_uniformity"] = df["suvpeak"] / df["suvmax"].clip(lower=1e-9)
    df.loc[:, "section_3_9_category"] = df.apply(categorise, axis=1)

    cat_counts = df["section_3_9_category"].value_counts().sort_index()
    print("=== §3.9 triage breakdown ===")
    for cat, n in cat_counts.items():
        print(f"  {cat:<22} {n}")
    print()

    flagged = df[df["section_3_9_category"].isin(
        ["A_extreme_suv", "B_small_high_suv", "C_focal_hotspot"]
    )]
    auto_d = df[df["section_3_9_category"] == "D_likely_real"]
    print(f"Flagged for image review (Cat A/B/C): {len(flagged)}")
    print(f"Auto-retain Cat D (50<SUVmax<=150 with normal uniformity, vol>=2 mL): {len(auto_d)}")
    print(f"Auto-retain (SUVmax<=50): {(df['section_3_9_category'] == 'auto_retain').sum()}")
    print()

    if len(flagged) > 0:
        args.flag_csv.parent.mkdir(parents=True, exist_ok=True)
        flag_cols = [
            "case_id", "lesion_id", "section_3_9_category",
            "suvmax", "suvpeak", "suvmean", "suv_uniformity",
            "volume_ml", "lesion_class", "centre_id", "vendor",
        ]
        flagged[flag_cols].to_csv(args.flag_csv, index=False)
        print(f"Wrote flag table: {args.flag_csv}")
        print()

    # Apply reviewer decisions if provided
    if args.decisions is not None:
        decisions = pd.read_csv(args.decisions)
        excluded = set(zip(
            decisions[decisions["decision"] == "exclude"]["case_id"],
            decisions[decisions["decision"] == "exclude"]["lesion_id"].astype(int),
        ))
        df["section_3_9_excluded"] = df.apply(
            lambda r: (r["case_id"], int(r["lesion_id"])) in excluded, axis=1
        )
        print(f"Applied {len(excluded)} exclusion decisions from {args.decisions}")
    else:
        # Auto-finalise path (no flagged lesions OR called with no decisions to apply)
        if len(flagged) > 0:
            print()
            print("WARNING: Cat A/B/C lesions exist but no --decisions CSV supplied.")
            print("This run did NOT write a reviewed parquet. Next steps:")
            print("  1. Render Cat A/B/C lesions for image review (adapt scripts/section_3_9_validation_render.py for HECKTOR)")
            print("  2. Capture reviewer decisions in CSV with columns: case_id, lesion_id, decision (retain|exclude)")
            print(f"  3. Re-run: python3 {Path(__file__).name} --decisions <path/to/decisions.csv>")
            return
        df.loc[:, "section_3_9_excluded"] = False
        print("No Cat A/B/C lesions; auto-finalising with all retained.")

    # Map category to final review label
    def review_label(row: pd.Series) -> str:
        cat = row["section_3_9_category"]
        if cat == "auto_retain":
            return "auto_retain"
        if cat == "D_likely_real":
            return "auto_retain"  # Cat D is auto-retained per pre-reg §3.9
        # Cat A/B/C with applied decisions
        if row.get("section_3_9_excluded", False):
            return "exclude"
        return "retain"

    df.loc[:, "section_3_9_review"] = df.apply(review_label, axis=1)

    # Drop the helper column before writing
    df = df.drop(columns=["suv_uniformity"])

    final_count = (~df["section_3_9_excluded"]).sum()
    excluded_count = df["section_3_9_excluded"].sum()
    print()
    print(f"Final cohort: {final_count} retained, {excluded_count} excluded "
          f"({excluded_count/n_total*100:.3f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    out_sha = _sha256(args.out)
    print()
    print(f"Wrote {args.out}")
    print(f"SHA-256: {out_sha}")
    print(f"Size:    {args.out.stat().st_size:,} bytes")

    # Audit JSON
    audit = {
        "input_parquet": str(args.parquet),
        "input_sha256": _sha256(args.parquet),
        "output_parquet": str(args.out),
        "output_sha256": out_sha,
        "n_total_lesions": int(n_total),
        "n_retained": int(final_count),
        "n_excluded": int(excluded_count),
        "exclusion_rate_pct": float(excluded_count / n_total * 100),
        "category_breakdown": {str(k): int(v) for k, v in cat_counts.items()},
        "thresholds": {
            "suvmax_flag": SUVMAX_FLAG_THRESHOLD,
            "suvmax_extreme": SUVMAX_EXTREME_THRESHOLD,
            "min_volume_ml": MIN_VOLUME_ML_THRESHOLD,
            "suv_uniformity": SUV_UNIFORMITY_THRESHOLD,
        },
    }
    audit_path = args.out.with_suffix("").with_name(args.out.stem + "_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Wrote audit: {audit_path}")


if __name__ == "__main__":
    main()
