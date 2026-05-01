"""Enumerate AutoPET-III same-patient serial PT pair counts (§3.5 gate).

Pre-reg §3.5 reads "stable-disease pairs from AutoPET" without restricting cohort.
The earlier (2026-04-28) gate decision counted AutoPET-I pairs only and concluded
zero in-window pairs. The P80 handoff later that day surfaced that AutoPET-III has
substantial multi-series structure (135 of 333 cohort patients have >=2 PT series),
which the earlier gate did not enumerate. This script closes that audit gap.

Inputs:
  - data/interim/lesion_tables/autopet_iii_lesions_reviewed.parquet (cohort: 333 pts / 497 series)
  - data/raw/autopet_iii/nbia_digest.xlsx (StudyDate per series_uid)

Outputs (stdout + JSON):
  - Pair counts at 4 / 8 / 12 / 16 / 26 / 52-week windows
  - Stratified by radionuclide-matched (legitimate test-retest candidates) vs
    radionuclide-mismatched (18F-PSMA vs 68Ga-PSMA -- not test-retest)
  - Pre-reg §3.5 gate verdict (>=50 pairs at 8 weeks, radionuclide-matched only)

Gate rule (combined cohort, radionuclide-matched only):
  N_AutoPET-I (=0) + N_AutoPET-III >= 50  -->  primary path; file Amendment 6
  otherwise                                -->  Poisson-noise fallback (already pre-registered)

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (sec 3.5)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = REPO_ROOT / "data/interim/lesion_tables/autopet_iii_lesions_reviewed.parquet"
DEFAULT_DIGEST = REPO_ROOT / "data/raw/autopet_iii/nbia_digest.xlsx"

# Pre-reg threshold (sec 3.5)
PRIMARY_GATE_PAIRS = 50
PRIMARY_GATE_WEEKS = 8
WINDOW_WEEKS = (4, 8, 12, 16, 26, 52)
# AutoPET-I in-window pair count from earlier enumeration (kaggle_notebooks/enumerate_autopet_i_serial_pairs.py)
AUTOPET_I_PAIRS_8W = 0


def parse_studydate(s: str) -> datetime | None:
    """NBIA digest StudyDate is ``MM-DD-YYYY``; tolerate ``YYYY-MM-DD`` too."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def canonicalise_radionuclide(r: str | float | None) -> str:
    """Two encoding variants for Ga-68 in the parquet (per PROGRESS.md 2026-04-28).

    Collapse them so radionuclide-matched grouping works.
    """
    if not isinstance(r, str):
        return "UNK"
    r = r.strip()
    if r in {"Ga-68", "^68^Gallium", "68Ga", "Ga68"}:
        return "Ga-68"
    if r in {"F-18", "18F", "F18", "^18^Fluorine"}:
        return "F-18"
    return r


def load_series_table(parquet_path: Path, digest_path: Path) -> pd.DataFrame:
    """Build one-row-per-series table with: case_id, series_uid, study_date, radionuclide, vendor, tracer."""
    lesions = pd.read_parquet(parquet_path)
    lesions = lesions[~lesions["section_3_9_excluded"]].copy()
    series = (
        lesions.groupby("series_uid", as_index=False)
        .agg(
            case_id=("case_id", "first"),
            radionuclide=("radionuclide", "first"),
            vendor=("vendor", "first"),
            tracer=("tracer", "first"),
            n_lesions=("lesion_id", "size"),
        )
        .copy()
    )
    series.loc[:, "radionuclide"] = series["radionuclide"].map(canonicalise_radionuclide)

    digest_full = pd.read_excel(digest_path, sheet_name="Metadata")
    digest = digest_full.loc[
        digest_full["Modality"] == "PT",
        ["SeriesInstanceUID", "PatientID", "StudyDate", "Manufacturer"],
    ].rename(columns={"SeriesInstanceUID": "series_uid", "StudyDate": "study_date_raw"}).copy()
    digest.loc[:, "study_date"] = digest["study_date_raw"].map(parse_studydate)

    merged = series.merge(digest[["series_uid", "study_date"]], on="series_uid", how="left")
    missing = merged["study_date"].isna().sum()
    if missing:
        raise RuntimeError(
            f"{missing} series_uid(s) lack a parseable StudyDate in the NBIA digest -- "
            "cannot enumerate intervals."
        )
    return merged


def enumerate_pairs(series: pd.DataFrame, max_weeks: int) -> list[dict]:
    """All within-patient PT pairs with interval <= max_weeks. Annotated with radionuclide match."""
    max_interval = timedelta(weeks=max_weeks)
    pairs: list[dict] = []
    for case_id, grp in series.groupby("case_id"):
        if len(grp) < 2:
            continue
        ordered = grp.sort_values("study_date").reset_index(drop=True)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                interval = ordered.loc[j, "study_date"] - ordered.loc[i, "study_date"]
                if interval <= max_interval:
                    rn_i = ordered.loc[i, "radionuclide"]
                    rn_j = ordered.loc[j, "radionuclide"]
                    vendor_i = ordered.loc[i, "vendor"]
                    vendor_j = ordered.loc[j, "vendor"]
                    pairs.append(
                        {
                            "case_id": case_id,
                            "series_1": ordered.loc[i, "series_uid"],
                            "series_2": ordered.loc[j, "series_uid"],
                            "interval_days": interval.days,
                            "radionuclide_match": rn_i == rn_j,
                            "radionuclide_1": rn_i,
                            "radionuclide_2": rn_j,
                            "vendor_match": vendor_i == vendor_j,
                            "vendor_1": vendor_i,
                            "vendor_2": vendor_j,
                            "n_lesions_1": int(ordered.loc[i, "n_lesions"]),
                            "n_lesions_2": int(ordered.loc[j, "n_lesions"]),
                        }
                    )
    return pairs


def summarise(series: pd.DataFrame) -> dict:
    """Patient/series counts that don't depend on the window."""
    n_patients = series["case_id"].nunique()
    series_per_patient = series.groupby("case_id").size()
    n_multi = int((series_per_patient >= 2).sum())
    return {
        "n_patients_cohort": int(n_patients),
        "n_series_cohort": int(len(series)),
        "n_patients_multi_series": n_multi,
        "multi_series_distribution": series_per_patient.value_counts().sort_index().to_dict(),
        "radionuclide_distribution": series["radionuclide"].value_counts().to_dict(),
        "vendor_distribution": series["vendor"].value_counts().to_dict(),
    }


def window_sweep(series: pd.DataFrame, windows_weeks: tuple[int, ...]) -> dict:
    """Return cumulative pair counts at each window, plus radionuclide-match breakdown."""
    out = {}
    max_w = max(windows_weeks)
    all_pairs = enumerate_pairs(series, max_weeks=max_w)
    df = pd.DataFrame(all_pairs)
    if df.empty:
        return {f"{w}w": {"all_pairs": 0, "radionuclide_matched": 0, "radionuclide_mismatched": 0} for w in windows_weeks}
    for w in windows_weeks:
        sub = df[df["interval_days"] <= w * 7]
        out[f"{w}w"] = {
            "all_pairs": int(len(sub)),
            "radionuclide_matched": int(sub["radionuclide_match"].sum()),
            "radionuclide_mismatched": int((~sub["radionuclide_match"]).sum()),
            "vendor_matched_within_radionuclide_matched": int(
                ((sub["radionuclide_match"]) & (sub["vendor_match"])).sum()
            ),
        }
    return out


def gate_verdict(autopet_iii_matched_8w: int) -> dict:
    combined = AUTOPET_I_PAIRS_8W + autopet_iii_matched_8w
    primary = combined >= PRIMARY_GATE_PAIRS
    return {
        "autopet_i_pairs_8w": AUTOPET_I_PAIRS_8W,
        "autopet_iii_pairs_8w_radionuclide_matched": autopet_iii_matched_8w,
        "combined_pairs_8w_radionuclide_matched": combined,
        "threshold": PRIMARY_GATE_PAIRS,
        "primary_path_activated": primary,
        "verdict": "PRIMARY_PATH" if primary else "POISSON_FALLBACK",
        "amendment_6_required": primary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--digest", type=Path, default=DEFAULT_DIGEST)
    ap.add_argument(
        "--out-pairs",
        type=Path,
        default=REPO_ROOT / "results/tables/section_3_5_autopet_iii_pairs.csv",
        help="CSV: every in-window pair (max window) with radionuclide/vendor metadata.",
    )
    ap.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT / "results/tables/section_3_5_autopet_iii_summary.json",
        help="JSON: counts + window sweep + §3.5 gate verdict.",
    )
    args = ap.parse_args()

    series = load_series_table(args.parquet, args.digest)
    summary = summarise(series)
    sweep = window_sweep(series, WINDOW_WEEKS)
    verdict = gate_verdict(sweep[f"{PRIMARY_GATE_WEEKS}w"]["radionuclide_matched"])

    print("=== AutoPET-III §3.5 cohort summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print("=== Window sweep (cumulative pair counts) ===")
    print(f"{'window':<8}{'all':>8}{'rn-match':>12}{'rn-mismatch':>14}{'rn+vendor-match':>18}")
    for w in WINDOW_WEEKS:
        d = sweep[f"{w}w"]
        print(
            f"{w:>4}w  "
            f"{d['all_pairs']:>8}"
            f"{d['radionuclide_matched']:>12}"
            f"{d['radionuclide_mismatched']:>14}"
            f"{d['vendor_matched_within_radionuclide_matched']:>18}"
        )
    print()
    print("=== §3.5 GATE VERDICT (combined AutoPET-I + AutoPET-III, radionuclide-matched) ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(enumerate_pairs(series, max_weeks=max(WINDOW_WEEKS))).to_csv(args.out_pairs, index=False)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(
            {"summary": summary, "window_sweep": sweep, "gate_verdict": verdict},
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {args.out_pairs}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
