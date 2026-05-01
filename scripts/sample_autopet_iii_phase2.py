"""Stratified random subsample of 50 PT series from the AutoPET-III post-§3.9
reviewed cohort, for the Amendment 6 primary Phase 2 reference.

Stratification: vendor (Siemens / GE) x radionuclide (F-18 / Ga-68) -> 4 strata.
Per-stratum allocation: proportional to each stratum's series count, with a
minimum-of-2 floor and integer rounding so the four strata sum to 50.
One series per patient (avoid leakage): if a patient has multiple PT series in
a stratum, only one is eligible (random pick within patient).

Pre-registered seed: 42 (matches AutoPET-I split allocation).

Output: results/tables/section_3_5_phase2_autopet_iii_subsample.csv
        results/tables/section_3_5_phase2_autopet_iii_subsample.json (manifest)

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN sec 3.5
Amendment 6: 2026-04-29
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = REPO_ROOT / "data/interim/lesion_tables/autopet_iii_lesions_reviewed.parquet"
DEFAULT_DIGEST = REPO_ROOT / "data/raw/autopet_iii/nbia_digest.xlsx"
TARGET_N = 50
SEED = 42


def canonicalise_radionuclide(r: str | float | None) -> str:
    if not isinstance(r, str):
        return "UNK"
    r = r.strip()
    if r in {"Ga-68", "^68^Gallium", "68Ga", "Ga68"}:
        return "Ga-68"
    if r in {"F-18", "18F", "F18", "^18^Fluorine"}:
        return "F-18"
    return r


def build_eligible_series_table(parquet_path: Path) -> pd.DataFrame:
    """One row per (case_id, series_uid) eligible for the subsample.

    Filters: section_3_9_excluded == False (post-§3.9 retained lesions only).
    Aggregates: vendor, radionuclide (canonicalised), n_lesions per series.
    """
    lesions = pd.read_parquet(parquet_path)
    lesions = lesions[~lesions["section_3_9_excluded"]].copy()
    lesions["radionuclide_canon"] = lesions["radionuclide"].map(canonicalise_radionuclide)
    series = (
        lesions.groupby(["case_id", "series_uid"], as_index=False)
        .agg(
            radionuclide=("radionuclide_canon", "first"),
            vendor=("vendor", "first"),
            n_lesions=("lesion_id", "size"),
        )
    )
    return series


def allocate_per_stratum(stratum_sizes: dict[tuple[str, str], int], target_n: int) -> dict[tuple[str, str], int]:
    """Largest-remainder proportional allocation with a min-of-2 floor.

    With 4 strata and target 50, each stratum gets max(2, round(p_k * 50)) and the
    largest residuals absorb any deficit / surplus to make the sum exact.
    """
    total = sum(stratum_sizes.values())
    if total == 0:
        return {k: 0 for k in stratum_sizes}
    raw = {k: target_n * v / total for k, v in stratum_sizes.items()}
    floor = {k: max(2, int(np.floor(v))) for k, v in raw.items()}
    deficit = target_n - sum(floor.values())
    # Allocate any remaining slots by largest fractional remainder
    remainders = sorted(raw.items(), key=lambda kv: kv[1] - np.floor(kv[1]), reverse=True)
    alloc = dict(floor)
    i = 0
    while deficit > 0:
        k = remainders[i % len(remainders)][0]
        alloc[k] += 1
        deficit -= 1
        i += 1
    while deficit < 0:
        # Surplus: trim the stratum with the smallest fractional remainder, but never below 2
        candidates = [k for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - np.floor(kv[1]))]
        for k in candidates:
            if alloc[k] > 2:
                alloc[k] -= 1
                deficit += 1
                if deficit == 0:
                    break
    return alloc


def stratified_subsample(series: pd.DataFrame, target_n: int = TARGET_N, seed: int = SEED) -> pd.DataFrame:
    """One series per patient per stratum, allocated proportionally.

    1. Group eligible series by (vendor, radionuclide).
    2. Within each stratum, randomly pick one series per patient (RNG seeded).
    3. Allocate stratum quotas proportional to per-stratum eligible-patient counts.
    4. Randomly sample without replacement up to the per-stratum quota.

    The final subsample has exactly target_n rows (one series per patient, never
    duplicating patients across strata because patients have a fixed vendor and
    -- in AutoPET-III -- a fixed radionuclide for any given series).
    """
    rng = np.random.RandomState(seed)
    series = series.copy()
    series.loc[:, "stratum"] = list(zip(series["vendor"], series["radionuclide"]))
    # Step 1: collapse to one-series-per-patient-per-stratum
    one_per_patient = []
    for (vendor, radionuclide, patient), grp in series.groupby(["vendor", "radionuclide", "case_id"]):
        if len(grp) == 1:
            one_per_patient.append(grp.iloc[0])
        else:
            pick = rng.randint(0, len(grp))
            one_per_patient.append(grp.iloc[pick])
    eligible = pd.DataFrame(one_per_patient).reset_index(drop=True)

    # Step 2: per-stratum allocation
    stratum_sizes = eligible.groupby("stratum").size().to_dict()
    quota = allocate_per_stratum(stratum_sizes, target_n)

    # Step 3: per-stratum random sampling
    picked = []
    for stratum, k in quota.items():
        pool = eligible[eligible["stratum"] == stratum]
        if k >= len(pool):
            picked.append(pool)
        else:
            idx = rng.choice(len(pool), size=k, replace=False)
            picked.append(pool.iloc[idx])
    out = pd.concat(picked, ignore_index=True).drop(columns=["stratum"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--target-n", type=int, default=TARGET_N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-csv", type=Path,
        default=REPO_ROOT / "results/tables/section_3_5_phase2_autopet_iii_subsample.csv",
    )
    ap.add_argument(
        "--out-json", type=Path,
        default=REPO_ROOT / "results/tables/section_3_5_phase2_autopet_iii_subsample.json",
    )
    args = ap.parse_args()

    series = build_eligible_series_table(args.parquet)
    sample = stratified_subsample(series, target_n=args.target_n, seed=args.seed)
    sample = sample.sort_values(["vendor", "radionuclide", "case_id"]).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out_csv, index=False)
    manifest = {
        "amendment_6_primary_reference_subsample": {
            "target_n": args.target_n,
            "actual_n": int(len(sample)),
            "seed": args.seed,
            "stratification": "vendor x radionuclide",
            "per_stratum": (
                sample.groupby(["vendor", "radionuclide"])
                .size()
                .reset_index(name="n")
                .to_dict(orient="records")
            ),
            "input_parquet": str(args.parquet),
            "input_filter": "section_3_9_excluded == False",
        }
    }
    with open(args.out_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== AutoPET-III Phase 2 subsample (Amendment 6 primary reference) ===")
    print(f"target_n={args.target_n}, actual_n={len(sample)}, seed={args.seed}")
    print()
    print("Per-stratum allocation:")
    print(sample.groupby(["vendor", "radionuclide"]).size().to_string())
    print()
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
