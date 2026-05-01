"""Sensitivity analysis: per-cohort recalibration robustness to holdout seed.

Action item PAI #6 from manuscript_evaluation_2026-05-01_v2.md.

Reruns the Mode 4 (per-cohort split-conformal recalibration) analysis with
multiple holdout seeds to confirm that the SA-31 PASS verdicts are not seed-specific.

The locked Phase 3 freeze used PERCOHORT_HOLDOUT_SEED = 42 (Amendment 11 §11d).
This script reports coverage and miss-pp at seed=42 (replication of locked) plus
seeds {1, 17, 99} for sensitivity.

Output: prints a table; writes results/phase3/amendment_11/sensitivity_seed.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.conformal.cqr import (
    calibrate_cqr,
    predict_intervals_array,
    train_quantile_regressors,
)
from src.conformal.coverage import compute_coverage
from src.conformal.weighted import patient_level_holdout_split
from scripts.phase3_evaluate import (
    ALPHA, NOMINAL, FEATURE_COLS,
    load_autopet_i, load_autopet_iii, load_hecktor,
    transform_target,
)


SEEDS = [42, 1, 17, 99]
OUT_DIR = PROJECT_ROOT / "results/phase3/amendment_11"
OUT_PATH = OUT_DIR / "sensitivity_seed.csv"


def main() -> None:
    print(f"Per-cohort recalibration seed sensitivity: seeds={SEEDS}")
    print("=" * 78)

    # Load cohorts (same loaders as locked driver)
    train, cal, test = load_autopet_i()
    autopet_iii = load_autopet_iii()
    hecktor = load_hecktor()

    # Train CQR base model on AutoPET-I train (same as locked driver)
    y_train = transform_target(train["suvmax"].to_numpy())
    X_train = train[FEATURE_COLS].to_numpy()
    lower_model, upper_model = train_quantile_regressors(X_train, y_train, alpha=ALPHA)

    def X(df: pd.DataFrame) -> np.ndarray:
        return df[FEATURE_COLS].to_numpy()

    def y(df: pd.DataFrame) -> np.ndarray:
        return transform_target(df["suvmax"].to_numpy())

    # Predict for the two external cohorts
    lo_t3, hi_t3 = lower_model.predict(X(autopet_iii)), upper_model.predict(X(autopet_iii))
    lo_h, hi_h = lower_model.predict(X(hecktor)), upper_model.predict(X(hecktor))

    rows = []
    for cohort_name, cohort_df, lo_c, hi_c in [
        ("autopet_iii", autopet_iii, lo_t3, hi_t3),
        ("hecktor", hecktor, lo_h, hi_h),
    ]:
        for seed in SEEDS:
            ho_df, ev_df = patient_level_holdout_split(cohort_df, seed=seed)
            ho_idx = cohort_df.index.isin(ho_df.index)
            ev_idx = cohort_df.index.isin(ev_df.index)
            cqr_local = calibrate_cqr(
                y(ho_df), lo_c[ho_idx], hi_c[ho_idx],
                alpha=ALPHA, stratum_name=f"{cohort_name}_seed{seed}",
            )
            q_hat = cqr_local.q_hat
            cov = compute_coverage(
                y(ev_df), lo_c[ev_idx] - q_hat, hi_c[ev_idx] + q_hat,
                nominal=NOMINAL, label=f"{cohort_name}_seed{seed}",
            )
            miss_pp = (cov.coverage - NOMINAL) * 100.0
            rows.append({
                "cohort": cohort_name,
                "seed": seed,
                "n_holdout_lesions": int(len(ho_df)),
                "n_holdout_patients": int(ho_df["case_id"].nunique()),
                "n_eval_lesions": int(len(ev_df)),
                "q_hat_local": float(q_hat),
                "coverage": float(cov.coverage),
                "miss_pp": float(miss_pp),
                "ci_low": float(cov.ci_lower),
                "ci_high": float(cov.ci_upper),
                "median_width_log": float(cov.median_width),
                "within_2pp_tolerance": abs(miss_pp) <= 2.0,
            })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(df.to_string(index=False))
    print()

    # Per-cohort range
    print("=" * 78)
    print("Per-cohort coverage range across seeds {42, 1, 17, 99}:")
    for cohort in ("autopet_iii", "hecktor"):
        sub = df[df["cohort"] == cohort]
        cov_min, cov_max = sub["coverage"].min(), sub["coverage"].max()
        miss_min, miss_max = sub["miss_pp"].min(), sub["miss_pp"].max()
        all_pass = bool(sub["within_2pp_tolerance"].all())
        print(f"  {cohort}: coverage range {cov_min:.4f}-{cov_max:.4f} "
              f"(miss_pp {miss_min:+.2f} to {miss_max:+.2f}); "
              f"all seeds within +/-2pp: {all_pass}")
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
