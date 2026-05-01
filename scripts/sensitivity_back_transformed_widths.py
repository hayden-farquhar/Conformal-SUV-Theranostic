"""Back-transformed median interval widths in natural-SUV units.

Action item PAI #7 from manuscript_evaluation_2026-05-01_v2.md.

The locked Phase 3 freeze reports interval widths in log(1 + SUVmax) space.
This script reproduces the four-mode prediction intervals, back-transforms to
natural SUVmax via exp(.) - 1, and reports median + IQR interval width per
cohort * mode.

Output: prints a table; writes results/phase3/amendment_11/widths_natural_suv.csv
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
    compute_nonconformity_scores,
    train_quantile_regressors,
)
from src.conformal.weighted import (
    PERCOHORT_HOLDOUT_FRACTION,
    PERCOHORT_HOLDOUT_SEED,
    WCP_EXTENDED_FEATURE_COLS,
    WCP_FEATURE_COLS,
    build_extended_wcp_features,
    fit_wcp_classifier,
    patient_level_holdout_split,
    weighted_conformal_threshold,
)
from scripts.phase3_evaluate import (
    ALPHA, NOMINAL, FEATURE_COLS,
    load_autopet_i, load_autopet_iii, load_hecktor,
    transform_target,
)

OUT_DIR = PROJECT_ROOT / "results/phase3/amendment_11"
OUT_PATH = OUT_DIR / "widths_natural_suv.csv"


def widths_natural(lo_log: np.ndarray, hi_log: np.ndarray, q_hat: float) -> np.ndarray:
    """Back-transform log(1+SUV) interval to natural-SUV interval and return widths."""
    lower_log = lo_log - q_hat
    upper_log = hi_log + q_hat
    lower_suv = np.exp(lower_log) - 1.0
    upper_suv = np.exp(upper_log) - 1.0
    lower_suv = np.maximum(lower_suv, 0.0)  # SUV is non-negative
    return upper_suv - lower_suv


def main() -> None:
    print("Back-transformed median interval widths in natural-SUV units")
    print("=" * 78)

    train, cal, test = load_autopet_i()
    autopet_iii = load_autopet_iii()
    hecktor = load_hecktor()

    y_train = transform_target(train["suvmax"].to_numpy())
    X_train = train[FEATURE_COLS].to_numpy()
    lower_model, upper_model = train_quantile_regressors(X_train, y_train, alpha=ALPHA)

    def X(df: pd.DataFrame) -> np.ndarray:
        return df[FEATURE_COLS].to_numpy()

    def y(df: pd.DataFrame) -> np.ndarray:
        return transform_target(df["suvmax"].to_numpy())

    # Predict on cal + all cohorts
    lo_cal, hi_cal = lower_model.predict(X(cal)), upper_model.predict(X(cal))
    lo_t1, hi_t1 = lower_model.predict(X(test)), upper_model.predict(X(test))
    lo_t3, hi_t3 = lower_model.predict(X(autopet_iii)), upper_model.predict(X(autopet_iii))
    lo_h, hi_h = lower_model.predict(X(hecktor)), upper_model.predict(X(hecktor))

    # Naive conformal threshold from AutoPET-I cal
    cal_cqr = calibrate_cqr(y(cal), lo_cal, hi_cal, alpha=ALPHA, stratum_name="autopet_i_cal")
    q_hat_naive = cal_cqr.q_hat
    print(f"q_hat_naive (AutoPET-I cal) = {q_hat_naive:.4f}")

    # Per-cohort recal q_hat values (using locked seed=42)
    ho_t3, ev_t3 = patient_level_holdout_split(autopet_iii)
    ho_t3_idx = autopet_iii.index.isin(ho_t3.index)
    cqr_t3 = calibrate_cqr(
        y(ho_t3), lo_t3[ho_t3_idx], hi_t3[ho_t3_idx],
        alpha=ALPHA, stratum_name="autopet_iii_holdout",
    )
    q_hat_t3_local = cqr_t3.q_hat

    ho_h, ev_h = patient_level_holdout_split(hecktor)
    ho_h_idx = hecktor.index.isin(ho_h.index)
    cqr_h = calibrate_cqr(
        y(ho_h), lo_h[ho_h_idx], hi_h[ho_h_idx],
        alpha=ALPHA, stratum_name="hecktor_holdout",
    )
    q_hat_h_local = cqr_h.q_hat

    # WCP-image weighted thresholds (use the same approach as locked driver)
    cal_features_image = cal[WCP_FEATURE_COLS].copy()
    autopet_iii_features_image = autopet_iii[WCP_FEATURE_COLS].copy()
    hecktor_features_image = hecktor[WCP_FEATURE_COLS].copy()

    wcp_t3 = fit_wcp_classifier(
        cal_features_image, autopet_iii_features_image,
        feature_cols=WCP_FEATURE_COLS,
    )
    cal_scores = compute_nonconformity_scores(y(cal), lo_cal, hi_cal)
    q_hat_W_t3 = weighted_conformal_threshold(cal_scores, wcp_t3.source_weights, alpha=ALPHA)

    wcp_h = fit_wcp_classifier(
        cal_features_image, hecktor_features_image,
        feature_cols=WCP_FEATURE_COLS,
    )
    q_hat_W_h = weighted_conformal_threshold(cal_scores, wcp_h.source_weights, alpha=ALPHA)

    rows = []

    # Mode 1: Naive transfer (q_hat_naive applied to all cohorts)
    for cohort_name, lo_c, hi_c in [
        ("autopet_i_test", lo_t1, hi_t1),
        ("autopet_iii", lo_t3, hi_t3),
        ("hecktor", lo_h, hi_h),
    ]:
        w_log = (hi_c + q_hat_naive) - (lo_c - q_hat_naive)
        w_suv = widths_natural(lo_c, hi_c, q_hat_naive)
        rows.append({
            "cohort": cohort_name,
            "mode": "naive",
            "n_lesions": int(len(lo_c)),
            "q_hat": float(q_hat_naive),
            "median_width_log": float(np.median(w_log)),
            "iqr25_width_log": float(np.percentile(w_log, 25)),
            "iqr75_width_log": float(np.percentile(w_log, 75)),
            "median_width_suv": float(np.median(w_suv)),
            "iqr25_width_suv": float(np.percentile(w_suv, 25)),
            "iqr75_width_suv": float(np.percentile(w_suv, 75)),
            "p90_width_suv": float(np.percentile(w_suv, 90)),
        })

    # Mode 2: WCP-image
    for cohort_name, lo_c, hi_c, q_hat in [
        ("autopet_iii", lo_t3, hi_t3, q_hat_W_t3),
        ("hecktor", lo_h, hi_h, q_hat_W_h),
    ]:
        w_log = (hi_c + q_hat) - (lo_c - q_hat)
        w_suv = widths_natural(lo_c, hi_c, q_hat)
        rows.append({
            "cohort": cohort_name,
            "mode": "wcp_image",
            "n_lesions": int(len(lo_c)),
            "q_hat": float(q_hat),
            "median_width_log": float(np.median(w_log)),
            "iqr25_width_log": float(np.percentile(w_log, 25)),
            "iqr75_width_log": float(np.percentile(w_log, 75)),
            "median_width_suv": float(np.median(w_suv)),
            "iqr25_width_suv": float(np.percentile(w_suv, 25)),
            "iqr75_width_suv": float(np.percentile(w_suv, 75)),
            "p90_width_suv": float(np.percentile(w_suv, 90)),
        })

    # Mode 4: per-cohort recal (using eval set only)
    for cohort_name, ev_df, lo_c, hi_c, q_hat, ev_idx_mask in [
        ("autopet_iii", ev_t3, lo_t3, hi_t3, q_hat_t3_local, ~ho_t3_idx),
        ("hecktor", ev_h, lo_h, hi_h, q_hat_h_local, ~ho_h_idx),
    ]:
        lo_e = lo_c[ev_idx_mask]
        hi_e = hi_c[ev_idx_mask]
        w_log = (hi_e + q_hat) - (lo_e - q_hat)
        w_suv = widths_natural(lo_e, hi_e, q_hat)
        rows.append({
            "cohort": cohort_name,
            "mode": "percohort_recal",
            "n_lesions": int(len(lo_e)),
            "q_hat": float(q_hat),
            "median_width_log": float(np.median(w_log)),
            "iqr25_width_log": float(np.percentile(w_log, 25)),
            "iqr75_width_log": float(np.percentile(w_log, 75)),
            "median_width_suv": float(np.median(w_suv)),
            "iqr25_width_suv": float(np.percentile(w_suv, 25)),
            "iqr75_width_suv": float(np.percentile(w_suv, 75)),
            "p90_width_suv": float(np.percentile(w_suv, 90)),
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print()
    print("Median (IQR) interval widths in natural-SUV units:")
    for _, r in df.iterrows():
        print(
            f"  {r['cohort']:<16} {r['mode']:<18} n={r['n_lesions']:>5}  q_hat={r['q_hat']:.4f}  "
            f"width_SUV={r['median_width_suv']:.2f} ({r['iqr25_width_suv']:.2f}-{r['iqr75_width_suv']:.2f})  "
            f"p90={r['p90_width_suv']:.2f}"
        )
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
