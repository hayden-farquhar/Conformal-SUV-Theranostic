"""CQR smoke test on AutoPET-I alone.

End-to-end exercise of the conformal pipeline (`src.conformal`) on the
AutoPET-I cohort BEFORE AutoPET-III data is available. This is NOT a
freeze run -- the pre-registered test split is not touched. Instead the
calibration split is internally divided into cal_a (60%) for conformal
calibration and cal_b (40%) as a pseudo-test set. Coverage on cal_b is
NOT a pre-registered result; this script's purpose is to catch
infrastructure bugs in the CQR pipeline early.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.2.3, §4.3)

Usage
-----
    python scripts/cqr_smoke_test_autopet_i.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conformal.cqr import (
    train_quantile_regressors,
    calibrate_cqr,
    predict_intervals_array,
)
from src.conformal.coverage import compute_coverage
from src.conformal.mondrian import assign_volume_quartiles


# Paths
LESIONS_PATH = PROJECT_ROOT / "data/interim/lesion_tables/autopet_i_lesions.parquet"
SPLITS_PATH = PROJECT_ROOT / "data/processed/autopet_i_splits.parquet"

# Smoke-test config (NOT pre-reg-frozen; for infrastructure validation only)
ALPHA = 0.10  # 90% nominal coverage
RANDOM_SEED = 42


def main():
    print("=" * 70)
    print("CQR smoke test on AutoPET-I (NOT a freeze run)")
    print("=" * 70)

    # --- Load data ---
    lesions = pd.read_parquet(LESIONS_PATH)
    splits = pd.read_parquet(SPLITS_PATH)
    print(f"\nLesions: {len(lesions)} rows, {lesions['case_id'].nunique()} patients")
    print(f"Splits:  {len(splits)} patient assignments")
    print(f"Split distribution: {splits['split'].value_counts().to_dict()}")

    # Filter §3.9 exclusions
    if "excluded" in lesions.columns:
        n_excluded = int(lesions["excluded"].sum())
        lesions = lesions[~lesions["excluded"].astype(bool)].copy()
        print(f"§3.9 exclusions removed: {n_excluded} -> {len(lesions)} retained lesions")

    # Merge split assignment (splits uses patient_id; lesions uses case_id -- same value)
    splits_renamed = splits[["patient_id", "split"]].rename(columns={"patient_id": "case_id"})
    lesions = lesions.merge(splits_renamed, on="case_id", how="left")
    assert lesions["split"].notna().all(), "every lesion must have a split assignment"

    # Subset
    train = lesions[lesions["split"] == "train"].reset_index(drop=True)
    cal = lesions[lesions["split"] == "calibration"].reset_index(drop=True)
    print(f"\nTrain: {len(train)} lesions ({train['case_id'].nunique()} patients)")
    print(f"Cal:   {len(cal)} lesions ({cal['case_id'].nunique()} patients)")

    # --- Internal cal_a / cal_b split (NOT touching pre-reg test) ---
    rng = np.random.RandomState(RANDOM_SEED)
    cal_patients = cal["case_id"].unique()
    rng.shuffle(cal_patients)
    n_cal_a = int(len(cal_patients) * 0.6)
    cal_a_patients = set(cal_patients[:n_cal_a])
    cal_a = cal[cal["case_id"].isin(cal_a_patients)].reset_index(drop=True)
    cal_b = cal[~cal["case_id"].isin(cal_a_patients)].reset_index(drop=True)
    print(f"\nInternal split (smoke-test only -- pre-reg test split is NOT touched):")
    print(f"  cal_a (calibration): {len(cal_a)} lesions ({cal_a['case_id'].nunique()} pat)")
    print(f"  cal_b (pseudo-test): {len(cal_b)} lesions ({cal_b['case_id'].nunique()} pat)")

    # --- Features ---
    # Pre-reg §4.2 lists volume, softmax_mean, softmax_entropy, tracer, vendor,
    # uptake_time, reconstruction_algorithm. AutoPET-I (FDAT NIfTI) has no DICOM
    # metadata so uptake_time / reconstruction_algorithm are missing; tracer and
    # vendor are constant (FDG / Siemens) and non-informative; softmax features
    # require nnU-Net inference and are not available for AutoPET-I (uses FDAT
    # ground-truth SEG). Feature set collapses to volume + centroid for AutoPET-I
    # alone -- this is appropriate for the smoke test, not for the freeze.
    feature_cols = ["volume_ml", "centroid_0", "centroid_1", "centroid_2"]
    target_col_raw = "suvmax"

    # Pre-reg specifies log-transform of SUV target (§5.2.3)
    def transform_target(y_raw):
        return np.log1p(y_raw.astype(np.float64))

    def inverse_target(y_log):
        return np.expm1(y_log)

    X_train = train[feature_cols].values
    y_train = transform_target(train[target_col_raw].values)
    X_cal_a = cal_a[feature_cols].values
    y_cal_a = transform_target(cal_a[target_col_raw].values)
    X_cal_b = cal_b[feature_cols].values
    y_cal_b = transform_target(cal_b[target_col_raw].values)

    # --- Train quantile regressors ---
    print("\nTraining LightGBM quantile regressors (lower + upper)...")
    lower_model, upper_model = train_quantile_regressors(
        X_train, y_train, alpha=ALPHA
    )

    # --- Calibrate on cal_a, predict on cal_b ---
    lo_a = lower_model.predict(X_cal_a)
    hi_a = upper_model.predict(X_cal_a)
    lo_b = lower_model.predict(X_cal_b)
    hi_b = upper_model.predict(X_cal_b)

    # Marginal calibration
    cqr_marginal = calibrate_cqr(
        y_cal_a, lo_a, hi_a, alpha=ALPHA, stratum_name="all"
    )
    print(f"\nMarginal calibration: q̂ = {cqr_marginal.q_hat:.4f}, "
          f"n_cal = {cqr_marginal.n_calibration}")

    # Marginal intervals on cal_b
    lower_b, upper_b, widths_b = predict_intervals_array(lo_b, hi_b, cqr_marginal.q_hat)
    cov_marginal = compute_coverage(y_cal_b, lower_b, upper_b, nominal=1 - ALPHA, label="marginal")
    print(f"\n--- MARGINAL COVERAGE on cal_b (target {(1-ALPHA)*100:.0f}%) ---")
    print(f"  empirical coverage: {cov_marginal.coverage:.4f} "
          f"(target {1 - ALPHA:.4f}, miss = {(cov_marginal.coverage - (1 - ALPHA)):+.4f})")
    print(f"  median width:       {np.median(widths_b):.4f} (log-SUV units)")
    print(f"  95% CI on coverage: [{cov_marginal.ci_lower:.4f}, {cov_marginal.ci_upper:.4f}]")

    nominal_in_ci = cov_marginal.ci_lower <= (1 - ALPHA) <= cov_marginal.ci_upper
    print(f"  nominal in 95% CI:  {'YES' if nominal_in_ci else 'NO (investigate)'}")

    # --- Mondrian by volume quartile (pre-reg §4.3) ---
    # AutoPET-I tracer and vendor are constant -> stratification collapses to
    # volume_quartile only.
    print(f"\n--- MONDRIAN COVERAGE by volume quartile ---")
    cal_a_q = assign_volume_quartiles(cal_a["volume_ml"].values)
    cal_b_q = assign_volume_quartiles(cal_b["volume_ml"].values)

    rows = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        mask_a = cal_a_q == q
        mask_b = cal_b_q == q
        if mask_a.sum() < 30 or mask_b.sum() < 10:
            print(f"  {q}: skipped (n_cal_a={mask_a.sum()}, n_cal_b={mask_b.sum()})")
            continue
        cqr_q = calibrate_cqr(
            y_cal_a[mask_a], lo_a[mask_a], hi_a[mask_a],
            alpha=ALPHA, stratum_name=q,
        )
        lo_q, hi_q, w_q = predict_intervals_array(
            lo_b[mask_b], hi_b[mask_b], cqr_q.q_hat
        )
        cov_q = compute_coverage(y_cal_b[mask_b], lo_q, hi_q, nominal=1 - ALPHA, label=q)
        rows.append({
            "stratum": q,
            "n_cal": mask_a.sum(),
            "n_test": mask_b.sum(),
            "q_hat": cqr_q.q_hat,
            "coverage": cov_q.coverage,
            "miss": cov_q.coverage - (1 - ALPHA),
            "median_width": float(np.median(w_q)),
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # --- Sanity gates ---
    print(f"\n--- SMOKE TEST SANITY GATES ---")
    gates = []

    g1 = abs(cov_marginal.coverage - (1 - ALPHA)) < 0.05
    gates.append(("Marginal coverage within 5pp of nominal", g1))

    if len(summary) > 0:
        max_miss = summary["miss"].abs().max()
        g2 = max_miss < 0.10
        gates.append(("All Mondrian strata within 10pp of nominal", g2))
        # Note: negative q̂ in a stratum is mathematically valid and means the
        # base regressor over-covered there; CQR tightens rather than widens.
        # The meaningful gate is that intervals are well-formed (lower < upper).
        g3 = bool((lower_b < upper_b).all())
        gates.append(("All marginal intervals well-formed (lower < upper)", g3))

    g4 = float(np.median(widths_b)) > 0.0 and float(np.median(widths_b)) < 5.0
    gates.append(("Median interval width plausible (0 < w < 5 in log-SUV)", g4))

    for name, passed in gates:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name}")

    if all(g[1] for g in gates):
        print(f"\nSMOKE TEST PASSED. Conformal infrastructure is operational on AutoPET-I.")
        return 0
    else:
        print(f"\nSMOKE TEST FAILED. Investigate before AutoPET-III lands.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
