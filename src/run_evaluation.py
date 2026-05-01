"""End-to-end evaluation pipeline.

This is the script referenced in the code freeze protocol (§3.10).
It reads the frozen split manifest and calibration artifacts, then
produces all primary results tables and figures.

Designed to be frozen (committed + tagged + SHA-256 hashed) before
test-set evaluation. No modifications permitted after freeze without
a documented amendment.

Usage:
    python -m src.run_evaluation --config configs/default.yaml

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§3.10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.conformal.cqr import (
    train_quantile_regressors,
    calibrate_cqr,
    predict_intervals_array,
)
from src.conformal.mondrian import (
    build_mondrian_strata,
    merge_small_strata,
    calibrate_mondrian,
    predict_mondrian_intervals,
)
from src.conformal.coverage import (
    compute_coverage,
    compute_conditional_coverage,
    compute_coverage_disparity,
    coverage_summary_table,
)
from src.clinical.vision_overlay import (
    apply_vision_overlay,
    compute_vision_indeterminacy_rate,
)
from src.clinical.percist_overlay import (
    apply_percist_overlay,
    compute_percist_indeterminacy_rate,
)
from src.clinical.indeterminacy_table import (
    build_indeterminacy_table,
    format_indeterminacy_table,
)
from src.evaluation.bootstrap import (
    bootstrap_coverage,
    bootstrap_median_width,
    bootstrap_indeterminacy_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def verify_split_manifest(manifest_path: str, expected_hash: str | None) -> pd.DataFrame:
    """Load and verify split manifest integrity."""
    df = pd.read_parquet(manifest_path)
    manifest_json = (
        df[["patient_id", "split"]]
        .sort_values("patient_id")
        .reset_index(drop=True)
        .to_json(orient="records", indent=None)
    )
    actual_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()

    if expected_hash and actual_hash != expected_hash:
        log.error(
            f"Split manifest hash mismatch!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}\n"
            f"  This indicates the split manifest has been modified after freeze."
        )
        sys.exit(1)

    log.info(f"Split manifest verified: {actual_hash}")
    return df


def run_primary_analysis(
    lesion_df: pd.DataFrame,
    split_df: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> dict:
    """Run the primary CQR + Mondrian analysis pipeline.

    Parameters
    ----------
    lesion_df : pd.DataFrame
        Lesion feature table with all CQR predictor variables and
        outcome variables (suvmax, suvpeak, suvmean, tlg).
    split_df : pd.DataFrame
        Split manifest (patient_id, split).
    config : dict
        Configuration from YAML.
    output_dir : Path
        Directory for results output.

    Returns
    -------
    dict
        Results dictionary with all primary metrics.
    """
    results = {}

    # Merge splits into lesion table
    df = lesion_df.merge(split_df[["patient_id", "split"]], on="patient_id", how="left")

    # Define feature columns for CQR
    feature_cols = [
        col for col in [
            "volume_ml", "surface_area_cm2", "sphericity",
            "pet_pixel_spacing_mm", "pet_slice_thickness_mm", "pet_voxel_volume_mm3",
            "age",
        ]
        if col in df.columns
    ]

    # Categorical features (encoded as dummies or passed to LightGBM as category)
    cat_cols = [col for col in ["vendor", "tracer_category", "scanner_model"] if col in df.columns]

    # Target SUV metrics
    suv_targets = ["suvmax", "suvpeak", "suvmean"]
    alpha_levels = config.get("cqr", {}).get("alpha_levels", [0.05, 0.10, 0.20])

    for target in suv_targets:
        if target not in df.columns:
            log.warning(f"Target {target} not in lesion table, skipping")
            continue

        log.info(f"Running CQR for {target}")

        # Prepare feature matrix
        train_df = df[df["split"] == "train"].dropna(subset=[target])
        cal_df = df[df["split"] == "calibration"].dropna(subset=[target])
        test_df = df[df["split"] == "test"].dropna(subset=[target])

        if len(train_df) == 0 or len(cal_df) == 0:
            log.warning(f"Insufficient data for {target}: train={len(train_df)}, cal={len(cal_df)}")
            continue

        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df[target].values
        X_cal = cal_df[feature_cols].fillna(0).values
        y_cal = cal_df[target].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df[target].values

        for alpha in alpha_levels:
            nominal = 1.0 - alpha
            log.info(f"  alpha={alpha} (target {nominal*100:.0f}% coverage)")

            # Step 1: Train quantile regressors
            lgb_params = config.get("cqr", {}).get("lightgbm", {})
            lower_model, upper_model = train_quantile_regressors(
                X_train, y_train, alpha=alpha, **lgb_params
            )

            # Step 2: Predict on calibration set
            lo_cal = lower_model.predict(X_cal)
            hi_cal = upper_model.predict(X_cal)

            # Step 3: Marginal calibration
            marginal_cal = calibrate_cqr(y_cal, lo_cal, hi_cal, alpha)

            # Step 4: Predict on test set
            lo_test = lower_model.predict(X_test)
            hi_test = upper_model.predict(X_test)
            lb, ub, widths = predict_intervals_array(lo_test, hi_test, marginal_cal.q_hat)

            # Step 5: Marginal coverage
            marginal_cov = compute_coverage(y_test, lb, ub, nominal, label="marginal")
            log.info(f"    Marginal coverage: {marginal_cov.coverage:.3f} "
                     f"(nominal {nominal:.2f}, gap {marginal_cov.coverage_gap*100:+.1f}pp)")

            # Step 6: Mondrian calibration
            if "tracer_category" in cal_df.columns and "vendor" in cal_df.columns:
                strata_cal = build_mondrian_strata(cal_df)
                strata_cal, merge_map = merge_small_strata(strata_cal)
                strata_test = build_mondrian_strata(test_df)
                # Apply same merge map to test
                for old, new in merge_map.items():
                    strata_test = strata_test.replace(old, new)

                mondrian_cal = calibrate_mondrian(
                    y_cal, lo_cal, hi_cal, strata_cal.values, alpha
                )
                lb_m, ub_m, widths_m = predict_mondrian_intervals(
                    lo_test, hi_test, strata_test.values, mondrian_cal
                )

                conditional = compute_conditional_coverage(
                    y_test, lb_m, ub_m, strata_test.values, nominal
                )
                disparity = compute_coverage_disparity(conditional)
                log.info(f"    Mondrian disparity: {disparity*100:.1f}pp across {len(conditional)} strata")
            else:
                conditional = []
                lb_m, ub_m, widths_m = lb, ub, widths

            # Step 7: Bootstrap CIs
            patient_ids_test = test_df["patient_id"].values
            boot_cov = bootstrap_coverage(patient_ids_test, y_test, lb_m, ub_m, seed=42)
            boot_width = bootstrap_median_width(patient_ids_test, lb_m, ub_m, seed=42)

            # Step 8: Coverage summary table
            summary = coverage_summary_table(marginal_cov, conditional)

            # Store results
            key = f"{target}_alpha{alpha}"
            results[key] = {
                "target": target,
                "alpha": alpha,
                "nominal": nominal,
                "marginal_coverage": marginal_cov.coverage,
                "marginal_coverage_ci": (boot_cov.ci_lower, boot_cov.ci_upper),
                "median_width": marginal_cov.median_width,
                "median_width_ci": (boot_width.ci_lower, boot_width.ci_upper),
                "n_test": len(y_test),
                "n_strata": len(conditional),
                "disparity": disparity if conditional else None,
                "q_hat_marginal": marginal_cal.q_hat,
            }

            # Save per-alpha outputs
            summary.to_csv(output_dir / f"coverage_{key}.csv", index=False)

            # Save intervals for clinical overlay
            test_df = test_df.copy()
            test_df[f"{target}_ci_lower"] = lb_m
            test_df[f"{target}_ci_upper"] = ub_m

        # Save test set with all intervals
        test_df.to_parquet(output_dir / f"test_intervals_{target}.parquet", index=False)

    return results


def run_vision_overlay(
    test_df: pd.DataFrame,
    liver_suvmean: dict[str, float],
    alpha_levels: list[float],
    output_dir: Path,
) -> list[dict]:
    """Run VISION eligibility overlay at each CI level."""
    vision_results = []

    for alpha in alpha_levels:
        ci_col_lower = "suvmax_ci_lower"
        ci_col_upper = "suvmax_ci_upper"

        if ci_col_lower not in test_df.columns:
            log.warning(f"No suvmax CI columns found for VISION overlay")
            continue

        overlay = apply_vision_overlay(
            test_df, liver_suvmean,
            ci_lower_col=ci_col_lower,
            ci_upper_col=ci_col_upper,
        )
        stats = compute_vision_indeterminacy_rate(overlay)
        stats["ci_level"] = 1.0 - alpha
        vision_results.append(stats)

        log.info(f"  VISION at {(1-alpha)*100:.0f}%: "
                 f"indeterminacy={stats['indeterminacy_pct']:.1f}% "
                 f"({stats['n_indeterminate']}/{stats['n_total']})")

    return vision_results


def save_results_summary(results: dict, output_dir: Path) -> None:
    """Save the overall results summary as JSON."""
    # Convert numpy types for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = json.loads(json.dumps(results, default=convert))

    out_path = output_dir / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Results summary saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run conformal SUV evaluation pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--lesion-table", type=str, required=True,
                        help="Path to lesion feature table (parquet)")
    parser.add_argument("--split-manifest", type=str, required=True,
                        help="Path to split manifest (parquet)")
    parser.add_argument("--split-hash", type=str, default=None,
                        help="Expected SHA-256 of split manifest (for verification)")
    parser.add_argument("--liver-suvmean", type=str, default=None,
                        help="Path to JSON mapping study_uid -> liver SUVmean")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for tables and figures")
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Conformal SUV Theranostic — Evaluation Pipeline")
    log.info(f"Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN")
    log.info(f"Config: {args.config}")
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info("=" * 60)

    # Load and verify data
    split_df = verify_split_manifest(args.split_manifest, args.split_hash)
    lesion_df = pd.read_parquet(args.lesion_table)
    log.info(f"Lesion table: {len(lesion_df)} lesions from {lesion_df['patient_id'].nunique()} patients")

    # Run primary analysis
    results = run_primary_analysis(
        lesion_df, split_df, config, output_dir / "tables"
    )

    # Run VISION overlay if liver reference available
    if args.liver_suvmean:
        with open(args.liver_suvmean) as f:
            liver_suvmean = json.load(f)

        # Load test intervals
        test_path = output_dir / "tables" / "test_intervals_suvmax.parquet"
        if test_path.exists():
            test_df = pd.read_parquet(test_path)
            alpha_levels = config.get("cqr", {}).get("alpha_levels", [0.10])
            vision_results = run_vision_overlay(
                test_df, liver_suvmean, alpha_levels, output_dir / "tables"
            )

            # Build indeterminacy table
            if vision_results:
                indet_table = build_indeterminacy_table(vision_results)
                indet_table.to_csv(output_dir / "tables" / "indeterminacy_table.csv", index=False)
                log.info(f"\n{format_indeterminacy_table(indet_table)}")
                results["vision_indeterminacy"] = vision_results

    # Save summary
    save_results_summary(results, output_dir)

    log.info("=" * 60)
    log.info("Pipeline complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
