"""Phase 3 diagnostic analysis.

Read-only diagnostic script that re-uses the Phase 3 driver's load + train +
calibrate logic to produce diagnostic artefacts WITHOUT writing to the locked
Phase 3 outputs. Goal: localise the under-coverage failure pattern surfaced
in the four-mode comparison (H1/H11/H12 FAIL with 8-9pp marginal miss on
external cohorts).

Diagnostics produced:

  1. Per-cohort residuals: predicted SUVmax point estimate ((lower+upper)/2)
     vs observed; Spearman rho per cohort (interpretation: does the model at
     least RANK lesions correctly even when its absolute calibration fails?).

  2. Coverage-by-bin curves: fine-grained coverage by volume_ml decile,
     surface_area_cm2 decile, sphericity bin, suvmax_observed quartile,
     softmax_mean (for AutoPET-III only). Localises where coverage breaks.

  3. q_hat sensitivity: what marginal coverage on each test cohort would have
     been if q_hat were calibrated on AutoPET-III cal split or HECKTOR cal
     split (counterfactual; AutoPET-III + HECKTOR are pure test cohorts so we
     can't actually do this for the freeze, but the counterfactual is
     informative about whether AutoPET-I-as-calibration is the bottleneck).

  4. Centre × vendor coverage heatmap (HECKTOR only): is the centre-level
     miss vendor-correlated, sample-size-correlated, or independent?

  5. GTVp/GTVn coverage ratio for HECKTOR (per-class anatomical asymmetry).

Outputs:
    results/phase3/diagnostics/
        residuals.parquet                  -- per-lesion predicted/observed/in-interval
        coverage_by_bin.parquet            -- per-bin coverage rates
        qhat_sensitivity.csv               -- counterfactual marginal coverage
        centre_vendor_coverage.csv         -- HECKTOR centre x vendor breakdown
        diagnostic_report.md               -- narrative summary
        residuals_per_cohort.png           -- 3-panel scatter plot
        coverage_by_bin.png                -- coverage drop pattern

This script does NOT write any locked Phase 3 outputs. Pre-reg-compliant
diagnostic only.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN
Amendments: OSF j5ry4 amendment_log.md v10
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

# Reuse Phase 3 driver constants directly so calibration logic stays bit-identical
import scripts.phase3_evaluate as p3
from scripts.phase3_evaluate import (
    ALPHA, NOMINAL, FEATURE_COLS, PATHS,
    load_autopet_i, load_autopet_iii, load_hecktor,
    transform_target,
    _compute_quartile_boundaries,
    _assign_volume_quartile_with_boundaries,
    build_mondrian_calibration,
)


OUT_DIR = PROJECT_ROOT / "results/phase3/diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "suvmax"

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def _per_lesion_records(
    cohort: str,
    df: pd.DataFrame,
    lo: np.ndarray,
    hi: np.ndarray,
    y_log: np.ndarray,
    q_hat_marginal: float,
) -> pd.DataFrame:
    """Compute per-lesion predicted/observed/in_interval table."""
    lower, upper, widths = predict_intervals_array(lo, hi, q_hat_marginal)
    point_est = (lower + upper) / 2.0
    in_interval = (y_log >= lower) & (y_log <= upper)
    rec = pd.DataFrame({
        "cohort": cohort,
        "case_id": df["case_id"].values,
        "lesion_id": df["lesion_id"].values,
        "y_log_observed": y_log,
        "y_log_lower": lower,
        "y_log_upper": upper,
        "y_log_point": point_est,
        "interval_width_log": widths,
        "in_interval": in_interval,
        "volume_ml": df["volume_ml"].values,
        "surface_area_cm2": df["surface_area_cm2"].values,
        "sphericity": df["sphericity"].values,
        "softmax_mean": df["softmax_mean"].values,
    })
    return rec


def _coverage_by_bin(rec: pd.DataFrame, col: str, n_bins: int = 10) -> pd.DataFrame:
    """Coverage rate by quantile bins of `col`."""
    if rec[col].nunique() < n_bins:
        return pd.DataFrame()
    bins = pd.qcut(rec[col], q=n_bins, duplicates="drop")
    out = (
        rec.groupby([rec["cohort"], bins], observed=True)
        .agg(
            n=("in_interval", "size"),
            coverage=("in_interval", "mean"),
            bin_low=(col, "min"),
            bin_high=(col, "max"),
        )
        .reset_index()
        .rename(columns={col: "bin"})
    )
    out["bin_label"] = out["bin"].astype(str)
    out["binned_by"] = col
    out = out.drop(columns=["bin"])
    return out


def main() -> None:
    print("=" * 78)
    print(f"Phase 3 diagnostics (Path C, target={TARGET}, alpha={ALPHA})")
    print("=" * 78)

    # Load all three cohorts (re-uses driver logic)
    train, cal, test = load_autopet_i()
    autopet_iii = load_autopet_iii()
    hecktor = load_hecktor()

    def X(df): return df[FEATURE_COLS].to_numpy()
    def y(df): return transform_target(df[TARGET].to_numpy())

    # Train QR + calibrate -- bit-identical to Phase 3 driver
    print("\nTraining LightGBM quantile regressors...")
    lower_model, upper_model = train_quantile_regressors(X(train), y(train), alpha=ALPHA)
    lo_cal,  hi_cal  = lower_model.predict(X(cal)),         upper_model.predict(X(cal))
    lo_t1,   hi_t1   = lower_model.predict(X(test)),        upper_model.predict(X(test))
    lo_t3,   hi_t3   = lower_model.predict(X(autopet_iii)), upper_model.predict(X(autopet_iii))
    lo_h,    hi_h    = lower_model.predict(X(hecktor)),     upper_model.predict(X(hecktor))

    cal_boundaries = _compute_quartile_boundaries(cal["volume_ml"].to_numpy())
    mondrian = build_mondrian_calibration(
        cal_df=cal, lo_cal=lo_cal, hi_cal=hi_cal, y_cal=y(cal),
        quartile_boundaries=cal_boundaries,
    )
    q_hat = mondrian.q_hat_marginal
    print(f"q_hat (AutoPET-I cal marginal) = {q_hat:.4f}")

    # ===== Diagnostic 1: per-lesion residuals =====
    print("\n--- Diagnostic 1: per-lesion residuals ---")
    records = pd.concat([
        _per_lesion_records("autopet_i",  test,         lo_t1, hi_t1, y(test),         q_hat),
        _per_lesion_records("autopet_iii", autopet_iii, lo_t3, hi_t3, y(autopet_iii),  q_hat),
        _per_lesion_records("hecktor",    hecktor,      lo_h,  hi_h,  y(hecktor),      q_hat),
    ], ignore_index=True)
    records["lesion_class"] = pd.NA
    if "lesion_class" in hecktor.columns:
        h_class = hecktor[["case_id", "lesion_id", "lesion_class"]]
        records = records.merge(h_class, on=["case_id", "lesion_id"], how="left", suffixes=("", "_h"))
        records["lesion_class"] = records["lesion_class_h"].combine_first(records["lesion_class"])
        records = records.drop(columns=["lesion_class_h"])
    if "centre_id" in hecktor.columns:
        h_ctr = hecktor[["case_id", "lesion_id", "centre_id", "centre_name", "vendor"]]
        records = records.merge(h_ctr, on=["case_id", "lesion_id"], how="left")
    records.to_parquet(OUT_DIR / "residuals.parquet", index=False)
    print(f"  Wrote {OUT_DIR / 'residuals.parquet'} ({len(records)} rows)")

    # Spearman rho per cohort: does the point estimate at least correlate with the truth?
    from scipy.stats import spearmanr
    print("\n  Spearman rho (predicted point estimate vs observed, log SUV):")
    for cohort, group in records.groupby("cohort", observed=True):
        rho, p = spearmanr(group["y_log_point"], group["y_log_observed"])
        print(f"    {cohort:<14} rho = {rho:+.3f}  p = {p:.2e}  n = {len(group)}")

    # ===== Diagnostic 2: coverage-by-bin =====
    print("\n--- Diagnostic 2: coverage-by-bin ---")
    bin_rows = []
    for col in ["volume_ml", "surface_area_cm2", "sphericity", "y_log_observed"]:
        for cohort, group in records.groupby("cohort", observed=True):
            sub = _coverage_by_bin(group.assign(cohort=cohort), col, n_bins=10)
            bin_rows.append(sub)
    cov_bins = pd.concat([r for r in bin_rows if not r.empty], ignore_index=True)
    cov_bins.to_parquet(OUT_DIR / "coverage_by_bin.parquet", index=False)
    print(f"  Wrote {OUT_DIR / 'coverage_by_bin.parquet'} ({len(cov_bins)} bins)")

    # Print the worst-coverage bins for quick interpretation
    print("\n  Worst-coverage bins (lowest empirical coverage; n>=20):")
    worst = cov_bins[cov_bins["n"] >= 20].nsmallest(10, "coverage")
    print(worst[["cohort", "binned_by", "bin_label", "n", "coverage"]].to_string(index=False))

    # ===== Diagnostic 3: q_hat sensitivity =====
    print("\n--- Diagnostic 3: q_hat sensitivity (counterfactual) ---")
    # What if we had calibrated q_hat on each cohort's own residuals?
    sens_rows = []
    for cohort, lo, hi, y_log, df in [
        ("autopet_i_test",  lo_t1,  hi_t1,  y(test),         test),
        ("autopet_iii",     lo_t3,  hi_t3,  y(autopet_iii),  autopet_iii),
        ("hecktor",         lo_h,   hi_h,   y(hecktor),      hecktor),
    ]:
        cqr = calibrate_cqr(y_log, lo, hi, alpha=ALPHA, stratum_name=cohort)
        sens_rows.append({
            "cohort": cohort,
            "q_hat_actual_from_autopet_i_cal": q_hat,
            "q_hat_counterfactual_from_self": cqr.q_hat,
            "q_hat_delta": cqr.q_hat - q_hat,
            "n": int(len(y_log)),
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT_DIR / "qhat_sensitivity.csv", index=False)
    print(sens_df.to_string(index=False))
    print()
    print("  Interpretation: a large 'q_hat_delta' means each cohort's residuals are")
    print("  systematically larger or smaller than AutoPET-I cal residuals. If")
    print("  external cohort q_hat_counterfactual >> q_hat_actual, the AutoPET-I")
    print("  calibration is too tight for that cohort -> coverage suffers.")

    # ===== Diagnostic 4: HECKTOR centre × vendor coverage =====
    print("\n--- Diagnostic 4: HECKTOR centre × vendor coverage ---")
    hek_rec = records[records["cohort"] == "hecktor"]
    cv_cov = (
        hek_rec.groupby(["centre_id", "centre_name", "vendor"])
        .agg(n=("in_interval", "size"), coverage=("in_interval", "mean"))
        .reset_index()
        .sort_values("coverage")
    )
    cv_cov["miss_pp"] = (cv_cov["coverage"] - NOMINAL) * 100
    cv_cov.to_csv(OUT_DIR / "centre_vendor_coverage.csv", index=False)
    print(cv_cov.to_string(index=False))

    # ===== Diagnostic 5: GTVp/GTVn per-class coverage =====
    print("\n--- Diagnostic 5: HECKTOR GTVp/GTVn per-class coverage ---")
    cls_cov = (
        hek_rec.groupby("lesion_class", observed=True)
        .agg(
            n=("in_interval", "size"),
            coverage=("in_interval", "mean"),
            volume_ml_median=("volume_ml", "median"),
            sphericity_median=("sphericity", "median"),
        )
        .reset_index()
    )
    cls_cov["lesion_class_label"] = cls_cov["lesion_class"].map(
        {1.0: "GTVp", 2.0: "GTVn", 3.0: "mixed"}
    )
    cls_cov["miss_pp"] = (cls_cov["coverage"] - NOMINAL) * 100
    print(cls_cov.to_string(index=False))

    # ===== Plots =====
    if HAVE_MPL:
        print("\n--- Generating plots ---")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for ax, cohort in zip(axes, ["autopet_i", "autopet_iii", "hecktor"]):
            sub = records[records["cohort"] == cohort]
            inside = sub[sub["in_interval"]]
            outside = sub[~sub["in_interval"]]
            ax.scatter(inside["y_log_observed"], inside["y_log_point"],
                       s=4, alpha=0.3, c="tab:blue", label=f"in interval (n={len(inside)})")
            ax.scatter(outside["y_log_observed"], outside["y_log_point"],
                       s=4, alpha=0.5, c="tab:red", label=f"missed (n={len(outside)})")
            lo_y = float(min(sub["y_log_observed"].min(), sub["y_log_point"].min()))
            hi_y = float(max(sub["y_log_observed"].max(), sub["y_log_point"].max()))
            ax.plot([lo_y, hi_y], [lo_y, hi_y], "k--", lw=1, alpha=0.5)
            ax.set_title(f"{cohort}  (cov = {sub['in_interval'].mean():.3f})")
            ax.set_xlabel("observed log(SUVmax+1)")
            ax.set_ylabel("predicted point estimate (log scale)")
            ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "residuals_per_cohort.png", dpi=120)
        plt.close(fig)
        print(f"  Wrote {OUT_DIR / 'residuals_per_cohort.png'}")

        # Coverage-by-volume curves per cohort
        fig, ax = plt.subplots(figsize=(10, 5))
        for cohort, group in cov_bins[cov_bins["binned_by"] == "volume_ml"].groupby("cohort", observed=True):
            ax.plot(range(len(group)), group["coverage"].values, marker="o",
                    label=f"{cohort} (n={int(group['n'].sum())})")
        ax.axhline(NOMINAL, color="k", ls="--", lw=1, label=f"nominal {NOMINAL:.2f}")
        ax.set_xlabel("volume_ml decile (low -> high)")
        ax.set_ylabel("empirical coverage")
        ax.set_title("Coverage by volume_ml decile")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "coverage_by_volume.png", dpi=120)
        plt.close(fig)
        print(f"  Wrote {OUT_DIR / 'coverage_by_volume.png'}")
    else:
        print("\n[matplotlib not available -- skipping plots]")

    # ===== Diagnostic report (markdown) =====
    print("\n--- Writing diagnostic report ---")
    lines = [
        "# Phase 3 Diagnostic Report",
        "",
        f"Generated by `scripts/phase3_diagnose.py` on {pd.Timestamp.now().date()}.",
        "Read-only diagnostic; no Phase 3 freeze artefacts touched.",
        "",
        "## 1. Spearman rho (predicted vs observed)",
        "",
        "Does the point estimate at least RANK lesions correctly even when its absolute calibration fails?",
        "",
        "| Cohort | rho | p | n |",
        "|---|---|---|---|",
    ]
    for cohort, group in records.groupby("cohort", observed=True):
        rho, p = spearmanr(group["y_log_point"], group["y_log_observed"])
        lines.append(f"| {cohort} | {rho:+.3f} | {p:.2e} | {len(group)} |")

    lines.extend([
        "",
        "Interpretation: high rho with poor coverage -> model ranks correctly but absolute scale is off (calibration issue, possibly fixable).",
        "Low rho -> model can't even rank correctly (deeper issue).",
        "",
        "## 2. q_hat sensitivity",
        "",
        "What would marginal coverage have looked like if q_hat were calibrated on each cohort's own residuals?",
        "",
        sens_df.to_markdown(index=False),
        "",
        "Interpretation: q_hat_delta > 0 means external residuals are larger than AutoPET-I cal residuals -> AutoPET-I-derived q_hat is too tight for that cohort -> coverage drops.",
        "",
        "## 3. HECKTOR centre × vendor coverage",
        "",
        cv_cov.to_markdown(index=False),
        "",
        "## 4. HECKTOR GTVp/GTVn per-class coverage",
        "",
        cls_cov.to_markdown(index=False),
        "",
        "## 5. Worst-coverage bins (n>=20)",
        "",
        worst[["cohort", "binned_by", "bin_label", "n", "coverage"]].to_markdown(index=False),
        "",
        "## Decision support",
        "",
        "Three paths from here:",
        "- **(A) Commit Phase 3 with honest negative finding**: marginal coverage is calibrated correctly under exchangeability but exchangeability fails under cohort/domain shift. Manuscript framing: 'consistent with [Tibshirani 2019; Barber 2023], we provide concrete external-validation quantification across three cohorts.'",
        "- **(B) Amend the protocol** to permit per-cohort recalibration / weighted conformal / jackknife+. Filed as a NEW Amendment; results clearly labelled exploratory.",
        "- **(C) Continue diagnosing**: if `q_hat_counterfactual_from_self` differs substantially from the AutoPET-I value, that's strong evidence the residual scale is cohort-specific -> per-cohort recalibration would close most of the gap. That motivates Amendment 11 (path B) more concretely.",
    ])
    (OUT_DIR / "diagnostic_report.md").write_text("\n".join(lines))
    print(f"  Wrote {OUT_DIR / 'diagnostic_report.md'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
