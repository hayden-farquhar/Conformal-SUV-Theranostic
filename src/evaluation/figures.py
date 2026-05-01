"""Pre-specified figure generation.

Creates the 13 figures specified in the pre-registration (§6.3).
Each function takes processed results DataFrames and produces a
publication-ready figure.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§6.3)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Consistent style
STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}
plt.rcParams.update(STYLE)
sns.set_palette("colorblind")


def fig2_coverage_calibration(
    coverage_results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Figure 2: Coverage calibration plot.

    Nominal coverage (x) vs empirical coverage (y) across alpha grid,
    with identity line and ±2pp tolerance band. Separate panels for
    marginal vs each Mondrian stratum.
    """
    strata = coverage_results["stratum"].unique()
    n_panels = len(strata)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)

    for i, stratum in enumerate(sorted(strata)):
        ax = axes[0, i]
        sub = coverage_results[coverage_results["stratum"] == stratum]

        nominals = sub["nominal"].values
        coverages = sub["coverage"].values

        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, label="Ideal")
        ax.fill_between([0.5, 1.0], [0.48, 0.98], [0.52, 1.02],
                        alpha=0.15, color="grey", label="±2pp")
        ax.scatter(nominals, coverages, s=40, zorder=3)
        ax.plot(nominals, coverages, alpha=0.5)

        if "ci_lower" in sub.columns and "ci_upper" in sub.columns:
            ax.vlines(nominals, sub["ci_lower"].values, sub["ci_upper"].values,
                      alpha=0.3, colors="C0")

        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(stratum)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.5, 1.05)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Coverage Calibration", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig3_interval_width_vs_volume(
    lesion_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Figure 3: Interval width vs lesion volume, coloured by tracer."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for tracer, color in [("FDG", "C0"), ("PSMA", "C1")]:
        sub = lesion_df[lesion_df["tracer_category"] == tracer]
        if len(sub) == 0:
            continue
        ax.scatter(sub["volume_ml"], sub["interval_width"],
                   s=8, alpha=0.3, color=color, label=tracer)

        # LOWESS smoother
        from statsmodels.nonparametric.smoothers_lowess import lowess
        if len(sub) > 20:
            smooth = lowess(sub["interval_width"].values, sub["volume_ml"].values,
                            frac=0.3, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], color=color, linewidth=2)

    ax.set_xlabel("Lesion volume (mL)")
    ax.set_ylabel("Conformal interval width (SUV)")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title("Interval Width vs Lesion Volume")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig4_vision_indeterminacy_zone(
    vision_df: pd.DataFrame,
    liver_suvmean: float,
    output_path: str | Path,
) -> None:
    """Figure 4: VISION indeterminacy zone plot.

    For each PSMA lesion: point estimate ± conformal interval as vertical
    bars, horizontal line at liver SUVmean, coloured by eligibility.
    """
    df = vision_df.sort_values("suvmax").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "eligible": "C2",
        "ineligible": "C3",
        "indeterminate": "C1",
    }

    for i, (_, row) in enumerate(df.iterrows()):
        decision = row.get("vision_decision", "indeterminate")
        color = colors.get(decision, "grey")
        ax.vlines(i, row["suvmax_ci_lower"], row["suvmax_ci_upper"],
                  colors=color, alpha=0.5, linewidth=1)
        ax.plot(i, row["suvmax"], ".", color=color, markersize=2)

    ax.axhline(liver_suvmean, color="red", linestyle="--", linewidth=1.5,
               label=f"Liver SUVmean = {liver_suvmean:.1f}")

    patches = [mpatches.Patch(color=c, label=l.capitalize())
               for l, c in colors.items()]
    patches.append(plt.Line2D([0], [0], color="red", linestyle="--", label="Threshold"))
    ax.legend(handles=patches, loc="upper left")

    ax.set_xlabel("Lesion index (sorted by SUVmax)")
    ax.set_ylabel("SUVmax")
    ax.set_title("VISION Eligibility with Conformal Intervals")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig5_percist_waterfall(
    percist_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Figure 5: PERCIST response classification waterfall plot.

    Waterfall of Δ% with conformal intervals, coloured by decision.
    """
    df = percist_df.sort_values("delta_pct_point").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "response": "C2",
        "stable": "C0",
        "progression": "C3",
        "indeterminate": "C1",
    }

    for i, (_, row) in enumerate(df.iterrows()):
        decision = row.get("percist_decision", "indeterminate")
        color = colors.get(decision, "grey")
        ax.bar(i, row["delta_pct_point"], color=color, alpha=0.6, width=0.8)
        ax.vlines(i, row["delta_pct_lower"], row["delta_pct_upper"],
                  colors="black", alpha=0.3, linewidth=0.5)

    ax.axhline(-30, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Response threshold (−30%)")
    ax.axhline(30, color="grey", linestyle=":", linewidth=1, alpha=0.7, label="Progression threshold (+30%)")
    ax.axhline(0, color="black", linewidth=0.5)

    patches = [mpatches.Patch(color=c, label=l.capitalize()) for l, c in colors.items()]
    ax.legend(handles=patches, loc="lower left", fontsize=8)

    ax.set_xlabel("Serial pair index (sorted by Δ%)")
    ax.set_ylabel("SULpeak change (%)")
    ax.set_title("PERCIST Response with Conformal Intervals")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig6_mondrian_heatmap(
    conditional_coverage: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Figure 6: Per-stratum coverage as a heatmap.

    Rows: volume quartile. Columns: tracer × vendor.
    """
    # Parse stratum labels to extract components
    df = conditional_coverage.copy()
    parts = df["stratum"].str.split("_", expand=True)
    if parts.shape[1] >= 3:
        df["tracer_vendor"] = parts[0] + "_" + parts[1]
        df["volume_q"] = parts[2]
    else:
        df["tracer_vendor"] = df["stratum"]
        df["volume_q"] = "all"

    pivot = df.pivot_table(values="coverage", index="volume_q",
                           columns="tracer_vendor", aggfunc="first")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0.90, vmin=0.70, vmax=1.0, ax=ax,
                linewidths=0.5, cbar_kws={"label": "Coverage"})
    ax.set_xlabel("Tracer × Vendor")
    ax.set_ylabel("Volume Quartile")
    ax.set_title("Mondrian Conditional Coverage")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig7_method_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Figure 7: Interval width vs coverage for all conformal methods."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in comparison_df["method"].unique():
        sub = comparison_df[comparison_df["method"] == method]
        ax.scatter(sub["coverage"], sub["median_width"], s=60, label=method, zorder=3)

    ax.axvline(0.90, color="grey", linestyle="--", alpha=0.5, label="90% target")
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("Median interval width (SUV)")
    ax.set_title("Conformal Method Comparison")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig8_qiba_benchmark(
    lesion_df: pd.DataFrame,
    earl_bounds: dict[str, float] | None = None,
    output_path: str | Path = "",
) -> None:
    """Figure 8: Conformal interval widths vs EARL/QIBA published bounds."""
    fig, ax = plt.subplots(figsize=(7, 5))

    volume_bins = [0, 2, 5, 10, 20, 50, 200]
    labels = ["<2", "2-5", "5-10", "10-20", "20-50", "50+"]

    if "interval_width" in lesion_df.columns and "volume_ml" in lesion_df.columns:
        lesion_df = lesion_df.copy()
        lesion_df["vol_bin"] = pd.cut(lesion_df["volume_ml"], bins=volume_bins, labels=labels)

        # Relative width as percentage of SUVmax
        if "suvmax" in lesion_df.columns:
            lesion_df["rel_width_pct"] = lesion_df["interval_width"] / lesion_df["suvmax"] * 100
            sns.boxplot(data=lesion_df, x="vol_bin", y="rel_width_pct", ax=ax,
                        color="C0", fliersize=2)

        # Overlay EARL bounds
        if earl_bounds:
            ax.axhline(earl_bounds.get("intra_scanner_cv", 10) * 2,
                       color="C2", linestyle="--", label="EARL intra-scanner (2×CV)")
            ax.axhline(earl_bounds.get("inter_scanner_cv", 22) * 2,
                       color="C3", linestyle="--", label="EARL inter-scanner (2×CV)")
            ax.legend()

    ax.set_xlabel("Lesion volume (mL)")
    ax.set_ylabel("Relative interval width (%)")
    ax.set_title("Conformal Intervals vs QIBA/EARL Benchmarks")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig9_meta_coverage_histogram(
    achieved_coverages: np.ndarray,
    target: float,
    output_path: str | Path,
) -> None:
    """Figure 9: Distribution of achieved coverage across resplits."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(achieved_coverages, bins=30, edgecolor="black", alpha=0.7, color="C0")
    ax.axvline(target, color="red", linestyle="--", linewidth=2,
               label=f"Target ({target*100:.0f}%)")
    ax.axvline(np.mean(achieved_coverages), color="C1", linestyle="-",
               linewidth=1.5, label=f"Mean ({np.mean(achieved_coverages)*100:.1f}%)")

    meta_cov = (achieved_coverages >= target).mean()
    ax.set_xlabel("Achieved coverage")
    ax.set_ylabel("Count (resplits)")
    ax.set_title(f"Meta-Coverage: {meta_cov*100:.0f}% of resplits achieve ≥{target*100:.0f}%")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig10_suvmax_vs_suvpeak_ratio(
    lesion_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Figure 10: SUVmax vs SUVpeak interval width ratio by volume quartile."""
    if "suvmax_width" not in lesion_df.columns or "suvpeak_width" not in lesion_df.columns:
        return

    df = lesion_df.copy()
    df["width_ratio"] = df["suvmax_width"] / df["suvpeak_width"].clip(lower=0.01)
    df["volume_quartile"] = pd.qcut(df["volume_ml"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="volume_quartile", y="width_ratio", ax=ax, color="C0")
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Volume quartile")
    ax.set_ylabel("Width ratio (SUVmax / SUVpeak)")
    ax.set_title("Interval Width: SUVmax vs SUVpeak by Lesion Size")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig11_patient_vs_lesion_indeterminacy(
    lesion_rate: float,
    patient_rate: float,
    lesion_ci: tuple[float, float],
    patient_ci: tuple[float, float],
    output_path: str | Path,
) -> None:
    """Figure 11: Paired bar chart of lesion-level vs patient-level indeterminacy."""
    fig, ax = plt.subplots(figsize=(5, 4))

    x = [0, 1]
    heights = [lesion_rate * 100, patient_rate * 100]
    errors = [
        [lesion_rate * 100 - lesion_ci[0] * 100, patient_rate * 100 - patient_ci[0] * 100],
        [lesion_ci[1] * 100 - lesion_rate * 100, patient_ci[1] * 100 - patient_rate * 100],
    ]

    bars = ax.bar(x, heights, yerr=errors, capsize=5, color=["C0", "C1"], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Lesion-level", "Patient-level"])
    ax.set_ylabel("Indeterminacy rate (%)")
    ax.set_title("VISION Indeterminacy: Lesion vs Patient Level")

    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig12_indeterminacy_cdf(
    margins: np.ndarray,
    interval_widths: np.ndarray,
    output_path: str | Path,
) -> None:
    """Figure 12: CDF of threshold margins with interval width overlay."""
    fig, ax1 = plt.subplots(figsize=(7, 5))

    sorted_margins = np.sort(np.abs(margins))
    cdf = np.arange(1, len(sorted_margins) + 1) / len(sorted_margins)
    ax1.plot(sorted_margins, cdf, color="C0", linewidth=2, label="CDF of |margin|")
    ax1.set_xlabel("|SUVmax − liver SUVmean|")
    ax1.set_ylabel("Cumulative fraction", color="C0")

    ax2 = ax1.twinx()
    median_width = np.median(interval_widths)
    ax1.axvline(median_width, color="C1", linestyle="--", linewidth=1.5,
                label=f"Median CI width = {median_width:.1f}")
    ax1.axvline(np.percentile(interval_widths, 75), color="C1", linestyle=":",
                alpha=0.5, label=f"75th pctl CI width")

    # Fraction affected at each width
    frac_at_median = (np.abs(margins) <= median_width).mean()
    ax1.annotate(f"{frac_at_median*100:.0f}% indeterminate\nat median CI width",
                 xy=(median_width, frac_at_median), xytext=(median_width * 1.5, frac_at_median - 0.1),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)

    ax1.legend(loc="lower right")
    ax1.set_title("Threshold Margin Distribution vs Conformal Interval Width")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fig13_ground_truth_decomposition(
    decomposition_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Figure 13: Stacked bar of segmentation error vs measurement uncertainty."""
    if "seg_error_fraction" not in decomposition_df.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    df = decomposition_df.sort_values("volume_quartile")
    x = range(len(df))
    seg_frac = df["seg_error_fraction"].values * 100
    meas_frac = (1 - df["seg_error_fraction"].values) * 100

    ax.bar(x, seg_frac, label="Segmentation error", color="C3", alpha=0.7)
    ax.bar(x, meas_frac, bottom=seg_frac, label="Measurement uncertainty", color="C0", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["volume_quartile"])
    ax.set_xlabel("Volume quartile")
    ax.set_ylabel("Fraction of interval width (%)")
    ax.set_title("Sources of SUV Uncertainty by Lesion Size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_all_figures(results_dir: str | Path) -> None:
    """Generate all figures from saved results.

    Looks for result files in results_dir and generates figures
    for any that exist.
    """
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # Check for each result file and generate corresponding figure
    coverage_path = tables_dir / "coverage_summary.csv"
    if coverage_path.exists():
        fig2_coverage_calibration(
            pd.read_csv(coverage_path),
            figures_dir / "fig2_coverage_calibration.png",
        )
        generated.append("fig2")

    meta_path = tables_dir / "meta_coverage.npz"
    if meta_path.exists():
        data = np.load(meta_path)
        fig9_meta_coverage_histogram(
            data["coverages"], data.get("target", 0.90),
            figures_dir / "fig9_meta_coverage.png",
        )
        generated.append("fig9")

    print(f"Generated {len(generated)} figures: {generated}")
    if not generated:
        print("No result files found. Run the evaluation pipeline first.")
