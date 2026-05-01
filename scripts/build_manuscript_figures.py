"""Build all manuscript figures + tables from the Phase 3 freeze on OSF.

Generates publication-quality figures (PNG 300dpi + PDF vector) and
formatted Markdown tables with full numerical content. Outputs:

    manuscript/figures/
        fig01_coverage_four_modes.{png,pdf}
        fig02_importance_weight_distributions.{png,pdf}
        fig03_per_feature_classifier_coefficients.{png,pdf}
        fig04_csi_vs_coverage_miss.{png,pdf}
        fig05_residuals_per_cohort.{png,pdf}
        fig06_coverage_by_volume_decile.{png,pdf}
        fig07_hecktor_centre_vendor_coverage.{png,pdf}
        fig08_gtvp_gtvn_coverage.{png,pdf}
        fig09_pipeline_overview.{png,pdf}     (schematic, drawn programmatically)

    manuscript/tables/
        table01_cohort_characteristics.md
        table02_phase2_wcv_summary.md
        table03_four_mode_coverage.md
        table04_support_overlap_diagnostic.md
        table05_csi.md
        table06_per_centre_coverage.md
        table07_per_class_coverage.md
        table08_per_quartile_coverage.md
        table09_hypothesis_verdicts.md

All figures use a consistent style (single-column = 3.5 in wide;
double-column = 7 in wide; 300 dpi rasters; PDF vector output).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Phase 3 freeze artefacts (OSF-locked SHAs)
PHASE3_DIR = PROJECT_ROOT / "results/phase3/amendment_11"
DIAG_DIR = PROJECT_ROOT / "results/phase3/diagnostics"
PHASE2_WCV_PARQUET = PROJECT_ROOT / "data/processed/phase2_autopet_iii_primary_wcv.parquet"

FIG_DIR = PROJECT_ROOT / "manuscript/figures"
TABLE_DIR = PROJECT_ROOT / "manuscript/tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Publication style: serif fonts, larger axis labels, white background
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.6,
})

# Colour scheme for the four modes (colour-blind-safe; consistent across figures)
MODE_COLORS = {
    "naive":              "#d62728",  # red
    "wcp":                "#1f77b4",  # blue
    "wcp_extended":       "#9467bd",  # purple
    "percohort_recal":    "#2ca02c",  # green
}
MODE_LABELS = {
    "naive":           "Naive transfer",
    "wcp":             "WCP-image",
    "wcp_extended":    "WCP-extended",
    "percohort_recal": "Per-cohort recal",
}
COHORT_LABELS = {
    "autopet_i":   "AutoPET-I (FDG WB)\nin-distribution",
    "autopet_iii": "AutoPET-III (PSMA WB)\nexternal",
    "hecktor":     "HECKTOR (FDG H&N)\nexternal",
}
NOMINAL = 0.90


# ===== Helpers =====


def _save(fig, name: str, dpi: int = 300):
    """Save fig as both PNG (raster, 300dpi) and PDF (vector)."""
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / name}.{{png,pdf}}")


def _ci_errorbar(p, n):
    """Clopper-Pearson 95% CI half-widths for binomial proportion."""
    from scipy.stats import beta
    if n == 0:
        return 0, 0
    k = int(round(p * n))
    lower = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    upper = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return p - lower, upper - p


def _load_phase3():
    cov = pd.read_parquet(PHASE3_DIR / "phase3_amendment_11_coverage_suvmax.parquet")
    csi = pd.read_csv(PHASE3_DIR / "phase3_amendment_11_csi_suvmax.csv")
    verdicts = pd.read_csv(PHASE3_DIR / "phase3_amendment_11_verdicts_suvmax.csv")
    with open(PHASE3_DIR / "phase3_amendment_11_metadata_suvmax.json") as f:
        meta = json.load(f)
    return cov, csi, verdicts, meta


# ===== Figure 1: Four-mode marginal coverage per cohort =====


def fig01_coverage_four_modes(cov: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    marg = cov[cov["stratum_kind"] == "marginal"].copy()
    cohorts = ["autopet_i", "autopet_iii", "hecktor"]
    modes = ["naive", "wcp", "wcp_extended", "percohort_recal"]
    n_modes = len(modes)
    width = 0.18

    for i, mode in enumerate(modes):
        sub = marg[marg["mode"] == mode]
        ys = []
        cis = []
        ns = []
        for cohort in cohorts:
            row = sub[sub["cohort"] == cohort]
            if len(row) == 0:
                ys.append(np.nan); cis.append((0, 0)); ns.append(0)
            else:
                p = float(row["coverage"].iloc[0])
                n = int(row["n_lesions"].iloc[0])
                ys.append(p); cis.append(_ci_errorbar(p, n)); ns.append(n)
        x = np.arange(len(cohorts)) + (i - n_modes / 2 + 0.5) * width
        ax.bar(x, ys, width=width, color=MODE_COLORS[mode], label=MODE_LABELS[mode],
               edgecolor="black", linewidth=0.4)
        # error bars
        err_low = [c[0] for c in cis]
        err_hi = [c[1] for c in cis]
        ax.errorbar(x, ys, yerr=[err_low, err_hi], fmt="none", ecolor="black",
                    elinewidth=0.6, capsize=2)

    ax.axhline(NOMINAL, ls="--", color="black", lw=0.8, alpha=0.7,
               label=f"Nominal {NOMINAL:.2f}")
    ax.set_xticks(np.arange(len(cohorts)))
    ax.set_xticklabels([COHORT_LABELS[c] for c in cohorts])
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.55, 1.00)
    ax.set_title("Phase 3 four-mode marginal coverage comparison (target SUV$_{max}$, α = 0.10)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    _save(fig, "fig01_coverage_four_modes")


# ===== Figure 2: Importance weight distributions =====


def fig02_importance_weight_distributions(meta: dict):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    cohort_names = ["autopet_iii", "hecktor"]

    for ax, cohort in zip(axes, cohort_names):
        wcp = meta["wcp_models"][cohort]
        wcp_ext = meta["wcp_ext_models"][cohort]

        labels = ["WCP-image\n(3 features)", "WCP-extended\n(16 features)"]
        # Use weight range as proxy for distribution; in real Fig 2 we'd have
        # the per-source-point weights, but the metadata has summary stats.
        clip_low = [wcp["weight_clip_low"], wcp_ext["weight_clip_low"]]
        clip_high = [wcp["weight_clip_high"], wcp_ext["weight_clip_high"]]
        median = [wcp["weight_median"], wcp_ext["weight_median"]]

        # Box-like representation (low, median, high)
        for i, (cl, m, ch) in enumerate(zip(clip_low, median, clip_high)):
            ax.plot([i, i], [cl, ch], color="black", linewidth=1.5)  # range
            ax.plot([i - 0.15, i + 0.15], [cl, cl], color="black", linewidth=1.5)
            ax.plot([i - 0.15, i + 0.15], [ch, ch], color="black", linewidth=1.5)
            ax.scatter([i], [m], color="red", zorder=5, s=40, marker="o")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_title(f"Target = {cohort.replace('_', '-').replace('autopet-', 'AutoPET-').replace('hecktor', 'HECKTOR')}\n"
                     f"AUC: img={wcp['classifier_auc']:.3f}, ext={wcp_ext['classifier_auc']:.3f}")
        ax.axhline(1.0, ls="--", color="grey", lw=0.6, alpha=0.6)

    axes[0].set_ylabel("Importance weight (log scale)")
    fig.suptitle("Importance-weight distributions: WCP-image vs WCP-extended", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig02_importance_weight_distributions")


# ===== Figure 3: Per-feature classifier coefficients =====


def fig03_per_feature_coefficients():
    """Hard-coded from SA-34 output until per-coef CSV is exported.
    Phase 3 driver currently prints these but doesn't write them; we use the
    metadata.json's `feature_means` along with reading the LR coefficients."""
    # Top 5 abs coefs per source-target pair, from the dry-run output:
    data_iii = [
        ("lesions_per_patient_in_cohort", +5.82),
        ("tracer_is_psma", +4.08),
        ("sphericity", +0.61),
        ("vendor_is_siemens", -0.39),
        ("vendor_is_ge", +0.32),
    ]
    data_h = [
        ("lesions_per_patient_in_cohort", -4.49),
        ("vendor_is_siemens", -0.94),
        ("vendor_is_ge", +0.74),
        ("centre_5", +0.53),
        ("centre_6", +0.44),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharex=True)
    for ax, (target, data) in zip(axes, [
        ("AutoPET-III (PSMA WB)", data_iii),
        ("HECKTOR (FDG H&N)", data_h),
    ]):
        names = [d[0].replace("_", " ") for d in data]
        coefs = [d[1] for d in data]
        colors = ["#1f77b4" if c > 0 else "#d62728" for c in coefs]
        ax.barh(np.arange(len(names)), coefs, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"Target = {target}")
        ax.set_xlabel("LR coefficient")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Per-feature classifier coefficients (top 5 |coef|; SA-34)", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig03_per_feature_classifier_coefficients")


# ===== Figure 4: CSI vs coverage miss =====


def fig04_csi_vs_coverage_miss(cov: pd.DataFrame, csi: pd.DataFrame):
    """Visualise relationship between CSI and naive-transfer coverage miss."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Match cohort names: csi has 'autopet_i_test', cov has 'autopet_i'
    csi_remap = csi.copy()
    csi_remap.loc[:, "cohort_match"] = csi_remap["cohort"].replace({"autopet_i_test": "autopet_i"})

    rows = []
    for _, c in csi_remap.iterrows():
        cohort = c["cohort_match"]
        marg = cov[(cov["mode"] == "naive") & (cov["cohort"] == cohort) & (cov["stratum_kind"] == "marginal")]
        if len(marg) == 0:
            continue
        rows.append({
            "cohort": cohort, "csi": float(c["csi"]),
            "coverage": float(marg["coverage"].iloc[0]),
            "miss_pp": float(marg["miss_pp"].iloc[0]),
            "n": int(marg["n_lesions"].iloc[0]),
        })
    df = pd.DataFrame(rows)

    sizes = df["n"] / df["n"].max() * 200 + 30
    ax.scatter(df["csi"], df["miss_pp"], s=sizes, c="#1f77b4",
               edgecolor="black", linewidth=0.6, alpha=0.85, zorder=5)

    for _, r in df.iterrows():
        offset = (0.15, 0.4) if r["cohort"] == "autopet_i" else (0.15, -0.7)
        ax.annotate(f"{r['cohort']}\nn={r['n']:,}",
                    xy=(r["csi"], r["miss_pp"]),
                    xytext=(r["csi"] + offset[0], r["miss_pp"] + offset[1]),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.4))

    ax.axvline(1.5, ls="--", color="black", lw=0.8, alpha=0.7)
    ax.text(1.52, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else -1,
            "CSI = 1.5\n(recalibration\nrecommended)", fontsize=8, va="top", color="black")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhspan(-2, 2, color="green", alpha=0.08, label="±2pp tolerance (H1, H11)")

    ax.set_xlabel("Calibration Shift Index (CSI)")
    ax.set_ylabel("Naive-transfer coverage miss (pp)")
    ax.set_title("CSI as a deployment diagnostic: predicts coverage shortfall")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, "fig04_csi_vs_coverage_miss")


# ===== Figure 5: Residuals per cohort (re-uses diagnostic plot) =====


def fig05_residuals_per_cohort():
    """Re-use the existing diagnostic residuals plot if present, else regenerate."""
    src = DIAG_DIR / "residuals_per_cohort.png"
    if src.exists():
        # Copy to manuscript figures
        import shutil
        shutil.copy2(src, FIG_DIR / "fig05_residuals_per_cohort.png")
        # Also try to copy PDF if exists
        src_pdf = src.with_suffix(".pdf")
        if src_pdf.exists():
            shutil.copy2(src_pdf, FIG_DIR / "fig05_residuals_per_cohort.pdf")
        print(f"  copied {src} -> {FIG_DIR / 'fig05_residuals_per_cohort.png'}")
        return

    # Regenerate from residuals.parquet
    res = pd.read_parquet(DIAG_DIR / "residuals.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), sharex=True, sharey=True)
    for ax, cohort in zip(axes, ["autopet_i", "autopet_iii", "hecktor"]):
        sub = res[res["cohort"] == cohort]
        inside = sub[sub["in_interval"]]
        outside = sub[~sub["in_interval"]]
        ax.scatter(inside["y_log_observed"], inside["y_log_point"],
                   s=4, alpha=0.3, c="#2ca02c", label=f"in (n={len(inside)})", rasterized=True)
        ax.scatter(outside["y_log_observed"], outside["y_log_point"],
                   s=4, alpha=0.5, c="#d62728", label=f"miss (n={len(outside)})", rasterized=True)
        lo = float(min(sub["y_log_observed"].min(), sub["y_log_point"].min()))
        hi = float(max(sub["y_log_observed"].max(), sub["y_log_point"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        cov = sub["in_interval"].mean()
        ax.set_title(f"{cohort}  (cov = {cov:.3f})")
        ax.set_xlabel("observed log(SUV$_{max}$ + 1)")
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel("predicted point estimate (log)")
    fig.tight_layout()
    _save(fig, "fig05_residuals_per_cohort")


# ===== Figure 6: Coverage by volume decile =====


def fig06_coverage_by_volume_decile():
    src = DIAG_DIR / "coverage_by_bin.parquet"
    if not src.exists():
        print(f"  skip: {src} missing")
        return
    bins = pd.read_parquet(src)
    vol_bins = bins[bins["binned_by"] == "volume_ml"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for cohort, grp in vol_bins.groupby("cohort"):
        ax.plot(np.arange(len(grp)), grp["coverage"].values, marker="o",
                linewidth=1.4, label=cohort)
    ax.axhline(NOMINAL, ls="--", color="black", lw=0.8, alpha=0.7,
               label=f"Nominal {NOMINAL:.2f}")
    ax.axhspan(NOMINAL - 0.02, NOMINAL + 0.02, color="green", alpha=0.08)
    ax.set_xlabel("Lesion volume decile (low → high)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage drift across the lesion-volume range (naive transfer)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    _save(fig, "fig06_coverage_by_volume_decile")


# ===== Figure 7: HECKTOR centre × vendor coverage =====


def fig07_hecktor_centre_vendor_coverage():
    src = DIAG_DIR / "centre_vendor_coverage.csv"
    if not src.exists():
        print(f"  skip: {src} missing")
        return
    df = pd.read_csv(src)
    df = df.sort_values("coverage")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bar_colors = []
    for _, r in df.iterrows():
        v = r["vendor"]
        bar_colors.append({"GE": "#1f77b4", "Siemens": "#ff7f0e", "Philips": "#2ca02c"}.get(v, "#7f7f7f"))
    labels = [f"{r['centre_name']}\n({r['vendor']}, n={int(r['n'])})" for _, r in df.iterrows()]
    bars = ax.barh(np.arange(len(df)), df["coverage"], color=bar_colors,
                   edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(labels)
    ax.axvline(NOMINAL, ls="--", color="black", lw=0.8, alpha=0.7,
               label=f"Nominal {NOMINAL:.2f}")
    ax.axvspan(NOMINAL - 0.05, NOMINAL + 0.05, color="green", alpha=0.08, label="±5pp tolerance (H12)")
    ax.set_xlim(0.55, 1.00)
    ax.set_xlabel("Empirical coverage")
    ax.set_title("HECKTOR per-centre coverage (naive transfer)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig07_hecktor_centre_vendor_coverage")


# ===== Figure 8: GTVp vs GTVn coverage =====


def fig08_gtvp_gtvn_coverage(cov: pd.DataFrame):
    cls = cov[(cov["cohort"] == "hecktor") & (cov["stratum_kind"] == "lesion_class")].copy()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    if len(cls):
        x = np.arange(len(cls))
        ax.bar(x, cls["coverage"], color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(cls)],
               edgecolor="black", linewidth=0.5)
        for i, (_, r) in enumerate(cls.iterrows()):
            err_low, err_high = _ci_errorbar(r["coverage"], int(r["n_lesions"]))
            ax.errorbar(i, r["coverage"], yerr=[[err_low], [err_high]],
                        fmt="none", ecolor="black", elinewidth=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['stratum_label']}\n(n={int(r['n_lesions'])})" for _, r in cls.iterrows()])
    ax.axhline(NOMINAL, ls="--", color="black", lw=0.8, alpha=0.7,
               label=f"Nominal {NOMINAL:.2f}")
    ax.axhspan(NOMINAL - 0.03, NOMINAL + 0.03, color="green", alpha=0.08, label="±3pp tolerance (SA-29)")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.55, 1.00)
    ax.set_title("HECKTOR per-class coverage (WCP-image; SA-29)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig08_gtvp_gtvn_coverage")


# ===== Figure 9: Pipeline overview schematic =====


def fig09_pipeline_overview():
    """Programmatic schematic of the end-to-end pipeline:

        Cohorts -> Phase 1 lesion extraction -> Phase 2 wCV reference
                -> Phase 3 four-mode conformal evaluation -> diagnostics
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()

    def box(x, y, w, h, label, fc, ec="black", fs=8.5):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.04,rounding_size=0.10",
                               linewidth=0.8, facecolor=fc, edgecolor=ec, zorder=2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=0.9, color="#444", zorder=1))

    cohort_fc = "#dbe9f6"
    p1_fc = "#fde8c4"
    p2_fc = "#e9d8f4"
    p3_fc = "#d4eedd"
    diag_fc = "#f8d7d7"

    # Row 1: cohorts
    box(0.2, 5.0, 2.8, 0.9,
        "AutoPET-I\n461 patients / 4,282 lesions\n(FDG WB, Siemens)", cohort_fc)
    box(3.5, 5.0, 2.9, 0.9,
        "AutoPET-III\n333 patients / 8,768 lesions\n(PSMA WB, Siemens+GE)", cohort_fc)
    box(6.9, 5.0, 2.9, 0.9,
        "HECKTOR 2025\n676 patients / 1,573 lesions\n(FDG H&N, 7 centres)", cohort_fc)

    # Row 2: Phase 1
    box(0.2, 3.6, 9.6, 0.8,
        "Phase 1 - lesion table extraction (>=1 mL filter, sec 3.9 image review)\n"
        "schema: volume_ml, surface_area_cm2, sphericity, softmax_mean, softmax_entropy, SUVmax/peak/mean",
        p1_fc, fs=8.2)

    # Row 3: Phase 2
    box(0.2, 2.4, 4.7, 0.8,
        "Phase 2 - within-replicate CV reference\n"
        "Poisson noise injection (sec 3.5 primary path)\n"
        "AutoPET-III, 333 patients, 2,178 (lesion x dose) rows",
        p2_fc, fs=8.0)

    # Row 4: Phase 3 (four modes)
    box(5.1, 2.4, 4.7, 0.8,
        "Phase 3 - split conformal calibration on AutoPET-I (n=2,489 cal)\n"
        "alpha = 0.10, target = log(1+SUVmax), LightGBM quantile heads",
        p3_fc, fs=8.0)

    # Row 5: Modes
    box(0.2, 1.2, 2.2, 0.7, "Mode 1\nNaive transfer", p3_fc, fs=8.5)
    box(2.6, 1.2, 2.2, 0.7, "Mode 2\nWCP-image", p3_fc, fs=8.5)
    box(5.0, 1.2, 2.2, 0.7, "Mode 3\nWCP-extended", p3_fc, fs=8.5)
    box(7.4, 1.2, 2.4, 0.7, "Mode 4\nPer-cohort recal", p3_fc, fs=8.5)

    # Row 6: Diagnostics
    box(0.2, 0.0, 9.6, 0.8,
        "Diagnostics: marginal + Mondrian (volume, centre, lesion class) coverage; "
        "support-overlap (AUC, ESS); CSI > 1.5 trigger; SHA-256 freeze on OSF j5ry4",
        diag_fc, fs=8.2)

    # Arrows: cohorts -> Phase 1
    for x in [1.6, 4.95, 8.35]:
        arrow(x, 5.0, x, 4.4)
    # Phase 1 -> Phase 2 and Phase 3
    arrow(2.55, 3.6, 2.55, 3.2)
    arrow(7.45, 3.6, 7.45, 3.2)
    # Phase 2 -> Phase 3 (lateral)
    arrow(4.9, 2.8, 5.1, 2.8)
    # Phase 3 -> modes
    for x in [1.3, 3.7, 6.1, 8.6]:
        arrow(7.45, 2.4, x, 1.9)
    # Modes -> diagnostics
    for x in [1.3, 3.7, 6.1, 8.6]:
        arrow(x, 1.2, x, 0.8)

    ax.text(5.0, 6.05, "Project 79: Conformal SUV Theranostic - end-to-end pipeline",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=cohort_fc, edgecolor="black", label="Cohorts (Phase 1 input)"),
        mpatches.Patch(facecolor=p1_fc, edgecolor="black", label="Phase 1 - lesion extraction"),
        mpatches.Patch(facecolor=p2_fc, edgecolor="black", label="Phase 2 - wCV reference"),
        mpatches.Patch(facecolor=p3_fc, edgecolor="black", label="Phase 3 - conformal calibration + 4 modes"),
        mpatches.Patch(facecolor=diag_fc, edgecolor="black", label="Diagnostics + freeze"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=7.5)

    fig.tight_layout()
    _save(fig, "fig09_pipeline_overview")


# ===== Figure 10: Clinical-implications threshold-flip rate =====


def fig10_threshold_flip_rate():
    """Proportion of lesions whose 90% conformal interval straddles a representative
    clinical decision threshold, plotted against the threshold value in natural
    SUVmax units (back-transformed from log-space intervals).

    A 'flip' occurs when the 90% interval contains the threshold — i.e., the lower
    bound is below and the upper bound is above. Conceptually this maps to the
    proportion of theranostic eligibility / response decisions that would be
    indeterminate under measurement repetition at the conformal coverage level.
    Liver SUVmean reference values from PSMA and FDG clinical literature are
    overlaid as vertical guides.
    """
    res_path = DIAG_DIR / "residuals.parquet"
    if not res_path.exists():
        print(f"  skip fig10: {res_path} missing")
        return
    res = pd.read_parquet(res_path)
    # Back-transform log(1+SUV) intervals to natural SUV.
    res = res.copy()
    res.loc[:, "suv_lower"] = np.expm1(res["y_log_lower"].astype(float))
    res.loc[:, "suv_upper"] = np.expm1(res["y_log_upper"].astype(float))
    res.loc[:, "suv_observed"] = np.expm1(res["y_log_observed"].astype(float))

    thresholds = np.linspace(0.5, 12.0, 121)
    cohorts = ["autopet_i", "autopet_iii", "hecktor"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax0, ax1 = axes

    for cohort in cohorts:
        sub = res[res["cohort"] == cohort]
        if len(sub) == 0:
            continue
        flips = []
        for thr in thresholds:
            straddle = (sub["suv_lower"] < thr) & (sub["suv_upper"] > thr)
            flips.append(float(straddle.mean()) * 100)
        color = {
            "autopet_i": "#999999",
            "autopet_iii": "#1f77b4",
            "hecktor": "#ff7f0e",
        }[cohort]
        label = COHORT_LABELS[cohort].replace("\n", " ")
        ax0.plot(thresholds, flips, color=color, lw=1.6, label=label)

    # Reference liver SUV bands
    ax0.axvspan(2.0, 3.5, color="#999999", alpha=0.10, label="Typical FDG liver SUVmean (~2-3.5)")
    ax0.axvspan(5.0, 7.0, color="#1f77b4", alpha=0.10, label="Typical PSMA liver SUVmean (~5-7)")

    ax0.set_xlabel("Decision threshold (SUVmax, natural scale)")
    ax0.set_ylabel("% of lesions with 90% CI straddling threshold")
    ax0.set_title("Clinical-implications: indeterminacy rate vs decision threshold")
    ax0.set_xlim(thresholds.min(), thresholds.max())
    ax0.set_ylim(0, 100)
    ax0.legend(loc="upper right", fontsize=7)

    # Right panel: indeterminacy at three benchmark thresholds, per cohort
    benchmarks = [2.5, 4.0, 6.0]  # representative liver / VISION-style thresholds
    bar_x = np.arange(len(benchmarks))
    width = 0.28
    for i, cohort in enumerate(cohorts):
        sub = res[res["cohort"] == cohort]
        if len(sub) == 0:
            continue
        bar_h = []
        for thr in benchmarks:
            straddle = (sub["suv_lower"] < thr) & (sub["suv_upper"] > thr)
            bar_h.append(float(straddle.mean()) * 100)
        color = {
            "autopet_i": "#999999",
            "autopet_iii": "#1f77b4",
            "hecktor": "#ff7f0e",
        }[cohort]
        offset = (i - 1) * width
        bars = ax1.bar(bar_x + offset, bar_h, width=width, color=color,
                       edgecolor="black", linewidth=0.4,
                       label=COHORT_LABELS[cohort].split("\n")[0])
        for b, h in zip(bars, bar_h):
            ax1.annotate(f"{h:.0f}%", xy=(b.get_x() + b.get_width()/2, h),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(bar_x)
    ax1.set_xticklabels([f"SUV={t:.1f}" for t in benchmarks])
    ax1.set_ylabel("% indeterminate")
    ax1.set_ylim(0, max(75, ax1.get_ylim()[1]))
    ax1.set_title("Indeterminacy at representative thresholds")
    ax1.legend(loc="upper right", fontsize=6.5)

    fig.suptitle("Figure 10. Conformal-interval indeterminacy at clinical decision thresholds",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    _save(fig, "fig10_clinical_indeterminacy")


# ===== Tables =====


def write_tables(cov: pd.DataFrame, csi: pd.DataFrame, verdicts: pd.DataFrame, meta: dict):
    """Write all manuscript tables as Markdown for easy review/conversion."""

    # Table 1: Cohort characteristics (manually compiled from PROGRESS / data_snapshot_log)
    t1 = """# Table 1. Cohort characteristics

| Cohort | n_patients | n_lesions | Tracer | Anatomy | Vendors | Centres | Role |
|---|---|---|---|---|---|---|---|
| AutoPET-I (FDAT) | 461 | 4,282 | FDG | Whole-body | Siemens (1) | 1 (Tübingen) | Train + cal + test |
| AutoPET-III (TCIA) | 333 | 8,768 | PSMA (¹⁸F + ⁶⁸Ga) | Whole-body | Siemens, GE | LMU Munich | External test |
| HECKTOR 2025 | 676 | 1,573 | FDG | Head & neck | Siemens, GE, Philips | 7 (CHUM, CHUS, CHUP, CHUV, MDA, USZ, HMR) | External test |
| **Total** | **1,470** | **14,621** | | | | | |

Notes: AutoPET-I lesion count after pre-registered ≥1 mL filter and Amendment-5 §3.9 image review (2 of 4,282 excluded). AutoPET-III lesion count after Amendment-5 INSUFFICIENT_AGREEMENT-branch full review (3 of 8,771 excluded). HECKTOR lesion count post-§3.9 with 0 exclusions (max SUVmax 43.14, below the >50 trigger).
"""
    (TABLE_DIR / "table01_cohort_characteristics.md").write_text(t1)

    # Table 2: Phase 2 wCV reference summary
    if PHASE2_WCV_PARQUET.exists():
        p2 = pd.read_parquet(PHASE2_WCV_PARQUET)
        n_lesions_unique = p2.groupby(["case_id", "lesion_id"]).ngroups
        n_patients = p2["case_id"].nunique()
        t2 = "# Table 2. Phase 2 within-replicate coefficient of variation (wCV) reference\n\n"
        t2 += (f"Source cohort: AutoPET-III ({n_patients} patients, {n_lesions_unique} unique lesions, "
               f"{len(p2):,} (lesion x dose-fraction) replicate rows; Poisson noise injection per pre-reg "
               f"section 3.5; primary path executed after section-3.5 decision gate).\n\n")
        t2 += "| Dose fraction | n_replicate_rows | n_lesions | wCV SUVmax % (median) | wCV SUVmax % (IQR) | wCV SUVpeak % (median) | wCV SUVmean % (median) |\n"
        t2 += "|---|---|---|---|---|---|---|\n"
        for df_val in sorted(p2["dose_fraction"].unique()):
            sub = p2[p2["dose_fraction"] == df_val]
            n_rep = len(sub)
            n_les = sub.groupby(["case_id", "lesion_id"]).ngroups
            mx_med = float(sub["wcv_suvmax_pct"].median())
            mx_q1 = float(sub["wcv_suvmax_pct"].quantile(0.25))
            mx_q3 = float(sub["wcv_suvmax_pct"].quantile(0.75))
            pk_med = float(sub["wcv_suvpeak_pct"].median())
            mn_med = float(sub["wcv_suvmean_pct"].median())
            t2 += (f"| {df_val:.2f} | {n_rep:,} | {n_les:,} | {mx_med:.2f} | "
                   f"[{mx_q1:.2f}, {mx_q3:.2f}] | {pk_med:.2f} | {mn_med:.2f} |\n")
        t2 += ("\nNotes: wCV is the within-replicate coefficient of variation across n_replicates Poisson-resampled "
               "PET volumes per (lesion, dose-fraction). dose_fraction = 1.0 corresponds to full clinical statistics; "
               "lower fractions emulate count-poor conditions. Per-scanner sensitivity calibration uses NEMA NU-2 "
               "coefficients (see supplementary S3.2). The full wCV reference parquet is recorded in "
               "osf/data_snapshot_log.md and informs the test-retest variability budget feeding Phase 3 calibration.\n")
        (TABLE_DIR / "table02_phase2_wcv_summary.md").write_text(t2)
    else:
        print(f"  warn: {PHASE2_WCV_PARQUET} missing; table02 not written")

    # Table 3: four-mode coverage
    marg = cov[cov["stratum_kind"] == "marginal"].copy()
    marg = marg.sort_values(["cohort", "mode"])
    rows = []
    for cohort in ["autopet_i", "autopet_iii", "hecktor"]:
        for mode in ["naive", "wcp", "wcp_extended", "percohort_recal"]:
            sub = marg[(marg["cohort"] == cohort) & (marg["mode"] == mode)]
            if len(sub) == 0:
                rows.append({"cohort": cohort, "mode": mode, "coverage": "—", "miss_pp": "—",
                             "ci": "—", "n": "—"})
                continue
            r = sub.iloc[0]
            ci_low = float(r["ci_lower"]) * 100
            ci_high = float(r["ci_upper"]) * 100
            rows.append({
                "cohort": cohort, "mode": mode,
                "coverage": f"{r['coverage']*100:.1f}%",
                "miss_pp": f"{r['miss_pp']:+.1f}pp",
                "ci": f"[{ci_low:.1f}, {ci_high:.1f}]",
                "n": f"{int(r['n_lesions']):,}",
            })
    table3 = "# Table 3. Phase 3 four-mode marginal coverage comparison (target SUVmax, α=0.10)\n\n"
    table3 += "| Cohort | Mode | Coverage | 95% CI (pp) | Miss (pp) | n_lesions |\n"
    table3 += "|---|---|---|---|---|---|\n"
    for r in rows:
        cohort_label = {"autopet_i": "AutoPET-I (test)", "autopet_iii": "AutoPET-III", "hecktor": "HECKTOR"}[r["cohort"]]
        table3 += f"| {cohort_label} | {MODE_LABELS[r['mode']]} | {r['coverage']} | {r['ci']} | {r['miss_pp']} | {r['n']} |\n"
    (TABLE_DIR / "table03_four_mode_coverage.md").write_text(table3)

    # Table 4: support-overlap diagnostic
    t4 = "# Table 4. WCP-extended support-overlap diagnostic per source-target pair (Amendment 12 §12b)\n\n"
    t4 += "| Source | Target | Classifier AUC | Weight dispersion | ESS / n_source | Verdict | Flagged reasons |\n"
    t4 += "|---|---|---|---|---|---|---|\n"
    for cohort, info in meta["wcp_ext_models"].items():
        verdict_label = {"red": "RED", "amber": "AMBER", "green": "GREEN"}.get(info["support_verdict"], info["support_verdict"])
        flagged = "; ".join(info["flagged_reasons"]) if info["flagged_reasons"] else "—"
        t4 += (f"| AutoPET-I cal | {cohort} | {info['classifier_auc']:.4f} | "
               f"{info['weight_dispersion']:.2f} | {info['ess_ratio']:.4f} | {verdict_label} | {flagged} |\n")
    t4 += "\nThresholds: AUC > 0.99 -> Red (support violation); AUC > 0.95 OR weight dispersion > 100 OR ESS/n < 0.30 -> Amber.\n"
    (TABLE_DIR / "table04_support_overlap_diagnostic.md").write_text(t4)

    # Table 5: CSI
    t5 = "# Table 5. Calibration Shift Index per cohort (Amendment 11 §11e)\n\n"
    t5 += "| Cohort | n_lesions | q̂(AutoPET-I cal) | q̂(self) | CSI | Recalibration recommended (CSI > 1.5)? |\n"
    t5 += "|---|---|---|---|---|---|\n"
    cohort_label_map = {"autopet_i_test": "AutoPET-I test (in-distribution)",
                        "autopet_iii": "AutoPET-III (external)",
                        "hecktor": "HECKTOR (external)"}
    for _, r in csi.iterrows():
        flag = "TRUE" if r["recalibration_recommended"] else "FALSE"
        t5 += (f"| {cohort_label_map.get(r['cohort'], r['cohort'])} | {int(r['n_lesions']):,} | "
               f"{float(r['q_hat_autopet_i_cal']):.4f} | {float(r['q_hat_self']):.4f} | "
               f"{float(r['csi']):.3f} | {flag} |\n")
    (TABLE_DIR / "table05_csi.md").write_text(t5)

    # Table 9: hypothesis verdicts
    t9 = "# Table 9. Hypothesis verdicts (Phase 3 freeze, locked on OSF j5ry4)\n\n"
    t9 += "| Hypothesis | Method | Tolerance | Result | Verdict |\n"
    t9 += "|---|---|---|---|---|\n"
    h_rows = [
        ("H1", "Naive transfer (AutoPET-III marginal)", "±2pp", "−8.85pp", "FAIL"),
        ("H11", "Naive transfer (HECKTOR marginal)", "±2pp", "−9.39pp", "FAIL"),
        ("H12", "Naive transfer (HECKTOR per-centre, n≥30)", "±5pp", "−23.87pp (worst)", "FAIL"),
        ("H13", "WCP-image marginal (Amendment 11)", "±2pp", "−9.83pp (max)", "FAIL"),
        ("H14", "WCP-image per-centre (Amendment 11, n≥30)", "±5pp", "−23.87pp (max)", "FAIL"),
        ("**H15**", "**WCP-extended marginal (Amendment 12)**", "**±2pp conditional**", "Support violation; AUC = 1.0000", "**NOT_TESTED** (Red diag.)"),
        ("**SA-31**", "**Per-cohort recalibration marginal (Amendment 11 §11d)**", "**±2pp**", "**+1.56pp / +0.58pp**", "**PASS**"),
        ("SA-29 (GTVp)", "Naive (HECKTOR GTVp-only)", "±3pp", "+2.86pp", "PASS"),
        ("SA-29 (GTVn)", "Naive (HECKTOR GTVn-only)", "±3pp", "−17.27pp", "FAIL"),
    ]
    for h, m, tol, res, v in h_rows:
        t9 += f"| {h} | {m} | {tol} | {res} | {v} |\n"
    (TABLE_DIR / "table09_hypothesis_verdicts.md").write_text(t9)

    # Table 6: HECKTOR per-centre coverage (naive transfer + WCP-image where present)
    cv_csv = DIAG_DIR / "centre_vendor_coverage.csv"
    centre_rows = cov[(cov["cohort"] == "hecktor") & (cov["stratum_kind"] == "centre")].copy()
    if cv_csv.exists():
        cvdf = pd.read_csv(cv_csv).copy()
        cvdf.loc[:, "centre_id_int"] = cvdf["centre_id"].astype(int)
    else:
        cvdf = None
    t6 = "# Table 6. HECKTOR per-centre coverage (naive transfer vs WCP-image; nominal 0.90; H12/H14)\n\n"
    t6 += "| Centre | Vendor | n_lesions | Naive coverage | Naive miss (pp) | WCP-image coverage | WCP-image miss (pp) |\n"
    t6 += "|---|---|---|---|---|---|---|\n"
    if cvdf is not None:
        for _, r in cvdf.sort_values("coverage").iterrows():
            cid = int(r["centre_id"])
            wcp_row = centre_rows[(centre_rows["mode"] == "wcp")
                                   & (centre_rows["stratum_label"].astype(str) == str(cid))]
            if len(wcp_row):
                wcp_cov = float(wcp_row["coverage"].iloc[0])
                wcp_miss = float(wcp_row["miss_pp"].iloc[0])
                wcp_cov_s = f"{wcp_cov*100:.1f}%"
                wcp_miss_s = f"{wcp_miss:+.1f}pp"
            else:
                wcp_cov_s, wcp_miss_s = "—", "—"
            t6 += (f"| {r['centre_name']} | {r['vendor']} | {int(r['n']):,} | "
                   f"{r['coverage']*100:.1f}% | {r['miss_pp']:+.2f}pp | "
                   f"{wcp_cov_s} | {wcp_miss_s} |\n")
        t6 += "\nNotes: Naive-transfer columns are produced by the diagnostic driver and reflect a single shared q_hat from the AutoPET-I calibration cohort; the WCP-image columns are the Amendment 11 conditional-coverage stratum (HECKTOR target only, n_lesions >= 30 per centre). H12 (>=5pp tolerance, n>=30) and H14 (WCP-image equivalent) both fail at multiple centres; see narrative.\n"
    else:
        t6 += "| (centre_vendor_coverage.csv missing) | | | | | | |\n"
    (TABLE_DIR / "table06_per_centre_coverage.md").write_text(t6)

    # Table 7: HECKTOR per-class (GTVp / GTVn) coverage
    cls_rows = cov[(cov["cohort"] == "hecktor") & (cov["stratum_kind"] == "lesion_class")].copy()
    t7 = "# Table 7. HECKTOR per-class coverage: primary tumour (GTVp) vs lymph node (GTVn) (SA-29; Amendment 8)\n\n"
    t7 += "| Mode | Lesion class | n_lesions | Coverage | 95% CI (proportion) | Miss (pp) | Median interval width (log) |\n"
    t7 += "|---|---|---|---|---|---|---|\n"
    if len(cls_rows):
        for _, r in cls_rows.sort_values(["mode", "stratum_label"]).iterrows():
            t7 += (f"| {MODE_LABELS.get(r['mode'], r['mode'])} | {r['stratum_label']} | "
                   f"{int(r['n_lesions']):,} | {r['coverage']*100:.1f}% | "
                   f"[{r['ci_lower']*100:.1f}, {r['ci_upper']*100:.1f}] | "
                   f"{r['miss_pp']:+.2f}pp | {r['median_width_log']:.4f} |\n")
        t7 += ("\nNotes: GTVp = HECKTOR primary tumour gross tumour volume, GTVn = involved lymph node gross "
               "tumour volume; voxel-majority class assignment per Amendment 8 section 8b. SA-29 tolerance is "
               "+/- 3pp. The GTVp arm passes; the GTVn arm fails substantially. Mechanism: typical lymph nodes "
               "are smaller and lower-uptake than primary tumours, shifting the residual distribution beyond "
               "what the AutoPET-I calibration q_hat covers. The per-class asymmetry motivates per-class "
               "quantile heads in future work.\n")
    else:
        t7 += "| (no per-class strata in freeze) | | | | | | |\n"
    (TABLE_DIR / "table07_per_class_coverage.md").write_text(t7)

    # Table 8: per-volume-decile coverage (naive transfer; from coverage_by_bin diagnostic)
    bins_path = DIAG_DIR / "coverage_by_bin.parquet"
    t8 = "# Table 8. Per-volume-decile coverage (naive transfer; nominal 0.90)\n\n"
    if bins_path.exists():
        bins = pd.read_parquet(bins_path)
        vol = bins[bins["binned_by"] == "volume_ml"].copy()
        t8 += "| Cohort | Decile | Volume range (mL) | n_lesions | Coverage | Miss (pp) |\n"
        t8 += "|---|---|---|---|---|---|\n"
        for cohort, grp in vol.groupby("cohort"):
            grp_sorted = grp.reset_index(drop=True)
            for i, r in grp_sorted.iterrows():
                miss_pp = (r["coverage"] - NOMINAL) * 100
                t8 += (f"| {cohort} | D{i+1} | "
                       f"[{r['bin_low']:.2f}, {r['bin_high']:.2f}] | {int(r['n']):,} | "
                       f"{r['coverage']*100:.1f}% | {miss_pp:+.2f}pp |\n")
        t8 += ("\nNotes: Deciles are computed within each cohort separately (so D1 of AutoPET-I is not directly "
               "comparable to D1 of HECKTOR in absolute mL terms). The per-decile coverage curve reveals where "
               "the naive-transfer interval misses most: in AutoPET-III, the smallest-volume decile drops to "
               "75.1% (-14.9pp) and the largest-volume decile to 71.4% (-18.6pp), confirming the H2 conditional-"
               "coverage failure. AutoPET-I (in-distribution test) decile coverage stays within ~7pp of nominal.\n")
    else:
        t8 += "| (coverage_by_bin.parquet missing) | | | | | |\n"
    (TABLE_DIR / "table08_per_quartile_coverage.md").write_text(t8)

    print(f"  wrote 9 manuscript tables to {TABLE_DIR}")


def main() -> None:
    print("=" * 70)
    print("Building manuscript figures + tables from Phase 3 freeze")
    print("=" * 70)

    cov, csi, verdicts, meta = _load_phase3()
    print(f"\nLoaded Phase 3 freeze: {len(cov)} coverage rows, {len(csi)} CSI rows")
    print(f"Output dirs:\n  {FIG_DIR}\n  {TABLE_DIR}\n")

    print("Figures:")
    fig01_coverage_four_modes(cov)
    fig02_importance_weight_distributions(meta)
    fig03_per_feature_coefficients()
    fig04_csi_vs_coverage_miss(cov, csi)
    fig05_residuals_per_cohort()
    fig06_coverage_by_volume_decile()
    fig07_hecktor_centre_vendor_coverage()
    fig08_gtvp_gtvn_coverage(cov)
    fig09_pipeline_overview()
    fig10_threshold_flip_rate()

    print("\nTables:")
    write_tables(cov, csi, verdicts, meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
