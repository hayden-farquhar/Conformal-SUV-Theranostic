"""Amendment 11 Phase 3 driver: side-by-side naive + WCP + per-cohort recalibration.

Produces the manuscript-ready Phase 3 results table comparing:

    Mode 1 (naive)      -- Amendment 9 baseline; AutoPET-I cal q_hat applied to
                            external cohorts. Tests H1, H11, H12. Reported as the
                            naive-transfer comparator (the negative finding from
                            the 2026-04-30 dry-run).

    Mode 2 (WCP)        -- Amendment 11 §11b primary methodology. Importance
                            weights estimated via discriminative classifier on
                            the §11c 3-feature subset, weighted threshold per
                            Tibshirani 2019. Tests H13 (marginal), H14 (per-centre).

    Mode 3 (per-cohort) -- Amendment 11 §11d secondary. 20% patient-level
                            holdout per external cohort -> cohort-local q_hat
                            -> evaluate on remaining 80%. Tests SA-31.

Plus the Calibration Shift Index (CSI) table per §11e and per-feature
importance-weight magnitudes per §11f SA-33.

Modes:
    --dry-run     evaluate all three modes + CSI; print results; do not write
                  locked artefacts.
    --commit      evaluate + write to results/phase3/amendment_11/. Once
                  committed, the WCP / per-cohort / CSI evaluations are final.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN
Amendment 11: 2026-04-30 (osf/amendment_log.md v11 SHA d48d1f1e...)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.conformal.cqr import (
    calibrate_cqr,
    compute_nonconformity_scores,
    predict_intervals_array,
    train_quantile_regressors,
)
from src.conformal.coverage import compute_coverage
from src.conformal.weighted import (
    PERCOHORT_HOLDOUT_FRACTION,
    PERCOHORT_HOLDOUT_SEED,
    WCP_EXTENDED_FEATURE_COLS,
    WCP_FEATURE_COLS,
    build_extended_wcp_features,
    calibration_shift_index,
    diagnose_support_overlap,
    fit_wcp_classifier,
    patient_level_holdout_split,
    weighted_conformal_threshold,
)

# Reuse all loading + feature config from the original Phase 3 driver
from scripts.phase3_evaluate import (
    ALPHA, NOMINAL, FEATURE_COLS, PATHS,
    TOL_CLASS_PP, TOL_CONDITIONAL_PP, TOL_MARGINAL_PP, MIN_CELL_SIZE,
    load_autopet_i, load_autopet_iii, load_hecktor,
    transform_target,
)


# Output dir for Amendment 11 results
OUT_DIR = PROJECT_ROOT / "results/phase3/amendment_11"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CoverageRow:
    cohort: str
    mode: str                          # 'naive' | 'wcp' | 'percohort_recal'
    stratum_kind: str                  # 'marginal' | 'centre' | 'lesion_class'
    stratum_label: str
    n_lesions: int
    coverage: float
    ci_lower: float
    ci_upper: float
    miss_pp: float                     # signed, percentage points
    median_width_log: float
    q_hat_used: float


def _evaluate_with_qhat(
    cohort: str, mode: str, stratum_kind: str, stratum_label: str,
    lo: np.ndarray, hi: np.ndarray, y_log: np.ndarray, q_hat: float,
) -> CoverageRow:
    lower, upper, widths = predict_intervals_array(lo, hi, q_hat)
    cov = compute_coverage(y_log, lower, upper, nominal=NOMINAL, label=f"{cohort}_{mode}_{stratum_label}")
    return CoverageRow(
        cohort=cohort, mode=mode, stratum_kind=stratum_kind, stratum_label=stratum_label,
        n_lesions=int(len(y_log)),
        coverage=float(cov.coverage),
        ci_lower=float(cov.ci_lower), ci_upper=float(cov.ci_upper),
        miss_pp=float((cov.coverage - NOMINAL) * 100),
        median_width_log=float(np.median(widths)),
        q_hat_used=float(q_hat),
    )


def run_amendment_11(target: str, commit: bool) -> dict:
    print("=" * 78)
    print(f"Amendment 11 Phase 3 evaluation -- target = {target}")
    print(f"Modes: naive (baseline) + WCP (primary) + per-cohort recal (secondary)")
    print("=" * 78)

    # Provenance: SHA-256 of every input artefact
    src_shas = {k: _sha256(p) for k, p in PATHS.items()
                if p.suffix in (".parquet", ".json", ".csv") and p.exists()}
    src_shas["driver_script"] = _sha256(Path(__file__))
    src_shas["weighted_module"] = _sha256(PROJECT_ROOT / "src/conformal/weighted.py")
    src_shas["cqr_module"] = _sha256(PROJECT_ROOT / "src/conformal/cqr.py")
    print()
    print("Input artefact SHA-256:")
    for k, v in src_shas.items():
        print(f"  {k:<25} {v[:16]}...")
    print()

    # Load
    train, cal, test = load_autopet_i()
    autopet_iii = load_autopet_iii()
    hecktor = load_hecktor()

    def X(df): return df[FEATURE_COLS].to_numpy()
    def y(df): return transform_target(df[target].to_numpy())

    print(f"\nTraining LightGBM quantile regressors (alpha={ALPHA})...")
    lower_model, upper_model = train_quantile_regressors(X(train), y(train), alpha=ALPHA)

    # Predictions on cal + each cohort
    lo_cal,  hi_cal  = lower_model.predict(X(cal)),         upper_model.predict(X(cal))
    lo_t1,   hi_t1   = lower_model.predict(X(test)),        upper_model.predict(X(test))
    lo_t3,   hi_t3   = lower_model.predict(X(autopet_iii)), upper_model.predict(X(autopet_iii))
    lo_h,    hi_h    = lower_model.predict(X(hecktor)),     upper_model.predict(X(hecktor))

    # Nonconformity scores on AutoPET-I cal (used by both naive q_hat AND WCP)
    cal_scores = compute_nonconformity_scores(y(cal), lo_cal, hi_cal)

    # === Mode 1: Naive (baseline) ===
    print("\n--- Mode 1: Naive transfer (Amendment 9 baseline) ---")
    cqr_marginal = calibrate_cqr(y(cal), lo_cal, hi_cal, alpha=ALPHA, stratum_name="all")
    q_hat_naive = cqr_marginal.q_hat
    print(f"  q_hat_naive = {q_hat_naive:.4f}")
    naive_rows: list[CoverageRow] = []
    naive_rows.append(_evaluate_with_qhat(
        "autopet_i", "naive", "marginal", "all",
        lo_t1, hi_t1, y(test), q_hat_naive,
    ))
    naive_rows.append(_evaluate_with_qhat(
        "autopet_iii", "naive", "marginal", "all",
        lo_t3, hi_t3, y(autopet_iii), q_hat_naive,
    ))
    naive_rows.append(_evaluate_with_qhat(
        "hecktor", "naive", "marginal", "all",
        lo_h, hi_h, y(hecktor), q_hat_naive,
    ))
    for r in naive_rows:
        print(f"    {r.cohort:<14} cov={r.coverage:.4f}  miss={r.miss_pp:+.2f}pp")

    # === Mode 2: WCP (Amendment 11 §11b primary) ===
    print("\n--- Mode 2: Weighted Conformal Prediction (Amendment 11 primary) ---")
    wcp_rows: list[CoverageRow] = []
    wcp_models: dict[str, dict] = {}
    for cohort_name, cohort_df, lo_c, hi_c in [
        ("autopet_iii", autopet_iii, lo_t3, hi_t3),
        ("hecktor",     hecktor,     lo_h,  hi_h),
    ]:
        print(f"  Fitting WCP classifier (AutoPET-I cal -> {cohort_name})...")
        wcp = fit_wcp_classifier(cal[WCP_FEATURE_COLS], cohort_df[WCP_FEATURE_COLS])
        q_hat_W = weighted_conformal_threshold(cal_scores, wcp.source_weights, alpha=ALPHA)
        print(f"    classifier AUC = {wcp.classifier_auc:.4f}  weight range [{wcp.weight_clip_low:.3f}, {wcp.weight_clip_high:.3f}]")
        print(f"    q_hat_W = {q_hat_W:.4f}  (vs naive {q_hat_naive:.4f}; ratio {q_hat_W / q_hat_naive:.2f})")
        row = _evaluate_with_qhat(
            cohort_name, "wcp", "marginal", "all", lo_c, hi_c, y(cohort_df), q_hat_W,
        )
        wcp_rows.append(row)
        print(f"    cov={row.coverage:.4f}  miss={row.miss_pp:+.2f}pp")
        wcp_models[cohort_name] = {
            "classifier_auc": wcp.classifier_auc,
            "weight_clip_low": wcp.weight_clip_low,
            "weight_clip_high": wcp.weight_clip_high,
            "weight_median": float(np.median(wcp.source_weights)),
            "weight_mean": float(np.mean(wcp.source_weights)),
            "q_hat_W": q_hat_W,
            "feature_means": wcp.feature_means,
        }

    # H14: per-centre WCP for HECKTOR (cells with n>=30)
    print("\n  Per-centre WCP for HECKTOR (H14, n>=30 floor):")
    for centre_id, cgroup in hecktor.groupby("centre_id"):
        n = len(cgroup)
        if n < MIN_CELL_SIZE:
            continue
        wcp_c = fit_wcp_classifier(cal[WCP_FEATURE_COLS], cgroup[WCP_FEATURE_COLS])
        q_hat_Wc = weighted_conformal_threshold(cal_scores, wcp_c.source_weights, alpha=ALPHA)
        idx = cgroup.index
        lo_c = lo_h[hecktor.index.isin(idx)]
        hi_c = hi_h[hecktor.index.isin(idx)]
        y_c = y(cgroup)
        row = _evaluate_with_qhat(
            "hecktor", "wcp", "centre", str(int(centre_id)),
            lo_c, hi_c, y_c, q_hat_Wc,
        )
        wcp_rows.append(row)
        print(f"    centre {int(centre_id):>2}  n={n:>3}  q_hat_W={q_hat_Wc:.4f}  cov={row.coverage:.4f}  miss={row.miss_pp:+.2f}pp")

    # SA-29 under WCP: HECKTOR per-class
    print("\n  HECKTOR per-class WCP (SA-29 secondary check):")
    for klass, label in [(1, "GTVp"), (2, "GTVn"), (3, "mixed")]:
        sub = hecktor[hecktor["lesion_class"] == klass]
        if len(sub) < 30:
            continue
        wcp_k = fit_wcp_classifier(cal[WCP_FEATURE_COLS], sub[WCP_FEATURE_COLS])
        q_hat_Wk = weighted_conformal_threshold(cal_scores, wcp_k.source_weights, alpha=ALPHA)
        idx = sub.index
        lo_c = lo_h[hecktor.index.isin(idx)]
        hi_c = hi_h[hecktor.index.isin(idx)]
        y_c = y(sub)
        row = _evaluate_with_qhat(
            "hecktor", "wcp", "lesion_class", label, lo_c, hi_c, y_c, q_hat_Wk,
        )
        wcp_rows.append(row)
        print(f"    class {label}  n={len(sub):>3}  q_hat_W={q_hat_Wk:.4f}  cov={row.coverage:.4f}  miss={row.miss_pp:+.2f}pp")

    # === Mode 2b: WCP-extended (Amendment 12 §12a primary) ===
    print("\n--- Mode 2b: Weighted Conformal Prediction with extended features (Amendment 12) ---")
    wcp_ext_rows: list[CoverageRow] = []
    wcp_ext_models: dict[str, dict] = {}
    cal_extended = build_extended_wcp_features(cal)
    for cohort_name, cohort_df, lo_c, hi_c in [
        ("autopet_iii", autopet_iii, lo_t3, hi_t3),
        ("hecktor",     hecktor,     lo_h,  hi_h),
    ]:
        cohort_extended = build_extended_wcp_features(cohort_df)
        print(f"  Fitting WCP-extended classifier (AutoPET-I cal -> {cohort_name})...")
        wcp_e = fit_wcp_classifier(cal_extended, cohort_extended, feature_cols=WCP_EXTENDED_FEATURE_COLS)
        diag = diagnose_support_overlap(wcp_e.classifier_auc, wcp_e.source_weights)
        q_hat_We = weighted_conformal_threshold(cal_scores, wcp_e.source_weights, alpha=ALPHA)
        print(f"    AUC = {wcp_e.classifier_auc:.4f}  weight range [{wcp_e.weight_clip_low:.2e}, {wcp_e.weight_clip_high:.2e}]")
        print(f"    support diagnostic = {diag.verdict.upper()}  ESS/n = {diag.ess_ratio:.4f}  dispersion = {diag.weight_dispersion:.2f}")
        if diag.flagged_reasons:
            for r in diag.flagged_reasons:
                print(f"      flagged: {r}")
        print(f"    q_hat_W (extended) = {q_hat_We:.4f}  (vs naive {q_hat_naive:.4f}; ratio {q_hat_We / q_hat_naive:.2f})")
        row = _evaluate_with_qhat(
            cohort_name, "wcp_extended", "marginal", "all", lo_c, hi_c, y(cohort_df), q_hat_We,
        )
        wcp_ext_rows.append(row)
        print(f"    cov={row.coverage:.4f}  miss={row.miss_pp:+.2f}pp")
        wcp_ext_models[cohort_name] = {
            "classifier_auc": wcp_e.classifier_auc,
            "weight_clip_low": wcp_e.weight_clip_low,
            "weight_clip_high": wcp_e.weight_clip_high,
            "weight_median": float(np.median(wcp_e.source_weights)),
            "weight_dispersion": diag.weight_dispersion,
            "ess_ratio": diag.ess_ratio,
            "support_verdict": diag.verdict,
            "flagged_reasons": list(diag.flagged_reasons),
            "q_hat_W_extended": q_hat_We,
            "feature_means": {k: list(v) for k, v in wcp_e.feature_means.items()},
        }

    # Per-feature classifier coefficients (SA-34: which axes drive the shift)
    print("\n  SA-34 per-feature LR coefficients (positive = pushes target prob up):")
    for cohort_name in ("autopet_iii", "hecktor"):
        cohort_df = autopet_iii if cohort_name == "autopet_iii" else hecktor
        cohort_extended = build_extended_wcp_features(cohort_df)
        wcp_e2 = fit_wcp_classifier(cal_extended, cohort_extended, feature_cols=WCP_EXTENDED_FEATURE_COLS)
        coefs = dict(zip(WCP_EXTENDED_FEATURE_COLS, wcp_e2.classifier.coef_[0]))
        sorted_c = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"    target={cohort_name}: top 5 coefs by |value|:")
        for fname, coef in sorted_c[:5]:
            print(f"      {fname:<35} {coef:+.4f}")

    # === Mode 3: Per-cohort recalibration (Amendment 11 §11d secondary) ===
    print("\n--- Mode 3: Per-cohort split-conformal recalibration (Amendment 11 SA-31) ---")
    percohort_rows: list[CoverageRow] = []
    percohort_meta: dict[str, dict] = {}
    for cohort_name, cohort_df, lo_c, hi_c in [
        ("autopet_iii", autopet_iii, lo_t3, hi_t3),
        ("hecktor",     hecktor,     lo_h,  hi_h),
    ]:
        ho_df, ev_df = patient_level_holdout_split(cohort_df)
        ho_idx = cohort_df.index.isin(ho_df.index)
        ev_idx = cohort_df.index.isin(ev_df.index)
        # Recalibrate on holdout
        cqr_local = calibrate_cqr(
            y(ho_df), lo_c[ho_idx], hi_c[ho_idx],
            alpha=ALPHA, stratum_name=f"{cohort_name}_holdout",
        )
        q_hat_local = cqr_local.q_hat
        print(f"  {cohort_name}: holdout n={len(ho_df)} patients={ho_df['case_id'].nunique()} "
              f"q_hat_local={q_hat_local:.4f}  eval n={len(ev_df)}")
        row = _evaluate_with_qhat(
            cohort_name, "percohort_recal", "marginal", "all",
            lo_c[ev_idx], hi_c[ev_idx], y(ev_df), q_hat_local,
        )
        percohort_rows.append(row)
        print(f"    cov={row.coverage:.4f}  miss={row.miss_pp:+.2f}pp")
        percohort_meta[cohort_name] = {
            "q_hat_local": q_hat_local,
            "n_holdout_patients": int(ho_df["case_id"].nunique()),
            "n_holdout_lesions": int(len(ho_df)),
            "n_eval_lesions": int(len(ev_df)),
        }

    # === CSI Table (Amendment 11 §11e) ===
    print("\n--- CSI Table (Amendment 11 §11e) ---")
    csi_rows = []
    for cohort_name, cohort_df, lo_c, hi_c in [
        ("autopet_i_test", test,        lo_t1, hi_t1),
        ("autopet_iii",    autopet_iii, lo_t3, hi_t3),
        ("hecktor",        hecktor,     lo_h,  hi_h),
    ]:
        cqr_self = calibrate_cqr(y(cohort_df), lo_c, hi_c, alpha=ALPHA, stratum_name=cohort_name)
        csi = calibration_shift_index(cqr_self.q_hat, q_hat_naive)
        recal_recommended = csi > 1.5
        csi_rows.append({
            "cohort": cohort_name,
            "n_lesions": int(len(cohort_df)),
            "q_hat_autopet_i_cal": q_hat_naive,
            "q_hat_self": cqr_self.q_hat,
            "csi": csi,
            "recalibration_recommended": recal_recommended,
        })
    csi_df = pd.DataFrame(csi_rows)
    print(csi_df.to_string(index=False))

    # === Side-by-side comparison ===
    all_rows = naive_rows + wcp_rows + wcp_ext_rows + percohort_rows
    summary = pd.DataFrame([asdict(r) for r in all_rows])

    print("\n" + "=" * 78)
    print("Side-by-side cohort marginal coverage comparison (Phase 3 Amendment 11)")
    print("=" * 78)
    pivot = (
        summary[summary["stratum_kind"] == "marginal"]
        .pivot_table(index="cohort", columns="mode",
                     values=["coverage", "miss_pp"], aggfunc="first")
    )
    print(pivot.round(4).to_string())

    # Hypothesis verdicts
    print("\n--- Amendment 11 + 12 hypothesis verdicts ---")
    h13_rows = [r for r in wcp_rows if r.stratum_kind == "marginal" and r.cohort != "autopet_i"]
    h14_rows = [r for r in wcp_rows if r.cohort == "hecktor" and r.stratum_kind == "centre"]
    h15_rows = wcp_ext_rows  # Amendment 12 primary
    sa31_rows = [r for r in percohort_rows if r.stratum_kind == "marginal"]
    h13_pass = all(abs(r.miss_pp) <= TOL_MARGINAL_PP * 100 for r in h13_rows)
    h14_max_miss = max(abs(r.miss_pp) for r in h14_rows) if h14_rows else 0
    h14_pass = h14_max_miss <= TOL_CONDITIONAL_PP * 100
    h15_pass = all(abs(r.miss_pp) <= TOL_MARGINAL_PP * 100 for r in h15_rows)
    sa31_pass = all(abs(r.miss_pp) <= TOL_MARGINAL_PP * 100 for r in sa31_rows)

    # H15 conditional on support-overlap diagnostic per Amendment 12 §12d
    h15_red = any(wcp_ext_models[c]["support_verdict"] == "red" for c in wcp_ext_models)
    h15_status = (
        "NOT_TESTED (support violation; Red diagnostic)" if h15_red
        else ("PASS" if h15_pass else "FAIL")
    )

    verdicts = pd.DataFrame([
        {"hypothesis": "H13", "scope": "WCP-image marginal coverage on external cohorts (±2pp)",
         "verdict": "PASS" if h13_pass else "FAIL",
         "max_abs_miss_pp": max(abs(r.miss_pp) for r in h13_rows)},
        {"hypothesis": "H14", "scope": "WCP-image per-centre coverage on HECKTOR (±5pp; n>=30)",
         "verdict": "PASS" if h14_pass else "FAIL",
         "max_abs_miss_pp": h14_max_miss},
        {"hypothesis": "H15", "scope": "WCP-extended marginal coverage on external cohorts (±2pp; conditional on Green/Amber support diagnostic)",
         "verdict": h15_status,
         "max_abs_miss_pp": max(abs(r.miss_pp) for r in h15_rows) if h15_rows else float("nan")},
        {"hypothesis": "SA-31", "scope": "Per-cohort split recalibration marginal coverage (±2pp)",
         "verdict": "PASS" if sa31_pass else "FAIL",
         "max_abs_miss_pp": max(abs(r.miss_pp) for r in sa31_rows)},
    ])
    print(verdicts.to_string(index=False))

    # === Outputs ===
    out = {
        "target": target, "alpha": ALPHA,
        "src_shas": src_shas,
        "summary": summary, "csi": csi_df, "verdicts": verdicts,
        "wcp_models": wcp_models, "wcp_ext_models": wcp_ext_models,
        "percohort_meta": percohort_meta,
    }

    if commit:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(OUT_DIR / f"phase3_amendment_11_coverage_{target}.parquet", index=False)
        csi_df.to_csv(OUT_DIR / f"phase3_amendment_11_csi_{target}.csv", index=False)
        verdicts.to_csv(OUT_DIR / f"phase3_amendment_11_verdicts_{target}.csv", index=False)
        with open(OUT_DIR / f"phase3_amendment_11_metadata_{target}.json", "w") as f:
            json.dump({
                "target": target, "alpha": ALPHA, "src_shas": src_shas,
                "wcp_models": wcp_models, "wcp_ext_models": wcp_ext_models,
                "percohort_meta": percohort_meta,
            }, f, indent=2, default=str)
        print(f"\nWrote outputs to {OUT_DIR}")
    else:
        print(f"\n[--dry-run mode: no artefacts written]")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="suvmax", choices=("suvmax", "suvpeak", "suvmean"))
    ap.add_argument("--commit", action="store_true",
                    help="Write Amendment 11 locked artefacts. Without this, dry-run only.")
    args = ap.parse_args()
    run_amendment_11(args.target, args.commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
