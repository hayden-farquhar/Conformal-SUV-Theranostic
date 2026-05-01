"""§3.9 validation kappa analysis -- AutoPET-III (Amendment 5).

Loads the filled-in blinded sample CSV + the key CSV, computes Cohen's
kappa with bootstrap 95% CI between reviewer decisions and the index test,
applies the pre-specified action rule, and writes a markdown report.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (Amendment 5)

Usage
-----
    python scripts/section_3_9_validation_analyse.py [--blinded PATH] [--key PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Pre-registered constants (Amendment 5)
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 4242
INDEX_TEST_RATIO_THRESHOLD = 0.4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "results/tables/section_3_9_validation"
DEFAULT_BLINDED = DEFAULT_DIR / "sample_blinded.csv"
DEFAULT_KEY = DEFAULT_DIR / "sample_key.csv"
DEFAULT_OUT = DEFAULT_DIR / "kappa_report.md"


def bootstrap_kappa_ci(
    y1: np.ndarray, y2: np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
):
    """Bootstrap 95% CI for Cohen's kappa via case-level resampling."""
    rng = np.random.RandomState(seed)
    n = len(y1)
    kappas = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            k = cohen_kappa_score(y1[idx], y2[idx])
            if not np.isnan(k):
                kappas.append(k)
        except ValueError:
            # Some bootstrap samples may be degenerate (all one class)
            continue
    kappas = np.array(kappas)
    ci_lower = float(np.percentile(kappas, 2.5))
    ci_upper = float(np.percentile(kappas, 97.5))
    return ci_lower, ci_upper, kappas


def grid_search_optimal_threshold(reviewer: np.ndarray, ratios: np.ndarray) -> tuple:
    """Find the SUVpeak/SUVmax ratio threshold that maximises kappa against reviewer.

    Returns (best_threshold, best_kappa).
    """
    candidates = np.linspace(0.20, 0.70, 51)
    best_t, best_k = 0.4, -1.0
    for t in candidates:
        pred = np.where(ratios >= t, "retain", "exclude")
        try:
            k = cohen_kappa_score(reviewer, pred)
            if not np.isnan(k) and k > best_k:
                best_k = k
                best_t = float(t)
        except ValueError:
            continue
    return best_t, best_k


def apply_action_rule(kappa_lower: float) -> tuple[str, str]:
    """Pre-specified action rule from Amendment 5."""
    if kappa_lower >= 0.75:
        return "SUBSTANTIAL_AGREEMENT", (
            "Apply the SUVpeak/SUVmax >= 0.4 retain rule to the remaining 199 review-needed "
            "lesions. Document kappa + CI in manuscript methods. The 50 reviewed lesions retain "
            "the reviewer's decision (not the index test's)."
        )
    if kappa_lower >= 0.60:
        return "MODERATE_AGREEMENT", (
            "Refine the ratio threshold via grid search on the 50-lesion validation sample. "
            "Draw a fresh 30-lesion sample (seed=43, same stratification) and re-validate at "
            "the optimised threshold. Apply the optimised rule only if the fresh sample also "
            "achieves kappa_lower_95CI >= 0.75; otherwise revert to full image review."
        )
    return "INSUFFICIENT_AGREEMENT", (
        "Revert to full image review for the remaining 199 review-needed lesions per the "
        "original §3.9 protocol. The ratio-based protocol is not a defensible substitute "
        "for this cohort."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blinded", default=str(DEFAULT_BLINDED),
                        help="Filled-in blinded sample CSV (with reviewer_decision)")
    parser.add_argument("--key", default=str(DEFAULT_KEY), help="Sample key CSV")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output markdown report path")
    args = parser.parse_args()

    blinded = pd.read_csv(args.blinded)
    key = pd.read_csv(args.key)

    # Validation
    if "reviewer_decision" not in blinded.columns:
        print(f"ERROR: blinded CSV missing 'reviewer_decision' column", file=sys.stderr)
        return 1
    blinded["reviewer_decision"] = blinded["reviewer_decision"].astype(str).str.strip().str.lower()
    valid_decisions = {"retain", "exclude"}
    bad = blinded[~blinded["reviewer_decision"].isin(valid_decisions)]
    if len(bad) > 0:
        print(f"ERROR: {len(bad)} rows have invalid reviewer_decision values "
              f"(must be 'retain' or 'exclude'):", file=sys.stderr)
        print(bad[["review_id", "reviewer_decision"]].to_string(index=False), file=sys.stderr)
        return 1

    merged = blinded.merge(key, on="review_id", how="inner", validate="one_to_one")
    if len(merged) != len(blinded):
        print(f"ERROR: review_id mismatch between blinded ({len(blinded)}) "
              f"and key ({len(key)}); merged {len(merged)}", file=sys.stderr)
        return 1

    print(f"Loaded {len(merged)} reviewed lesions")

    # --- Primary analysis ---
    reviewer = merged["reviewer_decision"].values
    index_test = merged["index_predicted_decision"].values

    kappa_point = cohen_kappa_score(reviewer, index_test)
    ci_low, ci_high, _ = bootstrap_kappa_ci(reviewer, index_test)

    pct_agreement = float((reviewer == index_test).sum() / len(merged))
    cm = confusion_matrix(reviewer, index_test, labels=["retain", "exclude"])

    decision, action_text = apply_action_rule(ci_low)

    # --- Per-category (secondary, diagnostic only) ---
    per_cat = []
    for cat in sorted(merged["triage_category"].unique()):
        sub = merged[merged["triage_category"] == cat]
        if len(sub) < 5:
            continue
        try:
            k = cohen_kappa_score(sub["reviewer_decision"], sub["index_predicted_decision"])
            agree = float((sub["reviewer_decision"] == sub["index_predicted_decision"]).sum() / len(sub))
            per_cat.append({
                "stratum": cat,
                "n": len(sub),
                "kappa": float(k) if not np.isnan(k) else None,
                "pct_agreement": agree,
            })
        except ValueError:
            per_cat.append({"stratum": cat, "n": len(sub), "kappa": None, "pct_agreement": None})
    per_cat_df = pd.DataFrame(per_cat)

    # --- Threshold grid search (informational; only matters if MODERATE_AGREEMENT) ---
    opt_t, opt_k = grid_search_optimal_threshold(reviewer, merged["suv_peakratio"].values)

    # --- Console summary ---
    print(f"\n--- Primary analysis ---")
    print(f"  Cohen's kappa (point):  {kappa_point:.4f}")
    print(f"  95% bootstrap CI:        [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Percent agreement:       {pct_agreement:.4f}  ({int((reviewer == index_test).sum())}/{len(merged)})")
    print(f"\n  Confusion matrix (rows = reviewer, cols = index test):")
    print(f"             retain  exclude")
    print(f"   retain    {cm[0,0]:>6d}  {cm[0,1]:>6d}")
    print(f"   exclude   {cm[1,0]:>6d}  {cm[1,1]:>6d}")
    print(f"\n  Per-stratum kappa (diagnostic):")
    for r in per_cat:
        k = r["kappa"]
        k_str = f"{k:.3f}" if k is not None else "n/a"
        a = r["pct_agreement"]
        a_str = f"{a:.3f}" if a is not None else "n/a"
        print(f"    {r['stratum']:20s}: n={r['n']:2d}, kappa={k_str}, agreement={a_str}")
    print(f"\n  Optimal ratio threshold via grid search: {opt_t:.3f} (kappa={opt_k:.3f})")
    print(f"\n--- ACTION (per pre-registered rule, Amendment 5) ---")
    print(f"  {decision}")
    print(f"  {action_text}")

    # --- Markdown report ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("# §3.9 Validation Kappa Report -- AutoPET-III\n")
    report.append("**Amendment 5** (https://doi.org/10.17605/OSF.IO/4KAZN)  \n")
    report.append(f"**Reviewer:** Hayden Farquhar (project investigator, blinded review protocol)  \n")
    report.append(f"**Sample size:** {len(merged)} lesions  \n")
    report.append(f"**Random seed (sample draw):** 42  \n")
    report.append(f"**Index test:** retain if SUVpeak/SUVmax >= {INDEX_TEST_RATIO_THRESHOLD}  \n\n")

    report.append("## Primary analysis\n\n")
    report.append("| Metric | Value |\n|---|---|\n")
    report.append(f"| Cohen's kappa (point estimate) | {kappa_point:.4f} |\n")
    report.append(f"| 95% bootstrap CI | [{ci_low:.4f}, {ci_high:.4f}] |\n")
    report.append(f"| Percent agreement | {pct_agreement:.4f} ({int((reviewer == index_test).sum())}/{len(merged)}) |\n")
    report.append(f"| Bootstrap resamples | {N_BOOTSTRAP} |\n")
    report.append(f"| Bootstrap seed | {BOOTSTRAP_SEED} |\n\n")

    report.append("## Confusion matrix\n\n")
    report.append("|  | index: retain | index: exclude |\n|---|---|---|\n")
    report.append(f"| **reviewer: retain** | {cm[0,0]} | {cm[0,1]} |\n")
    report.append(f"| **reviewer: exclude** | {cm[1,0]} | {cm[1,1]} |\n\n")

    report.append("## Per-stratum (diagnostic)\n\n")
    report.append("| Stratum | n | Cohen's kappa | Percent agreement |\n|---|---|---|---|\n")
    for r in per_cat:
        k_str = f"{r['kappa']:.3f}" if r["kappa"] is not None else "n/a"
        a_str = f"{r['pct_agreement']:.3f}" if r["pct_agreement"] is not None else "n/a"
        report.append(f"| {r['stratum']} | {r['n']} | {k_str} | {a_str} |\n")
    report.append("\n")

    report.append("## Threshold grid search (informational)\n\n")
    report.append(f"- Optimal SUVpeak/SUVmax threshold (maximises kappa against reviewer): "
                  f"**{opt_t:.3f}**  \n")
    report.append(f"- Kappa at optimal threshold: **{opt_k:.4f}**  \n")
    report.append(f"- Pre-specified threshold (Amendment 5): {INDEX_TEST_RATIO_THRESHOLD}  \n\n")

    report.append("## Decision (per Amendment 5 pre-specified action rule)\n\n")
    report.append(f"**{decision}** (kappa_lower_95CI = {ci_low:.4f})\n\n")
    report.append(f"{action_text}\n\n")

    report.append("## Reproducibility\n\n")
    report.append(f"- Blinded sample: `{args.blinded}`  \n")
    report.append(f"- Key: `{args.key}`  \n")
    report.append(f"- This report: `{args.out}`  \n")
    import hashlib
    blinded_sha = hashlib.sha256(Path(args.blinded).read_bytes()).hexdigest()
    key_sha = hashlib.sha256(Path(args.key).read_bytes()).hexdigest()
    report.append(f"- Blinded SHA-256: `{blinded_sha}`  \n")
    report.append(f"- Key SHA-256: `{key_sha}`  \n")

    out_path.write_text("".join(report))
    print(f"\nMarkdown report written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
