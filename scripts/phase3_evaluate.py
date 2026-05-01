"""Phase 3 cross-cohort conformal coverage evaluation (driver scaffold).

Trains + calibrates the CQR ONCE on the locked AutoPET-I cal split, then
evaluates marginal + Mondrian conditional coverage across three cohorts:

    AutoPET-I test  (held-out from the same data-generating process)
    AutoPET-III     (external: multi-vendor PSMA, whole-body)
    HECKTOR         (external: multi-centre FDG H&N)

Hypotheses tested (per pre-reg + Amendments 8 + 9):

    H1  AutoPET-III marginal coverage within ±2pp of nominal
    H2  AutoPET-III conditional Mondrian coverage within ±5pp of nominal on
        cells with n>=30 lesions; AutoPET-I smallest-quartile drop ≥5pp
        without Mondrian; Mondrian recovers
    H11 HECKTOR marginal coverage within ±2pp of nominal
    H12 HECKTOR per-centre conditional coverage within ±5pp of nominal
        (cells with n>=30; smaller cells pooled into "other")
    SA-29 HECKTOR GTVp-only and GTVn-only coverage within ±3pp each

(H3 is a separate analysis -- VISION/PERCIST threshold-straddling rate --
handled by `src/clinical/vision_overlay.py` and reported alongside.)

Locked decisions (Amendments 6, 7, 8, 9):

    Feature set (F1):  6 features  -- volume_ml, centroid_0/1/2, softmax_mean,
                                       softmax_entropy. Sentinel softmax
                                       (1.0, 0.0) for GT-seg cohorts (AutoPET-I,
                                       HECKTOR). Native softmax for AutoPET-III.

    Mondrian (M2):     per-cohort quartile boundaries on the test side; q_hat
                       per stratum calibrated once on AutoPET-I cal.

    wCV reference:     Amendment 7 AutoPET-III primary wCV parquet
                       (OSF SHA 619e740a..., 2178 lesion-doses).

Pre-reg gate -- this driver runs in two modes:

    --dry-run     loads data, trains, calibrates, evaluates, but does NOT
                  write the locked Phase 3 artefacts. Used to verify the
                  pipeline before the freeze.

    --commit      writes the locked Phase 3 artefacts. Once committed the
                  test-set evaluation is final per pre-reg discipline.

The driver records the source SHA-256 of every input artefact and the SHA-256
of the driver itself in the output, so the Phase 3 freeze is reproducible.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN
Amendments: https://osf.io/j5ry4/  (amendment_log.md v9 SHA 2ad1c920...)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
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
from src.conformal.mondrian import assign_volume_quartiles


# ====== LOCKED CONFIG (changes require an Amendment) ======
ALPHA = 0.10                         # 90% nominal coverage (pre-reg §5.2.3)
NOMINAL = 1 - ALPHA
TARGET_PRIMARY = "suvmax"            # primary outcome (pre-reg §4.1)
TARGETS_SECONDARY = ("suvpeak", "suvmean")
RANDOM_SEED = 42

# Feature set: pre-reg §4.2 5-feature (Amendment 9 §9b sentinel for GT cohorts)
# Centroids removed: pre-reg §4.2 lists surface_area_cm2 + sphericity as geometric
# predictors, NOT centroid coordinates (which are voxel coords with cohort-dependent
# meaning). The earlier scaffold's use of centroids was a mistake inherited from the
# AutoPET-I smoke test (which was an infrastructure check, not a freeze).
FEATURE_COLS = ["volume_ml", "surface_area_cm2", "sphericity",
                "softmax_mean", "softmax_entropy"]
SENTINEL_SOFTMAX_MEAN = 1.0
SENTINEL_SOFTMAX_ENTROPY = 0.0

# Mondrian config (Amendment 9 §9a: per-cohort boundaries on test side)
MIN_CELL_SIZE = 30                   # n>=30 floor for conditional coverage cells
MIN_CELL_SIZE_AUTOPET_I_CAL = 30     # for calibration-side stratum minimum

# Pre-reg coverage tolerances
TOL_MARGINAL_PP = 0.02               # H1, H11
TOL_CONDITIONAL_PP = 0.05            # H2, H12
TOL_CLASS_PP = 0.03                  # SA-29 GTVp/GTVn

# Default paths (locked artefacts)
PATHS = {
    "autopet_i_lesions":       PROJECT_ROOT / "data/interim/lesion_tables/autopet_i_lesions.parquet",
    "autopet_i_splits":        PROJECT_ROOT / "data/processed/autopet_i_splits.parquet",
    "autopet_iii_lesions":     PROJECT_ROOT / "data/interim/lesion_tables/autopet_iii_lesions_reviewed.parquet",
    "hecktor_lesions":         PROJECT_ROOT / "data/interim/lesion_tables/hecktor_lesions_reviewed.parquet",
    "phase2_wcv":              PROJECT_ROOT / "data/processed/phase2_autopet_iii_primary_wcv.parquet",
    "out_dir":                 PROJECT_ROOT / "results/phase3",
}


# ====== LOAD HELPERS ======

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
    """Apply §3.9 exclusion filter if the column exists."""
    for col in ("section_3_9_excluded", "excluded"):
        if col in df.columns:
            n_before = len(df)
            df = df[~df[col].astype(bool)].copy()
            n_after = len(df)
            if n_before != n_after:
                print(f"  §3.9 filter ({col}): {n_before} -> {n_after}")
    return df


def _add_sentinel_softmax(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per Amendment 9 §9b: add or overwrite softmax columns with sentinel values
    for GT-segmentation cohorts. AutoPET-III is left untouched."""
    df = df.copy()
    if "softmax_mean" not in df.columns:
        df["softmax_mean"] = SENTINEL_SOFTMAX_MEAN
        print(f"  [{label}] softmax_mean missing -- substituting sentinel {SENTINEL_SOFTMAX_MEAN}")
    if "softmax_entropy" not in df.columns:
        df["softmax_entropy"] = SENTINEL_SOFTMAX_ENTROPY
        print(f"  [{label}] softmax_entropy missing -- substituting sentinel {SENTINEL_SOFTMAX_ENTROPY}")
    return df


def load_autopet_i() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train, cal, test) AutoPET-I lesion tables, with §3.9 + sentinel applied."""
    print("Loading AutoPET-I...")
    lesions = pd.read_parquet(PATHS["autopet_i_lesions"])
    splits = pd.read_parquet(PATHS["autopet_i_splits"])
    lesions = _drop_excluded(lesions)

    # Merge split assignment by patient_id == case_id
    splits_renamed = splits[["patient_id", "split"]].rename(columns={"patient_id": "case_id"})
    lesions = lesions.merge(splits_renamed, on="case_id", how="left")
    if lesions["split"].isna().any():
        raise RuntimeError(f"{lesions['split'].isna().sum()} lesions have no split assignment")

    lesions = _add_sentinel_softmax(lesions, "autopet_i")
    train = lesions[lesions["split"] == "train"].reset_index(drop=True)
    cal   = lesions[lesions["split"] == "calibration"].reset_index(drop=True)
    test  = lesions[lesions["split"] == "test"].reset_index(drop=True)
    print(f"  train={len(train)}  cal={len(cal)}  test={len(test)}")
    return train, cal, test


def load_autopet_iii() -> pd.DataFrame:
    print("Loading AutoPET-III (external test)...")
    lesions = pd.read_parquet(PATHS["autopet_iii_lesions"])
    lesions = _drop_excluded(lesions)
    # AutoPET-III has native softmax_mean / softmax_entropy from nnU-Net inference.
    if "softmax_mean" not in lesions.columns or "softmax_entropy" not in lesions.columns:
        raise RuntimeError(
            "AutoPET-III parquet missing native softmax columns; expected from §4.2 schema"
        )
    print(f"  n_lesions={len(lesions)}  n_patients={lesions['case_id'].nunique()}")
    return lesions


def load_hecktor() -> pd.DataFrame:
    print("Loading HECKTOR (external test)...")
    lesions = pd.read_parquet(PATHS["hecktor_lesions"])
    lesions = _drop_excluded(lesions)
    lesions = _add_sentinel_softmax(lesions, "hecktor")
    print(f"  n_lesions={len(lesions)}  n_patients={lesions['case_id'].nunique()}")
    return lesions


def transform_target(y_raw: np.ndarray) -> np.ndarray:
    """Pre-reg §5.2.3: log1p transform of SUV target."""
    return np.log1p(np.asarray(y_raw, dtype=np.float64))


# ====== MONDRIAN APPLICATION (Amendment 9 §9a) ======

@dataclass
class MondrianModel:
    """Container for one calibrated Mondrian model.

    q_hat values keyed by stratum label. Strata not present in calibration
    fall back to marginal q_hat with a logged warning.
    """
    q_hat_marginal: float
    q_hat_per_stratum: dict[str, float]
    n_per_stratum: dict[str, int]


def build_mondrian_calibration(
    cal_df: pd.DataFrame,
    lo_cal: np.ndarray,
    hi_cal: np.ndarray,
    y_cal: np.ndarray,
    quartile_boundaries: np.ndarray,
) -> MondrianModel:
    """Calibrate Mondrian on AutoPET-I cal using volume quartile only (tracer +
    vendor are constant for AutoPET-I, so the Mondrian collapses to volume_q)."""
    # Marginal
    cqr = calibrate_cqr(y_cal, lo_cal, hi_cal, alpha=ALPHA, stratum_name="all")
    q_hat_marginal = cqr.q_hat
    n_marginal = cqr.n_calibration

    # Per stratum (Q1/Q2/Q3/Q4) using AutoPET-I cal-derived boundaries
    quartiles = _assign_volume_quartile_with_boundaries(
        cal_df["volume_ml"].to_numpy(), quartile_boundaries
    )
    q_hat_per_stratum = {}
    n_per_stratum = {}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        mask = quartiles == q
        n = int(mask.sum())
        n_per_stratum[q] = n
        if n < MIN_CELL_SIZE_AUTOPET_I_CAL:
            print(f"  Mondrian Q{q}: n={n} < {MIN_CELL_SIZE_AUTOPET_I_CAL} -- using marginal q_hat")
            q_hat_per_stratum[q] = q_hat_marginal
            continue
        cqr_q = calibrate_cqr(
            y_cal[mask], lo_cal[mask], hi_cal[mask],
            alpha=ALPHA, stratum_name=q,
        )
        q_hat_per_stratum[q] = cqr_q.q_hat
    return MondrianModel(
        q_hat_marginal=q_hat_marginal,
        q_hat_per_stratum=q_hat_per_stratum,
        n_per_stratum={"all": n_marginal, **n_per_stratum},
    )


def _compute_quartile_boundaries(volumes: np.ndarray) -> np.ndarray:
    """Return [25th, 50th, 75th] percentile of volumes (Q1|Q2|Q3|Q4 cuts)."""
    return np.percentile(np.asarray(volumes, dtype=np.float64), [25, 50, 75])


def _assign_volume_quartile_with_boundaries(
    volumes: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Assign Q1/Q2/Q3/Q4 labels using explicit boundary array [p25, p50, p75].

    Per Amendment 9 §9a: the test-side cohorts use their OWN boundaries; the
    calibration side uses AutoPET-I cal boundaries. This function does NOT
    enforce which boundaries are passed -- the caller controls the boundary
    source.
    """
    v = np.asarray(volumes, dtype=np.float64)
    p25, p50, p75 = boundaries
    out = np.full(v.shape, "Q4", dtype=object)
    out[v < p75] = "Q3"
    out[v < p50] = "Q2"
    out[v < p25] = "Q1"
    return out


# ====== COVERAGE EVALUATION ======

@dataclass
class CoverageRow:
    cohort: str
    stratum_kind: str          # 'marginal' | 'mondrian_volume' | 'centre' | 'lesion_class'
    stratum_label: str
    n_lesions: int
    coverage: float
    ci_lower: float
    ci_upper: float
    miss_pp: float             # signed: empirical - nominal, in percentage points
    median_width_log: float    # in log-SUV units
    nominal_in_ci: bool


def evaluate_cohort_marginal(
    cohort: str,
    df: pd.DataFrame,
    lo: np.ndarray,
    hi: np.ndarray,
    y_log: np.ndarray,
    q_hat: float,
) -> CoverageRow:
    lower, upper, widths = predict_intervals_array(lo, hi, q_hat)
    cov = compute_coverage(y_log, lower, upper, nominal=NOMINAL, label=f"{cohort}_marginal")
    return CoverageRow(
        cohort=cohort, stratum_kind="marginal", stratum_label="all",
        n_lesions=int(len(y_log)),
        coverage=float(cov.coverage),
        ci_lower=float(cov.ci_lower), ci_upper=float(cov.ci_upper),
        miss_pp=float((cov.coverage - NOMINAL) * 100),
        median_width_log=float(np.median(widths)),
        nominal_in_ci=bool(cov.ci_lower <= NOMINAL <= cov.ci_upper),
    )


def evaluate_cohort_mondrian(
    cohort: str,
    df: pd.DataFrame,
    lo: np.ndarray,
    hi: np.ndarray,
    y_log: np.ndarray,
    mondrian: MondrianModel,
    boundary_source: str,   # 'autopet_i_cal' | 'self'
    cal_boundaries: np.ndarray,
) -> list[CoverageRow]:
    """M2 (Amendment 9 §9a): per-cohort boundaries on the test side by default.

    `boundary_source='autopet_i_cal'` is also supported for the AutoPET-I test
    split (where boundaries naturally come from the cal split of the same
    cohort) and for diagnostic comparisons.
    """
    if boundary_source == "self":
        boundaries = _compute_quartile_boundaries(df["volume_ml"].to_numpy())
    else:
        boundaries = cal_boundaries
    quartiles = _assign_volume_quartile_with_boundaries(
        df["volume_ml"].to_numpy(), boundaries
    )
    rows = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        mask = quartiles == q
        n = int(mask.sum())
        if n < MIN_CELL_SIZE:
            rows.append(CoverageRow(
                cohort=cohort, stratum_kind="mondrian_volume", stratum_label=q,
                n_lesions=n, coverage=float("nan"),
                ci_lower=float("nan"), ci_upper=float("nan"),
                miss_pp=float("nan"), median_width_log=float("nan"),
                nominal_in_ci=False,
            ))
            continue
        q_hat = mondrian.q_hat_per_stratum.get(q, mondrian.q_hat_marginal)
        lower, upper, widths = predict_intervals_array(lo[mask], hi[mask], q_hat)
        cov = compute_coverage(y_log[mask], lower, upper,
                               nominal=NOMINAL, label=f"{cohort}_{q}")
        rows.append(CoverageRow(
            cohort=cohort, stratum_kind="mondrian_volume", stratum_label=q,
            n_lesions=n,
            coverage=float(cov.coverage),
            ci_lower=float(cov.ci_lower), ci_upper=float(cov.ci_upper),
            miss_pp=float((cov.coverage - NOMINAL) * 100),
            median_width_log=float(np.median(widths)),
            nominal_in_ci=bool(cov.ci_lower <= NOMINAL <= cov.ci_upper),
        ))
    return rows


def evaluate_hecktor_per_centre(
    df: pd.DataFrame,
    lo: np.ndarray,
    hi: np.ndarray,
    y_log: np.ndarray,
    q_hat_marginal: float,
) -> list[CoverageRow]:
    """H12: per-centre conditional coverage on HECKTOR. Cells with n<MIN_CELL_SIZE
    are pooled into 'other' rather than dropped (Amendment 8 §8d)."""
    centres = df["centre_id"].to_numpy()
    rows = []
    other_mask = np.zeros(len(df), dtype=bool)
    centre_counts = pd.Series(centres).value_counts()
    for cid, n in centre_counts.items():
        mask = centres == cid
        if n < MIN_CELL_SIZE:
            other_mask |= mask
            continue
        lower, upper, widths = predict_intervals_array(lo[mask], hi[mask], q_hat_marginal)
        cov = compute_coverage(y_log[mask], lower, upper,
                               nominal=NOMINAL, label=f"hecktor_centre_{cid}")
        rows.append(CoverageRow(
            cohort="hecktor", stratum_kind="centre", stratum_label=str(int(cid)),
            n_lesions=int(n),
            coverage=float(cov.coverage),
            ci_lower=float(cov.ci_lower), ci_upper=float(cov.ci_upper),
            miss_pp=float((cov.coverage - NOMINAL) * 100),
            median_width_log=float(np.median(widths)),
            nominal_in_ci=bool(cov.ci_lower <= NOMINAL <= cov.ci_upper),
        ))
    if other_mask.any():
        n = int(other_mask.sum())
        lower, upper, widths = predict_intervals_array(
            lo[other_mask], hi[other_mask], q_hat_marginal
        )
        cov = compute_coverage(y_log[other_mask], lower, upper,
                               nominal=NOMINAL, label="hecktor_centre_other")
        rows.append(CoverageRow(
            cohort="hecktor", stratum_kind="centre", stratum_label="other_pooled",
            n_lesions=n,
            coverage=float(cov.coverage),
            ci_lower=float(cov.ci_lower), ci_upper=float(cov.ci_upper),
            miss_pp=float((cov.coverage - NOMINAL) * 100),
            median_width_log=float(np.median(widths)),
            nominal_in_ci=bool(cov.ci_lower <= NOMINAL <= cov.ci_upper),
        ))
    return rows


def evaluate_hecktor_per_class(
    df: pd.DataFrame,
    lo: np.ndarray,
    hi: np.ndarray,
    y_log: np.ndarray,
    q_hat_marginal: float,
) -> list[CoverageRow]:
    """SA-29: GTVp-only and GTVn-only marginal coverage."""
    rows = []
    for klass, label in [(1, "GTVp"), (2, "GTVn"), (3, "mixed")]:
        mask = (df["lesion_class"] == klass).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        lower, upper, widths = predict_intervals_array(lo[mask], hi[mask], q_hat_marginal)
        cov = compute_coverage(y_log[mask], lower, upper,
                               nominal=NOMINAL, label=f"hecktor_class_{label}")
        rows.append(CoverageRow(
            cohort="hecktor", stratum_kind="lesion_class", stratum_label=label,
            n_lesions=n,
            coverage=float(cov.coverage),
            ci_lower=float(cov.ci_lower), ci_upper=float(cov.ci_upper),
            miss_pp=float((cov.coverage - NOMINAL) * 100),
            median_width_log=float(np.median(widths)),
            nominal_in_ci=bool(cov.ci_lower <= NOMINAL <= cov.ci_upper),
        ))
    return rows


# ====== HYPOTHESIS-MAPPING REPORT ======

def build_hypothesis_table(rows: list[CoverageRow]) -> pd.DataFrame:
    """Map locked rows to H1/H2/H11/H12/SA-29 verdicts."""
    by_key = {(r.cohort, r.stratum_kind, r.stratum_label): r for r in rows}
    records = []

    # H1: AutoPET-III marginal
    if ("autopet_iii", "marginal", "all") in by_key:
        r = by_key[("autopet_iii", "marginal", "all")]
        records.append({
            "hypothesis": "H1", "scope": "AutoPET-III marginal coverage",
            "tolerance_pp": TOL_MARGINAL_PP * 100,
            "n_lesions": r.n_lesions, "coverage": r.coverage,
            "miss_pp": r.miss_pp,
            "verdict": "PASS" if abs(r.miss_pp) <= TOL_MARGINAL_PP * 100 else "FAIL",
        })

    # H2 (partial): AutoPET-I smallest-quartile drop without Mondrian + recovery with Mondrian
    # (Full H2 evaluation requires marginal-q_hat-only baseline as well; reported separately.)

    # H11: HECKTOR marginal
    if ("hecktor", "marginal", "all") in by_key:
        r = by_key[("hecktor", "marginal", "all")]
        records.append({
            "hypothesis": "H11", "scope": "HECKTOR marginal coverage",
            "tolerance_pp": TOL_MARGINAL_PP * 100,
            "n_lesions": r.n_lesions, "coverage": r.coverage,
            "miss_pp": r.miss_pp,
            "verdict": "PASS" if abs(r.miss_pp) <= TOL_MARGINAL_PP * 100 else "FAIL",
        })

    # H12: HECKTOR per-centre conditional
    centre_rows = [r for r in rows if r.cohort == "hecktor" and r.stratum_kind == "centre"]
    centre_misses = [abs(r.miss_pp) for r in centre_rows if not np.isnan(r.miss_pp)]
    if centre_misses:
        max_miss = max(centre_misses)
        records.append({
            "hypothesis": "H12",
            "scope": f"HECKTOR per-centre conditional coverage (max abs miss across {len(centre_rows)} cells)",
            "tolerance_pp": TOL_CONDITIONAL_PP * 100,
            "n_lesions": sum(r.n_lesions for r in centre_rows),
            "coverage": float("nan"),  # multi-cell, no single coverage value
            "miss_pp": max_miss,
            "verdict": "PASS" if max_miss <= TOL_CONDITIONAL_PP * 100 else "FAIL",
        })

    # SA-29: HECKTOR GTVp + GTVn
    for label in ("GTVp", "GTVn"):
        key = ("hecktor", "lesion_class", label)
        if key in by_key:
            r = by_key[key]
            records.append({
                "hypothesis": f"SA-29 ({label})",
                "scope": f"HECKTOR {label}-only marginal coverage",
                "tolerance_pp": TOL_CLASS_PP * 100,
                "n_lesions": r.n_lesions, "coverage": r.coverage,
                "miss_pp": r.miss_pp,
                "verdict": "PASS" if abs(r.miss_pp) <= TOL_CLASS_PP * 100 else "FAIL",
            })

    return pd.DataFrame(records)


# ====== DRIVER ======

def run_phase3(target: str, commit: bool) -> dict:
    """Full Phase 3 evaluation for one target stat."""
    print("=" * 78)
    print(f"Phase 3 evaluation -- target = {target}, alpha = {ALPHA}, commit = {commit}")
    print("=" * 78)

    # Provenance: SHA the inputs we'll touch
    src_shas = {k: _sha256(p) for k, p in PATHS.items()
                if p.suffix in (".parquet", ".json", ".csv") and p.exists()}
    src_shas["driver_script"] = _sha256(Path(__file__))
    print()
    print("Input artefact SHA-256:")
    for k, v in src_shas.items():
        print(f"  {k:<25} {v[:16]}...")
    print()

    # Load
    train, cal, test = load_autopet_i()
    autopet_iii = load_autopet_iii()
    hecktor = load_hecktor()

    # Feature matrices (Amendment 9 §9b: 6-feature with sentinel for GT cohorts)
    def X(df): return df[FEATURE_COLS].to_numpy()
    def y(df): return transform_target(df[target].to_numpy())

    # Train QR on AutoPET-I train split
    print(f"\nTraining LightGBM quantile regressors (lower/upper at alpha={ALPHA})...")
    lower_model, upper_model = train_quantile_regressors(X(train), y(train), alpha=ALPHA)

    # Predict raw quantiles on cal + each test cohort
    lo_cal,  hi_cal  = lower_model.predict(X(cal)),  upper_model.predict(X(cal))
    lo_t1,   hi_t1   = lower_model.predict(X(test)),  upper_model.predict(X(test))
    lo_t3,   hi_t3   = lower_model.predict(X(autopet_iii)), upper_model.predict(X(autopet_iii))
    lo_h,    hi_h    = lower_model.predict(X(hecktor)), upper_model.predict(X(hecktor))

    # AutoPET-I cal-derived quartile boundaries (locked)
    cal_boundaries = _compute_quartile_boundaries(cal["volume_ml"].to_numpy())
    print(f"\nAutoPET-I cal volume quartile boundaries (mL): "
          f"p25={cal_boundaries[0]:.2f}  p50={cal_boundaries[1]:.2f}  p75={cal_boundaries[2]:.2f}")

    # Calibrate Mondrian
    mondrian = build_mondrian_calibration(
        cal_df=cal, lo_cal=lo_cal, hi_cal=hi_cal, y_cal=y(cal),
        quartile_boundaries=cal_boundaries,
    )
    print(f"\nMondrian calibration:")
    print(f"  q_hat_marginal = {mondrian.q_hat_marginal:.4f}")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        n = mondrian.n_per_stratum[q]
        qh = mondrian.q_hat_per_stratum[q]
        print(f"  q_hat_{q} = {qh:.4f}  (n_cal={n})")

    # Evaluate per cohort
    rows: list[CoverageRow] = []

    # AutoPET-I test
    print(f"\n--- AutoPET-I test split ---")
    rows.append(evaluate_cohort_marginal("autopet_i", test, lo_t1, hi_t1, y(test),
                                         q_hat=mondrian.q_hat_marginal))
    rows.extend(evaluate_cohort_mondrian(
        "autopet_i", test, lo_t1, hi_t1, y(test),
        mondrian=mondrian,
        boundary_source="self",   # M2: AutoPET-I test has its own boundaries
        cal_boundaries=cal_boundaries,
    ))

    # AutoPET-III
    print(f"\n--- AutoPET-III (external test) ---")
    rows.append(evaluate_cohort_marginal("autopet_iii", autopet_iii, lo_t3, hi_t3, y(autopet_iii),
                                         q_hat=mondrian.q_hat_marginal))
    rows.extend(evaluate_cohort_mondrian(
        "autopet_iii", autopet_iii, lo_t3, hi_t3, y(autopet_iii),
        mondrian=mondrian, boundary_source="self", cal_boundaries=cal_boundaries,
    ))

    # HECKTOR
    print(f"\n--- HECKTOR (external test) ---")
    rows.append(evaluate_cohort_marginal("hecktor", hecktor, lo_h, hi_h, y(hecktor),
                                         q_hat=mondrian.q_hat_marginal))
    rows.extend(evaluate_cohort_mondrian(
        "hecktor", hecktor, lo_h, hi_h, y(hecktor),
        mondrian=mondrian, boundary_source="self", cal_boundaries=cal_boundaries,
    ))
    rows.extend(evaluate_hecktor_per_centre(
        hecktor, lo_h, hi_h, y(hecktor), q_hat_marginal=mondrian.q_hat_marginal,
    ))
    rows.extend(evaluate_hecktor_per_class(
        hecktor, lo_h, hi_h, y(hecktor), q_hat_marginal=mondrian.q_hat_marginal,
    ))

    # Print summary
    summary = pd.DataFrame([r.__dict__ for r in rows])
    print(f"\n=== Coverage summary ({target}) ===")
    print(summary.to_string(index=False))

    hypotheses = build_hypothesis_table(rows)
    print(f"\n=== Hypothesis verdicts ({target}) ===")
    print(hypotheses.to_string(index=False))

    out = {
        "target": target,
        "alpha": ALPHA,
        "src_shas": src_shas,
        "mondrian": {
            "q_hat_marginal": mondrian.q_hat_marginal,
            "q_hat_per_stratum": mondrian.q_hat_per_stratum,
            "n_per_stratum": mondrian.n_per_stratum,
            "cal_boundaries_ml": cal_boundaries.tolist(),
        },
        "summary": summary,
        "hypotheses": hypotheses,
    }

    if commit:
        outdir = PATHS["out_dir"]
        outdir.mkdir(parents=True, exist_ok=True)
        summary_path = outdir / f"phase3_coverage_summary_{target}.parquet"
        hyp_path = outdir / f"phase3_hypothesis_verdicts_{target}.csv"
        meta_path = outdir / f"phase3_metadata_{target}.json"
        summary.to_parquet(summary_path, index=False)
        hypotheses.to_csv(hyp_path, index=False)
        with open(meta_path, "w") as f:
            json.dump({
                "target": target, "alpha": ALPHA,
                "src_shas": src_shas,
                "mondrian": {
                    "q_hat_marginal": mondrian.q_hat_marginal,
                    "q_hat_per_stratum": mondrian.q_hat_per_stratum,
                    "n_per_stratum": mondrian.n_per_stratum,
                    "cal_boundaries_ml": cal_boundaries.tolist(),
                },
            }, f, indent=2, default=str)
        print(f"\nWrote {summary_path}")
        print(f"Wrote {hyp_path}")
        print(f"Wrote {meta_path}")
    else:
        print(f"\n[--dry-run mode: no artefacts written]")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=("suvmax", "suvpeak", "suvmean", "all"),
                    default="suvmax",
                    help="Target SUV stat (primary=suvmax)")
    ap.add_argument("--commit", action="store_true",
                    help="Commit the Phase 3 freeze (write locked artefacts). Without "
                         "this flag the driver runs in --dry-run mode.")
    args = ap.parse_args()

    targets = (TARGET_PRIMARY,) + TARGETS_SECONDARY if args.target == "all" else (args.target,)
    for t in targets:
        run_phase3(target=t, commit=args.commit)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
