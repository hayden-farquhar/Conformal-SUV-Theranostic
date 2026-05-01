"""Weighted Conformal Prediction (WCP) under covariate shift (Amendment 11).

Implements the weighted split-conformal threshold from Tibshirani et al. 2019
("Conformal Prediction Under Covariate Shift", NeurIPS 2019). Importance
weights w(x) = p_target(x) / p_source(x) are estimated via a discriminative
classifier per Amendment 11 sec 11c.

Pre-specified design (Amendment 11):
    1. Combine source (AutoPET-I cal) + target (test cohort) lesions, label them.
    2. Fit L2-regularised logistic regression on a 3-feature subset
       (volume_ml, surface_area_cm2, sphericity).
    3. For each source point i: predicted prob p_i -> w_i = p_i / (1 - p_i).
    4. Clip weights to 1st-99th percentile to control variance from extreme weights.
    5. Apply weights in the WCP threshold formula:
           q_hat_W = inf{q : sum_i w_i * I(s_i <= q) / sum_i w_i >= 1 - alpha}

The 3-feature subset is the union of §4.2 features that have actual variance
in both AutoPET-I and external cohorts -- softmax_mean and softmax_entropy
are sentinel constants for AutoPET-I and HECKTOR (Amendments 8/9) and would
contribute zero discrimination.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN sec 5.2.3
Amendment 11: 2026-04-30 (osf/amendment_log.md sec 11b-c)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Pre-reg / Amendment 11 locked feature subset for importance-weight estimation
WCP_FEATURE_COLS = ["volume_ml", "surface_area_cm2", "sphericity"]

# Amendment 12 §12a extended feature set: continuous + categorical cohort-defining
WCP_EXTENDED_CONTINUOUS_COLS = [
    "volume_ml",
    "surface_area_cm2",
    "sphericity",
    "voxel_volume_ml",
    "lesions_per_patient_in_cohort",
]
WCP_EXTENDED_CATEGORICAL_COLS = [
    "tracer_is_psma",
    "vendor_is_siemens", "vendor_is_ge", "vendor_is_philips",
    "centre_1", "centre_2", "centre_3", "centre_5", "centre_6", "centre_7", "centre_8",
]
WCP_EXTENDED_FEATURE_COLS = WCP_EXTENDED_CONTINUOUS_COLS + WCP_EXTENDED_CATEGORICAL_COLS

# Amendment 11 §11c: weight clipping percentile range (variance control)
WCP_WEIGHT_CLIP_LOW = 1.0
WCP_WEIGHT_CLIP_HIGH = 99.0

# Amendment 12 §12b: support-overlap diagnostic thresholds
WCP_SUPPORT_AUC_AMBER = 0.95
WCP_SUPPORT_AUC_RED = 0.99
WCP_WEIGHT_DISPERSION_AMBER = 100.0
WCP_ESS_RATIO_AMBER = 0.30

# Logistic regression hyperparameters (Amendment 11 §11c)
LR_C = 1.0
LR_MAX_ITER = 1000


@dataclass
class WCPModel:
    """A fitted weighted-conformal model for one source -> target pair."""
    classifier: LogisticRegression
    feature_cols: list[str]
    source_weights: np.ndarray            # w_i for each source (cal) point
    weight_clip_low: float                # the actual clipping value (1st pct)
    weight_clip_high: float               # the actual clipping value (99th pct)
    n_source: int
    n_target: int
    classifier_auc: float                 # ROC-AUC of source-vs-target on train set
    feature_means: dict[str, tuple[float, float]]  # per-feature (source_mean, target_mean)


def fit_wcp_classifier(
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
    feature_cols: list[str] = WCP_FEATURE_COLS,
    *,
    seed: int = 42,
) -> WCPModel:
    """Estimate per-source importance weights via a discriminative classifier.

    Parameters
    ----------
    source_features : pd.DataFrame
        Feature rows from the source distribution (AutoPET-I cal).
    target_features : pd.DataFrame
        Feature rows from the target distribution (one external cohort).
    feature_cols : list[str]
        Pre-reg-locked 3-feature subset (Amendment 11 sec 11c).
    seed : int
        Random seed for the classifier (sklearn LogisticRegression doesn't
        actually use one for liblinear, but kept for documentation).

    Returns
    -------
    WCPModel
        Fitted classifier + per-source weights (already clipped).
    """
    if not all(c in source_features.columns for c in feature_cols):
        raise ValueError(f"source_features missing one of {feature_cols}")
    if not all(c in target_features.columns for c in feature_cols):
        raise ValueError(f"target_features missing one of {feature_cols}")
    n_s = int(len(source_features))
    n_t = int(len(target_features))
    if n_s == 0 or n_t == 0:
        raise ValueError(f"empty source ({n_s}) or target ({n_t}) cohort")

    X_s = source_features[feature_cols].to_numpy(dtype=np.float64)
    X_t = target_features[feature_cols].to_numpy(dtype=np.float64)
    X = np.vstack([X_s, X_t])
    y = np.concatenate([np.zeros(n_s, dtype=int), np.ones(n_t, dtype=int)])

    # L2 regularisation: future sklearn (>=1.10) deprecates penalty='l2' in
    # favour of l1_ratio=0 with elasticnet-aware solvers. lbfgs only supports L2,
    # so the default penalty (None means no regularisation; we explicitly want L2).
    clf = LogisticRegression(
        C=LR_C, max_iter=LR_MAX_ITER, solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X, y)

    # Predict on source points only; w_i = p_i / (1 - p_i)
    p_source = clf.predict_proba(X_s)[:, 1]
    # Avoid division by 0 / numerical issues
    p_source = np.clip(p_source, 1e-12, 1.0 - 1e-12)
    raw_weights = p_source / (1.0 - p_source)

    # Clip to 1st-99th percentile (Amendment 11 §11c; variance control)
    clip_low = float(np.percentile(raw_weights, WCP_WEIGHT_CLIP_LOW))
    clip_high = float(np.percentile(raw_weights, WCP_WEIGHT_CLIP_HIGH))
    weights = np.clip(raw_weights, clip_low, clip_high)

    # Classifier AUC on the same data (informational; reported in CSI table)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

    feature_means = {
        c: (float(source_features[c].mean()), float(target_features[c].mean()))
        for c in feature_cols
    }

    return WCPModel(
        classifier=clf,
        feature_cols=list(feature_cols),
        source_weights=weights,
        weight_clip_low=clip_low,
        weight_clip_high=clip_high,
        n_source=n_s,
        n_target=n_t,
        classifier_auc=auc,
        feature_means=feature_means,
    )


def weighted_conformal_threshold(
    nonconformity_scores: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Compute the weighted split-conformal threshold q_hat_W.

    Per Tibshirani et al. 2019:
        q_hat_W = inf{q : sum_i w_i * I(s_i <= q) / sum_i w_i >= 1 - alpha}

    Equivalently: the (1 - alpha) weighted quantile of the source nonconformity
    scores, where each source point i contributes w_i to the weighted distribution.

    Parameters
    ----------
    nonconformity_scores : np.ndarray, shape (n_source,)
        CQR nonconformity scores on the calibration set (e.g., max(lo - y, y - hi)).
    weights : np.ndarray, shape (n_source,)
        Importance weights from `fit_wcp_classifier`.
    alpha : float
        Miscoverage rate (alpha=0.10 -> 90% coverage).

    Returns
    -------
    float
        The weighted (1 - alpha) quantile of the score distribution.
    """
    if len(nonconformity_scores) != len(weights):
        raise ValueError(
            f"nonconformity_scores ({len(nonconformity_scores)}) and weights "
            f"({len(weights)}) must have same length"
        )
    if len(nonconformity_scores) == 0:
        raise ValueError("empty nonconformity_scores")
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    # Sort scores ascending; reorder weights to match
    order = np.argsort(nonconformity_scores)
    s_sorted = nonconformity_scores[order]
    w_sorted = weights[order]

    # Cumulative weighted distribution
    total_w = float(np.sum(w_sorted))
    if total_w <= 0:
        raise ValueError("sum of weights must be > 0")
    cum_w = np.cumsum(w_sorted) / total_w

    # First score where cumulative weight >= 1 - alpha
    target = 1.0 - alpha
    idx = np.searchsorted(cum_w, target, side="left")
    idx = min(idx, len(s_sorted) - 1)
    return float(s_sorted[idx])


def calibration_shift_index(
    q_hat_self: float,
    q_hat_source: float,
) -> float:
    """Calibration Shift Index (CSI) -- Amendment 11 sec 11e.

    CSI = q_hat_cohort_self / q_hat_AutoPET_I_cal

    CSI ~ 1 means no recalibration needed. CSI > 1.5 triggers the pre-registered
    "recalibration recommended" flag (Amendment 11 sec 11e clinical-deployment guideline).
    """
    if q_hat_source <= 0:
        return float("inf")
    return float(q_hat_self / q_hat_source)


# ===== Amendment 11 §11d: per-cohort split-conformal recalibration =====

PERCOHORT_HOLDOUT_FRACTION = 0.20         # Amendment 11 §11d locked
PERCOHORT_HOLDOUT_SEED = 42                # Amendment 11 §11d locked


def patient_level_holdout_split(
    df: pd.DataFrame,
    holdout_fraction: float = PERCOHORT_HOLDOUT_FRACTION,
    seed: int = PERCOHORT_HOLDOUT_SEED,
    patient_col: str = "case_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level random split for per-cohort recalibration (SA-31).

    Per Amendment 11 §11d: one-stage cluster random sample by patient_id with
    seed=42, no further stratification (external cohorts are pure test cohorts;
    no cal-vs-test allocation history to preserve).

    Returns
    -------
    (holdout_df, eval_df)
        holdout_df = the 20% used for cohort-specific q_hat calibration
        eval_df    = the 80% on which marginal coverage is evaluated
    """
    if patient_col not in df.columns:
        raise ValueError(f"df missing {patient_col!r} column")
    rng = np.random.RandomState(seed)
    patients = df[patient_col].unique()
    rng.shuffle(patients)
    n_holdout = int(round(len(patients) * holdout_fraction))
    if n_holdout == 0 or n_holdout == len(patients):
        raise ValueError(
            f"holdout_fraction {holdout_fraction} produces degenerate split "
            f"({n_holdout} of {len(patients)} patients)"
        )
    holdout_patients = set(patients[:n_holdout])
    holdout = df[df[patient_col].isin(holdout_patients)].copy()
    evalset = df[~df[patient_col].isin(holdout_patients)].copy()
    return holdout, evalset


# ===== Amendment 12: Extended feature builder + support-overlap diagnostic =====


def build_extended_wcp_features(lesion_df: pd.DataFrame) -> pd.DataFrame:
    """Construct the Amendment 12 §12a 16-feature extended subset for any cohort.

    Reads lesion-table columns (locked schema across cohorts post-Amendment 10)
    and constructs the continuous + categorical features for WCP-extended.

    Required input columns:
        volume_ml, surface_area_cm2, sphericity (Amendment 11 §11c)
        voxel_spacing_0/1/2 (computed -> voxel_volume_ml)
        case_id (computed -> lesions_per_patient_in_cohort)
        tracer (computed -> tracer_is_psma)
        vendor (computed -> vendor_is_siemens / _ge / _philips)
        centre_id (computed -> centre_<id>; absent for AutoPET-I/III which set 0)

    Returns
    -------
    pd.DataFrame
        16 columns matching `WCP_EXTENDED_FEATURE_COLS`. All numeric;
        categorical features are 0/1 indicators.
    """
    df = lesion_df.copy()
    out = pd.DataFrame(index=df.index)

    # Continuous geometric (Amendment 11 §11c, retained)
    out["volume_ml"] = df["volume_ml"].astype(float)
    out["surface_area_cm2"] = df["surface_area_cm2"].astype(float)
    out["sphericity"] = df["sphericity"].astype(float)

    # voxel_volume_ml from voxel_spacing
    if all(c in df.columns for c in ("voxel_spacing_0", "voxel_spacing_1", "voxel_spacing_2")):
        out["voxel_volume_ml"] = (
            df["voxel_spacing_0"].astype(float)
            * df["voxel_spacing_1"].astype(float)
            * df["voxel_spacing_2"].astype(float)
            / 1000.0
        )
    else:
        out["voxel_volume_ml"] = 0.0  # missing; LR with L2 will downweight

    # lesions_per_patient_in_cohort: median count of lesions per patient,
    # broadcast to every lesion (cohort-level burden indicator).
    if "case_id" in df.columns:
        per_patient = df.groupby("case_id").size()
        median_per_patient = float(per_patient.median())
        out["lesions_per_patient_in_cohort"] = median_per_patient
    else:
        out["lesions_per_patient_in_cohort"] = 0.0

    # Categorical: tracer (FDG vs PSMA)
    tracer_col = df.get("tracer", pd.Series([""] * len(df), index=df.index))
    out["tracer_is_psma"] = (tracer_col.astype(str).str.upper() == "PSMA").astype(int)

    # Categorical: vendor one-hot
    vendor_col = df.get("vendor", pd.Series([""] * len(df), index=df.index)).astype(str)
    out["vendor_is_siemens"] = (vendor_col.str.lower() == "siemens").astype(int)
    out["vendor_is_ge"]      = (vendor_col.str.lower() == "ge").astype(int)
    out["vendor_is_philips"] = (vendor_col.str.lower() == "philips").astype(int)

    # Categorical: HECKTOR centre indicators (AutoPET-I/III have no centre_id -> all 0)
    if "centre_id" in df.columns:
        centre_int = pd.to_numeric(df["centre_id"], errors="coerce").fillna(0).astype(int)
        for cid in (1, 2, 3, 5, 6, 7, 8):
            out[f"centre_{cid}"] = (centre_int == cid).astype(int)
    else:
        for cid in (1, 2, 3, 5, 6, 7, 8):
            out[f"centre_{cid}"] = 0

    # Sanity: all expected columns present in expected order
    out = out[WCP_EXTENDED_FEATURE_COLS]
    return out


@dataclass
class SupportOverlapDiagnostic:
    """Amendment 12 §12b support-overlap verdict for one (source, target) pair."""
    classifier_auc: float
    weight_dispersion: float        # max(w) / min(w)
    effective_sample_size: float    # (sum w)^2 / sum(w^2)
    ess_ratio: float                # ESS / n_source
    verdict: str                    # 'green' | 'amber' | 'red'
    flagged_reasons: list[str]


def diagnose_support_overlap(
    classifier_auc: float,
    weights: np.ndarray,
) -> SupportOverlapDiagnostic:
    """Produce the Amendment 12 §12b support-overlap diagnostic.

    Verdict thresholds (locked Amendment 12):
      - Red:    AUC > 0.99 (near-deterministic source-vs-target separation)
      - Amber:  any single threshold breached (AUC > 0.95, dispersion > 100, ESS/n < 0.30)
      - Green:  none breached
    """
    n = int(len(weights))
    if n == 0:
        raise ValueError("empty weights")
    w_max = float(np.max(weights))
    w_min = float(np.min(weights))
    dispersion = w_max / w_min if w_min > 0 else float("inf")
    sum_w = float(np.sum(weights))
    sum_w_sq = float(np.sum(weights ** 2))
    ess = (sum_w ** 2) / sum_w_sq if sum_w_sq > 0 else 0.0
    ess_ratio = ess / n

    flagged = []
    if classifier_auc > WCP_SUPPORT_AUC_RED:
        verdict = "red"
        flagged.append(f"classifier_auc={classifier_auc:.4f} > {WCP_SUPPORT_AUC_RED}")
    else:
        if classifier_auc > WCP_SUPPORT_AUC_AMBER:
            flagged.append(f"classifier_auc={classifier_auc:.4f} > {WCP_SUPPORT_AUC_AMBER}")
        if dispersion > WCP_WEIGHT_DISPERSION_AMBER:
            flagged.append(f"weight_dispersion={dispersion:.2f} > {WCP_WEIGHT_DISPERSION_AMBER}")
        if ess_ratio < WCP_ESS_RATIO_AMBER:
            flagged.append(f"ess_ratio={ess_ratio:.4f} < {WCP_ESS_RATIO_AMBER}")
        verdict = "amber" if flagged else "green"

    return SupportOverlapDiagnostic(
        classifier_auc=classifier_auc,
        weight_dispersion=dispersion,
        effective_sample_size=ess,
        ess_ratio=ess_ratio,
        verdict=verdict,
        flagged_reasons=flagged,
    )
