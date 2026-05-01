"""Conformalized Quantile Regression (CQR).

Implements the CQR framework from Romano et al. (2019):
1. Train quantile regressors for lower and upper conditional quantiles
2. Compute nonconformity scores on calibration data
3. Apply finite-sample corrected quantile as the conformal adjustment

Supports Mondrian stratification for group-conditional coverage.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.2.3)

Reference:
    Romano, Patterson, Candès (2019). "Conformalized Quantile Regression."
    NeurIPS 2019.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CQRCalibration:
    """Stores calibration result for a single alpha level and stratum."""

    alpha: float
    stratum: str  # "all" for marginal, or stratum name for Mondrian
    q_hat: float  # conformal adjustment (quantile of nonconformity scores)
    n_calibration: int
    scores: np.ndarray  # raw nonconformity scores (for diagnostics)


@dataclass
class PredictionInterval:
    """A single prediction interval for one data point."""

    lower: float
    upper: float
    width: float
    point_estimate: float  # midpoint of the quantile regressor interval


def train_quantile_regressors(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    alpha: float = 0.10,
    **lgb_params,
) -> tuple:
    """Train LightGBM quantile regressors for lower and upper quantiles.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training features.
    y_train : array-like, shape (n_samples,)
        Training targets (SUV values).
    alpha : float
        Miscoverage level. Trains quantiles at alpha/2 and 1-alpha/2.
    **lgb_params
        Additional LightGBM parameters. Defaults from pre-registration §5.2.3.

    Returns
    -------
    tuple[lightgbm.LGBMRegressor, lightgbm.LGBMRegressor]
        (lower_model, upper_model)
    """
    import lightgbm as lgb

    defaults = {
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1,
    }
    defaults.update(lgb_params)

    lower_quantile = alpha / 2
    upper_quantile = 1.0 - alpha / 2

    lower_model = lgb.LGBMRegressor(
        objective="quantile", alpha=lower_quantile, **defaults
    )
    upper_model = lgb.LGBMRegressor(
        objective="quantile", alpha=upper_quantile, **defaults
    )

    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    return lower_model, upper_model


def compute_nonconformity_scores(
    y_cal: np.ndarray,
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
) -> np.ndarray:
    """Compute CQR nonconformity scores.

    E_i = max(Q̂(α/2)(x_i) - y_i, y_i - Q̂(1-α/2)(x_i))

    A negative score means the true value was inside the raw quantile
    interval; a positive score means it was outside.

    Parameters
    ----------
    y_cal : array, shape (n,)
        True values on calibration set.
    lower_pred : array, shape (n,)
        Lower quantile predictions on calibration set.
    upper_pred : array, shape (n,)
        Upper quantile predictions on calibration set.

    Returns
    -------
    np.ndarray, shape (n,)
        Nonconformity scores.
    """
    return np.maximum(lower_pred - y_cal, y_cal - upper_pred)


def calibrate_conformal_threshold(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """Compute the finite-sample corrected conformal quantile.

    q̂ = ⌈(n+1)(1-α)⌉/n quantile of the nonconformity scores.

    Parameters
    ----------
    scores : array, shape (n,)
        Nonconformity scores from calibration set.
    alpha : float
        Miscoverage level.

    Returns
    -------
    float
        The conformal adjustment q̂.
    """
    n = len(scores)
    # Finite-sample correction: ceil((n+1)(1-alpha)) / n
    quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
    # Clip to [0, 1] for safety
    quantile_level = min(quantile_level, 1.0)
    return float(np.quantile(scores, quantile_level))


def calibrate_cqr(
    y_cal: np.ndarray,
    lower_pred_cal: np.ndarray,
    upper_pred_cal: np.ndarray,
    alpha: float,
    stratum_name: str = "all",
) -> CQRCalibration:
    """Full CQR calibration: compute scores and threshold.

    Parameters
    ----------
    y_cal : array
        True values on calibration set.
    lower_pred_cal : array
        Lower quantile predictions on calibration set.
    upper_pred_cal : array
        Upper quantile predictions on calibration set.
    alpha : float
        Miscoverage level.
    stratum_name : str
        Label for this stratum (for Mondrian).

    Returns
    -------
    CQRCalibration
    """
    scores = compute_nonconformity_scores(y_cal, lower_pred_cal, upper_pred_cal)
    q_hat = calibrate_conformal_threshold(scores, alpha)

    return CQRCalibration(
        alpha=alpha,
        stratum=stratum_name,
        q_hat=q_hat,
        n_calibration=len(scores),
        scores=scores,
    )


def predict_intervals(
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    q_hat: float,
) -> list[PredictionInterval]:
    """Apply conformal adjustment to produce prediction intervals.

    Interval for x_new: [Q̂(α/2)(x) - q̂, Q̂(1-α/2)(x) + q̂]

    Parameters
    ----------
    lower_pred : array
        Lower quantile predictions on test data.
    upper_pred : array
        Upper quantile predictions on test data.
    q_hat : float
        Conformal adjustment from calibration.

    Returns
    -------
    list[PredictionInterval]
    """
    intervals = []
    for lo, hi in zip(lower_pred, upper_pred):
        adj_lo = lo - q_hat
        adj_hi = hi + q_hat
        intervals.append(PredictionInterval(
            lower=adj_lo,
            upper=adj_hi,
            width=adj_hi - adj_lo,
            point_estimate=(lo + hi) / 2.0,
        ))
    return intervals


def predict_intervals_array(
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised version of predict_intervals.

    Returns
    -------
    tuple[lower_bounds, upper_bounds, widths]
        Each np.ndarray of shape (n,).
    """
    lower_bounds = lower_pred - q_hat
    upper_bounds = upper_pred + q_hat
    widths = upper_bounds - lower_bounds
    return lower_bounds, upper_bounds, widths
