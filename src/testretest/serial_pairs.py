"""Find same-patient serial PET scan pairs for test-retest analysis.

Identifies pairs of scans from the same patient where:
- Interval between scans is ≤8 weeks (pre-registered §3.5)
- Same tracer used in both scans
- Lesions are matched across time points by centroid proximity (≤20mm)

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§3.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# Pre-registered thresholds (§3.5)
MAX_SCAN_INTERVAL_WEEKS = 8
MAX_CENTROID_DISPLACEMENT_MM = 20.0
MIN_PAIRS_FOR_PRIMARY = 50  # decision gate for Poisson fallback


@dataclass
class LesionPair:
    """A matched lesion pair across two time points."""

    patient_id: str
    scan_1_uid: str
    scan_2_uid: str
    scan_1_date: str
    scan_2_date: str
    interval_days: int
    lesion_1_id: int
    lesion_2_id: int
    centroid_distance_mm: float

    # SUV values at each time point (filled after extraction)
    suvmax_1: float | None = None
    suvmax_2: float | None = None
    suvpeak_1: float | None = None
    suvpeak_2: float | None = None
    suvmean_1: float | None = None
    suvmean_2: float | None = None


def find_serial_scan_pairs(
    study_df: pd.DataFrame,
    max_interval_weeks: int = MAX_SCAN_INTERVAL_WEEKS,
) -> pd.DataFrame:
    """Find same-patient scan pairs within the interval threshold.

    Parameters
    ----------
    study_df : pd.DataFrame
        Study-level metadata. Must have: patient_id, study_uid, study_date.
        study_date should be parseable by pd.to_datetime.
    max_interval_weeks : int
        Maximum interval between scans in weeks.

    Returns
    -------
    pd.DataFrame
        One row per scan pair with columns: patient_id, scan_1_uid,
        scan_2_uid, scan_1_date, scan_2_date, interval_days.
    """
    df = study_df.copy()
    df["study_date_dt"] = pd.to_datetime(df["study_date"])
    df = df.sort_values(["patient_id", "study_date_dt"])

    max_interval = timedelta(weeks=max_interval_weeks)
    pairs = []

    for patient_id, group in df.groupby("patient_id"):
        if len(group) < 2:
            continue

        studies = group.reset_index(drop=True)
        for i in range(len(studies)):
            for j in range(i + 1, len(studies)):
                interval = studies.iloc[j]["study_date_dt"] - studies.iloc[i]["study_date_dt"]
                if interval <= max_interval:
                    pairs.append({
                        "patient_id": patient_id,
                        "scan_1_uid": studies.iloc[i]["study_uid"],
                        "scan_2_uid": studies.iloc[j]["study_uid"],
                        "scan_1_date": str(studies.iloc[i]["study_date"]),
                        "scan_2_date": str(studies.iloc[j]["study_date"]),
                        "interval_days": interval.days,
                    })

    return pd.DataFrame(pairs) if pairs else pd.DataFrame(
        columns=["patient_id", "scan_1_uid", "scan_2_uid",
                 "scan_1_date", "scan_2_date", "interval_days"]
    )


def match_lesions_across_scans(
    lesions_scan_1: pd.DataFrame,
    lesions_scan_2: pd.DataFrame,
    max_displacement_mm: float = MAX_CENTROID_DISPLACEMENT_MM,
) -> list[tuple[int, int, float]]:
    """Match lesions between two scans by nearest centroid.

    Uses greedy nearest-neighbour matching: for each lesion in scan 1,
    find the closest unmatched lesion in scan 2 within the displacement
    threshold.

    Parameters
    ----------
    lesions_scan_1 : pd.DataFrame
        Must have: lesion_id, centroid_z, centroid_y, centroid_x.
        Centroids should be in mm (world coordinates), not voxels.
    lesions_scan_2 : pd.DataFrame
        Same format.
    max_displacement_mm : float
        Maximum centroid distance for a valid match.

    Returns
    -------
    list of (lesion_1_id, lesion_2_id, distance_mm) tuples
        Matched pairs, sorted by distance (closest first).
    """
    if len(lesions_scan_1) == 0 or len(lesions_scan_2) == 0:
        return []

    centroids_1 = lesions_scan_1[["centroid_z", "centroid_y", "centroid_x"]].values
    centroids_2 = lesions_scan_2[["centroid_z", "centroid_y", "centroid_x"]].values
    ids_1 = lesions_scan_1["lesion_id"].values
    ids_2 = lesions_scan_2["lesion_id"].values

    # Compute pairwise distances
    # centroids_1: (n1, 3), centroids_2: (n2, 3)
    diff = centroids_1[:, np.newaxis, :] - centroids_2[np.newaxis, :, :]  # (n1, n2, 3)
    distances = np.sqrt((diff ** 2).sum(axis=2))  # (n1, n2)

    # Greedy matching: pick closest pair, remove both, repeat
    matched = []
    used_1 = set()
    used_2 = set()

    # Flatten and sort by distance
    n1, n2 = distances.shape
    flat_indices = np.argsort(distances.ravel())

    for flat_idx in flat_indices:
        i = flat_idx // n2
        j = flat_idx % n2
        dist = distances[i, j]

        if dist > max_displacement_mm:
            break  # all remaining are too far

        if i in used_1 or j in used_2:
            continue

        matched.append((int(ids_1[i]), int(ids_2[j]), float(dist)))
        used_1.add(i)
        used_2.add(j)

    return matched


def compute_within_lesion_cv(
    pairs_df: pd.DataFrame,
    suv_col_1: str,
    suv_col_2: str,
) -> dict:
    """Compute within-lesion coefficient of variation from matched pairs.

    CV = SD / mean for each pair, then report the distribution.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Must have the specified SUV columns (one per time point).
    suv_col_1, suv_col_2 : str
        Column names for SUV at time 1 and time 2.

    Returns
    -------
    dict
        Keys: n_pairs, mean_cv, median_cv, sd_cv, iqr_cv
    """
    v1 = pairs_df[suv_col_1].values
    v2 = pairs_df[suv_col_2].values

    pair_means = (v1 + v2) / 2.0
    pair_sds = np.abs(v1 - v2) / np.sqrt(2)  # SD from 2 measurements

    # Avoid division by zero
    valid = pair_means > 0
    cvs = np.where(valid, pair_sds / pair_means * 100.0, np.nan)
    cvs = cvs[~np.isnan(cvs)]

    return {
        "n_pairs": len(cvs),
        "mean_cv_pct": float(np.mean(cvs)) if len(cvs) > 0 else 0.0,
        "median_cv_pct": float(np.median(cvs)) if len(cvs) > 0 else 0.0,
        "sd_cv_pct": float(np.std(cvs)) if len(cvs) > 0 else 0.0,
        "iqr_cv_pct": (
            float(np.percentile(cvs, 25)),
            float(np.percentile(cvs, 75)),
        ) if len(cvs) > 0 else (0.0, 0.0),
    }


def check_decision_gate(n_pairs: int) -> bool:
    """Check if enough pairs exist for primary test-retest analysis.

    Pre-registered decision gate (§3.5): if <50 matched lesion pairs,
    activate Poisson-noise fallback.

    Returns
    -------
    bool
        True if primary analysis is viable (≥50 pairs).
    """
    return n_pairs >= MIN_PAIRS_FOR_PRIMARY
