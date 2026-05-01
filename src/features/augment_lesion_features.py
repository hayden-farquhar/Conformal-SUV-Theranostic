"""Augment existing lesion parquet with surface_area_cm2 + sphericity.

Pre-reg §4.2 lists `surface_area_cm2` and `sphericity` as CQR predictors. The
production extraction notebooks for AutoPET-I and AutoPET-III bypassed
`src/features/extract_lesion_features.py` (which computes both correctly) and
hand-rolled per-row dicts that omitted these geometric features. This module
adds them back to existing parquets without redoing SUV/softmax extraction.

Design: per case, re-load the SEG mask, re-run scipy.ndimage.label on the
binary mask, match each parquet lesion row to a component by centroid
proximity (matches the AutoPET-I Step 9 pattern), then compute SA + sphericity
from the matched component.

Sanity checks (any failure aborts that row, never silently writes wrong data):
  - Centroid match must be within 1 voxel
  - Matched component voxel count must match parquet's `n_voxels` (within 5%)

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN sec 4.2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from src.features.extract_lesion_features import compute_sphericity, compute_surface_area

# Matching tolerances
CENTROID_MATCH_TOL_VOX = 1.0       # max centroid distance in voxels
N_VOXELS_RELATIVE_TOL = 0.05       # max fractional difference in voxel count

# Pre-reg §3.2 minimum lesion volume
MIN_LESION_VOLUME_ML = 1.0


@dataclass
class AugmentationResult:
    """Per-case augmentation outcome."""
    case_id: str
    n_lesions_in: int               # rows in the input parquet for this case
    n_lesions_matched: int          # rows where SA + sphericity were filled
    n_lesions_unmatched: int        # rows that failed sanity checks
    error: str | None = None        # set on case-level errors (SEG missing, shape mismatch)


def augment_case(
    case_id: str,
    case_lesions: pd.DataFrame,
    seg: np.ndarray,
    voxel_spacing_zyx: tuple[float, float, float],
    min_volume_ml: float = MIN_LESION_VOLUME_ML,
) -> tuple[pd.DataFrame, AugmentationResult]:
    """Augment all parquet rows for one case with SA + sphericity.

    Parameters
    ----------
    case_id : str
        Patient/case identifier.
    case_lesions : pd.DataFrame
        Existing parquet rows for this case. Must include columns:
        case_id, lesion_id, n_voxels, volume_ml, centroid_0/1/2.
    seg : np.ndarray
        3D integer SEG mask (binary or multi-class) on the SUV grid.
    voxel_spacing_zyx : (sz, sy, sx) in mm
        Voxel spacing for the SEG/SUV grid.

    Returns
    -------
    augmented_df : pd.DataFrame
        Same rows as input, with `surface_area_cm2` and `sphericity` columns
        populated for matched lesions. Unmatched lesions get NaN for both
        new columns.
    result : AugmentationResult
    """
    if seg.ndim != 3:
        raise ValueError(f"seg must be 3D, got shape {seg.shape}")
    binary = (seg > 0).astype(np.int32)
    labelled, n_comp = ndimage.label(binary)

    voxel_volume_ml = float(np.prod(voxel_spacing_zyx)) / 1000.0

    # Cache per-component centroid + voxel count for fast matching
    if n_comp > 0:
        comp_ids = np.arange(1, n_comp + 1)
        comp_centroids = np.array(
            ndimage.center_of_mass(binary, labelled, list(comp_ids))
        )
        comp_sizes = np.array(
            ndimage.sum(binary, labelled, list(comp_ids)), dtype=np.int64
        )
        comp_volumes = comp_sizes * voxel_volume_ml
        # Filter to components above min_volume_ml (matches pre-reg §3.2)
        valid_mask = comp_volumes >= min_volume_ml
        valid_ids = comp_ids[valid_mask]
        valid_centroids = comp_centroids[valid_mask]
        valid_sizes = comp_sizes[valid_mask]
    else:
        valid_ids = np.array([], dtype=int)
        valid_centroids = np.zeros((0, 3))
        valid_sizes = np.array([], dtype=int)

    out_rows = []
    n_matched = 0
    n_unmatched = 0
    for _, row in case_lesions.iterrows():
        target_centroid = np.array([
            float(row["centroid_0"]),
            float(row["centroid_1"]),
            float(row["centroid_2"]),
        ])
        target_n_voxels = int(row["n_voxels"])

        new_row = row.to_dict()
        new_row["surface_area_cm2"] = np.nan
        new_row["sphericity"] = np.nan

        if len(valid_ids) == 0:
            n_unmatched += 1
            out_rows.append(new_row)
            continue

        dists = np.linalg.norm(valid_centroids - target_centroid, axis=1)
        best = int(np.argmin(dists))
        if dists[best] > CENTROID_MATCH_TOL_VOX:
            n_unmatched += 1
            out_rows.append(new_row)
            continue
        if abs(valid_sizes[best] - target_n_voxels) > N_VOXELS_RELATIVE_TOL * target_n_voxels:
            n_unmatched += 1
            out_rows.append(new_row)
            continue

        comp_mask = labelled == valid_ids[best]
        sa_cm2 = compute_surface_area(comp_mask, voxel_spacing_zyx)
        sph = compute_sphericity(float(row["volume_ml"]), sa_cm2)
        new_row["surface_area_cm2"] = float(sa_cm2)
        new_row["sphericity"] = float(sph)
        n_matched += 1
        out_rows.append(new_row)

    augmented_df = pd.DataFrame(out_rows)
    result = AugmentationResult(
        case_id=case_id,
        n_lesions_in=int(len(case_lesions)),
        n_lesions_matched=n_matched,
        n_lesions_unmatched=n_unmatched,
    )
    return augmented_df, result
