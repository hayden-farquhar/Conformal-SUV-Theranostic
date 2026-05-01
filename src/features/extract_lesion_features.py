"""Per-lesion feature extraction from SUV volumes and segmentation masks.

Extracts all pre-registered features (§4.1-4.2) for each connected
component (lesion) in a segmentation mask:

Outcome variables:
    - SUVmax, SUVpeak, SUVmean, TLG

Predictor variables (for CQR):
    - volume_ml, surface_area_cm2, sphericity
    - nnunet_softmax_mean, nnunet_softmax_entropy (if softmax available)

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§4.1-4.2)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

from src.features.suvpeak import compute_suvpeak


# Pre-registered minimum lesion volume (§3.2)
MIN_LESION_VOLUME_ML = 1.0


@dataclass
class LesionFeatures:
    """All pre-registered features for a single lesion."""

    # Identity
    patient_id: str
    study_uid: str
    lesion_id: int  # connected component label

    # Outcome variables (§4.1)
    suvmax: float
    suvpeak: float
    suvpeak_is_fallback: bool  # True if lesion too small for 1mL sphere
    suvmean: float
    tlg: float  # total lesion glycolysis = suvmean * volume_ml

    # Geometric features (§4.2)
    volume_ml: float
    surface_area_cm2: float
    sphericity: float  # 6√π × V^(2/3) / SA

    # Centroid (for lesion matching and body-region assignment)
    centroid_z: float
    centroid_y: float
    centroid_x: float

    # Voxel counts
    n_voxels: int


def extract_connected_components(
    segmentation_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    min_volume_ml: float = MIN_LESION_VOLUME_ML,
) -> list[tuple[int, np.ndarray]]:
    """Extract connected components from a binary segmentation mask.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        3D binary mask (0/1 or boolean).
    voxel_spacing_mm : tuple
        (z, y, x) spacing in mm.
    min_volume_ml : float
        Minimum lesion volume in mL. Components below this are excluded.

    Returns
    -------
    list of (label, component_mask) tuples
        Each component_mask is a boolean 3D array (same shape as input).
    """
    binary = (segmentation_mask > 0).astype(np.int32)
    labelled, n_labels = ndimage.label(binary)

    voxel_volume_ml = np.prod(voxel_spacing_mm) / 1000.0

    components = []
    for label_id in range(1, n_labels + 1):
        component_mask = labelled == label_id
        n_voxels = int(component_mask.sum())
        volume_ml = n_voxels * voxel_volume_ml

        if volume_ml >= min_volume_ml:
            components.append((label_id, component_mask))

    return components


def compute_surface_area(
    mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
) -> float:
    """Estimate surface area of a binary mask using voxel face counting.

    Counts exposed faces between foreground and background voxels,
    scaled by face area. This is a simple but robust approximation
    that avoids marching cubes dependencies.

    Parameters
    ----------
    mask : np.ndarray
        3D boolean mask.
    voxel_spacing_mm : tuple
        (z, y, x) spacing in mm.

    Returns
    -------
    float
        Surface area in cm².
    """
    sz, sy, sx = voxel_spacing_mm
    mask_int = mask.astype(np.int8)

    # Count exposed faces along each axis
    # Z-axis faces (area = sy * sx per face)
    z_faces = int(np.abs(np.diff(mask_int, axis=0)).sum())
    # Y-axis faces (area = sz * sx per face)
    y_faces = int(np.abs(np.diff(mask_int, axis=1)).sum())
    # X-axis faces (area = sz * sy per face)
    x_faces = int(np.abs(np.diff(mask_int, axis=2)).sum())

    # Add boundary faces (edges of the volume)
    z_faces += int(mask_int[0, :, :].sum()) + int(mask_int[-1, :, :].sum())
    y_faces += int(mask_int[:, 0, :].sum()) + int(mask_int[:, -1, :].sum())
    x_faces += int(mask_int[:, :, 0].sum()) + int(mask_int[:, :, -1].sum())

    # Surface area in mm², then convert to cm²
    sa_mm2 = z_faces * (sy * sx) + y_faces * (sz * sx) + x_faces * (sz * sy)
    return sa_mm2 / 100.0  # mm² -> cm²


def compute_sphericity(volume_ml: float, surface_area_cm2: float) -> float:
    """Compute sphericity: π^(1/3) × (6V)^(2/3) / SA.

    A perfect sphere has sphericity = 1.0.
    Irregular shapes have sphericity < 1.0.

    Parameters
    ----------
    volume_ml : float
        Volume in mL (= cm³).
    surface_area_cm2 : float
        Surface area in cm².

    Returns
    -------
    float
        Sphericity in [0, 1]. Returns 0 if surface area is 0.
    """
    if surface_area_cm2 <= 0:
        return 0.0
    # Standard sphericity: ψ = π^(1/3) × (6V)^(2/3) / SA
    # For a sphere: SA = π^(1/3) × (6 × 4/3 π r³)^(2/3) = 4πr² ✓
    return (np.pi ** (1.0 / 3.0) * (6.0 * volume_ml) ** (2.0 / 3.0)) / surface_area_cm2


def extract_features_single_lesion(
    suv_volume: np.ndarray,
    lesion_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    patient_id: str,
    study_uid: str,
    lesion_id: int,
) -> LesionFeatures:
    """Extract all pre-registered features for a single lesion.

    Parameters
    ----------
    suv_volume : np.ndarray
        3D SUV image (z, y, x).
    lesion_mask : np.ndarray
        3D boolean mask for this lesion.
    voxel_spacing_mm : tuple
        (z, y, x) spacing in mm.
    patient_id : str
    study_uid : str
    lesion_id : int
        Connected component label.

    Returns
    -------
    LesionFeatures
    """
    voxel_volume_ml = np.prod(voxel_spacing_mm) / 1000.0
    lesion_voxels = lesion_mask > 0
    n_voxels = int(lesion_voxels.sum())
    volume_ml = n_voxels * voxel_volume_ml

    # SUV statistics within lesion
    suv_in_lesion = suv_volume[lesion_voxels]
    suvmax = float(suv_in_lesion.max())
    suvmean = float(suv_in_lesion.mean())
    tlg = suvmean * volume_ml

    # SUVpeak (1 mL sphere)
    suvpeak, is_fallback = compute_suvpeak(suv_volume, lesion_mask, voxel_spacing_mm)

    # Geometry
    surface_area_cm2 = compute_surface_area(lesion_mask, voxel_spacing_mm)
    sphericity = compute_sphericity(volume_ml, surface_area_cm2)

    # Centroid in voxel coordinates
    coords = np.argwhere(lesion_voxels)
    centroid = coords.mean(axis=0)

    return LesionFeatures(
        patient_id=patient_id,
        study_uid=study_uid,
        lesion_id=lesion_id,
        suvmax=suvmax,
        suvpeak=suvpeak,
        suvpeak_is_fallback=is_fallback,
        suvmean=suvmean,
        tlg=tlg,
        volume_ml=volume_ml,
        surface_area_cm2=surface_area_cm2,
        sphericity=sphericity,
        centroid_z=float(centroid[0]),
        centroid_y=float(centroid[1]),
        centroid_x=float(centroid[2]),
        n_voxels=n_voxels,
    )


def extract_all_lesions(
    suv_volume: np.ndarray,
    segmentation_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    patient_id: str,
    study_uid: str,
    min_volume_ml: float = MIN_LESION_VOLUME_ML,
) -> list[LesionFeatures]:
    """Extract features for all lesions in a study.

    Parameters
    ----------
    suv_volume : np.ndarray
        3D SUV image.
    segmentation_mask : np.ndarray
        3D segmentation mask (may contain multiple lesions).
    voxel_spacing_mm : tuple
        (z, y, x) voxel spacing in mm.
    patient_id : str
    study_uid : str
    min_volume_ml : float
        Minimum lesion volume.

    Returns
    -------
    list[LesionFeatures]
        One entry per lesion meeting the volume threshold.
    """
    components = extract_connected_components(
        segmentation_mask, voxel_spacing_mm, min_volume_ml
    )

    features = []
    for label_id, component_mask in components:
        feat = extract_features_single_lesion(
            suv_volume=suv_volume,
            lesion_mask=component_mask,
            voxel_spacing_mm=voxel_spacing_mm,
            patient_id=patient_id,
            study_uid=study_uid,
            lesion_id=label_id,
        )
        features.append(feat)

    return features


def lesion_features_to_dataframe(
    features_list: list[LesionFeatures],
) -> "pd.DataFrame":
    """Convert a list of LesionFeatures to a DataFrame."""
    import pandas as pd
    return pd.DataFrame([asdict(f) for f in features_list])
