"""SUVpeak calculation per PERCIST 1.0 definition.

SUVpeak = mean SUV within a 1 mL (1 cm³) spherical ROI centred on the
voxel with the highest SUV value within the lesion.

For lesions with volume < 1.2 mL, SUVpeak is set to SUVmax (the sphere
cannot fit reliably within the lesion).

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§3.3)

Reference:
    Wahl et al. (2009) JNM 50(Suppl 1):122S-150S (PERCIST 1.0)
"""

from __future__ import annotations

import numpy as np


# 1 mL sphere radius in mm: r = (3V / 4π)^(1/3) where V = 1000 mm³
SPHERE_VOLUME_ML = 1.0
SPHERE_RADIUS_MM = (3.0 * SPHERE_VOLUME_ML * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)  # ≈6.20 mm

# Pre-registered threshold below which SUVpeak := SUVmax (§3.3)
MIN_VOLUME_FOR_PEAK_ML = 1.2


def compute_suvpeak(
    suv_volume: np.ndarray,
    lesion_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
) -> tuple[float, bool]:
    """Compute SUVpeak for a single lesion.

    Parameters
    ----------
    suv_volume : np.ndarray
        3D SUV image (z, y, x).
    lesion_mask : np.ndarray
        3D binary mask for this lesion (same shape as suv_volume).
    voxel_spacing_mm : tuple
        (z_spacing, y_spacing, x_spacing) in mm.

    Returns
    -------
    tuple[float, bool]
        (suvpeak_value, is_fallback) where is_fallback=True if the lesion
        was too small and SUVpeak was set to SUVmax.
    """
    voxel_volume_ml = (
        voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]
    ) / 1000.0  # mm³ -> mL

    lesion_voxels = lesion_mask > 0
    n_voxels = int(lesion_voxels.sum())
    lesion_volume_ml = n_voxels * voxel_volume_ml

    # Extract SUV values within the lesion
    suv_in_lesion = suv_volume[lesion_voxels]
    suvmax = float(suv_in_lesion.max())

    # Fallback for small lesions
    if lesion_volume_ml < MIN_VOLUME_FOR_PEAK_ML:
        return suvmax, True

    # Find the hottest voxel location
    masked_suv = np.where(lesion_voxels, suv_volume, -np.inf)
    hottest_idx = np.unravel_index(np.argmax(masked_suv), suv_volume.shape)

    # Build the 1 mL spherical ROI centred on the hottest voxel
    sphere_mask = _build_sphere_mask(
        volume_shape=suv_volume.shape,
        centre_voxel=hottest_idx,
        radius_mm=SPHERE_RADIUS_MM,
        voxel_spacing_mm=voxel_spacing_mm,
    )

    # Intersect sphere with the lesion mask to avoid sampling background
    # PERCIST specifies the sphere should be within the tumour; for voxels
    # at the lesion edge, only include voxels that are inside the lesion.
    sphere_in_lesion = sphere_mask & lesion_voxels
    n_sphere = int(sphere_in_lesion.sum())

    if n_sphere == 0:
        return suvmax, True

    suvpeak = float(suv_volume[sphere_in_lesion].mean())
    return suvpeak, False


def _build_sphere_mask(
    volume_shape: tuple[int, int, int],
    centre_voxel: tuple[int, int, int],
    radius_mm: float,
    voxel_spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    """Build a binary sphere mask on the voxel grid.

    Parameters
    ----------
    volume_shape : tuple
        (z, y, x) shape of the volume.
    centre_voxel : tuple
        (z, y, x) index of the sphere centre.
    radius_mm : float
        Sphere radius in mm.
    voxel_spacing_mm : tuple
        (z_spacing, y_spacing, x_spacing) in mm.

    Returns
    -------
    np.ndarray
        Binary mask (same shape as volume), dtype bool.
    """
    cz, cy, cx = centre_voxel
    sz, sy, sx = voxel_spacing_mm

    # Compute the search range in voxels (bounding box of the sphere)
    rz = int(np.ceil(radius_mm / sz)) + 1
    ry = int(np.ceil(radius_mm / sy)) + 1
    rx = int(np.ceil(radius_mm / sx)) + 1

    # Build coordinate grids relative to centre (in mm)
    z_range = np.arange(max(0, cz - rz), min(volume_shape[0], cz + rz + 1))
    y_range = np.arange(max(0, cy - ry), min(volume_shape[1], cy + ry + 1))
    x_range = np.arange(max(0, cx - rx), min(volume_shape[2], cx + rx + 1))

    zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing="ij")

    # Distance from centre in mm
    dist_sq = (
        ((zz - cz) * sz) ** 2
        + ((yy - cy) * sy) ** 2
        + ((xx - cx) * sx) ** 2
    )

    # Voxels within the sphere
    in_sphere = dist_sq <= radius_mm ** 2

    # Create full-volume mask
    mask = np.zeros(volume_shape, dtype=bool)
    mask[
        z_range[0] : z_range[-1] + 1,
        y_range[0] : y_range[-1] + 1,
        x_range[0] : x_range[-1] + 1,
    ] = in_sphere

    return mask
