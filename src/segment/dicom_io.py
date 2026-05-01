"""DICOM series I/O helpers for the segmentation pipeline.

Provides:
- `load_dicom_series`: zip-or-directory loader for DICOM series stored on
  Google Drive (or any path) with two heterogeneous storage formats coexisting.
  Used by both AutoPET-III segmentation (P79) and any cross-project pipeline
  reusing the same Drive layout (e.g. P80 PET FM Probe).
- `read_ct_as_hu_sitk`: read a CT DICOM directory into a SimpleITK image in HU.
- `resample_to_reference`: resample a moving image (typically CT) onto the grid
  of a reference image (typically PET/SUV).

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.2.2)

These helpers were originally inlined in `kaggle_notebooks/segment_autopet_iii.ipynb`
Step 7 and promoted here on 2026-04-28 to satisfy the Phase 3 code-freeze requirement
that all analysis code be unit-tested and importable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import Callable

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False


def load_dicom_series(
    series_uid: str,
    drive_root: str,
) -> tuple[str, Callable[[], None]]:
    """Load a DICOM series, handling both zip and extracted-directory storage.

    Tries `{drive_root}/{series_uid}.zip` first; falls back to
    `{drive_root}/{series_uid}/`. The zip path extracts to a temporary
    directory (which the caller must clean up via the returned callable);
    the directory path is returned as-is with a no-op cleanup so Drive
    data is never deleted.

    Parameters
    ----------
    series_uid : str
        DICOM SeriesInstanceUID (used as the file/directory key).
    drive_root : str
        Path to the directory containing either `{series_uid}.zip` or
        `{series_uid}/` (or both -- zip wins).

    Returns
    -------
    (dicom_dir, cleanup) : tuple
        - `dicom_dir`: filesystem path containing the DICOM files
          (suitable for SimpleITK ImageSeriesReader).
        - `cleanup`: zero-argument callable that removes any temp
          extraction directory. Safe to call multiple times.

    Raises
    ------
    FileNotFoundError
        If neither the zip nor the directory exists at `drive_root`.

    Notes
    -----
    Some TCIA zips contain a single inner directory (rather than the DICOM
    files at the top level). This helper detects that pattern and returns
    the inner path so callers can treat the result uniformly.
    """
    zip_path = os.path.join(drive_root, f"{series_uid}.zip")
    dir_path = os.path.join(drive_root, series_uid)

    if os.path.exists(zip_path):
        tmp_dir = tempfile.mkdtemp(prefix="dcmser_")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        # Flatten if the zip contained a single inner directory
        entries = os.listdir(tmp_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
            inner = os.path.join(tmp_dir, entries[0])
        else:
            inner = tmp_dir
        return inner, lambda: shutil.rmtree(tmp_dir, ignore_errors=True)

    if os.path.isdir(dir_path):
        return dir_path, lambda: None

    raise FileNotFoundError(
        f"Neither {zip_path} nor {dir_path} exists for series_uid={series_uid}"
    )


def read_ct_as_hu_sitk(ct_dir: str):  # type: ignore[no-untyped-def]
    """Read a CT DICOM series as a SimpleITK image in Hounsfield Units.

    Uses SimpleITK's ImageSeriesReader, which applies RescaleSlope/Intercept
    via GDCM during the read. For CT, this returns HU directly (the canonical
    semantic of CT pixel values after rescale). Unlike PET (where we explicitly
    handle per-slice rescale variability via pydicom), CT typically has uniform
    rescale across slices, so the GDCM path is appropriate.

    Parameters
    ----------
    ct_dir : str
        Directory containing CT DICOM files for a single series.

    Returns
    -------
    sitk.Image
        3D CT volume in HU with spatial metadata (origin, spacing, direction).

    Raises
    ------
    ImportError
        If SimpleITK is not installed.
    ValueError
        If no DICOM series is found in the directory.
    """
    if not _HAS_SITK:
        raise ImportError("SimpleITK is required for read_ct_as_hu_sitk")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(ct_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {ct_dir}")
    file_names = reader.GetGDCMSeriesFileNames(ct_dir, series_ids[0])
    reader.SetFileNames(file_names)
    return reader.Execute()


def resample_to_reference(
    moving_image,  # type: ignore[no-untyped-def]
    reference_image,  # type: ignore[no-untyped-def]
    default_value: float = -1024.0,
    interpolator=None,  # type: ignore[no-untyped-def]
):  # type: ignore[no-untyped-def]
    """Resample `moving_image` onto the grid of `reference_image`.

    Used to align CT (typically higher-resolution, larger FOV) onto the PET
    grid for two-channel nnU-Net inference. Uses identity transform: assumes
    both images are already in the same patient coordinate system (which is
    true for paired CT and PET acquired in the same study).

    Parameters
    ----------
    moving_image : sitk.Image
        Image to resample (e.g., CT in HU).
    reference_image : sitk.Image
        Image whose grid (size, spacing, origin, direction) defines the
        output (e.g., SUV PET).
    default_value : float
        Pixel value for out-of-FOV regions in the moving image. For CT,
        -1024 HU (air) is the natural default; for other modalities,
        choose a value that won't bias downstream models.
    interpolator : sitk.Interpolator, optional
        SimpleITK interpolator. Defaults to linear (sitk.sitkLinear).
        Use sitk.sitkNearestNeighbor for label/segmentation images.

    Returns
    -------
    sitk.Image
        Resampled moving_image on reference_image's grid; pixel type is
        preserved from the moving image.
    """
    if not _HAS_SITK:
        raise ImportError("SimpleITK is required for resample_to_reference")
    if interpolator is None:
        interpolator = sitk.sitkLinear
    return sitk.Resample(
        moving_image,
        reference_image,
        sitk.Transform(),  # identity
        interpolator,
        default_value,
        moving_image.GetPixelID(),
    )
