"""Build paired CT+PET NIfTI inputs for two-channel nnU-Net segmentation.

The AutoPET-III LesionTracer model (and the autoPET-III challenge format
generally) expects per-case input as two NIfTI files:
- `{case}_0000.nii.gz`: CT in HU, on the PET grid
- `{case}_0001.nii.gz`: PET in SUV body-weight

This module provides `build_paired_niftis` which orchestrates the full
DICOM-to-paired-NIfTI conversion using the validated SUV pipeline
(`src.preprocess.suv_conversion`) and the DICOM I/O helpers.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.2.2)
"""

from __future__ import annotations

import glob
import os

try:
    import SimpleITK as sitk
    import numpy as np
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

from .dicom_io import (
    load_dicom_series,
    read_ct_as_hu_sitk,
    resample_to_reference,
)


def build_paired_niftis(
    pt_uid: str,
    ct_uid: str,
    out_dir: str,
    drive_root: str,
    case_name: str | None = None,
    ct_default_value: float = -1024.0,
) -> dict:
    """Build the paired CT+PET NIfTI files for one case.

    Writes:
    - `{out_dir}/{case_name}_0000.nii.gz` -- CT in HU, resampled to PET grid
    - `{out_dir}/{case_name}_0001.nii.gz` -- PET in SUV body-weight

    The default `case_name = pt_uid` ensures that nnU-Net's prediction output
    file `{out_dir}/{pt_uid}.nii.gz` matches the path expected by downstream
    feature extraction (`process_autopet_iii.ipynb` Step 7).

    Parameters
    ----------
    pt_uid : str
        SeriesInstanceUID of the PET series.
    ct_uid : str
        SeriesInstanceUID of the matching CT series (same StudyInstanceUID).
    out_dir : str
        Directory to write paired NIfTI files. Created if missing.
    drive_root : str
        Root path containing `{uid}.zip` or `{uid}/` per series.
    case_name : str, optional
        Filename prefix. Defaults to `pt_uid`.
    ct_default_value : float
        Pixel value for out-of-FOV regions when CT is resampled onto the
        PET grid. -1024 HU (air) by default.

    Returns
    -------
    dict
        Diagnostic info: case, patient_id, manufacturer, radionuclide,
        pet_shape, pet_spacing, suv_max, ct_orig_shape, ct_orig_spacing.

    Raises
    ------
    ImportError
        If SimpleITK or numpy is unavailable.
    FileNotFoundError
        If either the PT or CT series is not found at `drive_root`.
    """
    if not _HAS_DEPS:
        raise ImportError("SimpleITK and numpy are required for build_paired_niftis")

    # Local import so the module imports cleanly even if pydicom is missing.
    from src.preprocess.suv_conversion import (
        dicom_series_to_suv_sitk,
        extract_pet_metadata,
    )

    case = case_name or pt_uid
    os.makedirs(out_dir, exist_ok=True)

    pt_dir, pt_cleanup = load_dicom_series(pt_uid, drive_root)
    ct_dir, ct_cleanup = load_dicom_series(ct_uid, drive_root)
    try:
        # Pick any DICOM file in the PT series for metadata extraction
        any_pt_dcm = next(iter(
            glob.glob(os.path.join(pt_dir, "*.dcm"))
            or [
                os.path.join(pt_dir, e)
                for e in os.listdir(pt_dir)
                if os.path.isfile(os.path.join(pt_dir, e))
            ]
        ))
        meta = extract_pet_metadata(any_pt_dcm)
        suv_image = dicom_series_to_suv_sitk(pt_dir, meta)

        ct_image = read_ct_as_hu_sitk(ct_dir)
        ct_resampled = resample_to_reference(
            ct_image, suv_image, default_value=ct_default_value
        )

        sitk.WriteImage(ct_resampled, os.path.join(out_dir, f"{case}_0000.nii.gz"))
        sitk.WriteImage(suv_image, os.path.join(out_dir, f"{case}_0001.nii.gz"))

        return {
            "case": case,
            "patient_id": meta.patient_id,
            "manufacturer": meta.manufacturer,
            "radionuclide": meta.radionuclide,
            "pet_shape": suv_image.GetSize(),
            "pet_spacing": suv_image.GetSpacing(),
            "suv_max": float(np.max(sitk.GetArrayFromImage(suv_image))),
            "ct_orig_shape": ct_image.GetSize(),
            "ct_orig_spacing": ct_image.GetSpacing(),
        }
    finally:
        pt_cleanup()
        ct_cleanup()
