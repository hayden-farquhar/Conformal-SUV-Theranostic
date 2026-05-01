"""Segmentation pipeline helpers for AutoPET-III nnU-Net inference.

Public API:
- `load_dicom_series`: zip-or-directory DICOM series loader
- `read_ct_as_hu_sitk`: read CT DICOM as SimpleITK image in HU
- `resample_to_reference`: resample one image onto another's grid
- `build_paired_niftis`: orchestrate DICOM -> paired CT+PET NIfTI for nnU-Net

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.2.2)
"""

from .dicom_io import (
    load_dicom_series,
    read_ct_as_hu_sitk,
    resample_to_reference,
)
from .paired_input import build_paired_niftis

__all__ = [
    "load_dicom_series",
    "read_ct_as_hu_sitk",
    "resample_to_reference",
    "build_paired_niftis",
]
