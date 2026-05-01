"""Tests for src/segment/ -- DICOM I/O and paired-NIfTI builder.

The DICOM-dependent tests use synthetic PT (reused from test_suv_conversion)
and synthetic CT slices written to tmpdir. Each test is isolated.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§5.2.2)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile

import numpy as np
import pytest

# Test the pure-Python load_dicom_series logic without DICOM deps
from src.segment.dicom_io import load_dicom_series

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    import SimpleITK as sitk

    from src.segment.dicom_io import (
        read_ct_as_hu_sitk,
        resample_to_reference,
    )
    from src.segment.paired_input import build_paired_niftis
    from tests.test_suv_conversion import _make_pt_slice  # reuse PT synthetic helper

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

requires_dicom = pytest.mark.skipif(
    not _HAS_DEPS, reason="pydicom + SimpleITK required for DICOM integration tests"
)

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"


# --- pure-python (no DICOM deps) tests for load_dicom_series ---


class TestLoadDicomSeriesStorageDispatch:
    """Test the zip-or-directory dispatch logic without any actual DICOM data."""

    def setup_method(self):
        self.root = tempfile.mkdtemp(prefix="test_seg_root_")

    def teardown_method(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_directory_storage_is_returned_with_noop_cleanup(self):
        uid = "1.2.3.4.5"
        os.makedirs(os.path.join(self.root, uid))
        # Touch a placeholder file so the dir is non-empty
        open(os.path.join(self.root, uid, "placeholder.dcm"), "wb").close()

        path, cleanup = load_dicom_series(uid, drive_root=self.root)
        assert path == os.path.join(self.root, uid)
        cleanup()  # no-op for dir storage; should not delete drive data
        assert os.path.exists(path)

    def test_zip_storage_extracts_to_temp(self):
        uid = "1.2.3.4.5"
        zip_path = os.path.join(self.root, f"{uid}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.dcm", b"placeholder1")
            zf.writestr("b.dcm", b"placeholder2")

        path, cleanup = load_dicom_series(uid, drive_root=self.root)
        assert path != os.path.join(self.root, uid)
        assert os.path.exists(os.path.join(path, "a.dcm"))
        assert os.path.exists(os.path.join(path, "b.dcm"))
        cleanup()
        assert not os.path.exists(path)  # tmpdir removed

    def test_zip_with_inner_directory_is_flattened(self):
        uid = "1.2.3.4.5"
        zip_path = os.path.join(self.root, f"{uid}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner/a.dcm", b"placeholder1")
            zf.writestr("inner/b.dcm", b"placeholder2")

        path, cleanup = load_dicom_series(uid, drive_root=self.root)
        # Should return the inner/ directory, not the tmp root
        assert os.path.basename(path) == "inner"
        assert os.path.exists(os.path.join(path, "a.dcm"))
        cleanup()

    def test_zip_takes_precedence_over_directory(self):
        uid = "1.2.3.4.5"
        # Create both zip and directory
        os.makedirs(os.path.join(self.root, uid))
        open(os.path.join(self.root, uid, "from_dir.dcm"), "wb").close()
        with zipfile.ZipFile(os.path.join(self.root, f"{uid}.zip"), "w") as zf:
            zf.writestr("from_zip.dcm", b"placeholder")

        path, cleanup = load_dicom_series(uid, drive_root=self.root)
        # Zip wins -- we should be in the tmp extraction, not the dir
        assert path != os.path.join(self.root, uid)
        assert os.path.exists(os.path.join(path, "from_zip.dcm"))
        cleanup()

    def test_missing_raises_filenotfounderror(self):
        with pytest.raises(FileNotFoundError):
            load_dicom_series("does_not_exist", drive_root=self.root)


# --- DICOM-dependent integration tests ---


def _make_ct_slice(
    path: str,
    slice_z_mm: float,
    raw_pixels: np.ndarray,
    rescale_slope: float = 1.0,
    rescale_intercept: float = -1024.0,
    series_uid: str = None,
    study_uid: str = None,
    frame_uid: str = None,
    instance_number: int = 1,
):
    """Write a minimal valid CT DICOM file. Default rescale maps stored
    values to HU (slope=1, intercept=-1024)."""
    if series_uid is None:
        series_uid = generate_uid()
    if study_uid is None:
        study_uid = generate_uid()
    if frame_uid is None:
        frame_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"
    ds.PatientBirthDate = "19500101"
    ds.PatientSex = "M"

    ds.StudyInstanceUID = study_uid
    ds.StudyDate = "20260101"
    ds.StudyTime = "083000"

    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = "1"
    ds.SeriesDate = "20260101"
    ds.SeriesTime = "083000"
    ds.Modality = "CT"
    ds.SeriesDescription = "Test CT"

    ds.FrameOfReferenceUID = frame_uid
    ds.PositionReferenceIndicator = ""

    ds.SOPClassUID = CT_IMAGE_STORAGE
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = str(instance_number)

    ds.PixelSpacing = [1.0, 1.0]
    ds.ImagePositionPatient = [0.0, 0.0, slice_z_mm]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceThickness = "1.0"

    raw = np.asarray(raw_pixels, dtype=np.uint16)
    ds.Rows, ds.Columns = raw.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.PixelData = raw.tobytes()

    ds.RescaleSlope = str(rescale_slope)
    ds.RescaleIntercept = str(rescale_intercept)
    ds.RescaleType = "HU"

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path, write_like_original=False)


@requires_dicom
class TestReadCTAsHU:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_seg_ct_")
        self.series_uid = generate_uid()
        self.study_uid = generate_uid()
        self.frame_uid = generate_uid()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_hu_with_default_rescale(self):
        """stored=1024 with slope=1 intercept=-1024 -> HU=0 (water)."""
        for i in range(3):
            _make_ct_slice(
                path=os.path.join(self.tmpdir, f"slice_{i:03d}.dcm"),
                slice_z_mm=float(i * 1.0),
                raw_pixels=np.full((4, 4), 1024, dtype=np.uint16),
                series_uid=self.series_uid,
                study_uid=self.study_uid,
                frame_uid=self.frame_uid,
                instance_number=i + 1,
            )
        img = read_ct_as_hu_sitk(self.tmpdir)
        arr = sitk.GetArrayFromImage(img)
        assert arr.shape == (3, 4, 4)
        np.testing.assert_allclose(arr, 0.0, atol=0.5)

    def test_rescale_slope_and_intercept_applied(self):
        """stored=2048 with slope=1 intercept=-1024 -> HU=1024 (denser tissue)."""
        for i in range(3):
            _make_ct_slice(
                path=os.path.join(self.tmpdir, f"slice_{i:03d}.dcm"),
                slice_z_mm=float(i * 1.0),
                raw_pixels=np.full((4, 4), 2048, dtype=np.uint16),
                series_uid=self.series_uid,
                study_uid=self.study_uid,
                frame_uid=self.frame_uid,
                instance_number=i + 1,
            )
        img = read_ct_as_hu_sitk(self.tmpdir)
        arr = sitk.GetArrayFromImage(img)
        np.testing.assert_allclose(arr, 1024.0, atol=0.5)

    def test_no_series_raises(self):
        with pytest.raises(ValueError, match="No DICOM series"):
            read_ct_as_hu_sitk(self.tmpdir)


@requires_dicom
class TestResampleToReference:
    def test_identity_when_grids_match(self):
        ref = sitk.Image(8, 8, 4, sitk.sitkFloat32)
        ref.SetSpacing((2.0, 2.0, 3.0))
        moving = sitk.Image(8, 8, 4, sitk.sitkFloat32)
        moving.SetSpacing((2.0, 2.0, 3.0))
        # Set a non-zero pixel value in moving
        arr = np.full((4, 8, 8), 7.5, dtype=np.float32)
        moving = sitk.GetImageFromArray(arr)
        moving.SetSpacing((2.0, 2.0, 3.0))

        out = resample_to_reference(moving, ref)
        out_arr = sitk.GetArrayFromImage(out)
        assert out_arr.shape == (4, 8, 8)
        np.testing.assert_allclose(out_arr, 7.5)

    def test_default_value_for_out_of_fov(self):
        """A reference larger than the moving image should fill -1024 outside FOV."""
        ref = sitk.Image(16, 16, 8, sitk.sitkFloat32)
        ref.SetSpacing((1.0, 1.0, 1.0))
        moving_arr = np.full((4, 4, 4), 50.0, dtype=np.float32)
        moving = sitk.GetImageFromArray(moving_arr)
        moving.SetSpacing((1.0, 1.0, 1.0))

        out = resample_to_reference(moving, ref, default_value=-1024.0)
        out_arr = sitk.GetArrayFromImage(out)
        assert out_arr.shape == (8, 16, 16)
        # Most voxels should be the default fill since moving is much smaller
        n_filled = int((out_arr == -1024.0).sum())
        assert n_filled > out_arr.size // 2


@requires_dicom
class TestBuildPairedNiftis:
    """End-to-end: synthetic PT + synthetic CT directories on a tmpdir 'drive'."""

    def setup_method(self):
        self.root = tempfile.mkdtemp(prefix="test_seg_paired_")
        self.out_dir = tempfile.mkdtemp(prefix="test_seg_paired_out_")
        self.pt_uid = generate_uid()
        self.ct_uid = generate_uid()
        self.study_uid = generate_uid()
        self.frame_uid = generate_uid()

        # Build PT directory
        pt_dir = os.path.join(self.root, self.pt_uid)
        os.makedirs(pt_dir)
        for i in range(3):
            _make_pt_slice(
                path=os.path.join(pt_dir, f"pt_{i:03d}.dcm"),
                slice_z_mm=float(i * 3.0),
                raw_pixels=np.full((4, 4), 1000, dtype=np.uint16),
                rescale_slope=0.001,
                rescale_intercept=0.0,
                series_uid=self.pt_uid,
                study_uid=self.study_uid,
                frame_uid=self.frame_uid,
                instance_number=i + 1,
            )

        # Build CT directory (twice as many slices, half the spacing -- realistic mismatch)
        ct_dir = os.path.join(self.root, self.ct_uid)
        os.makedirs(ct_dir)
        for i in range(6):
            _make_ct_slice(
                path=os.path.join(ct_dir, f"ct_{i:03d}.dcm"),
                slice_z_mm=float(i * 1.5),
                raw_pixels=np.full((4, 4), 1024, dtype=np.uint16),
                series_uid=self.ct_uid,
                study_uid=self.study_uid,
                frame_uid=self.frame_uid,
                instance_number=i + 1,
            )

    def teardown_method(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_writes_paired_niftis_with_canonical_naming(self):
        info = build_paired_niftis(
            self.pt_uid, self.ct_uid, self.out_dir, drive_root=self.root,
        )
        ct_path = os.path.join(self.out_dir, f"{self.pt_uid}_0000.nii.gz")
        pt_path = os.path.join(self.out_dir, f"{self.pt_uid}_0001.nii.gz")
        assert os.path.exists(ct_path)
        assert os.path.exists(pt_path)
        assert info["case"] == self.pt_uid
        assert info["pet_shape"] is not None
        assert info["suv_max"] > 0

    def test_ct_resampled_to_pet_grid(self):
        """After build_paired_niftis, CT should have the same shape as PET."""
        build_paired_niftis(
            self.pt_uid, self.ct_uid, self.out_dir, drive_root=self.root,
        )
        ct = sitk.ReadImage(os.path.join(self.out_dir, f"{self.pt_uid}_0000.nii.gz"))
        pt = sitk.ReadImage(os.path.join(self.out_dir, f"{self.pt_uid}_0001.nii.gz"))
        assert ct.GetSize() == pt.GetSize()
        assert ct.GetSpacing() == pt.GetSpacing()

    def test_custom_case_name(self):
        info = build_paired_niftis(
            self.pt_uid, self.ct_uid, self.out_dir, drive_root=self.root,
            case_name="custom_case_id",
        )
        assert info["case"] == "custom_case_id"
        assert os.path.exists(os.path.join(self.out_dir, "custom_case_id_0000.nii.gz"))
        assert os.path.exists(os.path.join(self.out_dir, "custom_case_id_0001.nii.gz"))
