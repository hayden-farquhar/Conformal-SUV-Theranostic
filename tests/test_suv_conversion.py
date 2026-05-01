"""Tests for SUV conversion logic.

Uses synthetic PETMetadata to validate SUV_bw, SUL, and LBM calculations
without requiring real DICOM files.
"""

import datetime
import math

import numpy as np
import pytest

from src.preprocess.suv_conversion import (
    PETMetadata,
    compute_suv_bw,
    compute_sul,
    lean_body_mass_janmahasatian,
    _parse_dicom_datetime,
    _infer_radionuclide,
)


def _make_meta(**overrides) -> PETMetadata:
    """Create a PETMetadata with sensible defaults, overridable."""
    defaults = dict(
        patient_id="TEST_001",
        patient_weight_kg=75.0,
        patient_height_m=1.75,
        patient_sex="M",
        patient_age="060Y",
        injected_dose_bq=370e6,  # 370 MBq
        injection_time=datetime.datetime(2020, 1, 1, 8, 0, 0),
        scan_time=datetime.datetime(2020, 1, 1, 9, 0, 0),  # 60 min uptake
        half_life_sec=6586.2,  # F-18
        radionuclide="F-18",
        manufacturer="SIEMENS",
        manufacturer_model="Biograph mCT",
        software_version="VG60A",
        pixel_spacing_mm=(4.07, 4.07),
        slice_thickness_mm=3.0,
        rescale_slope=1.0,
        rescale_intercept=0.0,
        series_uid="1.2.3.4.5",
        study_uid="1.2.3.4",
        uptake_time_sec=3600.0,
        decay_factor=2 ** (-3600.0 / 6586.2),
    )
    defaults.update(overrides)
    return PETMetadata(**defaults)


class TestComputeSuvBw:
    def test_basic_suv_calculation(self):
        """A known activity concentration should produce a predictable SUV."""
        meta = _make_meta()

        # If a voxel has activity = dose/weight (in Bq/mL / (Bq/g)),
        # then SUV = 1.0
        decay_corrected_dose = meta.injected_dose_bq * meta.decay_factor
        weight_g = meta.patient_weight_kg * 1000.0
        unity_activity = decay_corrected_dose / weight_g

        pixel = np.array([unity_activity])
        suv = compute_suv_bw(pixel, meta)
        assert abs(suv[0] - 1.0) < 1e-10

    def test_suv_scales_linearly(self):
        """Doubling activity should double SUV."""
        meta = _make_meta()
        pixel_1 = np.array([1000.0])
        pixel_2 = np.array([2000.0])

        suv_1 = compute_suv_bw(pixel_1, meta)
        suv_2 = compute_suv_bw(pixel_2, meta)

        assert abs(suv_2[0] / suv_1[0] - 2.0) < 1e-10

    def test_heavier_patient_higher_suv(self):
        """Same activity in a heavier patient should give higher SUV_bw."""
        pixel = np.array([5000.0])
        meta_70 = _make_meta(patient_weight_kg=70.0)
        meta_90 = _make_meta(patient_weight_kg=90.0)

        suv_70 = compute_suv_bw(pixel, meta_70)
        suv_90 = compute_suv_bw(pixel, meta_90)

        assert suv_90[0] > suv_70[0]
        assert abs(suv_90[0] / suv_70[0] - 90.0 / 70.0) < 1e-10

    def test_longer_uptake_higher_suv(self):
        """Longer uptake time means more decay, lower decay_corrected_dose,
        so for the same measured activity, SUV is higher."""
        pixel = np.array([5000.0])
        meta_60min = _make_meta(
            uptake_time_sec=3600,
            decay_factor=2 ** (-3600 / 6586.2),
        )
        meta_90min = _make_meta(
            uptake_time_sec=5400,
            decay_factor=2 ** (-5400 / 6586.2),
        )

        suv_60 = compute_suv_bw(pixel, meta_60min)
        suv_90 = compute_suv_bw(pixel, meta_90min)

        # More decay -> lower corrected dose -> higher SUV
        assert suv_90[0] > suv_60[0]

    def test_rescale_slope_intercept(self):
        """Rescale parameters should be applied before SUV calculation."""
        meta = _make_meta(rescale_slope=2.0, rescale_intercept=100.0)
        pixel = np.array([1000.0])

        suv = compute_suv_bw(pixel, meta)

        # Activity = 1000 * 2.0 + 100.0 = 2100.0
        meta_unity = _make_meta(rescale_slope=1.0, rescale_intercept=0.0)
        suv_direct = compute_suv_bw(np.array([2100.0]), meta_unity)

        assert abs(suv[0] - suv_direct[0]) < 1e-10

    def test_3d_array(self):
        """Should work on 3D arrays (typical PET volume)."""
        meta = _make_meta()
        pixel = np.random.rand(10, 128, 128).astype(np.float64) * 10000
        suv = compute_suv_bw(pixel, meta)

        assert suv.shape == pixel.shape
        assert suv.dtype == np.float64
        assert (suv >= 0).all()


class TestComputeSul:
    def test_sul_lower_than_suv_bw_for_overweight(self):
        """For an overweight patient, LBM < weight, so SUL < SUV_bw."""
        meta = _make_meta(patient_weight_kg=100.0, patient_height_m=1.70, patient_sex="M")
        suv_bw = np.array([5.0])
        sul = compute_sul(suv_bw, meta)

        assert sul is not None
        lbm = lean_body_mass_janmahasatian(100.0, 1.70, "M")
        assert lbm < 100.0  # LBM < total weight
        assert sul[0] < suv_bw[0]

    def test_sul_returns_none_without_height(self):
        meta = _make_meta(patient_height_m=None)
        suv_bw = np.array([5.0])
        result = compute_sul(suv_bw, meta)
        assert result is None

    def test_sul_returns_none_without_sex(self):
        meta = _make_meta(patient_sex=None)
        suv_bw = np.array([5.0])
        result = compute_sul(suv_bw, meta)
        assert result is None


class TestLeanBodyMass:
    def test_male_formula(self):
        """Verify Janmahasatian formula for a reference male."""
        # 75 kg, 1.75 m male -> BMI = 75 / 1.75^2 = 24.49
        lbm = lean_body_mass_janmahasatian(75.0, 1.75, "M")
        bmi = 75.0 / 1.75 ** 2
        expected = 9270.0 * 75.0 / (6680.0 + 216.0 * bmi)
        assert abs(lbm - expected) < 0.01

    def test_female_formula(self):
        """Verify Janmahasatian formula for a reference female."""
        lbm = lean_body_mass_janmahasatian(65.0, 1.65, "F")
        bmi = 65.0 / 1.65 ** 2
        expected = 9270.0 * 65.0 / (8780.0 + 244.0 * bmi)
        assert abs(lbm - expected) < 0.01

    def test_returns_none_for_unknown_sex(self):
        assert lean_body_mass_janmahasatian(75.0, 1.75, "X") is None

    def test_lbm_always_less_than_weight(self):
        """LBM should always be less than total body weight."""
        for w, h, s in [(60, 1.50, "F"), (80, 1.80, "M"), (120, 1.70, "M")]:
            lbm = lean_body_mass_janmahasatian(w, h, s)
            assert lbm < w


class TestDecayFactor:
    def test_zero_uptake_no_decay(self):
        """At time zero, decay factor should be 1.0."""
        factor = 2 ** (-0.0 / 6586.2)
        assert abs(factor - 1.0) < 1e-15

    def test_one_half_life(self):
        """After one half-life, factor should be 0.5."""
        factor = 2 ** (-6586.2 / 6586.2)
        assert abs(factor - 0.5) < 1e-15

    def test_typical_f18_60min(self):
        """F-18 at 60 min uptake: should decay ~32%."""
        factor = 2 ** (-3600 / 6586.2)
        # e^(-ln2 * 3600/6586.2) ≈ 0.679
        assert 0.67 < factor < 0.69


class TestDicomDatetimeParsing:
    def test_standard_format(self):
        dt = _parse_dicom_datetime("20200101", "090000")
        assert dt == datetime.datetime(2020, 1, 1, 9, 0, 0)

    def test_fractional_seconds(self):
        dt = _parse_dicom_datetime("20200101", "090000.123456")
        assert dt.microsecond == 123456

    def test_short_time(self):
        dt = _parse_dicom_datetime("20200101", "0900")
        assert dt.hour == 9
        assert dt.minute == 0


class TestInferRadionuclide:
    def test_f18(self):
        assert _infer_radionuclide(6586.2) == "F-18"

    def test_ga68(self):
        assert _infer_radionuclide(4062.0) == "Ga-68"

    def test_unknown(self):
        assert _infer_radionuclide(9999.0) == "Unknown"

    def test_close_match(self):
        # Within 1% of F-18
        assert _infer_radionuclide(6586.2 * 1.005) == "F-18"


# ---------------------------------------------------------------------------
# Integration tests with synthetic DICOM PT series.
# Skipped cleanly if pydicom or SimpleITK is missing, so the pure-math unit
# tests above keep running in lean environments (e.g. CI without SimpleITK).
# ---------------------------------------------------------------------------

import os
import shutil
import tempfile

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    import SimpleITK as sitk
    from src.preprocess.suv_conversion import (
        dicom_series_to_suv_sitk,
        extract_pet_metadata,
    )
    _HAS_DICOM_DEPS = True
except ImportError:
    _HAS_DICOM_DEPS = False

requires_dicom_stack = pytest.mark.skipif(
    not _HAS_DICOM_DEPS, reason="pydicom + SimpleITK required for integration tests"
)

PT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.128"


def _make_pt_slice(
    path: str,
    slice_z_mm: float,
    raw_pixels: np.ndarray,
    rescale_slope: float,
    rescale_intercept: float,
    series_uid: str,
    study_uid: str,
    frame_uid: str,
    instance_number: int,
    weight_kg: float = 70.0,
    dose_bq: float = 370e6,
    half_life_sec: float = 6586.2,
    inj_time: str = "083000.000",
    scan_time: str = "093000.000",
):
    """Write a minimal valid PT DICOM file with controllable raw pixel + rescale."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = PT_IMAGE_STORAGE
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"
    ds.PatientBirthDate = "19500101"
    ds.PatientSex = "M"
    ds.PatientAge = "060Y"
    ds.PatientWeight = str(weight_kg)
    ds.PatientSize = "1.75"

    ds.StudyInstanceUID = study_uid
    ds.StudyDate = "20260101"
    ds.StudyTime = "083000"
    ds.StudyID = "1"
    ds.AccessionNumber = "12345"

    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = "1"
    ds.SeriesDate = "20260101"
    ds.SeriesTime = scan_time
    ds.Modality = "PT"
    ds.SeriesDescription = "Test PT"

    ds.FrameOfReferenceUID = frame_uid
    ds.PositionReferenceIndicator = ""

    ds.SOPClassUID = PT_IMAGE_STORAGE
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = str(instance_number)

    ds.PixelSpacing = [4.0, 4.0]
    ds.ImagePositionPatient = [0.0, 0.0, slice_z_mm]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceThickness = "3.0"

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
    ds.RescaleType = "BQML"

    ds.Units = "BQML"
    ds.AcquisitionDate = "20260101"
    ds.AcquisitionTime = scan_time
    ds.CorrectedImage = ["DECY", "ATTN"]
    ds.DecayCorrection = "START"

    radio = Dataset()
    radio.RadiopharmaceuticalStartTime = inj_time
    radio.RadionuclideTotalDose = str(dose_bq)
    radio.RadionuclideHalfLife = str(half_life_sec)
    radionuclide_seq = Dataset()
    radionuclide_seq.CodeMeaning = "^18^Fluorine"
    radio.RadionuclideCodeSequence = [radionuclide_seq]
    ds.RadiopharmaceuticalInformationSequence = [radio]

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path, write_like_original=False)


def _expected_suv(raw: float, slope: float, intercept: float = 0.0,
                  weight_kg: float = 70.0, dose_bq: float = 370e6,
                  half_life: float = 6586.2, uptake_sec: float = 3600.0) -> float:
    activity = raw * slope + intercept
    decay_corrected = dose_bq * (2.0 ** (-uptake_sec / half_life))
    return activity * (weight_kg * 1000.0) / decay_corrected


@requires_dicom_stack
class TestDicomSeriesToSuvSitkIntegration:
    """Synthetic-DICOM regression tests for the rescale-applied-twice and
    per-slice-rescale-ignored bugs caught during pre-reg §3.3 validation
    (Amendment 3, 2026-04-26). Use a tmpdir per test for isolation."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_suv_dcm_")
        self.series_uid = generate_uid()
        self.study_uid = generate_uid()
        self.frame_uid = generate_uid()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_series(self, slopes, raw_value=1000):
        """Write N synthetic PT slices with the given slopes; raw value constant."""
        for i, slope in enumerate(slopes):
            _make_pt_slice(
                path=os.path.join(self.tmpdir, f"slice_{i:03d}.dcm"),
                slice_z_mm=float(i * 3.0),
                raw_pixels=np.full((4, 4), raw_value, dtype=np.uint16),
                rescale_slope=slope,
                rescale_intercept=0.0,
                series_uid=self.series_uid,
                study_uid=self.study_uid,
                frame_uid=self.frame_uid,
                instance_number=i + 1,
            )
        sample = sorted(os.listdir(self.tmpdir))[0]
        return extract_pet_metadata(os.path.join(self.tmpdir, sample))

    def test_uniform_rescale_yields_correct_suv(self):
        """Smoke test: constant raw + constant slope => deterministic SUV."""
        meta = self._build_series([0.001, 0.001, 0.001], raw_value=1000)
        suv_image = dicom_series_to_suv_sitk(self.tmpdir, meta)
        suv = sitk.GetArrayFromImage(suv_image)
        expected = _expected_suv(raw=1000, slope=0.001)
        assert suv.shape == (3, 4, 4)
        np.testing.assert_allclose(suv, expected, rtol=1e-6)

    def test_no_double_rescale(self):
        """REGRESSION: rescale must be applied exactly once.

        Pre-2026-04-26 bug: GDCM applied rescale during sitk read; compute_suv_bw
        applied it again; SUV was inflated by ~slope× (Siemens slope=7.4 -> 640%).
        """
        slope = 7.4
        meta = self._build_series([slope, slope, slope], raw_value=10)
        suv = sitk.GetArrayFromImage(dicom_series_to_suv_sitk(self.tmpdir, meta))
        expected_once = _expected_suv(raw=10, slope=slope)
        wrong_if_doubled = _expected_suv(raw=10, slope=slope * slope)

        actual = float(suv[0, 0, 0])
        np.testing.assert_allclose(actual, expected_once, rtol=1e-6)
        assert abs(actual - wrong_if_doubled) > expected_once * 0.5, (
            f"SUV {actual} matches doubled-rescale prediction {wrong_if_doubled} - bug returned"
        )

    def test_per_slice_variable_rescale_handled(self):
        """REGRESSION: each slice's RescaleSlope must drive its own SUV.

        Pre-2026-04-26 bug: meta.rescale_slope was a scalar from the first slice
        and applied to the whole 3D volume, ignoring per-slice variation common
        on Siemens scanners.
        """
        slopes = [0.5, 2.0, 10.0]
        meta = self._build_series(slopes, raw_value=1000)
        suv = sitk.GetArrayFromImage(dicom_series_to_suv_sitk(self.tmpdir, meta))

        # SimpleITK z-sorts ascending by ImagePositionPatient[2]; we wrote them
        # in ascending z order, so suv[i] corresponds to slopes[i].
        for i, slope in enumerate(slopes):
            expected = _expected_suv(raw=1000, slope=slope)
            actual = float(suv[i, 0, 0])
            np.testing.assert_allclose(
                actual, expected, rtol=1e-6,
                err_msg=f"Slice {i} (slope={slope}): per-slice rescale not honoured",
            )

    def test_metadata_extraction_roundtrip(self):
        """extract_pet_metadata reads back what _make_pt_slice wrote."""
        meta = self._build_series([1.0], raw_value=100)
        assert meta.patient_weight_kg == 70.0
        assert meta.injected_dose_bq == 370e6
        assert abs(meta.half_life_sec - 6586.2) < 0.01
        assert abs(meta.uptake_time_sec - 3600.0) < 0.01
        assert meta.radionuclide == "F-18" or "Fluor" in meta.radionuclide
