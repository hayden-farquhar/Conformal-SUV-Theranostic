"""SUV conversion from DICOM PET images.

Implements body-weight SUV (SUV_bw) and lean-body-mass SUV (SUL)
conversion from DICOM PET pixel data and metadata.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§3.3)

Half-lives:
    F-18:  6586.2 seconds (109.77 min)
    Ga-68: 4062.0 seconds (67.71 min)

References:
    - Kinahan & Fletcher (2010) JNM 51(Suppl 1):11S-20S
    - Adams et al. (2010) EJNMMI 37:181-200
    - Janmahasatian et al. (2005) Clin Pharmacokinet 44:1051-1065
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

# Pre-registered half-lives (§3.3)
HALF_LIVES = {
    "F-18": 6586.2,
    "Ga-68": 4062.0,
    "C-11": 1223.4,
}


@dataclass
class PETMetadata:
    """Metadata extracted from DICOM PET series for SUV calculation."""

    patient_id: str
    patient_weight_kg: float | None
    patient_height_m: float | None
    patient_sex: str | None
    patient_age: str | None
    injected_dose_bq: float
    injection_time: datetime.datetime
    scan_time: datetime.datetime
    half_life_sec: float
    radionuclide: str
    manufacturer: str
    manufacturer_model: str
    software_version: str | None
    pixel_spacing_mm: tuple[float, float]
    slice_thickness_mm: float
    rescale_slope: float
    rescale_intercept: float
    series_uid: str
    study_uid: str
    uptake_time_sec: float  # derived: scan_time - injection_time
    decay_factor: float  # derived: 2^(-uptake_time / half_life)


def extract_pet_metadata(dcm_path: str) -> PETMetadata:  # pragma: no cover (requires pydicom + DICOM files)
    """Extract all metadata needed for SUV conversion from a single DICOM file.

    Parameters
    ----------
    dcm_path : str
        Path to any DICOM file in the PET series (metadata is per-series).

    Returns
    -------
    PETMetadata
        Dataclass with all fields needed for SUV calculation.

    Raises
    ------
    ValueError
        If essential DICOM tags are missing.
    """
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    # Patient demographics
    patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
    patient_weight_kg = _get_float(ds, "PatientWeight")
    patient_height_m = _get_float(ds, "PatientSize")  # DICOM stores height in metres
    patient_sex = getattr(ds, "PatientSex", None)
    patient_age = getattr(ds, "PatientAge", None)

    if patient_weight_kg is None:
        raise ValueError(f"PatientWeight missing for {patient_id}")

    # Radiopharmaceutical info
    radio_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if radio_seq is None or len(radio_seq) == 0:
        raise ValueError(f"RadiopharmaceuticalInformationSequence missing for {patient_id}")

    radio = radio_seq[0]

    # Injected dose
    injected_dose_bq = _get_float(radio, "RadionuclideTotalDose")
    if injected_dose_bq is None:
        raise ValueError(f"RadionuclideTotalDose missing for {patient_id}")

    # Half-life
    half_life_sec = _get_float(radio, "RadionuclideHalfLife")
    if half_life_sec is None:
        raise ValueError(f"RadionuclideHalfLife missing for {patient_id}")

    # Radionuclide name
    radionuclide_seq = getattr(radio, "RadionuclideCodeSequence", None)
    if radionuclide_seq and len(radionuclide_seq) > 0:
        radionuclide = str(getattr(radionuclide_seq[0], "CodeMeaning", "Unknown"))
    else:
        # Infer from half-life
        radionuclide = _infer_radionuclide(half_life_sec)

    # Injection time
    injection_time_str = getattr(radio, "RadiopharmaceuticalStartTime", None)
    if injection_time_str is None:
        raise ValueError(f"RadiopharmaceuticalStartTime missing for {patient_id}")

    # Scan time (series time or acquisition time)
    scan_time_str = getattr(ds, "AcquisitionTime", None) or getattr(ds, "SeriesTime", None)
    if scan_time_str is None:
        raise ValueError(f"AcquisitionTime/SeriesTime missing for {patient_id}")

    # Parse times
    series_date = getattr(ds, "SeriesDate", None) or getattr(ds, "StudyDate", "19000101")
    injection_time = _parse_dicom_datetime(series_date, str(injection_time_str))
    scan_time = _parse_dicom_datetime(series_date, str(scan_time_str))

    # Handle midnight crossing
    if scan_time < injection_time:
        scan_time += datetime.timedelta(days=1)

    uptake_time_sec = (scan_time - injection_time).total_seconds()

    # Decay factor
    decay_factor = 2 ** (-uptake_time_sec / half_life_sec)

    # Scanner info
    manufacturer = str(getattr(ds, "Manufacturer", "Unknown"))
    manufacturer_model = str(getattr(ds, "ManufacturerModelName", "Unknown"))
    software_version = str(getattr(ds, "SoftwareVersions", None))

    # Geometry
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    pixel_spacing_mm = (float(pixel_spacing[0]), float(pixel_spacing[1]))
    slice_thickness_mm = float(getattr(ds, "SliceThickness", 1.0))

    # Rescale parameters (for converting stored pixel values to activity)
    rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    # UIDs
    series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
    study_uid = str(getattr(ds, "StudyInstanceUID", ""))

    return PETMetadata(
        patient_id=patient_id,
        patient_weight_kg=patient_weight_kg,
        patient_height_m=patient_height_m,
        patient_sex=patient_sex,
        patient_age=patient_age,
        injected_dose_bq=injected_dose_bq,
        injection_time=injection_time,
        scan_time=scan_time,
        half_life_sec=half_life_sec,
        radionuclide=radionuclide,
        manufacturer=manufacturer,
        manufacturer_model=manufacturer_model,
        software_version=software_version,
        pixel_spacing_mm=pixel_spacing_mm,
        slice_thickness_mm=slice_thickness_mm,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        series_uid=series_uid,
        study_uid=study_uid,
        uptake_time_sec=uptake_time_sec,
        decay_factor=decay_factor,
    )


def compute_suv_bw(
    pixel_array: np.ndarray,
    meta: PETMetadata,
) -> np.ndarray:
    """Convert PET pixel values to SUV body-weight (SUV_bw).

    SUV_bw = activity_concentration / (decay_corrected_dose / patient_weight)

    Parameters
    ----------
    pixel_array : np.ndarray
        Raw PET pixel values (stored values, not yet rescaled).
    meta : PETMetadata
        Metadata from extract_pet_metadata().

    Returns
    -------
    np.ndarray
        SUV_bw values (same shape as pixel_array), dtype float64.
    """
    # Convert stored pixel values to activity concentration (Bq/mL)
    activity = pixel_array.astype(np.float64) * meta.rescale_slope + meta.rescale_intercept

    # Decay-correct the injected dose to scan time
    decay_corrected_dose = meta.injected_dose_bq * meta.decay_factor

    # SUV_bw = activity / (dose / weight)
    # Weight in grams (DICOM stores kg, SUV convention uses g)
    weight_g = meta.patient_weight_kg * 1000.0

    suv = activity / (decay_corrected_dose / weight_g)

    return suv


def compute_sul(
    suv_bw: np.ndarray,
    meta: PETMetadata,
) -> np.ndarray | None:
    """Convert SUV_bw to SUL (lean body mass normalised).

    Uses the Janmahasatian (2005) formula for lean body mass, as specified
    by PERCIST 1.0.

    Parameters
    ----------
    suv_bw : np.ndarray
        SUV body-weight values.
    meta : PETMetadata
        Metadata (needs weight, height, sex).

    Returns
    -------
    np.ndarray or None
        SUL values, or None if height/sex unavailable.
    """
    if meta.patient_height_m is None or meta.patient_sex is None:
        return None

    lbm = lean_body_mass_janmahasatian(
        weight_kg=meta.patient_weight_kg,
        height_m=meta.patient_height_m,
        sex=meta.patient_sex,
    )

    if lbm is None or lbm <= 0:
        return None

    # SUL = SUV_bw * (weight / LBM)
    # Because SUV_bw = activity / (dose / weight), and SUL = activity / (dose / LBM)
    # => SUL = SUV_bw * (LBM / weight)  ... wait, that's the wrong direction
    # Actually: SUL = activity / (dose / LBM) = (activity / (dose / weight)) * (weight / LBM)
    #         ... no. Let me be careful.
    # SUV_bw = activity * weight / dose
    # SUL    = activity * LBM   / dose
    # => SUL = SUV_bw * (LBM / weight)
    sul = suv_bw * (lbm / meta.patient_weight_kg)

    return sul


def lean_body_mass_janmahasatian(
    weight_kg: float,
    height_m: float,
    sex: str,
) -> float | None:
    """Lean body mass via Janmahasatian (2005) formula.

    LBM_male   = 9270 * weight / (6680 + 216 * BMI)
    LBM_female = 9270 * weight / (8780 + 244 * BMI)

    Parameters
    ----------
    weight_kg : float
    height_m : float
    sex : str
        'M' or 'F'

    Returns
    -------
    float or None
        Lean body mass in kg. None if sex not M/F.
    """
    if height_m <= 0:
        return None

    bmi = weight_kg / (height_m ** 2)

    sex_upper = sex.upper().strip() if sex else ""
    if sex_upper.startswith("M"):
        return 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    elif sex_upper.startswith("F"):
        return 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)
    else:
        return None


def dicom_series_to_suv_sitk(
    dicom_dir: str,
    meta: PETMetadata,
) -> sitk.Image:
    """Read a DICOM PET series and convert to SUV_bw.

    Pixel data is read via pydicom on a per-slice basis with each slice's own
    RescaleSlope/Intercept applied. SimpleITK is used only for slice ordering
    (GDCM z-sort) and for the output image's spatial metadata.

    The previous implementation called sitk.ImageSeriesReader.Execute() to read
    pixels, then applied compute_suv_bw — which itself applies RescaleSlope/
    Intercept. GDCM also applies rescale during read, so the rescale was being
    applied twice, producing SUVs inflated by ~slope× (validation showed
    77–640% errors across Siemens/GE; see osf/amendment_log.md Amendment 3).
    Reading via pydicom per slice also correctly handles Siemens-style per-slice
    rescale variation, which the old scalar-from-first-slice path ignored.

    Parameters
    ----------
    dicom_dir : str
        Directory containing DICOM files for one PET series.
    meta : PETMetadata
        Pre-extracted series metadata (used for weight/dose/decay; NOT for rescale).

    Returns
    -------
    sitk.Image
        3D SUV_bw image with spatial metadata copied from the source series.
    """
    if pydicom is None:
        raise ImportError("pydicom is required for dicom_series_to_suv_sitk")
    if sitk is None:
        raise ImportError("SimpleITK is required for dicom_series_to_suv_sitk")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])

    # Per-slice pydicom reads with per-slice rescale.
    slices = []
    for f in file_names:
        d = pydicom.dcmread(f)
        try:
            raw = d.pixel_array
        except (AttributeError, NotImplementedError) as e:
            raise ValueError(f"Cannot read pixel data from {f}: {e}")
        rs = float(getattr(d, "RescaleSlope", 1.0))
        ri = float(getattr(d, "RescaleIntercept", 0.0))
        slices.append(raw.astype(np.float64) * rs + ri)

    if not slices:
        raise ValueError(f"No slices with pixel data in {dicom_dir}")

    activity_3d = np.stack(slices, axis=0)  # Bq/mL

    # SUV_bw = activity * weight_g / decay_corrected_dose
    decay_corrected_dose = meta.injected_dose_bq * meta.decay_factor
    weight_g = meta.patient_weight_kg * 1000.0
    suv_array = activity_3d * weight_g / decay_corrected_dose

    # Spatial metadata via SimpleITK (origin, spacing, direction)
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    suv_image = sitk.GetImageFromArray(suv_array)
    suv_image.CopyInformation(image)

    return suv_image


# --- Quality control ---


def qc_suv_range(suv_image: sitk.Image, patient_id: str) -> dict:
    """Quality-check SUV values per pre-registration §3.9.

    Returns
    -------
    dict
        Keys: 'pass', 'suvmax', 'has_negative', 'has_zero', 'flagged_high'
    """
    arr = sitk.GetArrayFromImage(suv_image)
    suvmax = float(arr.max())
    has_negative = bool((arr < 0).any())
    has_zero_in_mask = False  # checked separately with mask
    flagged_high = suvmax > 50.0

    passed = not has_negative
    if flagged_high:
        # Not an automatic fail, but flagged for manual review
        pass

    return {
        "patient_id": patient_id,
        "pass": passed,
        "suvmax": suvmax,
        "has_negative": has_negative,
        "flagged_high": flagged_high,
    }


# --- Private helpers ---


def _get_float(ds, tag: str) -> float | None:
    val = getattr(ds, tag, None)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_dicom_datetime(date_str: str, time_str: str) -> datetime.datetime:
    """Parse DICOM date (YYYYMMDD) and time (HHMMSS.ffffff) into datetime."""
    date_str = date_str.strip()
    time_str = time_str.strip()

    # Handle fractional seconds
    if "." in time_str:
        time_parts = time_str.split(".")
        time_main = time_parts[0].ljust(6, "0")
        frac = time_parts[1][:6].ljust(6, "0")
        time_str = f"{time_main}.{frac}"
        fmt = "%Y%m%d%H%M%S.%f"
    else:
        time_str = time_str.ljust(6, "0")
        fmt = "%Y%m%d%H%M%S"

    return datetime.datetime.strptime(date_str + time_str, fmt)


def _infer_radionuclide(half_life_sec: float) -> str:
    """Infer radionuclide from half-life if CodeSequence is missing."""
    for name, hl in HALF_LIVES.items():
        if abs(half_life_sec - hl) / hl < 0.01:  # within 1%
            return name
    return "Unknown"
