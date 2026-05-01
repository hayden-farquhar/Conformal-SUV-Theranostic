"""Validate SUV pipeline against an independent reference implementation.

Per pre-reg §3.3 (with Amendment 2): SUV pipeline validation is performed on
AutoPET-III DICOM. Target: <1% voxel-wise relative difference between our
production pipeline (suv_conversion.py) and an independent reference written here.

The reference uses only pydicom + numpy — no SimpleITK, no shared code with
suv_conversion.py — implementing Kinahan & Fletcher (2010) SUV_bw from scratch.
This is a true differential-implementation check: a bug in either pipeline that
both happen to share would not be caught, but any bug unique to one pipeline
will produce a >1% delta.

CLI:
    python -m src.preprocess.validate_suv_pipeline \\
        --data-dir /path/to/autopet_iii \\
        --series-list series_uids.txt \\
        --out suv_validation_report.csv

Importable:
    from src.preprocess.validate_suv_pipeline import validate_batch
    df = validate_batch(uids, data_dir, out_csv)
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

try:
    from src.preprocess.suv_conversion import (
        extract_pet_metadata,
        dicom_series_to_suv_sitk,
    )
except ImportError:
    # Standalone mode (e.g. uploaded alongside suv_conversion.py to Colab /content)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, '/content')
    from suv_conversion import extract_pet_metadata, dicom_series_to_suv_sitk

import SimpleITK as sitk


def compute_suv_reference(series_dir: str) -> tuple[np.ndarray, dict]:
    """Independent SUV computation per Kinahan & Fletcher (2010).

    Reads DICOM PET series using only pydicom + numpy. Sorts slices by
    ImagePositionPatient[2] (z). Applies per-slice rescale slope/intercept
    to convert stored values to Bq/mL, then SUV_bw = activity * weight_g / decay_dose.
    """
    dcm_files = sorted(glob.glob(os.path.join(series_dir, '*.dcm')))
    if not dcm_files:
        # Some TCIA series have no extension on files
        dcm_files = sorted(
            f for f in glob.glob(os.path.join(series_dir, '*'))
            if os.path.isfile(f) and not f.lower().endswith(('.txt', '.json', '.xml'))
        )
    if not dcm_files:
        raise FileNotFoundError(f'No DICOM files found in {series_dir}')

    ds0 = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)

    weight_kg = float(ds0.PatientWeight)
    weight_g = weight_kg * 1000.0

    radio = ds0.RadiopharmaceuticalInformationSequence[0]
    injected_dose = float(radio.RadionuclideTotalDose)
    half_life = float(radio.RadionuclideHalfLife)

    inj_time_str = str(radio.RadiopharmaceuticalStartTime)
    scan_time_str = str(getattr(ds0, 'AcquisitionTime', None) or ds0.SeriesTime)
    series_date = str(getattr(ds0, 'SeriesDate', None) or ds0.StudyDate)

    inj_time = _parse_dt(series_date, inj_time_str)
    scan_time = _parse_dt(series_date, scan_time_str)
    if scan_time < inj_time:
        scan_time += datetime.timedelta(days=1)

    uptake_sec = (scan_time - inj_time).total_seconds()
    decay_factor = 2.0 ** (-uptake_sec / half_life)
    decay_corrected_dose = injected_dose * decay_factor

    slices = []
    positions = []
    for f in dcm_files:
        d = pydicom.dcmread(f)
        try:
            arr = d.pixel_array
        except (AttributeError, NotImplementedError):
            continue
        rs = float(getattr(d, 'RescaleSlope', 1.0))
        ri = float(getattr(d, 'RescaleIntercept', 0.0))
        activity = arr.astype(np.float64) * rs + ri
        slices.append(activity)
        ipp = getattr(d, 'ImagePositionPatient', None)
        positions.append(float(ipp[2]) if ipp is not None else float(len(slices)))

    if not slices:
        raise ValueError(f'No slices with pixel data in {series_dir}')

    sort_idx = np.argsort(positions)
    activity_3d = np.stack([slices[i] for i in sort_idx], axis=0)
    suv_3d = activity_3d * weight_g / decay_corrected_dose

    metadata = {
        'patient_weight_kg': weight_kg,
        'injected_dose_bq': injected_dose,
        'half_life_sec': half_life,
        'uptake_time_sec': uptake_sec,
        'decay_factor': decay_factor,
        'manufacturer': str(getattr(ds0, 'Manufacturer', 'Unknown')),
        'manufacturer_model': str(getattr(ds0, 'ManufacturerModelName', 'Unknown')),
        'n_slices': len(slices),
    }
    return suv_3d, metadata


def validate_one_series(series_dir: str, series_uid: str | None = None) -> dict:
    """Run both pipelines on one series, return comparison metrics.

    Status values: pass, fail_above_1pct, fail_shape_mismatch, error
    """
    series_uid = series_uid or os.path.basename(series_dir.rstrip('/'))
    result: dict = {'series_uid': series_uid, 'status': 'unknown', 'error': None}

    try:
        ref_suv, ref_meta = compute_suv_reference(series_dir)

        # Production pipeline (suv_conversion.py)
        sample_dcm = sorted(
            f for f in glob.glob(os.path.join(series_dir, '*'))
            if os.path.isfile(f) and not f.lower().endswith(('.txt', '.json', '.xml'))
        )[0]
        our_meta = extract_pet_metadata(sample_dcm)
        our_suv_image = dicom_series_to_suv_sitk(series_dir, our_meta)
        our_suv = sitk.GetArrayFromImage(our_suv_image)

        if our_suv.shape != ref_suv.shape:
            result['status'] = 'fail_shape_mismatch'
            result['error'] = f'shape ours {our_suv.shape} vs ref {ref_suv.shape}'
            return result

        # Voxel-wise comparison; relative diff vs reference SUVmax avoids div-by-zero in air
        abs_diff = np.abs(our_suv - ref_suv)
        ref_max = float(ref_suv.max())
        our_max = float(our_suv.max())
        denom = max(ref_max, 1e-9)
        rel_diff_pct = abs_diff / denom * 100.0

        # Cross-check the metadata extraction agrees with the reference reader
        meta_consistent = (
            abs(float(our_meta.patient_weight_kg) - ref_meta['patient_weight_kg']) < 0.01
            and abs(float(our_meta.injected_dose_bq) - ref_meta['injected_dose_bq']) < 1.0
            and abs(float(our_meta.uptake_time_sec) - ref_meta['uptake_time_sec']) < 1.0
        )

        result.update({
            'manufacturer': ref_meta['manufacturer'],
            'model': ref_meta['manufacturer_model'],
            'n_slices': ref_meta['n_slices'],
            'shape': str(our_suv.shape),
            'patient_weight_kg': ref_meta['patient_weight_kg'],
            'injected_dose_bq': ref_meta['injected_dose_bq'],
            'half_life_sec': ref_meta['half_life_sec'],
            'uptake_time_sec': ref_meta['uptake_time_sec'],
            'decay_factor': ref_meta['decay_factor'],
            'metadata_consistent': meta_consistent,
            'our_suvmax': our_max,
            'ref_suvmax': ref_max,
            'max_abs_diff_suv': float(abs_diff.max()),
            'max_rel_diff_pct': float(rel_diff_pct.max()),
            'mean_abs_diff_suv': float(abs_diff.mean()),
            'mean_rel_diff_pct': float(rel_diff_pct.mean()),
            'status': 'pass' if rel_diff_pct.max() < 1.0 else 'fail_above_1pct',
        })
        return result

    except Exception as e:
        result['status'] = 'error'
        result['error'] = f'{type(e).__name__}: {e}'
        return result


def resolve_series_path(uid: str, data_dir: str) -> tuple[str, callable | None]:
    """Locate a series on disk in the heterogeneous AutoPET-III storage.

    Tries {data_dir}/{uid}/ first, then {data_dir}/{uid}.zip (extracted to a
    temp dir and returned with a cleanup callback).
    """
    dir_path = os.path.join(data_dir, uid)
    zip_path = os.path.join(data_dir, f'{uid}.zip')

    if os.path.isdir(dir_path) and os.listdir(dir_path):
        # If contents are inside a single subdir (TCIA convention), descend one level
        contents = [c for c in os.listdir(dir_path) if not c.startswith('.')]
        if len(contents) == 1 and os.path.isdir(os.path.join(dir_path, contents[0])):
            return os.path.join(dir_path, contents[0]), None
        return dir_path, None

    if os.path.exists(zip_path):
        tmp = tempfile.mkdtemp(prefix=f'val_{uid[-12:]}_')
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        contents = [c for c in os.listdir(tmp) if not c.startswith('.')]
        target = tmp
        if len(contents) == 1 and os.path.isdir(os.path.join(tmp, contents[0])):
            target = os.path.join(tmp, contents[0])
        return target, lambda: shutil.rmtree(tmp, ignore_errors=True)

    raise FileNotFoundError(f'Series {uid}: neither {dir_path} nor {zip_path} exists')


def validate_batch(
    series_uids: list[str],
    data_dir: str,
    out_csv: str,
) -> pd.DataFrame:
    """Validate multiple series, write CSV report, print summary."""
    rows = []
    n = len(series_uids)
    for i, uid in enumerate(series_uids, 1):
        print(f'[{i}/{n}] {uid}')
        cleanup = None
        try:
            path, cleanup = resolve_series_path(uid, data_dir)
            row = validate_one_series(path, series_uid=uid)
        except FileNotFoundError as e:
            row = {'series_uid': uid, 'status': 'not_found', 'error': str(e)}
        finally:
            if cleanup:
                cleanup()
        rows.append(row)

        s = row.get('status', 'unknown')
        if s == 'pass':
            print(f'  PASS  max_rel_diff={row["max_rel_diff_pct"]:.5f}% '
                  f'({row["manufacturer"]} {row["model"]})')
        else:
            print(f'  {s.upper()}  {row.get("error", "")}')

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print()
    print('=' * 68)
    print(f'Validated {len(df)} series')
    print(f'Status: {df["status"].value_counts().to_dict()}')
    passed = df[df['status'] == 'pass']
    if len(passed) > 0:
        print(f'Among {len(passed)} passing series:')
        print(f'  max rel diff: {passed["max_rel_diff_pct"].max():.5f}% '
              f'(threshold: 1.0%)')
        print(f'  median rel diff: {passed["max_rel_diff_pct"].median():.5f}%')
        if 'manufacturer' in passed.columns:
            print(f'  Coverage by manufacturer: '
                  f'{passed["manufacturer"].value_counts().to_dict()}')
            print(f'  Coverage by model: '
                  f'{passed["model"].value_counts().to_dict()}')
    if (df['status'] == 'pass').all():
        print()
        print('PRE-REG §3.3 VALIDATION: PASSED')
        print('Document this result in osf/data_snapshot_log.md.')
    else:
        print()
        print('PRE-REG §3.3 VALIDATION: NOT PASSED')
        print('Investigate failures before treating suv_conversion.py as production-validated.')
    print(f'Report saved: {out_csv}')
    return df


def _parse_dt(date_str: str, time_str: str) -> datetime.datetime:
    date_str = date_str.strip()
    time_str = time_str.strip()
    if '.' in time_str:
        t_main, t_frac = time_str.split('.', 1)
        t_str = t_main.ljust(6, '0') + '.' + t_frac[:6].ljust(6, '0')
        return datetime.datetime.strptime(date_str + t_str, '%Y%m%d%H%M%S.%f')
    return datetime.datetime.strptime(date_str + time_str.ljust(6, '0'), '%Y%m%d%H%M%S')


def main():
    ap = argparse.ArgumentParser(
        description='Validate suv_conversion.py vs independent reference (pre-reg §3.3 + Amendment 2)'
    )
    ap.add_argument('--data-dir', required=True,
                    help='AutoPET-III download directory containing {uid}/ dirs and/or {uid}.zip files')
    ap.add_argument('--series-list', required=True,
                    help='Path to text file with one SeriesInstanceUID per line (PT modality only)')
    ap.add_argument('--out', default='suv_validation_report.csv',
                    help='Output CSV path')
    args = ap.parse_args()

    with open(args.series_list) as f:
        uids = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    print(f'Loaded {len(uids)} series UIDs from {args.series_list}')
    validate_batch(uids, args.data_dir, args.out)


if __name__ == '__main__':
    main()
