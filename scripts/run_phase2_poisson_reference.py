"""Phase 2 Poisson-noise reference driver (Amendment 6 implementation).

Iterates a cohort manifest, runs the per-case Phase 2 pipeline, and writes
two parquets:

    {OUT_PREFIX}_replicates.parquet   -- long-form replicate trace
                                         (case_id, lesion_id, dose_fraction,
                                          replicate, suvmax, suvpeak, suvmean)
    {OUT_PREFIX}_wcv.parquet          -- per-(lesion, dose) wCV summary

Two cohort modes (Amendment 6 sec 6a):

    --mode autopet_iii_primary
        Subsample manifest: results/tables/section_3_5_phase2_autopet_iii_subsample.csv
        SUV NIfTI dir:      $WORK_DIR/autopet_iii/paired_inputs/{series_uid}_0001.nii.gz
        SEG NIfTI dir:      $WORK_DIR/autopet_iii/segmentations/{series_uid}.nii.gz
        DICOM dir:          $WORK_DIR/autopet_iii/dicom/{series_uid}/
        Acquisition params: extracted per series from DICOM (`src.preprocess.suv_conversion`)

    --mode autopet_i_sensitivity
        Cohort manifest: data/processed/autopet_i_splits.parquet (split=='calibration')
        SUV NIfTI dir:   $WORK_DIR/autopet_i/extracted/{patient_id}/{study_date}/SUV.nii.gz
        SEG NIfTI dir:   $WORK_DIR/autopet_i/extracted/{patient_id}/{study_date}/SEG.nii.gz
        Acquisition params: AutoPET-I literature defaults
                            (`src.testretest.defaults.autopet_i_defaults_for_patient`)
                            with per-patient weight from clinical_metadata.csv if available.

`$WORK_DIR` is whatever local or networked directory holds the raw cohort
data (DICOM / SUV NIfTI / SEG NIfTI). On Colab the conventional choice is
the mounted Google Drive root; locally, any directory with sufficient disk
space works. Pass the resolved paths to the corresponding CLI flags.

Resume support: rows already written to {OUT_PREFIX}_replicates.parquet are
skipped on re-run, so long sessions can pick up after a disconnection.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN sec 3.5
Amendment 6.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.testretest.defaults import (
    AcquisitionParams,
    autopet_i_defaults_for_patient,
    lookup_scanner_calibration,
)
from src.testretest.phase2_pipeline import CaseInputs, reduce_to_wcv, run_case


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Lazy import: nibabel is only needed inside Colab production runs."""
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata())
    # nibabel returns (x, y, z); we use (z, y, x) consistently across the project
    arr = np.transpose(arr, (2, 1, 0))
    sx, sy, sz = img.header.get_zooms()[:3]
    return arr, (float(sz), float(sy), float(sx))


def _load_autopet_iii_manifest(subsample_csv: Path) -> pd.DataFrame:
    return pd.read_csv(subsample_csv)


def _load_autopet_i_calibration_manifest(splits_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(splits_parquet)
    return df[df["split"] == "calibration"].reset_index(drop=True)


def _load_autopet_i_clinical_metadata(csv_path: Path | None) -> dict[str, float]:
    """Map patient_id -> patient_weight_kg, if a clinical CSV is provided.

    The AutoPET-I clinical CSV is shipped under CC BY 4.0 alongside the FDAT
    NIfTI release; if absent, all patients fall back to the cohort-typical
    default per Amendment 6.
    """
    if csv_path is None or not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    weight_cols = [c for c in df.columns if c.lower() in ("weight", "weight_kg", "patient_weight", "body_weight")]
    if not weight_cols:
        return {}
    id_cols = [c for c in df.columns if c.lower() in ("patient_id", "subject_id", "patient")]
    if not id_cols:
        return {}
    return dict(zip(df[id_cols[0]].astype(str), df[weight_cols[0]].astype(float)))


def _params_from_dicom(dicom_dir: Path) -> AcquisitionParams:
    """Extract DICOM-grounded acquisition parameters for one PT series.

    Wraps `extract_pet_metadata` from `src.preprocess.suv_conversion` (the §3.3
    validated extractor; reads dose / weight / decay from a single DICOM slice
    in the series). Adds frame_duration since PETMetadata doesn't expose it.
    """
    import pydicom

    from src.preprocess.suv_conversion import extract_pet_metadata

    pt_files = sorted(dicom_dir.glob("*.dcm"))
    if not pt_files:
        # Some Drive-extracted series have DICOMs without .dcm extension
        # (TCIA/NBIA quirk). Fall back to any regular file in the directory.
        pt_files = sorted(p for p in dicom_dir.iterdir() if p.is_file())
    if not pt_files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")
    first_dcm = str(pt_files[0])
    meta = extract_pet_metadata(first_dcm)

    # Frame duration: ActualFrameDuration is in ms, FrameDuration may also be in ms.
    # Some Siemens DICOMs put it in milliseconds; some scanners put scan time per bed
    # under FrameDuration as the alias. Fall back to literature value (120s) only as
    # a last resort with a loud warning.
    ds = pydicom.dcmread(first_dcm, stop_before_pixels=True)
    frame_dur_ms = float(getattr(ds, "ActualFrameDuration", getattr(ds, "FrameDuration", 0)) or 0)
    if frame_dur_ms <= 0:
        # ActualFrameDuration may be absent on whole-body series stitched from
        # multiple bed positions; try a few neighbouring slices before giving up.
        for cand in pt_files[1:6]:
            ds2 = pydicom.dcmread(str(cand), stop_before_pixels=True)
            frame_dur_ms = float(getattr(ds2, "ActualFrameDuration", getattr(ds2, "FrameDuration", 0)) or 0)
            if frame_dur_ms > 0:
                break
    if frame_dur_ms <= 0:
        raise ValueError(
            f"FrameDuration missing across first 5 slices of {dicom_dir} -- "
            "cannot compute DICOM-grounded acquisition parameters per Amendment 6."
        )
    # Amendment 7: per-scanner calibration_factor anchored to NEMA NU2 sensitivity
    # (substring lookup against ManufacturerModelName; cohort default for unknowns).
    calibration = lookup_scanner_calibration(getattr(meta, "manufacturer_model", None))
    return AcquisitionParams(
        injected_dose_bq=float(meta.injected_dose_bq),
        patient_weight_kg=float(meta.patient_weight_kg),
        decay_factor=float(meta.decay_factor),
        frame_duration_sec=frame_dur_ms / 1000.0,
        calibration_factor=calibration,
    )


def _resume_skiplist(out_replicates: Path) -> set[str]:
    """Return the set of (case_id, series_uid) keys already in the output parquet."""
    if not out_replicates.exists():
        return set()
    done = pd.read_parquet(out_replicates, columns=["case_id", "series_uid"]).drop_duplicates()
    return set(zip(done["case_id"], done["series_uid"]))


def _append_parquet(df: pd.DataFrame, path: Path) -> None:
    """Append a chunk by reading + concat + write (good enough for Phase 2 scale: <=100 cases)."""
    if df.empty:
        return
    if path.exists():
        prev = pd.read_parquet(path)
        df = pd.concat([prev, df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["autopet_iii_primary", "autopet_i_sensitivity"], required=True)
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="Output prefix; produces {prefix}_replicates.parquet and {prefix}_wcv.parquet")
    ap.add_argument("--subsample-csv", type=Path,
                    default=REPO_ROOT / "results/tables/section_3_5_phase2_autopet_iii_subsample.csv",
                    help="(autopet_iii_primary mode) stratified subsample manifest")
    ap.add_argument("--splits-parquet", type=Path,
                    default=REPO_ROOT / "data/processed/autopet_i_splits.parquet",
                    help="(autopet_i_sensitivity mode) splits manifest")
    ap.add_argument("--clinical-csv", type=Path, default=None,
                    help="(autopet_i_sensitivity mode) optional clinical CSV with per-patient weight")
    ap.add_argument("--suv-dir", type=Path, required=True,
                    help="Directory containing SUV NIfTI files (Drive-mounted on Colab)")
    ap.add_argument("--seg-dir", type=Path, required=True,
                    help="Directory containing SEG NIfTI files")
    ap.add_argument("--dicom-dir", type=Path, default=None,
                    help="(autopet_iii_primary mode) directory with per-series DICOM subfolders")
    ap.add_argument("--n-replicates", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N cases (debug / smoke test)")
    args = ap.parse_args()

    out_replicates = args.out_prefix.with_name(args.out_prefix.name + "_replicates.parquet")
    out_wcv = args.out_prefix.with_name(args.out_prefix.name + "_wcv.parquet")

    if args.mode == "autopet_iii_primary":
        manifest = _load_autopet_iii_manifest(args.subsample_csv)
        manifest = manifest.rename(columns={"case_id": "patient_id"}) if "case_id" in manifest.columns and "patient_id" not in manifest.columns else manifest
        cohort = "autopet_iii"
        params_source = "dicom"
    else:
        manifest = _load_autopet_i_calibration_manifest(args.splits_parquet)
        cohort = "autopet_i"
        params_source = "defaults"

    if args.limit:
        manifest = manifest.head(args.limit)

    skip = _resume_skiplist(out_replicates)
    weight_map = (
        _load_autopet_i_clinical_metadata(args.clinical_csv)
        if args.mode == "autopet_i_sensitivity" else {}
    )

    print(f"=== Phase 2 driver -- mode={args.mode}, n_cases={len(manifest)}, seed={args.seed} ===")
    print(f"out_replicates: {out_replicates}")
    print(f"out_wcv:        {out_wcv}")
    if skip:
        print(f"resuming: {len(skip)} (case, series) keys already in output, will skip")

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for i, row in manifest.iterrows():
        case_id = str(row.get("patient_id") or row.get("case_id"))
        series_uid = str(row.get("series_uid", "")) if args.mode == "autopet_iii_primary" else None

        if args.mode == "autopet_iii_primary":
            key = (case_id, series_uid)
            suv_path = args.suv_dir / f"{series_uid}_0001.nii.gz"
            seg_path = args.seg_dir / f"{series_uid}.nii.gz"
            dicom_subdir = args.dicom_dir / series_uid if args.dicom_dir else None
        else:
            # AutoPET-I FDAT layout: {patient_id}/{study_date}/{SUV,SEG}.nii.gz
            # If a patient has multiple study_date subdirs, pick the lexicographically-first
            # one (deterministic; aligns with the §3.4 split allocation which used the same rule).
            patient_dir = args.suv_dir / case_id
            study_dirs = sorted(d for d in patient_dir.glob("*") if d.is_dir()) if patient_dir.exists() else []
            if not study_dirs:
                series_uid = case_id
                key = (case_id, series_uid)
                suv_path = patient_dir / "SUV.nii.gz"  # will fail the existence check below
                seg_path = args.seg_dir / case_id / "SEG.nii.gz"
            else:
                study_date = study_dirs[0].name
                series_uid = f"{case_id}/{study_date}"  # composite key for resume tracking
                key = (case_id, series_uid)
                suv_path = study_dirs[0] / "SUV.nii.gz"
                seg_path = args.seg_dir / case_id / study_date / "SEG.nii.gz"
            dicom_subdir = None

        if key in skip:
            n_skipped += 1
            continue

        try:
            if not suv_path.exists() or not seg_path.exists():
                print(f"[{i+1}/{len(manifest)}] {case_id} -- SUV or SEG missing, skipping", flush=True)
                n_failed += 1
                continue

            suv_volume, voxel_spacing = _load_nifti(suv_path)
            seg_mask, _ = _load_nifti(seg_path)
            if seg_mask.shape != suv_volume.shape:
                print(f"[{i+1}/{len(manifest)}] {case_id} -- shape mismatch SUV {suv_volume.shape} vs SEG {seg_mask.shape}", flush=True)
                n_failed += 1
                continue

            if args.mode == "autopet_iii_primary":
                if dicom_subdir is None or not dicom_subdir.exists():
                    raise FileNotFoundError(f"DICOM dir missing for {series_uid}: {dicom_subdir}")
                params = _params_from_dicom(dicom_subdir)
            else:
                weight = weight_map.get(case_id)
                params = autopet_i_defaults_for_patient(patient_weight_kg=weight)

            inputs = CaseInputs(
                case_id=case_id, series_uid=series_uid,
                cohort=cohort, params_source=params_source,
                suv_volume=suv_volume, seg_mask=seg_mask,
                voxel_spacing_mm=voxel_spacing, params=params,
            )
            df = run_case(inputs, n_replicates=args.n_replicates, seed=args.seed)
            if df.empty:
                print(f"[{i+1}/{len(manifest)}] {case_id} -- no lesions >= 1mL after relabel; skipping", flush=True)
                continue
            _append_parquet(df, out_replicates)
            n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            print(f"[{i+1}/{len(manifest)}] {case_id} -- {df['lesion_id'].nunique()} lesions, "
                  f"{len(df)} replicate rows ({n_done} done, {rate:.2f} cases/s)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(manifest)}] {case_id} -- FAILED: {e}", flush=True)
            n_failed += 1
            continue

    # Always rebuild the wCV summary from the (cumulative) replicate parquet
    if out_replicates.exists():
        all_reps = pd.read_parquet(out_replicates)
        wcv = reduce_to_wcv(all_reps)
        wcv.to_parquet(out_wcv, index=False)
        print(f"\nwCV summary: {len(wcv)} (lesion x dose) rows across {wcv['case_id'].nunique()} cases -> {out_wcv}")
    print(f"\nDone: {n_done} processed, {n_skipped} skipped (resume), {n_failed} failed.")
    print(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
