"""Phase 2 Colab preflight check (Amendment 6, both cohorts).

Walks the Drive-mounted directories and reports per-input availability for
the AutoPET-III primary subsample (50 series) and the AutoPET-I sensitivity
calibration cohort (92 patients). Catches missing SUV / SEG / DICOM before
any expensive run kicks off, so we don't burn an hour of Colab time on a
trivially-fixable layout problem.

Outputs:
    {OUT_DIR}/phase2_preflight_autopet_iii.csv
    {OUT_DIR}/phase2_preflight_autopet_i.csv
    {OUT_DIR}/phase2_preflight_summary.json

Usage on Colab:
    !python scripts/phase2_preflight.py \\
        --autopet-iii-suv-dir "/content/drive/MyDrive/P79 Data/autopet_iii/paired_inputs" \\
        --autopet-iii-seg-dir "/content/drive/MyDrive/P79 Data/autopet_iii/segmentations" \\
        --autopet-iii-dicom-dir "/content/drive/MyDrive/P79 Data/autopet_iii/dicom" \\
        --autopet-i-suv-dir "/content/drive/MyDrive/P79 Data/autopet_i/extracted" \\
        --autopet-i-seg-dir "/content/drive/MyDrive/P79 Data/autopet_i/extracted" \\
        --out-dir "/content/drive/MyDrive/P79 Data/phase2"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSAMPLE = REPO_ROOT / "results/tables/section_3_5_phase2_autopet_iii_subsample.csv"
DEFAULT_AUTOPET_I_SPLITS = REPO_ROOT / "data/processed/autopet_i_splits.parquet"


def check_autopet_iii(
    subsample_csv: Path,
    suv_dir: Path,
    seg_dir: Path,
    dicom_dir: Path | None,
) -> pd.DataFrame:
    """For each of the 50 stratified-subsample series, check input readiness.

    Drive layout for AutoPET-III DICOM is heterogeneous: per-series UIDs appear
    either as extracted directories `{dicom_dir}/{series_uid}/*.dcm` (~200-450
    files per PT series) or as zipped `{dicom_dir}/{series_uid}.zip` archives.
    The preflight reports both -- the regen step (Cell 5) unzips on demand.
    """
    sub = pd.read_csv(subsample_csv)
    rows = []
    for _, r in sub.iterrows():
        suid = r["series_uid"]
        suv_path = suv_dir / f"{suid}_0001.nii.gz"
        seg_path = seg_dir / f"{suid}.nii.gz"
        dicom_subdir = (dicom_dir / suid) if dicom_dir else None
        dicom_zip = (dicom_dir / f"{suid}.zip") if dicom_dir else None
        n_dcm = (
            len(list(dicom_subdir.glob("*.dcm")))
            if (dicom_subdir and dicom_subdir.exists()) else 0
        )
        # Some Drive layouts use lowercase or no extension on some DICOM files;
        # also accept any file inside the dir as a DICOM proxy if .dcm count is 0.
        if dicom_subdir and dicom_subdir.exists() and n_dcm == 0:
            n_dcm = sum(1 for _ in dicom_subdir.iterdir() if _.is_file())
        zip_present = bool(dicom_zip and dicom_zip.exists())
        zip_size = dicom_zip.stat().st_size if zip_present else 0
        # Heuristic: real DICOM-content zips are >1 MB; smaller zips are likely
        # SEG-only or empty (already extracted to /segmentations/).
        zip_has_content = zip_present and zip_size > 1_000_000
        dir_present = n_dcm > 0
        rows.append({
            "case_id": r["case_id"],
            "series_uid": suid,
            "vendor": r["vendor"],
            "radionuclide": r["radionuclide"],
            "suv_present": suv_path.exists(),
            "seg_present": seg_path.exists(),
            "dicom_dir_present": dir_present,
            "dicom_zip_present": zip_has_content,
            "dicom_n_files": n_dcm,
            "dicom_zip_size_bytes": zip_size,
            "needs_dicom_unzip": (not dir_present) and zip_has_content,
            "needs_suv_regen": (not suv_path.exists()) and (dir_present or zip_has_content),
            "blocked": (
                (not suv_path.exists() and not (dir_present or zip_has_content))
                or (not seg_path.exists())
            ),
            "suv_path": str(suv_path),
            "seg_path": str(seg_path),
            "dicom_dir": str(dicom_subdir) if dicom_subdir else "",
            "dicom_zip": str(dicom_zip) if dicom_zip else "",
        })
    return pd.DataFrame(rows)


def check_autopet_i(
    splits_parquet: Path,
    suv_dir: Path,
    seg_dir: Path,
) -> pd.DataFrame:
    """For each of the 92 calibration patients, check input readiness in the FDAT layout.

    Layout: {suv_dir}/{patient_id}/{study_date}/{SUV,SEG}.nii.gz
    A patient may have multiple study_date subdirs; pick the lexicographically-first.
    """
    splits = pd.read_parquet(splits_parquet)
    cal = splits[splits["split"] == "calibration"].reset_index(drop=True)
    rows = []
    for _, r in cal.iterrows():
        pid = r["patient_id"]
        patient_dir = suv_dir / pid
        study_dirs = sorted(d for d in patient_dir.glob("*") if d.is_dir()) if patient_dir.exists() else []
        if not study_dirs:
            rows.append({
                "patient_id": pid, "vendor": r["vendor"], "tracer_category": r["tracer_category"],
                "study_date": "", "suv_present": False, "seg_present": False,
                "blocked": True, "suv_path": "", "seg_path": "",
            })
            continue
        sdir = study_dirs[0]
        suv_path = sdir / "SUV.nii.gz"
        seg_path = (seg_dir / pid / sdir.name / "SEG.nii.gz")
        rows.append({
            "patient_id": pid, "vendor": r["vendor"], "tracer_category": r["tracer_category"],
            "study_date": sdir.name,
            "suv_present": suv_path.exists(),
            "seg_present": seg_path.exists(),
            "blocked": not (suv_path.exists() and seg_path.exists()),
            "suv_path": str(suv_path), "seg_path": str(seg_path),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsample-csv", type=Path, default=DEFAULT_SUBSAMPLE)
    ap.add_argument("--splits-parquet", type=Path, default=DEFAULT_AUTOPET_I_SPLITS)
    ap.add_argument("--autopet-iii-suv-dir", type=Path, required=True)
    ap.add_argument("--autopet-iii-seg-dir", type=Path, required=True)
    ap.add_argument("--autopet-iii-dicom-dir", type=Path, default=None)
    ap.add_argument("--autopet-i-suv-dir", type=Path, required=True)
    ap.add_argument("--autopet-i-seg-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    iii = check_autopet_iii(
        args.subsample_csv, args.autopet_iii_suv_dir, args.autopet_iii_seg_dir,
        args.autopet_iii_dicom_dir,
    )
    i = check_autopet_i(args.splits_parquet, args.autopet_i_suv_dir, args.autopet_i_seg_dir)

    iii_csv = args.out_dir / "phase2_preflight_autopet_iii.csv"
    i_csv = args.out_dir / "phase2_preflight_autopet_i.csv"
    iii.to_csv(iii_csv, index=False)
    i.to_csv(i_csv, index=False)

    summary = {
        "autopet_iii_primary": {
            "n_total": int(len(iii)),
            "n_ready": int((~iii["blocked"]).sum()),
            "n_blocked": int(iii["blocked"].sum()),
            "n_need_suv_regen": int(iii["needs_suv_regen"].sum()),
            "n_need_dicom_unzip": int(iii["needs_dicom_unzip"].sum()),
            "n_suv_present": int(iii["suv_present"].sum()),
            "n_seg_present": int(iii["seg_present"].sum()),
            "n_dicom_dir_present": int(iii["dicom_dir_present"].sum()),
            "n_dicom_zip_present": int(iii["dicom_zip_present"].sum()),
            "blocked_examples": iii.loc[iii["blocked"], ["case_id", "series_uid"]]
                .head(5).to_dict(orient="records"),
        },
        "autopet_i_sensitivity": {
            "n_total": int(len(i)),
            "n_ready": int((~i["blocked"]).sum()),
            "n_blocked": int(i["blocked"].sum()),
            "n_suv_present": int(i["suv_present"].sum()),
            "n_seg_present": int(i["seg_present"].sum()),
            "blocked_examples": i.loc[i["blocked"], ["patient_id"]].head(5).to_dict(orient="records"),
        },
    }
    summary_path = args.out_dir / "phase2_preflight_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Phase 2 preflight ===")
    print()
    print("AutoPET-III primary subsample (50 series, DICOM-grounded):")
    s3 = summary["autopet_iii_primary"]
    print(f"  ready:               {s3['n_ready']}/{s3['n_total']}")
    print(f"  blocked:             {s3['n_blocked']}")
    print(f"  needs SUV regen:     {s3['n_need_suv_regen']} (have DICOM but no paired SUV NIfTI)")
    print(f"  needs DICOM unzip:   {s3['n_need_dicom_unzip']} (zipped, no extracted dir yet)")
    print(f"  SUV present:         {s3['n_suv_present']}")
    print(f"  SEG present:         {s3['n_seg_present']}")
    print(f"  DICOM dir present:   {s3['n_dicom_dir_present']}")
    print(f"  DICOM zip present:   {s3['n_dicom_zip_present']}")
    print()
    print("AutoPET-I sensitivity calibration cohort (92 patients, literature-default):")
    s1 = summary["autopet_i_sensitivity"]
    print(f"  ready:           {s1['n_ready']}/{s1['n_total']}")
    print(f"  blocked:         {s1['n_blocked']}")
    print(f"  SUV present:     {s1['n_suv_present']}")
    print(f"  SEG present:     {s1['n_seg_present']}")
    print()
    print(f"Wrote {iii_csv}")
    print(f"Wrote {i_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
