"""Build unified study-level metadata table from AutoPET-III sources.

Merges the clinical TSV (patient-level: age, manufacturer, model, tracer,
contrast) with the NBIA digest XLSX (series-level: pixel spacing, slice
thickness, software version, image count, file size) into a single
per-study table with all CQR predictor variables.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§4.2)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_clinical_tsv(path: str | Path) -> pd.DataFrame:
    """Load the AutoPET-III clinical metadata TSV.

    Columns: case_identifier_number, study_date, Age_at_Imaging,
    Manufacturer, manufacturer_model_name, pet_radionuclide, ct_contrast_agent
    """
    df = pd.read_csv(path, sep="\t").copy()
    df = df.rename(columns={
        "case_identifier_number": "patient_id",
        "study_date": "study_date",
        "Age_at_Imaging": "age",
        "Manufacturer": "manufacturer",
        "manufacturer_model_name": "scanner_model",
        "pet_radionuclide": "tracer",
        "ct_contrast_agent": "ct_contrast",
    })
    # Normalise manufacturer names
    df["vendor"] = df["manufacturer"].map({
        "SIEMENS": "Siemens",
        "GE MEDICAL SYSTEMS": "GE",
    }).fillna("Other")
    return df


def load_nbia_digest(path: str | Path) -> pd.DataFrame:
    """Load the NBIA digest XLSX and extract per-study metadata.

    Returns one row per study (aggregating across CT/PT/SEG series).
    """
    df = pd.read_excel(path, engine="openpyxl")

    # Separate by modality
    pet_df = df[df["Modality"] == "PT"].copy()
    ct_df = df[df["Modality"] == "CT"].copy()

    # Per-study PET metadata
    pet_meta = pet_df[[
        "PatientID", "StudyInstanceUID", "SeriesInstanceUID",
        "Manufacturer", "ManufacturerModelName",
        "PixelSpacing(mm)-Row", "SliceThickness(mm)",
        "SoftwareVersions", "ImageCount",
    ]].rename(columns={
        "PatientID": "patient_id",
        "StudyInstanceUID": "study_uid",
        "SeriesInstanceUID": "pet_series_uid",
        "PixelSpacing(mm)-Row": "pet_pixel_spacing_mm",
        "SliceThickness(mm)": "pet_slice_thickness_mm",
        "SoftwareVersions": "pet_software_version",
        "ImageCount": "pet_image_count",
        "Manufacturer": "manufacturer_nbia",
        "ManufacturerModelName": "scanner_model_nbia",
    })

    # Per-study CT metadata
    ct_meta = ct_df[[
        "StudyInstanceUID", "SeriesInstanceUID",
        "PixelSpacing(mm)-Row", "SliceThickness(mm)", "ImageCount",
    ]].rename(columns={
        "StudyInstanceUID": "study_uid",
        "SeriesInstanceUID": "ct_series_uid",
        "PixelSpacing(mm)-Row": "ct_pixel_spacing_mm",
        "SliceThickness(mm)": "ct_slice_thickness_mm",
        "ImageCount": "ct_image_count",
    })

    # SEG series UIDs
    seg_df = df[df["Modality"] == "SEG"].copy()
    seg_meta = seg_df[[
        "StudyInstanceUID", "SeriesInstanceUID",
    ]].rename(columns={
        "StudyInstanceUID": "study_uid",
        "SeriesInstanceUID": "seg_series_uid",
    })

    # Merge on study_uid
    merged = pet_meta.merge(ct_meta, on="study_uid", how="left")
    merged = merged.merge(seg_meta, on="study_uid", how="left")

    return merged


def build_study_metadata(
    clinical_tsv_path: str | Path,
    nbia_digest_path: str | Path,
) -> pd.DataFrame:
    """Build the unified per-study metadata table.

    Merges clinical TSV and NBIA digest into a single DataFrame with
    all variables needed for CQR predictors and Mondrian stratification.

    Returns
    -------
    pd.DataFrame
        One row per study with columns for all CQR predictor variables.
    """
    clinical = load_clinical_tsv(clinical_tsv_path)
    nbia = load_nbia_digest(nbia_digest_path)

    # The clinical TSV has one row per study (patient_id + study_date combo)
    # The NBIA digest has one row per study (patient_id + study_uid combo)
    # Merge on patient_id — but there may be multiple studies per patient
    # Use study ordering to align (both sorted by patient then date)

    # Create a study index within each patient for both DataFrames
    clinical = clinical.sort_values(["patient_id", "study_date"]).reset_index(drop=True)
    clinical["study_idx"] = clinical.groupby("patient_id").cumcount()

    nbia = nbia.sort_values(["patient_id", "study_uid"]).reset_index(drop=True)
    nbia["study_idx"] = nbia.groupby("patient_id").cumcount()

    # Merge on patient_id + study_idx
    merged = clinical.merge(nbia, on=["patient_id", "study_idx"], how="inner")

    # Compute derived features
    # Voxel volume in mm³
    merged["pet_voxel_volume_mm3"] = (
        merged["pet_pixel_spacing_mm"] ** 2 * merged["pet_slice_thickness_mm"]
    )

    # Vendor (from clinical TSV, already normalised)
    # Tracer category
    merged["tracer_category"] = merged["tracer"].map({
        "68Ga": "PSMA",
        "18F": "FDG",
    }).fillna("Other")

    # Ensure string columns don't have mixed types
    for col in ["pet_software_version", "pet_series_uid", "ct_series_uid", "seg_series_uid", "study_uid"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).replace("nan", pd.NA)

    # Drop intermediate columns
    merged = merged.drop(columns=["study_idx", "manufacturer_nbia", "scanner_model_nbia"])

    return merged


def summarise_metadata(df: pd.DataFrame) -> None:
    """Print a summary of the study metadata table."""
    print(f"Studies: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"\nVendor distribution:")
    print(df["vendor"].value_counts().to_string(header=False))
    print(f"\nScanner model distribution:")
    print(df["scanner_model"].value_counts().to_string(header=False))
    print(f"\nTracer distribution:")
    print(df["tracer_category"].value_counts().to_string(header=False))
    print(f"\nPET pixel spacing (mm):")
    print(df["pet_pixel_spacing_mm"].describe().to_string())
    print(f"\nPET slice thickness (mm):")
    print(df["pet_slice_thickness_mm"].describe().to_string())
    print(f"\nAge:")
    print(df["age"].describe().to_string())
    print(f"\nCT contrast:")
    print(df["ct_contrast"].value_counts().to_string(header=False))


if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parents[2] / "data" / "raw" / "autopet_iii"
    clinical_path = base / "clinical_metadata.tsv"
    nbia_path = base / "nbia_digest.xlsx"

    if not clinical_path.exists() or not nbia_path.exists():
        print(f"Metadata files not found in {base}")
        sys.exit(1)

    df = build_study_metadata(clinical_path, nbia_path)
    summarise_metadata(df)

    # Save to interim
    out_path = Path(__file__).resolve().parents[2] / "data" / "interim" / "autopet_iii_study_metadata.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
