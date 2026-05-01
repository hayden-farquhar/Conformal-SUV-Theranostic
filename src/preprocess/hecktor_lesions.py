"""HECKTOR per-lesion feature extraction (Amendment 8 implementation).

Reads HECKTOR pre-SUV-converted PT NIfTI + multi-class GT SEG NIfTI, finds
connected components on the BINARY mask (GTVp ∪ GTVn merged so adjacent
primary-and-nodal components count as one lesion), and produces per-lesion
features matching the AutoPET-III parquet schema where possible.

Key Amendment-8 design decisions encoded:
  - Lesion definition: connected components on binary (GT_p > 0 OR GT_n > 0)
    with min volume ≥ 1 mL (pre-reg sec 3.2 unchanged).
  - lesion_class via voxel-majority within each component:
      pure GTVp  -> 1
      pure GTVn  -> 2
      mixed (both classes present, neither >50%) -> 3
      class-dominant (>=50% one class with the other present) -> dominant class
  - Softmax predictors: sentinel substitution per Amendment 8 §8c
      softmax_mean = 1.0, softmax_entropy = 0.0  (GT = perfect confidence)
  - Output schema parallels AutoPET-III parquet for downstream conformal code
    reuse, with HECKTOR-specific columns appended (centre_id, lesion_class,
    task1_patient, task2_patient, relapse, rfs_days, t/n/m_stage, hpv_status).

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN
Amendment 8: 2026-04-30
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from src.features.suvpeak import compute_suvpeak

# Pre-reg sec 3.2 (unchanged across amendments)
MIN_LESION_VOLUME_ML = 1.0

# Amendment 8 sec 8c sentinels (GT-defined lesions = perfect confidence)
GT_SOFTMAX_MEAN = 1.0
GT_SOFTMAX_ENTROPY = 0.0

# lesion_class encoding (matches P80 schema)
LESION_CLASS_GTVP = 1
LESION_CLASS_GTVN = 2
LESION_CLASS_MIXED = 3


@dataclass
class HecktorLesionFeatures:
    """One row in hecktor_lesions.parquet."""
    case_id: str
    lesion_id: int
    series_uid: str  # HECKTOR is single-time-point per patient; series_uid = patient_id
    suvmax: float
    suvmean: float
    suvpeak: float
    tlg: float
    volume_ml: float
    n_voxels: int
    surface_area_cm2: float  # pre-reg §4.2 predictor
    sphericity: float        # pre-reg §4.2 predictor
    centroid_0: float  # z (voxel)
    centroid_1: float  # y (voxel)
    centroid_2: float  # x (voxel)
    voxel_spacing_0: float  # z (mm)
    voxel_spacing_1: float  # y (mm)
    voxel_spacing_2: float  # x (mm)
    dataset: str  # 'hecktor'
    tracer: str   # 'FDG'
    vendor: str   # from per-centre lookup (Amendment 8 sec 8d)
    radionuclide: str  # 'F-18'
    softmax_mean: float    # = GT_SOFTMAX_MEAN per Amendment 8
    softmax_entropy: float  # = GT_SOFTMAX_ENTROPY per Amendment 8
    centre_id: int | None
    centre_name: str
    lesion_class: int  # 1=GTVp, 2=GTVn, 3=mixed (per Amendment 8 sec 8b)
    task1_patient: bool
    task2_patient: bool
    relapse: int | None
    rfs_days: float | None
    t_stage: str | None
    n_stage: str | None
    m_stage: str | None
    hpv_status: str | None


def classify_component(seg_multi: np.ndarray, component_mask: np.ndarray) -> int:
    """Voxel-majority class for a connected component on the multi-class SEG.

    Returns
    -------
    int
        1=GTVp, 2=GTVn, 3=mixed. A "mixed" label is returned when both classes
        are present AND neither has >=50% of the component voxels. If one class
        has >=50% of voxels, that class wins (the minority voxels are treated as
        annotation ambiguity at the GTVp/GTVn boundary, not as a "mixed lesion").
    """
    inside = seg_multi[component_mask]
    n_total = inside.size
    if n_total == 0:
        return LESION_CLASS_MIXED
    n_p = int((inside == 1).sum())
    n_n = int((inside == 2).sum())
    if n_p == 0 and n_n == 0:
        return LESION_CLASS_MIXED  # shouldn't happen but safe default
    if n_p == 0:
        return LESION_CLASS_GTVN
    if n_n == 0:
        return LESION_CLASS_GTVP
    if n_p / n_total >= 0.5:
        return LESION_CLASS_GTVP
    if n_n / n_total >= 0.5:
        return LESION_CLASS_GTVN
    return LESION_CLASS_MIXED


def _surface_area_cm2(mask: np.ndarray, voxel_spacing_mm: tuple[float, float, float]) -> float:
    """Voxel-face surface-area approximation (matches AutoPET pipeline)."""
    sz, sy, sx = voxel_spacing_mm
    m = mask.astype(np.int8)
    z_faces = int(np.abs(np.diff(m, axis=0)).sum())
    y_faces = int(np.abs(np.diff(m, axis=1)).sum())
    x_faces = int(np.abs(np.diff(m, axis=2)).sum())
    z_faces += int(m[0, :, :].sum()) + int(m[-1, :, :].sum())
    y_faces += int(m[:, 0, :].sum()) + int(m[:, -1, :].sum())
    x_faces += int(m[:, :, 0].sum()) + int(m[:, :, -1].sum())
    sa_mm2 = z_faces * (sy * sx) + y_faces * (sz * sx) + x_faces * (sz * sy)
    return sa_mm2 / 100.0


def _sphericity(volume_ml: float, sa_cm2: float) -> float:
    if sa_cm2 <= 0:
        return 0.0
    return (np.pi ** (1.0 / 3.0) * (6.0 * volume_ml) ** (2.0 / 3.0)) / sa_cm2


def extract_hecktor_lesions(
    suv: np.ndarray,
    seg_multi: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    case_id: str,
    centre_id: int | None,
    centre_name: str,
    vendor: str,
    patient_meta: dict,
    min_volume_ml: float = MIN_LESION_VOLUME_ML,
) -> list[HecktorLesionFeatures]:
    """Per-lesion feature extraction for one HECKTOR patient.

    Parameters
    ----------
    suv : np.ndarray
        3D float SUV volume on the PT grid.
    seg_multi : np.ndarray
        3D multi-class SEG (uint8), already resampled to the PT grid via NN.
        Values: 0=background, 1=GTVp, 2=GTVn.
    voxel_spacing_mm : (sz, sy, sx)
        PT grid spacing in mm.
    case_id : str
        Patient ID (e.g., "MDA-001").
    centre_id : int | None
        EHR CenterID.
    centre_name : str
        Centre name from HECKTOR_CENTRE_VENDOR lookup.
    vendor : str
        Dominant scanner vendor for the centre.
    patient_meta : dict
        EHR row for this patient (Task1, Task2, Relapse, RFS, T-stage, ...).
    """
    if suv.shape != seg_multi.shape:
        raise ValueError(
            f"SUV shape {suv.shape} != SEG shape {seg_multi.shape} for {case_id}; "
            "ensure SEG is resampled to PT grid (NN) before calling."
        )

    binary = (seg_multi > 0).astype(np.int32)
    labelled, n_lab = ndimage.label(binary)
    voxel_volume_ml = float(np.prod(voxel_spacing_mm)) / 1000.0

    rows: list[HecktorLesionFeatures] = []
    for lid in range(1, n_lab + 1):
        comp = labelled == lid
        n_voxels = int(comp.sum())
        volume_ml = n_voxels * voxel_volume_ml
        if volume_ml < min_volume_ml:
            continue

        suv_in = suv[comp]
        suvmax = float(suv_in.max())
        suvmean = float(suv_in.mean())
        suvpeak, _is_fallback = compute_suvpeak(suv, comp, voxel_spacing_mm)
        suvpeak = float(suvpeak)
        tlg = suvmean * volume_ml

        sa_cm2 = _surface_area_cm2(comp, voxel_spacing_mm)
        sphericity = _sphericity(volume_ml, sa_cm2)

        coords = np.argwhere(comp)
        cz, cy, cx = coords.mean(axis=0)
        klass = classify_component(seg_multi, comp)

        rows.append(HecktorLesionFeatures(
            case_id=case_id, lesion_id=lid, series_uid=case_id,
            suvmax=suvmax, suvmean=suvmean, suvpeak=suvpeak, tlg=tlg,
            volume_ml=volume_ml, n_voxels=n_voxels,
            surface_area_cm2=float(sa_cm2),
            sphericity=float(sphericity),
            centroid_0=float(cz), centroid_1=float(cy), centroid_2=float(cx),
            voxel_spacing_0=float(voxel_spacing_mm[0]),
            voxel_spacing_1=float(voxel_spacing_mm[1]),
            voxel_spacing_2=float(voxel_spacing_mm[2]),
            dataset="hecktor", tracer="FDG", vendor=vendor, radionuclide="F-18",
            softmax_mean=GT_SOFTMAX_MEAN, softmax_entropy=GT_SOFTMAX_ENTROPY,
            centre_id=centre_id, centre_name=centre_name, lesion_class=klass,
            task1_patient=bool(patient_meta.get("task1_patient", False)),
            task2_patient=bool(patient_meta.get("task2_patient", False)),
            relapse=patient_meta.get("Relapse"),
            rfs_days=patient_meta.get("RFS"),
            t_stage=patient_meta.get("T-stage"),
            n_stage=patient_meta.get("N-stage"),
            m_stage=patient_meta.get("M-stage"),
            hpv_status=patient_meta.get("HPV Status"),
        ))
    return rows
