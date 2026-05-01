"""§3.9 validation rendering -- AutoPET-III (Amendment 5).

For each lesion in the validation sample (drawn by
`section_3_9_validation_sample.py`), renders a per-lesion review page
showing axial CT+SUV thumbnails + whole-body coronal/sagittal MIPs.

Critical: the reviewer's PNG filename and on-image labels reveal ONLY
the `review_id`. case_id, triage_category, SUVpeak/SUVmax ratio, and
the index-test predicted decision are NEVER drawn on the image.

Designed to run on Colab CPU runtime with Drive mounted. Loads each
PT series once and renders all sampled lesions for that case before
moving on — minimises Drive I/O.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (Amendment 5)

Usage
-----
    # `$WORK_DIR` is whatever local or networked directory holds the raw
    # AutoPET-III cohort data; on Colab the conventional choice is the
    # mounted Google Drive root.
    pip install -q pydicom SimpleITK nibabel scipy matplotlib
    python scripts/section_3_9_validation_render.py \\
        --key        "$WORK_DIR/autopet_iii/sample_key.csv" \\
        --drive-root "$WORK_DIR/autopet_iii" \\
        --suv-module src/preprocess/suv_conversion.py \\
        --out        "$WORK_DIR/autopet_iii/section_3_9_validation_pngs"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Rendering parameters
N_AXIAL_SLICES = 5             # number of axial slices to render around centroid
SLICE_STRIDE = 2               # voxels between rendered slices (skip a slice each side)
SUV_DISPLAY_MAX = 30.0         # SUV colormap clipped to [0, this] for display
CT_HU_WINDOW = (-200, 250)     # soft-tissue window for CT background
FIGSIZE = (18, 12)             # high-res figure
DPI = 180                      # pixel density (was 110; ~2.7x area)
MARKER_SIZE = 14               # cyan crosshair marker
CROSSHAIR_SPAN = 25            # voxels each side of centroid for crosshair lines


def import_suv_module(path: str):
    """Dynamically load src/preprocess/suv_conversion.py from arbitrary location."""
    spec = importlib.util.spec_from_file_location("suv_conversion", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load suv_conversion from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so @dataclass introspection works.
    # (dataclass looks up cls.__module__ in sys.modules to resolve namespace; if the
    # module isn't registered, dataclass raises AttributeError on None.__dict__.)
    sys.modules["suv_conversion"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_dicom_series(series_uid: str, drive_root: str):
    """Returns (dicom_dir, cleanup) -- handles zip-or-directory storage."""
    zip_path = os.path.join(drive_root, f"{series_uid}.zip")
    dir_path = os.path.join(drive_root, series_uid)
    if os.path.exists(zip_path):
        tmp_dir = tempfile.mkdtemp(prefix="dcmser_")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        entries = os.listdir(tmp_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
            inner = os.path.join(tmp_dir, entries[0])
        else:
            inner = tmp_dir
        return inner, lambda: shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.isdir(dir_path):
        return dir_path, lambda: None
    raise FileNotFoundError(f"{series_uid} not on Drive")


def load_seg_aligned_to_pet(seg_path: str, suv_shape: tuple) -> np.ndarray:
    """Load SEG NIfTI; transpose if shape (x,y,z) instead of (z,y,x) to match SUV array."""
    seg = nib.load(seg_path).get_fdata().astype(np.int32)
    if seg.shape == suv_shape:
        return seg
    if seg.T.shape == suv_shape:
        return seg.T
    raise ValueError(f"SEG shape {seg.shape} does not match SUV {suv_shape} (or its transpose)")


def find_lesion_component(seg: np.ndarray, target_centroid: tuple, max_dist_vox: float = 1.5):
    """Find the connected-component label whose centroid is closest to the target.

    Returns the binary mask of the matched component, or raises if no match within tol.
    """
    binary = (seg > 0).astype(np.int32)
    labelled, n_comp = ndimage.label(binary)
    if n_comp == 0:
        raise ValueError("No lesions in SEG mask")

    centroids = np.array(ndimage.center_of_mass(binary, labelled, list(range(1, n_comp + 1))))
    target = np.asarray(target_centroid, dtype=np.float64)
    dists = np.linalg.norm(centroids - target, axis=1)
    best = int(np.argmin(dists))
    if dists[best] > max_dist_vox:
        raise ValueError(
            f"No connected component within {max_dist_vox} voxels of target "
            f"centroid {target_centroid}; closest was {dists[best]:.2f} voxels"
        )
    return (labelled == (best + 1))


def axial_panel(ax, ct_slice: np.ndarray, suv_slice: np.ndarray, seg_slice: np.ndarray):
    """Render a single axial slice with CT grayscale background, SUV hot overlay, SEG cyan outline."""
    # CT in HU window
    ax.imshow(ct_slice.T[::-1], cmap="gray", vmin=CT_HU_WINDOW[0], vmax=CT_HU_WINDOW[1],
              origin="upper")
    # SUV overlay
    suv_masked = np.ma.masked_where(suv_slice < 0.5, suv_slice)
    ax.imshow(suv_masked.T[::-1], cmap="hot", vmin=0, vmax=SUV_DISPLAY_MAX,
              alpha=0.55, origin="upper")
    # SEG outline
    if seg_slice.any():
        ax.contour(seg_slice.T[::-1], levels=[0.5], colors=["cyan"], linewidths=1.0,
                   origin="upper")
    ax.set_xticks([]); ax.set_yticks([])


def mip_panel(
    ax,  # type: ignore[no-untyped-def]
    suv_volume: np.ndarray,
    axis: int,
    lesion_centroid_zyx: tuple,
    spacing_zyx: tuple,
):
    """Render a maximum-intensity projection with a localised cyan crosshair
    pointing at the lesion's exact projected position.

    Replaces the previous full-width axhline (which marked the correct z-row but
    was visually disconnected from small lesions when other bright structures
    dominated the same row at different lateral positions). The crosshair
    consists of two short orthogonal segments centred on the lesion centroid's
    projection in MIP coordinates.
    """
    mip = np.max(suv_volume, axis=axis)
    n_z = mip.shape[0]
    cz, cy, cx = lesion_centroid_zyx
    # In flipped (head-up) MIP, the row is n_z - 1 - cz
    row_y = n_z - 1 - cz
    if axis == 1:  # coronal: project along Y -> result is (Z, X)
        ax.imshow(mip[::-1], cmap="gray_r", vmin=0, vmax=SUV_DISPLAY_MAX,
                  aspect=spacing_zyx[0]/spacing_zyx[2])
        col_x = cx
        ax.set_title("Coronal MIP", fontsize=10)
    elif axis == 2:  # sagittal: project along X -> result is (Z, Y)
        ax.imshow(mip[::-1], cmap="gray_r", vmin=0, vmax=SUV_DISPLAY_MAX,
                  aspect=spacing_zyx[0]/spacing_zyx[1])
        col_x = cy
        ax.set_title("Sagittal MIP", fontsize=10)
    else:
        raise ValueError(f"Unsupported MIP axis {axis}")
    # Cyan crosshair at the lesion's projected (col_x, row_y) position.
    # Two short orthogonal segments rather than full-width lines so the
    # marker is precise to the lesion location, not just its z-row.
    ax.plot([col_x - CROSSHAIR_SPAN, col_x - 4], [row_y, row_y],
            color="cyan", linewidth=1.4, alpha=0.95)
    ax.plot([col_x + 4, col_x + CROSSHAIR_SPAN], [row_y, row_y],
            color="cyan", linewidth=1.4, alpha=0.95)
    ax.plot([col_x, col_x], [row_y - CROSSHAIR_SPAN, row_y - 4],
            color="cyan", linewidth=1.4, alpha=0.95)
    ax.plot([col_x, col_x], [row_y + 4, row_y + CROSSHAIR_SPAN],
            color="cyan", linewidth=1.4, alpha=0.95)
    ax.plot(col_x, row_y, "o", color="cyan", markersize=MARKER_SIZE,
            markerfacecolor="none", markeredgewidth=1.4)
    ax.set_xticks([]); ax.set_yticks([])


def render_lesion(
    review_id: int,
    suv: np.ndarray,
    ct: np.ndarray,
    seg_mask: np.ndarray,
    spacing_zyx: tuple,
    out_path: str,
):
    """Render a single per-lesion review page.

    Layout: 5 axial thumbnails on top row + coronal MIP + sagittal MIP underneath.
    Title shows ONLY "Review ID: NN" (anonymisation discipline).
    Crosshair marker on each MIP points at the lesion centroid's projected (col, row)
    position rather than a full-width row line.
    """
    # Find lesion centroid in (z, y, x) voxel coordinates
    coords = np.argwhere(seg_mask)
    if len(coords) == 0:
        raise ValueError("Empty lesion mask")
    centroid_z = int(round(coords[:, 0].mean()))
    centroid_y = int(round(coords[:, 1].mean()))
    centroid_x = int(round(coords[:, 2].mean()))
    centroid_zyx = (centroid_z, centroid_y, centroid_x)

    # Build figure
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(3, N_AXIAL_SLICES, height_ratios=[1.2, 1.6, 1.6])

    # Axial slices around centroid
    z_offsets = np.arange(-((N_AXIAL_SLICES - 1) // 2), (N_AXIAL_SLICES + 1) // 2) * SLICE_STRIDE
    for col, dz in enumerate(z_offsets):
        z = centroid_z + dz
        ax = fig.add_subplot(gs[0, col])
        if 0 <= z < suv.shape[0]:
            axial_panel(ax, ct[z], suv[z], seg_mask[z])
            ax.set_title(f"axial z+{dz:+d}", fontsize=10)
        else:
            ax.set_visible(False)

    # Coronal MIP (project along axis 1 = Y); crosshair at (centroid_x, centroid_z)
    ax_cor = fig.add_subplot(gs[1, :])
    mip_panel(ax_cor, suv, axis=1, lesion_centroid_zyx=centroid_zyx,
              spacing_zyx=spacing_zyx)

    # Sagittal MIP (project along axis 2 = X); crosshair at (centroid_y, centroid_z)
    ax_sag = fig.add_subplot(gs[2, :])
    mip_panel(ax_sag, suv, axis=2, lesion_centroid_zyx=centroid_zyx,
              spacing_zyx=spacing_zyx)

    fig.suptitle(f"Review ID: {review_id:02d}", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Path to sample_key.csv")
    parser.add_argument("--drive-root", required=True,
                        help="Drive root containing series UID zips/dirs and segmentations/")
    parser.add_argument("--suv-module", required=True,
                        help="Path to src/preprocess/suv_conversion.py")
    parser.add_argument("--out", required=True, help="Output PNG directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic import of the SUV pipeline (so this script doesn't require the project
    # to be importable as a package on Colab)
    suv_mod = import_suv_module(args.suv_module)
    extract_pet_metadata = suv_mod.extract_pet_metadata
    dicom_series_to_suv_sitk = suv_mod.dicom_series_to_suv_sitk

    import SimpleITK as sitk

    # CT loader + resampler (lifted from src/segment/dicom_io.py to keep this script
    # portable to Colab without uploading the whole src/ tree)
    def read_ct_as_hu_sitk(ct_dir):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(ct_dir)
        if not series_ids:
            raise ValueError(f"No DICOM series in {ct_dir}")
        file_names = reader.GetGDCMSeriesFileNames(ct_dir, series_ids[0])
        reader.SetFileNames(file_names)
        return reader.Execute()

    def resample_to_reference(moving, reference, default_value=-1024.0):
        return sitk.Resample(moving, reference, sitk.Transform(), sitk.sitkLinear,
                             default_value, moving.GetPixelID())

    key = pd.read_csv(args.key)
    print(f"Sample key: {len(key)} lesions")

    # Group by series_uid so each PT series is loaded only once
    seg_dir = os.path.join(args.drive_root, "segmentations")
    n_done = 0
    n_failed = 0
    for series_uid, group in key.groupby("series_uid"):
        # We need the matching CT series for this PT. If we have paired_inputs,
        # we can reuse them; otherwise we need to look up CT via the manifest.
        manifest_path = os.path.join(args.drive_root, "_pt_ct_pairs.csv")
        manifest = pd.read_csv(manifest_path)
        match = manifest[manifest["pt_uid"] == series_uid]
        if len(match) == 0:
            print(f"[skip] {series_uid}: no PT/CT pair in manifest")
            n_failed += len(group)
            continue
        ct_uid = match.iloc[0]["ct_uid"]

        print(f"\nLoading {series_uid[-12:]} ({len(group)} lesions to render)...")
        try:
            pt_dir, pt_cleanup = load_dicom_series(series_uid, args.drive_root)
            ct_dir, ct_cleanup = load_dicom_series(ct_uid, args.drive_root)
            try:
                # Find any DICOM file in the PT series for metadata
                import glob
                any_pt_dcm = next(iter(
                    glob.glob(os.path.join(pt_dir, "*.dcm"))
                    or [os.path.join(pt_dir, e) for e in os.listdir(pt_dir)
                        if os.path.isfile(os.path.join(pt_dir, e))]
                ))
                meta = extract_pet_metadata(any_pt_dcm)
                suv_sitk = dicom_series_to_suv_sitk(pt_dir, meta)
                ct_sitk = read_ct_as_hu_sitk(ct_dir)
                ct_resampled = resample_to_reference(ct_sitk, suv_sitk, default_value=-1024.0)

                suv = sitk.GetArrayFromImage(suv_sitk).astype(np.float64)         # (z,y,x)
                ct = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
                spacing_xyz = suv_sitk.GetSpacing()
                spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

                seg_path = os.path.join(seg_dir, f"{series_uid}.nii.gz")
                seg = load_seg_aligned_to_pet(seg_path, suv.shape)

                # Render each lesion in this group
                for _, row in group.iterrows():
                    review_id = int(row["review_id"])
                    target_centroid = (row["centroid_0"], row["centroid_1"], row["centroid_2"])
                    try:
                        lesion_mask = find_lesion_component(seg, target_centroid)
                        out_path = out_dir / f"review_{review_id:02d}.png"
                        render_lesion(review_id, suv, ct, lesion_mask, spacing_zyx, str(out_path))
                        print(f"  rendered review_{review_id:02d}.png")
                        n_done += 1
                    except Exception as e:
                        print(f"  [FAIL] review_{review_id:02d}: {type(e).__name__}: {e}")
                        n_failed += 1
            finally:
                pt_cleanup()
                ct_cleanup()
        except Exception as e:
            print(f"  [FAIL series-level] {type(e).__name__}: {e}")
            n_failed += len(group)

    print(f"\n=== Render summary ===")
    print(f"  Rendered: {n_done}")
    print(f"  Failed:   {n_failed}")
    print(f"  Output:   {out_dir}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
