"""Render per-lesion review thumbnails for §3.9 image review.

For each of the 7 review lesions, produces a multi-panel PNG showing:
  - 5 axial slices centred on the lesion (centroid-2, centroid-1, centroid, centroid+1, centroid+2)
  - CT in greyscale background, SUV overlay in hot colormap (alpha 0.5)
  - SEG mask outlined in cyan on each slice
  - Text overlay with case_id, lesion_id, SUVmax/peak/ratio/volume

Output: PNG per lesion in the same folder as the manifest. Open in Preview, decide,
fill in section_3_9_review_decisions.csv.

Usage (local, where files are already downloaded):
    python3 render_section_3_9_thumbnails.py /path/to/section_3_9_review
"""

import sys
import os
import csv

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REVIEW_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Downloads/section_3_9_review')
MANIFEST = os.path.join(REVIEW_DIR, 'navigation_manifest.csv')

# Load manifest
with open(MANIFEST) as f:
    rows = list(csv.DictReader(f))
print('Lesions to render: ' + str(len(rows)))

# Cache loaded images per case to avoid re-reading
_image_cache = {}

def load_case(case_id):
    if case_id in _image_cache:
        return _image_cache[case_id]
    case_dir = os.path.join(REVIEW_DIR, case_id)
    suv = nib.load(os.path.join(case_dir, 'SUV.nii.gz')).get_fdata()
    seg = nib.load(os.path.join(case_dir, 'SEG.nii.gz')).get_fdata()
    # CTres is registered to PET grid; preferred for overlay
    ct_path = os.path.join(case_dir, 'CTres.nii.gz')
    if not os.path.exists(ct_path):
        ct_path = os.path.join(case_dir, 'CT.nii.gz')
    ct = nib.load(ct_path).get_fdata()
    print('  loaded ' + case_id + ' SUV={} SEG={} CT={}'.format(suv.shape, seg.shape, ct.shape))
    _image_cache[case_id] = (suv, seg, ct)
    return suv, seg, ct


def render_lesion(row):
    case_id = row['case_id']
    lesion_id = int(row['lesion_id'])
    cz = int(round(float(row['centroid_2'])))
    cy = int(round(float(row['centroid_1'])))
    cx = int(round(float(row['centroid_0'])))
    suvmax = float(row['suvmax'])
    suvpeak = float(row['suvpeak'])
    ratio = float(row['ratio'])
    volume = float(row['volume_ml'])
    triage = row['triage_category']

    suv, seg, ct = load_case(case_id)

    # NIfTI nibabel returns (X, Y, Z) typically — but our centroids were stored
    # from numpy arrays loaded the same way. For SUV/SEG written in the same
    # convention (which they are, since extracted by the same pipeline), the
    # axes match. Centroid axis 2 = "z" (axial slice through the patient).
    n_slices = suv.shape[2]
    z_indices = [max(0, min(n_slices - 1, cz + d)) for d in (-4, -2, 0, 2, 4)]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    # Window/level for CT (Hounsfield range -200 to 400 = soft tissue/lung mix)
    ct_vmin, ct_vmax = -200, 400
    suv_vmax = max(15.0, suvpeak * 1.2)  # scale to SUVpeak so colour range is meaningful

    for col, z in enumerate(z_indices):
        # Top row: CT only
        ax_top = axes[0, col]
        ax_top.imshow(np.rot90(ct[:, :, z]), cmap='gray', vmin=ct_vmin, vmax=ct_vmax,
                      aspect='equal')
        ax_top.set_title('CT  z={}'.format(z), fontsize=10)
        ax_top.axis('off')

        # Bottom row: CT + SUV overlay + SEG outline
        ax_bot = axes[1, col]
        ax_bot.imshow(np.rot90(ct[:, :, z]), cmap='gray', vmin=ct_vmin, vmax=ct_vmax,
                      aspect='equal')
        # SUV overlay — only show where SUV > 1.5 to avoid washing out background
        suv_slice = np.rot90(suv[:, :, z])
        masked = np.ma.masked_where(suv_slice < 1.5, suv_slice)
        ax_bot.imshow(masked, cmap='hot', vmin=0, vmax=suv_vmax, alpha=0.55,
                      aspect='equal')
        # SEG outline
        seg_slice = np.rot90(seg[:, :, z])
        if seg_slice.any():
            ax_bot.contour(seg_slice > 0, levels=[0.5], colors='cyan', linewidths=1.0)
        # Crosshair at centroid (only on the centroid slice itself)
        if z == cz:
            ax_bot.axhline(suv.shape[1] - cy, color='lime', linewidth=0.5, alpha=0.7)
            ax_bot.axvline(cx, color='lime', linewidth=0.5, alpha=0.7)
        ax_bot.set_title('PET+CT  z={}'.format(z), fontsize=10)
        ax_bot.axis('off')

    title = ('{}  lesion {}   |   {}\n'
             'SUVmax {:.1f}   SUVpeak {:.1f}   ratio {:.3f}   volume {:.2f} mL'
             '   |   centroid (vox)=({}, {}, {})').format(
        case_id, lesion_id, triage, suvmax, suvpeak, ratio, volume, cx, cy, cz
    )
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(
        REVIEW_DIR,
        'thumb__{}__lesion_{:03d}.png'.format(case_id, lesion_id)
    )
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('  saved ' + os.path.basename(out_path))


for row in rows:
    print('Rendering ' + row['case_id'] + ' lesion ' + row['lesion_id'])
    render_lesion(row)

print()
print('Done. Open the PNG files in Preview (Finder -> double-click).')
print('Each PNG shows:')
print('  TOP row:    CT only at 5 axial slices around the lesion centroid')
print('  BOTTOM row: same slices with PET overlay (hot colormap) and SEG mask outline (cyan)')
print('  Green crosshair on the centre column marks the lesion centroid.')
print()
print('Thumbnails saved next to navigation_manifest.csv:')
print('  ' + REVIEW_DIR)
