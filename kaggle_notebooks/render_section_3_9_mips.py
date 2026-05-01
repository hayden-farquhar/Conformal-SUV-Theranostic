"""Render coronal + sagittal MIPs per case with lesion z-positions marked.

Lets us confirm anatomic location for each §3.9 review lesion by showing the
full-body PET MIP and marking the axial slice index of each lesion.
"""

import os
import csv
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

REVIEW_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Downloads/section_3_9_review')
MANIFEST = os.path.join(REVIEW_DIR, 'navigation_manifest.csv')

with open(MANIFEST) as f:
    rows = list(csv.DictReader(f))

# Group by case
by_case = {}
for r in rows:
    by_case.setdefault(r['case_id'], []).append(r)

for case_id, lesions in by_case.items():
    suv_path = os.path.join(REVIEW_DIR, case_id, 'SUV.nii.gz')
    suv_img = nib.load(suv_path)
    suv = suv_img.get_fdata()
    spacing = suv_img.header.get_zooms()[:3]
    n_z = suv.shape[2]

    # Coronal MIP: max along anterior-posterior axis (axis 1)
    coronal_mip = suv.max(axis=1)  # shape (X, Z)
    # Sagittal MIP: max along left-right axis (axis 0)
    sagittal_mip = suv.max(axis=0)  # shape (Y, Z)

    # Display: rotate so superior is up, anatomy oriented correctly
    coronal_disp = np.rot90(coronal_mip)  # now (Z, X) with Z superior up
    sagittal_disp = np.rot90(sagittal_mip)  # now (Z, Y)

    # Aspect ratio: Z spacing 3.0 vs X/Y spacing 2.04
    aspect_cor = spacing[2] / spacing[0]  # for coronal (Z, X)
    aspect_sag = spacing[2] / spacing[1]  # for sagittal (Z, Y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 11))

    # Coronal MIP
    ax = axes[0]
    ax.imshow(coronal_disp, cmap='hot', vmin=0, vmax=10, aspect=aspect_cor, origin='upper')
    # Mark lesion z-positions with horizontal lines (slices count from bottom of array
    # in the original orientation; after np.rot90, what was "slice 0 = bottom" is now
    # "row index = n_z - 1 - z_at_slice_in_orig_coords" wait let me think...
    # Original shape (X, Y, Z). After max along axis 1 -> (X, Z). After np.rot90 -> (Z, X).
    # np.rot90 rotates 90 deg counter-clockwise. Original axes (X, Z) -> new axes (Z_reversed, X)?
    # Actually for a 2D array of shape (X, Z), np.rot90 produces shape (Z, X) where
    # row 0 of the rotated = column (last) of original.
    # If original column z=0 is at the inferior end and column z=N-1 at superior end,
    # after rot90 the row 0 of rotated = column z=N-1 of original = SUPERIOR end.
    # So row index 0 in displayed coronal = superior, row index n_z-1 = inferior.
    # To convert lesion slice index (0-based in original array) to display row:
    #   display_row = n_z - 1 - slice_index
    for lesion in lesions:
        z = int(round(float(lesion['centroid_2'])))
        x = int(round(float(lesion['centroid_0'])))
        display_row = n_z - 1 - z
        ax.axhline(display_row, color='cyan', linewidth=0.8, alpha=0.7)
        ax.scatter(x, display_row, color='lime', s=30, marker='+', linewidth=1.5)
        ax.text(suv.shape[0] + 5, display_row, 'L{} z={}'.format(lesion['lesion_id'], z),
                color='cyan', fontsize=9, va='center')
    ax.set_title('Coronal MIP (anterior view)\n{} | n_slices={} | z_extent={:.0f} mm'.format(
        case_id, n_z, n_z * spacing[2]), fontsize=11)
    ax.set_xlabel('Left-Right')
    ax.set_ylabel('Inferior <- Z -> Superior')
    ax.set_xticks([]); ax.set_yticks([])

    # Sagittal MIP
    ax = axes[1]
    ax.imshow(sagittal_disp, cmap='hot', vmin=0, vmax=10, aspect=aspect_sag, origin='upper')
    for lesion in lesions:
        z = int(round(float(lesion['centroid_2'])))
        y = int(round(float(lesion['centroid_1'])))
        display_row = n_z - 1 - z
        ax.axhline(display_row, color='cyan', linewidth=0.8, alpha=0.7)
        ax.scatter(y, display_row, color='lime', s=30, marker='+', linewidth=1.5)
        ax.text(suv.shape[1] + 5, display_row, 'L{} z={}'.format(lesion['lesion_id'], z),
                color='cyan', fontsize=9, va='center')
    ax.set_title('Sagittal MIP (left view)\n{}'.format(case_id), fontsize=11)
    ax.set_xlabel('Anterior-Posterior')
    ax.set_ylabel('Inferior <- Z -> Superior')
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    out_path = os.path.join(REVIEW_DIR, 'mip__{}.png'.format(case_id))
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print('saved ' + os.path.basename(out_path))

print()
print('Coronal+sagittal MIPs saved to ' + REVIEW_DIR)
print('Cyan lines mark lesion z-positions; green crosses mark X/Y centroid.')
