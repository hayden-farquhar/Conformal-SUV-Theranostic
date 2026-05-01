"""Extract the 4 cases needed for §3.9 image review from the FDAT zip on Drive.

Selects the 7 residual lesions:
  - Cat A: PETCT_7b42056ee3 lesion 1, PETCT_27d69a8466 lesion 11
  - Cat B (non-1285b86bea): all from PETCT_73597f33fe and PETCT_f65602d938

For each unique case (4 cases), extracts:
  - SUV.nii.gz  (PET in SUV units)
  - SEG.nii.gz  (lesion segmentation mask)
  - CTres.nii.gz  (resampled CT, if present in zip — gives anatomic context)

Outputs go to {WORK_DIR}/autopet_i/section_3_9_review/
plus a navigation manifest CSV + a decision-tracking template CSV.
"""


# Working-directory configuration:
# Set the WORK_DIR environment variable to point at the local or networked
# folder that holds the raw cohort data (DICOM / SUV NIfTI / SEG NIfTI).
# Default is `<repo_root>/work_dir`; on Colab the conventional choice is
# the mounted Google Drive root (e.g. /content/drive/MyDrive/<your-folder>).
import os as _os
from pathlib import Path as _Path
WORK_DIR = _os.environ.get(
    "WORK_DIR",
    str(_Path(__file__).resolve().parent.parent / "work_dir") if "__file__" in globals()
    else "/content/work_dir",
)

import os
import zipfile
import shutil
import pandas as pd
import numpy as np

ZIP_PATH = f'{WORK_DIR}/autopet_i/fdg-pet-ct-lesions.zip'
LESION_PARQUET = f'{WORK_DIR}/autopet_i/lesion_features_v2.parquet'
REVIEW_CSV = f'{WORK_DIR}/autopet_i/suv_outlier_review.csv'
OUT_DIR = f'{WORK_DIR}/autopet_i/section_3_9_review'

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Identify the 7 review lesions from the triage CSV
review = pd.read_csv(REVIEW_CSV)
needs = (
    (review['triage_category'] == 'A_extreme_suv')
    | (
        (review['triage_category'] == 'B_small_high_suv')
        & (review['case_id'] != 'PETCT_1285b86bea')
    )
)
to_review = review[needs].copy()
print('Lesions needing image review: ' + str(len(to_review)))
print('Unique cases: ' + str(to_review['case_id'].nunique()))
print()
print(to_review[['case_id', 'lesion_id', 'triage_category', 'suvmax', 'suvmean', 'volume_ml']].to_string(index=False))
print()

# 2. Enrich with SUVpeak + ratio + spatial metadata from the v2 parquet
# Drop overlapping columns from the review CSV first — Step 7 wrote a subset of
# spatial columns (centroid_2, centroid_z_mm, voxel_spacing_2) into the CSV, and
# pandas would otherwise suffix them as _x / _y on merge.
to_review = to_review.drop(
    columns=['centroid_2', 'centroid_z_mm', 'voxel_spacing_2'],
    errors='ignore',
)

df = pd.read_parquet(LESION_PARQUET)
df['ratio'] = df['suvpeak'] / df['suvmax']
key_cols = ['case_id', 'lesion_id', 'suvpeak', 'ratio',
            'centroid_0', 'centroid_1', 'centroid_2',
            'voxel_spacing_0', 'voxel_spacing_1', 'voxel_spacing_2']
to_review = to_review.merge(df[key_cols], on=['case_id', 'lesion_id'], how='left')

# Convert centroid voxel indices to physical mm (relative to image origin)
to_review['centroid_z_mm'] = to_review['centroid_2'] * to_review['voxel_spacing_2']
to_review['centroid_y_mm'] = to_review['centroid_1'] * to_review['voxel_spacing_1']
to_review['centroid_x_mm'] = to_review['centroid_0'] * to_review['voxel_spacing_0']

# 3. Index the FDAT zip and find files for our 4 cases
cases_to_extract = sorted(to_review['case_id'].unique())
print('Cases to extract from FDAT zip: ' + str(cases_to_extract))
print()

print('Indexing FDAT zip (this is fast — only reads the central directory) ...')
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    all_names = zf.namelist()

case_files = {}
for name in all_names:
    if not name.endswith('.nii.gz'):
        continue
    parts = name.split('/')
    if len(parts) < 3:
        continue
    case_id = parts[1]
    if case_id in cases_to_extract:
        filename = parts[-1]
        case_files.setdefault(case_id, {})[filename] = name

# Show what we found per case
print('Files available per case:')
for c in cases_to_extract:
    files = sorted(case_files.get(c, {}).keys())
    print('  ' + c + ': ' + ', '.join(files))
print()

# 4. Extract SUV / SEG / CTres (if present) per case
WANTED = ['SUV.nii.gz', 'SEG.nii.gz', 'CTres.nii.gz', 'CT.nii.gz', 'PET.nii.gz']
extracted_log = []
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for case_id in cases_to_extract:
        case_out = os.path.join(OUT_DIR, case_id)
        os.makedirs(case_out, exist_ok=True)
        files = case_files.get(case_id, {})
        for w in WANTED:
            if w not in files:
                continue
            src = files[w]
            dst = os.path.join(case_out, w)
            if os.path.exists(dst):
                print('  skip (exists): ' + dst)
                extracted_log.append({'case_id': case_id, 'file': w, 'status': 'exists', 'size_mb': round(os.path.getsize(dst)/(1024**2), 1)})
                continue
            print('  extract: ' + src + ' -> ' + dst)
            with zf.open(src) as zsrc, open(dst, 'wb') as out:
                shutil.copyfileobj(zsrc, out, length=8 * 1024 * 1024)
            extracted_log.append({'case_id': case_id, 'file': w, 'status': 'extracted', 'size_mb': round(os.path.getsize(dst)/(1024**2), 1)})
print()

extracted_df = pd.DataFrame(extracted_log)
print('Extraction summary:')
print(extracted_df.to_string(index=False))
print()
print('Total extracted size: {:.1f} MB'.format(extracted_df['size_mb'].sum()))
print()

# 5. Write the navigation manifest (one row per lesion to review)
nav_cols = [
    'case_id', 'lesion_id', 'triage_category',
    'suvmax', 'suvmean', 'suvpeak', 'ratio', 'volume_ml',
    'centroid_0', 'centroid_1', 'centroid_2',
    'centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm',
    'voxel_spacing_0', 'voxel_spacing_1', 'voxel_spacing_2',
]
nav = to_review[nav_cols].sort_values(['case_id', 'lesion_id'])
nav_path = os.path.join(OUT_DIR, 'navigation_manifest.csv')
nav.to_csv(nav_path, index=False)
print('Navigation manifest saved: ' + nav_path)

# 6. Write decision-tracking template (one row per lesion, blank decision/notes)
decisions = nav[['case_id', 'lesion_id', 'triage_category', 'suvmax', 'suvpeak', 'ratio', 'volume_ml']].copy()
decisions['decision'] = ''  # 'retain' / 'exclude'
decisions['anatomic_context'] = ''  # free text: 'bladder bleed-in', 'true avid lesion', etc.
decisions['reviewer_notes'] = ''
dec_path = os.path.join(OUT_DIR, 'section_3_9_review_decisions.csv')
decisions.to_csv(dec_path, index=False)
print('Decision template saved: ' + dec_path)
print()
print('NEXT STEPS:')
print('  1. Download {} folder locally (or sync via Drive Desktop)'.format(OUT_DIR))
print('     Files needed: each case folder contains SUV.nii.gz + SEG.nii.gz (+ CTres if present)')
print('  2. Open in 3D Slicer or ITK-SNAP. For each lesion in navigation_manifest.csv:')
print('     - Load SUV.nii.gz as background, SEG.nii.gz as segmentation overlay, CTres.nii.gz if anatomic context needed')
print('     - Navigate to (centroid_x_mm, centroid_y_mm, centroid_z_mm) in physical coordinates')
print('     - Inspect: is the masked region a real avid tumour, or physiologic uptake (bladder/brain/injection)?')
print('  3. Record decisions in section_3_9_review_decisions.csv (decision column: retain / exclude)')
print('  4. Re-upload the completed CSV; we will then update the lesion table and close §3.9.')
