"""Apply §3.9 image-review decisions to the lesion table.

Reads the filled-in decisions CSV (uploaded back to Drive after review) and
the SUVpeak-augmented lesion features parquet. Adds two columns to every row:
  - excluded (bool): True iff the lesion was reviewed and decided to exclude
  - exclusion_reason (str): the anatomic_context label for excluded lesions; '' otherwise

Output: lesion_features_reviewed.parquet on Drive (the §3.9-final lesion table
that downstream conformal calibration should use).

Run on Colab after uploading section_3_9_review_decisions.csv to Drive.
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
import pandas as pd

DECISIONS_CSV = f'{WORK_DIR}/autopet_i/section_3_9_review/section_3_9_review_decisions.csv'
INPUT_PARQUET = f'{WORK_DIR}/autopet_i/lesion_features_v2.parquet'
OUTPUT_PARQUET = f'{WORK_DIR}/autopet_i/lesion_features_reviewed.parquet'

assert os.path.exists(DECISIONS_CSV), 'Upload section_3_9_review_decisions.csv to ' + DECISIONS_CSV + ' first'
assert os.path.exists(INPUT_PARQUET), 'lesion_features_v2.parquet not found at ' + INPUT_PARQUET

decisions = pd.read_csv(DECISIONS_CSV)
df = pd.read_parquet(INPUT_PARQUET)

print('Lesion table: ' + str(len(df)) + ' rows')
print('Decisions:    ' + str(len(decisions)) + ' rows')

# Sanity: every decision row must match a lesion in the table
matched = decisions.merge(df[['case_id', 'lesion_id']], on=['case_id', 'lesion_id'], how='inner')
if len(matched) != len(decisions):
    print('WARNING: ' + str(len(decisions) - len(matched)) + ' decision rows did not match any lesion')

# Sanity: decision must be retain or exclude
bad = decisions[~decisions['decision'].isin(['retain', 'exclude'])]
if len(bad) > 0:
    print('ERROR: rows with invalid decision (must be retain or exclude):')
    print(bad[['case_id', 'lesion_id', 'decision']].to_string())
    raise ValueError('invalid decision values')

# Build exclusion lookup: (case_id, lesion_id) -> exclusion_reason (or None for retained)
exclusions = {}
for _, r in decisions[decisions['decision'] == 'exclude'].iterrows():
    exclusions[(r['case_id'], int(r['lesion_id']))] = str(r['anatomic_context'])

print('To exclude: ' + str(len(exclusions)) + ' lesions')
for k, v in exclusions.items():
    print('  ' + str(k) + ': ' + v)

# Apply
def is_excluded(row):
    return (row['case_id'], int(row['lesion_id'])) in exclusions

def reason(row):
    return exclusions.get((row['case_id'], int(row['lesion_id'])), '')

df['excluded'] = df.apply(is_excluded, axis=1)
df['exclusion_reason'] = df.apply(reason, axis=1)

n_in = len(df)
n_excluded = int(df['excluded'].sum())
n_retained = n_in - n_excluded
print('')
print('Final disposition:')
print('  Total lesions before review:  ' + str(n_in))
print('  Excluded by §3.9 review:      ' + str(n_excluded))
print('  Retained for conformal calib: ' + str(n_retained))
print('  Exclusion rate:                {:.4%}'.format(n_excluded / n_in))

df.to_parquet(OUTPUT_PARQUET, index=False)
print('')
print('Saved: ' + OUTPUT_PARQUET)
print('Schema: original v2 columns + excluded (bool) + exclusion_reason (str)')
print('')
print('Downstream code should filter on "not excluded" before running conformal calibration:')
print('  df = pd.read_parquet(OUTPUT_PARQUET)')
print('  df = df[~df["excluded"]].reset_index(drop=True)')
