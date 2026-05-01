"""Enumerate AutoPET-I same-patient serial scan pairs from the FDAT zip.

Pre-reg §3.5 decision gate: if <50 same-patient stable-disease pairs within 8 weeks,
the test-retest reference falls back to Poisson-noise injection. This script reads
the FDAT zip namelist (NO content extraction; metadata only) to count:
  - patients with multiple study_date directories
  - in-window pairs (interval ≤ 8 weeks)

Run this in a Colab CPU notebook with Drive mounted. Reads only the zip's central
directory (~few MB), not file contents -- runs in seconds even on the 282.9 GB zip.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN (§3.5)
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
from collections import defaultdict
from datetime import datetime, timedelta

ZIP_PATH = f'{WORK_DIR}/autopet_i/fdg-pet-ct-lesions.zip'
MAX_INTERVAL_WEEKS = 8  # pre-reg §3.5

assert os.path.exists(ZIP_PATH), f'FDAT zip not found at {ZIP_PATH}'

# Read the zip's central directory only (no extraction)
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    names = zf.namelist()

# FDAT structure has an outer wrapper directory:
#   fdg-pet-ct-lesions/{patient_id}/{study_date}/{SUV.nii.gz, SEG.nii.gz, ...}
# Defensively locate the PETCT_<hex> component regardless of wrapper depth.
patient_studies = defaultdict(set)  # patient_id -> set of study_date strings
for n in names:
    if not n.endswith('.nii.gz'):
        continue
    parts = n.split('/')
    petct_idx = next((i for i, p in enumerate(parts) if p.startswith('PETCT_')), None)
    if petct_idx is None or len(parts) <= petct_idx + 1:
        continue
    pid = parts[petct_idx]
    sdate = parts[petct_idx + 1]
    patient_studies[pid].add(sdate)

n_patients_total = len(patient_studies)
n_patients_multi = sum(1 for s in patient_studies.values() if len(s) >= 2)
total_studies = sum(len(s) for s in patient_studies.values())
print(f'Patients (total):              {n_patients_total}')
print(f'Patients with ≥2 studies:      {n_patients_multi} ({n_patients_multi/n_patients_total*100:.1f}%)')
print(f'Total study_date directories:  {total_studies}')

# Try to parse study_date as a date and count pairs within 8 weeks
def try_parse(s):
    # FDAT/TCIA format: "MM-DD-YYYY-NA-PET-CT <description>-<accession>"
    # First 10 chars are the date; the rest is free-text description.
    if len(s) < 10:
        return None
    prefix = s[:10]
    for fmt in ('%m-%d-%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y%m%d'):
        try:
            return datetime.strptime(prefix, fmt)
        except ValueError:
            continue
    return None

max_interval = timedelta(weeks=MAX_INTERVAL_WEEKS)
pairs_in_window = []
unparseable = 0
for pid, sdates in patient_studies.items():
    if len(sdates) < 2:
        continue
    parsed = [(s, try_parse(s)) for s in sdates]
    parseable = [(s, d) for s, d in parsed if d is not None]
    if len(parseable) < len(parsed):
        unparseable += 1
    if len(parseable) < 2:
        continue
    # Sort chronologically (by parsed date), NOT alphabetically -- string sort
    # would put 01-20-2006 before 10-16-2005 even though 10-16-2005 came first
    # in time. With chronological sort, j > i always means parseable[j] is later.
    parseable.sort(key=lambda x: x[1])
    for i in range(len(parseable)):
        for j in range(i + 1, len(parseable)):
            interval = parseable[j][1] - parseable[i][1]
            if interval <= max_interval:
                pairs_in_window.append({
                    'patient_id': pid,
                    'study_1': parseable[i][0],
                    'study_2': parseable[j][0],
                    'interval_days': interval.days,
                })

print(f'\n--- Pairs within {MAX_INTERVAL_WEEKS} weeks (pre-reg §3.5 threshold) ---')
print(f'Total pairs:                   {len(pairs_in_window)}')
if unparseable > 0:
    print(f'Patients with unparseable dates: {unparseable} (excluded from pair counts)')

if len(pairs_in_window) > 0:
    intervals = [p['interval_days'] for p in pairs_in_window]
    print(f'\nInterval distribution (days):')
    print(f'  min:    {min(intervals)}')
    print(f'  median: {sorted(intervals)[len(intervals)//2]}')
    print(f'  max:    {max(intervals)}')
    print(f'\nFirst 10 pairs:')
    for p in pairs_in_window[:10]:
        print(f"  {p['patient_id']}: {p['study_1']} -> {p['study_2']} ({p['interval_days']} days)")

# Pre-reg §3.5 decision gate
print(f'\n=== PRE-REG §3.5 DECISION GATE ===')
n_pairs = len(pairs_in_window)
if n_pairs >= 50:
    print(f'  {n_pairs} pairs >= 50 threshold -- PRIMARY PATH (real serial pairs)')
    print(f'  Phase 2 will use real same-patient pairs for test-retest reference.')
else:
    print(f'  {n_pairs} pairs < 50 threshold -- POISSON-NOISE FALLBACK ACTIVATED')
    print(f'  Phase 2 will use Poisson-noise injection per pre-reg §3.5.')
    print(f'  Note: AutoPET-I extraction kept only ONE study per patient (~12% of cohort had multiple);')
    print(f'  so the real-pair pool is structurally limited even before the 8-week window filter.')
