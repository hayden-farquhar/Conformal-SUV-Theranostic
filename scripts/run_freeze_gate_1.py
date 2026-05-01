"""Freeze Gate 1: AutoPET-I patient-level split allocation.

Produces data/processed/autopet_i_splits.parquet with patient_id assigned to
one of {train, calibration, test, serial} per pre-reg §3.4 fractions
(40/20/20/20). Computes the SHA-256 hash that gets recorded in
osf/data_snapshot_log.md before any conformal calibration.

Patient-level assumption: AutoPET-I FDAT release uses per-study case_ids
(format: PETCT_<hex>) without an explicit patient mapping. We have no
clinical metadata CSV locally to map studies → patients. We therefore treat
each case_id as a patient_id. This is conservative because (a) the AutoPET-I
publication reports 900 patients across 1014 studies — ~12% of patients have
multiple studies — but (b) we only have 461 disease-positive cases, where the
study-per-patient ratio is likely closer to 1:1, and (c) any over-counting of
patients still produces patient-level (not lesion-level) splits — the only
risk is a small number of patients having lesions split across train/test,
which is mitigated by the small magnitude of multi-study patients.

This assumption is documented in osf/data_snapshot_log.md when the manifest
is recorded.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess.split_allocation import (
    allocate_autopet_i_splits,
    compute_manifest_hash,
    save_split_manifest,
    print_split_summary,
    SEED,
    AUTOPET_I_FRACTIONS,
)

LESION_TABLE = ROOT / 'data/interim/lesion_tables/autopet_i_lesions.parquet'
OUTPUT = ROOT / 'data/processed/autopet_i_splits.parquet'

print('=' * 68)
print('Freeze Gate 1: AutoPET-I patient-level split allocation')
print('=' * 68)
print(f'Pre-reg §3.4 fractions: {AUTOPET_I_FRACTIONS}')
print(f'Random seed: {SEED}')
print(f'Input lesion table: {LESION_TABLE}')
print(f'Output split manifest: {OUTPUT}')
print()

lesions = pd.read_parquet(LESION_TABLE)
print(f'Lesion table: {len(lesions)} rows, {lesions["case_id"].nunique()} cases')
print(f'Excluded: {int(lesions["excluded"].sum())} | Retained: {int((~lesions["excluded"]).sum())}')

# Build patient-level frame (one row per case_id = patient_id).
# Only include patients with at least one retained lesion (excluded patients
# would be those whose ONLY lesions were §3.9-excluded — for our 2-exclusion
# case this affects 0 patients since both excluded lesions are from cases that
# also have many retained lesions, but we check anyway).
retained = lesions[~lesions['excluded']]
patients_with_retained = set(retained['case_id'])
patients_all = set(lesions['case_id'])
patients_dropped = patients_all - patients_with_retained
print(f'Patients with >=1 retained lesion: {len(patients_with_retained)} '
      f'(all {len(patients_all)}; dropped: {len(patients_dropped)})')

patient_df = pd.DataFrame({
    'patient_id': sorted(patients_with_retained),
})
# AutoPET-I: single vendor, single tracer (per Amendment 2)
patient_df['vendor'] = 'Siemens'
patient_df['tracer_category'] = 'FDG'

print()
print(f'Patient-level frame: {len(patient_df)} patients')
print('  vendor distribution:    ', patient_df['vendor'].value_counts().to_dict())
print('  tracer distribution:    ', patient_df['tracer_category'].value_counts().to_dict())
print()

# Allocate
split_df = allocate_autopet_i_splits(patient_df, seed=SEED)

print('Split allocation result:')
print(split_df['split'].value_counts().sort_index())
print()
print('Fraction realised vs pre-registered:')
n = len(split_df)
for split_name, target in AUTOPET_I_FRACTIONS.items():
    actual = (split_df['split'] == split_name).sum() / n
    print(f'  {split_name:11s}  target={target:.2%}  actual={actual:.2%}  n={int((split_df["split"] == split_name).sum())}')
print()

# Save manifest + compute hash (in one call)
hash_value = save_split_manifest(split_df, OUTPUT)

# Also write a separate metadata sidecar with provenance
import json, datetime
sidecar = {
    'created_utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'pre_reg_doi': '10.17605/OSF.IO/4KAZN',
    'pre_reg_section': '3.4',
    'random_seed': SEED,
    'fractions': AUTOPET_I_FRACTIONS,
    'input_lesion_table': str(LESION_TABLE.relative_to(ROOT)),
    'input_lesion_count_total': int(len(lesions)),
    'input_lesion_count_excluded': int(lesions['excluded'].sum()),
    'input_lesion_count_retained': int((~lesions['excluded']).sum()),
    'patients_assigned': int(len(split_df)),
    'patients_dropped_no_retained_lesions': len(patients_dropped),
    'patient_level_assumption': 'case_id treated as patient_id (FDAT release lacks per-study patient mapping)',
    'split_counts': split_df['split'].value_counts().to_dict(),
    'manifest_sha256': hash_value,
}
sidecar_path = OUTPUT.with_suffix('.metadata.json')
with open(sidecar_path, 'w') as f:
    json.dump(sidecar, f, indent=2)
print(f'Sidecar metadata: {sidecar_path}')

print()
print('=' * 68)
print('FREEZE GATE 1 — manifest produced')
print('=' * 68)
print(f'SHA-256 to record in osf/data_snapshot_log.md:')
print(f'  {hash_value}')
print()
print('Lesion-level retention by split (sanity check, using retained lesions only):')
lesions_retained = lesions[~lesions['excluded']].merge(
    split_df.rename(columns={'patient_id': 'case_id'})[['case_id', 'split']],
    on='case_id', how='left'
)
print(lesions_retained.groupby('split').size().to_string())
