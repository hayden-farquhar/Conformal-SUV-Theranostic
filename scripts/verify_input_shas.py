"""Verify SHA-256 of locked Phase 3 input artefacts against the OSF freeze.

This script provides a single-command pre-flight check before re-running the Phase 3
driver. It hashes each locked input file on disk and compares against the SHA-256
values recorded in `phase3_amendment_11_metadata_suvmax.json` under `src_shas`.

Exit codes:
  0 -- all locked artefacts present and SHAs match
  1 -- one or more artefacts missing
  2 -- one or more SHAs differ from the locked values

Usage:
  python3 scripts/verify_input_shas.py
  python3 scripts/verify_input_shas.py --metadata path/to/alternative_metadata.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "results/phase3/amendment_11/phase3_amendment_11_metadata_suvmax.json"

# Map logical artefact name (from metadata.src_shas keys) to project-relative path.
# Update this map when new locked inputs are added to the metadata schema.
ARTEFACT_PATHS: dict[str, str] = {
    "autopet_i_lesions": "data/interim/lesion_tables/autopet_i_lesions.parquet",
    "autopet_i_splits": "data/processed/autopet_i_splits.parquet",
    "autopet_iii_lesions": "data/interim/lesion_tables/autopet_iii_lesions_reviewed.parquet",
    "hecktor_lesions": "data/interim/lesion_tables/hecktor_lesions_reviewed.parquet",
    "phase2_wcv": "data/processed/phase2_autopet_iii_primary_wcv.parquet",
    "driver_script": "scripts/phase3_evaluate_amendment_11.py",
    "cqr_module": "src/conformal/cqr.py",
    "weighted_module": "src/conformal/weighted.py",
}


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="Path to phase3_amendment_11_metadata_suvmax.json")
    args = parser.parse_args()

    if not args.metadata.exists():
        print(f"ERROR: metadata file not found: {args.metadata}", file=sys.stderr)
        return 1

    with args.metadata.open() as f:
        meta = json.load(f)

    locked_shas: dict[str, str] = meta.get("src_shas", {})
    if not locked_shas:
        print(f"ERROR: metadata.src_shas is empty in {args.metadata}", file=sys.stderr)
        return 1

    print(f"Verifying {len(locked_shas)} locked artefacts against {args.metadata.name}")
    print("=" * 70)

    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []
    matched: list[str] = []

    for name, expected_sha in sorted(locked_shas.items()):
        rel_path = ARTEFACT_PATHS.get(name)
        if rel_path is None:
            print(f"  UNKNOWN   {name}: no path mapping; update ARTEFACT_PATHS")
            missing.append(name)
            continue
        candidate = PROJECT_ROOT / rel_path
        label = f"{name} ({rel_path})"
        if not candidate.exists():
            print(f"  MISSING   {label}")
            missing.append(label)
            continue
        actual_sha = sha256_of(candidate)
        if actual_sha == expected_sha:
            print(f"  OK        {label}")
            matched.append(label)
        else:
            print(f"  MISMATCH  {label}")
            print(f"            expected {expected_sha}")
            print(f"            actual   {actual_sha}")
            mismatched.append((label, expected_sha, actual_sha))

    print("=" * 70)
    print(f"Summary: {len(matched)} OK, {len(mismatched)} mismatched, {len(missing)} missing")

    if missing:
        print("\nMissing artefacts must be retrieved from OSF j5ry4 before the driver can run.",
              file=sys.stderr)
        return 1
    if mismatched:
        print("\nMismatched artefacts indicate the local copy diverges from the OSF freeze.",
              file=sys.stderr)
        print("Refuse to proceed until the local copy is restored to the locked SHA.",
              file=sys.stderr)
        return 2
    print("\nAll locked artefacts verified. Phase 3 driver may run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
