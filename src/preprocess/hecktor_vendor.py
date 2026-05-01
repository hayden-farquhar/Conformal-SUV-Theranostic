"""HECKTOR per-centre scanner vendor lookup (Amendment 8 §8d).

HECKTOR PT NIfTIs come pre-SUV-converted by the challenge organisers; vendor
metadata is NOT preserved in the NIfTI headers. To support the Mondrian
stratification (`vendor x volume_quartile x centre`), this module maps the
EHR `CenterID` to the dominant scanner vendor per centre, derived from the
HECKTOR documentation (Andrearczyk et al. 2022, Oreiller et al. 2022).

**Provenance and editing rule.** The mapping below is initial best-effort
based on the published HECKTOR challenge papers; centres where vendor
heterogeneity is documented (mixed scanners within a centre) are tagged as
"Mixed" and require special handling in the Mondrian (the centre is then
excluded from per-vendor cells but still appears in the per-centre marginal).
Update this table only by editing this file and re-running lesion extraction;
do NOT silently override at runtime.

Pre-registration: https://doi.org/10.17605/OSF.IO/4KAZN sec 4.3
Amendment 8: 2026-04-30 (osf/amendment_log.md)
"""
from __future__ import annotations


# Centre-ID -> (centre_name, dominant_vendor) per HECKTOR documentation.
# `None` for unverified centres -- those are flagged at extraction time and
# fall through to the "Unknown" vendor with a logged warning.
HECKTOR_CENTRE_VENDOR = {
    # Integer-encoded CenterID per the HECKTOR challenge EHR (the CenterID
    # column uses integer codes, NOT letter-prefix codes). Mapping below pairs
    # the integer code to the centre name reported in HECKTOR challenge papers
    # and the dominant scanner vendor for that centre.
    1: ("CHUM",  "GE"),       # Centre Hospitalier de l'Universite de Montreal
    2: ("CHUS",  "Philips"),  # CHU Sherbrooke
    3: ("CHUP",  "Siemens"),  # CHU Poitiers (or CHUS depending on HECKTOR cohort version)
    # 4: gap in the EHR enumeration -- not present in the defaced training release
    5: ("MDA",   "GE"),       # MD Anderson Cancer Center -- dominant centre (~58-61%)
    6: ("CHUV",  "Siemens"),  # CHU Vaudois (Lausanne)
    7: ("USZ",   "GE"),       # University Hospital Zurich
    8: ("HMR",   "GE"),       # Hopital Maisonneuve-Rosemont (Montreal)
}

UNKNOWN_VENDOR = "Unknown"


def lookup_centre(center_id: int | str | None) -> tuple[str, str]:
    """Resolve EHR CenterID -> (centre_name, vendor).

    Parameters
    ----------
    center_id : int / str / None
        EHR CenterID value. May be an integer code, a string prefix, or None.

    Returns
    -------
    (centre_name, vendor) : tuple of str
        ("Unknown", "Unknown") if center_id is missing or not in the lookup.
    """
    if center_id is None:
        return ("Unknown", UNKNOWN_VENDOR)
    try:
        cid = int(center_id)
    except (TypeError, ValueError):
        cid = None
    if cid is not None and cid in HECKTOR_CENTRE_VENDOR:
        return HECKTOR_CENTRE_VENDOR[cid]
    # String fallback: attempt prefix match against centre names
    if isinstance(center_id, str):
        s = center_id.strip().upper()
        for _cid, (name, vendor) in HECKTOR_CENTRE_VENDOR.items():
            if s.startswith(name):
                return (name, vendor)
    return ("Unknown", UNKNOWN_VENDOR)
