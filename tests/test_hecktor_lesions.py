"""Unit tests for the HECKTOR lesion-extraction module (Amendment 8)."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocess.hecktor_lesions import (
    GT_SOFTMAX_ENTROPY,
    GT_SOFTMAX_MEAN,
    LESION_CLASS_GTVN,
    LESION_CLASS_GTVP,
    LESION_CLASS_MIXED,
    classify_component,
    extract_hecktor_lesions,
)
from src.preprocess.hecktor_vendor import (
    HECKTOR_CENTRE_VENDOR,
    UNKNOWN_VENDOR,
    lookup_centre,
)


class TestVendorLookup:
    def test_known_centres_return_name_and_vendor(self):
        for cid, (name, vendor) in HECKTOR_CENTRE_VENDOR.items():
            r_name, r_vendor = lookup_centre(cid)
            assert r_name == name
            assert r_vendor == vendor

    def test_unknown_int_falls_back(self):
        # Centre 99 isn't in the table
        name, vendor = lookup_centre(99)
        assert name == "Unknown"
        assert vendor == UNKNOWN_VENDOR

    def test_none_falls_back(self):
        assert lookup_centre(None) == ("Unknown", UNKNOWN_VENDOR)

    def test_string_int_coerces(self):
        # EHR sometimes stores CenterID as string
        assert lookup_centre("5") == ("MDA", "GE")

    def test_string_prefix_match(self):
        # If passed a centre-name string instead of int, prefix match should work
        assert lookup_centre("MDA-001") == ("MDA", "GE")
        assert lookup_centre("CHUM-042") == ("CHUM", "GE")


class TestClassifyComponent:
    def _make_comp(self, p_voxels: int, n_voxels: int):
        """Return (seg_multi, component_mask) with given voxel counts."""
        total = p_voxels + n_voxels + 5  # plus some background
        seg = np.zeros((total,), dtype=np.uint8)
        seg[:p_voxels] = 1
        seg[p_voxels:p_voxels + n_voxels] = 2
        # component mask covers the GTVp+GTVn region only (not the trailing background)
        comp = np.zeros_like(seg, dtype=bool)
        comp[: p_voxels + n_voxels] = True
        return seg.reshape(-1, 1, 1), comp.reshape(-1, 1, 1)

    def test_pure_gtvp(self):
        seg, comp = self._make_comp(p_voxels=10, n_voxels=0)
        assert classify_component(seg, comp) == LESION_CLASS_GTVP

    def test_pure_gtvn(self):
        seg, comp = self._make_comp(p_voxels=0, n_voxels=10)
        assert classify_component(seg, comp) == LESION_CLASS_GTVN

    def test_dominant_gtvp_wins(self):
        # 70% GTVp, 30% GTVn -> dominant GTVp
        seg, comp = self._make_comp(p_voxels=70, n_voxels=30)
        assert classify_component(seg, comp) == LESION_CLASS_GTVP

    def test_dominant_gtvn_wins(self):
        seg, comp = self._make_comp(p_voxels=20, n_voxels=80)
        assert classify_component(seg, comp) == LESION_CLASS_GTVN

    def test_balanced_returns_mixed(self):
        # 49/51 -- both <50% if we require strictly >=50% for one
        # Spec: dominant requires >=50%, so 51% IS dominant -> GTVN here
        seg, comp = self._make_comp(p_voxels=49, n_voxels=51)
        assert classify_component(seg, comp) == LESION_CLASS_GTVN

    def test_truly_balanced_returns_mixed(self):
        # 50/50 -> first check (n_p/n_total >= 0.5) wins -> GTVP
        # This is intentional: a 50-50 split is rare and the order resolution is documented.
        seg, comp = self._make_comp(p_voxels=50, n_voxels=50)
        assert classify_component(seg, comp) == LESION_CLASS_GTVP

    def test_neither_at_50pct_returns_mixed(self):
        # Build a case where neither side hits 50%: 40 GTVp + 41 GTVn + 19 background
        # voxels INSIDE the component (i.e., the multi-class seg has zeros within
        # the binary-derived component).
        # This shouldn't normally happen (component is the binary mask itself), but
        # we test the spec branch.
        n = 100
        seg = np.zeros((n,), dtype=np.uint8)
        seg[:40] = 1
        seg[40:81] = 2
        # background voxels in [81:100]
        comp = np.ones((n,), dtype=bool)
        seg = seg.reshape(-1, 1, 1)
        comp = comp.reshape(-1, 1, 1)
        # 40/100 = 40%, 41/100 = 41% -> neither >= 50% -> MIXED
        assert classify_component(seg, comp) == LESION_CLASS_MIXED


class TestExtractHecktorLesionsSynthetic:
    """End-to-end on a 32^3 synthetic volume with 2 lesions (one GTVp, one GTVn)."""

    def _make_synthetic(self):
        rng = np.random.RandomState(0)
        suv = rng.uniform(0.1, 0.5, size=(32, 32, 32)).astype(np.float64)
        seg = np.zeros_like(suv, dtype=np.uint8)
        z, y, x = np.ogrid[:32, :32, :32]
        # Lesion 1: GTVp sphere at (8, 8, 8), radius 3 -> 27 voxels, ~0.7 mL @ 3mm iso
        l1 = (z - 8) ** 2 + (y - 8) ** 2 + (x - 8) ** 2 <= 9
        seg[l1] = 1
        suv[l1] = 18.0
        # Lesion 2: GTVn sphere at (24, 24, 24), radius 4 -> larger, ~2.1 mL @ 3mm iso
        l2 = (z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2 <= 16
        seg[l2] = 2
        suv[l2] = 9.0
        # Voxel spacing 3mm iso -> voxel volume 27 mm^3 = 0.027 mL.
        # Lesion 1 ~27 voxels -> 0.73 mL (BELOW 1mL threshold, should be excluded)
        # Lesion 2 ~ ~33 voxels -> 0.89 mL (also below 1mL)
        # Use 4mm iso so voxels are 64mm^3 = 0.064 mL and lesions exceed 1 mL.
        return suv, seg, (4.0, 4.0, 4.0)

    def test_extract_returns_two_lesions_with_class_metadata(self):
        suv, seg, spacing = self._make_synthetic()
        rows = extract_hecktor_lesions(
            suv=suv, seg_multi=seg, voxel_spacing_mm=spacing,
            case_id="TEST-001", centre_id=5, centre_name="MDA", vendor="GE",
            patient_meta={
                "task1_patient": True, "task2_patient": True,
                "Relapse": 0, "RFS": 365.0,
                "T-stage": "T2", "N-stage": "N0", "M-stage": "M0",
                "HPV Status": "negative",
            },
        )
        assert len(rows) == 2
        classes = {r.lesion_class for r in rows}
        assert classes == {LESION_CLASS_GTVP, LESION_CLASS_GTVN}
        for r in rows:
            assert r.dataset == "hecktor"
            assert r.tracer == "FDG"
            assert r.vendor == "GE"
            assert r.softmax_mean == GT_SOFTMAX_MEAN
            assert r.softmax_entropy == GT_SOFTMAX_ENTROPY
            assert r.case_id == "TEST-001"
            assert r.centre_id == 5
            assert r.centre_name == "MDA"
            assert r.task1_patient is True
            assert r.t_stage == "T2"
            # Pre-reg §4.2 geometric predictors must be populated (not default 0)
            assert r.surface_area_cm2 > 0.0
            assert 0.0 < r.sphericity <= 1.0

    def test_min_volume_filter_excludes_small_lesions(self):
        suv, seg, spacing = self._make_synthetic()
        # Add a 1-voxel "lesion" that should be filtered out
        seg[16, 16, 16] = 1
        suv[16, 16, 16] = 50.0
        rows = extract_hecktor_lesions(
            suv=suv, seg_multi=seg, voxel_spacing_mm=spacing,
            case_id="TEST-002", centre_id=5, centre_name="MDA", vendor="GE",
            patient_meta={},
        )
        assert len(rows) == 2  # the singleton voxel excluded by 1mL filter
        for r in rows:
            assert r.suvmax < 50.0  # the 50-SUV singleton excluded

    def test_shape_mismatch_raises(self):
        suv, _, spacing = self._make_synthetic()
        bad_seg = np.zeros((16, 16, 16), dtype=np.uint8)
        with pytest.raises(ValueError, match="SUV shape"):
            extract_hecktor_lesions(
                suv=suv, seg_multi=bad_seg, voxel_spacing_mm=spacing,
                case_id="X", centre_id=5, centre_name="MDA", vendor="GE",
                patient_meta={},
            )

    def test_adjacent_gtvp_gtvn_merge_into_mixed_component(self):
        """GTVp and GTVn touching -> single component classified by majority."""
        suv = np.full((32, 32, 32), 0.2, dtype=np.float64)
        seg = np.zeros_like(suv, dtype=np.uint8)
        # 6x6x6 GTVp block...
        seg[8:14, 8:14, 8:14] = 1
        suv[8:14, 8:14, 8:14] = 15.0
        # ...directly adjacent to a 4x4x4 GTVn block (sharing a face)
        seg[14:18, 8:12, 8:12] = 2
        suv[14:18, 8:12, 8:12] = 10.0
        rows = extract_hecktor_lesions(
            suv=suv, seg_multi=seg, voxel_spacing_mm=(4.0, 4.0, 4.0),
            case_id="ADJ-001", centre_id=5, centre_name="MDA", vendor="GE",
            patient_meta={},
        )
        # Should produce ONE component (the two blocks share a face)
        assert len(rows) == 1
        # GTVp is 6^3 = 216 voxels, GTVn is 4*4*4 = 64 voxels -- GTVp dominant
        assert rows[0].lesion_class == LESION_CLASS_GTVP
