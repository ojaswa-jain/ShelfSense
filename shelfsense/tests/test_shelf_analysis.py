from copy import deepcopy

import pytest

from src.annotation_editor import apply_corrections, editor_rows, manual_detection, validate_box, validate_regions
from src.shelf_analyzer import analyze_shelf, assign_region, compare_audits, potential_gaps, union_area

REGIONS = [{"name": "row1", "box": [0, 0, 100, 100]}]


def test_union_overlap():
    assert union_area([[0, 0, 60, 100], [40, 0, 100, 100]]) == 10000


def test_assign_once_boundary():
    regions = REGIONS + [{"name": "row2", "box": [0, 100, 100, 200]}]
    assert assign_region([10, 90, 30, 110], regions) == "row2"


def test_gaps():
    gaps = potential_gaps([0, 0, 100, 100], [[0, 0, 30, 100], [60, 0, 100, 100]], 0.2)
    assert [g["box"] for g in gaps] == [[30, 0, 60, 100]]


def test_empty_shelf_is_only_potential_gap():
    result = analyze_shelf([], REGIONS, [])
    assert result["visible_facings"] == 0
    assert result["regions"][0]["estimated_visual_occupancy"] == 0
    assert result["regions"][0]["potential_gaps"][0]["label"] == "Potential gap — verify"


def test_shortfall_and_misplacement():
    product = {"product_id": "a", "minimum_facings": 3, "expected_region": "row1"}
    detections = [manual_detection([0, 0, 40, 100], "a")]
    result = analyze_shelf(detections, REGIONS, [product])
    assert result["targets"][0]["display_facing_shortfall"] == 2
    assert result["regions"][0]["estimated_visual_occupancy"] == 0.4


def test_missing_expected_region_not_claimed_shortfall():
    result = analyze_shelf([], REGIONS, [{"product_id": "a", "minimum_facings": 3, "expected_region": "missing"}])
    assert result["targets"][0]["display_facing_shortfall"] is None


def test_corrections_preserve_original_and_update_analysis():
    original = [manual_detection([0, 0, 40, 100])]
    snapshot = deepcopy(original)
    rows = editor_rows(original)
    rows[0].update(x2=80, product_id="a")
    corrected, events = apply_corrections(original, rows, 100, 100, {"a"})
    assert original == snapshot
    assert corrected[0]["status"] == "Manual identity"
    assert len(events) == 1
    assert analyze_shelf(corrected, REGIONS, [])["regions"][0]["estimated_visual_occupancy"] == 0.8


def test_remove_and_add():
    original = [manual_detection([0, 0, 40, 100])]
    rows = editor_rows(original)
    rows[0]["remove"] = True
    rows.append({"id": None, "x1": 50, "y1": 0, "x2": 100, "y2": 100, "product_id": ""})
    result, events = apply_corrections(original, rows, 100, 100, set())
    assert len(result) == 1 and len(events) == 2
    assert result[0]["box"] == [50, 0, 100, 100]


def test_invalid_boxes_and_overlapping_regions():
    with pytest.raises(ValueError):
        validate_box([0, 0, float("nan"), 20], 100, 100)
    with pytest.raises(ValueError):
        validate_regions([{"name": "a", "x1": 0, "y1": 0, "x2": 20, "y2": 20},
                          {"name": "b", "x1": 10, "y1": 10, "x2": 30, "y2": 30}], 100, 100)


def test_comparison_uses_product_evidence():
    reference = analyze_shelf([manual_detection([0, 0, 40, 100], "a")], REGIONS, [])
    current = analyze_shelf([], REGIONS, [])
    changes = compare_audits(current, reference)
    assert changes[0]["visible_facing_delta"] == -1
    assert changes[0]["possible_missing_facings"] == {"a": 1}
