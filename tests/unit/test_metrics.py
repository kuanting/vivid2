import math

import numpy as np
import pytest

from vivid2.metrics.control import point_to_polyline
from vivid2.metrics.mission import match_reports
from vivid2.metrics.perception import average_precision, iou
from vivid2.metrics.planning import points_in_polygon
from vivid2.metrics.safety import distance_outside_polygon
from vivid2.scoring import bootstrap_ci, overall_score

TARGETS = [{"id": "A", "ned": [0, 0, 0]}, {"id": "B", "ned": [100, 0, 0]}]


def test_match_reports_tolerance_duplicates_and_false_alarms():
    reports = [
        {"t": 1, "ned": [3, 4, 0]},  # A, 5 m
        {"t": 2, "ned": [1, 1, 0]},  # repeat of A: not a false alarm
        {"t": 3, "ned": [50, 50, 0]},  # false alarm
        {"t": 4, "ned": [111, 0, 0]},  # outside 10 m of B -> false alarm
    ]
    m = match_reports(reports, TARGETS, tolerance_m=10)
    assert set(m["matched"]) == {"A"}
    assert m["matched"]["A"]["error_m"] == pytest.approx(5.0)
    assert len(m["false_alarms"]) == 2


def test_iou_and_ap():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert average_precision([0.9, 0.8], [False, True], 1) == pytest.approx(0.5)
    assert average_precision([], [], 3) == 0.0
    assert math.isnan(average_precision([0.5], [False], 0))


def test_polygon_helpers():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    inside = points_in_polygon(np.array([[5, 5], [15, 5], [-1, -1]]), sq)
    assert inside.tolist() == [True, False, False]
    assert distance_outside_polygon(np.array([5.0, 5.0]), np.asarray(sq, float)) == 0.0
    assert distance_outside_polygon(np.array([13.0, 5.0]), np.asarray(sq, float)) == pytest.approx(3.0)


def test_point_to_polyline():
    line = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], float)
    assert point_to_polyline(np.array([5, 2, 0.0]), line) == pytest.approx(2.0)
    assert point_to_polyline(np.array([12, 5, 0.0]), line) == pytest.approx(2.0)


def test_scoring_skips_missing_layers_and_caps_on_collision():
    scores = {"mission": 100, "perception": None, "reasoning": None, "planning": 100, "control": None, "safety": 100}
    assert overall_score(scores, collisions=0) == pytest.approx(100)
    assert overall_score(scores, collisions=1) == 25


def test_bootstrap_ci():
    ci = bootstrap_ci([50, 60, 70, 80])
    assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["n"] == 4
    assert bootstrap_ci([None])["mean"] is None
