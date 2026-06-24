"""Tests for the decomposed AnomalyDetector state machine.

Uses lightweight stub messages so no live SITL is needed. The thresholds and
classification strings are asserted to match the legacy behaviour.
"""
from types import SimpleNamespace

import pytest

import nyctea.anomaly as anomaly_mod
from nyctea.anomaly import (
    CRASH, DEVIATION, PASS, PREARM_FAILED, TIMEOUT, THRUST_LOSS,
    AnomalyDetector, point_to_line_distance,
)
from nyctea.params import Location


def _statustext(text, severity):
    """Build a STATUSTEXT-like stub message."""
    return SimpleNamespace(get_type=lambda: "STATUSTEXT", text=text, severity=severity)


def _other(t):
    """A non-STATUSTEXT message is ignored by on_status."""
    return SimpleNamespace(get_type=lambda: t)


def test_on_status_pass_on_landing():
    det = AnomalyDetector(Location(0, 0), Location(1, 1))
    assert det.result is None
    det.on_status(_statustext("Disarming", 6))
    assert det.result == PASS


def test_on_status_crash_on_severity_0():
    det = AnomalyDetector(Location(0, 0), Location(1, 1))
    det.on_status(_statustext("SIM Hit ground", 0))
    assert det.result == CRASH


def test_on_status_thrust_loss():
    det = AnomalyDetector(Location(0, 0), Location(1, 1))
    det.on_status(_statustext("Potential Thrust Loss", 2))
    assert det.result == THRUST_LOSS


def test_on_status_prearm_failed():
    det = AnomalyDetector(Location(0, 0), Location(1, 1))
    det.on_status(_statustext("PreArm check", 2))
    assert det.result == PREARM_FAILED


def test_on_status_ignores_non_statustext():
    det = AnomalyDetector(Location(0, 0), Location(1, 1))
    det.on_status(_other("ATTITUDE"))
    assert det.result is None


def test_point_to_line_distance_zero_for_degenerate():
    a = Location(1, 1)
    # point at an endpoint, segment length zero → 0
    assert point_to_line_distance(Location(1, 1), a, Location(1, 1)) == 0.0


def test_point_to_line_distance_positive_off_line():
    # Use realistic nonzero latitudes (Location.distance treats lat==0 as
    # "unset" and returns 0 — the legacy guard). A point offset north of an
    # east-west segment must report a positive perpendicular distance.
    a = Location(-35.363, 149.165)
    b = Location(-35.363, 149.175)   # ~1 km east
    p = Location(-35.364, 149.170)   # ~111 m north of the segment midpoint
    d = point_to_line_distance(p, a, b)
    assert d > 0.0


def test_timed_out_sets_timeout():
    det = AnomalyDetector(Location(0, 0), Location(1, 1))
    # Force the start_time far in the past.
    det.start_time = det.start_time - (anomaly_mod.MISSION_TIMEOUT_S + 1)
    assert det.timed_out() is True
    assert det.result == TIMEOUT


def test_seeds_pre_location_from_first_waypoint():
    a = Location(1.0, 2.0)
    det = AnomalyDetector(a, Location(3, 3))
    assert det.pre_location.x == 1.0
    assert det.pre_location.y == 2.0
