"""Tests for the frozen config singleton (mode resolution, freeze, paths)."""
import os

import pytest

import nyctea.config as cfg_module
from nyctea.config import REPO_ROOT, ToolConfig


def test_default_mode_is_ardupilot():
    c = ToolConfig()
    assert c.MODE == "Ardupilot"
    assert c.SIM == "SITL"
    assert c.PARAM_PART and isinstance(c.PARAM_PART, list)


def test_freeze_blocks_uppercase_set():
    c = ToolConfig()
    with pytest.raises(c.ConstError):
        c.MODE = "PX4"
    with pytest.raises(c.ConstError):
        c.SPEED = 5


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        ToolConfig(mode="Bogus")


def test_env_override_picks_px4(monkeypatch):
    monkeypatch.setenv("NYCTEA_MODE", "PX4")
    c = ToolConfig()
    assert c.MODE == "PX4"
    assert c.SIM == "Jmavsim"
    # PX4 interleaves the BiasA..D columns into STATUS_ORDER.
    for col in ("BiasA", "BiasB", "BiasC", "BiasD"):
        assert col in c.STATUS_ORDER


def test_exe_when_subset_equals_full_set(monkeypatch):
    # In PX4 the PARAM_PART length equals the full PARAM set length → EXE == ''.
    monkeypatch.setenv("NYCTEA_MODE", "PX4")
    c = ToolConfig()
    assert c.EXE == ""


def test_derived_lengths_consistent():
    c = ToolConfig()
    assert c.STATUS_LEN == len(c.STATUS_ORDER) - 1  # drops TimeS
    assert c.PARAM_LEN == len(c.PARAM)


def test_mavlink_port_single_source_of_truth():
    c = ToolConfig()
    assert c.mavlink_port(0) == 14540
    assert c.mavlink_port(12) == 14552  # the legacy i<10 bug would give 14552 only by luck


def test_instances_env_override(monkeypatch):
    monkeypatch.setenv("NYCTEA_INSTANCES", "4")
    c = ToolConfig()
    assert c.INSTANCES == 4


def test_instances_invalid_rejected(monkeypatch):
    monkeypatch.setenv("NYCTEA_INSTANCES", "notanint")
    with pytest.raises(ValueError):
        ToolConfig()


def test_model_dir_is_mode_scoped():
    c = ToolConfig()
    assert c.model_dir().endswith(f"/model/{c.MODE}")
    assert c.buffer_dir().endswith(f"/buffer/{c.MODE}")


def test_params_csv_resolves_icsearcher_result():
    c = ToolConfig()
    path = c.params_csv()
    assert path.endswith(f"/{c.MODE}/params{c.EXE}.csv")
