"""Tests for the pure action↔config mapping."""
import numpy as np

from nyctea.rl.actions import action2config, create_random_params


def test_action2config_midpoint_quantized():
    # action 0.5 over range [0,10] step 0.1 → 5.0 quantized to 0.1 → 4.9 (49 steps)
    rng = np.array([[0.0, 10.0]], dtype=float)
    step = np.array([0.1], dtype=float)
    cfg = action2config(np.array([0.5]), rng, step)
    assert cfg.shape == (1,)
    assert cfg[0] == pytest_approx(4.9)


def test_action2config_zero_action_gives_low_bound():
    rng = np.array([[2.0, 8.0], [0.0, 1.0]], dtype=float)
    step = np.array([0.5, 0.25], dtype=float)
    cfg = action2config(np.array([0.0, 0.0]), rng, step)
    assert cfg[0] == pytest_approx(2.0)
    assert cfg[1] == pytest_approx(0.0)


def test_action2config_full_action_gives_high_bound_quantized():
    rng = np.array([[0.0, 10.0]], dtype=float)
    step = np.array([0.5], dtype=float)
    cfg = action2config(np.array([1.0]), rng, step)
    # 1.0 * (10-0) // 0.5 * 0.5 = 20 * 0.5 = 10.0
    assert cfg[0] == pytest_approx(10.0)


def test_create_random_params_in_range_and_quantized():
    import pandas as pd
    spec = pd.DataFrame(
        {"P1": {"range": [0.0, 10.0], "step": 0.5, "default": 5.0},
         "P2": {"range": [1.0, 3.0], "step": 0.25, "default": 2.0}})
    for _ in range(20):
        out = create_random_params(["P1", "P2"], spec)
        assert 0.0 <= out["P1"] <= 10.0
        assert 1.0 <= out["P2"] <= 3.0
        # quantized to step
        assert abs(out["P1"] / 0.5 - round(out["P1"] / 0.5)) < 1e-9
        assert abs(out["P2"] / 0.25 - round(out["P2"] / 0.25)) < 1e-9


def pytest_approx(expected, **kw):
    """Local approx to avoid importing pytest at module top for the helper."""
    import pytest
    return pytest.approx(expected, **kw)
