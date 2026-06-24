"""Tests for the pure reward function (logic equivalence to legacy env.step)."""
import numpy as np
import pandas as pd
import pytest

from nyctea.rl.reward import REWARD_CLIP, compute_reward, deviation_from_state


def test_reward_positive_scaled_and_clipped():
    # reward = (10-2)/max(1,2) * 1.5 = 8/2*1.5 = 6.0
    assert compute_reward(10.0, 2.0, 1.5) == pytest.approx(6.0)


def test_reward_negative_clipped():
    # reward = 2-10 = -8 (within [-100,0])
    assert compute_reward(2.0, 10.0, 1.0) == -8.0


def test_reward_clips_at_plus100():
    r = compute_reward(1000.0, 0.001, 100.0)
    assert r == REWARD_CLIP


def test_reward_clips_at_minus100():
    r = compute_reward(0.0, 1000.0, 0.0)
    assert r == -REWARD_CLIP


def test_reward_zero_deviation_change_is_zero():
    # cur == played → reward 0, not positive, clipped to max(-100, 0) = 0.
    assert compute_reward(5.0, 5.0, 1.0) == 0.0


def test_reward_acc_ratio_scales_positive_only():
    base = compute_reward(8.0, 2.0, 1.0)   # (8-2)/max(1,2)*1 = 3.0
    scaled = compute_reward(8.0, 2.0, 2.0)  # *2 = 6.0
    assert base == pytest.approx(3.0)
    assert scaled == pytest.approx(6.0)
    # negative reward ignores acc_ratio.
    assert compute_reward(2.0, 8.0, 5.0) == -6.0


def test_deviation_ardupilot_uses_des_minus_ach():
    df = pd.DataFrame({
        "TimeS": [0.0],
        "Roll": [10.0], "DesRoll": [12.0],
        "Pitch": [0.0], "DesPitch": [0.0],
        "Yaw": [0.0], "DesYaw": [0.0],
        "RateRoll": [0.0], "DesRateRoll": [0.0],
        "RatePitch": [0.0], "DesRatePitch": [0.0],
        "RateYaw": [0.0], "DesRateYaw": [0.0],
    })
    out_state, dev = deviation_from_state(df, "Ardupilot")
    # deviation = sum(|radians(Des - Ach)|) = |radians(2)| over Roll only.
    assert dev == pytest.approx(abs(np.radians(2.0)))
    # The desired columns are dropped from the returned observation.
    for col in ("DesRoll", "DesPitch", "DesYaw", "TimeS"):
        assert col not in out_state.columns


def test_deviation_px4_uses_bias_sum():
    df = pd.DataFrame({
        "TimeS": [0.0],
        "BiasA": [1.0], "BiasB": [2.0], "BiasC": [3.0], "BiasD": [4.0],
        "Roll": [0.0], "Pitch": [0.0], "Yaw": [0.0],
    })
    _, dev = deviation_from_state(df, "PX4")
    assert dev == pytest.approx(10.0)  # 1+2+3+4
