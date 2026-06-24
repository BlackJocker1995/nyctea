"""RL reward computation — extracted from the legacy ``env.step`` as a pure fn.

The reward formula is preserved verbatim so the trained policy keeps behaving
identically; only the code structure changed (it was inlined in a 60-line
``step`` that also did MAVLink I/O, so it could not be unit-tested).

Semantics:

- ``reward = cur_deviation - played_deviation`` — positive means the uploaded
  config reduced the flight-status deviation (helped the drone), negative means
  it intensified the instability.
- If positive: scale by ``max(1, played_deviation)`` (so a fix from a large
  deviation is rewarded proportionally less), multiply by the mean |AccX|
  (accelerometer activity — bigger is better), clip to ``[0, 100]``.
- If non-positive: clip to ``[-100, 0]``.
"""
import numpy as np


REWARD_CLIP = 100


def compute_reward(cur_deviation: float, played_deviation: float, acc_ratio: float) -> float:
    """Pure reward function.

    Args:
        cur_deviation: status deviation before the action (the agent's input state).
        played_deviation: status deviation after the action took effect.
        acc_ratio: ``mean(|AccX|)`` of the post-action state segment.

    Returns:
        The scalar reward in ``[-100, 100]``.
    """
    reward = cur_deviation - played_deviation
    if reward > 0:
        # Greater reward if the config shrinks a large deviation down past 1.
        reward = reward / max(1, played_deviation)
        # A more active accelerometer is better.
        reward = reward * acc_ratio
        return min(REWARD_CLIP, reward)
    else:
        # Negative: a bad change.
        return max(-REWARD_CLIP, reward)


def deviation_from_state(pd_state, mode: str):
    """Compute the (achieved-state, deviation) pair from a status DataFrame.

    Mirrors ``DroneEnv.get_deviation`` exactly:

    - PX4: deviation = ``sum(|BiasA..D|)`` (the estimator bias channels).
    - Ardupilot: deviation = ``sum(|radians(Des* - Ach*)|)`` over the 6 attitude
      channels (Roll/Pitch/Yaw + their rates).

    Returns ``(out_state_df, deviation_scalar)`` where ``out_state_df`` is the
    input with the deviation-defining columns dropped (the RL observation).
    """
    if mode == "PX4":
        bias = pd_state[['BiasA', 'BiasB', 'BiasC', 'BiasD']]
        out_state = pd_state.drop(
            ['TimeS', 'BiasA', 'BiasB', 'BiasC', 'BiasD'], axis=1)
        deviation = float(np.array(bias).sum())
    else:
        desired_state = pd_state[['DesRoll', 'DesPitch', 'DesYaw',
                                  'DesRateRoll', 'DesRatePitch', 'DesRateYaw']]
        achieved_state = pd_state[['Roll', 'Pitch', 'Yaw',
                                   'RateRoll', 'RatePitch', 'RateYaw']]
        out_state = pd_state.drop(
            ['TimeS', 'DesRoll', 'DesPitch', 'DesYaw',
             'DesRateRoll', 'DesRatePitch', 'DesRateYaw'], axis=1)
        deviation = float(np.abs(np.radians(desired_state.values - achieved_state.values)).sum())
    return out_state, deviation
