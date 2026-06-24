"""Action↔configuration mapping — pure functions extracted from the agent.

The DDPG actor outputs actions in ``[0, 1]``; these map them to the real
parameter ranges (step-quantized). Extracted as pure functions so the mapping
can be unit-tested without a live agent/SITL.

Mirrors ``ReLearningAgent.action2config`` (``learning_agent.py``) and the
``create_random_params`` helper verbatim — the trained policy must keep behaving
identically.
"""
import random

import numpy as np


def action2config(action, sub_value_range, step_unit):
    """Map a ``[0,1]`` action vector to step-quantized real parameters.

    Args:
        action: ``[0,1]`` action vector (one entry per parameter).
        sub_value_range: ``(n_params, 2)`` array of ``[low, high]`` per param.
        step_unit: ``(n_params,)`` array of per-param step sizes.

    Returns:
        ``np.ndarray`` of real parameter values, each quantized to its step.
    """
    step_increase = (action * (sub_value_range[:, 1] - sub_value_range[:, 0])
                     // step_unit) * step_unit
    return np.array(sub_value_range[:, 0] + step_increase, dtype=float)


def create_random_params(param_choice, param_spec):
    """Build a random step-quantized config over ``param_choice``.

    Args:
        param_choice: iterable of parameter names to randomize.
        param_spec: the full param DataFrame (columns = param names, rows
            ``range`` / ``step`` / ``default``).

    Returns:
        ``{param: value}`` dict with each value quantized to its step.
    """
    out_dict = {}
    for key in param_choice:
        param_range = param_spec[key]
        value = round(random.uniform(param_range['range'][0], param_range['range'][1])
                      / param_range['step']) * param_range['step']
        out_dict[key] = value
    return out_dict
