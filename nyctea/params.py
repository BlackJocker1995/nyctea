"""Parameter loading, scaling, and geometry helpers.

Mirrors ICSearcher's ``icsearcher/params.py``: the :class:`Location` geometry,
param-JSON loading/scaling, and min-max transforms. The parameter spec lives in
``data/param_{ardu,px4}.json`` (each parameter carries ``range`` / ``step`` /
``default`` rows); the active subset for repair is ``toolConfig.PARAM_PART``.

A lazy ``_config()`` accessor reads ``toolConfig`` at call time (not import
time) so tests can repoint the singleton per mode.
"""
import json
import os

import numpy as np
import pandas as pd
from pymavlink import mavextra


def _config():
    """Lazy import so tests can repoint the toolConfig singleton per mode.

    Importing at module top-level binds the singleton object captured at first
    import, which defeats per-mode monkeypatching. Reading it through the module
    at call time picks up any reassignment of ``nyctea.config.toolConfig``.
    """
    from nyctea.config import toolConfig
    return toolConfig


class Location:
    """A 2-D geographic point (lat, lon) with an optional timestamp.

    Wraps a (x=lat, y=lon) pair so the deviation geometry and the mission
    waypoint loader share one type. Distance is the great-circle distance via
    ``mavextra.distance_lat_lon`` (matches the legacy formula).
    """
    def __init__(self, x, y=None, timeS=0):
        if y is None:
            self.x = x.x
            self.y = x.y
        else:
            self.x = x
            self.y = y
        self.timeS = timeS
        self.npa = np.array([x, y])

    def __sub__(self, other):
        return Location(self.x - other.x, self.y - other.y)

    def __str__(self):
        return f"X: {self.x} ; Y: {self.y}"

    def sum(self):
        return self.npa.sum()

    @classmethod
    def distance(cls, point1, point2):
        if point1.x == 0 or point2.x == 0:
            return 0
        return mavextra.distance_lat_lon(point1.x, point1.y,
                                         point2.x, point2.y)


def load_param():
    """Load the full parameter DataFrame for the current mode."""
    cfg = _config()
    with open(cfg._param_file(cfg.MODE), 'r') as f:
        return pd.DataFrame(json.loads(f.read()))


def load_sub_param():
    """Load only the repair subset (PARAM_PART) for the current mode."""
    cfg = _config()
    with open(cfg._param_file(cfg.MODE), 'r') as f:
        return pd.DataFrame(json.loads(f.read()))[cfg.PARAM_PART]


def get_default_values(para_dict):
    """Return the ``default`` row of the param spec (a single-row DataFrame)."""
    return para_dict.loc[['default']]


def select_sub_dict(para_dict, param_choice):
    """Select a subset of columns from the param spec."""
    return para_dict[param_choice]


def read_range_from_dict(para_dict):
    """Return the ``range`` rows as a ``(n_params, 2)`` array of [low, high]."""
    return np.array(para_dict.loc['range'].to_list())


def read_unit_from_dict(para_dict):
    """Return the ``step`` row coerced to float.

    The param JSON mixes integer and decimal step values, so pandas infers an
    object dtype; coercing here keeps ``param * step_unit`` numeric downstream.
    """
    return para_dict.loc['step'].to_numpy(dtype=float)


# --------------------------------------------------------------------- file ops
def read_path_specified_file(log_path, exe):
    """List ``log_path`` files ending in ``.{exe}``, sorted."""
    file_list = []
    for filename in os.listdir(log_path):
        if filename.endswith(f'.{exe}'):
            file_list.append(filename)
    file_list.sort()
    return file_list


def rename_bin(log_path, ranges):
    """Rename ``.BIN`` log files to zero-padded sequence numbers."""
    file_list = read_path_specified_file(log_path, 'BIN')
    for file, num in zip(file_list, range(ranges[0], ranges[1])):
        name, _ = file.split('.')
        os.rename(f"{log_path}/{file}", f"{log_path}/{str(num).zfill(8)}.BIN")


# --------------------------------------------------------------------- scaling
def min_max_scaler_param(param_value):
    """Scale param values to [0, 1] using each param's [low, high] range."""
    if param_value.shape[1] != load_param().shape[1]:
        para_dict = load_sub_param()
    else:
        para_dict = load_param()

    param_bounds = read_range_from_dict(para_dict)
    lb = param_bounds[:, 0]
    ub = param_bounds[:, 1]
    param_value = (param_value - lb) / (ub - lb)
    return param_value.astype(float)


def return_min_max_scaler_param(param_value):
    """Inverse of :func:`min_max_scaler_param`: [0, 1] → real param values."""
    param = load_param()
    param_bounds = read_range_from_dict(param)
    lb = param_bounds[:, 0]
    ub = param_bounds[:, 1]
    param_value = (param_value * (ub - lb)) + lb
    return param_value


def min_max_scaler(trans, values):
    """Scale a combined (status | param) row block: status via fitted scaler,
    params via their [low, high] ranges."""
    status_len = _config().STATUS_LEN
    status_value = values[:, :status_len]
    param_value = values[:, status_len:]

    param_value = min_max_scaler_param(param_value)
    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def return_min_max_scaler(trans, values):
    """Inverse combined scaling: status via inverse transform, params to range."""
    status_len = _config().STATUS_LEN
    status_value = values[:, :status_len]
    param_value = values[:, status_len:]

    param_value = return_min_max_scaler_param(param_value)
    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def pad_configuration_default_value(params_value):
    """Embed a repaired PARAM_PART vector into a full-param row of defaults."""
    para_dict = load_param()
    all_default_value = para_dict.loc[['default']]
    all_default_value = pd.concat([all_default_value] * params_value.shape[0])
    participle_param = _config().PARAM_PART
    all_default_value[participle_param] = params_value
    return all_default_value.values
