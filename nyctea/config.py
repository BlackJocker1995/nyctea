# coding:utf-8
"""Project configuration.

The :data:`toolConfig` singleton is constructed once at import time from
``data/config.yaml``. The mode (``PX4`` or ``Ardupilot``) is **frozen at load
time**: it comes from ``config.yaml``'s ``mode`` field and may be overridden by
the ``NYCTEA_MODE`` environment variable. All mode-derived constants
(``STATUS_ORDER``, ``PARAM``, ``PARAM_PART``, simulator paths, the buffer/checkpoint
directories, derived lengths) are computed once during construction.

There is no runtime ``select_mode`` anymore — it was removed for the same reason
ICSearcher dropped it: the legacy write-once ``__setattr__`` dance let every
``_px4`` script mutate the singleton after the fact, which silently broke long
training runs. To run the pipeline in a different mode, set ``mode`` in
``data/config.yaml`` (or the ``NYCTEA_MODE`` env var) before importing anything
that reads ``toolConfig``.

Override precedence for the magic fields: CLI flag > env var > yaml > default.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

# Absolute path to the repo root. Every relative path in the project is
# resolved against this so the pipeline no longer depends on the CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent

# If the file-based resolution doesn't point at the project root (e.g. under
# some editable-install configurations where __file__ resolves to a synthetic
# path), fall back to the working directory. Both methods are validated by
# checking for a known marker (the nyctea/config.py itself).
if not (REPO_ROOT / "nyctea" / "config.py").is_file():
    cwd = Path.cwd()
    if (cwd / "nyctea" / "config.py").is_file():
        REPO_ROOT = cwd
    else:
        raise RuntimeError(
            f"Cannot locate project root: {REPO_ROOT} (from __file__) "
            f"and {cwd} (from CWD) neither contains nyctea/config.py. "
            "Run from the project directory."
        )

DATA_DIR = REPO_ROOT / "data"

VALID_MODES = ("Ardupilot", "PX4")

# Per-mode parameter subsets that participate in repair. These differ from
# ICSearcher's PARAM_PART because the two projects fuzz/repair different slices
# of the controller — nyctea inherits the legacy subset it was trained against.
PARAM_PART_ARDUPILOT = [
    "PSC_VELXY_P", "PSC_VELXY_I", "PSC_VELXY_D",
    "ATC_ANG_RLL_P", "ATC_RAT_RLL_P", "ATC_RAT_RLL_I", "ATC_RAT_RLL_D",
    "ATC_ANG_PIT_P", "ATC_RAT_PIT_P", "ATC_RAT_PIT_I", "ATC_RAT_PIT_D",
    "ATC_ANG_YAW_P", "ATC_RAT_YAW_P", "ATC_RAT_YAW_I", "ATC_RAT_YAW_D",
]
PARAM_PART_PX4 = [
    "MC_ROLL_P", "MC_PITCH_P", "MC_YAW_P", "MC_YAW_WEIGHT",
    "MPC_XY_P", "MPC_Z_P",
    "MC_PITCHRATE_P", "MC_ROLLRATE_P", "MC_YAWRATE_P",
    "MIS_YAW_ERR", "MPC_TKO_SPEED",
]

# Ordered status channels. The leading TimeS column is excluded from STATUS_LEN.
# Identical to ICSearcher's STATUS_ORDER_COMMON (same .BIN/.ulg telemetry layout).
STATUS_ORDER_COMMON = [
    'TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
    'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
    'MagX', 'MagY', 'MagZ', 'VibeX', 'VibeY', 'VibeZ',
]

# PX4 online status uses the BiasA..D columns the RL state reads; injected into
# STATUS_ORDER at _apply_mode time for that mode (mirrors the legacy layout).
PX4_BIAS_COLS = ['BiasA', 'BiasB', 'BiasC', 'BiasD']


class ToolConfig:
    """Frozen-at-load config singleton.

    Construct with ``ToolConfig(mode=...)`` (mode defaults to the yaml/env
    value). After construction every attribute is read-only; attempts to set a
    new uppercase constant raise ``ConstError`` so accidental mutation fails
    loudly instead of silently corrupting a long training run.
    """

    class ConstError(PermissionError):
        pass

    def __init__(self, mode=None):
        self.__dict__["yaml_config"] = self._load_yaml_config()
        self._init_defaults()
        # If paths from yaml don't exist, try to auto-detect from sims/.
        self._detect_sims()
        # Resolve the definitive mode once: explicit arg > env var > yaml.
        resolved = mode or os.environ.get("NYCTEA_MODE") or self.__dict__["MODE"]
        if resolved not in VALID_MODES:
            raise ValueError(f"Invalid MODE {resolved!r}; expected one of {VALID_MODES}")
        self._apply_mode(resolved)

    # ------------------------------------------------------------------ loading
    def _load_yaml_config(self):
        """Load YAML config with fallback to empty dict.

        Priority:
          1. ``data/config.yaml`` (machine-specific, gitignored — generated
             by copying the .example template).
          2. ``data/config.yaml.example`` (committed template).
          3. Empty dict (all defaults used).
        """
        config_path = DATA_DIR / 'config.yaml'
        example_path = DATA_DIR / 'config.yaml.example'

        # If the machine-specific file doesn't exist yet, create it from the
        # example template so the user gets a ready-to-edit copy.
        if not config_path.exists() and example_path.exists():
            try:
                import shutil
                shutil.copy2(example_path, config_path)
                print(f"Created {config_path} from {example_path.name} — edit it if needed.")
            except OSError as e:
                print(f"Warning: could not copy {example_path} to {config_path}: {e}")

        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not load config.yaml ({str(e)}), using defaults")
            return {}

    def _get_yaml_value(self, *keys, default=None):
        """Safely get nested YAML config value with fallback."""
        config = self.yaml_config
        for key in keys:
            if not isinstance(config, dict):
                return default
            config = config.get(key, default)
        return config

    def _param_file(self, mode):
        """Resolve the parameter JSON path for the given mode.

        Reads ``param_files.<mode>`` from config.yaml (relative to the repo
        root by default) and falls back to ``data/param_<mode>.json``.
        """
        key = 'ardupilot' if mode == 'Ardupilot' else 'px4'
        rel = self._get_yaml_value('param_files', key, default=f'data/param_{key}.json')
        return str(REPO_ROOT / rel) if not os.path.isabs(rel) else rel

    # ------------------------------------------------------------------ path API
    def resolve(self, rel):
        """Resolve a repo-relative path to an absolute string.

        Absolute paths are returned unchanged so machine-specific simulator
        locations keep working.
        """
        if not rel:
            return rel
        return rel if os.path.isabs(rel) else str(REPO_ROOT / rel)

    def mission_file(self):
        """Absolute path to the fit-collection mission for the current mode."""
        if self.HOME is None:
            name = 'data/mission_px4.txt' if self.MODE == 'PX4' else 'data/mission.txt'
        else:
            name = 'data/fitCollection_px4.txt' if self.MODE == 'PX4' else 'data/fitCollection.txt'
        rel = self._get_yaml_value('missions', 'fit_collection', self.MODE.lower(), default=name)
        return self.resolve(rel)

    # ------------------------------------------------------------------ multi-instance paths
    def _instance_subdir(self, i):
        """Render the per-instance subdirectory name from INSTANCE_DIR."""
        return self.INSTANCE_DIR.replace('{i}', str(int(i)))

    def ardu_instance_path(self, i):
        """Per-instance ArduPilot working directory.

        Each concurrent SITL instance gets its own directory under
        ``ARDUPILOT_LOG_PATH`` so its ``eeprom.bin`` / ``mav.parm`` / ``logs/``
        do not collide with sibling instances. ``i`` is the 0-based instance index.
        """
        return os.path.join(self.ARDUPILOT_LOG_PATH, self._instance_subdir(i))

    def ardu_instance_log_path(self, i):
        """Per-instance ArduPilot ``logs/`` directory (created lazily by callers)."""
        return os.path.join(self.ardu_instance_path(i), 'logs')

    def px4_instance_path(self, i):
        """Per-instance PX4 build directory (``instance_{i}`` under the build tree)."""
        return os.path.join(
            self.PX4_RUN_PATH, 'build', 'px4_sitl_default',
            self._instance_subdir(i),
        )

    # ------------------------------------------------------------------ RL artefact paths
    def model_dir(self):
        """Absolute path to the checkpoint directory for the current mode.

        Checkpoints live under ``model_dir/{MODE}/`` so ArduPilot and PX4 runs
        don't clobber each other.
        """
        rel = self._get_yaml_value('paths', 'model_dir', default='model')
        return os.path.join(self.resolve(rel), self.MODE)

    def buffer_dir(self):
        """Absolute path to the replay-buffer shard directory for the current mode."""
        rel = self._get_yaml_value('paths', 'buffer_dir', default='model/buffer')
        return os.path.join(self.resolve(rel), self.MODE)

    def icsearcher_result(self):
        """Absolute path to ICSearcher's result dir (holds params{EXE}.csv).

        Override via ``NYCTEA_ICSEARCHER_RESULT`` (priority: env var > yaml).
        Relative paths are resolved against the repo root.
        """
        env_path = os.environ.get("NYCTEA_ICSEARCHER_RESULT")
        if env_path:
            return env_path if os.path.isabs(env_path) else str(REPO_ROOT / env_path)
        rel = self._get_yaml_value('paths', 'icsearcher_result', default='../ICSearcher/result')
        # icsearcher_result is conventionally a sibling of the nyctea repo, so
        # resolve relative to REPO_ROOT (.. goes above the repo, as expected).
        return rel if os.path.isabs(rel) else str((REPO_ROOT / rel).resolve())

    def params_csv(self):
        """Absolute path to the destabilizing-config CSV nyctea consumes.

        Reads ``params{EXE}.csv`` from ICSearcher's result dir for the current
        mode. ``EXE`` is '' when the repaired subset equals the full param set.
        """
        return os.path.join(self.icsearcher_result(), self.MODE, f"params{self.EXE}.csv")

    # ------------------------------------------------------------------ multi-instance ports
    # The MAVLink GCS port for instance ``i`` is 14540+i. This is the single
    # source of truth: both the SITL launch and the MAVLink monitor must agree
    # on it. Centralising here fixes the legacy bug where the port was built by
    # string-appending the index (``f"1455{device}"``), which only worked for
    # device indices < 10.
    BASE_MAVLINK_PORT = 14540

    def mavlink_port(self, i):
        """MAVLink UDP port the GCS/monitor listens on for instance ``i``."""
        return self.BASE_MAVLINK_PORT + int(i)

    # ------------------------------------------------------------------ defaults
    def _init_defaults(self):
        """Initialize with YAML values or defaults."""
        sim = self._get_yaml_value('simulation', default={}) or {}
        self.__dict__["MODE"] = self._get_yaml_value('mode', default="Ardupilot")
        self.__dict__["SPEED"] = sim.get('speed', 3)
        self.__dict__["HOME"] = sim.get('home', "AVC_plane")
        self.__dict__["DEBUG"] = sim.get('debug', True)
        self.__dict__["WIND_RANGE"] = sim.get('wind_range', [8, 10.7])

        window = sim.get('window', {}) or {}
        self.__dict__["WIDTH"] = window.get('width', 640)
        self.__dict__["HEIGHT"] = window.get('height', 480)

        altitude = sim.get('altitude', {}) or {}
        self.__dict__["LIMIT_H"] = altitude.get('limit_high', 50)
        self.__dict__["LIMIT_L"] = altitude.get('limit_low', 40)

        paths = self._get_yaml_value('paths', default={}) or {}
        self.__dict__["ARDUPILOT_LOG_PATH"] = paths.get('ardupilot_log', '/media/rain/data')
        self.__dict__["SITL_PATH"] = paths.get('sitl', "/home/rain/ardupilot/Tools/autotest/sim_vehicle.py")
        self.__dict__["PX4_RUN_PATH"] = paths.get('px4_run', '/home/rain/PX4-Autopilot')
        self.__dict__["JMAVSIM_PATH"] = paths.get('jmavsim', "/home/rain/PX4-Autopilot/Tools/jmavsim_run.sh")
        self.__dict__["MORSE_PATH"] = paths.get('morse', "/home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.py")

        model = self._get_yaml_value('model', default={}) or {}
        self.__dict__["HIDDEN"] = model.get('hidden', 256)
        self.__dict__["CAPACITY"] = model.get('capacity', 10000)
        self.__dict__["BATCH_SIZE"] = model.get('batch_size', 64)

        # Parallel multi-instance SITL. Override the count per-run with the
        # NYCTEA_INSTANCES env var (priority: env var > yaml).
        parallel = self._get_yaml_value('parallel', default={}) or {}
        env_instances = os.environ.get("NYCTEA_INSTANCES")
        if env_instances:
            try:
                instances = int(env_instances)
            except ValueError:
                raise ValueError(f"NYCTEA_INSTANCES must be an int, got {env_instances!r}")
        else:
            instances = int(parallel.get('instances', 1))
        if instances < 1:
            raise ValueError(f"parallel.instances must be >= 1, got {instances}")
        self.__dict__["INSTANCES"] = instances
        self.__dict__["INSTANCE_DIR"] = parallel.get('instance_dir', 'instance_{i}')

    # ------------------------------------------------------------------ sims auto-detect
    def _detect_sims(self):
        """Override non-existent paths with ones under ``sims/`` if available.

        If the user has placed the simulators in ``sims/`` (the default paths in
        the template point at home-directory locations that likely don't exist
        on this machine), redirect each missing path to its ``sims/`` equivalent.
        """
        sims = REPO_ROOT / "sims"
        if not sims.is_dir():
            return  # nothing to auto-detect

        def _lookup(key, sims_rel):
            """If the current path for *key* doesn't exist, try sims/sims_rel."""
            cur = self.__dict__.get(key)
            if cur and os.path.exists(cur):
                return  # already valid
            candidate = (sims / sims_rel).resolve()
            if candidate.exists():
                self.__dict__[key] = str(candidate)
                print(f"  auto-detected {key} = {candidate}")

        _lookup("ARDUPILOT_LOG_PATH", "data")
        _lookup("SITL_PATH",          "ardupilot/Tools/autotest/sim_vehicle.py")
        _lookup("PX4_RUN_PATH",       "PX4-Autopilot")
        _lookup("JMAVSIM_PATH",       "PX4-Autopilot/Tools/jmavsim_run.sh")
        _lookup("MORSE_PATH",         "ardupilot/libraries/SITL/examples/Morse/quadcopter.py")

    # ------------------------------------------------------------------ mode
    def _apply_mode(self, mode):
        """Populate all mode-derived constants. Called once during __init__."""
        self.__dict__["MODE"] = mode

        if mode == "Ardupilot":
            self.__dict__["SIM"] = "SITL"
            self.__dict__["STATUS_ORDER"] = list(STATUS_ORDER_COMMON)
            self.__dict__["PARAM_PART"] = list(PARAM_PART_ARDUPILOT)
        else:  # PX4
            self.__dict__["SIM"] = "Jmavsim"
            now_time = time.strftime("%Y-%m-%d", time.localtime())
            self.__dict__["PX4_LOG_PATH"] = f"{self.__dict__['PX4_RUN_PATH']}/build/px4_sitl_default/logs/{now_time}"
            # PX4 status interleaves the BiasA..D columns the RL state reads.
            status = list(STATUS_ORDER_COMMON)
            # Insert the bias columns after the rate channels (index 7).
            status[7:7] = PX4_BIAS_COLS
            self.__dict__["STATUS_ORDER"] = status
            self.__dict__["PARAM_PART"] = list(PARAM_PART_PX4)

        with open(self._param_file(mode), 'r') as f:
            param_name = pd.DataFrame(json.loads(f.read())).columns.tolist()
        self.__dict__["PARAM"] = param_name

        # EXE: '' when the repaired subset equals the full param set.
        self.__dict__["EXE"] = "" if len(self.PARAM_PART) == len(self.PARAM) else len(self.PARAM_PART)

        # ---- derived lengths ----
        self.__dict__["STATUS_LEN"] = len(self.STATUS_ORDER) - 1            # drop TimeS
        self.__dict__["PARAM_LEN"] = len(self.PARAM)

        self.validate_config()

    # ------------------------------------------------------------------ helpers
    def get(self, key, default=None):
        """Safe config getter with a default value."""
        return self.__dict__.get(key, default)

    def validate_config(self):
        """Validate critical configuration values."""
        for key in ('MODE', 'SITL_PATH', 'PARAM'):
            if not self.__dict__.get(key):
                raise ValueError(f"Missing required config: {key}")
        if self.MODE not in VALID_MODES:
            raise ValueError(f"Invalid MODE {self.MODE!r}")

        # Warn (do not fail) when a configured simulator path is absent; the
        # operator may be running a pipeline stage that does not need it.
        for path_key in ('SITL_PATH', 'PX4_RUN_PATH', 'ARDUPILOT_LOG_PATH'):
            path = self.__dict__.get(path_key)
            if path and not os.path.exists(path):
                print(f"Warning: Path does not exist: {path_key}={path}")

    # ------------------------------------------------------------------ dunder
    def __setattr__(self, name, value):
        # After construction the config is effectively frozen.
        raise self.ConstError(
            f"toolConfig is frozen at load time; cannot set {name!r}. "
            "Change mode via data/config.yaml or the NYCTEA_MODE env var."
        )

    def __getattr__(self, item):
        # Only called when the attribute is genuinely missing.
        raise AttributeError(item)


# The singleton, constructed once from config.yaml (mode overridable via
# NYCTEA_MODE). Importing this module is the only way to read config.
toolConfig = ToolConfig()
