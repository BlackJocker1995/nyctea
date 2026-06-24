# Nyctea — UAV Reinforcement Learning Repair Framework

A reinforcement-learning framework that **repairs destabilizing UAV controller
configurations**. Given a controller config that destabilizes a drone (found by
the companion fuzzer [ICSearcher](https://github.com/BlackJoker1995/ICSearcher)),
nyctea's DDPG agent learns to upload a corrected set of parameters mid-flight
that brings the drone's attitude back onto the desired trajectory. Supports both
ArduPilot and PX4.

```
ICSearcher  ── params.csv (destabilizing configs) ──>  nyctea  ── repaired configs + deviation histories
```

This is the modernized successor to the legacy `Cptool/`+`Rl/` codebase: a single
`nyctea/` package with a 5-stage pipeline, YAML config, stdlib parallelism
(no Ray), and a test suite.

## Requirements

Python ≥ 3.10. Install with [uv](https://docs.astral.sh/uv/) (or pip):

```bash
uv pip install -e ".[dev]"     # editable install + pytest
# or: pip install -e ".[dev]"
```

Simulation: ArduPilot [SITL](https://github.com/ArduPilot/ardupilot) (run via
`python3 .../sim_vehicle.py`); PX4 uses JMavSim (source build required).

## Configuration

All configuration lives in `data/config.yaml` (copy `data/config.yaml.example`
on first run). The config is **frozen at load time**; mode is resolved once as
`CLI flag > env var > yaml > default`.

| Field        | Env var                  | Meaning                                              |
|--------------|--------------------------|------------------------------------------------------|
| `mode`       | `NYCTEA_MODE`            | `Ardupilot` (default) or `PX4`                       |
| `instances`  | `NYCTEA_INSTANCES`       | concurrent simulator instances (default 1)           |
| `debug`      | `NYCTEA_DEBUG`           | `1`/`true` for DEBUG logging                         |
| `icsearcher_result` | `NYCTEA_ICSEARCHER_RESULT` | path to ICSearcher's `result/` dir (holds params.csv) |
| `model_dir`  | —                        | checkpoint dir (default `model/`, mode-scoped)       |
| `buffer_dir` | —                        | replay-buffer shard dir (default `model/buffer/`)    |

Simulator paths (`paths.sitl`, `paths.px4_run`, …) point at home-directory
builds in the template; if they don't exist and you've placed the simulators
under `sims/`, they are auto-detected.

### PX4 multi-instance setup

1. Create `Tools/sitl_multiple_run_single.sh` in your PX4 tree (one instance
   per call, isolated working dir).
2. Set the flight home in `Tools/jmavsim_run.sh` (`PX4_HOME_LAT/LON/ALT`).
3. Run headless with `HEADLESS=1`.
4. Switch mode via `NYCTEA_MODE=PX4` (or set `mode: PX4` in config.yaml).

## Pipeline

Five decoupled stages, each consuming the previous stage's on-disk artefacts
and re-runnable independently:

| Stage              | Command           | Reads                              | Writes                              |
|--------------------|-------------------|------------------------------------|-------------------------------------|
| 1. collect         | `nyctea-collect`  | ICSearcher `params{EXE}.csv`       | npz buffer shards                   |
| 2. train           | `nyctea-train`    | buffer shards                      | `model/{MODE}/*.pth` checkpoints    |
| 3. repair          | `nyctea-repair`   | `params{EXE}.csv`, checkpoints     | `params_repair{EXE}.csv`, dev npz   |
| 4. validate        | `nyctea-validate` | `params_repair{EXE}.csv`           | `params_validate{EXE}.csv`          |
| 5. analyze         | `nyctea-analyze`  | repair/validate CSVs, dev npz      | statistics + figures                |

```bash
# Example: train ArduPilot with 4 parallel simulator instances
NYCTEA_INSTANCES=4 nyctea-train

# Repair-test bad configs with wind disturbance (replaces the legacy _wind script)
nyctea-repair --disturbance wind

# Switch to PX4 for the whole pipeline
NYCTEA_MODE=PX4 nyctea-collect
```

The `--disturbance {none,wind,sensor,random}` flag on `nyctea-repair` replaces
the legacy `1.fix_test_{wind,sensor,random}.py` copy-pasted variants.

## Package layout

```
nyctea/                 core package
  config.py             frozen YAML config singleton + mode resolution
  logging_config.py     loguru sink + stdlib bridge
  concurrency.py        MultiInstanceRunner / LockedCsv (stdlib, no Ray)
  sim.py                SimManager / FixSimManager / MonitorFlight
  comms.py              DroneMavlink / MavlinkAPM / MavlinkPX4
  board_mavlink.py      .BIN / .ulg online log reader (RL state)
  anomaly.py            decomposed AnomalyDetector state machine
  params.py             param loading / scaling / Location geometry
  mavtool.py            log→CSV extraction + plotting helpers
  buffer.py             npz-shard replay buffer (replaces shared_memory)
  model_io.py           flock-guarded checkpoint read/write
  rl/
    env.py              DroneEnv (reset/step)
    actor_critic.py     Actor / Critic PyTorch MLPs
    learning_agent.py   DDPGAgent / ReLearningAgent
    reward.py           pure reward function
    actions.py          pure action↔config mapping
pipelines/              5-stage entry points + dispatcher
  __main__.py, collect.py, train.py, repair.py, validate.py, analyze.py
tests/                  pytest unit tests (no SITL required)
data/                   config.yaml(.example), param_*.json, missions
```

## Testing

```bash
pytest                   # 43 unit tests, no SITL needed
```

Covers the frozen config, `LockedCsv` concurrency, the pure reward/action
functions, the `AnomalyDetector` state machine, and the npz replay buffer.

## Relationship to ICSearcher

nyctea and ICSearcher are sibling projects that chain together: ICSearcher
fuzzes for destabilizing controller configs and emits `params.csv`; nyctea reads
those (`result != "pass"` rows) and learns to repair them. This refactor mirrors
ICSearcher's engineering patterns (YAML config, stdlib parallelism, pipeline
stages, npz artefacts, loguru, pytest) while keeping nyctea an independent,
separately-released project.

## Experimental Setup

![APM](/fig/airsim.png)
![Jmavsim](/fig/jmavsim.jpg)
![Real Drone](/fig/zd500.jpg)
