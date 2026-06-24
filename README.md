# Nyctea

Reinforcement-learning **attitude repair** for UAV autopilot controllers. Given a
controller configuration that destabilizes a drone (found by the companion
fuzzer [ICSearcher](https://github.com/BlackJoker1995/ICSearcher)), nyctea's
DDPG agent learns to upload a corrected set of parameters mid-flight that brings
the drone's attitude back onto the desired trajectory. It supports both
**ArduPilot** and **PX4** firmware.

```
ICSearcher  ── params.csv (destabilizing configs) ──>  nyctea  ── repaired configs + deviation histories
```

This is the modernized successor to the legacy `Cptool/`+`Rl/` codebase: a single
`nyctea/` package with a 5-stage pipeline, YAML config, stdlib parallelism
(no Ray), and a test suite — mirroring ICSearcher's engineering patterns.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Deployment walkthrough](#deployment-walkthrough)
   - [Step 1 — Install the Python environment](#step-1--install-the-python-environment)
   - [Step 2 — Provision the simulators](#step-2--provision-the-simulators)
   - [Step 3 — Configure the run](#step-3--configure-the-run)
   - [Step 4 — Run the pipeline](#step-4--run-the-pipeline)
5. [Configuration reference](#configuration-reference)
6. [Testing](#testing)
7. [Notes & troubleshooting](#notes--troubleshooting)

---

## How it works

nyctea is a five-stage pipeline. Each stage is a standalone console command in
`pipelines/`; the firmware (ArduPilot / PX4) is selected once in
`data/config.yaml` and every stage branches on it.

```
Stage 1  collect     Fly ICSearcher's destabilizing configs, gather DDPG transitions
Stage 2  train       Train the DDPG actor/critic on the replay buffer
Stage 3  repair      Repair-test bad configs with the trained agent
Stage 4  validate    Re-fly repaired configs to confirm stability
Stage 5  analyze     Loss / detection / deviation analysis and figures
```

The repair agent is a PyTorch DDPG actor-critic. The replay buffer is a set of
npz shards (workers append, the learner reloads into memory and samples there) —
no Ray, no `shared_memory.ShareableList`.

---

## Repository layout

```
nyctea/                   Core package
  config.py               Frozen toolConfig singleton (loaded from data/config.yaml)
  logging_config.py       Unified loguru setup
  concurrency.py          Multi-instance SITL orchestration (multiprocessing + flock)
  params.py               Parameter loading / scaling / Location geometry
  comms.py                MAVLink command/control (DroneMavlink, APM/PX4)
  sim.py                  Simulator lifecycle (SimManager / FixSimManager / MonitorFlight)
  board_mavlink.py        Onboard .BIN / .ulg log reader (RL state segments)
  anomaly.py              In-flight anomaly detector (decomposed state machine)
  mavtool.py              Log→CSV extraction + plotting helpers
  buffer.py               npz-shard replay buffer (replaces shared_memory)
  model_io.py             flock-guarded checkpoint read/write
  rl/                     The DDPG agent (env, actor_critic, learning_agent, reward, actions)
data/                     config.yaml, param_*.json, mission*.txt, fitCollection*.txt
pipelines/                The five stage entry points (collect, train, repair, validate, analyze)
scripts/setup_sims.sh     Bootstrap: clone & build ArduPilot SITL and PX4 + JMavSim
tests/                    Pure-function unit tests (no SITL required)
pyproject.toml            Project manifest (deps + nyctea-* console entry points)
```

---

## Requirements

- **OS:** Ubuntu 20.04+. The simulators and their build
  toolchains are Linux-centric.
- **Python:** 3.9 – 3.11. Python 3.12 has known compatibility issues with
  ArduPilot's build system (the `imp` module was removed) and is not supported.
- **Simulators:** ArduPilot SITL and/or PX4-Autopilot with JMavSim. The
  bootstrap script builds them for you (see Step 2).

### Prerequisites (system packages — install once, needs sudo)

Before anything else, install the system build tools.

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git \
    python3 python3-pip python3-dev python3-venv \
    build-essential \
    ccache \
    wget curl
```

---

## Deployment walkthrough

### Step 1 — Install the Python environment

All Python dependencies are declared in `pyproject.toml` and managed with
[uv](https://docs.astral.sh/uv/). One command resolves the tree, creates
`.venv`, installs the project (plus its `nyctea-*` console commands), and
writes `uv.lock`:

```bash
# 1a. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1b. From the repository root, sync everything
cd nyctea
uv sync

# 1c. (ArduPilot only) add the firmware build/runtime tools the SITL needs
uv sync --group ardupilot
```

This brings in the scientific stack (numpy, pandas, scipy, scikit-learn), the
drone-comms stack (pymavlink, pyulog, pexpect), the RL backend (PyTorch), and
dev tools (pytest). **PX4-only users can stop at `uv sync`.** ArduPilot users
add the `ardupilot` group (MAVProxy, dronekit-sitl, etc.) so `sim_vehicle.py`
runs inside the project venv — no separate system `pip install` needed.

### Step 2 — Provision the simulators

nyctea re-flies real firmware to train and evaluate the repair agent, so you
need the ArduPilot and/or PX4 **SITL** (Software-In-The-Loop) simulators built
from source. The `scripts/setup_sims.sh` helper does all of it and is written
as a teaching script — it prints what it's doing at each step.

```bash
# Make the script executable (one-time — needed after a fresh clone)
chmod +x scripts/setup_sims.sh

# Clone + build both simulators. Everything lands under ./sims/ inside the
# repo, so the whole install is self-contained and removed by `rm -rf sims`.
./scripts/setup_sims.sh

# Or build only one firmware
./scripts/setup_sims.sh --ardupilot
./scripts/setup_sims.sh --px4
```

**What it clones + builds** (all under the repository, ~10 GB):

```
nyctea/
├── sims/                         created by the script
│   ├── ardupilot/                cloned from github.com/ardupilot/ardupilot
│   ├── PX4-Autopilot/            cloned from github.com/PX4/PX4-Autopilot
│   └── data/                     flight logs (.BIN / .ulg) live here
├── data/config.yaml              the script rewrites 'paths:' to match
└── ...
```

Run it as your **normal user** (the same one that ran the Prerequisites
`sudo apt-get`). The script itself does not use `sudo` — but the firmware
repos' own setup scripts (ArduPilot's `install-prereqs-ubuntu.sh` and PX4's
`ubuntu.sh`) do, and will prompt for your password when they run. Do **not**
wrap the whole script in `sudo` (`sudo ./scripts/...`): it would run `uv`/builds
as root and break the project venv. The first build downloads a compiler
toolchain and is slow (20–60 min); later pipeline runs reuse the binaries.

> **Custom locations?** Override with env vars (absolute paths):
> ```bash
> SIM_ROOT=/opt/sims DATA_DIR=/var/lib/nyctea ./scripts/setup_sims.sh
> ```
> **Different firmware version?**
> ```bash
> ARDUPILOT_BRANCH=Copter-4.5.2 PX4_BRANCH=v1.14.0 ./scripts/setup_sims.sh
> ```
> **Uninstall:** `rm -rf sims` removes everything the script created.

### Step 3 — Configure the run

If you ran `setup_sims.sh`, the `paths:` block in `data/config.yaml` is already
pointed at `sims/ardupilot`, `sims/PX4-Autopilot`, and `sims/data` — no manual
editing needed. The one thing you must still choose is the **firmware mode**,
which selects which simulator the pipeline drives and is **frozen at load
time** (there is no runtime switching):

```yaml
mode: Ardupilot    # or "PX4"
```

> **Quick mode switch without editing the file:** set the `NYCTEA_MODE`
> environment variable before running any stage:
> ```bash
> NYCTEA_MODE=PX4 uv run nyctea-collect
> ```
> Priority is: env var > `data/config.yaml`'s `mode` field.

> **Point nyctea at ICSearcher's output.** The repair stages read ICSearcher's
> destabilizing configs from `params{EXE}.csv`. Set `paths.icsearcher_result`
> in `data/config.yaml` to ICSearcher's `result/` directory (or the
> `NYCTEA_ICSEARCHER_RESULT` env var).

See [Configuration reference](#configuration-reference) for every field.

### Step 4 — Run the pipeline

**Before running, make sure you've cloned + built the simulators first**
(`./scripts/setup_sims.sh`, see Step 2 above). The pipeline launches SITL
simulators — without them it cannot fly or repair.

Run the stages in order. Each stage is a console command — no `python` or path
needed (the `nyctea-*` entry points are installed by `uv sync`):

```bash
# Stage 1 — collect RL transitions (flies ICSearcher's bad configs)
uv run nyctea-collect
uv run nyctea-collect --instances 4         # 4 parallel SITLs (faster)

# Stage 2 — train the DDPG repair agent on the buffer shards
uv run nyctea-train
uv run nyctea-train --instances 4

# Stage 3 — repair-test bad configs with the trained agent
uv run nyctea-repair
uv run nyctea-repair --disturbance wind     # replaces the legacy _wind script

# Stage 4 — re-fly repaired configs to verify
uv run nyctea-validate

# Stage 5 — loss / detection / deviation analysis
uv run nyctea-analyze
```

Prefer the module dispatcher? `uv run python -m pipelines <stage> [args...]`
works too (e.g. `uv run python -m pipelines repair --disturbance wind`).

**Outputs** (git-ignored):

- `model/{MODE}/` — DDPG checkpoints (`actor.pth`, `critic.pth`, optimizers).
- `model/buffer/{MODE}/` — replay-buffer npz shards.
- `validation/{MODE}/` — repair/validate CSVs and per-config deviation npz.

---

## Configuration reference

`data/config.yaml` is the single source of configuration.

| Field | Description |
|-------|-------------|
| `mode` | `Ardupilot` or `PX4`. Frozen at load time; override per-run with `NYCTEA_MODE`. |
| `simulation.speed` | SITL simulation speed factor. |
| `simulation.home` | ArduPilot `--location` / PX4 home region tag. |
| `simulation.debug` | Verbose logging when true (or `NYCTEA_DEBUG`). |
| `simulation.wind_range` | Wind speed range for sampling. |
| `simulation.altitude.{limit_high,limit_low}` | Altitude bounds. |
| `paths.ardupilot_log` | ArduPilot log directory. **Must contain** `logs/LASTLOG.TXT` (auto-created by `setup_sims.sh`). |
| `paths.sitl` | Path to ArduPilot's `sim_vehicle.py`. |
| `paths.px4_run` | PX4-Autopilot source root. The PX4 log path is derived from this automatically. |
| `paths.jmavsim` | Path to PX4's `jmavsim_run.sh`. |
| `paths.morse` | Optional alternate simulator launcher. |
| `paths.icsearcher_result` | Path to ICSearcher's `result/` dir (holds `params{EXE}.csv`). Override with `NYCTEA_ICSEARCHER_RESULT`. |
| `paths.model_dir` | Checkpoint directory (default `model/`, mode-scoped). |
| `paths.buffer_dir` | Replay-buffer shard directory (default `model/buffer/`). |
| `model.{hidden,capacity,batch_size}` | DDPG network width + replay buffer hyperparameters. |
| `parallel.instances` | Number of concurrent SITL instances (default 1). Override per-run with `NYCTEA_INSTANCES` or `--instances`. |
| `parallel.instance_dir` | Per-instance working subdir template (`{i}` = index, default `instance_{i}`). |
| `param_files.{ardupilot,px4}` | Parameter-definition JSONs (default `data/param_*.json`). |
| `missions.fit_collection.{ardupilot,px4}` | Mission files used for fitness collection. |

---

## Testing

Pure-function unit tests that do **not** require a live SITL simulator:

```bash
uv run pytest
```

Coverage spans the frozen config loader, `LockedCsv` concurrency, the pure
reward/action functions, the decomposed `AnomalyDetector` state machine, and the
npz replay buffer. Tests requiring heavy backends (torch, pymavlink) live in the
RL modules, which are exercised end-to-end against SITL rather than unit-tested.

---

## Notes & troubleshooting

**No GUI needed for unattended training.** PX4 SITL launches with `HEADLESS=1`
(no JMavSim 3D window); ArduPilot SITL never opens a GUI either. The anomaly
detector reads flight telemetry over MAVLink, not the 3D view. To see the
JMavSim window for debugging, remove `HEADLESS=1` from the PX4 build in
`scripts/setup_sims.sh`.

**The lockfile is not committed.** Generate it locally with `uv lock`
(the Ray-free dependency graph resolves in a couple of seconds). Commit
`uv.lock` if you want reproducible installs across machines.

**Multi-instance (parallel) collect & train.** Both SITL-bound stages run N
simulator instances concurrently — each on its own UDP port (`14540 + i`) and in
its own working directory, coordinated by the replay-buffer shards. Set the
count any of these ways (priority: flag > env var > yaml):

```bash
uv run nyctea-collect --instances 4
NYCTEA_INSTANCES=4 uv run nyctea-train
# or set it permanently in data/config.yaml:
#   parallel:
#     instances: 4
```

Per-instance ArduPilot state is isolated under `ARDUPILOT_LOG_PATH/instance_{i}/`,
PX4 under the build tree's `instance_{i}/`, so concurrent instances never
collide.

**Logging.** Every module uses [loguru](https://github.com/Deligan/loguru)
through `nyctea/logging_config.py`. `setup_logging(debug=...)` configures one
unified stderr sink and bridges any remaining stdlib `logging` calls into it,
so nothing is silenced.

## Experimental Setup

![APM](/fig/airsim.png)
![Jmavsim](/fig/jmavsim.jpg)
![Real Drone](/fig/zd500.jpg)
