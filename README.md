# Nyctea

Reinforcement-learning attitude repair for UAV autopilot controllers. A DDPG agent learns to upload corrected parameters mid-flight to bring a destabilized drone back onto the desired trajectory. Supports ArduPilot and PX4 firmware.

```
ICSearcher ── params.csv ──> nyctea ── repaired configs + deviations
```

5-stage pipeline (collect → train → repair → validate → analyze), YAML config, multi-process SITL orchestration (no Ray).

---

## How it works

Each stage is a console command in `pipelines/`; firmware (ArduPilot / PX4) is selected once in `data/config.yaml`.

| Stage | Task |
|-------|------|
| collect | Fly destabilizing configs, gather DDPG transitions |
| train | Train the actor/critic on the replay buffer |
| repair | Repair-test bad configs with the trained agent |
| validate | Re-fly repaired configs to confirm stability |
| analyze | Loss / detection / deviation analysis |

---

## Requirements

- **OS:** Ubuntu 20.04+
- **Python:** 3.9 – 3.11
- **Simulators:** ArduPilot SITL and/or PX4 SITL

```bash
# System deps
sudo apt-get install -y git python3 python3-venv build-essential ccache wget curl

# Python env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv sync --group ardupilot  # ArduPilot only
```

---

## Setup

Build simulators (self-contained under `./sims`, removed by `rm -rf sims`):

```bash
chmod +x scripts/setup_sims.sh
./scripts/setup_sims.sh
./scripts/setup_sims.sh --ardupilot  # or just one firmware
```

Override locations: `SIM_ROOT=/opt/sims DATA_DIR=/var/lib/nyctea`

---

## Config

`data/config.yaml` — set `mode: Ardupilot` or `mode: PX4`. Override per-run with `NYCTEA_MODE=PX4`.  
Point `paths.icsearcher_result` at ICSearcher's `result/` dir (or use `NYCTEA_ICSEARCHER_RESULT`).

| Field | Description |
|-------|-------------|
| `mode` | `Ardupilot` / `PX4` (overridable via `NYCTEA_MODE`) |
| `simulation.speed` | SITL simulation speed factor |
| `simulation.home` | ArduPilot `--location` / PX4 home region |
| `simulation.debug` | Verbose logging (`NYCTEA_DEBUG`) |
| `simulation.wind_range` | Wind speed range |
| `paths.ardupilot_log` | ArduPilot log dir |
| `paths.sitl` | Path to `sim_vehicle.py` |
| `paths.px4_run` | PX4 source root |
| `paths.jmavsim` | Path to `jmavsim_run.sh` |
| `paths.icsearcher_result` | ICSearcher result dir |
| `paths.model_dir` | Checkpoint dir (default `model/`) |
| `paths.buffer_dir` | Replay-buffer shard dir (default `model/buffer/`) |
| `model.{hidden,capacity,batch_size}` | DDPG network width and buffer size |
| `parallel.instances` | Concurrent SITL instances (default 1) |

---

## Run

```bash
uv run nyctea-collect
uv run nyctea-collect --instances 4
uv run nyctea-train
uv run nyctea-repair --disturbance wind
uv run nyctea-validate
uv run nyctea-analyze

# module dispatcher:
# uv run python -m pipelines <stage> [args...]
```

Outputs (git-ignored): `model/{MODE}/` (checkpoints), `model/buffer/{MODE}/` (replay shards), `validation/{MODE}/` (CSVs + npz).

---

## Test

```bash
uv run pytest
```

Covers config loader, `LockedCsv` concurrency, reward/action functions, `AnomalyDetector`, npz buffer. Heavy backends (torch, pymavlink) are exercised against SITL.
