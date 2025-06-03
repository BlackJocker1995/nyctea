# Nyctea - UAV Reinforcement Learning Framework

A reinforcement learning based drone control framework supporting both Ardupilot and PX4 flight control systems.

## Directory Structure

```
nyctea/
├── Cptool/                # Core tool modules
│   ├── config.py          # Configuration file
│   ├── simManager.py      # Simulator manager
│   └── ...                # Other core modules
├── Rl/                    # Reinforcement learning code
│   ├── actor_critic.py    # Actor-Critic implementation
│   ├── env.py             # RL environment
│   └── learning_agent.py  # Learning agent
├── model/                 # Trained models
│   ├── Ardupilot/         # Ardupilot models
│   └── PX4/               # PX4 models
├── validation/            # Validation data
├── csv_data/              # Datasets
├── fig/                   # Image resources
└── ...                    # Other files and directories
```

## Requirements

### Software Dependencies
- Python 3.6+
- Required packages:
  ```bash
  pip3 install numpy pandas pymavlink pyulog keras tensorflow torch
  ```

### Simulator Support
1. **Ardupilot SITL**:
   ```bash
   python3 {Ardupilot_path}/Tools/autotest/sim_vehicle.py --location=AVC_plane --out=127.0.0.1:14550 -v ArduCopter -w -S {toolConfig.SPEED}
   ```

2. **PX4 jMAVSim**:
   - Modify home position in `Tools/jmavsim_run.sh`
   - Set `HEADLESS=1` for no-GUI mode

## Quick Start

1. Configure `Cptool/config.py`:
   ```python
   # Select simulator type
   toolConfig.select_mode("Ardupilot")  # or "PX4"
   
   # Set paths
   ARDUPILOT_PATH = "your_ardupilot_path"
   PX4_PATH = "your_px4_path"
   ```

2. Train model:
   ```bash
   python3 0.train_ddpg.py
   ```

3. Test fixes:
   ```bash
   python3 1.fix_test.py
   ```

## Main Features

### Training Scripts
- `0.train_ddpg.py`: Main training script
- `0.train_ddpg_thread.py`: Multi-thread training version

### Testing Scripts
- `1.fix_test.py`: Basic testing
- `1.fix_test_thread.py`: Multi-thread testing
- `1.fix_test_wind.py`: Wind disturbance testing

### Data Analysis
- `2.average_loss.py`: Loss analysis
- `3.distribution.py`: Data distribution analysis

## Configuration Options

In `Cptool/config.py` you can configure:
- `MODE`: Simulator type (Ardupilot/PX4)
- `SPEED`: Simulation speed
- `DEBUG`: Debug mode
- Various path settings

## Experimental Setup

![APM](/fig/airsim.png) 
![Jmavsim](/fig/jmavsim.jpg)
![Real Drone](/fig/zd500.jpg)
