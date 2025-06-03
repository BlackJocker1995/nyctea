# Nyctea - UAV Reinforcement Learning Framework

A reinforcement learning based drone attitude correction framework supporting both Ardupilot and PX4 flight control systems.

## Requirement

Python package requirement: numpy ; pandas ; pymavlink ; pyulog ; pytorch

OS: The program is only tested in Ubuntu 18.04 and 20.04 (recommend).

```bash
pip3 install numpy pandas pymavlink pyulog eventlet keras tensorflow torch
```

Simulation requirement: Ardupilot [SITL](https://github.com/ArduPilot/ardupilot). We suggest applying python3 to run SITL simulator. Jmavsim for PX4, which requires source build in PX4 file.

The initializer of Ardupilot simulator needs to change the path in the file `Cptool.config.py` with item `SITL_PATH`.

For example,
```bash
python3 {Your Ardupilot path}/Tools/autotest/sim_vehicle.py --location=AVC_plane --out=127.0.0.1:14550 -v ArduCopter -w -S {toolConfig.SPEED} "
```

If you want to run PX4 evaluation in multiple threads, you should change the following code in PX4-Ardupilot.

1. Create the file `Tools/sitl_multiple_run_single.sh` and add content next:

```bash
#!/bin/bash
sitl_num=0
[ -n "$1" ] && sitl_num="$1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
src_path="$SCRIPT_DIR/.."
build_path=${src_path}/build/px4_sitl_default
pkill  -f "px4 -i $sitl_num"
sleep 1
export PX4_SIM_MODEL=iris
working_dir="$build_path/instance_$sitl_num"
[ ! -d "$working_dir" ] && mkdir -p "$working_dir"
pushd "$working_dir" &>/dev/null
echo "starting instance $sitl_num in $(pwd)"
../bin/px4 -i $sitl_num -d "$build_path/etc" -s etc/init.d-posix/rcS # >out.log 2>err.log &
popd &>/dev/null
```
2. Change the flight home point in `Tools/jmavsim_run.sh`

```bash
export PX4_HOME_LAT=40.072842
export PX4_HOME_LON=-105.230575
export PX4_HOME_ALT=0.000000
export PX4_SIM_SPEED_FACTOR=3 # speed
```

3. To use the simulation without the jMAVSim GUI, please set the following environment variable `HEADLESS=1`

4. Change the default mode in `Cptool.config.py` to ` toolConfig = ToolConfig() and
toolConfig.select_mode("PX4")`

## Deployment

The configuration is in `Cptool.config.py`.

If you want to try PX4 simulation, change the sentence `toolConfig.select_mode("Ardupilot")` to `toolConfig.select_mode("PX4")`

## Configuration of System config.py

* ARDUPILOT_LOG_PATH: log path of ardupilot running, noted that, the path needs to have a flag file "logs/LASTLOG.TXT". Or you can manually run the simulation at first in {ARDUPILOT_LOG_PATH} to auto generate flag file.

The log path for PX4 is in `{PX4_Path}/build/px4_sitl_default/logs/`, which is no need to change.

* MODE: 1. Ardupilot 2. PX4.

* SPEED: simulation speed, 3 default.

* HOME: 1. None, use the default location to start; 2. AVC, use AVC location and mission.

* DEBUG: display debug information.

* ARDUPILOT_LOG_PATH: Path to save the Ardupilot logs.

* AIRSIM_PATH: if select airsim, you should set the execution path.

* PX4_PATH: if select PX4, you should set the execution path, like `xx/xx/PX4-Autopilot`.

* ARDUPILOT_PATH: Ardupilot running path, e.g., `xx/xx/Autopilot`.

## Main Features

### Training Scripts
- [`0.train_ddpg.py`](0.train_ddpg.py): Main training script
- [`0.train_ddpg_thread.py`](0.train_ddpg_thread.py): Multi-thread training version

### Testing Scripts
- [`1.fix_test.py`](1.fix_test.py): Basic testing
- [`1.fix_test_thread.py`](1.fix_test_thread.py): Multi-thread testing
- [`1.fix_test_wind.py`](1.fix_test_wind.py): Wind disturbance testing

### Data Analysis
- [`2.average_loss.py`](2.average_loss.py): Loss analysis
- [`3.distribution.py`](3.distribution.py): Data distribution analysis

## Experimental Setup

![APM](/fig/airsim.png)
![Jmavsim](/fig/jmavsim.jpg)
![Real Drone](/fig/zd500.jpg)
