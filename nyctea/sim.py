"""Simulator lifecycle management.

Mirrors ICSearcher's ``icsearcher/sim.py`` in structure: a :class:`SimManager`
that spawns and tears down the SITL/JMavSim process via ``pexpect``, wires up
the online MAVLink connection (:mod:`nyctea.comms`), the onboard log reader
(:class:`BoardMavlink`), and the in-flight anomaly monitor (:class:`MonitorFlight`).
:class:`FixSimManager` is the runnable variant (also a ``multiprocessing.Process``)
used by the RL env.

Key changes from the legacy ``Cptool/simManager.py``:

- **Per-instance port/dir isolation** via ``toolConfig.mavlink_port(i)`` and
  ``ardu/px4_instance_path(i)`` — the single source of truth. The legacy path
  built ports by string-appending the index (``f"1455{drone_i}"``) and shared one
  cwd, so parallel instances raced on ``eeprom.bin`` and only worked for i<10.
- **PX4 home/speed via a per-call ``env`` dict** instead of mutating the global
  ``os.environ``, so concurrent instances don't overwrite each other's
  ``PX4_HOME_*``.
- **MonitorFlight uses the decomposed :class:`AnomalyDetector`** state machine
  instead of a 200-line inline loop.
- **No gnome-terminal spawning** for the simulator (kept inline via pexpect).
- **``logging`` → ``loguru``.**
"""
import math
import multiprocessing
import os
import time
from typing import Type

import pexpect
from loguru import logger
from pymavlink import mavwp

from nyctea.anomaly import AnomalyDetector, build_detector
from nyctea.board_mavlink import BoardMavlink, BoardMavlinkAPM, BoardMavlinkPX4
from nyctea.comms import DroneMavlink, MavlinkAPM, MavlinkPX4
from nyctea.config import toolConfig
from nyctea.params import Location


class SimManager:
    """Owns the SITL/JMavSim lifecycle and the MAVLink/board/monitor wiring.

    A single instance corresponds to one simulator (one UDP port + one working
    directory). For parallel training, build one :class:`FixSimManager` per
    worker via :class:`~nyctea.concurrency.MultiInstanceRunner`.
    """

    def __init__(self, debug: bool = False):
        self._sim_task = None
        self._sitl_task = None
        self.online_mavlink: DroneMavlink = None
        self.board_mavlink: BoardMavlink = None
        self.mav_monitor: "MonitorFlight" = None

    # --------------------------------------------------------------- simulator
    def start_multiple_sim(self, drone_i=0):
        """Start the JMavSim visualization backend for PX4 instance ``drone_i``."""
        if toolConfig.SIM == 'Jmavsim':
            port = 4560 + int(drone_i)
            cmd = f'{toolConfig.JMAVSIM_PATH} -p {port} -l'
            self._sim_task = pexpect.spawn(
                cmd, cwd=toolConfig.PX4_RUN_PATH, timeout=30, encoding='utf-8')
            logger.debug("Init px4 Jmavsim description.")

    def start_multiple_sitl(self, drone_i=0):
        """Start one SITL instance for ``drone_i`` with isolated port + cwd.

        ArduPilot: the instance gets its own ``instance_{i}`` working dir under
        ``ARDUPILOT_LOG_PATH`` (so ``eeprom.bin`` / ``mav.parm`` / ``logs/`` don't
        collide) and ports derived from ``toolConfig.mavlink_port``.
        PX4: home/speed are passed via a per-call ``env`` dict rather than
        mutating ``os.environ`` so concurrent instances stay independent.
        """
        if toolConfig.MODE == 'Ardupilot':
            work_dir = toolConfig.ardu_instance_path(drone_i)
            os.makedirs(work_dir, exist_ok=True)
            for stale in ('eeprom.bin', 'mav.parm'):
                stale_path = os.path.join(work_dir, stale)
                if os.path.exists(stale_path):
                    os.remove(stale_path)

            base_port = toolConfig.mavlink_port(drone_i)
            if toolConfig.HOME is not None:
                cmd = (f"python3 {toolConfig.SITL_PATH} --location={toolConfig.HOME} "
                       f"--out=127.0.0.1:{base_port + 10} --out=127.0.0.1:{base_port} "
                       f"--out=127.0.0.1:{base_port + 20} "
                       f"-v ArduCopter -w -S {toolConfig.SPEED} --instance {drone_i}")
            else:
                cmd = (f"python3 {toolConfig.SITL_PATH} "
                       f"--out=127.0.0.1:{base_port + 10} --out=127.0.0.1:{base_port} "
                       f"--out=127.0.0.1:{base_port + 20} "
                       f"-v ArduCopter -w -S {toolConfig.SPEED} --instance {drone_i}")
            self._sitl_task = pexpect.spawn(cmd, cwd=work_dir, timeout=30, encoding='utf-8')

        elif toolConfig.MODE == 'PX4':
            instance_path = toolConfig.px4_instance_path(drone_i)
            params_file = os.path.join(instance_path, 'eeprom', 'parameters_10016')
            if os.path.exists(params_file):
                os.remove(params_file)

            if toolConfig.SIM == 'Jmavsim':
                cmd = f"{toolConfig.PX4_RUN_PATH}/Tools/sitl_multiple_run_single.sh {drone_i}"
                # Per-call env dict — does NOT mutate the global os.environ, so
                # concurrent PX4 instances keep independent home/speed.
                env = dict(os.environ)
                env['PX4_SIM_SPEED_FACTOR'] = f"{toolConfig.SPEED}"
                if toolConfig.HOME is None:
                    env['PX4_HOME_LAT'] = "-35.363261"
                    env['PX4_HOME_LON'] = "149.165230"
                    env['PX4_HOME_ALT'] = "583.730592"
                else:
                    env['PX4_HOME_LAT'] = "40.072842"
                    env['PX4_HOME_LON'] = "-105.230575"
                    env['PX4_HOME_ALT'] = "0.000000"
                self._sitl_task = pexpect.spawn(
                    cmd, cwd=toolConfig.PX4_RUN_PATH, timeout=30, encoding='utf-8', env=env)

        logger.info(f"Start {toolConfig.MODE} --> [{toolConfig.SIM} - {drone_i}]")

    # --------------------------------------------------------------- wiring
    def online_mavlink_init(self, mavlink_class: Type[DroneMavlink] = DroneMavlink,
                            drone_i=0):
        """Build + connect the online command/control MAVLink for instance ``drone_i``."""
        self.online_mavlink = mavlink_class(drone_i)
        self.online_mavlink.connect()
        if toolConfig.MODE == 'Ardupilot':
            if self.online_mavlink.ready2fly():
                return True
        elif toolConfig.MODE == 'PX4':
            while True:
                line = self._sitl_task.readline()
                if 'notify' in line:
                    self._sitl_task.send("param set NAV_RCL_ACT 0 \n")
                    time.sleep(0.1)
                    self._sitl_task.send("param set NAV_DLL_ACT 0 \n")
                    time.sleep(0.1)
                    self._sitl_task.send("param set CBRK_FLIGHTTERM 0 \n")
                    return True

    def board_mavlink_init(self):
        """Build the onboard log reader for the current mode."""
        if toolConfig.MODE == "PX4":
            self.board_mavlink = BoardMavlinkPX4()
        else:
            self.board_mavlink = BoardMavlinkAPM()

    def mav_monitor_init(self, port):
        """Build the in-flight anomaly monitor bound to ``port``."""
        self.mav_monitor = MonitorFlight(port)

    # --------------------------------------------------------------- teardown
    def stop_sitl(self):
        if self._sitl_task is not None:
            self._sitl_task.sendcontrol('c')
            while True:
                line = self._sitl_task.readline()
                if not line:
                    break
            self._sitl_task.close(force=True)
            logger.info('Stop SITL task.')

    def stop_sim(self):
        if self._sim_task is not None:
            self._sim_task.sendcontrol('c')
            self._sim_task.close(force=True)
            logger.info('Stop Sim task.')


class FixSimManager(SimManager, multiprocessing.Process):
    """Runnable :class:`SimManager` used by the RL env (a separate process)."""

    def __init__(self, debug: bool = False):
        SimManager.__init__(self, debug)
        multiprocessing.Process.__init__(self)


class MonitorFlight(multiprocessing.Process):
    """In-flight anomaly monitor, driven by the :class:`AnomalyDetector`.

    Listens on ``port`` for STATUSTEXT/position/mission messages, feeds them to
    the decomposed detector state machine, and pushes the terminal outcome onto
    ``msg_queue`` as ``[result, timestamp]`` when the flight ends. This replaces
    the legacy 200-line inline loop with a thin driver over the testable
    :class:`AnomalyDetector`.
    """

    def __init__(self, port):
        super().__init__()
        self.msg_queue = multiprocessing.Queue()
        from pymavlink import mavutil
        self.master = mavutil.mavlink_connection(f'udp:0.0.0.0:{port}')
        result = self.master.wait_heartbeat(timeout=30)
        if result is None:
            raise ValueError(f"Fail to connect 'udp:0.0.0.0:{port}'.")
        logger.info("Heartbeat from system (system {} component {}) from {}".format(
            self.master.target_system, self.master.target_component, port))

    def get_msg(self, msg_type, block=False):
        return self.master.recv_match(type=msg_type, blocking=block)

    def gcs_msg_request(self):
        """Send a GCS heartbeat (PX4 needs manual GCS heartbeats)."""
        from pymavlink import mavutil
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def run(self) -> None:
        logger.info('Start error monitor.')
        detector = build_detector(toolConfig.mission_file())
        while True:
            if toolConfig.MODE == "PX4":
                self.gcs_msg_request()
            status_message = self.get_msg(["STATUSTEXT"])
            position_msg = self.get_msg(["GLOBAL_POSITION_INT", "MISSION_CURRENT"])

            detector.on_status(status_message)
            # The two position message types come back from one recv_match call;
            # dispatch each to the right handler.
            if position_msg is not None:
                if position_msg.get_type() == "MISSION_CURRENT":
                    detector.on_mission_current(position_msg)
                elif position_msg.get_type() == "GLOBAL_POSITION_INT":
                    detector.on_position(position_msg)

            if detector.result is not None:
                break
            if detector.timed_out():
                break

        logger.info(f"Monitor result: {detector.result}")
        self.msg_queue.put([detector.result, time.time()])
