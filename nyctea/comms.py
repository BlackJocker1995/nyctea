"""MAVLink command/control layer.

Mirrors ICSearcher's ``icsearcher/comms.py`` in structure: a base
:class:`DroneMavlink` that owns the master connection and the mission/param
command surface, plus ``MavlinkAPM`` / ``MavlinkPX4`` firmware subclasses that
also run as a background ``multiprocessing.Process`` for the recv loop.

Key changes from the legacy ``Cptool/mavlink.py``:

- **No Ray.** the legacy module imported ``ray`` but only used it elsewhere;
  removed entirely.
- **Port via the config singleton.** the port is taken from
  ``toolConfig.mavlink_port(i)`` (the single source of truth) instead of being
  string-appended (``1455{drone_i}``), which fixes the i>=10 truncation bug.
- **``logging`` → ``loguru``.**
"""
import multiprocessing
import random
import time
from typing import Type

from loguru import logger
from pymavlink import mavutil, mavwp

from nyctea.config import toolConfig
from nyctea.params import load_param, select_sub_dict


class DroneMavlink:
    """Online MAVLink command/control connection to a single simulator instance.

    Not a process itself; the APM/PX4 subclasses add the recv-loop process. The
    instance binds to ``toolConfig.mavlink_port(drone_i)`` for the GCS port.
    """

    def __init__(self, drone_i=0, recv_msg_queue=None, send_msg_queue=None):
        self.recv_msg_queue = recv_msg_queue
        self.send_msg_queue = send_msg_queue
        self._master = None
        # The port is derived from the config singleton (single source of truth).
        self._port = toolConfig.mavlink_port(drone_i)
        self._drone_i = int(drone_i)
        self.takeoff = False

    # --------------------------------------------------------------- connection
    def connect(self):
        """Connect to the drone's MAVLink stream and wait for a heartbeat."""
        self._master = mavutil.mavlink_connection(f'udp:0.0.0.0:{self._port}')
        try:
            self._master.wait_heartbeat(timeout=30)
        except TimeoutError:
            return False
        logger.info("Heartbeat from system (system {} component {}) from {}".format(
            self._master.target_system, self._master.target_component, self._port))
        return True

    def ready2fly(self) -> bool:
        """Wait until the IMU/GPS is ready (mode-specific status text)."""
        try:
            while True:
                message = self._master.recv_match(type=['STATUSTEXT'], blocking=True, timeout=30)
                message = message.to_dict()["text"]
                if toolConfig.MODE == "Ardupilot" and "IMU0 is using GPS" in message:
                    logger.debug("Ready to fly.")
                    return True
                if toolConfig.MODE == "PX4" and "home set" in message:
                    logger.debug("Ready to fly.")
                    return True
        except Exception as e:
            logger.debug(f"Error {e}")
            return False

    # --------------------------------------------------------------- mission
    def set_mission(self, mission_file, israndom: bool = False) -> bool:
        """Upload a waypoint mission. Returns True on a MISSION_ACK."""
        if not self._master:
            logger.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first !')

        loader = mavwp.MAVWPLoader()
        loader.target_system = self._master.target_system
        loader.target_component = self._master.target_component
        loader.load(mission_file)
        logger.debug(f"Load mission file {mission_file}")

        # PX4 must set home first.
        if toolConfig.MODE == "PX4":
            if not self.px4_set_home():
                logger.warning("PX4 set home failed!")
                return False

        if israndom:
            loader = self.random_mission(loader)
        self._master.waypoint_clear_all_send()
        self._master.waypoint_count_send(loader.count())
        seq_list = [True] * loader.count()
        try:
            while True in seq_list:
                msg = self._master.recv_match(type=['MISSION_REQUEST'], blocking=True)
                if msg is not None and seq_list[msg.seq] is True:
                    self._master.mav.send(loader.wp(msg.seq))
                    seq_list[msg.seq] = False
                    logger.debug(f'Sending waypoint {msg.seq}')
            self._master.recv_match(type=['MISSION_ACK'], blocking=True, timeout=5)
            logger.info('Upload mission finish.')
        except TimeoutError:
            logger.warning('Upload mission timeout!')
            return False
        return True

    def start_mission(self):
        """Arm and start the auto mission."""
        if not self._master:
            logger.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        if toolConfig.MODE == "PX4":
            self._master.set_mode_auto()
            self._master.arducopter_arm()
            self._master.set_mode_auto()
        else:
            self._master.arducopter_arm()
            self._master.set_mode_auto()
        logger.info('Arm and start.')

    # --------------------------------------------------------------- params
    def set_param(self, param: str, value: float) -> None:
        """Set a single parameter and wait for its PARAM_VALUE echo."""
        if not self._master:
            raise ValueError('Connect at first!')
        self._master.param_set_send(param, value)
        message = self._master.recv_match(type='PARAM_VALUE', blocking=True, timeout=3)
        if message is not None:
            message = message.to_dict()
            logger.debug('name: %s\t value: %f' % (message['param_id'], message['param_value']))

    def set_params(self, params_dict: dict) -> None:
        """Set multiple parameters ({param: value, ...})."""
        for param, value in params_dict.items():
            self.set_param(param, value)

    def get_param(self, param: str) -> float:
        """Get the current value of a parameter."""
        self._master.param_fetch_one(param)
        while True:
            message = self._master.recv_match(type=['PARAM_VALUE', 'PARM'], blocking=True).to_dict()
            if message['param_id'] == param:
                logger.debug('name: %s\t value: %f' % (message['param_id'], message['param_value']))
                break
        return message['param_value']

    def get_params(self, params: list) -> dict:
        """Get the current values of several parameters."""
        return {param: self.get_param(param) for param in params}

    # --------------------------------------------------------------- misc
    def get_msg(self, msg_type, block=False):
        """Receive a MAVLink message of ``msg_type``."""
        return self._master.recv_match(type=msg_type, blocking=block)

    def wait_waypoint(self, waypoint: int = 2) -> bool:
        """Block until the vehicle reaches the given mission waypoint seq."""
        if toolConfig.MODE == "PX4":
            waypoint -= 1
        while True:
            time.sleep(0.1)
            mission_current = self.get_msg(["MISSION_CURRENT"])
            if mission_current is not None and int(mission_current.seq) == waypoint:
                break
        return True

    def px4_set_home(self):
        """Tell PX4 to set its home position (mode-dependent coords)."""
        if toolConfig.HOME is None:
            lat, lon, alt = -35.362758, 149.165135, 583.730592
        else:
            lat, lon, alt = 40.072842, -105.230575, 0.000000
        self._master.mav.command_long_send(
            self._master.target_system, self._master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME, 1,
            0, 0, 0, 0, lat, lon, alt)
        msg = self._master.recv_match(type=['COMMAND_ACK'], blocking=True, timeout=5)
        if msg is not None:
            logger.debug(f"Home set callback: {msg.command}")
            return True
        return False

    def gcs_msg_request(self):
        """Send a GCS heartbeat (PX4 requires manual GCS heartbeats)."""
        self._master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def get_connection(self):
        return self._master

    # --------------------------------------------------------------- static
    @staticmethod
    def create_random_params(param_choice):
        """Build a random config dict over ``param_choice`` (step-quantized)."""
        para_dict = load_param()
        param_choice_dict = select_sub_dict(para_dict, param_choice)
        out_dict = {}
        for key, param_range in param_choice_dict.items():
            value = round(random.uniform(param_range['range'][0], param_range['range'][1])
                          / param_range['step']) * param_range['step']
            out_dict[key] = value
        return out_dict

    @staticmethod
    def random_mission(loader):
        """Shuffle a mission's interior waypoints (keep first two + last)."""
        index = random.sample(loader.wpoints[2:loader.count() - 1], loader.count() - 3)
        index = loader.wpoints[0:2] + index
        index.append(loader.wpoints[-1])
        for i, points in enumerate(index):
            points.seq = i
        loader.wpoints = index
        return loader


class MavlinkAPM(DroneMavlink, multiprocessing.Process):
    """ArduPilot MAVLink connection + background recv loop.

    The recv loop pushes an ``'error'`` marker onto ``send_msg_queue`` when a
    critical (severity 0/2) STATUSTEXT is observed, so the env can terminate
    the episode.
    """

    def __init__(self, drone_i=0, recv_msg_queue=None, send_msg_queue=None):
        DroneMavlink.__init__(self, drone_i, recv_msg_queue, send_msg_queue)
        multiprocessing.Process.__init__(self)

    def run(self):
        while True:
            msg = self._master.recv_match(type=['STATUSTEXT'], blocking=False)
            if msg is not None:
                msg = msg.to_dict()
                if msg['severity'] in [0, 2]:
                    logger.info('ArduCopter detect Crash.')
                    self.send_msg_queue._put('error')
                    break


class MavlinkPX4(DroneMavlink, multiprocessing.Process):
    """PX4 MAVLink connection. PX4 needs manual GCS heartbeats in its loop."""

    def __init__(self, drone_i=0, recv_msg_queue=None, send_msg_queue=None):
        DroneMavlink.__init__(self, drone_i, recv_msg_queue, send_msg_queue)
        multiprocessing.Process.__init__(self)

    def gcs_msg_request(self):
        self._master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
