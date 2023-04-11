import logging
import glob
import logging
import multiprocessing
import os
import random
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import ray
from pymavlink import mavutil, mavwp
from pymavlink.DFReader import DFMessage
from pymavlink.mavutil import mavserial
from pyulog import ULog
from tqdm import tqdm

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, select_sub_dict, read_path_specified_file, sort_result_detect_repair


class DroneMavlink:
    def __init__(self, port, recv_msg_queue=None, send_msg_queue=None):
        super(DroneMavlink, self).__init__()
        self.recv_msg_queue = recv_msg_queue
        self.send_msg_queue = send_msg_queue
        self._master: mavserial = None
        self._port = port
        self.takeoff = False

    # Mavlink common operation

    def connect(self):
        """
        Connect drone
        :return:
        """
        self._master = mavutil.mavlink_connection('udp:0.0.0.0:{}'.format(self._port))
        try:
            self._master.wait_heartbeat(timeout=30)
        except TimeoutError:
            return False
        logging.info("Heartbeat from system (system %u component %u) from %u" % (
            self._master.target_system, self._master.target_component, self._port))
        return True

    def ready2fly(self) -> bool:
        """
        wait for IMU can work
        :return:
        """
        try:
            while True:
                message = self._master.recv_match(type=['STATUSTEXT'], blocking=True, timeout=30)
                message = message.to_dict()["text"]
                # print(message)
                if toolConfig.MODE == "Ardupilot" and "IMU0 is using GPS" in message:
                    logging.debug("Ready to fly.")
                    return True
                # print(message)
                if toolConfig.MODE == "PX4" and "home set" in message:
                    logging.debug("Ready to fly.")
                    return True
        except Exception as e:
            logging.debug(f"Error {e}")
            return False

    def set_mission(self, mission_file, israndom: bool = False, timeout=30) -> bool:
        """
        Set mission
        :param israndom: random mission order
        :param mission_file: mission file
        :param timeout:
        :return: success
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first !')

        loader = mavwp.MAVWPLoader()
        loader.target_system = self._master.target_system
        loader.target_component = self._master.target_component
        loader.load(mission_file)
        logging.debug(f"Load mission file {mission_file}")

        # if px4, set home at first
        if toolConfig.MODE == "PX4":
            self.px4_set_home()

        if israndom:
            loader = self.random_mission(loader)
        # clear the waypoint
        self._master.waypoint_clear_all_send()
        # send the waypoint count
        self._master.waypoint_count_send(loader.count())
        seq_list = [True] * loader.count()
        try:
            # looping to send each waypoint information
            # Ardupilot method
            while True in seq_list:
                msg = self._master.recv_match(type=['MISSION_REQUEST'], blocking=True)
                if msg is not None and seq_list[msg.seq] is True:
                    self._master.mav.send(loader.wp(msg.seq))
                    seq_list[msg.seq] = False
                    logging.debug(f'Sending waypoint {msg.seq}')
            mission_ack_msg = self._master.recv_match(type=['MISSION_ACK'], blocking=True, timeout=timeout)
            logging.info(f'Upload mission finish.')
        except TimeoutError:
            logging.warning('Upload mission timeout!')
            return False
        return True

    def start_mission(self):
        """
        Arm and start the flight
        :return:
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        # self._master.set_mode_loiter()

        if toolConfig.MODE == "PX4":
            self._master.set_mode_auto()
            self._master.arducopter_arm()
            self._master.set_mode_auto()
        else:
            self._master.arducopter_arm()
            self._master.set_mode_auto()

        logging.info('Arm and start.')

    def set_param(self, param: str, value: float) -> None:
        """
        set a value of specific parameter
        :param param: name of the parameter
        :param value: float value want to set
        """
        if not self._master:
            raise ValueError('Connect at first!')
        self._master.param_set_send(param, value)
        # self.get_param(param)

    def set_params(self, params_dict: dict) -> None:
        """
        set multiple parameter
        :param params_dict: a dict consist of {parameter:values}...
        """
        for param, value in params_dict.items():
            self.set_param(param, value)

    def reset_params(self):
        self.set_param("FORMAT_VERSION", 0)

    def get_param(self, param: str) -> float:
        """
        get current value of a parameter.
        :param param: name
        :return: value of parameter
        """
        self._master.param_fetch_one(param)
        while True:
            message = self._master.recv_match(type=['PARAM_VALUE', 'PARM'], blocking=True).to_dict()
            if message['param_id'] == param:
                logging.debug('name: %s\t value: %f' % (message['param_id'], message['param_value']))
                break
        return message['param_value']

    def get_params(self, params: list) -> dict:
        """
        get current value of a parameters.
        :param params:
        :return: value of parameter
        """
        out_dict = {}
        for param in params:
            out_dict[param] = self.get_param(param)
        return out_dict

    def get_msg(self, msg_type, block=False):
        """
        receive the mavlink message
        :param msg_type:
        :param block:
        :return:
        """
        msg = self._master.recv_match(type=msg_type, blocking=block)
        return msg

    def set_mode(self, mode: str):
        """
        Set flight mode
        :param mode: string type of a mode, it will be convert to an int values.
        :return:
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        mode_id = self._master.mode_mapping()[mode]

        self._master.mav.set_mode_send(self._master.target_system,
                                       mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                       mode_id)
        while True:
            message = self._master.recv_match(type='COMMAND_ACK', blocking=True).to_dict()
            if message['command'] == mavutil.mavlink.MAVLINK_MSG_ID_SET_MODE:
                logging.debug(f'Mode: {mode} Set successful')
                break

    # Special operation
    def set_random_param_and_start(self):
        param_configuration = self.create_random_params(toolConfig.PARAM)
        self.set_params(param_configuration)
        # Unlock the uav
        self.start_mission()

    def px4_set_home(self):
        if toolConfig.HOME is None:
            self._master.mav.command_long_send(self._master.target_system, self._master.target_component,
                                               mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                                               1,
                                               0,
                                               0,
                                               0,
                                               0,
                                               -35.362758,
                                               149.165135,
                                               583.730592)
        else:
            self._master.mav.command_long_send(self._master.target_system, self._master.target_component,
                                               mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                                               1,
                                               0,
                                               0,
                                               0,
                                               0,
                                               40.072842,
                                               -105.230575,
                                               0.000000)
        msg = self._master.recv_match(type=['COMMAND_ACK'], blocking=True, timeout=30)
        logging.debug(f"Home set callback: {msg.command}")

    def gcs_msg_request(self):
        # If it requires manually send the gsc packets. (PX4)
        """
               PX4 needs manual send the heartbeat for GCS
               :return:
               """
        self._master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def wait_complete(self, remain_fail=False, timeout=60 * 5):
        """
        Wait the flight mission complete
        :param remain_fail:
        :param timeout:
        :return:
        """
        try:
            timeout_start = time.time()
            while time.time() < timeout_start + timeout:
                # PX4 will manually send the heartbeat for GCS
                self.gcs_msg_request()
                message = self._master.recv_match(type=['STATUSTEXT'], blocking=True, timeout=30)
                if message is None:
                    continue
                message = message.to_dict()
                line = message['text']
                if message["severity"] == 6:
                    if "Land" in line:
                        # if successful landed, break the loop and return true
                        logging.info(f"Successful break the loop.")
                        return True
                elif message["severity"] == 2 or message["severity"] == 0:
                    # Appear error, break loop and return false
                    if "PreArm" in line or remain_fail:
                        # "PreArm" failure will not generate log file, so it not need to delete log
                        # remain_fail means keep this log
                        logging.info(f"Get error with {message['text']}")
                        return True
                    return False
        except (TimeoutError, KeyboardInterrupt) as e:
            # Mission point time out, change other params
            logging.warning(f'Wp timeout! or Key bordInterrupt! exit: {e}')
            return False
        return False

    def wait_waypoint(self, waypoint: int = 2) -> bool:
        if toolConfig.MODE == "PX4":
            waypoint -= 1
        while True:
            time.sleep(0.1)
            mission_current = self.get_msg(["MISSION_CURRENT"])
            if mission_current is not None and int(mission_current.seq) == waypoint:
                break
        return True

    """
    Static method
    """

    @staticmethod
    def create_random_params(param_choice):
        para_dict = load_param()
        param_choice_dict = select_sub_dict(para_dict, param_choice)

        out_dict = {}
        for key, param_range in param_choice_dict.items():
            value = round(random.uniform(param_range['range'][0], param_range['range'][1]) / param_range['step']) * \
                    param_range['step']
            out_dict[key] = value
        return out_dict

    @staticmethod
    def random_mission(loader):
        """
        create random order of a mission
        :param loader: waypoint loader
        :return:
        """
        index = random.sample(loader.wpoints[2:loader.count() - 1], loader.count() - 3)
        index = loader.wpoints[0:2] + index
        index.append(loader.wpoints[-1])
        for i, points in enumerate(index):
            points.seq = i
        loader.wpoints = index
        return loader


class MavlinkAPM(DroneMavlink):
    """
    Mainly responsible for initiating the communication link to interact with UAV
    """

    def __init__(self, port, recv_msg_queue, send_msg_queue):
        super(MavlinkAPM, self).__init__(port, recv_msg_queue, send_msg_queue)


    """
    Thread
    """

    def run(self):
        """
        loop check
        :return:
        """

        while True:
            msg = self._master.recv_match(type=['STATUSTEXT'], blocking=False)
            if msg is not None:
                msg = msg.to_dict()
                # print(msg2)
                if msg['severity'] in [0, 2]:
                    # self.send_msg_queue.put('crash')
                    logging.info('ArduCopter detect Crash.')
                    self.send_msg_queue._put('error')
                    break


class MavlinkPX4(DroneMavlink):
    """
    Mainly responsible for initiating the communication link to interact with UAV
    """

    def __init__(self, port, recv_msg_queue, send_msg_queue):
        super(MavlinkPX4, self).__init__(port, recv_msg_queue, send_msg_queue)

    """
    Method
    """

    def gcs_msg_request(self):
        """
        PX4 needs manual send the heartbeat for GCS
        :return:
        """
        self._master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
