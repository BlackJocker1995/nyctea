"""
SimManager Version: 4.4 23-03-02
"""
import logging
import math
import multiprocessing
import os
import time
from typing import Type

import pexpect
from pexpect import spawn
from pymavlink import mavwp

from Cptool.boardMavlink import BoardMavlink, BoardMavlinkAPM, BoardMavlinkPX4
from Cptool.config import toolConfig
from Cptool.mavlink import DroneMavlink
from Cptool.mavtool import Location
from Cptool.monitor import MonitorFlight
from Cptool.simSimulator import SimSimulator


class SimManager:
    def __init__(self, debug: bool = False):
        self._sim_task = None
        self._sitl_task = None
        self.sim_simulator: SimSimulator = None
        self.online_mavlink: DroneMavlink = None
        self.board_mavlink: BoardMavlink = None
        self.mav_monitor: MonitorFlight = None
        self._even = None
        self.sim_msg_queue = multiprocessing.Queue()
        self.mav_msg_queue = multiprocessing.Queue()

        # clear previous logging handler
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        if debug:
            logging.basicConfig(format='%(asctime)s-PID(%(process)d) %(filename)s[%(lineno)d]: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s-PID(%(process)d) %(filename)s[%(lineno)d]: %(message)s',
                                level=logging.INFO)

    """
    Base Function
    """

    def start_sim(self):
        """
        start simulator
        :return:
        """
        # Airsim
        cmd = None
        if toolConfig.SIM == 'Airsim':
            cmd = f'gnome-terminal -- {toolConfig.AIRSIM_PATH} ' \
                  f'-ResX={toolConfig.HEIGHT} -ResY={toolConfig.WEIGHT} -windowed'
        if toolConfig.SIM == 'Jmavsim':
            cmd = f'gnome-terminal -- bash {toolConfig.JMAVSIM_PATH}'
        if toolConfig.SIM == 'Morse':
            cmd = f'gnome-terminal -- morse run {toolConfig.MORSE_PATH}'
        if toolConfig.SIM == 'Gazebo':
            cmd = f'gnome-terminal -- gazebo --verbose worlds/iris_arducopter_runway.world'
        if cmd is None:
            raise ValueError('Not support mode')
        logging.info(f'Start Simulator {toolConfig.SIM}')
        self._sim_task = pexpect.spawn(cmd)

    def start_multiple_sim(self, drone_i=0):
        """
        start multiple simulator (only jmavsim now)
        :return:
        """
        # Airsim
        cmd = None
        if toolConfig.SIM == 'Jmavsim':
            port = 4560 + int(drone_i)
            cmd = f'{toolConfig.JMAVSIM_PATH} -p {port} -l'
        self._sim_task = pexpect.spawn(cmd, cwd=toolConfig.PX4_RUN_PATH, timeout=30, encoding='utf-8')
        logging.debug("Init px4 Jmavsim description.")

    def start_sitl(self):
        """
        Start SITL PX4 or Ardupilot
        :return:
        """

        global cmd
        if toolConfig.MODE == "Ardupilot":
            if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin"):
                os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin")
            if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm"):
                os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm")

            if toolConfig.SIM == 'Airsim':
                if toolConfig.HOME is not None:
                    cmd = f"python3 {toolConfig.SITL_PATH} -v ArduCopter " \
                          f"--location={toolConfig.HOME}" \
                          f" -f airsim-copter --out=127.0.0.1:14550 --out=127.0.0.1:14540 " \
                          f" -S {toolConfig.SPEED}"
                else:
                    cmd = f"python3 {toolConfig.SITL_PATH} -v ArduCopter -f airsim-copter " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -S {toolConfig.SPEED}"
            if toolConfig.SIM == 'Morse':
                cmd = f"python3 {toolConfig.SITL_PATH}  -v ArduCopter --model morse-quad " \
                      f"--add-param-file=/home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.parm  " \
                      f"--out=127.0.0.1:14550 -S {toolConfig.SPEED}"
            if toolConfig.SIM == 'Gazebo':
                cmd = f'python3 {toolConfig.SITL_PATH} -f gazebo-iris -v ArduCopter ' \
                      f'--out=127.0.0.1:14550 -S {toolConfig.SPEED}'
            if toolConfig.SIM == 'SITL':
                if toolConfig.HOME is not None:
                    cmd = f"python3 {toolConfig.SITL_PATH} --location={toolConfig.HOME} " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540  -v ArduCopter -w -S {toolConfig.SPEED} "
                else:
                    cmd = f"python3 {toolConfig.SITL_PATH}  " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -v ArduCopter -w -S {toolConfig.SPEED} "

            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.ARDUPILOT_LOG_PATH, timeout=30, encoding='utf-8')

        if toolConfig.MODE == "PX4":
            if os.path.exists(f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/tmp/rootfs/eeprom/parameters_10016"):
                os.remove(f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/tmp/rootfs/eeprom/parameters_10016")

                if toolConfig.HOME is None:
                    pre_argv = f"PX4_HOME_LAT=-35.362758 " \
                               f"PX4_HOME_LON=149.165135 " \
                               f"PX4_HOME_ALT=583.730592 " \
                               f"PX4_SIM_SPEED_FACTOR={toolConfig.SPEED}"
                else:
                    pre_argv = f"PX4_HOME_LAT=40.072842 " \
                               f"PX4_HOME_LON=-105.230575 " \
                               f"PX4_HOME_ALT=0.000000 " \
                               f"PX4_SIM_SPEED_FACTOR={toolConfig.SPEED}"

                if toolConfig.SIM == 'Airsim':
                    cmd = f'make {pre_argv} px4_sitl_default none_iris'
                if toolConfig.SIM == 'Jmavsim':
                    cmd = f"make {pre_argv} px4_sitl_default jmavsim"
            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.PX4_RUN_PATH, timeout=30, encoding='utf-8')
            logging.info(f"Start {toolConfig.MODE} --> [{toolConfig.SIM}]")

    def start_multiple_sitl(self, drone_i=0):
        """
        start multiple simulators (not support PX4 now)
        :param drone_i:
        :return:
        """
        if toolConfig.MODE == 'Ardupilot':
            if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{drone_i}/eeprom.bin"):
                os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{drone_i}/eeprom.bin")
            if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{drone_i}/mav.parm"):
                os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{drone_i}/mav.parm")

            if toolConfig.HOME is not None:
                cmd = f"python3 {toolConfig.SITL_PATH} --location={toolConfig.HOME} " \
                      f"--out=127.0.0.1:1455{drone_i} --out=127.0.0.1:1454{drone_i} --out=127.0.0.1:1456{drone_i} " \
                      f"-v ArduCopter -w -S {toolConfig.SPEED} --instance {drone_i}"
            else:
                cmd = f"python3 {toolConfig.SITL_PATH} " \
                      f"--out=127.0.0.1:1455{drone_i} --out=127.0.0.1:1454{drone_i} --out=127.0.0.1:1456{drone_i} " \
                      f"-v ArduCopter -w -S {toolConfig.SPEED} --instance {drone_i}"

            self._sitl_task = (pexpect.spawn(cmd, cwd=f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{drone_i}",
                                             timeout=30, encoding='utf-8'))

        if toolConfig.MODE == 'PX4':

            if os.path.exists(
                    f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/instance_{drone_i}/eeprom/parameters_10016") \
                    and toolConfig.MODE == "PX4":
                os.remove(
                    f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/instance_{drone_i}/eeprom/parameters_10016")

            if toolConfig.SIM == 'Jmavsim':
                cmd = f"{toolConfig.PX4_RUN_PATH}/Tools/sitl_multiple_run_single.sh {drone_i}"

            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.PX4_RUN_PATH, timeout=30, encoding='utf-8')

        logging.info(f"Start {toolConfig.MODE} --> [{toolConfig.SIM} - {drone_i}]")

    def online_mavlink_init(self, mavlink_class: Type[DroneMavlink] = DroneMavlink, drone_i=0):
        """
        Runtime mavlink monitor
        :return:
        """
        self.online_mavlink = mavlink_class(14540 + int(drone_i),
                                            recv_msg_queue=self.sim_msg_queue,
                                            send_msg_queue=self.mav_msg_queue)
        self.online_mavlink.connect()
        if toolConfig.MODE == 'Ardupilot':
            if self.online_mavlink.ready2fly():
                return True
        elif toolConfig.MODE == 'PX4':
            while True:
                line = self._sitl_task.readline()
                # print(line)
                if 'notify' in line:
                    # Disable the fail warning and return
                    self._sitl_task.send("param set NAV_RCL_ACT 0 \n")
                    time.sleep(0.1)
                    self._sitl_task.send("param set NAV_DLL_ACT 0 \n")
                    time.sleep(0.1)
                    # Enable detector
                    self._sitl_task.send("param set CBRK_FLIGHTTERM 0 \n")
                    return True

    def sim_simulator_init(self, simulator_class):
        """
        Init  monitor
        :return:
        """
        self.sim_simulator = simulator_class(recv_msg_queue=self.mav_msg_queue, send_msg_queue=self.sim_msg_queue)
        time.sleep(3)

    def board_mavlink_init(self):
        """
        onboard message manager
        """
        if toolConfig.MODE == "PX4":
            self.board_mavlink = BoardMavlinkPX4()
        else:
            self.board_mavlink = BoardMavlinkAPM()

    def mav_monitor_init(self, port):
        """
        Error monitor
        """
        self.mav_monitor = MonitorFlight(port)

    """
    Mavlink Operation
    """

    def mav_monitor_connect(self):
        """
        Mavlnik 连接
        :return:
        """
        return self.online_mavlink.connect()

    def mav_monitor_start_mission(self):
        """
        开始任务
        :return:
        """
        self.online_mavlink.start_mission()

    def mav_monitor_set_mission(self, mission_file, random: bool = False):
        """
        Set mission
        :param mission_file: file path
        :param random:
        :return:
        """
        return self.online_mavlink.set_mission(mission_file, random)

    def mav_monitor_set_random_param(self):
        """
        initial airsim monitor
        :return:
        """
        params_dict = self.online_mavlink.load_param()
        params_value = self.online_mavlink.random_param_value(params_dict)
        self.online_mavlink.set_params(params_value)

    """
    Simulator Operation
    """

    def kill_mavproxy(self):
        os.kill(int(pid), signal.SIGKILL)

    def stop_sitl(self):
        self._sitl_task.sendcontrol('c')
        while True:
            line = self._sitl_task.readline()
            if not line:
                break
        self._sitl_task.close(force=True)
        logging.info('Stop SITL task.')

    def stop_sim(self):
        self._sim_task.sendcontrol('c')
        self._sim_task.close(force=True)
        logging.info('Stop Sim task.')

    """
    Other get/set
    """

    def sitl_task(self) -> spawn:
        return self._sitl_task

    def airsim_task(self) -> spawn:
        return self._sim_task


class FixSimManager(SimManager, multiprocessing.Process):

    def __init__(self, debug: bool = False):
        super(FixSimManager, self).__init__(debug)
        super(multiprocessing.Process, self).__init__()