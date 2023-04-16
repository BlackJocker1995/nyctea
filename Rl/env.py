import csv
import logging
import os
import random
import time
from copy import deepcopy
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
from Cptool.config import toolConfig
from Cptool.mavlink import MavlinkAPM, MavlinkPX4
from Cptool.simManager import FixSimManager


class DroneEnv:
    """
    Drone Flight environment
    """

    def __init__(self, device=0, tail_n=20, debug: bool = False):
        """
        Game environment
        """
        # which simulator device
        self.manager: FixSimManager = None
        self.device = device
        # number of the least state get
        self.tail_n = tail_n
        # deviation threshold
        if toolConfig.MODE == "PX4":
            self.deviation_threshold = 2.3
        else:
            # 12.25
            self.deviation_threshold = 6.1
        # parameter_shape
        self.parameter_shape = len(toolConfig.PARAM)

        #
        self.cur_deviation = None
        self.cur_state = None
        self.current_incorrect_configuration = None
        # display debug information
        self.debug = debug

    def get_random_incorrent_configuration(self, deduplicate=False):
        """
        set a random incorrent configuration from ICSearcher
        @param deduplicate:
        @return:
        """
        # Read incorrect configuration
        configurations = pd.read_csv(f"validation/{toolConfig.MODE}/params{toolConfig.EXE}.csv")
        incorrect_configuration = configurations[configurations["result"] != "pass"]

        # Stochastic order
        incorrect_configuration: pd.DataFrame = shuffle(incorrect_configuration)
        global config
        if deduplicate:
            for index, row in incorrect_configuration.iterrows():
                config = row.drop(["score", "result"]).astype(float)
                # check if used
                if os.path.exists(f'validation/{toolConfig.MODE}/params_trained{toolConfig.EXE}.csv'):
                    # check file access (whether the file is being written)
                    while not os.access(f"validation/{toolConfig.MODE}/params_trained{toolConfig.EXE}.csv", os.R_OK):
                        continue
                    # read
                    read_data = pd.read_csv(f'validation/{toolConfig.MODE}/params_trained{toolConfig.EXE}.csv')
                    exit_data = read_data.drop(['used'], axis=1, inplace=False)
                    # If the value has been used, change another
                    if ((exit_data - config).sum(axis=1).abs() < 0.00001).sum() > 0:
                        continue
                else:
                    data = pd.DataFrame(columns=(toolConfig.PARAM_PART + ['used']))
                    data.to_csv(f'validation/{toolConfig.MODE}/params_trained{toolConfig.EXE}.csv', index=False)
                break
        else:
            config = incorrect_configuration.iloc[0].drop(["score", "result"]).astype(float)
        self.current_incorrect_configuration = config.to_dict()

    @staticmethod
    def get_deviation(pd_state):
        """

        @param pd_state:
        @return: achieved state, deviation
        """
        if toolConfig.MODE == "PX4":
            bias = pd_state[['BiasA', 'BiasB', 'BiasC']]
            out_state = pd_state.drop(['TimeS', 'BiasA', 'BiasB', 'BiasC'], axis=1)
            deviation = np.array(bias)
        else:
            desired_state = pd_state[['DesRoll', 'DesPitch', 'DesYaw', 'DesRateRoll', 'DesRatePitch', 'DesRateYaw']]
            achieved_state = pd_state[['Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw']]
            out_state = pd_state.drop(['TimeS', 'DesRoll', 'DesPitch', 'DesYaw',
                                       'DesRateRoll', 'DesRatePitch', 'DesRateYaw'], axis=1)
            deviation = np.radians(desired_state.values - achieved_state.values)
        # observed deviation
        deviation = abs(deviation.sum())

        return out_state, deviation

    def reset(self, delay=True):
        """
        Reset Environment
        """
        # if alive, close
        if self.manager is not None:
            print(self.manager)
            self.manager.mav_monitor.terminate()
            self.manager.board_mavlink.terminate()
            self.manager.stop_sitl()
            if toolConfig.MODE == "PX4":
                self.manager.stop_sim()
            self.manager.board_mavlink.delete_current_log(self.device)
        logging.debug("Stop previous simulation successfully.")

        # Manager
        self.manager = FixSimManager(self.debug)

        # init environment
        self.manager.start_multiple_sitl(self.device)
        if toolConfig.MODE == "PX4":
            self.manager.start_multiple_sim(self.device)
            self.manager.online_mavlink_init(MavlinkPX4, self.device)
            mission_file = 'Cptool/fitCollection_px4.txt'
            self.manager.mav_monitor_init(int(14550) + int(self.device))
        else:
            self.manager.online_mavlink_init(MavlinkAPM, self.device)
            mission_file = 'Cptool/fitCollection.txt'
            self.manager.mav_monitor_init(int(14560) + int(self.device))
        self.manager.board_mavlink_init()
        logging.debug("Start new simulation environment.")

        # Set mission_ file
        set_result = self.manager.online_mavlink.set_mission(mission_file, False)
        if not set_result:
            logging.warning("Mission file set failed!")
            return False

        # PX4 requires waiting 2 seconds.
        if toolConfig.MODE == "PX4":
            time.sleep(2)
        # take off
        self.manager.online_mavlink.start_mission()
        # load bin log file
        self.manager.board_mavlink.init_binary_log_file(self.device)
        # Bin manager
        self.manager.board_mavlink.wait_bin_ready()

        # monitor error
        self.manager.mav_monitor.start()
        # wait flight
        self.manager.online_mavlink.wait_waypoint()
        # wait to play game
        if delay:
            x = random.randint(0, 2)
            logging.info(f"Game play with {x} seconds later.")
            time.sleep(x)
        # set configuration
        self.manager.online_mavlink.set_params(self.current_incorrect_configuration)
        # read check parameters
        # self.manager.online_mavlink.get_params(toolConfig.PARAM_PART)

        logging.debug("Set parameters successfully.")

        self.manager.board_mavlink.start()

        return True

    def catch_state(self):

        # catch a state segment
        df_array = self.manager.board_mavlink.bin_read_last_seg()
        if df_array is None:
            return None
        # current state
        self.cur_state, self.cur_deviation = self.get_deviation(df_array)

        return self.cur_state.to_numpy().reshape(-1)

    def step(self, configuration):
        """
        Perform an action on the current environment
        @param configuration: configuration
        @return:
        """
        # 上传数据，然后观察下个segment 有没有减小，减的越多越好
        # Upload this configuration
        configuration = pd.DataFrame(configuration.reshape((-1, self.parameter_shape)),
                                     columns=toolConfig.PARAM).iloc[0].to_dict()
        # set and read check
        self.manager.online_mavlink.set_params(configuration)
        # self.manager.online_mavlink.get_params(toolConfig.PARAM_PART)

        # Wait a minimum
        if toolConfig.MODE == "Ardupilot":
            time.sleep(self.tail_n / 10)
        # reward rules:
        reward = 0
        # finish flag
        finish = False
        # 1. deviation between played_deviation and current_deviation.
        # Observe state
        play_state = self.manager.board_mavlink.bin_read_last_seg()
        # state the action resulted, and its deviation compared to the previous state.
        next_state, played_deviation = self.get_deviation(play_state)
        acc_ratio = abs(np.average(next_state["AccX"].to_numpy()))

        # Positive means this configuration help droned; negative intensify the unstable.
        reward = self.cur_deviation - played_deviation
        if reward > 0:
            # greater reward if the configuration reduce the deviation from larger to smaller that "1"
            reward = reward / max(1, played_deviation)
            # acc bigger is better
            reward = reward * acc_ratio
        else:
            # negative: bad change
            reward = max(-100, reward)

        # 2. if received mavlink message about events (e.g., thrust loss),then finish the game and punish the agent -10
        #  if received mavlink land message, give reward to agent 10
        if not self.manager.mav_monitor.msg_queue.empty():
            # receive error result
            manager_msg, manager_msg_timestamp = self.manager.mav_monitor.msg_queue.get()
            if manager_msg == "pass":
                logging.info("Land, finish this mission.")
                # neutral result, current game is ended.
                reward = 10
                finish = True
            else:
                # serious error result, current game is ended.
                logging.info("Unstable events, finish this mission.")
                reward = reward * 2
                finish = True

        logging.info(f"Change Deviation: {round(self.cur_deviation, 4)} -> {round(played_deviation, 4)}: reward: "
                     f"{round(reward, 4)}, done: {finish}")

        self.cur_state = next_state
        self.cur_deviation = played_deviation
        return next_state.to_numpy().reshape(-1), reward, finish
