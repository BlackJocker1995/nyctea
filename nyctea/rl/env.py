"""The DDPG repair environment.

Rewritten from the legacy ``Rl/env.py``. The reward computation and the
deviation geometry are delegated to pure functions in :mod:`nyctea.rl.reward`
and :mod:`nyctea.rl.actions` so they can be unit-tested; the MAVLink I/O stays
in the env. The ``os.access`` busy-wait in ``get_random_incorrect_configuration``
is replaced by :class:`~nyctea.concurrency.LockedCsv`, and the mission/port
lookups go through the config singleton.
"""
import os
import random
import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.utils import shuffle

from nyctea.comms import MavlinkAPM, MavlinkPX4
from nyctea.config import toolConfig
from nyctea.concurrency import LockedCsv
from nyctea.rl.reward import compute_reward, deviation_from_state
from nyctea.sim import FixSimManager


class DroneEnv:
    """Drone flight environment for the DDPG repair agent."""

    def __init__(self, device=0, tail_n=20, debug: bool = False):
        self.manager: FixSimManager = None
        self.device = device
        # number of state rows to keep as the observation window.
        self.tail_n = tail_n
        # deviation threshold below which no repair action is taken.
        if toolConfig.MODE == "PX4":
            self.deviation_threshold = 2.5
        else:
            self.deviation_threshold = 6.1
        self.parameter_shape = len(toolConfig.PARAM)

        self.cur_deviation = None
        self.cur_state = None
        self.current_incorrect_configuration = None
        self.debug = debug

    # --------------------------------------------------------------- config load
    def get_random_incorrect_configuration(self, param_file, deduplicate=False):
        """Pick a destabilizing config from ICSearcher's params.csv.

        The CSV is produced by ICSearcher's validate stage; rows with
        ``result != "pass"`` are the destabilizing configs the agent learns to
        repair. ``deduplicate`` skips configs already used (tracked in a
        LockedCsv so concurrent workers don't both pick the same one).
        """
        configurations = pd.read_csv(param_file)
        incorrect_configuration = configurations[configurations["result"] != "pass"]
        incorrect_configuration = shuffle(incorrect_configuration)

        if deduplicate:
            used_csv = LockedCsv(
                f'validation/{toolConfig.MODE}/params_trained{toolConfig.EXE}.csv',
                header=(toolConfig.PARAM_PART + ['used']),
            )
            used_csv.ensure_created()
            for index, row in incorrect_configuration.iterrows():
                config = row.drop(["score", "result"]).astype(float)
                # Skip configs already used (matched within float tolerance).
                used = used_csv.rows_as_dataframe()
                if not used.empty:
                    used_params = used.drop(['used'], axis=1, inplace=False)
                    if ((used_params - config).sum(axis=1).abs() < 0.00001).sum() > 0:
                        continue
                # Claim this config by recording it.
                used_csv.append_row(list(config.values) + [1])
                break
        else:
            config = incorrect_configuration.iloc[0].drop(["score", "result"]).astype(float)
        self.current_incorrect_configuration = config.to_dict()

    # --------------------------------------------------------------- state
    @staticmethod
    def get_deviation(pd_state):
        """(achieved-state, deviation) for the current status segment."""
        return deviation_from_state(pd_state, toolConfig.MODE)

    # --------------------------------------------------------------- lifecycle
    def init_drone_env(self, delete_log=False):
        """Boot one SITL instance + wire MAVLink/board/monitor for this device."""
        if self.manager is not None:
            self.close_env()
            if delete_log:
                self.manager.board_mavlink.delete_current_log(self.device)
            del self.manager

        self.manager = FixSimManager(self.debug)
        self.manager.start_multiple_sitl(self.device)
        if toolConfig.MODE == "PX4":
            self.manager.start_multiple_sim(self.device)
            self.manager.online_mavlink_init(MavlinkPX4, self.device)
            self.manager.mav_monitor_init(toolConfig.mavlink_port(self.device) + 20)
        else:
            self.manager.online_mavlink_init(MavlinkAPM, self.device)
            self.manager.mav_monitor_init(toolConfig.mavlink_port(self.device) + 20)
        self.manager.board_mavlink_init()
        logger.debug("Start new simulation environment.")

        set_result = self.manager.online_mavlink.set_mission(toolConfig.mission_file(), False)
        if not set_result:
            logger.warning("Mission file set failed!")
            return False

        if toolConfig.MODE == "PX4":
            time.sleep(2)

    def reset(self, delay=True, delete_log=False) -> bool:
        """Reset the environment: boot SITL, upload the bad config, take off."""
        self.init_drone_env(delete_log)
        self.manager.online_mavlink.start_mission()
        self.manager.board_mavlink.init_binary_log_file(self.device)
        self.manager.board_mavlink.wait_bin_ready()
        self.manager.mav_monitor.start()
        self.manager.online_mavlink.wait_waypoint()
        if delay:
            x = random.randint(0, 2)
            logger.info(f"Game play with {x} seconds later.")
            time.sleep(x)
        self.manager.online_mavlink.set_params(self.current_incorrect_configuration)
        logger.debug("Set parameters successfully.")
        self.manager.board_mavlink.start()
        return True

    def catch_state(self):
        """Capture the latest status segment as the RL observation."""
        df_array = self.manager.board_mavlink.bin_read_last_seg()
        if df_array is None:
            return None
        self.cur_state, self.cur_deviation = self.get_deviation(df_array)
        return self.cur_state.to_numpy().reshape(-1)

    def step(self, configuration):
        """Apply ``configuration``, observe the resulting state, return (s', r, done)."""
        configuration = pd.DataFrame(configuration.reshape((-1, self.parameter_shape)),
                                     columns=toolConfig.PARAM).iloc[0].to_dict()
        self.manager.online_mavlink.set_params(configuration)

        if toolConfig.MODE == "Ardupilot":
            time.sleep(self.tail_n / 10)

        finish = False
        play_state = self.manager.board_mavlink.bin_read_last_seg()
        next_state, played_deviation = self.get_deviation(play_state)
        acc_ratio = abs(np.average(next_state["AccX"].to_numpy()))

        reward = compute_reward(self.cur_deviation, played_deviation, acc_ratio)

        # Monitor events terminate the episode and adjust the reward.
        if not self.manager.mav_monitor.msg_queue.empty():
            manager_msg, _ = self.manager.mav_monitor.msg_queue.get()
            if manager_msg == "pass":
                logger.info("Land, finish this mission.")
                reward = 10
                finish = True
            else:
                logger.info("Unstable events, finish this mission.")
                reward = reward * 2
                finish = True

        logger.info(f"Change Deviation: {round(self.cur_deviation, 4)} -> "
                    f"{round(played_deviation, 4)}: reward: {round(reward, 4)}, done: {finish}")

        self.cur_state = next_state
        self.cur_deviation = played_deviation
        return next_state.to_numpy().reshape(-1), reward, finish

    # --------------------------------------------------------------- teardown
    def close_env(self):
        try:
            self._try_kill_mavproxy()
            self.manager.stop_sitl()
            if toolConfig.MODE == "PX4":
                self.manager.stop_sim()
            if self.manager.mav_monitor is not None and self.manager.mav_monitor.is_alive():
                self.manager.mav_monitor.terminate()
            if self.manager.board_mavlink is not None and self.manager.board_mavlink.is_alive():
                self.manager.board_mavlink.terminate()
        except Exception:
            pass
        logger.debug("Stop previous simulation successfully.")

    def _try_kill_mavproxy(self):
        """Kill any lingering mavproxy bound to this device's port.

        Uses psutil to find the process by port (more targeted than the legacy
        full-process cmdline scan, but kept functionally equivalent).
        """
        import signal
        import psutil
        port = str(toolConfig.mavlink_port(self.device))
        for proc in psutil.process_iter():
            try:
                cmdline = proc.cmdline()
            except psutil.NoSuchProcess:
                continue
            if (len(cmdline) >= 2 and os.path.basename(cmdline[1]) == "mavproxy.py"
                    and port in cmdline[3:]):
                os.kill(proc.pid, signal.SIGKILL)
