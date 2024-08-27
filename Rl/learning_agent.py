import copy
import fcntl
import json
import logging
import os
import pickle
import random
import sys
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

from Rl.actor_critic import Actor, Critic

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, select_sub_dict, read_unit_from_dict, get_default_values, read_range_from_dict, \
    send_notice
from multiprocessing import shared_memory
from Rl.env import DroneEnv


def create_random_params(param_choice):
    para_dict = load_param()

    param_choice_dict = select_sub_dict(para_dict, param_choice)

    out_dict = {}
    for key, param_range in param_choice_dict.items():
        value = round(random.uniform(param_range['range'][0], param_range['range'][1]) / param_range['step']) * \
                param_range['step']
        out_dict[key] = value
    return out_dict


class ReLearningAgent():
    def __init__(self):
        self.participle_param = toolConfig.PARAM_PART
        para_dict = load_param()

        # default value, step and boundary
        # self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        self.step_unit = read_unit_from_dict(para_dict)
        self.default_pop = (get_default_values(para_dict)).to_numpy(dtype=int)
        self.sub_value_range = read_range_from_dict(para_dict)
        # self.value_range = read_range_from_dict(para_dict)
        # boundary limitation

    @staticmethod
    def create_random_params() -> np.ndarray:
        """
        generate a configuration with random parameter values.
        @return:
        """
        out_values = np.random.random(len(toolConfig.PARAM))
        return out_values

    def action2config(self, action):
        """
        output action is [0, 1], transform them to original range of parameters.
        @param action: configuration action
        @return:
        """
        step_increase = (action * (self.sub_value_range[:, 1] - self.sub_value_range[:, 0])
                         // self.step_unit) * self.step_unit
        return np.array(self.sub_value_range[:, 0] + step_increase, dtype=float)

    @abstractmethod
    def save_point(self):
        pass

    @abstractmethod
    def check_point(self):
        pass


class DDPGAgent(ReLearningAgent):
    def __init__(self, train=True, device=0):
        super().__init__()
        self.device = int(device)
        # play environment
        self.env = DroneEnv(device=device, debug=toolConfig.DEBUG)
        # gamma
        self.gamma = 0.99
        # actor learning rate
        self.actor_lr = 0.01
        # critic learning rate
        self.critic_lr = 0.001
        # tau
        self.tau = 0.02
        # buffer capacity
        self.capacity = 10000
        # learning batch size
        self.batch_size = 64

        # Varies by environment
        # state shape (tail_n, 12) -> (1, tail_n * 12)
        state_shape = self.env.tail_n * 12
        # parameter element in configuration
        action_dim = len(toolConfig.PARAM)

        # Buffer length
        self.buffer_length = 0
        self.buffer = None

        # actor network
        self.actor = Actor(state_shape, 256, action_dim)
        self.actor_target = Actor(state_shape, 256, action_dim)
        # critic network
        self.critic = Critic(state_shape + action_dim, 256, action_dim)
        self.critic_target = Critic(state_shape + action_dim, 256, action_dim)
        # network map
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        if self.device == 0:
            self.check_point()
            try:
                _shm = shared_memory.ShareableList(name="rl_buffer")
                _shm.shm.close()
                _shm.shm.unlink()
            except Exception as e:
                pass
            if train:
                self.buffer = self.load_buffer()
                if self.buffer is None:
                    # destroy the shared memory
                    a = (np.ones(12 * 20, dtype=float), np.ones(len(toolConfig.PARAM), dtype=float),
                         float(-101), np.ones(12 * 20, dtype=float))
                    bit_a = pickle.dumps(a)
                    _tmp = [0]
                    for _ in range(self.capacity):
                        _tmp.append(bit_a)
                    self.buffer = shared_memory.ShareableList(_tmp, name="rl_buffer")
            # self.restore_buffer()
            # None and create
        else:
            if train:
                self.buffer = shared_memory.ShareableList(name="rl_buffer")

    def select_action(self, state):
        """
        select an action for "state"
        @param state: current state
        @return: action
        """
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        return action

    def _put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def put_once_buffer(self, *transition):
        # others thread wait first 0 thread loading data at first.
        if self.device != 0:
            while self.buffer[0] == 0:
                continue
        # load current index
        _index = self.buffer[0]
        # over capacity
        if _index > self.capacity - 3:
            self.buffer[0] = 0
        # to bytes
        pik_data = pickle.dumps(transition)
        # notify
        logging.debug(f"Item add to buffer: {_index} - len:{len(pik_data)}")
        # add new data
        self.buffer[self.buffer[0] + 1] = pik_data
        self.buffer[0] += 1

    def restore_buffer(self):
        if toolConfig.MODE == "PX4":
            pathfile = f"{toolConfig.BUFFER_PATH}/PX4/buffer.pkl"
        else:
            pathfile = f"{toolConfig.BUFFER_PATH}/Ardupilot/buffer.pkl"
        if not os.path.exists(pathfile):
            return None
        # wait for read
        while not os.access(pathfile, os.R_OK):
            continue
        with open(pathfile, "rb") as fp:
            # fcntl.flock(fp, fcntl.LOCK_EX)
            _buffer = pickle.load(fp)
            # fcntl.flock(fp, fcntl.LOCK_UN)

        for i in range(_buffer[0]):
            self.buffer[i] = _buffer[i]

    def load_buffer(self):
        if toolConfig.MODE == "PX4":
            pathfile = f"{toolConfig.BUFFER_PATH}/PX4/buffer.pkl"
        else:
            pathfile = f"{toolConfig.BUFFER_PATH}/Ardupilot/buffer.pkl"
        if not os.path.exists(pathfile):
            return None

        while not os.access(pathfile, os.R_OK):
            continue
        with open(pathfile, "rb") as fp:
            # fcntl.flock(fp, fcntl.LOCK_EX)
            _buffer = pickle.load(fp)
            # fcntl.flock(fp, fcntl.LOCK_UN)
        return shared_memory.ShareableList(_buffer, name="rl_buffer")

    def save_buffer(self):
        if toolConfig.MODE == "PX4":
            pathfile = f"{toolConfig.BUFFER_PATH}/PX4/buffer.pkl"
        else:
            pathfile = f"{toolConfig.BUFFER_PATH}/Ardupilot/buffer.pkl"
        # return
        if self.buffer[0] == 0:
            return
        # save all content
        _tmp_file = []
        for i in range(self.capacity):
            _tmp_file.append(self.buffer[i])
        while os.path.exists(pathfile) and not os.access(pathfile, os.R_OK):
            continue
        with open(pathfile, "wb") as fp:
            # fcntl.flock(fp, fcntl.LOCK_EX)
            pickle.dump(_tmp_file, fp)
            # fcntl.flock(fp, fcntl.LOCK_UN)

    def learn(self):
        """
        learning a buffer
        @return:
        """
        # Load buffer
        # self.buffer = self.load_buffer()
        # return if buffer is too small
        if self.buffer[0] <= self.batch_size:
            logging.debug(f"Current buffer size is {self.buffer[0]}, skip.")
            return False
        if self.buffer[0] % self.batch_size > 10:
            return False

        logging.info(f"Take example and learn, example/buffer size: {self.batch_size}/{self.buffer[0]}.")
        # restore
        _tmp_buffer = []
        for i in range(self.buffer[0]-1):
            content = pickle.loads(self.buffer[i + 1])
            if content[2] < -100:
                continue
            _tmp_buffer.append(content)
        # take a sample
        samples = random.sample(_tmp_buffer, self.batch_size)
        # tuple input
        state_current, action, reward, state_result = zip(*samples)

        # transform to tensor data
        state_current = torch.tensor(np.array(state_current, dtype=np.float32), dtype=torch.float)
        action = torch.tensor(np.array(action, dtype=np.float32), dtype=torch.float)
        reward = torch.tensor(np.array(reward, dtype=np.float32), dtype=torch.float).view(self.batch_size, -1)
        state_result = torch.tensor(np.array(state_result, dtype=np.float32), dtype=torch.float)

        def critic_learn():
            """
            critic learning
            @return:
            """
            a1 = self.actor_target(state_result).detach()
            y_true = reward + self.gamma * self.critic_target(state_result, a1).detach()

            y_pred = self.critic(state_current, action)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(state_current, self.actor(state_current)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
        return True

    def save_point(self, ext=None):
        if toolConfig.MODE == "PX4":
            filepath = "model/PX4"
        else:
            filepath = "model/Ardupilot"

        with open(f"{filepath}/actor.pth", "wb") as fp1, \
                open(f"{filepath}/critic.pth", "wb") as fp2, \
                open(f"{filepath}/actor_optimizer.pth", "wb") as fp3, \
                open(f"{filepath}/critic_optimizer.pth", "wb") as fp4:
            fcntl.flock(fp1, fcntl.LOCK_EX)
            fcntl.flock(fp2, fcntl.LOCK_EX)
            fcntl.flock(fp3, fcntl.LOCK_EX)
            fcntl.flock(fp4, fcntl.LOCK_EX)

            torch.save(self.actor, fp1)
            torch.save(self.critic, fp2)
            torch.save(self.actor_optim, fp3)
            torch.save(self.critic_optim, fp4)

            fcntl.flock(fp1, fcntl.LOCK_UN)
            fcntl.flock(fp2, fcntl.LOCK_UN)
            fcntl.flock(fp3, fcntl.LOCK_UN)
            fcntl.flock(fp4, fcntl.LOCK_UN)

    def check_point(self):
        if toolConfig.MODE == "PX4":
            filepath = "model/PX4"
        else:
            filepath = "model/Ardupilot"

        if os.path.exists(f"{filepath}/actor.pth"):

            while not (os.access(f"{filepath}/actor.pth", os.W_OK) and
                       os.access(f"{filepath}/critic.pth", os.W_OK) and
                       os.access(f"{filepath}/actor_optimizer.pth", os.W_OK) and
                       os.access(f"{filepath}/critic_optimizer.pth", os.W_OK)):
                continue

            with open(f"{filepath}/actor.pth", "rb") as fp1, \
                    open(f"{filepath}/critic.pth", "rb") as fp2, \
                    open(f"{filepath}/actor_optimizer.pth", "rb") as fp3, \
                    open(f"{filepath}/critic_optimizer.pth", "rb") as fp4:

                fcntl.flock(fp1, fcntl.LOCK_EX)
                fcntl.flock(fp2, fcntl.LOCK_EX)
                fcntl.flock(fp3, fcntl.LOCK_EX)
                fcntl.flock(fp4, fcntl.LOCK_EX)

                self.actor = torch.load(fp1)
                self.critic = torch.load(fp2)
                self.actor_optim = torch.load(fp3)
                self.actor_optim = torch.load(fp4)

                fcntl.flock(fp1, fcntl.LOCK_UN)
                fcntl.flock(fp2, fcntl.LOCK_UN)
                fcntl.flock(fp3, fcntl.LOCK_UN)
                fcntl.flock(fp4, fcntl.LOCK_UN)

            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)

        logging.info("Load previous model.")

    def train_from_incorrect(self, param_file):
        run_round = 1
        while True:
            try:
                time.sleep(1)
                print("\n--------------------------------------------------------------------------------------\n")

                logging.info("Change configuration.")
                self.env.get_random_incorrect_configuration(param_file)
                set_result = self.env.reset(delete_log=True)
                # reset fault, retry
                if not set_result:
                    continue

                # other device should load model at first
                if self.device != 0 and run_round % 16 == 0:
                    self.check_point()
                small_deviation = 0
                previous_deviation = 0
                for _ in range(80):
                    # monitor report error
                    if not self.env.manager.mav_monitor.msg_queue.empty():
                        self.env.manager.mav_monitor.msg_queue.get(block=True)
                        break
                    if small_deviation > 6:
                        break

                    # observe
                    cur_state = self.env.catch_state()
                    if cur_state is None:
                        # Not catch a state
                        break
                    logging.info(f"Observation deviation {self.env.cur_deviation}")
                    # break if the deviation is not change
                    if previous_deviation == self.env.cur_deviation:
                        break
                    # update previous deviation
                    previous_deviation = self.env.cur_deviation

                    # check threshold
                    if self.env.cur_deviation < self.env.deviation_threshold:
                        logging.debug(f"Deviation {round(self.env.cur_deviation, 4)} is small, no need to action.")
                        small_deviation += 1
                        # else keep flight
                        continue
                    else:
                        small_deviation = 0
                    logging.info("Deviation over threshold, start repair.")
                    # select action
                    action_0 = self.select_action(cur_state)
                    # translate the action to configuration
                    action_0 = self.action2config(action_0)
                    obe_state, reward, done = self.env.step(action_0)

                    if done:
                        self.save_point()
                        break
                    # print(cur_state.dtype, action_0.dtype, obe_state.dtype)
                    # state - action save to buffer
                    cur_state = cur_state.astype(float)
                    action_0 = action_0.astype(float)
                    obe_state = obe_state.astype(float)
                    self.put_once_buffer(cur_state, action_0, float(reward), obe_state)

                    # only device 0 is learning other device provides buffer
                    if int(self.device) == 0:
                        if not self.learn():
                            continue
                        else:
                            self.save_point()
                            self.save_buffer()

                    # index +1
                    run_round += 1
            except KeyboardInterrupt:
                if int(self.device) == 0:
                    self.save_point()
                    self.save_buffer()

                if int(self.device) == 0:
                    self.buffer.shm.close()
                    self.buffer.shm.unlink()
                self.env.close_env()
                operation = input("Any key to exit...")
                if operation == "c":
                    continue
                else:
                    return
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                frame = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # self.save_point()
                logging.info(f"Exception: {exc_type}, {frame}, {exc_tb.tb_lineno}, {e}.")
                # send_notice(self.device, self.buffer_length, [frame, exc_tb.tb_lineno, e])
                # input("Any key to continue...")
                continue

    def online_bin_monitor_rl(self):
        action_num = 0
        small_deviation = 0
        deviations_his = []
        while True:
            try:
                time.sleep(1)
                if not self.env.manager.mav_monitor.msg_queue.empty():
                    # receive error result
                    if not self.env.manager.mav_monitor.msg_queue.empty():
                        manager_msg, _ = self.env.manager.mav_monitor.msg_queue.get(block=True)
                        return manager_msg, action_num, deviations_his
                if small_deviation > 6:
                    return "pass", action_num, deviations_his

                cur_state = self.env.catch_state()
                logging.info(f"Observation deviation {self.env.cur_deviation}")
                deviations_his.append(self.env.cur_deviation)
                if self.env.cur_deviation > self.env.deviation_threshold:
                    logging.info(f"Deviation {self.env.cur_deviation} detect.")
                    action_0 = self.select_action(cur_state)
                    # translate the action to configuration
                    action_0 = self.action2config(action_0)
                    configuration = pd.DataFrame(action_0.reshape((-1, self.env.parameter_shape)),
                                                 columns=toolConfig.PARAM)
                    configuration = configuration.iloc[0].to_dict()
                    # print(configuration)
                    self.env.manager.online_mavlink.set_params(configuration)
                    action_num += 1
                    time.sleep(2)
                    small_deviation = 0
                else:
                    small_deviation += 1
            except KeyboardInterrupt:
                input("Any key to exit...")
                return "timeout", action_num, deviations_his
            except Exception as e:
                print(e)

    def close(self):
        self.env.close_env()
