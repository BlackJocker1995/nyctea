import fcntl
import json
import logging
import os
import pickle
import random
import time
from abc import abstractmethod
from copy import deepcopy

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Rl.actor_critic import Actor, Critic

tf.compat.v1.disable_eager_execution()

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, select_sub_dict, read_unit_from_dict, get_default_values, read_range_from_dict, \
    send_notice

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

    def action2range(self, action):
        """
        output action is [0, 1], transform them to original range of parameters.
        @param action: configuration action
        @return:
        """
        step_increase = (action * (self.sub_value_range[:, 1] - self.sub_value_range[:, 0])
                         // self.step_unit) * self.step_unit
        return self.sub_value_range[:, 0] + step_increase

    @abstractmethod
    def save_point(self):
        pass

    @abstractmethod
    def check_point(self):
        pass


class DDPGAgent(ReLearningAgent):
    def __init__(self, device=0, *args):
        super().__init__()
        self.device = int(device)
        # play environment
        self.env = DroneEnv(device=device, debug=toolConfig.DEBUG)
        # gamma
        self.gamma = 0.99
        # actor learning rate
        self.actor_lr = 0.001
        # critic learning rate
        self.critic_lr = 0.001
        # tau
        self.tau = 0.02
        # buffer capacity
        self.capacity = 20000
        # learning batch size
        self.batch_size = 64

        # Varies by environment
        # state shape (tail_n, 18) -> (1, tail_n * 18)
        state_shape = self.env.tail_n * 18
        # parameter element in configuration
        action_dim = len(toolConfig.PARAM)

        # Buffer length
        self.buffer_length = 0

        if len(args) > 0:
            self.actor = args[0]
            self.actor_target = args[1]
            self.critic = args[2]
            self.critic_target = args[3]
            self.actor_optim = args[4]
            self.critic_optim = args[5]
        else:
            # actor network
            self.actor = Actor(state_shape, 1024, action_dim)
            self.actor_target = Actor(state_shape, 1024, action_dim)
            # critic network
            self.critic = Critic(state_shape + action_dim, 1024, action_dim)
            self.critic_target = Critic(state_shape + action_dim, 1024, action_dim)
            # network map
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            # optimizer
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

            self.check_point()

        # learning buffer, to save memory
        self.buffer = []

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

    @classmethod
    def save_once_buffer(cls, *transition):
        logging.debug("Load buffer and put transition")
        if toolConfig.MODE == "PX4":
            pathfile = f"{toolConfig.BUFFER_PATH}/PX4/buffer.pkl"
        else:
            pathfile = f"{toolConfig.BUFFER_PATH}/Ardupilot/buffer.pkl"

        # read local buffer file
        if os.path.exists(pathfile):
            while os.path.exists(pathfile) and not os.access(pathfile, os.W_OK):
                continue
            with open(pathfile, "ab+") as fp:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pik_data = pickle.dumps(transition)
                fp.write(pik_data)
                fp.write(b";;\n")
                fcntl.flock(fp, fcntl.LOCK_UN)
        else:
            with open(pathfile, "wb+") as fp:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pik_data = pickle.dumps(transition)
                fp.write(pik_data)
                fp.write(b";;\n")
                fcntl.flock(fp, fcntl.LOCK_UN)
        #     with open(pathfile, "rb") as fp:
        #         _buffer = pickle.load(fp)
        # # If over buffer size
        # if len(_buffer) == self.capacity:
        #     _buffer.pop(0)
        # _buffer.append(transition)
        # self.buffer_length = len(_buffer)
        # # Save to local
        # while os.path.exists(pathfile) and not os.access(pathfile, os.W_OK):
        #     continue
        # with open(pathfile, "wb") as fp:
        #     pickle.dump(_buffer, fp)

    @classmethod
    def load_buffer(cls):
        raw_datas = []
        if toolConfig.MODE == "PX4":
            pathfile = f"{toolConfig.BUFFER_PATH}/PX4/buffer.pkl"
        else:
            pathfile = f"{toolConfig.BUFFER_PATH}/Ardupilot/buffer.pkl"
        while os.path.exists(pathfile) and not os.access(pathfile, os.R_OK):
            continue
        with open(pathfile, "rb") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            tmp_obj = fp.read()
            fcntl.flock(fp, fcntl.LOCK_UN)
        # binary split
        tmp_obj_array = tmp_obj.split(b";;\n")
        for item in tmp_obj_array:
            if len(item) != 0:
                raw_datas.append(pickle.loads(item))
        return raw_datas

    def learn(self):
        """
        learning a buffer
        @return:
        """
        # Load buffer
        self.buffer = self.load_buffer()
        # return if buffer is too small
        if len(self.buffer) < self.batch_size:
            return False

        logging.info(f"Take example and learn, example/buffer size: {self.batch_size}/{len(self.buffer)}.")
        # take a sample
        samples = random.sample(self.buffer, self.batch_size)
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

        if self.device == 0:
            if os.path.exists(f"{filepath}/actor.pth"):
                while not (os.access(f"{filepath}/actor.pth", os.W_OK) and
                           os.access(f"{filepath}/critic.pth", os.W_OK) and
                           os.access(f"{filepath}/actor_target.pth", os.W_OK) and
                           os.access(f"{filepath}/critic_target.pth", os.W_OK)
                ):
                    continue
            torch.save(self.actor, f"{filepath}/actor.pth")
            torch.save(self.critic, f"{filepath}/critic.pth")
            torch.save(self.actor_target, f"{filepath}/actor_target.pth")
            torch.save(self.critic_target, f"{filepath}/critic_target.pth")

        if self.device == 0 and ext is not None:
            ext = int(ext)
            torch.save(self.actor, f"{filepath}/his/actor{ext}.pth")
            torch.save(self.critic, f"{filepath}/his/critic{ext}.pth")
            torch.save(self.actor_target, f"{filepath}/his/actor_targe{ext}.pth")
            torch.save(self.critic_target, f"{filepath}/his/critic_target{ext}.pth")

    def check_point(self):
        if toolConfig.MODE == "PX4":
            filepath = "model/PX4"
        else:
            filepath = "model/Ardupilot"
        if os.path.exists(f"{filepath}/actor.pth"):
            while not (os.access(f"{filepath}/actor.pth", os.R_OK) and
                       os.access(f"{filepath}/critic.pth", os.R_OK) and
                       os.access(f"{filepath}/actor_target.pth", os.R_OK) and
                       os.access(f"{filepath}/critic_target.pth", os.R_OK)):
                continue

            self.actor = torch.load(f"{filepath}/actor.pth")
            self.critic = torch.load(f"{filepath}/critic.pth")
            self.actor_target = torch.load(f"{filepath}/actor_target.pth")
            self.critic_target = torch.load(f"{filepath}/critic_target.pth")
            # logging.info("Load previous model.")

    def train_from_incorrent(self):
        while True:
            try:
                time.sleep(1)
                print("\n--------------------------------------------------------------------------------------\n")

                logging.info("Change configuration.")
                self.env.get_random_incorrent_configuration()
                self.env.reset()

                previous_deviation = 0
                for i in range(80):
                    # monitor report error
                    if not self.env.manager.mav_monitor.msg_queue.empty():
                        self.env.manager.mav_monitor.msg_queue.get(block=True)
                        break

                    # observe
                    cur_state = self.env.catch_state()
                    if cur_state is None:
                        # Not catch a state
                        continue
                    logging.info(f"Observation deviation {self.env.cur_deviation}")
                    # break
                    if previous_deviation == self.env.cur_deviation:
                        break
                    previous_deviation = self.env.cur_deviation
                    # check threshold
                    if self.env.cur_deviation < self.env.deviation_threshold:
                        logging.debug(f"Deviation {round(self.env.cur_deviation, 4)} is small, no need to action.")
                        # else keep flight
                        continue

                    # only device 0 main model
                    if self.device != 0:
                        self.check_point()
                    # select action
                    action_0 = self.select_action(cur_state)
                    action_0 = self.action2range(action_0)
                    obe_state, reward, done = self.env.step(action_0)

                    if done:
                        self.save_point()
                        break
                    # state - action save to buffer
                    self.save_once_buffer(cur_state, action_0, reward, obe_state)

                    # only device 0 is learning other device provides buffer
                    if int(self.device) == 0:
                        if not self.learn():
                            continue

                        if self.buffer_length % 100 < 7:
                            self.save_point(ext=(self.buffer_length // 100) * 100)
                        else:
                            self.save_point()

            except KeyboardInterrupt:
                self.save_point()
                return
            except Exception as e:
                logging.info(f"Exception: {e}, and save model")
                # self.save_point()
                send_notice(self.device, self.buffer_length, e)
                exit(-1)
