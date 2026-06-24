"""The DDPG repair agent.

Rewritten from the legacy ``Rl/learning_agent.py``. The RL math (DDPG
hyperparameters, the actor/critic update, soft-update τ) is preserved verbatim;
the engineering around it changed:

- **No Ray, no ``shared_memory.ShareableList``.** the replay buffer is now
  :class:`~nyctea.buffer.ReplayBuffer` (npz shards + in-memory sampling).
- **No ``tensorflow``/``pickle``.** the unused ``import tensorflow`` is gone;
  buffer/checkpoint I/O uses torch + npz.
- **No ``os.access`` busy-waits.** checkpoint read/write goes through
  :mod:`nyctea.model_io` (fcntl.flock); the buffer uses atomic temp-rename.
- **``action2config`` / random params** delegated to :mod:`nyctea.rl.actions`.

The agent still supports the multi-instance training topology: workers with
``device != 0`` only collect transitions (they ``add`` to a buffer whose shards
the device-0 learner ``reload``s and learns from); only device 0 calls ``learn``.
"""
import copy
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from nyctea.buffer import ReplayBuffer
from nyctea.config import toolConfig
from nyctea.model_io import load_checkpoint, save_checkpoint
from nyctea.params import (
    get_default_values, load_param, read_range_from_dict, read_unit_from_dict,
)
from nyctea.rl.actions import action2config as _action2config
from nyctea.rl.actor_critic import Actor, Critic
from nyctea.rl.env import DroneEnv


class ReLearningAgent:
    """Holds the param spec (ranges/steps/defaults) shared by all agents."""

    def __init__(self):
        self.participle_param = toolConfig.PARAM_PART
        para_dict = load_param()
        self.step_unit = read_unit_from_dict(para_dict)
        self.default_pop = get_default_values(para_dict).to_numpy(dtype=int)
        self.sub_value_range = read_range_from_dict(para_dict)

    def action2config(self, action):
        """Map a ``[0,1]`` action to step-quantized real params (pure fn)."""
        return _action2config(action, self.sub_value_range, self.step_unit)

    @staticmethod
    def create_random_params() -> np.ndarray:
        """A ``[0,1]`` random action vector of full param length."""
        return np.random.random(len(toolConfig.PARAM))


class DDPGAgent(ReLearningAgent):
    """Full DDPG: actor/critic + targets, replay buffer, online repair loop."""

    def __init__(self, train=True, device=0):
        super().__init__()
        self.device = int(device)
        self.env = DroneEnv(device=device, debug=toolConfig.DEBUG)
        # DDPG hyperparameters (unchanged from the legacy agent).
        self.gamma = 0.99
        self.actor_lr = 0.01
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = toolConfig.CAPACITY
        self.batch_size = toolConfig.BATCH_SIZE

        # state shape (tail_n, 12) -> (1, tail_n * 12); action = full param set.
        state_shape = self.env.tail_n * 12
        action_dim = len(toolConfig.PARAM)

        self.buffer = ReplayBuffer(
            buffer_dir=toolConfig.buffer_dir(), worker_id=self.device,
            capacity=self.capacity, state_dim=state_shape, action_dim=action_dim)

        # networks
        hidden = toolConfig.HIDDEN
        self.actor = Actor(state_shape, hidden, action_dim)
        self.actor_target = Actor(state_shape, hidden, action_dim)
        self.critic = Critic(state_shape + action_dim, hidden, action_dim)
        self.critic_target = Critic(state_shape + action_dim, hidden, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        if self.device == 0:
            ckpt = load_checkpoint()
            if ckpt is not None:
                actor, critic, actor_opt, critic_opt = ckpt
                self.actor, self.critic = actor, critic
                self.actor_optim, self.critic_optim = actor_opt, critic_opt
                self.actor_target = copy.deepcopy(self.actor)
                self.critic_target = copy.deepcopy(self.critic)
            # Seed the in-memory view from existing shards (resumes a run).
            self.buffer.reload()

    # --------------------------------------------------------------- acting
    def select_action(self, state):
        """Run the actor on ``state``; return a detached ``[0,1]`` action."""
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        return action

    def put_once_buffer(self, state, action, reward, next_state):
        """Append a transition to this worker's buffer shard."""
        self.buffer.add(state, action, reward, next_state)

    # --------------------------------------------------------------- learning
    def learn(self):
        """One DDPG update step. Returns True if a step was taken.

        Skips if the buffer is too small or not on a batch boundary (matches the
        legacy gating: only learn every ``batch_size`` new transitions).
        """
        # device 0 reloads periodically; if the buffer has no memory view yet,
        # reload now.
        if self.buffer._mem_len == 0:
            self.buffer.reload()
        n = len(self.buffer)
        if n <= self.batch_size:
            logger.debug(f"Current buffer size is {n}, skip.")
            return False
        if n % self.batch_size > 10:
            return False

        logger.info(f"Take example and learn, example/buffer size: {self.batch_size}/{n}.")
        state_current, action, reward, state_result = self.buffer.sample(self.batch_size)

        state_current = torch.tensor(state_current, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float).view(self.batch_size, -1)
        state_result = torch.tensor(state_result, dtype=torch.float)

        # --- critic update ---
        a1 = self.actor_target(state_result).detach()
        y_true = reward + self.gamma * self.critic_target(state_result, a1).detach()
        y_pred = self.critic(state_current, action)
        loss = nn.MSELoss()(y_pred, y_true)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        # --- actor update ---
        actor_loss = -torch.mean(self.critic(state_current, self.actor(state_current)))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- soft target update ---
        self._soft_update(self.critic_target, self.critic, self.tau)
        self._soft_update(self.actor_target, self.actor, self.tau)
        return True

    @staticmethod
    def _soft_update(target_net, net, tau):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # --------------------------------------------------------------- save
    def save_point(self):
        """Persist actor/critic + optimizers under flock."""
        save_checkpoint(self.actor, self.critic,
                        self.actor_optim, self.critic_optim)

    def check_point(self):
        """Reload the checkpoint into this agent (non-device-0 workers)."""
        ckpt = load_checkpoint()
        if ckpt is None:
            return
        actor, critic, actor_opt, critic_opt = ckpt
        self.actor, self.critic = actor, critic
        self.actor_optim, self.critic_optim = actor_opt, critic_opt
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

    # --------------------------------------------------------------- loops
    def train_from_incorrect(self, param_file):
        """Main training loop: fly a bad config, repair it, learn from the fix."""
        run_round = 1
        while True:
            try:
                time.sleep(1)
                logger.info("Change configuration.")
                self.env.get_random_incorrect_configuration(param_file)
                if not self.env.reset(delete_log=True):
                    continue

                # non-primary workers periodically resync the policy.
                if self.device != 0 and run_round % 16 == 0:
                    self.check_point()

                small_deviation = 0
                previous_deviation = 0
                for _ in range(80):
                    if not self.env.manager.mav_monitor.msg_queue.empty():
                        self.env.manager.mav_monitor.msg_queue.get(block=True)
                        break
                    if small_deviation > 6:
                        break

                    cur_state = self.env.catch_state()
                    if cur_state is None:
                        break
                    logger.info(f"Observation deviation {self.env.cur_deviation}")
                    if previous_deviation == self.env.cur_deviation:
                        break
                    previous_deviation = self.env.cur_deviation

                    if self.env.cur_deviation < self.env.deviation_threshold:
                        logger.debug(f"Deviation {round(self.env.cur_deviation, 4)} is small, no need to action.")
                        small_deviation += 1
                        continue
                    else:
                        small_deviation = 0
                    logger.info("Deviation over threshold, start repair.")

                    action_0 = self.select_action(cur_state)
                    action_0 = self.action2config(action_0)
                    obe_state, reward, done = self.env.step(action_0)

                    if done:
                        self.save_point()
                        break
                    self.put_once_buffer(
                        cur_state.astype(float), action_0.astype(float),
                        float(reward), obe_state.astype(float))

                    # Only device 0 learns; others just collect transitions.
                    if self.device == 0:
                        if self.learn():
                            self.save_point()
                            self.buffer.flush()
                    run_round += 1
            except KeyboardInterrupt:
                if self.device == 0:
                    self.save_point()
                    self.buffer.flush()
                self.env.close_env()
                operation = input("Any key to exit...")
                if operation == "c":
                    continue
                return
            except Exception as e:
                exc_type, _, exc_tb = sys.exc_info()
                frame = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.info(f"Exception: {exc_type}, {frame}, {exc_tb.tb_lineno}, {e}.")
                continue

    def online_bin_monitor_rl(self):
        """Inference loop: observe deviation, upload a repair config if over threshold.

        Returns ``(result, action_num, deviations_history)``.
        """
        action_num = 0
        small_deviation = 0
        deviations_his = []
        while True:
            try:
                time.sleep(1)
                if not self.env.manager.mav_monitor.msg_queue.empty():
                    manager_msg, _ = self.env.manager.mav_monitor.msg_queue.get(block=True)
                    return manager_msg, action_num, deviations_his
                if small_deviation > 6:
                    return "pass", action_num, deviations_his

                cur_state = self.env.catch_state()
                logger.info(f"Observation deviation {self.env.cur_deviation}")
                deviations_his.append(self.env.cur_deviation)
                if self.env.cur_deviation > self.env.deviation_threshold:
                    logger.info(f"Deviation {self.env.cur_deviation} detect.")
                    action_0 = self.select_action(cur_state)
                    action_0 = self.action2config(action_0)
                    configuration = pd.DataFrame(action_0.reshape((-1, self.env.parameter_shape)),
                                                 columns=toolConfig.PARAM).iloc[0].to_dict()
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
                logger.warning(e)

    def close(self):
        self.env.close_env()
