import os

import argparse
import gym
import torch

from Cptool.config import toolConfig
from Rl.learning_agent import DDPGAgent


def apm_run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--device', dest='device', type=str, help='Name of the candidate')
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = 0
    param_file = f"validation/{toolConfig.MODE}/params.csv"
    ddpg_agent = DDPGAgent(device=device)
    ddpg_agent.train_from_incorrect(param_file)
    ddpg_agent.close()


if __name__ == "__main__":
    apm_run()
