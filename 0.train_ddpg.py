import os

import argparse
import gym
import torch

from Rl.learning_agent import DDPGAgent



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--device', dest='device', type=str, help='Name of the candidate')
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = 0
    ddpg_agent = DDPGAgent(device=device)
    ddpg_agent.train_from_incorrent()
