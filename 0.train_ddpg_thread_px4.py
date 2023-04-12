import argparse
import os

import os

import argparse
import time

import gym
import torch

from Cptool.config import toolConfig
from Rl.learning_agent import DDPGAgent



if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--thread', dest='thread', type=int, help='Name of the candidate', default=1)
    args = parser.parse_args()
    thread = args.thread
    thread = int(thread)
    print(thread)

    for i in range(thread):
        cmd = f'gnome-terminal --tab --working-directory={os.getcwd()} -e ' \
              f'"python3 {os.getcwd()}/0.train_ddpg.py --device {i}"'
        os.system(cmd)
