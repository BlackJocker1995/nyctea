import argparse
import os

import os

import argparse
import time

import gym
import ray
import torch

from Cptool.config import toolConfig
from Rl.learning_agent import DDPGAgent


@ray.remote
def run_train(param_file, device):

    ddpg_agent = DDPGAgent(device=device)
    ddpg_agent.train_from_incorrent(param_file)
    ddpg_agent.close()


if __name__ == '__main__':
    """
    If you running this program in a cmd environment, you can use this script.
    127.0.0.1:8080 will demonstrate the detail of all processes.
    """
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--thread', dest='thread', type=int, help='Name of the candidate', default=1)
    args = parser.parse_args()
    thread = args.thread
    thread = int(thread)
    print(thread)
    param_file = f"validation/{toolConfig.MODE}/params.csv"
    threat_manage = []
    ray.init(include_dashboard=True, dashboard_host="127.0.0.1", dashboard_port=8088)
    for i in range(thread):
        threat_manage.append(run_train.remote(param_file,i))
    ray.get(threat_manage)
    ray.shutdown()
