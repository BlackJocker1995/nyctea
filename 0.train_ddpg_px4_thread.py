import argparse
import os

import os

import argparse
import time

import gym
import torch

from Rl.learning_agent import DDPGAgent


def save_point(filepath, *arges):
    torch.save(arges[0], f"{filepath}/actor.pth")
    torch.save(arges[1], f"{filepath}/critic.pth")
    torch.save(arges[2], f"{filepath}/actor_target.pth")
    torch.save(arges[3], f"{filepath}/critic_target.pth")


def check_point(filepath):
    actor = torch.load(f"{filepath}/actor.pth")
    critic = torch.load(f"{filepath}/critic.pth")
    actor_target = torch.load(f"{filepath}/actor_target.pth")
    critic_target = torch.load(f"{filepath}/critic_target.pth")
    return actor, critic, actor_target, critic_target


if __name__ == '__main__':
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
