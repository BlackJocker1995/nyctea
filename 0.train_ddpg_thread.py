import argparse
import os

import os

import argparse
import time

import gym
import ray
import torch

from Rl.learning_agent import DDPGAgent


# def save_point(filepath, *arges):
#     torch.save(arges[0], f"{filepath}/actor.pth")
#     torch.save(arges[1], f"{filepath}/critic.pth")
#     torch.save(arges[2], f"{filepath}/actor_target.pth")
#     torch.save(arges[3], f"{filepath}/critic_target.pth")
#
#
# def check_point(filepath):
#     actor = torch.load(f"{filepath}/actor.pth")
#     critic = torch.load(f"{filepath}/critic.pth")
#     actor_target = torch.load(f"{filepath}/actor_target.pth")
#     critic_target = torch.load(f"{filepath}/critic_target.pth")
#     return actor, critic, actor_target, critic_target

@ray.remote
def run_train(device):
    ddpg_agent = DDPGAgent(device=device)
    ddpg_agent.train_from_incorrect()
    ddpg_agent.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--thread', dest='thread', type=int, help='Name of the candidate', default=1)
    args = parser.parse_args()
    thread = args.thread
    thread = int(thread)
    print(thread)

    # threat_manage = []
    # ray.init(include_dashboard=True, dashboard_host="127.0.0.1", dashboard_port=8088)
    # for i in range(thread):
    #     threat_manage.append(run_train.remote(i))
    # ray.get(threat_manage)
    # ray.shutdown()

    for i in range(thread):
        cmd = f'gnome-terminal --tab --working-directory={os.getcwd()} -e ' \
              f'"python3 {os.getcwd()}/0.train_ddpg.py --device {i}"'
        os.system(cmd)
