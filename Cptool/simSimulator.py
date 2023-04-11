import logging
import multiprocessing
import random
import time

import airsim

from Cptool.config import toolConfig


class SimSimulator(multiprocessing.Process):
    def __init__(self, recv_msg_queue: multiprocessing.Queue, send_msg_queue: multiprocessing.Queue):
        super(SimSimulator, self).__init__()
        self.recv_msg_queue = recv_msg_queue
        self.send_msg_queue = send_msg_queue

        if toolConfig.DEBUG:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.INFO)


class GaSimSimulator(SimSimulator):
    def __init__(self, recv_msg_queue: multiprocessing.Queue, send_msg_queue: multiprocessing.Queue):
        super(SimSimulator, self).__init__()
        self.recv_msg_queue = recv_msg_queue
        self.send_msg_queue = send_msg_queue
        self._client = airsim.MultirotorClient()
        self._wind = False

    def confirm_api(self):
        """
        使用API链接Airsim模拟器
        :return:
        """
        self._client.confirmConnection()
        self._client.enableApiControl(True)

    def reset_item(self):
        """
        模拟器重设
        :return:
        """
        logging.info('Reset item!')
        self._client.reset()
        self._client.enableApiControl(True)

    def arm_disarm(self, arm: bool):
        """
        通过Airsim解锁UAV
        :param arm: 是否解锁
        :return:
        """
        self._client.armDisarm(arm)

    def set_wind(self, x, y, z):
        """
        添加阵风，单位是m/s
        :param x:
        :param y:
        :param z:
        :return:
        """
        logging.info(f'Set wind {x} - {y} - {z}.')
        wind = airsim.Vector3r(x, y, z)
        self._client.simSetWind(wind)

    def set_wind_random(self, button, top):
        """
        添加阵风，单位是m/s
        :param x:
        :param y:
        :param z:
        :return:
        """
        value = [-1, 1]

        x = random.uniform(button, top) * random.choice(value)
        y = random.uniform(button, top) * random.choice(value)
        z = random.uniform(button, top) * random.choice(value)
        self.set_wind(x, y, z)

    def clear_wind(self):
        wind = airsim.Vector3r(0, 0, 0)
        self._client.simSetWind(wind)

    def pause(self):
        self._client.simPause(True)

    def resume(self):
        self._client.simPause(False)

    def run(self) -> None:
        self.set_wind(0, 0, 6)
        time.sleep(2)
        self.clear_wind()