import glob
import logging
import multiprocessing
import os
import random
import time
from abc import abstractmethod, ABC
from queue import Queue

import numpy as np
import pandas as pd
import ray
from pymavlink import mavutil
from pymavlink.DFReader import DFMessage
from pyulog import ULog
from tqdm import tqdm

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, StackBuffer
from collections import deque


class BoardMavlink(multiprocessing.Process):
    def __init__(self):
        super().__init__()
        self.data_segments = multiprocessing.Queue(1)
        self.log_file_path = None

    def wait_bin_ready(self):
        while True:
            time.sleep(0.1)
            if os.path.exists(self.log_file_path):
                break

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def extract_log_file(self, log_file, keep_des=False):
        pass

    @abstractmethod
    def fill_and_process_pd_log(self, pd_array: pd.DataFrame):
        pass

    @abstractmethod
    def init_binary_log_file(self, device_i=None):
        pass

    @abstractmethod
    def bin_read_last_seg(self, tail_n: int = 20):
        pass

    """
    Static Method
    """

    @staticmethod
    def extract_log_file_des_and_ach(log_file):
        """
        extract log message form a bin file with att desired and achieved
        :param log_file:
        :return:
        """

        logs = mavutil.mavlink_connection(log_file)
        # init
        out_data = []

        while True:
            msg = logs.recv_match(type=["ATT"])
            if msg is None:
                break
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'Roll': msg.Roll,
                'DesRoll': msg.DesRoll,
                'Pitch': msg.Pitch,
                'DesPitch': msg.DesPitch,
                'Yaw': msg.Yaw,
                'DesYaw': msg.DesYaw
            }
            out_data.append(out)

        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')
        return pd_array

    @ray.remote
    def extract_log_path_threat(self, log_path, file_list, keep_des, skip):
        """
        threat method to extract data from log.
        :param log_path:
        :param file_list:
        :param keep_des: whether keep desired value of ATT and RATE
        :param skip:
        :return:
        """
        for file in tqdm(file_list):
            name, _ = file.split('.')
            if skip and os.path.exists(f'{log_path}/csv/{name}.csv'):
                continue
            try:
                csv_data = self.extract_log_file(log_path + f'/{file}', keep_des)
                csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
            except Exception as e:
                logging.warning(f"Error processing {file} : {e}")
                continue
        return True

    @staticmethod
    def _fill_and_process_public(pd_array: pd.DataFrame):
        """
        Public process of "fill_and_process_pd_log" function
        :param pd_array:
        :return:
        """
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')
        # merge data in same TimeS
        df_array = pd.DataFrame(columns=pd_array.columns)
        for group, group_item in pd_array.groupby('TimeS'):
            # fillna
            group_item = group_item.fillna(method='ffill')
            group_item = group_item.fillna(method='bfill')
            df_array.loc[len(df_array.index)] = group_item.mean()
        # Drop nan
        df_array = df_array.fillna(method='ffill')
        df_array = df_array.dropna()

        return df_array

    @staticmethod
    def _order_sort(df_array):
        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))
        # Status value + Parameter name
        order_name.extend(param_name)
        df_array = df_array[order_name]
        return df_array


class BoardMavlinkAPM(BoardMavlink):
    def __init__(self):
        super().__init__()
        self.flight_log = None
        self.param_current = dict()

    def init_binary_log_file(self, device_i=None):
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"
        # Read last index
        with open(log_index, 'r') as f:
            num = int(f.readline())  + 1
            if num == 501:
                num = 1
            # To string
        num = f'{num}'
        self.log_file_path = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/{num.rjust(8, '0')}.BIN"
        logging.info(f"Current log file: {self.log_file_path}")

    def wait_bin_ready(self):
        while True:
            time.sleep(0.1)
            if os.path.exists(self.log_file_path):
                break

    def init_current_param(self):
        # inti param value
        accpet_param = load_param().columns.to_list()
        while len(self.param_current) <= len(accpet_param):
            msg = self.flight_log.recv_match(type=["PARM"], blocking=True)
            if msg.Name in accpet_param:
                self.param_current.update(self.log_extract_apm(msg))
        self.param_current.pop('TimeS')

    def bin_read_last_seg(self, tail_n: int = 20):
        while self.data_segments.empty():
            time.sleep(0.1)
        pd_array = pd.DataFrame(self.data_segments.get())
        # Switch sequence, fill,  and return
        # pd_array = self.fill_and_process_pd_log(pd_array)
        pd_array = pd_array.tail(tail_n)
        return pd_array

    @staticmethod
    def log_extract_apm(msg: DFMessage, keep_des=False):
        """
        parse the msg of mavlink
        :param keep_des: whether keep att and rate desired and achieve value
        :param msg:
        :return:
        """
        out = None
        if msg.get_type() == 'ATT':
            # if len(toolConfig.LOG_MAP):
            if not keep_des:
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    'Roll': msg.Roll,
                    'Pitch': msg.Pitch,
                    'Yaw': msg.Yaw,
                }
            else:
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    "DesRoll": msg.DesRoll,
                    'Roll': msg.Roll,
                    'DesPitch': msg.DesPitch,
                    'Pitch': msg.Pitch,
                    'DesYaw': msg.DesYaw,
                    'Yaw': msg.Yaw,
                }
        elif msg.get_type() == 'RATE':
            if not keep_des:
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    # deg to rad
                    'RateRoll': msg.R,
                    'RatePitch': msg.P,
                    'RateYaw': msg.Y,
                }
            else:
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    # deg to rad
                    'DesRateRoll': msg.RDes,
                    'RateRoll': msg.R,
                    'DesRatePitch': msg.PDes,
                    'RatePitch': msg.P,
                    'DesRateYaw': msg.YDes,
                    'RateYaw': msg.Y,
                }
        # elif msg.get_type() == 'POS':
        #     out = {
        #         'TimeS': msg.TimeUS / 1000000,
        #         # deglongtitude
        #         'Lat': msg.Lat,
        #         'Lng': msg.Lng,
        #         'Alt': msg.Alt,
        #     }
        elif msg.get_type() == 'IMU':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'AccX': msg.AccX,
                'AccY': msg.AccY,
                'AccZ': msg.AccZ,
                'GyrX': msg.GyrX,
                'GyrY': msg.GyrY,
                'GyrZ': msg.GyrZ,
            }
        elif msg.get_type() == 'VIBE':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                # m/s^2
                'VibeX': msg.VibeX,
                'VibeY': msg.VibeY,
                'VibeZ': msg.VibeZ,
            }
        elif msg.get_type() == 'MAG':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'MagX': msg.MagX,
                'MagY': msg.MagY,
                'MagZ': msg.MagZ,
            }
        elif msg.get_type() == 'PARM':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                msg.Name: msg.Value
            }
        elif msg.get_type() == 'GPS':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'Lat': msg.Lat,
                'Lng': msg.Lng,
                'Alt': msg.Alt,
            }
        return out

    @staticmethod
    def delete_current_log():
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"

        # Read last index
        with open(log_index, 'r') as f:
            num = int(f.readline())
        # To string
        num = f'{num}'
        log_file = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/{num.rjust(8, '0')}.BIN"
        # Remove file
        if os.path.exists(log_file):
            os.remove(log_file)
            # Fix last index number
            last_num = f"{int(num) - 1}"
            with open(log_index, 'w') as f:
                f.write(last_num)

    def extract_log_file(self, log_file, keep_des=False):
        """
        extract log message form a bin file.
        :param keep_des:
        :param log_file:
        :return:
        """
        accept_item = toolConfig.LOG_MAP

        logs = mavutil.mavlink_connection(log_file)
        # init
        out_data = []
        accept_param = load_param().columns.to_list()

        while True:
            msg = logs.recv_match(type=accept_item)
            if msg is None:
                break
            # Skip if not index 0 sensor
            # SKip is param is not we want
            if (hasattr(msg, "I") and msg.I != 0) or \
                    (hasattr(msg, "IMU") and msg.IMU != 0) or \
                    (msg.get_type() == 'PARM' and msg.Name not in accept_param):
                continue
            # Otherwise Record
            out_data.append(self.log_extract_apm(msg, keep_des))
        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array = self.fill_and_process_pd_log(pd_array)
        return pd_array

    def fill_and_process_pd_log(self, pd_array: pd.DataFrame):
        """
        pre-process the data collected.
        :param pd_array:
        :return:
        """
        # Remain timestamp .1 and drop duplicate
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        df_array = self._fill_and_process_public(pd_array)
        # Sort
        df_array = self._order_sort(df_array)
        return df_array

    def get_time_index_bin(self, msg):
        """
        As different message have different time unit. It needs to convert to same second unit.
        :return:
        """
        return msg.TimeUS / 1000000

    def init_current_param(self):
        # inti param value
        accpet_param = load_param().columns.to_list()
        while len(self.param_current) <= len(accpet_param):
            msg = self.flight_log.recv_match(type=["PARM"], blocking=True)
            if msg.Name in accpet_param:
                self.param_current.update(self.log_extract_apm(msg))
        self.param_current.pop('TimeS')

    def read_status_patch_bin(self, time_last, time_unit: float):
        time_unit = float(time_unit)
        out_data = []
        accept_item = toolConfig.LOG_MAP.copy()
        accept_item_ex_param = accept_item.copy()
        accept_item_ex_param.remove("PARM")
        accpet_param = load_param().columns.to_list()
        self.init_current_param()

        # Walk to the first message
        while True:
            msg = self.flight_log.recv_match(type=accept_item_ex_param)
            if msg is None:
                return None
            elif msg.TimeUS > time_last:
                # Get first message
                first_msg = msg
                first_time = self.get_time_index_bin(first_msg)
                first_msg = self.log_extract_apm(first_msg)
                # logging.debug(f"Current status at {first_time} second.")

                first_msg.update(self.param_current)
                out_data.append(first_msg)
                new_time = first_time
                break

        # Collect data in one time_unit
        while new_time < first_time + time_unit:
            msg = self.flight_log.recv_match(type=accept_item)
            if msg is None:
                break
            if msg.get_type() in ['ATT', 'RATE']:
                out_data.append(self.log_extract_apm(msg, True))
            elif msg.get_type() in ['IMU', 'MAG'] and msg.I == 0:
                out_data.append(self.log_extract_apm(msg))
            elif msg.get_type() == 'VIBE' and msg.IMU == 0:
                out_data.append(self.log_extract_apm(msg))
            elif msg.get_type() == 'PARM' and msg.Name in accpet_param:
                tmp = self.log_extract_apm(msg)
                print(tmp)
                tmp.pop("TimeS")
                continue
            new_time = self.get_time_index_bin(msg)
        # To DataFrame
        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array = self.fill_and_process_pd_log(pd_array)
        if pd_array.shape[0] < (time_unit * 10):
            return False
        return pd_array

    def run(self):
        time_last = 0
        # Wait for bin file created
        self.wait_bin_ready()
        _file = open(self.log_file_path, 'rb')
        accept_item = toolConfig.LOG_MAP.copy()

        while True:

            time.sleep(1)
            # Flush write buffer
            _file.flush()
            # Load current log file
            self.flight_log = mavutil.mavlink_connection(self.log_file_path)
            self.init_current_param()
            try:
                # Read flight status
                status_data = self.read_status_patch_bin(time_last, 2)

                # Check landed or read failure
                if status_data is None:
                    time.sleep(0.1)
                    logging.info("Successful break the loop.")
                    return "pass", "repair"
                elif status_data is False:
                    logging.debug("Reading status failure, try again.")
                    continue
                # send this message
                self.data_segments.put(status_data)
            except Exception as e:
                logging.warning(f"{e}, then continue looping")

            # Drop old message
            while True:
                msg = self.flight_log.recv_match(type=accept_item)
                if msg is None:
                    break
                else:
                    # Update timestamp
                    time_last = msg.TimeUS


class BoardMavlinkPX4(BoardMavlink, ABC):
    def __init__(self):
        super().__init__()

    def fill_and_process_pd_log(self, pd_array: pd.DataFrame):
        df_array = self._fill_and_process_public(pd_array)
        return df_array

    def extract_log_file(self, log_file, keep_des=False):
        """
        extract log message form a bin file.
        :param log_file:
        :return:
        """

        ulog = ULog(log_file)

        if keep_des:
            bias = pd.DataFrame(ulog.get_dataset('estimator_sensor_bias').data)[["timestamp", "accel_bias[0]",
                                                                                 "accel_bias[1]", "accel_bias[2]"]]
            bias.columns = ["TimeS", "BiasA", "BiasB", "BiasC"]

        att = pd.DataFrame(ulog.get_dataset('vehicle_attitude_setpoint').data)[["timestamp",
                                                                                "roll_body", "pitch_body", "yaw_body"]]
        rate = pd.DataFrame(ulog.get_dataset('vehicle_rates_setpoint').data)[["timestamp",
                                                                              "roll", "pitch", "yaw"]]
        acc_gyr = pd.DataFrame(ulog.get_dataset('sensor_combined').data)[["timestamp",
                                                                          "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]",
                                                                          "accelerometer_m_s2[0]",
                                                                          "accelerometer_m_s2[1]",
                                                                          "accelerometer_m_s2[2]"]]
        mag = pd.DataFrame(ulog.get_dataset('sensor_mag').data)[["timestamp", "x", "y", "z"]]
        vibe = pd.DataFrame(ulog.get_dataset('sensor_accel').data)[["timestamp", "x", "y", "z"]]
        # Param
        param = pd.Series(ulog.initial_parameters)
        param = param[toolConfig.PARAM]
        # select parameters
        for t, name, value in ulog.changed_parameters:
            if name in toolConfig.PARAM:
                param[name] = round(value, 5)

        att.columns = ["TimeS", "Roll", "Pitch", "Yaw"]
        rate.columns = ["TimeS", "RateRoll", "RatePitch", "RateYaw"]
        acc_gyr.columns = ["TimeS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
        mag.columns = ["TimeS", "MagX", "MagY", "MagZ"]
        vibe.columns = ["TimeS", "VibeX", "VibeY", "VibeZ"]
        # Merge values
        if keep_des:
            pd_array = pd.concat([att, rate, acc_gyr, mag, vibe, bias]).sort_values(by='TimeS')
        else:
            pd_array = pd.concat([att, rate, acc_gyr, mag, vibe]).sort_values(by='TimeS')

        # Process
        df_array = self.fill_and_process_pd_log(pd_array)
        # Add parameters
        param_values = np.tile(param.values, df_array.shape[0]).reshape(df_array.shape[0], -1)
        df_array[toolConfig.PARAM] = param_values

        # Sort
        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))

        return df_array

    @classmethod
    def delete_current_log(cls):
        log_path = f"{toolConfig.PX4_LOG_PATH}/*.ulg"

        list_of_files = glob.glob(log_path)  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        # Remove file
        if os.path.exists(latest_file):
            os.remove(latest_file)
