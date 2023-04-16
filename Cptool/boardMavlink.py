import glob
import logging
import multiprocessing
import os
import random
import time
from abc import abstractmethod, ABC
from queue import Queue

import eventlet
import numpy as np
import pandas as pd
import ray
from pymavlink import mavutil
from pymavlink.DFReader import DFMessage
from pyulog import ULog
from tqdm import tqdm

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, StackBuffer, read_path_specified_file
from collections import deque


class BoardMavlink(multiprocessing.Process):
    def __init__(self):
        super().__init__()
        self.data_segments = multiprocessing.Queue(1)
        self.log_file_path = None
        self.flight_log = None
        self.param_current = dict()

    def wait_bin_ready(self):
        while True:
            time.sleep(0.1)
            if os.path.exists(self.log_file_path):
                logging.debug("Bin file ready.")
                break

    def bin_read_last_seg(self, tail_n: int = 20):
        try:
            with eventlet.Timeout(10, True):
                # wait share queue.
                while self.data_segments.empty():
                    time.sleep(0.1)
            # capture a data
            states, timestamp = self.data_segments.get()
            pd_array = pd.DataFrame(states)
            # only capture last tail_n vector
            pd_array = pd_array.tail(tail_n)

            logging.debug("Return a captured segment.")
            return pd_array
        except TimeoutError:
            logging.warning("Read segment timed out!")
            return None

    def delete_tmp_log(self):
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

    @classmethod
    def extract_log_path(cls, log_path, skip=True, keep_des=False, thread=None):
        """
        extract and convert bin file to csv
        :param log_path: file path.
        :param skip: skip process if the csv is existing.
        :param keep_des: extract the desired state from logs.
        :param thread: multiple threat, if none will not start thread
        :return:
        """
        # If px4, the log is ulg, if Ardupilot the log is bin
        end_flag = 'ulg' if (toolConfig.MODE == "PX4") else 'BIN'
        file_list = read_path_specified_file(log_path, end_flag)
        if not os.path.exists(f"{log_path}/csv"):
            os.makedirs(f"{log_path}/csv")

        # multiple
        if thread is not None:
            arrays = np.array_split(file_list, thread)
            threat_manage = []
            ray.init(include_dashboard=True, dashboard_host="127.0.0.1", dashboard_port=8088)

            for array in arrays:
                threat_manage.append(cls.extract_log_path_threat.remote(cls, log_path, array,
                                                                        keep_des, skip))
            ray.get(threat_manage)
            ray.shutdown()
        else:
            cls.extract_log_path_threat(cls, log_path, file_list, keep_des, skip)

    @abstractmethod
    def run(self):
        pass

    @classmethod
    @abstractmethod
    def extract_log_file(cls, log_file, keep_des=False):
        pass

    @classmethod
    @ray.remote
    def extract_log_path_threat(cls, log_path, file_list, keep_des, skip):
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
                csv_data = cls.extract_log_file(f'{log_path}/{file}', keep_des)
                csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
            except Exception as e:
                logging.warning(f"Error processing {file} : {e}")
                continue
        return True

    @abstractmethod
    def fill_and_process_pd_log(self, pd_array: pd.DataFrame):
        pass

    @abstractmethod
    def init_binary_log_file(self, device_i=None):
        pass

    @abstractmethod
    def delete_current_log(self, device=None):
        pass

    """
    Static Method
    """

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
            # filling
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

    def init_binary_log_file(self, device_i=None):
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{device_i}/logs/LASTLOG.TXT"
        # Read last index
        with open(log_index, 'r') as f:
            num = int(f.readline()) + 1
            if num == 501:
                num = 1
            # To string
        num = f'{num}'
        self.log_file_path = f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{device_i}/logs/{num.rjust(8, '0')}.BIN"
        logging.info(f"Current log file: {self.log_file_path}")

    def wait_bin_ready(self):
        while True:
            time.sleep(0.1)
            if os.path.exists(self.log_file_path):
                break

    def read_status_patch_bin(self, time_last):
        out_data = []
        accept_item = toolConfig.LOG_MAP.copy()
        accept_item_ex_param = accept_item.copy()
        accept_item_ex_param.remove("PARM")

        # Walk to the first message
        while True:
            msg = self.flight_log.recv_match(type=accept_item_ex_param)
            if msg is None:
                break
            elif msg.TimeUS > time_last:
                out_data.append(self.log_extract_apm(msg, True))

        # To DataFrame
        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array = self.fill_and_process_pd_log(pd_array)
        if pd_array.shape[0] < 20:
            return False
        return pd_array

    def delete_current_log(self, device=None):
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{device}/logs/LASTLOG.TXT"

        # Read last index
        with open(log_index, 'r') as f:
            num = int(f.readline())
        # To string
        num = f'{num}'
        log_file = f"{toolConfig.ARDUPILOT_LOG_PATH}/drone{device}/logs/{num.rjust(8, '0')}.BIN"
        # Remove file
        if os.path.exists(log_file):
            os.remove(log_file)
            # Fix last index number
            last_num = f"{int(num) - 1}"
            with open(log_index, 'w') as f:
                f.write(last_num)

    @classmethod
    def extract_log_file(cls, log_file, keep_des=False):
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
            out_data.append(cls.log_extract_apm(msg, keep_des))
        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array = cls.fill_and_process_pd_log(pd_array)
        return pd_array

    @classmethod
    def fill_and_process_pd_log(cls, pd_array: pd.DataFrame):
        """
        pre-process the data collected.
        :param pd_array:
        :return:
        """
        # Remain timestamp .1 and drop duplicate
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        df_array = cls._fill_and_process_public(pd_array)
        # Sort
        df_array = cls._order_sort(df_array)
        return df_array

    """
    Static Method
    """

    @staticmethod
    def get_time_index_bin(msg):
        """
        As different message have different time unit. It needs to convert to same second unit.
        :return:
        """
        return msg.TimeUS / 1000000

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

    """
    background thread
    """

    def run(self):
        time_last = 0
        # Wait for bin file created
        self.wait_bin_ready()
        _file = open(self.log_file_path, 'rb')
        accept_item = toolConfig.LOG_MAP.copy()

        while True:
            time.sleep(2)
            # Flush write buffer
            _file.flush()
            # Load current log file
            self.flight_log = mavutil.mavlink_connection(self.log_file_path)
            try:
                # Read flight status
                status_data = self.read_status_patch_bin(time_last)

                if status_data is False:
                    logging.debug("Reading status failure, try again.")
                    continue
                # send this message
                self.data_segments.put([status_data, round(time_last / 1000000, 1)])
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


class BoardMavlinkPX4(BoardMavlink):
    def __init__(self):
        super().__init__()

    def init_binary_log_file(self, device_i=None):
        if device_i is None:
            log_path = f"{toolConfig.PX4_LOG_PATH}/*.ulg"

            list_of_files = glob.glob(log_path)  # * means all if you need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            self.log_file_path = latest_file
            logging.info(f"Current log file: {latest_file}")
        else:
            now = time.localtime()
            now_time = time.strftime("%Y-%m-%d", now)
            log_path = f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/instance_{device_i}/log/{now_time}/*.ulg"

            list_of_files = glob.glob(log_path)  # * means all if you need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            self.log_file_path = latest_file
            logging.info(f"Current log file: {latest_file}")

    def init_current_param(self):
        # inti param value
        param = pd.Series(self.flight_log.initial_parameters)
        # select parameters
        self.param_current = param[toolConfig.PARAM]

    def read_status_patch_ulg(self, time_last):
        time_last = float(time_last)

        bias = pd.DataFrame(self.flight_log.get_dataset('estimator_sensor_bias').data)[["timestamp", "accel_bias[0]",
                                                                                        "accel_bias[1]",
                                                                                        "accel_bias[2]"]]
        bias.columns = ["TimeS", "BiasA", "BiasB", "BiasC"]
        bias = bias[bias["TimeS"] > time_last]

        att = pd.DataFrame(self.flight_log.get_dataset('vehicle_attitude_setpoint').data)[["timestamp",
                                                                                           "roll_body", "pitch_body",
                                                                                           "yaw_body"]]
        att.columns = ["TimeS", "Roll", "Pitch", "Yaw"]
        att = att[att["TimeS"] > time_last]

        rate = pd.DataFrame(self.flight_log.get_dataset('vehicle_rates_setpoint').data)[["timestamp",
                                                                                         "roll", "pitch", "yaw"]]
        rate.columns = ["TimeS", "RateRoll", "RatePitch", "RateYaw"]
        rate = rate[rate["TimeS"] > time_last]

        acc_gyr = pd.DataFrame(self.flight_log.get_dataset('sensor_combined').data)[["timestamp",
                                                                                     "gyro_rad[0]", "gyro_rad[1]",
                                                                                     "gyro_rad[2]",
                                                                                     "accelerometer_m_s2[0]",
                                                                                     "accelerometer_m_s2[1]",
                                                                                     "accelerometer_m_s2[2]"]]
        acc_gyr.columns = ["TimeS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
        acc_gyr = acc_gyr[acc_gyr["TimeS"] > time_last]

        mag = pd.DataFrame(self.flight_log.get_dataset('sensor_mag').data)[["timestamp", "x", "y", "z"]]
        mag.columns = ["TimeS", "MagX", "MagY", "MagZ"]
        mag = mag[mag["TimeS"] > time_last]

        vibe = pd.DataFrame(self.flight_log.get_dataset('sensor_accel').data)[["timestamp", "x", "y", "z"]]
        vibe.columns = ["TimeS", "VibeX", "VibeY", "VibeZ"]
        vibe = vibe[vibe["TimeS"] > time_last]

        # Merge values
        pd_array = pd.concat([att, rate, acc_gyr, mag, vibe, bias]).sort_values(by='TimeS')
        pd_array["TimeS"] = pd_array["TimeS"] / 1000000

        # Process
        df_array = self.fill_and_process_pd_log(pd_array)
        # print(df_array.shape)
        if pd_array.shape[0] < 20:
            return False

        return df_array

    def delete_current_log(self, device=None):
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

    @classmethod
    def fill_and_process_pd_log(cls, pd_array: pd.DataFrame):
        df_array = cls._fill_and_process_public(pd_array)
        return df_array

    @classmethod
    def extract_log_file(cls, log_file, keep_des=False):
        """
        extract log message form a bin file.
        :param log_file:
        :param keep_des:
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

        pd_array['TimeS'] = pd_array['TimeS'] / 1000000
        # Process
        df_array = cls.fill_and_process_pd_log(pd_array)
        # Add parameters
        param_values = np.tile(param.values, df_array.shape[0]).reshape(df_array.shape[0], -1)
        df_array[toolConfig.PARAM] = param_values

        # Sort
        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))

        return df_array

    """
    background thread
    """

    def run(self):
        logging.info("Start Bin monitor.")
        time_last = 0
        _file = open(self.log_file_path, 'rb')
        include = ['estimator_sensor_bias', 'vehicle_attitude_setpoint', 'vehicle_rates_setpoint', 'sensor_combined',
                   'sensor_mag', 'sensor_accel']
        while True:
            time.sleep(2)
            # Flush write buffer
            _file.flush()
            # Load current log file

            self.flight_log = ULog(self.log_file_path, message_name_filter_list=include)
            # inti param value
            self.init_current_param()
            try:
                # Read flight status
                status_data = self.read_status_patch_ulg(time_last)
                if status_data is False:
                    logging.debug("Reading status failure, try again.")
                    continue
                # send this message
                # print(status_data.shape, round(time_last / 1000000, 1))
                self.data_segments.put([status_data, round(time_last / 1000000, 1)])
            except Exception as e:
                logging.warning(f"{e}, then continue looping")

            # Drop old message
            msg = self.flight_log.last_timestamp
            time_last = msg
