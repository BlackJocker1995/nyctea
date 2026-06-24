"""Onboard flight-log reader for the RL state.

Two responsibilities:

1. **Online (the RL hot path):** :meth:`BoardMavlink.run` tails the live
   ``.BIN``/``.ulg`` log as a background process, parses new status rows since
   the last read, and pushes them onto ``data_segments``; :meth:`bin_read_last_seg`
   hands the env the most recent ``tail_n`` rows as the RL observation.
2. **Offline (batch log→CSV):** :meth:`extract_log_path` converts a directory of
   logs to CSVs for the analyze stage — now via stdlib
   ``concurrent.futures.ProcessPoolExecutor`` (no Ray).

Key changes from the legacy ``Cptool/boardMavlink.py``:

- **No Ray.** the ``@ray.remote`` ``extract_log_path_threat`` is replaced by a
  ``ProcessPoolExecutor``-mapped single-file converter.
- **No ``os.access`` busy-wait.**
- **Per-instance log paths** via ``toolConfig.ardu_instance_log_path`` /
  ``toolConfig.px4_instance_path`` (the single source of truth), so parallel
  instances read their own logs.
- **``logging`` → ``loguru``.**
"""
import glob
import multiprocessing
import os
import time
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from loguru import logger
from pymavlink import mavutil
from pymavlink.DFReader import DFMessage
from pyulog import ULog
from tqdm import tqdm

from nyctea.config import toolConfig
from nyctea.params import load_param, read_path_specified_file


class BoardMavlink(multiprocessing.Process):
    """Tails the live flight log and serves recent status segments to the env."""

    def __init__(self):
        super().__init__()
        self.data_segments = multiprocessing.Queue(1)
        self.log_file_path = None
        self.flight_log = None
        self.param_current = dict()

    def wait_bin_ready(self):
        """Block until the current log file appears on disk."""
        while True:
            time.sleep(0.1)
            if os.path.exists(self.log_file_path):
                logger.debug("Bin file ready.")
                break

    def bin_read_last_seg(self, tail_n: int = 20):
        """Return the most recent ``tail_n`` status rows (the RL observation).

        Uses an ``eventlet`` timeout to avoid blocking the env forever if no new
        segment arrives. Returns ``None`` on timeout.
        """
        try:
            import eventlet
            with eventlet.Timeout(10, True):
                while self.data_segments.empty():
                    time.sleep(0.1)
            states, timestamp = self.data_segments.get()
            if states is None:
                raise ValueError("Can not read segment from log file anymore.")
            pd_array = pd.DataFrame(states)
            pd_array = pd_array.tail(tail_n)
            logger.debug("Return a captured segment.")
            return pd_array
        except TimeoutError:
            logger.warning("Read segment timed out!")
            return None

    def delete_tmp_log(self):
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

    # --------------------------------------------------------- batch log→CSV
    @classmethod
    def extract_log_path(cls, log_path, skip=True, keep_des=False, workers=None):
        """Convert every log in ``log_path`` to a CSV under ``log_path/csv``.

        ``workers`` > 1 fans the files out across a stdlib
        ``ProcessPoolExecutor`` (replaces the legacy Ray remote). ``workers=None``
        runs serially.
        """
        end_flag = 'ulg' if (toolConfig.MODE == "PX4") else 'BIN'
        file_list = read_path_specified_file(log_path, end_flag)
        csv_dir = os.path.join(log_path, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        if workers and workers > 1:
            args = [(cls, log_path, f, keep_des, skip) for f in file_list]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for _ in tqdm(ex.map(_convert_one, args), total=len(args)):
                    pass
        else:
            cls.extract_log_path_single(log_path, file_list, keep_des, skip)

    @classmethod
    def extract_log_path_single(cls, log_path, file_list, keep_des, skip):
        """Serial log→CSV conversion."""
        for file in tqdm(file_list):
            name, _ = file.split('.')
            if skip and os.path.exists(f'{log_path}/csv/{name}.csv'):
                continue
            try:
                csv_data = cls.extract_log_file(f'{log_path}/{file}', keep_des)
                csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
            except Exception as e:
                logger.warning(f"Error processing {file} : {e}")
                continue
        return True

    # --------------------------------------------------------- abstract hooks
    @abstractmethod
    def run(self):
        pass

    @classmethod
    @abstractmethod
    def extract_log_file(cls, log_file, keep_des=False):
        pass

    @abstractmethod
    def fill_and_process_pd_log(self, pd_array: pd.DataFrame):
        pass

    @abstractmethod
    def init_binary_log_file(self, device_i=None):
        pass

    @abstractmethod
    def delete_current_log(self, device=None):
        pass

    # --------------------------------------------------------- static helpers
    @staticmethod
    def _fill_and_process_public(pd_array: pd.DataFrame):
        """Round TimeS to 0.1s, dedup, forward/backward-fill, group-merge same TimeS."""
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')
        df_array = pd.DataFrame(columns=pd_array.columns)
        for group, group_item in pd_array.groupby('TimeS'):
            group_item = group_item.fillna(method='ffill')
            group_item = group_item.fillna(method='bfill')
            df_array.loc[len(df_array.index)] = group_item.mean()
        df_array = df_array.fillna(method='ffill')
        df_array = df_array.dropna()
        return df_array

    @staticmethod
    def _order_sort(df_array):
        """Reorder columns: STATUS_ORDER first, then params in param-file order."""
        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))
        order_name.extend(param_name)
        df_array = df_array[order_name]
        return df_array


def _convert_one(args):
    """Top-level worker fn for ProcessPoolExecutor.map (must be picklable)."""
    cls, log_path, file, keep_des, skip = args
    name, _ = file.split('.')
    if skip and os.path.exists(f'{log_path}/csv/{name}.csv'):
        return
    try:
        csv_data = cls.extract_log_file(f'{log_path}/{file}', keep_des)
        csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
    except Exception as e:
        logger.warning(f"Error processing {file} : {e}")


class BoardMavlinkAPM(BoardMavlink):
    """ArduPilot ``.BIN`` log reader."""

    def __init__(self):
        super().__init__()

    def init_binary_log_file(self, device_i=None):
        log_dir = toolConfig.ardu_instance_log_path(device_i)
        log_index = os.path.join(log_dir, "LASTLOG.TXT")
        with open(log_index, 'r') as f:
            num = int(f.readline()) + 1
            if num == 501:
                num = 1
        num = f'{num}'
        self.log_file_path = os.path.join(log_dir, f"{num.rjust(8, '0')}.BIN")
        logger.info(f"Current log file: {self.log_file_path}")

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

        while True:
            msg = self.flight_log.recv_match(type=accept_item_ex_param)
            if msg is None:
                break
            elif msg.TimeUS > time_last:
                out_data.append(self.log_extract_apm(msg, True))

        pd_array = pd.DataFrame(out_data)
        pd_array = self.fill_and_process_pd_log(pd_array)
        if pd_array.shape[0] < 20:
            return False
        return pd_array

    def delete_current_log(self, device=None):
        log_dir = toolConfig.ardu_instance_log_path(device)
        log_index = os.path.join(log_dir, "LASTLOG.TXT")
        with open(log_index, 'r') as f:
            num = int(f.readline())
        num = f'{num}'
        log_file = os.path.join(log_dir, f"{num.rjust(8, '0')}.BIN")
        if os.path.exists(log_file):
            os.remove(log_file)
            last_num = f"{int(num) - 1}"
            with open(log_index, 'w') as f:
                f.write(last_num)

    @classmethod
    def extract_log_file(cls, log_file, keep_des=False):
        """Extract a DataFrame from a single ArduPilot ``.BIN`` log."""
        accept_item = toolConfig.LOG_MAP
        logs = mavutil.mavlink_connection(log_file)
        out_data = []
        accept_param = load_param().columns.to_list()

        while True:
            msg = logs.recv_match(type=accept_item)
            if msg is None:
                break
            # Skip non-index-0 sensors and unwanted params.
            if (hasattr(msg, "I") and msg.I != 0) or \
                    (hasattr(msg, "IMU") and msg.IMU != 0) or \
                    (msg.get_type() == 'PARM' and msg.Name not in accept_param):
                continue
            out_data.append(cls.log_extract_apm(msg, keep_des))
        pd_array = pd.DataFrame(out_data)
        pd_array = cls.fill_and_process_pd_log(pd_array)
        return pd_array

    @classmethod
    def fill_and_process_pd_log(cls, pd_array: pd.DataFrame):
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        df_array = cls._fill_and_process_public(pd_array)
        df_array = cls._order_sort(df_array)
        return df_array

    # ----------------------------------------------------- static extractors
    @staticmethod
    def get_time_index_bin(msg):
        return msg.TimeUS / 1000000

    @staticmethod
    def log_extract_apm(msg: DFMessage, keep_des=False):
        """Parse a single ArduPilot log message into a status-row dict."""
        out = None
        if msg.get_type() == 'ATT':
            if not keep_des:
                out = {'TimeS': msg.TimeUS / 1000000, 'Roll': msg.Roll,
                       'Pitch': msg.Pitch, 'Yaw': msg.Yaw}
            else:
                out = {'TimeS': msg.TimeUS / 1000000, "DesRoll": msg.DesRoll,
                       'Roll': msg.Roll, 'DesPitch': msg.DesPitch, 'Pitch': msg.Pitch,
                       'DesYaw': msg.DesYaw, 'Yaw': msg.Yaw}
        elif msg.get_type() == 'RATE':
            if not keep_des:
                out = {'TimeS': msg.TimeUS / 1000000, 'RateRoll': msg.R,
                       'RatePitch': msg.P, 'RateYaw': msg.Y}
            else:
                out = {'TimeS': msg.TimeUS / 1000000, 'DesRateRoll': msg.RDes,
                       'RateRoll': msg.R, 'DesRatePitch': msg.PDes, 'RatePitch': msg.P,
                       'DesRateYaw': msg.YDes, 'RateYaw': msg.Y}
        elif msg.get_type() == 'IMU':
            out = {'TimeS': msg.TimeUS / 1000000, 'AccX': msg.AccX, 'AccY': msg.AccY,
                   'AccZ': msg.AccZ, 'GyrX': msg.GyrX, 'GyrY': msg.GyrY, 'GyrZ': msg.GyrZ}
        elif msg.get_type() == 'VIBE':
            out = {'TimeS': msg.TimeUS / 1000000, 'VibeX': msg.VibeX,
                   'VibeY': msg.VibeY, 'VibeZ': msg.VibeZ}
        elif msg.get_type() == 'MAG':
            out = {'TimeS': msg.TimeUS / 1000000, 'MagX': msg.MagX,
                   'MagY': msg.MagY, 'MagZ': msg.MagZ}
        elif msg.get_type() == 'PARM':
            out = {'TimeS': msg.TimeUS / 1000000, msg.Name: msg.Value}
        elif msg.get_type() == 'GPS':
            out = {'TimeS': msg.TimeUS / 1000000, 'Lat': msg.Lat,
                   'Lng': msg.Lng, 'Alt': msg.Alt}
        return out

    # ----------------------------------------------------- background thread
    def run(self):
        time_last = 0
        self.wait_bin_ready()
        _file = open(self.log_file_path, 'rb')
        accept_item = toolConfig.LOG_MAP.copy()

        while True:
            time.sleep(2)
            _file.flush()
            self.flight_log = mavutil.mavlink_connection(self.log_file_path)
            try:
                status_data = self.read_status_patch_bin(time_last)
                if status_data is False:
                    logger.debug("Reading status failure, try again.")
                    continue
                self.data_segments.put([status_data, round(time_last / 1000000, 1)])
                logger.debug("Read a segment successfully send to main thread.")
            except Exception as e:
                logger.warning(f"{e}, then continue looping")

            # Drop old messages and advance the read cursor.
            while True:
                msg = self.flight_log.recv_match(type=accept_item)
                if msg is None:
                    break
                else:
                    time_last = msg.TimeUS


class BoardMavlinkPX4(BoardMavlink):
    """PX4 ``.ulg`` log reader."""

    def __init__(self):
        super().__init__()

    def init_binary_log_file(self, device_i=None):
        if device_i is None:
            log_path = f"{toolConfig.PX4_LOG_PATH}/*.ulg"
        else:
            now_time = time.strftime("%Y-%m-%d", time.localtime())
            instance_dir = toolConfig.px4_instance_path(device_i)
            log_path = f"{instance_dir}/log/{now_time}/*.ulg"
        list_of_files = glob.glob(log_path)
        latest_file = max(list_of_files, key=os.path.getctime)
        self.log_file_path = latest_file
        logger.info(f"Current log file: {latest_file}")

    def init_current_param(self):
        param = pd.Series(self.flight_log.initial_parameters)
        self.param_current = param[toolConfig.PARAM]

    def read_status_patch_ulg(self, time_last):
        time_last = float(time_last)

        acc_bias = pd.DataFrame(self.flight_log.get_dataset('estimator_sensor_bias').data)[
            ["timestamp", "accel_bias[0]", "accel_bias[1]", "accel_bias[2]"]]
        att_bias = pd.DataFrame(self.flight_log.get_dataset('estimator_status').data)[
            ["timestamp", "output_tracking_error[0]"]]
        acc_bias.columns = ["TimeS", "BiasA", "BiasB", "BiasC"]
        att_bias.columns = ["TimeS", "BiasD"]
        acc_bias = acc_bias[acc_bias["TimeS"] > time_last]
        att_bias = att_bias[att_bias["TimeS"] > time_last]

        att = pd.DataFrame(self.flight_log.get_dataset('vehicle_attitude_setpoint').data)[
            ["timestamp", "roll_body", "pitch_body", "yaw_body"]]
        att.columns = ["TimeS", "Roll", "Pitch", "Yaw"]
        att = att[att["TimeS"] > time_last]

        rate = pd.DataFrame(self.flight_log.get_dataset('vehicle_rates_setpoint').data)[
            ["timestamp", "roll", "pitch", "yaw"]]
        rate.columns = ["TimeS", "RateRoll", "RatePitch", "RateYaw"]
        rate = rate[rate["TimeS"] > time_last]

        acc_gyr = pd.DataFrame(self.flight_log.get_dataset('sensor_combined').data)[
            ["timestamp", "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]",
             "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]]
        acc_gyr.columns = ["TimeS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
        acc_gyr = acc_gyr[acc_gyr["TimeS"] > time_last]

        pd_array = pd.concat([att, rate, acc_gyr, acc_bias, att_bias]).sort_values(by='TimeS')
        pd_array["TimeS"] = pd_array["TimeS"] / 1000000
        df_array = self.fill_and_process_pd_log(pd_array)
        if pd_array.shape[0] < 20:
            return False
        return df_array

    def delete_current_log(self, device=None):
        if self.log_file_path is not None and os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

    @classmethod
    def fill_and_process_pd_log(cls, pd_array: pd.DataFrame):
        return cls._fill_and_process_public(pd_array)

    @classmethod
    def extract_log_file(cls, log_file, keep_des=False):
        """Extract a DataFrame from a single PX4 ``.ulg`` log."""
        include = ['estimator_sensor_bias', 'vehicle_attitude_setpoint',
                   'vehicle_rates_setpoint', 'sensor_combined', "estimator_status"]
        ulog = ULog(log_file, message_name_filter_list=include)

        if keep_des:
            acc_bias = pd.DataFrame(ulog.get_dataset('estimator_sensor_bias').data)[
                ["timestamp", "accel_bias[0]", "accel_bias[1]", "accel_bias[2]"]]
            att_bias = pd.DataFrame(ulog.get_dataset('estimator_status').data)[
                ["timestamp", "output_tracking_error[0]"]]
            acc_bias.columns = ["TimeS", "BiasA", "BiasB", "BiasC"]
            att_bias.columns = ["TimeS", "BiasD"]

        att = pd.DataFrame(ulog.get_dataset('vehicle_attitude_setpoint').data)[
            ["timestamp", "roll_body", "pitch_body", "yaw_body"]]
        rate = pd.DataFrame(ulog.get_dataset('vehicle_rates_setpoint').data)[
            ["timestamp", "roll", "pitch", "yaw"]]
        acc_gyr = pd.DataFrame(ulog.get_dataset('sensor_combined').data)[
            ["timestamp", "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]",
             "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]]
        # Params
        param = pd.Series(ulog.initial_parameters)
        param = param[toolConfig.PARAM]
        for t, name, value in ulog.changed_parameters:
            if name in toolConfig.PARAM:
                param[name] = round(value, 5)

        att.columns = ["TimeS", "Roll", "Pitch", "Yaw"]
        rate.columns = ["TimeS", "RateRoll", "RatePitch", "RateYaw"]
        acc_gyr.columns = ["TimeS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
        if keep_des:
            pd_array = pd.concat([att, rate, acc_gyr, acc_bias, att_bias]).sort_values(by='TimeS')
        else:
            pd_array = pd.concat([att, rate, acc_gyr]).sort_values(by='TimeS')
        pd_array['TimeS'] = pd_array['TimeS'] / 1000000
        df_array = cls.fill_and_process_pd_log(pd_array)
        param_values = np.tile(param.values, df_array.shape[0]).reshape(df_array.shape[0], -1)
        df_array[toolConfig.PARAM] = param_values

        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))
        return df_array

    # ----------------------------------------------------- background thread
    def run(self):
        logger.info("Start Bin monitor.")
        time_last = 0
        _file = open(self.log_file_path, 'rb')
        include = ['estimator_sensor_bias', 'vehicle_attitude_setpoint',
                   'vehicle_rates_setpoint', 'sensor_combined', "estimator_status"]
        failure_num = 0
        while True:
            time.sleep(2)
            if failure_num > 5:
                self.data_segments.put([None, round(time_last / 1000000, 1)])
            _file.flush()
            self.flight_log = ULog(self.log_file_path, message_name_filter_list=include)
            self.init_current_param()
            try:
                status_data = self.read_status_patch_ulg(time_last)
                if status_data is False:
                    logger.debug("Reading status failure, try again.")
                    failure_num += 1
                    continue
                else:
                    failure_num = 0
                self.data_segments.put([status_data, round(time_last / 1000000, 1)])
            except Exception as e:
                logger.warning(f"{e}, then continue looping")
            msg = self.flight_log.last_timestamp
            time_last = msg
