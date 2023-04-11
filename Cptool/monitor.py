import logging
import math
import multiprocessing
import os
import time
from abc import abstractmethod, ABC

from pymavlink import mavwp, mavutil

from Cptool.config import toolConfig
from Cptool.mavtool import Location


class MonitorFlight(multiprocessing.Process):
    def __init__(self, port):
        super().__init__()
        self.msg_queue = multiprocessing.Queue()
        self.master = mavutil.mavlink_connection('udp:0.0.0.0:{}'.format(port))
        self.master.wait_heartbeat(timeout=30)

    def get_msg(self, msg_type, block=False):
        """
        receive the mavlink message
        :param msg_type:
        :param block:
        :return:
        """
        msg = self.master.recv_match(type=msg_type, blocking=block)
        return msg

    def gcs_msg_request(self):
        # If it requires manually send the gsc packets.
        """
               PX4 needs manual send the heartbeat for GCS
               :return:
               """
        self.master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                       mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    @abstractmethod
    def init_current_param(self):
        pass

    def run(self) -> None:
        """
        monitor error during the flight
        :return:
        """
        # A New receiver
        _monitor = self.master
        # _monitor.connect()

        logging.info('Start error monitor.')
        # Setting
        mission_time_out_th = 180
        result = 'pass'
        # Waypoint
        loader = mavwp.MAVWPLoader()
        if toolConfig.MODE == "PX4":
            loader.load('Cptool/fitCollection_px4.txt')
        else:
            loader.load('Cptool/fitCollection.txt')
        #
        lpoint1 = Location(loader.wpoints[0])
        lpoint2 = Location(loader.wpoints[1])
        pre_location = Location(loader.wpoints[0])
        # logger
        small_move_num = 0
        deviation_num = 0
        low_lat_num = 0
        # Flag
        start_check = False
        current_mission = 0
        pre_alt = 0
        last_time = 0

        start_time = time.time()
        while True:
            # time.sleep(0.01)
            if toolConfig.MODE == "PX4":
                self.gcs_msg_request()
            status_message = self.get_msg(["STATUSTEXT"])
            position_msg = self.get_msg(["GLOBAL_POSITION_INT", "MISSION_CURRENT"])

            # System status message
            if status_message is not None and status_message.get_type() == "STATUSTEXT":
                line = status_message.text
                # print(status_message)
                if status_message.severity == 6:
                    if "Disarming" in line or "landed" in line or "Landing" in line or "Land" in line:
                        # if successful landed, break the loop and return true
                        logging.info("Successful break the loop.")
                        break
                    if "preflight disarming" in line:
                        result = 'PreArm Failed'
                        break
                elif status_message.severity == 2 or status_message.severity == 0:
                    # Appear error, break loop and return false
                    if "SIM Hit ground at" in line \
                            or "Crash" in line \
                            or "Failsafe enabled: no global position" in line \
                            or "failure detected" in line:
                        result = 'crash'
                        break
                    elif "Potential Thrust Loss" in line:
                        result = 'Thrust Loss'
                        break
                    elif "PreArm" in line or "speed has been constrained by max speed" in line:
                        result = 'PreArm Failed'
                        break

            if position_msg is not None and position_msg.get_type() == "MISSION_CURRENT":
                # print(position_msg)
                if int(position_msg.seq) > current_mission and int(position_msg.seq) != 6:
                    logging.debug(f"Mission change {current_mission} -> {position_msg.seq}")
                    lpoint1 = Location(loader.wpoints[current_mission])
                    lpoint2 = Location(loader.wpoints[position_msg.seq])

                    # Start Check
                    if int(position_msg.seq) == 2:
                        start_check = True
                    # # can start check
                    # if int(position_msg.seq) == 2:
                    #     self.mav_monitor.recv_msg_queue.put(["start2point", time.time()])
                    current_mission = int(position_msg.seq)
                    if toolConfig.MODE == "PX4" and int(position_msg.seq) == 5:
                        start_check = False
            elif position_msg is not None and position_msg.get_type() == "GLOBAL_POSITION_INT":
                # print(position_msg)
                # Check deviation
                position_lat = position_msg.lat * 1.0e-7
                position_lon = position_msg.lon * 1.0e-7
                alt = position_msg.relative_alt / 1000
                time_usec = position_msg.time_boot_ms * 1e-6
                position = Location(position_lat, position_lon, time_usec)

                # Calculate distance
                moving_dis = Location.distance(pre_location, position)
                time_step = position.timeS - pre_location.timeS
                alt_change = abs(pre_alt - alt)
                # Update position
                pre_location.x = position_lat
                pre_location.y = position_lon
                pre_alt = alt

                if start_check:
                    if alt < 1:
                        low_lat_num += 1
                    else:
                        small_move_num = 0

                    velocity = moving_dis / time_step
                    # logging.debug(f"Velocity {velocity}.")
                    # Is small move?
                    # logging.debug(f"alt_change {alt_change}.")
                    if velocity < 1 and alt_change < 0.1 and small_move_num != 0:
                        logging.debug(f"Small moving {small_move_num}, num++, num now - {small_move_num}.")
                        small_move_num += 1
                    else:
                        small_move_num = 0

                    # Point2line distance
                    a = Location.distance(position, lpoint1)
                    b = Location.distance(position, lpoint2)
                    c = Location.distance(lpoint1, lpoint2)

                    if c != 0:
                        p = (a + b + c) / 2
                        deviation_dis = 2 * math.sqrt(p * (p - a) * (p - b) * (p - c) + 0.01) / c
                    else:
                        deviation_dis = 0
                    # Is deviation ?
                    # logging.debug(f"Point2line distance {deviation_dis}.")
                    if deviation_dis > 10:
                        logging.debug(f"Deviation {deviation_dis}, num++, num now - {deviation_num}.")
                        deviation_num += 1
                    else:
                        deviation_num = 0

                    # deviation
                    if deviation_num > 15:
                        result = 'deviation'
                        break
                    # Threshold; Judgement
                    # Timeout
                    if small_move_num > 10:
                        result = 'timeout'
                        break
                # ============================ #

            # Timeout Check if stack at one point
            mid_point_time = time.time()
            last_time = mid_point_time
            if (mid_point_time - start_time) > mission_time_out_th:
                result = 'timeout'
                break

        logging.info(f"Monitor result: {result}")
        self.msg_queue.put([result, time.time()])
