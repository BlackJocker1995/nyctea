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
    def __init__(self, arg):
        super().__init__()
        self.msg_queue = multiprocessing.Queue()
        # port
        # if isinstance(arg, int):
        self.master = mavutil.mavlink_connection('udp:0.0.0.0:{}'.format(arg))
        self.master.wait_heartbeat(timeout=30)
        logging.info("Heartbeat from system (system %u component %u) from %u" % (
            self.master.target_system, self.master.target_component, arg))

        # if isinstance(arg, pymavlink.mavutil.mavudp):
        #     self.master = arg
        # else:
        #     raise TypeError(f"Not supported type: {type(arg)}")

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
        # Flag
        start_check = False
        current_mission = 0
        pre_alt = 0

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
                logging.debug(f"Status message: {status_message}")
                # print(status_message)
                if status_message.severity == 6:
                    if "Disarming" in line or "landed" in line or "Landing" in line or "Land" in line:
                        # if successful landed, break the loop and return true
                        logging.info("Successful break the loop.")
                        break
                    if "preflight disarming" in line:
                        result = 'PreArm Failed'
                        break
                    # if "SIM Hit ground" in line:
                    #     result = 'crash'
                    #     break
                elif status_message.severity == 2 or status_message.severity == 0:
                    # Appear error, break loop and return false
                    if "SIM Hit ground" in line \
                            or "Crash" in line \
                            or "Failsafe enabled: no global position" in line \
                            or "failure detected" in line:
                        result = 'crash'
                        break
                    elif "Potential Thrust Loss" in line:
                        result = 'Thrust Loss'
                        break
                    elif "PreArm" in line: # or "speed has been constrained by max speed" in line:
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
                    if int(position_msg.seq) == 1 and toolConfig.MODE == "PX4":
                        start_check = True

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
                # time
                time_step = position.timeS - pre_location.timeS
                # altitude change
                alt_change = abs(pre_alt - alt)
                # Update position
                pre_location.x = position_lat
                pre_location.y = position_lon
                pre_alt = alt

                if start_check:
                    velocity = moving_dis / time_step
                    # print(f"Velocity {velocity}.")
                    # Is small move? velocity smaller than 1 and altitude change  smaller than 0.1
                    if velocity < 1 and alt_change < 0.1:
                        small_move_num += 1
                        logging.debug(f"Small moving {small_move_num}, num++, num now - {small_move_num}.")
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
                        if deviation_dis % 5 == 0:
                            logging.debug(f"Deviation {round(deviation_dis, 4)}, "
                                          f"num++, num now - {deviation_num}.")
                        deviation_num += 1
                    else:
                        deviation_num = 0

                    # # Threshold; deviation judgement
                    if deviation_num > 15:
                        result = 'deviation'
                        break

                    # Timeout
                    if small_move_num > 10:
                        result = 'timeout'
                        break
                # ============================ #

            # Timeout Check if stack at one point
            mid_point_time = time.time()
            if (mid_point_time - start_time) > mission_time_out_th:
                result = 'timeout'
                break

        logging.info(f"Monitor result: {result}")
        self.msg_queue.put([result, time.time()])
