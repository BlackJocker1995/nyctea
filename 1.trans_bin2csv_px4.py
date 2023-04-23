import os

import numpy as np
import pandas as pd

from Cptool.boardMavlink import BoardMavlinkAPM, BoardMavlinkPX4
from Cptool.config import toolConfig

if __name__ == '__main__':
    # Please change the mode in Config.py as the multiple thread mode init the wrong environment configuration.
    toolConfig.select_mode("PX4")
    BoardMavlinkPX4.extract_log_path(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/ulg_unstable", skip=False,
                                     keep_des=True, thread=6)
