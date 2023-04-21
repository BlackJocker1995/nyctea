import argparse
import csv
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

import Cptool
from Cptool.config import toolConfig
from sklearn.utils import shuffle

from Cptool.simManager import SimManager, FixSimManager
from Rl.learning_agent import DDPGAgent

if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Device
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--device', dest='device', type=str, help='Name of the candidate')
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = 0
    # Read incorrect configuration
    configurations = pd.read_csv(f"validation/{toolConfig.MODE}/params{toolConfig.EXE}.csv")
    incorrect_configuration = configurations[configurations["result"] != "pass"]

    # Stochastic order
    incorrect_configuration = shuffle(incorrect_configuration)
    for index, row in incorrect_configuration.iterrows():
        ddpg_agent = DDPGAgent(device=device)
        ddpg_agent.check_point()
        time.sleep(1)
        print(f'======================={index} / {incorrect_configuration.shape[0]} ==========================')
        config = row.drop(["score", "result"]).astype(float)
        if not os.path.exists(f'validation/{toolConfig.MODE}/{toolConfig.EXE}'):
            os.mkdir(f'validation/{toolConfig.MODE}/{toolConfig.EXE}')

        if os.path.exists(f'validation/{toolConfig.MODE}/params_repair{toolConfig.EXE}.csv'):
            while not os.access(f"validation/{toolConfig.MODE}/params_repair{toolConfig.EXE}.csv", os.R_OK):
                continue
            data = pd.read_csv(f'validation/{toolConfig.MODE}/params_repair{toolConfig.EXE}.csv')
            exit_data = data.drop(['repair_result', 'result'], axis=1, inplace=False)
            # If the value has been validate
            if ((exit_data - config).sum(axis=1).abs() < 0.00001).sum() > 0:
                continue

        ddpg_agent.env.current_incorrect_configuration = config.to_dict()
        ddpg_agent.env.reset(delay=False)

        result, action_num, deviations_his = ddpg_agent.online_bin_monitor_rl()

        ddpg_agent.env.close_env()
        ddpg_agent.env.manager.board_mavlink.delete_current_log(device)

        # if the result have no instability, skip.
        if not os.path.exists(f'validation/{toolConfig.MODE}/params_repair{toolConfig.EXE}.csv'):
            data = pd.DataFrame(columns=(toolConfig.PARAM_PART + ['result', 'repair_result', 'repair_time', 'hash']))
            data.to_csv(f'validation/{toolConfig.MODE}/params_repair{toolConfig.EXE}.csv', index=False)

        while not os.access(f"validation/{toolConfig.MODE}/params_repair{toolConfig.EXE}.csv", os.W_OK):
            continue
        # Add instability result
        tmp_row = list(config.to_dict().values())
        hash_code = hash(str(tmp_row))
        tmp_row.append(row["result"])
        tmp_row.append(result)
        tmp_row.append(action_num)
        tmp_row.append(hash_code)
        with open(f'validation/{toolConfig.MODE}/{toolConfig.EXE}/{hash_code}.pkl', "wb") as f:
            pickle.dump(deviations_his, f)

        # Write Row
        with open(f"validation/{toolConfig.MODE}/params_repair{toolConfig.EXE}.csv", 'a+') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(tmp_row)
            logging.debug("Write row to params.csv.")


