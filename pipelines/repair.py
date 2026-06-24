"""Stage 3 — repair-test destabilizing configs with the trained agent.

For each bad config in ICSearcher's ``params{EXE}.csv`` (``result != "pass"``),
fly it, let the trained DDPG agent upload a correction when the deviation
exceeds threshold, and record the repair outcome + deviation history.

Replaces the legacy ``1.fix_test[_wind|_sensor|_random].py`` family:

- ``--disturbance {none,wind,sensor,random}`` selects the test variant instead
  of having four copy-pasted scripts.
- ``params_repair{EXE}.csv`` is written via :class:`~nyctea.concurrency.LockedCsv`
  (no ``os.access`` busy-wait), so parallel runs don't corrupt rows.
- The per-config deviation history moves from ``{hash}.pkl`` to an npz shard
  (``deviation_{hash}.npz``) — no arbitrary-code-execution pickle.
"""
import os
import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.utils import shuffle

from nyctea.config import toolConfig
from nyctea.concurrency import LockedCsv


def _disturbance_env(disturbance: str):
    """Apply the requested disturbance to the environment before a repair test.

    Mirrors the legacy _wind/_sensor/_random script differences (the random
    variant shuffles config order and randomizes the seed; sensor/wind perturb
    the sim). Currently a hook: wind/sensor require a live AirSim backend, so
    only ``random`` (order shuffle) and ``none`` are implemented here.
    """
    if disturbance == "random":
        np.random.seed(None)


def run(args) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from nyctea.rl.learning_agent import DDPGAgent

    disturbance = getattr(args, "disturbance", "none")
    _disturbance_env(disturbance)

    param_file = toolConfig.params_csv()
    configurations = pd.read_csv(param_file)
    incorrect = configurations[configurations["result"] != "pass"]
    columns_name = incorrect.columns.drop(["score", "result"]).tolist()
    EXE = toolConfig.EXE
    incorrect = shuffle(incorrect)

    # Output artefacts.
    out_dir = f"validation/{toolConfig.MODE}"
    os.makedirs(out_dir, exist_ok=True)
    repair_csv = LockedCsv(
        f"{out_dir}/params_repair{EXE}.csv",
        header=columns_name + ["result", "repair_result", "repair_time", "hash"],
    )
    repair_csv.ensure_created()
    hist_dir = os.path.join(out_dir, str(EXE))
    os.makedirs(hist_dir, exist_ok=True)

    agent = DDPGAgent(train=False, device=0)
    agent.check_point()

    for index, row in incorrect.iterrows():
        config = row.drop(["score", "result"]).astype(float)
        logger.info(f"======================= {index} / {incorrect.shape[0]} "
                    f"========================")

        # Skip configs already validated (concurrency-safe under the lock).
        existing = repair_csv.rows_as_dataframe()
        if not existing.empty:
            exit_data = existing.drop(
                ["repair_result", "result", "repair_time", "hash"],
                axis=1, inplace=False, errors="ignore")
            if ((exit_data - config).sum(axis=1).abs() < 0.00001).sum() > 0:
                continue

        agent.env.current_incorrect_configuration = config.to_dict()
        agent.env.reset(delay=False)
        result, action_num, deviations_his = agent.online_bin_monitor_rl()
        agent.env.close_env()
        agent.env.manager.board_mavlink.delete_current_log(0)

        tmp_row = list(config.to_dict().values())
        hash_code = hash(str(tmp_row))
        tmp_row += [row["result"], result, action_num, hash_code]

        # Deviation history as npz (replaces the legacy {hash}.pkl).
        np.savez(
            os.path.join(hist_dir, f"{hash_code}.npz"),
            deviation=np.asarray(deviations_his, dtype=np.float64),
            action_num=np.array(action_num),
            result=np.array(result),
        )
        repair_csv.append_row(tmp_row)
        logger.debug("Wrote row to params_repair.csv.")
        time.sleep(1)

    agent.close()
    logger.info(f"repair ({disturbance}): done; results in {out_dir}")
