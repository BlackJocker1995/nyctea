"""Stage 4 — validate repaired configs by re-flying them.

Reads ``params_repair{EXE}.csv`` (the repair outcomes from stage 3), re-flies
each repaired config in SITL, and records whether the flight stays stable
(``pass``) or re-diverges. This is the ground-truth check that the agent's
online repair generalised, not just fit one flight.

Output: ``params_validate{EXE}.csv`` with a ``validate_result`` column.
"""
import os

import pandas as pd
from loguru import logger

from nyctea.config import toolConfig
from nyctea.concurrency import LockedCsv


def run(args) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from nyctea.rl.learning_agent import DDPGAgent

    out_dir = f"validation/{toolConfig.MODE}"
    EXE = toolConfig.EXE
    repair_path = f"{out_dir}/params_repair{EXE}.csv"
    if not os.path.exists(repair_path):
        logger.error(f"repair output not found: {repair_path}; run nyctea-repair first")
        return

    repaired = pd.read_csv(repair_path)
    columns_name = toolConfig.PARAM
    validate_csv = LockedCsv(
        f"{out_dir}/params_validate{EXE}.csv",
        header=list(columns_name) + ["repair_result", "validate_result", "hash"],
    )
    validate_csv.ensure_created()

    agent = DDPGAgent(train=False, device=0)
    agent.check_point()

    for index, row in repaired.iterrows():
        config = row.drop(
            ["result", "repair_result", "repair_time", "hash"],
            errors="ignore").astype(float)
        logger.info(f"======================= {index} / {repaired.shape[0]} "
                    f"========================")
        # Fly the repaired config without any agent action; just observe.
        agent.env.current_incorrect_configuration = config.to_dict()
        agent.env.reset(delay=False)
        result, _, _ = agent.online_bin_monitor_rl()
        agent.env.close_env()
        agent.env.manager.board_mavlink.delete_current_log(0)

        validate_csv.append_row(
            list(config.to_dict().values())
            + [row.get("repair_result"), result, row.get("hash")])

    agent.close()
    logger.info(f"validate: done; results in {out_dir}/params_validate{EXE}.csv")
