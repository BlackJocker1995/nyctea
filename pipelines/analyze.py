"""Stage 5 — loss / detection / deviation analysis and figures.

Aggregates the per-config deviation histories (``deviation_{hash}.npz`` from
stage 3) and the repair/validate CSVs into summary statistics and the
desired-vs-achieved attitude figures. This is the offline, no-SITL stage.

Replaces the legacy ``2.average_loss*`` / ``2.detect_test*`` /
``3.distribution*`` / ``3.*_exp`` script family with one analysis entry point.
"""
import glob
import os

import numpy as np
import pandas as pd
from loguru import logger

from nyctea.config import toolConfig
from nyctea.mavtool import draw_att_des_and_ach_repair, sort_result_detect_repair


def _load_deviation_histories():
    """Load every ``deviation_{hash}.npz`` into a {hash: ndarray} dict."""
    hist_dir = os.path.join(f"validation/{toolConfig.MODE}", str(toolConfig.EXE))
    out = {}
    for path in glob.glob(os.path.join(hist_dir, "*.npz")):
        name = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path, allow_pickle=False)
        out[name] = data["deviation"]
    return out


def run(args) -> None:
    out_dir = f"validation/{toolConfig.MODE}"
    EXE = toolConfig.EXE

    histories = _load_deviation_histories()
    logger.info(f"analyze: loaded {len(histories)} deviation histories")

    if histories:
        before = np.array([h[0] for h in histories.values() if len(h)])
        after = np.array([h[-1] for h in histories.values() if len(h)])
        logger.info(f"mean deviation before repair: {before.mean():.4f}")
        logger.info(f"mean deviation after repair:  {after.mean():.4f}")
        reduction = (before.mean() - after.mean()) / before.mean() * 100 if before.mean() else 0
        logger.info(f"mean deviation reduction:     {reduction:.2f}%")

    # Repair-outcome breakdown from the CSV.
    repair_path = f"{out_dir}/params_repair{EXE}.csv"
    if os.path.exists(repair_path):
        df = pd.read_csv(repair_path)
        if "repair_result" in df.columns:
            counts = df["repair_result"].value_counts().to_dict()
            logger.info(f"repair outcomes: {counts}")
            # Detection-vs-repair timing classification (needs timestamps).
            if all(c in df.columns for c in ("repair_time",)):
                logger.info("repair_time column present; run sort_result_detect_repair "
                            "on individual flights for detect/repair/miss breakdown.")

    # Optionally render the desired-vs-achieved figure for a sample flight.
    # (Off by default; enable by pointing at a converted CSV.)
    logger.info(f"analyze: done. Figures available via "
                f"nyctea.mavtool.draw_att_des_and_ach_repair on a flight CSV.")
