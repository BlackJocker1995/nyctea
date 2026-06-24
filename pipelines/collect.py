"""Stage 1 — collect RL training data.

Flies ICSearcher's destabilizing configs (``params{EXE}.csv``, ``result !=
"pass"``) across N simulator instances in parallel and gathers DDPG
transitions into npz buffer shards under ``toolConfig.buffer_dir()``.

This is the data-gathering half of training: each worker builds a
``DDPGAgent(train=False)`` (no learning), runs ``train_from_incorrect`` which
flies a bad config, observes, repairs, and appends transitions to its buffer.
The learner (stage 2, device 0) reloads those shards and learns.

Replaces the legacy ``0.train_ddpg_thread.py`` + ``collect.py`` (Ray +
gnome-terminal spawning) with :class:`~nyctea.concurrency.MultiInstanceRunner`.
"""
import os

from loguru import logger

from nyctea.config import toolConfig
from nyctea.concurrency import MultiInstanceRunner, WorkerContext


def _collect_worker(ctx: WorkerContext) -> None:
    """One collector worker: build an agent and fly/repair to gather transitions."""
    # CUDA off — the collector workers don't learn.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from nyctea.rl.learning_agent import DDPGAgent
    agent = DDPGAgent(train=False, device=ctx.instance_id)
    try:
        agent.train_from_incorrect(toolConfig.params_csv())
    finally:
        agent.buffer.flush()
        agent.close()


def run(args) -> None:
    n = toolConfig.INSTANCES
    logger.info(f"collect: launching {n} worker(s) to gather transitions")
    runner = MultiInstanceRunner(
        n_instances=n, worker_fn=_collect_worker, debug=bool(args.debug))
    runner.run()
    logger.info("collect: done; buffer shards written to "
                f"{toolConfig.buffer_dir()}")
