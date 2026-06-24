"""Stage 2 — train the DDPG repair agent.

Device 0 reloads every buffer shard into memory and runs the DDPG update loop;
optional sibling workers (``--instances``) keep collecting fresh transitions in
parallel, which the learner picks up on periodic ``buffer.reload()`` calls.

Replaces the legacy ``0.train_ddpg[_thread].py``. The device-0 learner path is
preserved; only the orchestration changed (stdlib ``multiprocessing`` instead
of Ray / gnome-terminal).
"""
import os

from loguru import logger

from nyctea.config import toolConfig
from nyctea.concurrency import MultiInstanceRunner, WorkerContext


def _train_worker(ctx: WorkerContext) -> None:
    """device 0 learns; others collect transitions to feed it."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from nyctea.rl.learning_agent import DDPGAgent
    agent = DDPGAgent(train=True, device=ctx.instance_id)
    try:
        agent.train_from_incorrect(toolConfig.params_csv())
    finally:
        if ctx.instance_id == 0:
            agent.buffer.flush()
        agent.close()


def run(args) -> None:
    n = toolConfig.INSTANCES
    logger.info(f"train: launching {n} worker(s) (device 0 learns)")
    runner = MultiInstanceRunner(
        n_instances=n, worker_fn=_train_worker, debug=bool(args.debug))
    runner.run()
    logger.info(f"train: done; checkpoints under {toolConfig.model_dir()}")
