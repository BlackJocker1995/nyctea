"""RL checkpoint read/write, guarded by ``fcntl.flock``.

The legacy ``learning_agent.py`` guarded ``model/*.pth`` writes with flock but
guarded reads with racy ``os.access`` busy-waits. This module centralises both
under proper flock so concurrent workers can't read a half-written checkpoint.

Checkpoint layout (under ``toolConfig.model_dir()``):

- ``actor.pth`` / ``critic.pth`` — full module objects (legacy used
  ``torch.save(module)``; kept for back-compat with existing trained models).
- ``actor_optimizer.pth`` / ``critic_optimizer.pth`` — optimizer states.

The legacy ``save_point``/``check_point`` also locked four fds independently;
this serialises the whole checkpoint under one lock file to avoid partial
states appearing to a concurrent reader.
"""
import fcntl
import os
from typing import Optional, Tuple

import torch
from loguru import logger

from nyctea.config import toolConfig

CHECKPOINT_NAMES = (
    "actor.pth", "critic.pth",
    "actor_optimizer.pth", "critic_optimizer.pth",
)


def _lock_path(model_dir: str) -> str:
    return os.path.join(model_dir, ".checkpoint.lock")


def checkpoint_exists(model_dir: Optional[str] = None) -> bool:
    """Whether a trained actor.pth + critic.pth are present."""
    model_dir = model_dir or toolConfig.model_dir()
    return (os.path.exists(os.path.join(model_dir, "actor.pth"))
            and os.path.exists(os.path.join(model_dir, "critic.pth")))


def save_checkpoint(actor, critic, actor_optim, critic_optim,
                    model_dir: Optional[str] = None) -> None:
    """Write all four checkpoint files atomically under one flock.

    Writes to ``*.tmp`` then renames, so a concurrent reader never sees a
    partially-written file; the lock serialises two writers.
    """
    model_dir = model_dir or toolConfig.model_dir()
    os.makedirs(model_dir, exist_ok=True)
    payloads = {
        "actor.pth": actor, "critic.pth": critic,
        "actor_optimizer.pth": actor_optim, "critic_optimizer.pth": critic_optim,
    }
    with open(_lock_path(model_dir), "w") as lockfp:
        fcntl.flock(lockfp.fileno(), fcntl.LOCK_EX)
        try:
            for name, obj in payloads.items():
                path = os.path.join(model_dir, name)
                tmp = path + ".tmp"
                torch.save(obj, tmp)
                os.replace(tmp, path)
        finally:
            fcntl.flock(lockfp.fileno(), fcntl.LOCK_UN)


def load_checkpoint(device="cpu", model_dir: Optional[str] = None):
    """Load the four checkpoint files under a shared flock.

    Returns ``(actor, critic, actor_optimizer, critic_optimizer)``. Returns
    ``None`` if no checkpoint exists.
    """
    model_dir = model_dir or toolConfig.model_dir()
    if not checkpoint_exists(model_dir):
        return None

    with open(_lock_path(model_dir), "w") as lockfp:
        fcntl.flock(lockfp.fileno(), fcntl.LOCK_SH)
        try:
            actor = torch.load(os.path.join(model_dir, "actor.pth"), map_location=device)
            critic = torch.load(os.path.join(model_dir, "critic.pth"), map_location=device)
            actor_opt = torch.load(os.path.join(model_dir, "actor_optimizer.pth"),
                                   map_location=device)
            critic_opt = torch.load(os.path.join(model_dir, "critic_optimizer.pth"),
                                    map_location=device)
        finally:
            fcntl.flock(lockfp.fileno(), fcntl.LOCK_UN)
    logger.info("Loaded previous checkpoint.")
    return actor, critic, actor_opt, critic_opt
