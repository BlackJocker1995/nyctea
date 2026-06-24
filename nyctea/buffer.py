"""RL replay buffer — npz-shard-backed, replacing ``shared_memory.ShareableList``.

The legacy code pooled transitions in a single ``shared_memory.ShareableList``
named ``"rl_buffer"``. This module replaces it with a **file-as-persistence,
memory-as-sampling** design:

- **Workers append** transitions (:meth:`ReplayBuffer.add`), which buffer
  in-process and flush to a rolling shard ``buffer_{worker}_{chunk}.npz`` every
  ``SHARD_SIZE`` transitions (or on :meth:`flush`). Each flush is written under
  ``fcntl.flock`` so a concurrent learner can't read a half-written shard.
- **The learner periodically reloads** (:meth:`ReplayBuffer.reload`) all shards
  into a contiguous in-memory ndarray via :func:`np.load` (``mmap_mode='r'`` for
  large buffers), then :meth:`sample` draws a minibatch from memory — so the
  training hot path pays no per-step file I/O.

This keeps the sampling speed of the old shared-memory design (memory-resident)
while gaining crash recoverability (shards survive a worker death) and dropping
the Ray/shared-memory dependency. Capacity is a circular cap applied at reload.

Transition layout: ``(state, action, reward, next_state)`` stored as four
arrays keyed ``s``/``a``/``r``/``s2`` in each shard.
"""
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


# Default transitions per shard file. Larger = fewer files / fewer flushes;
# smaller = finer-grained recovery. Tunable per-instance.
SHARD_SIZE = 500


class ReplayBuffer:
    """Rolling npz-shard replay buffer with in-memory sampling.

    Args:
        buffer_dir: directory to write shards into (created lazily).
        worker_id: this writer's 0-based id (names its shards ``buffer_{id}_*``).
        capacity: max transitions kept in memory after :meth:`reload` (FIFO drop).
        state_dim / action_dim: shapes of a single transition (for preallocation).
    """

    def __init__(self, buffer_dir: str, worker_id: int = 0, capacity: int = 10000,
                 state_dim: Optional[int] = None, action_dim: Optional[int] = None):
        self.buffer_dir = buffer_dir
        self.worker_id = int(worker_id)
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        os.makedirs(buffer_dir, exist_ok=True)

        # In-flight transitions not yet flushed to a shard.
        self._pending: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        self._chunk = 0

        # In-memory view used for sampling (populated by reload).
        self._mem_states: Optional[np.ndarray] = None
        self._mem_actions: Optional[np.ndarray] = None
        self._mem_rewards: Optional[np.ndarray] = None
        self._mem_next_states: Optional[np.ndarray] = None
        self._mem_len = 0

    # --------------------------------------------------------------- writing
    def add(self, state, action, reward, next_state):
        """Append one transition; flush a shard when ``SHARD_SIZE`` accumulate."""
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if self.state_dim is None:
            self.state_dim = state.shape[0]
        if self.action_dim is None:
            self.action_dim = action.shape[0]
        self._pending.append((state, action, float(reward), next_state))
        if len(self._pending) >= SHARD_SIZE:
            self.flush()

    def shard_path(self, chunk: int) -> str:
        return os.path.join(self.buffer_dir, f"buffer_{self.worker_id}_{chunk}.npz")

    def flush(self):
        """Write all pending transitions to a new shard (flock-guarded)."""
        if not self._pending:
            return
        states = np.stack([t[0] for t in self._pending])
        actions = np.stack([t[1] for t in self._pending])
        rewards = np.array([t[2] for t in self._pending], dtype=np.float32)
        next_states = np.stack([t[3] for t in self._pending])

        path = self.shard_path(self._chunk)
        # Atomic-ish write: build to a temp file then rename under the lock.
        # np.savez always appends '.npz', so name the temp WITHOUT the suffix
        # (np adds it → 'path.tmp.npz') and rename that to the final path.
        import fcntl
        np.savez(path + ".tmp", s=states, a=actions, r=rewards, s2=next_states)
        tmp = path + ".tmp.npz"
        with open(path + ".lock", "w") as lockfp:
            fcntl.flock(lockfp.fileno(), fcntl.LOCK_EX)
            try:
                os.replace(tmp, path)
            finally:
                fcntl.flock(lockfp.fileno(), fcntl.LOCK_UN)
        logger.debug(f"flushed {len(self._pending)} transitions to {path}")
        self._pending.clear()
        self._chunk += 1

    # --------------------------------------------------------------- reading
    def reload(self):
        """Load every shard in ``buffer_dir`` into memory for sampling.

        Concatenates all ``buffer_*.npz`` (from every worker), applies the FIFO
        capacity cap, and stores four contiguous ndarrays. Safe to call while
        workers are appending (flushes are atomic; a half-written shard is
        skipped via the temp-rename dance).
        """
        shards = sorted(glob.glob(os.path.join(self.buffer_dir, "buffer_*.npz")))
        if not shards:
            self._mem_len = 0
            return

        all_s, all_a, all_r, all_s2 = [], [], [], []
        for shard in shards:
            try:
                data = np.load(shard, allow_pickle=False)
                all_s.append(data["s"])
                all_a.append(data["a"])
                all_r.append(data["r"])
                all_s2.append(data["s2"])
            except (OSError, KeyError) as e:
                # A shard mid-flush or corrupt — skip it.
                logger.debug(f"skipping unreadable shard {shard}: {e}")
                continue

        states = np.concatenate(all_s, axis=0)
        actions = np.concatenate(all_a, axis=0)
        rewards = np.concatenate(all_r, axis=0)
        next_states = np.concatenate(all_s2, axis=0)

        # FIFO cap: keep the most recent ``capacity`` transitions.
        if states.shape[0] > self.capacity:
            states = states[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_states = next_states[-self.capacity:]

        self._mem_states = states
        self._mem_actions = actions
        self._mem_rewards = rewards
        self._mem_next_states = next_states
        self._mem_len = states.shape[0]
        logger.debug(f"buffer reloaded: {self._mem_len} transitions in memory")

    def __len__(self):
        return self._mem_len

    def sample(self, batch_size: int):
        """Draw a random minibatch from the in-memory view.

        Returns ``(state, action, reward, next_state)`` ndarrays. Call
        :meth:`reload` first (typically periodically, not per-step).
        """
        import random
        if self._mem_len < batch_size:
            raise ValueError(
                f"buffer has {self._mem_len} transitions, need {batch_size}")
        idx = random.sample(range(self._mem_len), batch_size)
        return (self._mem_states[idx], self._mem_actions[idx],
                self._mem_rewards[idx], self._mem_next_states[idx])

    # --------------------------------------------------------------- migration
    @staticmethod
    def load_legacy_pickle(path: str):
        """Read a legacy ``buffer.pkl`` (a list of pickled transitions).

        Kept for one-time migration: the old ``shared_memory.ShareableList``
        was itself dumped to ``buffer.pkl``. Returns a list of
        ``(state, action, reward, next_state)`` tuples, or ``None`` if absent.
        """
        import pickle
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        # The legacy dump stored [count, t1, t2, ...] where tN is a pickled
        # (state, action, reward, next_state) tuple.
        out = []
        for item in data[1:int(data[0]) + 1] if isinstance(data, list) and data and isinstance(data[0], int) else data:
            out.append(pickle.loads(item) if isinstance(item, bytes) else item)
        return out
