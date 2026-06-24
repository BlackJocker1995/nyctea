"""Tests for the npz-shard replay buffer (write → reload → sample)."""
import numpy as np
import pytest

from nyctea.buffer import ReplayBuffer, SHARD_SIZE


def _fake_transition(i, state_dim=12, action_dim=4):
    s = np.full(state_dim, i, dtype=np.float32)
    a = np.full(action_dim, i, dtype=np.float32)
    n = np.full(state_dim, i + 1, dtype=np.float32)
    return s, a, float(i), n


def test_add_then_flush_writes_shard(tmp_path):
    buf = ReplayBuffer(str(tmp_path), worker_id=0, capacity=100,
                       state_dim=12, action_dim=4)
    for i in range(SHARD_SIZE):
        buf.add(*_fake_transition(i))
    # SHARD_SIZE additions trigger an auto-flush.
    import glob, os
    shards = glob.glob(os.path.join(str(tmp_path), "buffer_*.npz"))
    assert len(shards) == 1
    data = np.load(shards[0], allow_pickle=False)
    assert data["s"].shape == (SHARD_SIZE, 12)
    assert data["a"].shape == (SHARD_SIZE, 4)
    assert data["r"][0] == 0.0
    assert data["r"][-1] == SHARD_SIZE - 1


def test_reload_loads_all_shards(tmp_path):
    # capacity large enough that the FIFO cap doesn't drop anything.
    cap = 2 * SHARD_SIZE + 10
    buf = ReplayBuffer(str(tmp_path), worker_id=0, capacity=cap,
                       state_dim=12, action_dim=4)
    # Write two shards from two simulated workers.
    for w in (0, 1):
        wbuf = ReplayBuffer(str(tmp_path), worker_id=w, capacity=cap,
                            state_dim=12, action_dim=4)
        for i in range(SHARD_SIZE):
            wbuf.add(*_fake_transition(i + w * 1000))
        wbuf.flush()

    buf.reload()
    assert len(buf) == 2 * SHARD_SIZE


def test_sample_returns_correct_shapes(tmp_path):
    buf = ReplayBuffer(str(tmp_path), worker_id=0, capacity=100,
                       state_dim=8, action_dim=3)
    for i in range(30):
        buf.add(*_fake_transition(i, state_dim=8, action_dim=3))
    buf.flush()
    buf.reload()
    s, a, r, n = buf.sample(16)
    assert s.shape == (16, 8)
    assert a.shape == (16, 3)
    assert r.shape == (16,)


def test_sample_requires_enough_data(tmp_path):
    buf = ReplayBuffer(str(tmp_path), worker_id=0, capacity=100,
                       state_dim=8, action_dim=3)
    for i in range(5):
        buf.add(*_fake_transition(i, state_dim=8, action_dim=3))
    buf.flush()
    buf.reload()
    with pytest.raises(ValueError):
        buf.sample(16)


def test_capacity_fifo_cap(tmp_path):
    buf = ReplayBuffer(str(tmp_path), worker_id=0, capacity=10,
                       state_dim=4, action_dim=2)
    for i in range(30):
        buf.add(np.full(4, i, dtype=np.float32),
                np.full(2, i, dtype=np.float32), float(i),
                np.full(4, i + 1, dtype=np.float32))
    buf.flush()
    buf.reload()
    # Only the most recent 10 transitions survive the FIFO cap.
    assert len(buf) == 10
    s, _, r, _ = buf.sample(10)
    assert r.min() == pytest.approx(20.0)
    assert r.max() == pytest.approx(29.0)


def test_reload_handles_empty_dir(tmp_path):
    buf = ReplayBuffer(str(tmp_path), worker_id=0, capacity=100)
    buf.reload()
    assert len(buf) == 0
