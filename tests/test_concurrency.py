"""Tests for LockedCsv (concurrent-safe appends) and MultiInstanceRunner."""
import csv
import multiprocessing as mp
import os
import tempfile

from nyctea.concurrency import LockedCsv, MultiInstanceRunner, WorkerContext


def _write_rows(path, header, rows):
    """A worker fn: append ``rows`` to the shared CSV under the lock."""
    def _fn(ctx):
        lc = LockedCsv(path, header=header)
        lc.ensure_created()
        for r in rows:
            lc.append_row(r)
    return _fn


def test_locked_csv_append_and_read():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.csv")
        lc = LockedCsv(path, header=["a", "b"])
        lc.ensure_created()
        lc.append_row([1, 2])
        lc.append_row([3, 4])
        rows = lc.read_rows()
        assert rows[0] == ["a", "b"]
        assert rows[1] == ["1", "2"]
        assert rows[2] == ["3", "4"]


def test_locked_csv_concurrent_no_corruption():
    """N workers appending to one CSV must not interleave or lose rows."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "shared.csv")
        header = ["worker", "i"]
        n_workers, n_rows = 6, 50

        runner = MultiInstanceRunner(
            n_workers,
            worker_fn=lambda ctx: LockedCsv(path, header=header).ensure_created()
            or [LockedCsv(path, header=header).append_row([ctx.instance_id, j])
                for j in range(n_rows)],
        )
        runner.run()

        with open(path) as f:
            data_rows = list(csv.reader(f))[1:]  # drop header
        assert len(data_rows) == n_workers * n_rows
        # Every row has exactly 2 fields (no interleaving broke a record).
        assert all(len(r) == 2 for r in data_rows)


def test_locked_csv_rows_as_dataframe_empty():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "missing.csv")
        lc = LockedCsv(path, header=["x"])
        df = lc.rows_as_dataframe()
        assert list(df.columns) == ["x"]
        assert len(df) == 0


def test_multi_instance_runner_surfaces_failures():
    """A crashing worker makes run() raise RuntimeError listing the instance."""
    def boom(ctx: WorkerContext):
        if ctx.instance_id == 1:
            raise RuntimeError("worker 1 died")

    runner = MultiInstanceRunner(3, worker_fn=boom)
    try:
        runner.run()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "1" in str(e)


def test_multi_instance_runner_rejects_zero():
    try:
        MultiInstanceRunner(0, worker_fn=lambda ctx: None)
        assert False, "expected ValueError"
    except ValueError:
        pass
