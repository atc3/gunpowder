"""ProducerPool must fail loudly, never hang, when a worker dies (e.g. SIGKILLed by the OOM killer).

Before the fix, the watchdog reported the death through the result queue, which the killed worker
could leave unusable; the consumer then blocked in get() forever while the job kept its GPU.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from gunpowder.producer_pool import ProducerPool, WorkersDied

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ok():
    time.sleep(0.01)
    return 1


def _die_after_a_moment():
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGKILL)


class HalfWriteThenDie:
    """A worker that leaves the result queue corrupted exactly as a mid-write SIGKILL does: it writes a
    message header promising 10 MB, then only 100 bytes, then dies. A consumer in get() reads the header
    and blocks in recv forever -- the failure mode the out-of-band signal and the escalation exist for."""

    pool = None

    def __call__(self):
        import struct

        writer = self.pool._ProducerPool__result_queue._writer
        os.write(writer.fileno(), struct.pack("!i", 10_000_000) + b"x" * 100)
        os.kill(os.getpid(), signal.SIGKILL)


def test_consumer_blocked_on_a_half_written_batch_is_terminated():
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {REPO!r})
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from gunpowder.producer_pool import ProducerPool
        from producer_pool_failure import HalfWriteThenDie
        if __name__ == "__main__":
            worker = HalfWriteThenDie()
            pool = ProducerPool([worker], queue_size=2)
            worker.pool = pool
            pool.start()
            pool.get()
            print("RETURNED")
        """
    )
    env = dict(os.environ, GUNPOWDER_PRODUCER_POOL_HANG_TIMEOUT="3")
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert res.returncode == -signal.SIGTERM, (
        res.returncode,
        res.stdout,
        res.stderr[-2000:],
    )
    assert "did not pick up the worker failure" in res.stderr
    assert time.time() - t0 < 30


def test_healthy_pool_produces_and_stops():
    pool = ProducerPool([_ok, _ok], queue_size=4)
    pool.start()
    try:
        assert [pool.get() for _ in range(20)] == [1] * 20
    finally:
        pool.stop()


def test_sigkilled_worker_raises_instead_of_hanging():
    pool = ProducerPool([_ok, _die_after_a_moment], queue_size=4)
    pool.start()
    t0 = time.time()
    try:
        with pytest.raises(WorkersDied, match="SIGKILL"):
            while time.time() - t0 < 30:
                pool.get()
    finally:
        pool.stop()
    assert time.time() - t0 < 15


def test_consumer_that_never_collects_the_failure_is_terminated():
    # a consumer stuck elsewhere (here: sleeping instead of calling get()) stands in for one blocked on a
    # corrupted queue; the watchdog must SIGTERM it after the hang timeout
    code = textwrap.dedent(
        f"""
        import os, signal, sys, time
        sys.path.insert(0, {REPO!r})
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from gunpowder.producer_pool import ProducerPool
        from producer_pool_failure import _ok, _die_after_a_moment
        if __name__ == "__main__":
            pool = ProducerPool([_ok, _die_after_a_moment], queue_size=2)
            pool.start()
            time.sleep(60)
            print("NOT TERMINATED")
        """
    )
    env = dict(os.environ, GUNPOWDER_PRODUCER_POOL_HANG_TIMEOUT="3")
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert res.returncode == -signal.SIGTERM, (
        res.returncode,
        res.stdout,
        res.stderr[-2000:],
    )
    assert "NOT TERMINATED" not in res.stdout
    assert time.time() - t0 < 30


def test_escalation_can_be_disabled():
    code = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {REPO!r})
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from gunpowder.producer_pool import ProducerPool, WorkersDied
        from producer_pool_failure import _ok, _die_after_a_moment
        if __name__ == "__main__":
            pool = ProducerPool([_ok, _die_after_a_moment], queue_size=2)
            pool.start()
            time.sleep(6)
            try:
                while True:
                    pool.get()
            except WorkersDied as e:
                print("RAISED", e)
            pool.stop()
        """
    )
    env = dict(os.environ, GUNPOWDER_PRODUCER_POOL_HANG_TIMEOUT="0")
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    assert "RAISED a producer worker died (killed by SIGKILL" in res.stdout
