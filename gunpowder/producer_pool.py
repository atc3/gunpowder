import logging
import multiprocessing
import os
import queue as Queue
import signal
import sys

import numpy as np

logger = logging.getLogger(__name__)

# Seconds the watchdog waits for the consumer to pick up a worker failure before it terminates the
# consumer (SIGTERM). A worker killed while writing to the result queue (e.g. by the OOM killer) can leave
# the queue unusable -- a half-written batch or a held write lock -- and the consumer would then block in
# get() forever with the job still "running". 0 disables the escalation.
HANG_TIMEOUT_ENV = "GUNPOWDER_PRODUCER_POOL_HANG_TIMEOUT"
DEFAULT_HANG_TIMEOUT = 600


class NoResult(Exception):
    pass


class ParentDied(Exception):
    pass


class WorkersDied(Exception):
    pass


def _describe_exitcode(code):
    if code is None:
        return "still running"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        hint = (
            " -- likely the OOM killer; check the job's memory limit"
            if -code == signal.SIGKILL
            else ""
        )
        return f"killed by {name}{hint}"
    return f"exit code {code}"


class ProducerPool(object):
    def __init__(self, callables, queue_size=10):
        self.__watch_dog = multiprocessing.Process(
            target=self._run_watch_dog, args=(callables,)
        )
        self.__stop = multiprocessing.Event()
        self.__result_queue = multiprocessing.Queue(queue_size)
        # Worker failure is signalled OUT OF BAND: a dead worker can corrupt the result queue, so the
        # failure must not depend on that queue being readable. `failed` is set by the watchdog,
        # `acknowledged` by the consumer once it has raised, and `dead_exitcode` carries the first dead
        # worker's exit code (sentinel 1 << 30 = unknown) into the consumer's exception message.
        self.__failed = multiprocessing.Event()
        self.__acknowledged = multiprocessing.Event()
        self.__dead_exitcode = multiprocessing.Value("i", 1 << 30)

    def __del__(self):
        self.stop()

    def start(self):
        """Start the pool of producers."""

        if self.__watch_dog is None:
            raise RuntimeError("can't start a ProducerPool a second time")

        if self.__watch_dog.is_alive():
            logger.warning("trying to start workers, but they are already running")
            return

        self.__stop.clear()
        self.__watch_dog.start()

    def _raise_if_failed(self):
        if self.__failed.is_set():
            self.__acknowledged.set()
            code = self.__dead_exitcode.value
            detail = "" if code == 1 << 30 else f" ({_describe_exitcode(code)})"
            raise WorkersDied(f"a producer worker died{detail}")

    def get(self, timeout=0):
        """Return the next result from the producer pool.

        If timeout is set and there is not result after the given number of
        seconds, exception NoResult is raised.

        Raises WorkersDied as soon as the watchdog has seen a worker die, even
        if the result queue still holds (or appears to hold) batches.
        """

        block = False
        if timeout == 0:
            timeout = 1
            block = True

        item = None
        while item is None:
            self._raise_if_failed()
            watch_dog = self.__watch_dog
            if (
                watch_dog is not None
                and watch_dog.pid is not None
                and not watch_dog.is_alive()
            ):
                if not self.__stop.is_set():
                    self._raise_if_failed()
                    raise WorkersDied(
                        f"producer pool watchdog exited ({_describe_exitcode(watch_dog.exitcode)})"
                    )
            try:
                item = self.__result_queue.get(timeout=timeout)
            except Queue.Empty:
                if not block:
                    raise NoResult()

        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        """Stop the pool of producers.

        Items currently being produced will not be waited for and be discarded."""

        if self.__watch_dog is None:
            return

        self.__stop.set()
        if self.__watch_dog._popen is not None:  # ty: ignore[missing-attribute]
            self.__watch_dog.join(timeout=60)
            if self.__watch_dog.is_alive():
                logger.error(
                    "producer pool watchdog did not exit within 60 s, terminating it"
                )
                self.__watch_dog.terminate()
                self.__watch_dog.join(timeout=10)
        self.__watch_dog = None

    def _hang_timeout(self):
        try:
            return float(os.environ.get(HANG_TIMEOUT_ENV, DEFAULT_HANG_TIMEOUT))
        except ValueError:
            return DEFAULT_HANG_TIMEOUT

    def _run_watch_dog(self, callables):
        parent_pid = os.getppid()

        logger.debug("watchdog started with PID " + str(os.getpid()))
        logger.debug("parent PID " + str(parent_pid))

        workers = [
            multiprocessing.Process(target=self._run_worker, args=(c,))
            for c in callables
        ]

        try:
            logger.debug("starting %d workers" % len(workers))
            for worker in workers:
                worker.start()

            while not self.__stop.wait(1):
                if os.getppid() != parent_pid:
                    logger.error("parent of producer pool died, shutting down")
                    self.__result_queue.put(ParentDied())
                    break
                dead = [w for w in workers if not w.is_alive()]
                if dead:
                    for w in dead:
                        logger.error(
                            "producer worker PID %s died: %s",
                            w.pid,
                            _describe_exitcode(w.exitcode),
                        )
                    if dead[0].exitcode is not None:
                        self.__dead_exitcode.value = dead[0].exitcode
                    logger.error("at least one of my workers died, shutting down")
                    self.__failed.set()
                    self._escalate_if_not_acknowledged(parent_pid)
                    break
        except:
            pass

        finally:
            # a put from this process may be stuck behind a lock the dead worker held; never let that
            # block this process's exit (and with it the consumer's stop())
            self.__result_queue.cancel_join_thread()

            logger.info("terminating workers...")
            for worker in workers:
                worker.terminate()

            logger.info("joining workers...")
            for worker in workers:
                worker.join(timeout=10)

            logger.info("done")

    def _escalate_if_not_acknowledged(self, parent_pid):
        """Wait for the consumer to raise WorkersDied; if it does not within the hang timeout, it is
        presumably blocked on a corrupted result queue -- terminate it so the job fails instead of
        holding its resources indefinitely."""
        timeout = self._hang_timeout()
        if timeout <= 0:
            return
        waited = 0.0
        while waited < timeout:
            if self.__acknowledged.wait(1) or self.__stop.is_set():
                return
            if os.getppid() != parent_pid:
                return
            waited += 1
        logger.error(
            "consumer (PID %s) did not pick up the worker failure within %.0f s -- the result queue is "
            "probably corrupted by the dead worker; sending SIGTERM so the job fails instead of hanging "
            "(set %s=0 to disable)",
            parent_pid,
            timeout,
            HANG_TIMEOUT_ENV,
        )
        os.kill(parent_pid, signal.SIGTERM)

    def _run_worker(self, target):
        parent_pid = os.getppid()

        logger.debug("worker started with PID " + str(os.getpid()))
        logger.debug("parent PID " + str(parent_pid))

        result = None
        np.random.seed(None)
        while True:
            if os.getppid() != parent_pid:
                logger.debug("worker %d: watch-dog died, stopping" % os.getpid())
                break

            if result is None:
                try:
                    result = target()
                except Exception as e:
                    result = e
                    # don't stop on normal exceptions -- place them in result queue
                    # and let them be handled by caller
                except:
                    logger.error("received error: " + str(sys.exc_info()[0]))
                    # this is most likely a keyboard interrupt, stop process
                    break

            try:
                self.__result_queue.put(result, timeout=1)
                result = None
            except Queue.Full:
                logger.debug(
                    "worker %d: result queue is full, waiting to place my result"
                    % os.getpid()
                )

        logger.debug("worker with PID " + str(os.getpid()) + " exiting")
        os._exit(1)

    def _all_workers_alive(self, workers):
        return all([worker.is_alive() for worker in workers])
