import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    """
    Exception class for timeout

    """

    pass


@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException('Timeout')

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
