"""Close evaluation workers on normal return, errors and catchable termination."""
from contextlib import contextmanager
import signal


@contextmanager
def evaluation_environment(factory):
    previous = {}
    closing = False
    env = None

    def interrupt(signum, frame):
        if closing:
            return  # Let the bounded worker shutdown complete on repeated signals.
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupt)
        env = factory()
        yield env
    finally:
        closing = True
        try:
            if env is not None:
                env.close()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
