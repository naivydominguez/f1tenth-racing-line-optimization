"""Small helpers."""
import contextlib
import os
import sys


@contextlib.contextmanager
def quiet_stdout():
    """Silence C-level stdout (e.g. the qpOASES banner) inside the block."""
    sys.stdout.flush()
    fd = sys.stdout.fileno()
    saved = os.dup(fd)
    with open(os.devnull, 'w') as dn:
        os.dup2(dn.fileno(), fd)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, fd)
            os.close(saved)
