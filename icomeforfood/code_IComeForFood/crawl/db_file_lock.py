"""Short cross-process lock for access to the shared DuckDB file."""

from contextlib import contextmanager
import os
from pathlib import Path
import time


@contextmanager
def db_file_lock(database: Path, timeout: float = 120):
    path = Path(database).parent / 'locks' / 'status.duckdb.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        while True:
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Timed out waiting for DuckDB lock: {path}') from exc
                time.sleep(0.2)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
