"""Keep long local jobs from freezing the dev laptop (16 logical CPUs, 13.7 GB RAM, Windows).

lower_priority(): below-normal CPU priority for this process; on Windows child processes (ProcessPoolExecutor
workers) inherit the priority class, on Linux os.nice applies to children forked afterwards.
Rules learned the hard way (two crashes, 2026-10-14): read page shards by row group (crawl.manifest.read_html_rows),
keep big matrices in float16 and multiply them block by block, at most 3 worker processes, one heavy job at a time.
"""

from __future__ import annotations

import os
import sys


def lower_priority() -> None:
    if sys.platform == "win32":
        import ctypes

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000  # noqa: N806
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    else:
        try:
            os.nice(10)
        except OSError:
            pass
