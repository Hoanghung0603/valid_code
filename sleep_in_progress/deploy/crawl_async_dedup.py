"""Shared dedup sets for the async broad recovery job.

Owns: exact source corpus ID sets that must never be re-fetched:
  - original-ok: trunglt-198 state terminal `ok` (exact literal only)
  - b660 scope: cnkang.com + 120ask.com corpus IDs (outside trung scope)
  - ask20: 20 already-harvested ask.39.net IDs (verified zero-overlap)
  - browser2: 2 nodriver-recovered cnkang IDs (outside trung scope anyway)

Queue rule: queue = trung_scope_corpus_ids - original_ok - ask20.
`permanent` is INCLUDED in the queue (user explicitly requires permanent
extraction / short-content recovery). Unresolved (transient/blocked/
redirect) and never-tried are included. cnkang/120ask excluded entirely.
"""
from pathlib import Path

B660_HOSTS = ("cnkang.com", "120ask.com")

# Verified in local build step (see docs/crawl/ASYNC_RECOVERY.md):
# corpus 4394718; b660-scope 1881917; trung scope 2512801;
# trung-ok 1153950; ask20 20; queue = 2512801-1153950-20 = 1358831.
ASK20_IDS = frozenset([
    2136726, 2153594, 2170474, 2187381, 2204234, 2220408, 2236517,
    2253195, 2269177, 2286065, 2302757, 2319367, 2336353, 2353137,
    2369802, 2389718, 2406608, 2423508, 2440354, 2457223,
])

# nodriver browser recoveries (cnkang scope, not in trung queue anyway).
BROWSER2_IDS = frozenset([1881723, 1900274])

# 1 leftover from the 535 retry batch (text 181 < 200); cnkang scope.
RETRY_LEFTOVER_PERMANENT = frozenset([1677181])


def is_b660_scope(host: str) -> bool:
    return host in B660_HOSTS


def dedup_ok(queue_ids: set, original_ok: set, ask20: set = ASK20_IDS) -> set:
    """Return queue minus already-harvested sets. Pure set op, no I/O."""
    return set(queue_ids) - set(original_ok) - set(ask20)
