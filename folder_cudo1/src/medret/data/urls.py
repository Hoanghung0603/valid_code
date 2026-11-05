"""URL canonicalisation and host metadata for links_corpus.

`canonical_key` only merges URLs that are guaranteed to address the same resource (scheme, www.,
fragment, percent-encoding, trailing slash, tracking params). Content-level duplicates (syndicated
news, same article under several paths) need text comparison after crawling; see
docs/analysis/data-analysis.md section 4.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

# Query params that never change the page content in this corpus (baidu `from/sf_*`, pmphai `sessionId`).
TRACKING_PARAMS = frozenset({"from", "sf_ref", "sf_ch", "sessionid", "utm_source", "utm_medium", "utm_campaign"})

# Vietnamese hosts whose TLD is not .vn. Every other non-.vn host in the corpus is Chinese.
_VI_NON_VN_HOSTS = frozenset(
    {"vinmec.com", "hellobacsi.com", "vnexpress.net", "benhvienvietduc.org", "pmc-ecm-healthblog.beta.pharmacity.io"}
)


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def host_language(host: str) -> str:
    """'vi' or 'zh'. The corpus has no English-language host (see data-analysis.md section 2)."""
    return "vi" if host.endswith(".vn") or host in _VI_NON_VN_HOSTS else "zh"


def canonical_key(url: str) -> str:
    parts = urlsplit(url)
    path = unquote(parts.path).rstrip("/")
    query = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS
    )
    key = host_of(url) + path
    return key + "?" + urlencode(query) if query else key
