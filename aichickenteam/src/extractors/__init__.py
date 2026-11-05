"""Main-content extraction entry point.

`extract(body, url, header_charset) -> dict(title, text, lang, method, encoding)`
Site-specific extractors (src/extractors/sites.py) run first for known hosts; if they return
nothing usable we fall back to trafilatura + density heuristics.
"""
from __future__ import annotations

from urllib.parse import urlsplit

try:
    from selectolax.parser import HTMLParser
except ImportError:  # selectolax >= 1.0 bỏ backend modest -> dùng lexbor (API tương thích)
    from selectolax.lexbor import LexborHTMLParser as HTMLParser

from .generic import generic_extract
from .sites import find_site_extractor
from .text import decode_html, detect_lang, join_paras, norm_para, normalize_text  # noqa: F401


def extract_html(html: str, url: str, min_chars: int = 80) -> dict:
    host = (urlsplit(url).hostname or "").lower()
    tree = HTMLParser(html)
    title, text, method = "", "", ""
    fn = find_site_extractor(host)
    if fn is not None:
        try:
            res = fn(tree, url)
        except Exception as e:  # never let a site rule kill the page
            res = None
            method = f"site_error:{type(e).__name__}"
        if res:
            title, text = res
            method = f"site:{fn.__name__}"
    # a site rule's output is trusted even when short (Q&A pages with one-line answers);
    # generic extraction only runs when there is no rule or the rule found (almost) nothing
    trusted = method.startswith("site:") and (text or getattr(fn, "strict", False))
    if not trusted and len(text) < min_chars:
        t2, x2, m2 = generic_extract(html, url, tree=HTMLParser(html), min_chars=min_chars)
        if len(x2) > len(text):
            text, method = x2, m2
        title = title or t2
    title = norm_para(title)
    if title and title in text[: len(title) * 3 + 200]:
        # the heading is kept in `title`; don't duplicate it (or a "Tin tức" crumb) as first paragraphs
        paras = text.split("\n\n")
        head = paras[:3]
        if title in head:
            k = head.index(title)
            if all(len(p) <= 12 for p in head[:k]) and len(paras) > k + 1:
                text = "\n\n".join(paras[k + 1:])
    return {"title": title, "text": text, "lang": detect_lang(title + "\n" + text, host),
            "method": method}


def extract(body: bytes, url: str, header_charset: str | None = None, min_chars: int = 80) -> dict:
    html, enc = decode_html(body, header_charset)
    out = extract_html(html, url, min_chars=min_chars)
    out["encoding"] = enc
    return out
