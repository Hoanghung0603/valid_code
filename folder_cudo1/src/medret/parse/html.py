"""HTML bytes -> structured blocks (heading / paragraph / list item / bare text) via resiliparse.

resiliparse's `minimal_html` output keeps <h1-6>, <p>, <li>, <br> around the main-content text, which
gives block structure without a per-site DOM walker. Block text is entity-decoded but otherwise as
extracted; normalisation and boilerplate removal live in `medret.parse.clean`.
"""

from __future__ import annotations

import html as htmllib
import re
from dataclasses import dataclass

_META_CHARSET = re.compile(rb"<meta[^>]+charset=[\"']?\s*([A-Za-z0-9_\-]+)", re.I)
_HEADER_CHARSET = re.compile(r"charset=([\w\-]+)", re.I)
_GB_FAMILY = {"gb2312", "gbk", "gb18030", "x-gbk", "cp936"}


def decode_html(raw: bytes, content_type: str = "") -> tuple[str, str]:
    """Decode with header / <meta> charset (GBK family -> GB18030, its superset), else detect."""
    m = _HEADER_CHARSET.search(content_type or "")
    header_cs = m.group(1).lower() if m else None
    m = _META_CHARSET.search(raw[:4096])
    meta_cs = m.group(1).decode("ascii", "ignore").lower() if m else None
    # Chinese sites often send "utf-8" in the header while the page itself is GBK.
    cs = header_cs
    if meta_cs and (not cs or (cs.replace("-", "") == "utf8" and meta_cs in _GB_FAMILY)):
        cs = meta_cs
    if cs in _GB_FAMILY:
        cs = "gb18030"
    if cs:
        try:
            return raw.decode(cs), cs
        except (LookupError, UnicodeDecodeError):
            pass
    for cs in ("utf-8", "gb18030"):
        try:
            return raw.decode(cs), cs
        except UnicodeDecodeError:
            pass
    from charset_normalizer import from_bytes

    best = from_bytes(raw).best()
    return (str(best), best.encoding) if best else (raw.decode("utf-8", "replace"), "utf-8?")


@dataclass
class RawBlock:
    kind: str  # h1..h6 | p | li | text
    text: str
    link_ratio: float = 0.0  # share of non-space chars inside <a>
    in_main: bool = True  # kept by resiliparse's main-content heuristic
    in_region: bool = True  # lies between the first and last main-content block


@dataclass
class Extracted:
    title: str  # <title>
    h1: str  # first <h1> in the whole page (not only main content)
    blocks: list[RawBlock]  # full-page blocks in page order, flagged in_main / in_region


_TAG_SPLIT = re.compile(r"(<(/?)(h[1-6]|p|li|ul|ol|pre|br|a)(?:\s[^>]*)?/?>)", re.I)


def _nonspace(s: str) -> int:
    return len(s) - sum(c.isspace() for c in s)


def _blocks_from_minimal_html(markup: str) -> list[RawBlock]:
    """Split resiliparse minimal_html into blocks. Text outside any block tag becomes one block per line.

    Text is buffered as (segment, inside_link) pairs so every block, and every bare line, gets a link ratio.
    """
    blocks: list[RawBlock] = []
    open_kind: str | None = None
    in_link = False
    buf: list[tuple[str, bool]] = []

    def make(kind: str, segs: list[tuple[str, bool]]) -> None:
        text = "".join(t for t, _ in segs)
        total = _nonspace(text)
        if total:
            linked = sum(_nonspace(t) for t, a in segs if a)
            blocks.append(RawBlock(kind, text, linked / total))

    def flush(kind: str | None) -> None:
        segs = [(htmllib.unescape(t), a) for t, a in buf]
        buf.clear()
        if kind is not None:
            make(kind, segs)
            return
        line: list[tuple[str, bool]] = []
        for t, a in segs:
            for i, part in enumerate(t.split("\n")):
                if i:
                    make("text", line)
                    line = []
                line.append((part, a))
        make("text", line)

    pos = 0
    for m in _TAG_SPLIT.finditer(markup):
        buf.append((markup[pos : m.start()], in_link))
        pos = m.end()
        closing, tag = m.group(2) == "/", m.group(3).lower()
        if tag == "a":
            in_link = not closing
        elif tag == "br":
            buf.append(("\n", False))
        elif tag in ("ul", "ol", "pre"):
            continue
        elif not closing:
            flush(open_kind)
            open_kind = "li" if tag == "li" else tag
        else:
            flush(open_kind)
            open_kind = None
    buf.append((markup[pos:], in_link))
    flush(open_kind)
    return blocks


def extract(html: str, skip_elements: list[str] | None = None, min_overlap: float = 0.5) -> Extracted:
    """Full-page blocks (verbatim text), each flagged with whether main_content kept it.

    main_content output is used only as a selection signal, never as text: it silently drops some inline
    link texts inside paragraphs ("中午用茶下" -> "中午用下"), so its text is not verbatim page text.
    """
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree

    tree = HTMLTree.parse(html)
    h1 = tree.document.query_selector("h1")
    opts = dict(preserve_formatting="minimal_html", list_bullets=False, alt_texts=False, links=True,
                skip_elements=skip_elements)
    full = _blocks_from_minimal_html(extract_plain_text(tree, main_content=False, **opts))
    main = _blocks_from_minimal_html(extract_plain_text(tree, main_content=True, **opts))
    flag_main(full, main, min_overlap)
    return Extracted(title=(tree.title or "").strip(), h1=h1.text.strip() if h1 is not None else "", blocks=full)


_N = 4


def _grams(key: str) -> set[str]:
    return {key[i : i + _N] for i in range(len(key) - _N + 1)}


def flag_main(full: list[RawBlock], main: list[RawBlock], min_overlap: float = 0.5) -> None:
    """Set in_main / in_region on full-page blocks from the main-content extraction.

    The two modes segment and (for main) alter text differently, so a block is "in main" when at least
    `min_overlap` of its character 4-grams occur in the main text; blocks shorter than 4 chars need an exact
    match with a main block. in_region marks blocks between the first and last in-main block.
    """
    main_keys = {_key(m.text) for m in main}
    main_grams: set[str] = set()
    for k in main_keys:
        main_grams |= _grams(k)
    flags = []
    for b in full:
        k = _key(b.text)
        g = _grams(k)
        flags.append(k in main_keys if not g else len(g & main_grams) / len(g) >= min_overlap)
    hits = [i for i, f in enumerate(flags) if f]
    lo, hi = (hits[0], hits[-1]) if hits else (len(full), -1)
    for i, (b, f) in enumerate(zip(full, flags, strict=True)):
        b.in_main = f
        b.in_region = lo < i < hi


def _key(text: str) -> str:
    return re.sub(r"\s+", "", text)
