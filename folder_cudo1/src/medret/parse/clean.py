"""Block normalisation and boilerplate removal.

Normalisation is limited to what the official matcher ignores anyway (Unicode NFC, whitespace), so the
cleaned text stays verbatim page text: no case folding, no punctuation or script conversion.

Boilerplate removal works on whole lines/blocks only (never edits inside a sentence):
  - host templates: lines that occur on a large share of one host's pages (menus, footers, widgets);
  - leading chrome: blocks before the article <h1> (breadcrumbs, nav, login links);
  - related-content tail: everything after a "related articles / Tin liên quan"-style marker;
  - symbol-only lines (pagination, separators).
Each step can be switched off in `CleanConfig` so its effect is measurable.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from medret.parse.html import RawBlock

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_SPACES = re.compile(r"[^\S\n]+")  # any whitespace except newline (incl. \xa0, 　)


def normalize_block(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def template_key(line: str) -> str:
    return re.sub(r"\s+", "", line)


@dataclass
class HostTemplates:
    """Lines frequent across one host's pages, learned from a sample of that host.

    `lines` applies to body lines; `headings` (a stricter threshold) to h2-h6, so that recurring widget
    titles ("Theo dõi chúng tôi") go while ordinary section headings ("病因", "Triệu chứng") stay.
    """

    n_docs: int = 0
    lines: set[str] = field(default_factory=set)
    headings: set[str] = field(default_factory=set)


def learn_templates(
    docs: Iterable[list[RawBlock]], min_df: float = 0.1, min_docs: int = 3, heading_min_df: float = 0.5
) -> HostTemplates:
    df: Counter[str] = Counter()
    n = 0
    for blocks in docs:
        n += 1
        keys = {template_key(line) for b in blocks for line in normalize_block(b.text).split("\n")}
        df.update(k for k in keys if k)
    lines = {k for k, c in df.items() if c >= max(min_docs, min_df * n)}
    headings = {k for k, c in df.items() if c >= max(min_docs, heading_min_df * n)}
    return HostTemplates(n_docs=n, lines=lines, headings=headings)


_RELATED_ZH = [
    r"(本文)?(相关|热门|推荐|最新|热点|精彩|延伸|更多|其他|同类|猜你|大家)的?"
    r"(文章|阅读|推荐|医院|医生|专家|资讯|排行|问答|问题|话题|专题|标签|疾病|药品|链接|搜索|内容|新闻|百科|喜欢|关注|"
    r"都在看|都在搜|还看了)(推荐|排行|列表|榜)?",
    r"您可能(关注|感兴趣)的?(问题|文章)?",
    r"你可能(还)?感兴趣",
    r"看了本文的(网友|人)还看了",
    r"合作医院",
    r"免费提问",
    r"热点排行",
    r"排行榜",
]
_RELATED_VI = [
    "tin liên quan",
    "bài viết liên quan",
    "tin cùng chuyên mục",
    "có thể bạn quan tâm",
    "bài đọc nhiều",
    "đọc nhiều( nhất)?",
    "tin mới( nhất)?",
    "tin khác",
    "các tin khác",
    "tin nổi bật",
    "bài viết cùng chủ đề",
    "cùng chuyên mục",
    "xem nhiều( nhất)?",
    "bài viết khác",
    "tin tức liên quan",
    "bạn có thể quan tâm",
    "có thể bạn chưa biết",
    "tin cùng chủ đề",
    "bài liên quan",
    "tin xem nhiều",
    "tin đọc nhiều",
    "bài viết mới nhất",
    "bài mới",
    "tin tức mới nhất",
    "đọc thêm",
    "xem thêm",
    "bài viết nổi bật",
    "tin bài liên quan",
    "chủ đề liên quan",
    "tin cùng loại",
]
_RELATED = re.compile("^(" + "|".join(_RELATED_ZH + _RELATED_VI) + ")$", re.I)
_TAIL_MARKER = re.compile(r"^疾病百科\s*[|｜]")
# excerpt of another page: ends with an explicit "read more" mark
_TEASER_END = re.compile(r"(更多\s*>>|>>\s*$|\[(详细|说明书|全文|更多)\]|xem (thêm|tiếp|chi tiết)\s*(>>|»)?)\s*$", re.I)
_BASE64 = re.compile(r"[A-Za-z0-9+/=]{80,}")
_BROKEN = re.compile(r"[\ufffd\x00-\x08\x0b-\x1f\x7f]")  # replacement / control chars: binary decoded as text


def is_garbage(line: str) -> bool:
    """Base64 blobs or binary data (an image decoded as text) rendered into the page."""
    return bool(_BASE64.search(line)) or len(_BROKEN.findall(line)) > 0.1 * len(line)
_HAS_LETTER = re.compile(r"[^\W\d_]")
NL = "\n"


def is_related_marker(text: str) -> bool:
    return bool(_RELATED.match(text.strip(" :：>»|-—")))


@dataclass
class CleanConfig:
    """Defaults are the benchmarked choice (parse_bench v6/v7, docs/experiments/chunking-iterations.md)."""

    base: str = "gapfill"  # "main": blocks kept by resiliparse main_content; "gapfill": + prose it skipped
    max_link_ratio: float | None = None  # drop any non-title block this link-dense (rejected: <a>-wrapped pages)
    templates: bool = True
    trim_before_h1: bool = True
    safe_trim: bool = True  # never trim a prefix that contains a prose paragraph
    cut_related_tail: bool = True
    related_mode: str = "box"  # "tail": drop everything after a marker; "box": drop only the related box
    drop_symbol_lines: bool = True
    gap_max_link_ratio: float = 0.5
    short_link_ratio: float | None = 0.8  # drop short blocks with at least this link ratio (titles, menus)
    page_link_guard: float = 0.5  # ...unless this share of the page text is links (article wrapped in <a>)
    drop_teasers: bool = True  # blocks ending in an explicit "more >>" / "[详细]" mark (excerpts of other pages)
    heading_templates: bool = True  # h2-h6 on >= HostTemplates.headings share of the host's pages
    drop_garbage: bool = True  # lines carrying base64-like blobs rendered as text
    fix_headings: bool = True  # split multi-line headings, demote over-long ones to paragraphs
    tail_markers: bool = True  # cut closing widgets (_TAIL_MARKER) to the end of the page
    link_lists: bool = True  # short_link also drops long link-only runs without sentence punctuation
    dedup_blocks: bool = True  # drop a block repeating an earlier block of the same page verbatim


@dataclass
class Block:
    kind: str  # h1..h6 | p | li | text
    text: str
    link_ratio: float = 0.0


_CJK = re.compile(r"[\u3400-\u9fff]")
_PUNCT = re.compile(r"[。，；！？、,.;!?:：]")


def is_prose(text: str) -> bool:
    """Sentence-like text: has clause/sentence punctuation and is long enough for its script."""
    min_len = 8 if _CJK.search(text) else 25
    return len(text) >= min_len and bool(_PUNCT.search(text))


_BREADCRUMB = re.compile(r"→|>>|›|»|\s>\s")


def _is_long_prose(text: str) -> bool:
    """A real paragraph: long prose that is not a breadcrumb ("有问必答网 → 妇产科 → ... → <title>")."""
    return is_prose(text) and len(text) >= (25 if _CJK.search(text) else 60) and len(_BREADCRUMB.findall(text)) < 2


def _is_short(text: str) -> bool:
    return len(text) < (15 if _CJK.search(text) else 40)


def _is_short_title(text: str) -> bool:
    return len(text) < (50 if _CJK.search(text) else 120)


def related_box_end(blocks: list[Block], i: int, link: float = 0.5) -> int:
    """End (exclusive) of the related-content box whose marker is blocks[i].

    The box runs over link items, short lines (dates, "Chia sẻ") and teaser excerpts, i.e. prose that
    directly follows a link item and is followed (after short lines) by another link item or the page end.
    It stops at the first ordinary paragraph, so a "Tin liên quan" box inside an article body only removes
    the box, not the rest of the article.
    """
    j = i + 1
    while j < len(blocks):
        b = blocks[j]
        if b.kind == "h1":  # a marker in the navigation above the article: never swallow the title
            break
        if b.link_ratio >= link or _is_short(b.text):
            j += 1
            continue
        if blocks[j - 1].link_ratio >= link and j > i + 1:
            k = j + 1
            while k < len(blocks) and _is_short(blocks[k].text) and blocks[k].link_ratio < link:
                k += 1
            if k >= len(blocks) or blocks[k].link_ratio >= link:
                j = k
                continue
        break
    return j


def select_blocks(blocks: list[RawBlock], cfg: CleanConfig) -> tuple[list[RawBlock], list[tuple[str, str]]]:
    """Main-content blocks, plus (gapfill) prose blocks main_content skipped inside the main region."""
    kept, dropped = [], []
    for b in blocks:
        if b.in_main or (
            cfg.base == "gapfill"
            and b.in_region
            and b.link_ratio < cfg.gap_max_link_ratio
            and is_prose(normalize_block(b.text))
        ):
            kept.append(b)
        else:
            dropped.append(("not_main", b.text))
    return kept, dropped


def clean_blocks(
    blocks: list[RawBlock], page_h1: str, templates: HostTemplates | None, cfg: CleanConfig
) -> tuple[list[Block], list[tuple[str, str]]]:
    """Return kept blocks and the dropped (reason, text) pieces.

    Order matters: related boxes are found before template / link filtering, because their markers and
    link items are often templates or short links themselves and the box detector needs them.
    """
    raw, dropped = select_blocks(blocks, cfg)
    total = sum(len(b.text) for b in raw)
    page_link = sum(len(b.text) * b.link_ratio for b in raw) / total if total else 0.0

    out: list[Block] = []
    for b in raw:
        text = normalize_block(b.text)
        if not text:
            continue
        if cfg.max_link_ratio is not None and b.kind != "h1" and b.link_ratio >= cfg.max_link_ratio:
            dropped.append(("link_dense", text))
            continue
        if cfg.drop_symbol_lines:
            lines = text.split(NL)
            dropped.extend(("symbol", x) for x in lines if not _HAS_LETTER.search(x))
            text = NL.join(x for x in lines if _HAS_LETTER.search(x))
        if cfg.drop_garbage and text:
            lines = text.split(NL)
            dropped.extend(("garbage", x) for x in lines if is_garbage(x))
            text = NL.join(x for x in lines if not is_garbage(x))
        if not text:
            continue
        kind = b.kind
        if cfg.fix_headings and kind.startswith("h"):
            # <h2>title<br>body...</h2>: keep the first line as the heading, the rest is a paragraph
            first, _, rest = text.partition(NL)
            if rest:
                out.append(Block(kind, first, b.link_ratio))
                out.append(Block("p", rest, b.link_ratio))
                continue
            if len(text) > (60 if _CJK.search(text) else 150):
                kind = "p"
        out.append(Block(kind, text, b.link_ratio))

    if cfg.cut_related_tail and cfg.related_mode == "tail":
        for i, b in enumerate(out):
            if i > 0 and is_related_marker(b.text):
                dropped.extend(("related_tail", x.text) for x in out[i:])
                out = out[:i]
                break
    elif cfg.cut_related_tail and cfg.related_mode == "box":
        kept_blocks: list[Block] = []
        i = 0
        while i < len(out):
            if i > 0 and is_related_marker(out[i].text):
                end = related_box_end(out, i)
                dropped.extend(("related_box", x.text) for x in out[i:end])
                i = end
            else:
                kept_blocks.append(out[i])
                i += 1
        out = kept_blocks
    if cfg.tail_markers:
        # widgets that close a page (120ask "疾病百科| <disease>" encyclopedia card): cut to the end,
        # but only once some prose came before, so a page that is only the widget keeps it
        for i, b in enumerate(out):
            if _TAIL_MARKER.match(b.text) and any(_is_long_prose(x.text) for x in out[:i]):
                dropped.extend(("tail_widget", x.text) for x in out[i:])
                out = out[:i]
                break

    short_link = cfg.short_link_ratio is not None and page_link < cfg.page_link_guard
    filtered: list[Block] = []
    seen: set[str] = set()
    titles = {template_key(b.text) for b in out if b.kind == "h1"}
    if page_h1:
        titles.add(template_key(normalize_block(page_h1)))
    for b in out:
        if short_link and b.kind != "h1" and b.link_ratio >= cfg.short_link_ratio and (
            _is_short_title(b.text) or (cfg.link_lists and not _PUNCT.search(b.text))
        ):  # a menu item / title, or a run of link titles with no sentence punctuation
            dropped.append(("short_link", b.text))
            continue
        key = template_key(b.text)
        # a sapo or question title printed twice; short repeated lines are list structure ("Lần 2: ...") and stay
        if cfg.dedup_blocks and b.kind != "h1" and (key in titles or (key in seen and _is_long_prose(b.text))):
            dropped.append(("repeat", b.text))
            continue
        seen.add(key)
        if cfg.drop_teasers and b.kind != "h1" and _TEASER_END.search(b.text):
            dropped.append(("teaser", b.text))
            continue
        if b.kind.startswith("h") and b.kind != "h1":
            if cfg.heading_templates and templates is not None and template_key(b.text) in templates.headings:
                dropped.append(("template_heading", b.text))
                continue
        elif cfg.templates and templates is not None:
            lines = b.text.split(NL)
            dropped.extend(("template", x) for x in lines if template_key(x) in templates.lines)
            text = NL.join(x for x in lines if template_key(x) not in templates.lines)
            if not text:
                continue
            b = Block(b.kind, text, b.link_ratio)
        filtered.append(b)
    out = filtered

    if cfg.trim_before_h1:
        h1_key = template_key(normalize_block(page_h1)) if page_h1 else ""
        idx = next(
            (i for i, b in enumerate(out) if b.kind == "h1" or (h1_key and template_key(b.text) == h1_key)), None
        )
        if idx and not (cfg.safe_trim and any(_is_long_prose(b.text) for b in out[:idx])):
            dropped.extend(("before_h1", b.text) for b in out[:idx])
            out = out[idx:]
        elif idx is None and page_h1:
            # main-content extraction dropped the title: restore it as the first block (it is page text)
            h1 = normalize_block(page_h1)
            if h1:
                out.insert(0, Block("h1", h1))
    return out, dropped
