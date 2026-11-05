"""Generic main-content extraction: trafilatura (precision) with a selectolax density fallback."""
from __future__ import annotations

import re

try:
    from selectolax.parser import HTMLParser
except ImportError:  # selectolax >= 1.0 bỏ backend modest -> dùng lexbor (API tương thích)
    from selectolax.lexbor import LexborHTMLParser as HTMLParser

from .text import join_paras, norm_para, normalize_text

_JUNK_TAGS = ("script", "style", "noscript", "iframe", "svg", "form", "button", "select",
              "nav", "header", "footer", "aside", "template", "link", "meta")
_JUNK_ATTR = re.compile(
    r"(^|[\s_-])(nav|menu|breadcrumb|footer|header|sidebar|side-bar|share|social|comment|related|"
    r"recommend|advert|ads?|banner|popup|modal|login|tags?|copyright|hot|rank|toolbar|pager|pagination)"
    r"([\s_-]|$)", re.I)
_TITLE_SPLIT = re.compile(r"\s*(?:[|_]|\s-\s|–|—|-{2,})\s*")


def get_title(tree: HTMLParser) -> str:
    for sel in ('meta[property="og:title"]', 'meta[name="twitter:title"]'):
        n = tree.css_first(sel)
        if n is not None:
            t = norm_para(n.attributes.get("content") or "")
            if 3 <= len(t) <= 300:
                return t
    h1 = tree.css_first("h1")
    if h1 is not None:
        t = norm_para(h1.text(separator=" "))
        if 3 <= len(t) <= 300:
            return t
    n = tree.css_first("title")
    if n is not None:
        t = norm_para(n.text())
        parts = [p for p in _TITLE_SPLIT.split(t) if p]
        if parts:
            return parts[0] if len(parts[0]) >= 4 or len(parts) == 1 else max(parts[:2], key=len)
        return t
    return ""


def strip_junk(root) -> None:
    """Remove script/nav/etc. in place (selectolax node or tree)."""
    for tag in _JUNK_TAGS:
        for n in root.css(tag):
            n.decompose()


_BLOCK_TAGS = frozenset((
    "p", "div", "li", "ul", "ol", "dl", "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table",
    "tbody", "thead", "section", "article", "blockquote", "pre", "figcaption", "figure", "header",
    "footer", "main", "aside", "center", "address", "hr", "caption"))
_SKIP_TAGS = frozenset(("script", "style", "noscript", "iframe", "svg", "button", "select", "option",
                        "template", "textarea", "input", "img", "video", "audio", "canvas", "map",
                        "object", "embed", "-comment", "_comment", "head", "link", "meta"))
_HIDDEN = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)


def html_to_paras(node, skip_hidden: bool = True) -> list[str]:
    """Flatten a DOM subtree to paragraph strings: block elements and <br> start a new paragraph,
    inline text is concatenated verbatim, table cells are joined with ' | '."""
    out: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            s = "".join(buf)
            buf.clear()
            if s.strip():
                out.append(s)

    def walk(n, depth=0):
        if depth > 400:
            return
        child = n.child
        while child is not None:
            tag = child.tag
            if tag == "-text":
                buf.append(child.text(deep=False) or "")
            elif tag in _SKIP_TAGS:
                pass
            elif skip_hidden and _HIDDEN.search(child.attributes.get("style") or ""):
                pass
            elif tag == "br":
                flush()
            elif tag in ("td", "th"):
                if buf and "".join(buf).strip():
                    buf.append(" | ")
                walk(child, depth + 1)
            elif tag in _BLOCK_TAGS:
                flush()
                walk(child, depth + 1)
                flush()
            else:
                walk(child, depth + 1)
            child = child.next

    walk(node)
    flush()
    return out


def node_paragraphs(node, block_sel: str = "p, h2, h3, h4, li, tr, blockquote, pre") -> list[str]:
    """Paragraph texts from a container: block-level children if any, else the container text split
    on <br>/newlines."""
    paras: list[str] = []
    blocks = node.css(block_sel)
    if blocks:
        seen = set()
        for b in blocks:
            # skip blocks nested inside another selected block (avoid duplicates)
            par, nested = b.parent, False
            while par is not None and par is not node:
                if par.mem_id in seen:
                    nested = True
                    break
                par = par.parent
            if nested:
                continue
            seen.add(b.mem_id)
            t = b.text(separator="\n" if b.tag in ("pre",) else " ", deep=True)
            if b.tag == "tr":
                t = " | ".join(norm_para(td.text(separator=" ")) for td in b.css("td, th"))
            paras.append(t)
        # also text directly in the container outside blocks is lost; acceptable
    else:
        html = node.html or ""
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
        sub = HTMLParser(html)
        paras = (sub.body or sub.root).text(separator="").split("\n") if (sub.body or sub.root) else []
    return paras


def density_extract(html: str) -> str:
    """Readability-style fallback: pick the element whose <p> children carry the most text."""
    tree = HTMLParser(html)
    strip_junk(tree)
    body = tree.body
    if body is None:
        return ""
    score: dict[int, list] = {}
    for p in body.css("p"):
        t = p.text(separator=" ").strip()
        if len(t) < 10:
            continue
        par = p.parent
        if par is None:
            continue
        ent = score.setdefault(par.mem_id, [par, 0])
        ent[1] += len(t)
    if score:
        best = max(score.values(), key=lambda x: x[1])[0]
        return join_paras(html_to_paras(best))
    # no <p>: the div with most text and low link density
    best, best_len = None, 0
    for d in body.css("div, td, article, section"):
        txt = d.text(separator=" ", deep=True)
        n = len(txt)
        if n < 200:
            continue
        link_len = sum(len(a.text()) for a in d.css("a"))
        eff = n - 2 * link_len
        if eff > best_len:
            best, best_len = d, eff
    return join_paras(html_to_paras(best)) if best is not None else ""


def trafilatura_extract(html: str, url: str | None = None) -> str:
    import trafilatura
    try:
        txt = trafilatura.extract(
            html, url=url, favor_precision=True, include_comments=False, include_tables=True,
            include_images=False, include_links=False, deduplicate=False, output_format="txt",
            with_metadata=False,
        )
    except Exception:
        txt = None
    return normalize_text(txt) if txt else ""


def generic_extract(html: str, url: str | None = None, tree: HTMLParser | None = None,
                    min_chars: int = 80) -> tuple[str, str, str]:
    """-> (title, text, method)"""
    tree = tree or HTMLParser(html)
    title = get_title(tree)
    text = trafilatura_extract(html, url)
    method = "trafilatura"
    if len(text) < min_chars:
        alt = density_extract(html)
        if len(alt) > len(text):
            text, method = alt, "density"
    return title, text, method
