"""Text utilities: charset decoding, Unicode/whitespace normalisation, language detection."""
from __future__ import annotations

import re
import unicodedata

# ----------------------------------------------------------------------------- charset

_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.I)
_ALIASES = {
    "gb2312": "gb18030", "gbk": "gb18030", "gb_2312-80": "gb18030", "x-gbk": "gb18030",
    "gb18030": "gb18030", "cp936": "gb18030", "hz-gb-2312": "gb18030",
    "big5": "big5hkscs", "utf8": "utf-8", "unicode": "utf-8", "utf-16": "utf-8",
    "windows-1252": "cp1252", "iso-8859-1": "cp1252", "latin1": "cp1252", "us-ascii": "utf-8",
}


def _norm_enc(name: str | None) -> str | None:
    if not name:
        return None
    n = name.strip().strip("\"'").lower()
    n = _ALIASES.get(n, n)
    import codecs
    try:
        return codecs.lookup(n).name
    except LookupError:
        return None


def sniff_meta_charset(body: bytes) -> str | None:
    m = _META_CHARSET.search(body[:8192])
    return _norm_enc(m.group(1).decode("ascii", "ignore")) if m else None


def decode_html(body: bytes, header_charset: str | None = None) -> tuple[str, str]:
    """Decode HTML bytes -> (text, encoding used).

    Order: BOM -> header charset (unless latin-1 default) -> <meta charset> -> utf-8 -> gb18030
    -> charset_normalizer -> utf-8 with replacement. GBK/GB2312 are decoded as GB18030 (superset).
    A candidate is accepted if it decodes with very few errors.
    """
    if body.startswith(b"\xef\xbb\xbf"):
        return body[3:].decode("utf-8", "replace"), "utf-8"
    cands: list[str] = []
    h = _norm_enc(header_charset)
    meta = sniff_meta_charset(body)
    if h and h != "cp1252":
        cands.append(h)
    if meta:
        cands.append(meta)
    cands += ["utf-8", "gb18030"]
    seen = set()
    for enc in cands:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return body.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            # tolerate a handful of broken bytes (truncated pages, mixed junk)
            try:
                txt = body.decode(enc, "replace")
            except LookupError:
                continue
            if txt.count("�") <= max(3, len(txt) // 2000):
                return txt, enc
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(body[:200_000]).best()
        if best is not None and best.encoding:
            enc = _norm_enc(best.encoding) or best.encoding
            return body.decode(enc, "replace"), enc
    except Exception:
        pass
    return body.decode("utf-8", "replace"), "utf-8?"


# ----------------------------------------------------------------------------- normalisation

_ZW = re.compile("[​‌‍⁠﻿­]")
_WS = re.compile(r"[ \t\r\n\f\v   -     　]+")


def norm_para(s: str) -> str:
    """NFC + remove zero-width + collapse whitespace inside one paragraph."""
    s = unicodedata.normalize("NFC", s)
    s = _ZW.sub("", s)
    return _WS.sub(" ", s).strip()


def join_paras(paras, min_len: int = 1) -> str:
    """Normalise each paragraph, drop empties and consecutive duplicates, join with blank line."""
    out: list[str] = []
    for p in paras:
        p = norm_para(p)
        if len(p) < min_len or not any(ch.isalnum() for ch in p):  # drop "?" / "|" / "..." debris
            continue
        if out and out[-1] == p:
            continue
        out.append(p)
    return "\n\n".join(out)


def normalize_text(text: str) -> str:
    """Split on newlines (trafilatura txt output = one paragraph per line) and re-join."""
    return join_paras(text.splitlines())


# ----------------------------------------------------------------------------- language

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_VI = re.compile(
    r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I
)
_LATIN = re.compile(r"[A-Za-z]")

VI_HOSTS = ("hellobacsi.com", "vinmec.com", "youmed.vn", "benhvienvietduc.org",
            "pharmacity.io", "suckhoeviet.org.vn")


def lang_from_host(host: str) -> str | None:
    host = host.lower()
    if host.endswith(".vn") or any(host.endswith(h) for h in VI_HOSTS):
        return "vi"
    if host.endswith(".cn") or host.endswith(".com.cn"):
        return "zh"
    return None


def detect_lang(text: str, host: str = "") -> str:
    """vi / zh / en. Character heuristics on a prefix, falling back to host TLD."""
    sample = text[:4000]
    n_cjk = len(_CJK.findall(sample))
    n_lat = len(_LATIN.findall(sample))
    n_vi = len(_VI.findall(sample))
    total = n_cjk + n_lat + n_vi
    by_host = lang_from_host(host)
    if total < 20:
        return by_host or "zh"
    if n_cjk / total > 0.3:
        return "zh"
    if n_vi / max(1, n_lat + n_vi) > 0.03:
        return "vi"
    if by_host:
        return by_host
    return "en" if n_lat > n_cjk else "zh"
