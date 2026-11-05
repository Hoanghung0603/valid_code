"""Paragraph/sentence-aware chunking with exact character offsets.

Invariant (asserted in tests): doc["text"][c["start"]:c["end"]] == c["text"].

Modes (cfg["chunk"]["mode"], str or list -> union of granularities):
  packed     consecutive sentences packed up to `target` size (never above `max` unless a
             single unsplittable piece), soft paragraph boundaries, optional sentence overlap
  sentence   one sentence per chunk (tiny sentences merged with the next one up to `min`)
  paragraph  one paragraph per chunk; over-long paragraphs packed like `packed`
PubMed docs (source == "pubmed"): title (first paragraph) is its own chunk when
`pubmed_title_chunk`, abstract section labels (BACKGROUND:, METHODS: ...) start new paragraphs.
Sizes are in characters (`size_unit: chars`) or approximate tokens (`approx_tokens`:
CJK char = 1, other word = 1.3).
"""
from __future__ import annotations

import re
from typing import Iterable

DEFAULTS = {
    "mode": "packed",
    "size_unit": "chars",
    "target": {"vi": 600, "zh": 300, "en": 800},
    "max": {"vi": 1000, "zh": 500, "en": 1200},
    "min": {"vi": 80, "zh": 40, "en": 100},
    "overlap_sentences": 0,
    "respect_paragraphs": True,
    "para_break_frac": 0.5,   # a paragraph boundary ends a chunk once it reaches this fraction of target
    "pubmed_title_chunk": True,
    "max_chunks_per_doc": 300,
    "min_chunk_chars": 8,
}

# sentence terminators ---------------------------------------------------------
_CLOSE = "\"'”’」』）)】]"
_ZH_END = re.compile(r"[。！？；!?;…]+[" + re.escape(_CLOSE) + r"]*|[.](?=\s)[" + re.escape(_CLOSE) + r"]*")
_CJK_END = re.compile(r"[。！？；]+[" + re.escape(_CLOSE) + r"]*")
_LATIN_END = re.compile(r"[.?!…]+[" + re.escape(_CLOSE) + r"]*(?=\s|$)")
_PARA = re.compile(r"\n[ \t　]*\n\s*")
_LINE = re.compile(r"\n")
_SECTION = re.compile(r"(?<=[\s.])(?=(?:[A-Z][A-Z/&,\- ]{2,40}):\s)")
_SOFT_BREAK = re.compile(r"[,，、:：]\s*|\s+")
_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯]")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _get(cfg: dict, key: str, lang: str):
    v = cfg.get(key, DEFAULTS[key])
    if isinstance(v, dict):
        return v.get(lang, v.get("vi", next(iter(v.values()))))
    return v


def _size_fn(unit: str):
    if unit == "approx_tokens":
        def f(s: str) -> float:
            n_cjk = len(_CJK.findall(s))
            return n_cjk + 1.3 * len(_WORD.findall(_CJK.sub(" ", s)))
        return f
    return len


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def paragraphs(text: str, pubmed: bool = False) -> list[tuple[int, int]]:
    """Paragraph spans (split on blank lines; PubMed section labels also split)."""
    spans = []
    pos = 0
    for m in _PARA.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    if pubmed:
        out = []
        for s, e in spans:
            cuts = [s] + [s + m.start() for m in _SECTION.finditer(text[s:e]) if m.start() > 0] + [e]
            out.extend(zip(cuts[:-1], cuts[1:]))
        spans = out
    res = []
    for s, e in spans:
        s, e = _trim(text, s, e)
        if e > s:
            res.append((s, e))
    return res


def sentences(text: str, s: int, e: int, lang: str) -> list[tuple[int, int]]:
    """Sentence spans inside text[s:e]; newlines are also boundaries."""
    pat = _ZH_END if lang == "zh" else _LATIN_END
    seg = text[s:e]
    cuts = {0, len(seg)}
    for m in pat.finditer(seg):
        cuts.add(m.end())
    for m in _LINE.finditer(seg):
        cuts.add(m.start())
    if lang != "zh":  # mixed docs: CJK terminators always split
        for m in _CJK_END.finditer(seg):
            cuts.add(m.end())
    cuts = sorted(cuts)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        a2, b2 = _trim(text, s + a, s + b)
        if b2 > a2:
            out.append((a2, b2))
    return out


def _hard_split(text: str, s: int, e: int, mx: int, size) -> list[tuple[int, int]]:
    """Split an over-long sentence near `mx` at commas/whitespace (else hard cut)."""
    out = []
    while size(text[s:e]) > mx:
        # find the char position where size reaches mx
        lo, hi = s + 1, e
        while lo < hi:
            mid = (lo + hi) // 2
            if size(text[s:mid]) <= mx:
                lo = mid + 1
            else:
                hi = mid
        limit = max(s + 1, lo - 1)
        window = text[s:limit]
        cut = None
        for m in _SOFT_BREAK.finditer(window):
            if m.end() - 0 > len(window) * 0.5:
                cut = s + m.end()
        if cut is None or cut <= s:
            cut = limit
        a, b = _trim(text, s, cut)
        if b > a:
            out.append((a, b))
        s, _ = _trim(text, cut, e)
        if s >= e:
            break
    a, b = _trim(text, s, e)
    if b > a:
        out.append((a, b))
    return out


def _pack(text: str, units: list[tuple[int, int, int]], target: float, mx: float, mn: float,
          size, overlap: int, respect_par: bool, pmin: float | None = None) -> list[tuple[int, int]]:
    """units: (start, end, paragraph_idx). Returns chunk spans."""
    chunks: list[tuple[int, int]] = []
    pmin = mn if pmin is None else max(mn, pmin)
    i = 0
    n = len(units)
    while i < n:
        j = i
        s0 = units[i][0]
        while j + 1 < n:
            nxt = units[j + 1]
            cur = size(text[s0:units[j][1]])
            if respect_par and nxt[2] != units[j][2] and cur >= pmin:
                break
            if size(text[s0:nxt[1]]) > (target if cur >= mn else mx):
                break
            j += 1
        chunks.append((s0, units[j][1]))
        if j + 1 >= n:
            break
        i = max(i + 1, j + 1 - overlap) if overlap > 0 else j + 1
    # merge a too-small tail into the previous chunk if it fits
    if len(chunks) >= 2 and size(text[chunks[-1][0]:chunks[-1][1]]) < mn \
            and size(text[chunks[-2][0]:chunks[-1][1]]) <= mx and overlap == 0:
        chunks[-2:] = [(chunks[-2][0], chunks[-1][1])]
    return chunks


def chunk_spans(text: str, lang: str, cfg: dict | None = None, pubmed: bool = False,
                mode: str | None = None) -> list[tuple[int, int]]:
    cfg = cfg or {}
    mode = mode or cfg.get("mode", DEFAULTS["mode"])
    size = _size_fn(cfg.get("size_unit", DEFAULTS["size_unit"]))
    target = _get(cfg, "target", lang)
    mx = max(_get(cfg, "max", lang), target)
    mn = min(_get(cfg, "min", lang), target)
    overlap = int(cfg.get("overlap_sentences", DEFAULTS["overlap_sentences"]))
    respect = bool(cfg.get("respect_paragraphs", DEFAULTS["respect_paragraphs"]))
    paras = paragraphs(text, pubmed=pubmed)
    spans: list[tuple[int, int]] = []
    if pubmed and paras and cfg.get("pubmed_title_chunk", DEFAULTS["pubmed_title_chunk"]):
        spans.append(paras[0])
        paras = paras[1:]
    units: list[tuple[int, int, int]] = []
    for pi, (ps, pe) in enumerate(paras):
        for ss, se in sentences(text, ps, pe, lang):
            for a, b in (_hard_split(text, ss, se, mx, size) if size(text[ss:se]) > mx else [(ss, se)]):
                units.append((a, b, pi))
    if mode == "sentence":
        # merge tiny sentences forward (e.g. "Chào bác sĩ.") within a paragraph
        i = 0
        while i < len(units):
            j = i
            while j + 1 < len(units) and units[j + 1][2] == units[i][2] and \
                    size(text[units[i][0]:units[j][1]]) < mn and size(text[units[i][0]:units[j + 1][1]]) <= mx:
                j += 1
            spans.append((units[i][0], units[j][1]))
            i = j + 1
    elif mode == "paragraph":
        for pi in range(len(paras)):
            pu = [u for u in units if u[2] == pi]
            if not pu:
                continue
            if size(text[pu[0][0]:pu[-1][1]]) <= mx:
                spans.append((pu[0][0], pu[-1][1]))
            else:
                spans.extend(_pack(text, pu, target, mx, mn, size, overlap, False))
    elif mode == "packed":
        pmin = float(cfg.get("para_break_frac", DEFAULTS["para_break_frac"])) * target
        spans.extend(_pack(text, units, target, mx, mn, size, overlap, respect, pmin))
    else:
        raise ValueError(f"unknown chunk mode {mode!r}")
    return spans


def chunk_doc(doc: dict, cfg: dict | None = None) -> list[dict]:
    """Chunk one Doc -> list of Chunk dicts (schema in docs/INTERFACES.md).

    `cfg` may be the full config (uses cfg["chunk"]) or the chunk block itself.
    """
    if cfg and "chunk" in cfg and isinstance(cfg["chunk"], dict):
        cfg = cfg["chunk"]
    cfg = cfg or {}
    text = doc.get("text") or ""
    lang = doc.get("lang") or "vi"
    pubmed = doc.get("source") == "pubmed"
    modes = cfg.get("mode", DEFAULTS["mode"])
    modes = [modes] if isinstance(modes, str) else list(modes)
    min_chars = int(cfg.get("min_chunk_chars", DEFAULTS["min_chunk_chars"]))
    cap = int(cfg.get("max_chunks_per_doc", DEFAULTS["max_chunks_per_doc"]))
    seen = set()
    spans = []
    for m in modes:
        for sp in chunk_spans(text, lang, cfg, pubmed=pubmed, mode=m):
            if sp not in seen:
                seen.add(sp)
                spans.append(sp)
    if len(spans) > 1:
        spans = [sp for sp in spans if sp[1] - sp[0] >= min_chars] or spans[:1]
    spans = spans[:cap]
    did = str(doc["doc_id"])
    out = []
    for i, (s, e) in enumerate(spans):
        out.append({"chunk_id": f"{did}#{i}", "doc_id": did, "lang": lang,
                    "source": doc.get("source", "corpus"), "start": int(s), "end": int(e),
                    "text": text[s:e]})
    return out


def chunk_docs(docs: Iterable[dict], cfg: dict | None = None) -> list[dict]:
    out = []
    for d in docs:
        out.extend(chunk_doc(d, cfg))
    return out
