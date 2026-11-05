"""Cut cleaned documents (docs schema v2) into retrieval chunks (PHAN_CONG.md 2.2, KE_HOACH.md 12.3).

Every chunk is a contiguous slice of the document cut at block boundaries, so
chunk.text == doc.text[char_start:char_end] and the text stays verbatim. Granularity comes from a
config file (chunking/configs/*.json):
    paragraph  consecutive blocks of one section packed up to `target_units`
    section    a whole section (heading + body) up to `max_units`, split into balanced parts above that
Q&A blocks (qa_q / qa_a) give one `qa` chunk per question with its answers. Reference lists get chunks of
their own; table-of-contents and disclaimer blocks are left out. A block longer than `max_units` is cut at
sentence ends into `window` chunks, but only for languages whose words are space-separated: the official
scorer is assumed to split on whitespace, so a Chinese paragraph is a single token and a chunk holding
half of it would match nothing. Chinese blocks are cut only above `hard_max_units`.

Units: tokens of eval/r2ai_eval.py tokenize(mode="cjk"), i.e. one unit per Vietnamese syllable, English
word or Chinese character after the organizers' normalization. Stored as `n_tokens`.

Usage:
    python -I chunking/chunker.py chunk --config chunking/configs/p150.json --docs D:/r2ai_data/docs --out D:/r2ai_data/chunks
    python -I chunking/chunker.py check --docs D:/r2ai_data/docs --chunks D:/r2ai_data/chunks/p150
--docs takes docs_XXXXX.parquet files, folders of them, or folders of <id>.json docs (gold outputs).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from r2ai_eval import tokenize  # noqa: E402

CHUNK_SCHEMA = pa.schema([
    ("chunk_id", pa.string()), ("doc_id", pa.int64()), ("chunk_order", pa.int32()), ("text", pa.string()),
    ("char_start", pa.int64()), ("char_end", pa.int64()), ("heading_path", pa.string()),
    ("chunk_type", pa.string()), ("lang", pa.string()), ("n_tokens", pa.int32()), ("chunk_config", pa.string()),
])
# sentence ends: Latin punctuation followed by space, CJK punctuation anywhere
_SENT_END = re.compile(r"(?<=[.!?…;])\s+|(?<=[。！？；])")
# one unit as counted by tokenize(mode="cjk"): a CJK character or a run of other non-space characters
_CJK_CHARS = "㐀-䶿一-鿿豈-﫿　-〿＀-￯"
_UNIT = re.compile(rf"[{_CJK_CHARS}]|[^\s{_CJK_CHARS}]+")


@dataclass(frozen=True)
class Config:
    name: str
    mode: str                  # "paragraph" or "section"
    target_units: int          # size a chunk is packed up to
    max_units: int             # never exceed by packing; blocks above it are cut (see split_langs). A window
                               # can pass it by the few units of the headings that open it
    min_units: int             # smaller chunks are merged into a neighbour of the same section
    overlap_units: int         # overlap between window pieces of one long block
    break_level: int           # section mode: headings at or above this level open a new section
    heading_max_units: int     # a level>=2 "heading" longer than this is body text (sapo tagged as h2)
    split_langs: tuple         # languages whose long blocks may be cut at sentence ends
    hard_max_units: int        # cut any block above this, whatever the language
    skip_roles: tuple          # roles never put into a chunk
    own_roles: tuple           # roles kept in chunks of their own

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        for f in fields(cls):
            if f.type == "tuple":
                raw[f.name] = tuple(raw[f.name])
        cfg = cls(**raw)
        if cfg.mode not in ("paragraph", "section"):
            raise ValueError(f"unknown mode {cfg.mode!r}")
        if not 0 < cfg.min_units <= cfg.target_units <= cfg.max_units <= cfg.hard_max_units:
            raise ValueError("need 0 < min_units <= target_units <= max_units <= hard_max_units")
        return cfg


def count_units(text: str) -> int:
    return len(tokenize(text, "cjk"))


# ---------- blocks -> sections ----------

def _blocks(doc: dict) -> list[dict]:
    bl = doc.get("blocks")
    if isinstance(bl, str):
        bl = json.loads(bl)
    return list(bl or [])


def _kind(b: dict, units: int, cfg: Config) -> str:
    role, typ, level = b.get("role"), b.get("type"), b.get("level") or 0
    if role in cfg.skip_roles:
        return "skip"
    if role in cfg.own_roles:
        return "own"
    if typ == "qa_q":
        return "qa_q"
    if typ == "heading" and role != "lead" and (level <= 1 or units <= cfg.heading_max_units):
        return "heading"
    return "body"


def _path(title: str, stack: list[tuple[int, str]]) -> str:
    parts = [title] if title else []
    for _, h in stack:
        if not parts or h != parts[-1]:
            parts.append(h)
    return " > ".join(parts)


def _sections(blocks: list[dict], units: list[int], kinds: list[str], title: str, cfg: Config) -> list[dict]:
    """Runs of consecutive blocks. A run never crosses a skipped block, so its text is one slice."""
    secs: list[dict] = []
    stack: list[tuple[int, str]] = []
    cur = None

    def open_(kind: str, path: str):
        nonlocal cur
        cur = {"kind": kind, "path": path, "items": []}
        secs.append(cur)

    for i, (b, k) in enumerate(zip(blocks, kinds)):
        if k == "skip":
            cur = None
            continue
        if k == "heading":
            level = max(1, b.get("level") or 1)
            stack[:] = [(lv, h) for lv, h in stack if lv < level]
            if level > 1 or b["text"].strip() != title:
                stack.append((level, b["text"].strip()))
            new = (cfg.mode == "paragraph" or level <= cfg.break_level or cur is None
                   or cur["kind"] in ("own", "qa"))
            if new:
                open_("normal", _path(title, stack))
        elif k == "qa_q":
            open_("qa", _path(title, stack))
        elif k == "own":
            if cur is None or cur["kind"] != "own":
                open_("own", _path(title, stack))
        elif cur is None or cur["kind"] == "own":
            open_("normal", _path(title, stack))
        cur["items"].append(i)
    return secs


# ---------- sections -> chunks ----------

def _pieces(text: str, base: int, lang: str, cfg: Config) -> list[tuple[int, int]]:
    """Sentence spans of one block, as absolute offsets, each at most max_units."""
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    out = []
    for s, e in spans:
        piece = text[s:e]
        if count_units(piece) <= cfg.max_units:
            out.append((s, e))
            continue
        # a sentence longer than max_units (no punctuation, or Chinese text without spaces):
        # cut between units, i.e. between words or between CJK characters
        toks = [(m.start() + s, m.end() + s) for m in _UNIT.finditer(piece)]
        step = cfg.target_units if lang in cfg.split_langs else cfg.max_units
        for j in range(0, len(toks), step):
            out.append((toks[j][0], toks[min(j + step, len(toks)) - 1][1]))
    return [(base + s, base + e) for s, e in out if e > s]


def _windows(pieces: list[tuple[int, int]], text: str, cfg: Config) -> list[tuple[int, int]]:
    """Pack sentence pieces into overlapping windows of about target_units."""
    sizes = [count_units(text[s:e]) for s, e in pieces]
    out, i = [], 0
    while i < len(pieces):
        j, total = i, 0
        while j < len(pieces) and (j == i or total + sizes[j] <= cfg.target_units):
            total += sizes[j]
            j += 1
        out.append((pieces[i][0], pieces[j - 1][1]))
        if j >= len(pieces):
            break
        back, k = 0, j
        while k - 1 > i and back + sizes[k - 1] <= cfg.overlap_units:
            k -= 1
            back += sizes[k]
        i = k
    return out


def _split_items(items: list[int], units: list[int], cfg: Config, limit: int) -> list[list[int]]:
    """Greedy packing of whole blocks up to `limit` units (a lone oversized block stays alone)."""
    groups, cur, total = [], [], 0
    for i in items:
        if cur and total + units[i] > limit:
            groups.append(cur)
            cur, total = [], 0
        cur.append(i)
        total += units[i]
    if cur:
        groups.append(cur)
    return groups


def _balanced(items: list[int], units: list[int], cfg: Config) -> list[list[int]]:
    """Section mode: split an oversized section into the fewest parts of similar size."""
    total = sum(units[i] for i in items)
    if total <= cfg.max_units:
        return [items]
    n = -(-total // cfg.target_units)
    return _split_items(items, units, cfg, max(cfg.target_units, -(-total // n)))


def chunk_doc(doc: dict, cfg: Config) -> list[dict]:
    text = doc.get("text") or ""
    blocks = _blocks(doc)
    if not text.strip() or not blocks:
        return []
    for b in blocks:
        if text[b["char_start"]:b["char_end"]] != b["text"]:
            raise ValueError(f"doc {doc.get('id')}: block offsets do not match text at {b['char_start']}")
    lang = doc.get("lang") or ""
    title = (doc.get("title") or "").strip()
    units = [count_units(b["text"]) for b in blocks]
    kinds = [_kind(b, u, cfg) for b, u in zip(blocks, units)]

    spans: list[dict] = []   # {start, end, type, path, sec}

    def emit(start: int, end: int, typ: str, path: str, sec: int):
        spans.append({"start": start, "end": end, "type": typ, "path": path, "sec": sec})

    for si, sec in enumerate(_sections(blocks, units, kinds, title, cfg)):
        items = sec["items"]
        base_type = "qa" if sec["kind"] == "qa" else cfg.mode
        # long blocks are cut out first; the blocks between them are packed whole
        run: list[int] = []

        def flush_run():
            if not run:
                return
            # a question stays with its answers up to max_units in both modes
            if cfg.mode == "section" or sec["kind"] == "qa":
                groups = _balanced(run, units, cfg)
            else:
                groups = _split_items(run, units, cfg, cfg.target_units)
            for g in groups:
                emit(blocks[g[0]]["char_start"], blocks[g[-1]]["char_end"], base_type, sec["path"], si)
            run.clear()

        for i in items:
            cuttable = units[i] > cfg.max_units and (lang in cfg.split_langs or units[i] > cfg.hard_max_units)
            if not cuttable:
                run.append(i)
                continue
            # headings right before the long block open its first window
            lead_start = None
            while run and kinds[run[-1]] == "heading":
                lead_start = blocks[run.pop()]["char_start"]
            flush_run()
            b = blocks[i]
            wins = _windows(_pieces(b["text"], b["char_start"], lang, cfg), text, cfg)
            for w, (s, e) in enumerate(wins):
                emit(lead_start if (w == 0 and lead_start is not None) else s, e, "window", sec["path"], si)
        flush_run()

    spans = _merge_small(spans, text, cfg)
    out = []
    for order, sp in enumerate(spans):
        chunk_text = text[sp["start"]:sp["end"]]
        out.append({
            "chunk_id": f"{doc['id']}_{order}", "doc_id": int(doc["id"]), "chunk_order": order,
            "text": chunk_text, "char_start": sp["start"], "char_end": sp["end"], "heading_path": sp["path"],
            "chunk_type": sp["type"], "lang": lang, "n_tokens": count_units(chunk_text), "chunk_config": cfg.name,
        })
    return out


def _merge_small(spans: list[dict], text: str, cfg: Config) -> list[dict]:
    """Merge neighbouring chunks that carry too little on their own.

    - a chunk that is only a heading joins the chunk after it (the path becomes the later one);
    - a chunk below min_units joins its neighbour when both are in the same section, or are sibling
      sections under the same parent heading (path = the common parent).
    Windows of a cut block are never merged, and the result never exceeds max_units.
    """
    def units(sp: dict) -> int:
        return count_units(text[sp["start"]:sp["end"]])

    out: list[dict] = []
    for sp in spans:
        sp = {**sp, "units": units(sp)}
        sp["heading_only"] = sp["type"] != "window" and _only_heading(text[sp["start"]:sp["end"]], sp["path"])
        prev = out[-1] if out else None
        if (prev and "window" not in (prev["type"], sp["type"]) and not text[prev["end"]:sp["start"]].strip()
                and prev["units"] + sp["units"] <= cfg.max_units):
            siblings = prev["sec"] == sp["sec"] or _parent(prev["path"]) == _parent(sp["path"])
            if prev["heading_only"]:
                out[-1] = {**sp, "start": prev["start"], "units": prev["units"] + sp["units"]}
                continue
            if siblings and min(prev["units"], sp["units"]) < cfg.min_units and not sp["heading_only"]:
                path = prev["path"] if prev["sec"] == sp["sec"] else _parent(prev["path"])
                out[-1] = {**prev, "end": sp["end"], "path": path, "units": prev["units"] + sp["units"],
                           "heading_only": False}
                continue
        out.append(sp)
    return out


def _parent(path: str) -> str:
    return path.rsplit(" > ", 1)[0] if " > " in path else ""


def _only_heading(chunk_text: str, path: str) -> bool:
    # the chunk is just the last heading of its own path: a heading with no body
    return bool(path) and chunk_text.strip() == path.rsplit(" > ", 1)[-1]


# ---------- I/O ----------

def iter_doc_sources(paths: list[str]):
    """Yield (stem, docs) per input file or JSON folder."""
    for p in map(Path, paths):
        if p.is_file() and p.suffix == ".parquet":
            yield p.stem, pq.read_table(p).to_pylist()
        elif p.is_dir() and any(p.glob("*.parquet")):
            for f in sorted(p.glob("*.parquet")):
                yield f.stem, pq.read_table(f).to_pylist()
        elif p.is_dir():
            docs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(p.glob("*.json"))]
            yield f"{p.parent.name}_{p.name}", [d for d in docs if isinstance(d, dict) and "text" in d]
        else:
            raise FileNotFoundError(p)


def cmd_chunk(cfg: Config, doc_paths: list[str], out_root: Path) -> None:
    out_dir = out_root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, docs in iter_doc_sources(doc_paths):
        rows, errors = [], 0
        for d in docs:
            try:
                rows.extend(chunk_doc(d, cfg))
            except ValueError as e:
                errors += 1
                print("SKIP", e)
        name = "chunks_" + stem.removeprefix("docs_")
        pq.write_table(pa.Table.from_pylist(rows, schema=CHUNK_SCHEMA), out_dir / f"{name}.parquet")
        print(f"{stem}: {len(docs)} docs -> {len(rows)} chunks ({errors} docs skipped) -> {out_dir / name}.parquet")


def cmd_check(doc_paths: list[str], chunk_dir: Path) -> int:
    """Invariant (text == doc slice), unique ids, contiguous order, and coverage of every kept block."""
    chunks: dict[int, list[dict]] = {}
    for f in sorted(chunk_dir.glob("chunks_*.parquet")):
        for c in pq.read_table(f).to_pylist():
            chunks.setdefault(c["doc_id"], []).append(c)
    cfg_name = next((cs[0]["chunk_config"] for cs in chunks.values()), "?")
    cfg = Config.load(Path(__file__).parent / "configs" / f"{cfg_name}.json")
    bad = {"text_mismatch": 0, "bad_order": 0, "dup_id": 0, "uncovered_blocks": 0, "docs_without_chunks": 0}
    seen_ids, n_docs, examples = set(), 0, []
    stats: dict[tuple, list[int]] = {}
    for _, docs in iter_doc_sources(doc_paths):
        for d in docs:
            n_docs += 1
            cs = sorted(chunks.get(int(d["id"]), []), key=lambda c: c["chunk_order"])
            text = d.get("text") or ""
            if not cs:
                if text.strip():
                    bad["docs_without_chunks"] += 1
                continue
            if [c["chunk_order"] for c in cs] != list(range(len(cs))):
                bad["bad_order"] += 1
            for c in cs:
                if c["chunk_id"] in seen_ids:
                    bad["dup_id"] += 1
                seen_ids.add(c["chunk_id"])
                if text[c["char_start"]:c["char_end"]] != c["text"]:
                    bad["text_mismatch"] += 1
                    examples.append(c["chunk_id"])
                stats.setdefault((c["lang"], c["chunk_type"]), []).append(c["n_tokens"])
            covered = sorted((c["char_start"], c["char_end"]) for c in cs)
            for b in _blocks(d):
                if b.get("role") in cfg.skip_roles or not b["text"].strip():
                    continue
                if not _covered(b["char_start"], b["char_end"], covered, text):
                    bad["uncovered_blocks"] += 1
                    if len(examples) < 10:
                        examples.append(f"{d['id']}@{b['char_start']}")
    print(f"docs {n_docs} | docs with chunks {len(chunks)} | chunks {sum(map(len, chunks.values()))}")
    print("problems:", bad, "| examples:", examples[:10])
    for (lang, typ), v in sorted(stats.items()):
        v = sorted(v)
        print(f"  {lang:3s} {typ:9s} n {len(v):8d}  units median {statistics.median(v):6.0f}  p90 {v[int(len(v) * .9)]:6d}"
              f"  max {v[-1]:6d}  >max_units {sum(x > cfg.max_units for x in v)}")
    return sum(bad.values())


def _covered(s: int, e: int, spans: list[tuple[int, int]], text: str) -> bool:
    """Every non-space character of text[s:e] lies inside some chunk (windows meet at a space)."""
    pos = s
    for cs, ce in spans:
        if cs > pos and text[pos:min(cs, e)].strip():
            break
        if ce > pos:
            pos = ce
            if pos >= e:
                return True
    return pos >= e or not text[pos:e].strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["chunk", "check"])
    ap.add_argument("--config", type=Path)
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, help="chunk: output root, chunks go to <out>/<config name>/")
    ap.add_argument("--chunks", type=Path, help="check: folder of chunks_*.parquet")
    args = ap.parse_args()
    if args.cmd == "chunk":
        cmd_chunk(Config.load(args.config), args.docs, args.out)
    else:
        sys.exit(1 if cmd_check(args.docs, args.chunks) else 0)


if __name__ == "__main__":
    main()
