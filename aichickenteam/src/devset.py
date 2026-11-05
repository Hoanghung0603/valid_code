"""Dev-set construction tools.

(a) Synthetic dev set (chunk -> LLM-generated Vietnamese patient question):

    python -m src.devset sample   --src data/chunks --n 600 --out data/dev/synth_prompts.jsonl \
                                  [--per-lang vi=200,zh=200,en=200]
    python -m src.devset generate --prompts data/dev/synth_prompts.jsonl --out data/dev/synth_gen.jsonl
    python -m src.devset build    --gen data/dev/synth_gen.jsonl \
                                  --queries data/dev/synth_queries.jsonl --gold data/dev/synth_gold.json

    ``--src`` may be Chunk parquet (data/chunks) or Doc parquet (data/docs): for docs a
    random paragraph (split on ``\\n\\n``) is used as pseudo-chunk, with offsets.
    ``generate`` lazily imports ``src.llm.generate(prompts, max_new_tokens, json=True)``.
    Gold = that chunk (+ its doc), in submission format.

(b) Pooling for manual labeling:

    python -m src.devset pool --runs a.zip b.json ... --queries q.jsonl --out pool.jsonl \
                              [--depth-docs 10 --depth-chunks 10]
    # edit "label" fields (1 = relevant, 0 = not), then
    python -m src.devset pool2gold --pool pool.jsonl --out gold.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

PROMPT_TEMPLATE = """Bạn là người bệnh/người nhà bệnh nhân Việt Nam đang tìm thông tin y tế.
Đọc đoạn tài liệu y khoa dưới đây (có thể bằng tiếng Việt, tiếng Anh hoặc tiếng Trung) và viết MỘT câu hỏi bằng tiếng Việt, theo giọng văn tự nhiên của người dân (không dùng thuật ngữ quá chuyên môn nếu không cần), mà đoạn tài liệu này trả lời được.
Yêu cầu:
- Không nhắc tới "đoạn văn", "bài viết", "tài liệu", tên tác giả hay số liệu nghiên cứu cụ thể.
- Câu hỏi dài 10–40 từ, có thể kèm bối cảnh ngắn (tuổi, triệu chứng).
- Chỉ trả về JSON: {{"question": "..."}}

Đoạn tài liệu ({lang}):
\"\"\"{text}\"\"\"
"""

LANG_NAMES = {"vi": "tiếng Việt", "en": "tiếng Anh", "zh": "tiếng Trung"}


# ---------------------------------------------------------------- sampling
def _parquet_files(src: str | Path | list) -> list[str]:
    srcs = [src] if isinstance(src, (str, Path)) else list(src)
    files: list[str] = []
    for s in map(Path, srcs):
        files += [str(f) for f in sorted(s.rglob("*.parquet"))] if s.is_dir() else [str(s)]
    return files


def _paragraph_chunk(doc: dict, rng: random.Random, min_chars: int, max_chars: int) -> dict | None:
    text = doc.get("text") or ""
    spans, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    good = [(s, e) for s, e in spans if min_chars <= len(text[s:e].strip()) <= max_chars]
    if not good:
        return None
    s, e = rng.choice(good)
    # trim whitespace but keep offsets exact
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return {"chunk_id": f"{doc['doc_id']}#p{s}", "doc_id": str(doc["doc_id"]), "lang": doc.get("lang"),
            "source": doc.get("source"), "start": s, "end": e, "text": text[s:e]}


def sample_chunks(src, n: int = 500, seed: int = 13, per_lang: dict[str, int] | None = None,
                  min_chars: int = 80, max_chars: int = 1500, oversample: int = 4) -> list[dict]:
    """Randomly sample chunks from Chunk parquet (has chunk_id) or Doc parquet (has text only)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    files = _parquet_files(src)
    if not files:
        raise FileNotFoundError(f"no parquet files under {src}")
    dset = ds.dataset(files, format="parquet")
    names = set(dset.schema.names)
    is_chunk = "chunk_id" in names
    rng = random.Random(seed)
    cols = [c for c in ("chunk_id", "doc_id", "lang", "source", "start", "end", "text", "title")
            if c in names]
    groups = dict(per_lang) if per_lang else {None: n}
    rows = []
    for lang, want in groups.items():  # sample inside each language so rare langs get filled
        flt = (pc.field("lang") == lang) if lang is not None else None
        total = dset.scanner(columns=["doc_id"], filter=flt).count_rows()
        k = min(total, max(want * oversample, want))
        if k <= 0:
            continue
        idx = sorted(rng.sample(range(total), k))
        rows += dset.scanner(columns=cols, filter=flt).take(idx).to_pylist()
    rng.shuffle(rows)
    quota = dict(per_lang) if per_lang else None
    out, seen_docs = [], set()
    for r in rows:
        if r.get("doc_id") is None:
            continue
        r["doc_id"] = str(r["doc_id"])
        if r["doc_id"] in seen_docs:  # one question per doc keeps the set diverse
            continue
        if quota is not None:
            if quota.get(r.get("lang"), 0) <= 0:
                continue
        if is_chunk:
            t = r.get("text") or ""
            if not (min_chars <= len(t.strip()) <= max_chars):
                continue
            c = r
        else:
            c = _paragraph_chunk(r, rng, min_chars, max_chars)
            if c is None:
                continue
        out.append(c)
        seen_docs.add(r["doc_id"])
        if quota is not None:
            quota[r.get("lang")] -= 1
            if all(v <= 0 for v in quota.values()):
                break
        elif len(out) >= n:
            break
    return out


def make_prompt(chunk: dict) -> str:
    return PROMPT_TEMPLATE.format(lang=LANG_NAMES.get(chunk.get("lang"), chunk.get("lang") or "?"),
                                  text=(chunk.get("text") or "").strip())


def write_prompts(chunks: Iterable[dict], out: str | Path, start_id: int = 1) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, start=start_id):
            rec = {"id": i, "chunk_id": c.get("chunk_id"), "doc_id": c["doc_id"], "lang": c.get("lang"),
                   "source": c.get("source"), "start": c.get("start"), "end": c.get("end"),
                   "chunk_text": c["text"], "prompt": make_prompt(c)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# -------------------------------------------------------------- generation
def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_question(out: Any) -> str | None:
    """Extract the question from an LLM output (dict, JSON string, or raw text)."""
    if out is None:
        return None
    if isinstance(out, dict):
        q = out.get("question") or out.get("query") or out.get("q")
        return q.strip() if isinstance(q, str) and q.strip() else None
    s = str(out).strip()
    try:  # shared robust parser (handles fences / truncated JSON) if available
        from src.llm import parse_json
        obj = parse_json(s)
        if isinstance(obj, dict):
            return parse_question(obj)
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return parse_question(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    s = s.strip().strip('"').strip()
    return s if 5 <= len(s) <= 600 and "{" not in s else None


def generate_questions(prompts_path: str | Path, out: str | Path, max_new_tokens: int = 160,
                       batch_size: int = 16, generate_fn=None) -> int:
    """Call src.llm.generate on each prompt; write records + "question" to ``out`` (resumable)."""
    if generate_fn is None:
        try:
            import src.llm as llm_mod  # lazy: heavy module
        except ImportError as e:
            raise RuntimeError("src.llm is not available (needs generate(prompts, max_new_tokens, "
                               "json=True) or get_llm().generate); pass generate_fn=") from e
        if hasattr(llm_mod, "generate"):
            generate_fn = llm_mod.generate
        elif hasattr(llm_mod, "get_llm"):
            generate_fn = llm_mod.get_llm().generate
        else:
            raise RuntimeError("src.llm exposes neither generate() nor get_llm()")
    recs = _read_jsonl(prompts_path)
    out = Path(out)
    done = {r["id"] for r in _read_jsonl(out)} if out.exists() else set()
    todo = [r for r in recs if r["id"] not in done]
    n_ok = 0
    with out.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            outs = generate_fn([r["prompt"] for r in batch], max_new_tokens=max_new_tokens, json=True)
            for r, o in zip(batch, outs):
                q = parse_question(o)
                rec = {k: v for k, v in r.items() if k != "prompt"}
                rec["question"] = q
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += q is not None
            f.flush()
            print(f"[devset] generated {min(i + batch_size, len(todo))}/{len(todo)}", file=sys.stderr)
    return n_ok


def build_synthetic(gen_path: str | Path, queries_out: str | Path, gold_out: str | Path) -> int:
    """gen jsonl -> queries jsonl ({id, query}) + gold json (submission format)."""
    recs = [r for r in _read_jsonl(gen_path) if r.get("question")]
    queries = [{"id": int(r["id"]), "query": r["question"]} for r in recs]
    gold = [{"id": int(r["id"]), "relevant_docs": [str(r["doc_id"])],
             "relevant_chunks": [{"doc_id": str(r["doc_id"]), "chunk_text": r["chunk_text"]}]}
            for r in recs]
    for p in (queries_out, gold_out):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(queries_out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(gold_out, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=1)
    return len(gold)


# ----------------------------------------------------------------- pooling
def pool(runs: dict[str, list[dict]] | list, depth_docs: int = 10, depth_chunks: int = 10,
         queries: dict[int, str] | None = None) -> list[dict]:
    """Merge several systems' submissions into one labeling pool per query.

    ``runs``: {name: submission list} (or a list of submissions, named run0..).
    Returns [{id, query, docs: [{doc_id, systems, best_rank, label}],
    chunks: [{doc_id, chunk_text, systems, best_rank, label}]}], items sorted by
    (#systems desc, best_rank asc). ``label`` is None, to be filled by a human.
    """
    from src.evaluate import norm_text
    if isinstance(runs, list):
        runs = {f"run{i}": r for i, r in enumerate(runs)}
    per_q: dict[int, dict] = {}
    for name, sub in runs.items():
        for e in sub:
            qid = int(e["id"])
            slot = per_q.setdefault(qid, {"docs": {}, "chunks": {}})
            for rank, d in enumerate((e.get("relevant_docs") or [])[:depth_docs], 1):
                it = slot["docs"].setdefault(str(d), {"doc_id": str(d), "systems": [], "best_rank": rank})
                if name not in it["systems"]:
                    it["systems"].append(name)
                it["best_rank"] = min(it["best_rank"], rank)
            for rank, c in enumerate((e.get("relevant_chunks") or [])[:depth_chunks], 1):
                key = (str(c["doc_id"]), norm_text(c.get("chunk_text") or ""))
                it = slot["chunks"].setdefault(key, {"doc_id": key[0], "chunk_text": c["chunk_text"],
                                                     "systems": [], "best_rank": rank})
                if name not in it["systems"]:
                    it["systems"].append(name)
                it["best_rank"] = min(it["best_rank"], rank)
    out = []
    order = lambda it: (-len(it["systems"]), it["best_rank"])  # noqa: E731
    for qid in sorted(per_q):
        slot = per_q[qid]
        out.append({
            "id": qid, "query": (queries or {}).get(qid),
            "docs": [dict(it, label=None) for it in sorted(slot["docs"].values(), key=order)],
            "chunks": [dict(it, label=None) for it in sorted(slot["chunks"].values(), key=order)],
        })
    return out


def pool_to_gold(pool_recs: Iterable[dict], chunk_implies_doc: bool = True) -> list[dict]:
    """Labeled pool -> gold (submission format). label truthy = relevant.

    Unlabeled items (label None) are treated as not relevant. With
    ``chunk_implies_doc`` a relevant chunk makes its doc relevant too.
    """
    gold = []
    for r in pool_recs:
        docs = [d["doc_id"] for d in r.get("docs", []) if d.get("label")]
        chunks = [{"doc_id": c["doc_id"], "chunk_text": c["chunk_text"]}
                  for c in r.get("chunks", []) if c.get("label")]
        if chunk_implies_doc:
            for c in chunks:
                if c["doc_id"] not in docs:
                    docs.append(c["doc_id"])
        gold.append({"id": int(r["id"]), "relevant_docs": docs, "relevant_chunks": chunks})
    return gold


def _load_queries(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        t = pq.read_table(p, columns=["id", "query"])
        return dict(zip(map(int, t.column("id").to_pylist()), t.column("query").to_pylist()))
    return {int(r["id"]): r["query"] for r in _read_jsonl(p)}


def _parse_per_lang(s: str | None) -> dict[str, int] | None:
    if not s:
        return None
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dev set tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--src", default="data/chunks")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--per-lang", help="e.g. vi=200,zh=200,en=100")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--min-chars", type=int, default=80)
    s.add_argument("--max-chars", type=int, default=1500)
    s.add_argument("--start-id", type=int, default=1)
    s.add_argument("--out", default="data/dev/synth_prompts.jsonl")
    g = sub.add_parser("generate")
    g.add_argument("--prompts", default="data/dev/synth_prompts.jsonl")
    g.add_argument("--out", default="data/dev/synth_gen.jsonl")
    g.add_argument("--max-new-tokens", type=int, default=160)
    g.add_argument("--batch-size", type=int, default=16)
    b = sub.add_parser("build")
    b.add_argument("--gen", default="data/dev/synth_gen.jsonl")
    b.add_argument("--queries", default="data/dev/synth_queries.jsonl")
    b.add_argument("--gold", default="data/dev/synth_gold.json")
    p = sub.add_parser("pool")
    p.add_argument("--runs", nargs="+", required=True, help="submission .json/.zip files")
    p.add_argument("--queries")
    p.add_argument("--depth-docs", type=int, default=10)
    p.add_argument("--depth-chunks", type=int, default=10)
    p.add_argument("--out", required=True)
    pg = sub.add_parser("pool2gold")
    pg.add_argument("--pool", required=True)
    pg.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "sample":
        chunks = sample_chunks(a.src, a.n, a.seed, _parse_per_lang(a.per_lang), a.min_chars, a.max_chars)
        n = write_prompts(chunks, a.out, a.start_id)
        print(f"wrote {n} prompts to {a.out}")
    elif a.cmd == "generate":
        n = generate_questions(a.prompts, a.out, a.max_new_tokens, a.batch_size)
        print(f"generated {n} questions -> {a.out}")
    elif a.cmd == "build":
        n = build_synthetic(a.gen, a.queries, a.gold)
        print(f"wrote {n} queries -> {a.queries}, gold -> {a.gold}")
    elif a.cmd == "pool":
        from src.evaluate import load_json_any
        runs = {Path(r).stem: load_json_any(r) for r in a.runs}
        recs = pool(runs, a.depth_docs, a.depth_chunks, _load_queries(a.queries))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote pool for {len(recs)} queries -> {a.out}")
    elif a.cmd == "pool2gold":
        gold = pool_to_gold(_read_jsonl(a.pool))
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(gold, f, ensure_ascii=False, indent=1)
        print(f"wrote gold for {len(gold)} queries -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
