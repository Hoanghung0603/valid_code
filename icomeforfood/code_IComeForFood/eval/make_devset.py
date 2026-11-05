"""Build a synthetic dev set from cleaned docs (KE_HOACH.md 6 and 28.2 item 3).

There are no labelled queries, so an open LLM writes Vietnamese questions for passages of our own docs,
and the passage is the positive. Steps, each resumable and writing under --out:

    sample    one section chunk (s400) per doc, stratified by language, capped per site
    generate  the LLM reads the passage with numbered blocks and writes 1-3 colloquial Vietnamese
              questions, each with an intent (the organizers' seven) and the blocks that answer it
    verify    a second call at temperature 0 checks every question: understandable without the passage,
              fully answered by it, and by which blocks
    build     dev_gold.json in the format of eval/score_submission.py, queries, metadata, stats

Gold: relevant_docs = the source doc plus docs with the same text; relevant_chunks = each answering
block of those docs as its own reference chunk. Blocks are the units every chunk config is cut at, so
the gold does not favour one chunk size. Questions are filtered by the LLM, not by a retriever, so the
dev set does not favour BM25 or a dense model.

Known gap: only the source passage is labelled, while other docs may answer the question too. Doc
precision is therefore underestimated; compare systems on recall@k and chunk recall first.

The LLM is served by llama.cpp's llama-server (OpenAI-compatible API), e.g.
    llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 16384 -np 4 --port 8080

Usage:
    python -I eval/make_devset.py sample   --docs D:/r2ai_data/from_tv/sample2_v2/docs_sample2.parquet \
        --chunks D:/r2ai_data/chunks40k/s400 --out D:/r2ai_data/devset/v0 --n-vi 300 --n-zh 200
    python -I eval/make_devset.py generate --out D:/r2ai_data/devset/v0 [--limit 10]
    python -I eval/make_devset.py verify   --out D:/r2ai_data/devset/v0
    python -I eval/make_devset.py build    --docs ... --out D:/r2ai_data/devset/v0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2ai_eval import tokenize  # noqa: E402

QUERIES = Path("D:/r2ai_data/raw/query.parquet")
# the organizers' intent labels and their shares among the test queries (KE_HOACH.md 26)
INTENTS = {"Diagnosis": 31.9, "Therapy": 16.3, "Advice": 12.6, "Caution": 10.3,
           "Metric Interpretation": 9.4, "Outcome": 7.8, "Other": 11.7}
# question length styles, matched to the test queries (median 18 words, 75th percentile 40)
STYLES = [
    (0.50, "short", "ngắn, 6 đến 15 từ, hỏi thẳng một ý"),
    (0.25, "medium", "vừa, 15 đến 35 từ, có thể nêu tuổi, triệu chứng hoặc hoàn cảnh"),
    (0.25, "long", "dài, 40 đến 120 từ: kể hoàn cảnh cá nhân trước (tuổi, triệu chứng, đã khám gì, "
                   "kết quả xét nghiệm nếu có) rồi mới hỏi"),
]
SKIP_ROLES = {"toc", "disclaimer", "reference"}
LANG_NAME = {"vi": "tiếng Việt", "zh": "tiếng Trung", "en": "tiếng Anh"}
# a question that points at the passage instead of standing alone
_REFERS = re.compile(r"bài viết|bài báo|đoạn văn|đoạn trên|văn bản|tài liệu|theo bài|trong bài|nêu trên", re.I)


# ---------------------------------------------------------------- data

def load_docs(path: str) -> dict[int, dict]:
    cols = ["id", "url", "lang", "title", "site", "text", "blocks", "n_chars", "sha1_text", "quality_flags"]
    docs = {}
    for r in pq.read_table(path, columns=cols).to_pylist():
        if r["n_chars"] and r["text"]:
            docs[r["id"]] = r
    return docs


def passage_blocks(doc: dict, start: int, end: int) -> list[dict]:
    """Blocks inside [start, end); headings are context, the rest are numbered from 1."""
    out, n = [], 0
    for b in doc["blocks"] or []:
        if b["char_start"] < start or b["char_end"] > end or b.get("role") in SKIP_ROLES:
            continue
        item = {"type": b["type"], "char_start": b["char_start"], "char_end": b["char_end"],
                "text": doc["text"][b["char_start"]:b["char_end"]]}
        if b["type"] != "heading":
            n += 1
            item["n"] = n
        out.append(item)
    return out


def render(blocks: list[dict]) -> str:
    lines = []
    for b in blocks:
        if "n" in b:
            tag = {"qa_q": "câu hỏi", "qa_a": "trả lời"}.get(b["type"])
            lines.append(f"[{b['n']}]" + (f" ({tag})" if tag else "") + f" {b['text']}")
        else:
            lines.append(f"## {b['text']}")
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- sample

def cmd_sample(args) -> None:
    rng = random.Random(args.seed)
    docs = load_docs(args.docs)
    by_doc: dict[int, list[dict]] = {}
    for f in sorted(Path(args.chunks).glob("*.parquet")):
        for c in pq.read_table(f, columns=["doc_id", "char_start", "char_end", "chunk_type", "n_tokens",
                                           "heading_path"]).to_pylist():
            if c["chunk_type"] != "reference" and c["n_tokens"] >= args.min_units and c["doc_id"] in docs:
                by_doc.setdefault(c["doc_id"], []).append(c)

    picked = []
    for lang, quota in (("vi", args.n_vi), ("zh", args.n_zh), ("en", args.n_en)):
        ids = sorted(d for d in by_doc if docs[d]["lang"] == lang)
        rng.shuffle(ids)
        # English is PubMed only: one site, so no per-site cap
        cap = quota if lang == "en" else max(2, round(quota * args.site_cap))
        per_site: Counter = Counter()
        for d in ids:
            if len([p for p in picked if p["lang"] == lang]) >= quota:
                break
            doc = docs[d]
            if per_site[doc["site"]] >= cap:
                continue
            c = rng.choice(by_doc[d])
            blocks = passage_blocks(doc, c["char_start"], c["char_end"])
            if sum("n" in b for b in blocks) == 0:
                continue
            per_site[doc["site"]] += 1
            style = rng.choices(STYLES, weights=[s[0] for s in STYLES])[0]
            picked.append({"pid": len(picked), "doc_id": d, "lang": lang, "site": doc["site"],
                           "title": doc["title"] or "", "url": doc["url"],
                           "char_start": c["char_start"], "char_end": c["char_end"],
                           "heading_path": c["heading_path"], "style": style[1], "blocks": blocks,
                           "examples": []})

    # style examples: real test queries of the same length band, different for every passage
    real = pq.read_table(QUERIES).column("query").to_pylist()
    band = {"short": [q for q in real if len(q.split()) <= 15],
            "medium": [q for q in real if 15 < len(q.split()) <= 35],
            "long": [q for q in real if len(q.split()) > 35]}
    for p in picked:
        p["examples"] = rng.sample(band[p["style"]], 3) + rng.sample(real, 3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"passages: {len(picked)}", Counter(p["lang"] for p in picked), Counter(p["style"] for p in picked))
    print("sites:", len({p["site"] for p in picked}), Counter(p["site"] for p in picked).most_common(8))


# ---------------------------------------------------------------- LLM

def chat(args, messages: list[dict], schema: dict, temperature: float) -> dict:
    body = {"model": "local", "messages": messages, "temperature": temperature, "top_p": 0.95,
            "max_tokens": 1500, "seed": args.seed,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{args.server}/v1/chat/completions", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
            return json.loads(resp["choices"][0]["message"]["content"])
        except Exception as e:  # noqa: BLE001 - retry server hiccups and malformed JSON alike
            err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed: {err}")


GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "skip": {"type": "boolean"},
        "questions": {"type": "array", "maxItems": 3, "items": {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": list(INTENTS)},
                           "question": {"type": "string"},
                           "paragraphs": {"type": "array", "items": {"type": "integer"}, "minItems": 1,
                                          "maxItems": 4}},
            "required": ["intent", "question", "paragraphs"]}},
    },
    "required": ["skip", "questions"],
}

GEN_PROMPT = """Dưới đây là một đoạn trích từ trang web sức khỏe ({lang_name}). Các đoạn con được đánh số [n].

Tiêu đề trang: {title}
{passage}

Nhiệm vụ: viết tối đa {k} câu hỏi bằng tiếng Việt mà một người Việt (bệnh nhân hoặc người nhà) có thể gõ lên trang hỏi đáp sức khỏe, và đoạn trích trên trả lời được.

Yêu cầu:
- Chỉ hỏi về kiến thức sức khỏe: bệnh, triệu chứng, thuốc, xét nghiệm, chăm sóc, dinh dưỡng. Không hỏi về sự kiện, chính sách, số liệu thống kê, giá cả.
- Câu trả lời phải có trong đoạn trích. Không hỏi chi tiết mà đoạn trích không nói tới.
- Mỗi câu hỏi phải tự hiểu được khi không có đoạn trích: nêu rõ bệnh, triệu chứng, thuốc, xét nghiệm hoặc hoàn cảnh đang hỏi. Không viết "bài viết", "đoạn văn", "theo tài liệu".
- Viết như người thật: từ ngữ đời thường, có thể kể tuổi, triệu chứng, kết quả xét nghiệm, có thể thiếu dấu câu. Không chép nguyên cụm từ của đoạn trích.
- Độ dài: {style}. Chi tiết cá nhân thêm vào không được làm câu hỏi vượt ra ngoài những gì đoạn trích trả lời.
- Mỗi câu hỏi gắn một intent: Diagnosis (hỏi đây là bệnh gì, có phải bị gì không), Therapy (điều trị, dùng thuốc), Advice (nên làm gì, chăm sóc, ăn uống), Caution (có nguy hiểm không, lưu ý, chống chỉ định), Metric Interpretation (đọc chỉ số, kết quả xét nghiệm), Outcome (tiên lượng, có khỏi không, biến chứng), Other. Các câu nên khác intent nếu đoạn trích cho phép.
- "paragraphs": số thứ tự các đoạn con chứa câu trả lời (1 đến 3 số).
- Nếu đoạn trích không có kiến thức sức khỏe mà người bệnh sẽ hỏi (tin thời sự, sự kiện, chính sách, số ca bệnh, quảng cáo, chính trị, danh sách liên kết), trả về skip = true và không có câu hỏi.

Ví dụ câu hỏi thật của người dùng (chỉ để học văn phong, chủ đề không liên quan):
{examples}"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "health_question": {"type": "boolean"},
        "self_contained": {"type": "boolean"},
        "answered": {"type": "string", "enum": ["full", "partial", "no"]},
        "paragraphs": {"type": "array", "items": {"type": "integer"}, "maxItems": 4},
    },
    "required": ["health_question", "self_contained", "answered", "paragraphs"],
}

VERIFY_PROMPT = """Câu hỏi: {question}

Đoạn trích ({lang_name}), các đoạn con được đánh số [n]:
Tiêu đề trang: {title}
{passage}

Trả lời bằng JSON:
- health_question: true nếu đây là câu hỏi về sức khỏe mà người bệnh hay người nhà sẽ hỏi (bệnh, triệu chứng, thuốc, xét nghiệm, chăm sóc, dinh dưỡng); false nếu hỏi về sự kiện, cơ sở y tế, chính sách, số liệu, giá cả.
- self_contained: true nếu người chưa đọc đoạn trích vẫn hiểu câu hỏi đang hỏi về bệnh, triệu chứng, thuốc, xét nghiệm hay hoàn cảnh cụ thể nào; false nếu câu hỏi mơ hồ hoặc nhắc tới "bài viết", "đoạn trên".
- answered: "full" nếu đoạn trích trả lời đủ ý chính của câu hỏi, "partial" nếu chỉ một phần, "no" nếu không trả lời.
- paragraphs: số thứ tự các đoạn con chứa câu trả lời (rỗng nếu answered = "no")."""


def _gen_one(args, p: dict) -> dict:
    prompt = GEN_PROMPT.format(lang_name=LANG_NAME[p["lang"]], title=p["title"], passage=render(p["blocks"]),
                               k=args.k, style=dict((s[1], s[2]) for s in STYLES)[p["style"]],
                               examples="\n".join(f"- {q}" for q in p["examples"]))
    out = chat(args, [{"role": "user", "content": prompt}], GEN_SCHEMA, args.temperature)
    return {"pid": p["pid"], "skip": out["skip"], "questions": out["questions"][:args.k],
            "model": args.model_name}


def _verify_one(args, p: dict, qi: int, q: dict) -> dict:
    prompt = VERIFY_PROMPT.format(question=q["question"], lang_name=LANG_NAME[p["lang"]], title=p["title"],
                                  passage=render(p["blocks"]))
    out = chat(args, [{"role": "user", "content": prompt}], VERIFY_SCHEMA, 0.0)
    return {"pid": p["pid"], "qi": qi, **out, "model": args.model_name}


def _run(path: Path, jobs: list, fn, workers: int) -> None:
    done, t0 = 0, time.time()
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for res in ex.map(lambda j: fn(*j), jobs):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)


def cmd_generate(args) -> None:
    out = Path(args.out)
    passages = read_jsonl(out / "passages.jsonl")
    seen = {r["pid"] for r in read_jsonl(out / "generated.jsonl")}
    todo = [p for p in passages if p["pid"] not in seen][:args.limit or None]
    _run(out / "generated.jsonl", [(args, p) for p in todo], _gen_one, args.workers)


def cmd_verify(args) -> None:
    out = Path(args.out)
    passages = {p["pid"]: p for p in read_jsonl(out / "passages.jsonl")}
    seen = {(r["pid"], r["qi"]) for r in read_jsonl(out / "verified.jsonl")}
    jobs = [(args, passages[g["pid"]], qi, q)
            for g in read_jsonl(out / "generated.jsonl") if not g["skip"]
            for qi, q in enumerate(g["questions"]) if (g["pid"], qi) not in seen]
    _run(out / "verified.jsonl", jobs[:args.limit or None], _verify_one, args.workers)


# ---------------------------------------------------------------- build

def copied_run(question: str, passage: str) -> int:
    """Longest run of consecutive question tokens that also appears in the passage."""
    q, p = tokenize(question, "cjk"), " " + " ".join(tokenize(passage, "cjk")) + " "
    best = 0
    for i in range(len(q)):
        j = i + best + 1
        while j <= len(q) and f" {' '.join(q[i:j])} " in p:
            best = j - i
            j += 1
    return best


def cmd_build(args) -> None:
    out = Path(args.out)
    docs = load_docs(args.docs)
    twins: dict[str, list[int]] = {}
    for d in docs.values():
        twins.setdefault(d["sha1_text"], []).append(d["id"])
    passages = {p["pid"]: p for p in read_jsonl(out / "passages.jsonl")}
    verified = {(v["pid"], v["qi"]): v for v in read_jsonl(out / "verified.jsonl")}

    rows, reasons = [], Counter()
    for g in read_jsonl(out / "generated.jsonl"):
        p = passages[g["pid"]]
        if g["skip"]:
            reasons["skip_not_medical"] += 1
            continue
        numbered = {b["n"]: b for b in p["blocks"] if "n" in b}
        passage_text = " ".join(b["text"] for b in p["blocks"])
        for qi, q in enumerate(g["questions"]):
            v = verified.get((g["pid"], qi))
            text = " ".join(q["question"].split())
            why = None
            if v is None:
                why = "not_verified"
            elif len(text.split()) < 4:
                why = "too_short"
            elif _REFERS.search(text):
                why = "refers_to_passage"
            elif not v.get("health_question", True):
                why = "not_health_question"
            elif not v["self_contained"]:
                why = "not_self_contained"
            elif v["answered"] != "full":
                why = f"answered_{v['answered']}"
            nums = [n for n in (v["paragraphs"] if v and v["paragraphs"] else q["paragraphs"]) if n in numbered]
            if why is None and not nums:
                why = "no_answer_block"
            if why:
                reasons[why] += 1
                continue
            reasons["kept"] += 1
            doc = docs[p["doc_id"]]
            gold_docs = sorted(twins[doc["sha1_text"]])
            gold_blocks = [{"char_start": numbered[n]["char_start"], "char_end": numbered[n]["char_end"],
                            "text": numbered[n]["text"]} for n in sorted(set(nums))]
            rows.append({"source_pid": p["pid"], "source_qi": qi, "query": text, "intent": q["intent"],
                         "style": p["style"], "n_words": len(text.split()),
                         "copied_run": copied_run(text, passage_text), "doc_id": p["doc_id"],
                         "doc_lang": p["lang"], "site": p["site"], "title": p["title"], "url": p["url"],
                         "passage_start": p["char_start"], "passage_end": p["char_end"],
                         "gold_docs": gold_docs, "gold_blocks": gold_blocks, "model": g["model"]})

    # ids from --id-start, so sets built separately can be merged; split by source doc so a doc never sits
    # in both dev and test
    for i, r in enumerate(rows, args.id_start):
        r["id"] = i
        h = int(hashlib.sha1(str(r["doc_id"]).encode()).hexdigest(), 16) % 100
        r["split"] = "test" if h < args.test_pct else "dev"
    gold = [{"id": r["id"], "relevant_docs": r["gold_docs"],
             "relevant_chunks": [{"doc_id": d, "chunk_text": b["text"]} for d in r["gold_docs"]
                                 for b in r["gold_blocks"]],
             "split": r["split"]} for r in rows]
    with open(out / "devset.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        part = [g for g in gold if g["split"] == split]
        (out / f"{split}_gold.json").write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")
    (out / "all_gold.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")

    stats = {"funnel": dict(reasons), "queries": len(rows),
             "split": dict(Counter(r["split"] for r in rows)),
             "doc_lang": dict(Counter(r["doc_lang"] for r in rows)),
             "intent_pct": {k: round(100 * v / len(rows), 1) for k, v in Counter(r["intent"] for r in rows).items()},
             "intent_pct_test_queries": INTENTS,
             "words_median": statistics.median(r["n_words"] for r in rows),
             "words_p75": statistics.quantiles([r["n_words"] for r in rows], n=4)[2],
             "copied_run_ge6": sum(r["copied_run"] >= 6 for r in rows),
             "sites": len({r["site"] for r in rows}),
             "gold_docs_gt1": sum(len(r["gold_docs"]) > 1 for r in rows),
             "gold_blocks_mean": round(statistics.fmean(len(r["gold_blocks"]) for r in rows), 2)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "generate", "verify", "build"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs")
    ap.add_argument("--chunks")
    ap.add_argument("--n-vi", type=int, default=300)
    ap.add_argument("--n-zh", type=int, default=200)
    ap.add_argument("--n-en", type=int, default=0)
    ap.add_argument("--site-cap", type=float, default=0.04, help="max share of a language quota per site")
    ap.add_argument("--min-units", type=int, default=80)
    ap.add_argument("--k", type=int, default=2, help="questions per passage")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--model-name", default="Qwen3.5-9B-Q4_K_M")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--test-pct", type=int, default=20)
    ap.add_argument("--id-start", type=int, default=1)
    args = ap.parse_args()
    {"sample": cmd_sample, "generate": cmd_generate, "verify": cmd_verify, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
