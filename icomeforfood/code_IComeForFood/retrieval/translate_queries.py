"""Translate Vietnamese queries to Chinese and English with an open LLM (KE_HOACH.md 11, 12.7).

BM25 cannot cross languages, so every query also gets a Chinese and an English version plus a few standard
medical terms per language. The Chinese text keeps the colloquial voice of the question, because most
Chinese docs are patient-doctor Q&A; the terms add the formal names that articles use. English is for
PubMed abstracts once they are crawled.

Output is keyed by the sha1 of the query text, so dev queries and the 1,200 test queries share one file and
reruns skip what is done. The LLM is llama.cpp's llama-server, as in eval/make_devset.py.

Usage:
    python -I retrieval/translate_queries.py --queries D:/r2ai_data/devset/v0/devset.jsonl \
        D:/r2ai_data/raw/query.parquet --out D:/r2ai_data/qtrans/v1/translations.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

SCHEMA = {
    "type": "object",
    "properties": {
        "zh": {"type": "string"},
        "zh_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "en": {"type": "string"},
        "en_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": ["zh", "zh_terms", "en", "en_terms"],
}

PROMPT = """Dịch câu hỏi sức khỏe tiếng Việt dưới đây sang tiếng Trung giản thể và tiếng Anh, để tìm tài liệu y khoa.

Câu hỏi: {query}

Trả lời bằng JSON:
- zh: bản dịch tiếng Trung giản thể, đủ ý, giữ giọng văn nói như người bệnh hỏi bác sĩ trên trang hỏi đáp y tế Trung Quốc. Giữ nguyên số, đơn vị, tên xét nghiệm và tên thuốc.
- zh_terms: 2 đến 8 thuật ngữ y khoa tiếng Trung chuẩn cho các bệnh, triệu chứng, thuốc, xét nghiệm, bộ phận cơ thể mà câu hỏi nói tới (thêm tên gọi khác thường dùng nếu có).
- en: bản dịch tiếng Anh, đủ ý, dùng thuật ngữ y khoa chuẩn.
- en_terms: 2 đến 8 thuật ngữ y khoa tiếng Anh chuẩn (kiểu MeSH) cho các khái niệm đó.
Chỉ dịch, không trả lời câu hỏi, không thêm bệnh mà câu hỏi không nhắc tới.

Ví dụ:
Câu hỏi: Bé nhà em 2 tuổi bị sốt 39 độ kèm nổi mẩn đỏ ở lòng bàn tay, có phải tay chân miệng không ạ?
{{"zh": "我家宝宝2岁，发烧39度，手心起红疹，是不是手足口病啊？", "zh_terms": ["手足口病", "发热", "皮疹", "小儿发烧"], "en": "My 2-year-old child has a fever of 39 degrees with a red rash on the palms. Is it hand, foot and mouth disease?", "en_terms": ["hand, foot and mouth disease", "fever", "exanthema", "child"]}}"""


def key_of(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()


def read_queries(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if p.endswith(".parquet"):
            out += pq.read_table(p, columns=["query"]).column("query").to_pylist()
        else:
            out += [json.loads(l)["query"] for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def chat(server: str, query: str) -> dict:
    body = {"model": "local", "messages": [{"role": "user", "content": PROMPT.format(query=query)}],
            "temperature": 0.0, "max_tokens": 1200, "seed": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode("utf-8"),
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


def load(path: Path) -> dict[str, dict]:
    """Translations by query key; the last line wins."""
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["key"]: r for r in rows}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", required=True, help="jsonl with a 'query' field or parquet")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    done = load(args.out)
    todo = list(dict.fromkeys(q for q in read_queries(args.queries) if key_of(q) not in done))[:args.limit or None]
    print(f"done {len(done)}, todo {len(todo)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for i, (q, res) in enumerate(zip(todo, ex.map(lambda q: chat(args.server, q), todo)), 1):
            f.write(json.dumps({"key": key_of(q), "query": q, **res}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
