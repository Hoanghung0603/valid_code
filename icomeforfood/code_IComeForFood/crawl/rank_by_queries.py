"""Order a domain's URLs by how well their slugs match the test queries, so a slow crawl gets the useful pages first.

Article URLs carry the title as an unaccented slug (nhathuoclongchau.com.vn/bai-viet/ba-bau-tam-bon-co-an-toan-khong-57524.html).
BM25 over the slugs (unaccented syllables + bigrams) is queried with every test query; the order is a round robin over
queries (each query's best slug, then each query's second best, ...), so every query gets its closest pages early.
Slugs no query reaches come last. Ids already fetched (status.csv of crawl_home.py) are left out.

Usage (env with bm25s):
    python -I crawl/rank_by_queries.py --corpus D:/r2ai_data/raw/links_corpus_vi_en_zh.parquet \
        --queries D:/r2ai_data/raw/query.parquet --domain nhathuoclongchau.com.vn \
        --done D:/r2ai_data/crawl_home/status.csv --out D:/r2ai_data/crawl_home/longchau_ranked.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from urllib.parse import unquote, urlparse

import bm25s
import pyarrow.parquet as pq

_WORD = re.compile(r"[a-z0-9]+")


def unaccent(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def tokens(text: str) -> list[str]:
    words = _WORD.findall(unaccent(text))
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def slug(url: str) -> str:
    last = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])
    last = re.sub(r"\.html?$", "", last)
    return re.sub(r"-\d+$", "", last).replace("-", " ")  # drop the trailing article number


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--done", help="status.csv of crawl_home.py: ids with status ok/http_error are left out")
    ap.add_argument("--depth", type=int, default=200, help="slugs taken per query in the round robin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    done: set[int] = set()
    if args.done:
        with open(args.done, encoding="utf-8") as f:
            done = {int(r["id"]) for r in csv.DictReader(f)
                    if (r.get("id") or "").isdigit() and r.get("status") in ("ok", "http_error")}
    rows = [r for r in pq.read_table(args.corpus, columns=["id", "url"]).to_pylist()
            if (urlparse(r["url"]).hostname or "").lower().removeprefix("www.") == args.domain and r["id"] not in done]
    queries = pq.read_table(args.queries, columns=["query"]).column("query").to_pylist()

    index = bm25s.BM25()
    index.index([tokens(slug(r["url"])) for r in rows], show_progress=False)
    hits, scores = index.retrieve([tokens(q) for q in queries], k=min(args.depth, len(rows)), show_progress=False)

    order: list[int] = []
    taken: set[int] = set()
    for rank in range(hits.shape[1]):
        for qi in range(len(queries)):
            i = int(hits[qi, rank])
            if scores[qi, rank] > 0 and i not in taken:
                taken.add(i)
                order.append(i)
    order += [i for i in range(len(rows)) if i not in taken]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "url"])
        for i in order:
            w.writerow([rows[i]["id"], rows[i]["url"]])
    print(f"{len(rows)} urls ({len(done)} done skipped), {len(taken)} matched by a query -> {args.out}")


if __name__ == "__main__":
    main()
