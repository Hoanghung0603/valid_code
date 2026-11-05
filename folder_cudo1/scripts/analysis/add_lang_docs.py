"""Add the top docs of another language (PubMed en, zh) to an existing submission: docs and whole-doc chunks.

  python scripts/analysis/add_lang_docs.py --base sub-vi-docs41k-win2x10-v1 --channel en_dense_q --k 30 --chunks 20 \
         --name sub-vi-best-en30c20-v1
  python scripts/analysis/allocate_sub.py pack --name sub-vi-best-en30c20-v1
Appends the first k docs of the channel (data/dev/bench/<channel>.parquet, e.g. lang_rank.py output) to every query's
relevant_docs, and the whole text of the first `chunks` of them (PubMed abstracts are short) to relevant_chunks; the
base run's docs and chunks are kept as they are, so the LB difference measures what that language adds. Motivation
(2026-10-24): a teammate's doc-only probe rose from doc recall 0.699 (1,000 vi docs) to 0.885 with 300 en + 300 zh.
--max-docs 200: best + en30 and best + zh30 (up to 230 docs per query) both lost 22% of chunk recall with the base
chunks unchanged, while every earlier run had at most 200 docs (2026-10-25).
"""

from __future__ import annotations

import argparse
import gc
import glob
import json

import pandas as pd
import pyarrow.parquet as pq

from medret.config import get_config
from medret.index.meta import DB_FILE, MetaStore
from medret.submit import iter_submission, validate_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=7000)
    ap.add_argument("--max-docs", type=int, default=0, help="cap on relevant_docs per query (0: none)")
    ap.add_argument("--extra-docs", default=None,
                    help="glob of parsed doc parquets read for texts the meta store lacks (e.g. "
                         "'data/parsed/documents/*/part-gloryvu-20261023-zh.parquet')")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    cfg = get_config()
    ch = pd.read_parquet(cfg.paths.dev_dir / "bench" / f"{a.channel}.parquet")
    ch = ch[ch["rank"] < a.k].sort_values(["query_id", "rank"])
    extra = ch.groupby("query_id")["doc_id"].agg(list).to_dict()
    need = set(ch[ch["rank"] < a.chunks]["doc_id"].tolist())  # --chunks 0: docs only, chunks as the base run
    text = pd.Series(dtype=str)
    if need:
        store = MetaStore(cfg.paths.index_dir / DB_FILE)
        text = store.documents(need, with_text=True).set_index("doc_id")["text"]
        store.con.close()  # frees DuckDB's buffer pool before the 1 GB base is streamed
    if need and a.extra_docs:  # docs imported after the meta store was built (not in it until the next rebuild)
        lack = sorted(need - set(text.index))
        extra_text = [pq.read_table(f, columns=["doc_id", "text"], filters=[("doc_id", "in", lack)]).to_pandas()
                      for f in glob.glob(a.extra_docs)]
        if extra_text:
            text = pd.concat([text, pd.concat(extra_text).set_index("doc_id")["text"]])
        print(f"texts for {len(text) / len(need):.1%} of the {len(need):,} docs that get chunks")
    out = cfg.paths.runs_dir / a.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "submission.json"
    added_docs = added_chunks = 0
    qids = []
    with open(path, "w", encoding="utf-8") as f:  # base order kept (the pack step reorders queries anyway)
        f.write("[")
        for i, x in enumerate(iter_submission(cfg.paths.runs_dir / a.base / "submission.json")):
            qids.append(x["id"])
            have = {int(d) for d in x["relevant_docs"]}
            new = [d for d in extra.get(x["id"], []) if d not in have]
            if a.max_docs:
                new = new[: max(0, a.max_docs - len(x["relevant_docs"]))]
            chunks = list(x["relevant_chunks"])
            for d in new[: a.chunks]:
                t = text.get(d, "")
                if t.strip():
                    chunks.append({"doc_id": int(d), "chunk_text": t[: a.max_chars]})  # verbatim document text
                    added_chunks += 1
            added_docs += len(new)
            item = {"id": x["id"], "relevant_docs": list(x["relevant_docs"]) + [int(d) for d in new],
                    "relevant_chunks": chunks}
            f.write((", " if i else "") + json.dumps(item, ensure_ascii=False))
        f.write("]")
    del text
    gc.collect()
    qids.sort()
    errors = validate_submission(path, qids)
    info = {"base": a.base, "channel": a.channel, "k": a.k, "chunks": a.chunks, "max_chars": a.max_chars,
            "max_docs": a.max_docs, "extra_docs": a.extra_docs}
    (out / "config.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"added {added_docs / len(qids):.1f} docs and {added_chunks / len(qids):.1f} chunks per query; "
          f"validate: {'OK' if not errors else errors[:5]} -> {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
