"""Cross-encoder rerank of (query, doc) pairs on Kaggle 2xT4 (self-contained kernel script; rendered by
scripts/kaggle/launch_rerank.py).

Input : dataset with `pairs-*.parquet` (query_id, doc_id, lang, src_rank; one shard = a block of queries) and
        `queries.parquet` (id, query), plus the team snapshot gloryvu/vibiomir-crawl-snapshot for the page texts:
        `chunks_{vi,zh}-*.parquet` (`context` rows joined in chunk_idx order = the page text) and
        `extra_chunks.parquet` (URL / title stand-ins for pages without text, doc-level scoring only).
Model : Qwen3-Reranker (decoder; score = P("yes") of the final token, the model card's prompt) or any
        AutoModelForSequenceClassification cross-encoder (kind = "seqcls", e.g. BAAI/bge-reranker-v2-m3).
Doc   : "<page title>\\n<page text>", cut to `doc_chars` characters, then the whole input to `max_len` tokens.
Output: per shard `<shard>.scores.parquet` (query_id, doc_id, lang, src_rank, score) and `<shard>.done.json`;
        RUN_SUMMARY.json. Shards in `skip_shards` or already done are not redone. One worker process per GPU takes
        every n-th shard; no shard starts after the session deadline (session_limit - safety).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIG_JSON = "__CONFIG__"  # replaced by the launcher with a JSON string

DEFAULTS: dict = dict(
    run="local", model="Qwen/Qwen3-Reranker-0.6B", kind="qwen3", dtype="fp16", max_len=512, doc_chars=1600,
    budget=16384, instruction="Given a Vietnamese medical question, judge whether the document (in Vietnamese or "
    "Chinese) contains information that answers the question",
    pairs_glob="/kaggle/input/**/pairs-*.parquet", queries_glob="/kaggle/input/**/queries.parquet",
    snapshot_glob="/kaggle/input/**/vibiomir-crawl-snapshot", out_dir="/kaggle/working/out", skip_shards=[],
    gpus=[0, 1], session_limit_s=43200, safety_s=1800, est_shard_s=600.0,
    max_rank={"vi": 100000, "zh": 100000},  # pairs with src_rank >= max_rank[lang] are not scored (larger models)
    min_rank={"vi": 0, "zh": 0},  # pairs with src_rank < min_rank[lang] are not scored (already done by a run)
    part=None,  # [i, n]: this kernel takes the shards whose index % n == i (parallel sessions)
    only_shards=None,  # explicit shard names for this kernel (splits sized to each account's GPU quota)
)
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_JSON != "__CONFIG__":
        cfg.update(json.loads(CONFIG_JSON))
    return cfg


def find(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern, recursive=True))


def make_batches(lens: list[int], budget: int) -> list[list[int]]:
    """Length-sorted batches with at most `budget` padded tokens."""
    batches, cur = [], []
    for i in sorted(range(len(lens)), key=lens.__getitem__):
        if cur and (len(cur) + 1) * lens[i] > budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def doc_texts(cfg: dict, ids: set[int]) -> dict[int, str]:
    import pandas as pd
    import pyarrow.parquet as pq

    own = find(cfg.get("texts_glob", "/kaggle/input/**/texts-*.parquet"))
    if own:  # self-contained pairs dataset (doc_id, text already cut): no access to the team snapshot needed
        t = pd.concat(pd.read_parquet(f) for f in own)
        return {int(d): x for d, x in zip(t["doc_id"], t["text"], strict=True) if int(d) in ids}

    snap = find(cfg["snapshot_glob"])
    root = snap[0] if snap else str(Path(find("/kaggle/input/**/chunks_zh-0000.parquet")[0]).parent)
    want = sorted(ids)
    parts = []
    for f in sorted(glob.glob(f"{root}/chunks_vi-*.parquet") + glob.glob(f"{root}/chunks_zh-*.parquet")):
        parts.append(pq.read_table(f, columns=["doc_id", "chunk_idx", "context", "page_title"],
                                   filters=[("doc_id", "in", want)]).to_pandas())
    ch = pd.concat(parts).sort_values(["doc_id", "chunk_idx"])
    head = ch.groupby("doc_id").agg(title=("page_title", "first"), text=("context", "\n".join))
    out = {int(d): (f"{t}\n{x}" if isinstance(t, str) and t.strip() else x)[: cfg["doc_chars"]]
           for d, t, x in zip(head.index, head["title"], head["text"], strict=True)}
    extra = f"{root}/extra_chunks.parquet"
    if os.path.exists(extra):  # stand-ins for pages without text (e.g. www.120ask.com answers HTTP 521)
        ex = pd.read_parquet(extra)
        idcol = "doc_id" if "doc_id" in ex.columns else "id"
        txt = "context" if "context" in ex.columns else ex.columns[-1]
        for d, t in zip(ex[idcol], ex[txt], strict=True):
            if int(d) in ids and int(d) not in out and isinstance(t, str):
                out[int(d)] = t[: cfg["doc_chars"]]
    return out


def worker(cfg: dict, gpu: int, shards: list[str]) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"  # the orchestrator sets CUDA_VISIBLE_DEVICES per worker
    dtype = torch.float16 if cfg["dtype"] == "fp16" and dev == "cuda" else torch.float32
    out = Path(cfg["out_dir"])
    qs = pd.read_parquet(find(cfg["queries_glob"])[0]).set_index("id")["query"]
    pairs = {}
    for s in shards:
        p = pd.read_parquet(s)
        hi = p["lang"].map(cfg["max_rank"]).fillna(10**9)
        lo = p["lang"].map(cfg["min_rank"]).fillna(0)
        pairs[s] = p[(p["src_rank"] < hi) & (p["src_rank"] >= lo)].reset_index(drop=True)
    # optional text_id column: read the page text of another id (the crawled http/https twin, `exact_equivalent` in
    # the snapshot's corpus_annotated.parquet: 81k zh candidate pairs, all of www.120ask.com, have text only there)
    tid = {s: (p["text_id"] if "text_id" in p.columns else p["doc_id"]).astype("int64") for s, p in pairs.items()}
    texts = doc_texts(cfg, {int(d) for t in tid.values() for d in t})
    tok = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
    if cfg["kind"] == "qwen3":
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype).to(dev).eval()
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        pre, suf = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], torch_dtype=dtype).to(dev).eval()
    t_start = time.time()
    for s in shards:
        name = Path(s).stem
        if (out / f"{name}.done.json").exists():
            continue
        if time.time() - T0 + cfg["est_shard_s"] > cfg["session_limit_s"] - cfg["safety_s"]:
            print(f"[gpu{gpu}] deadline: {name} not started", flush=True)
            break
        t0 = time.time()
        p = pairs[s]
        docs = [texts.get(int(d), "") for d in tid[s]]
        if cfg["kind"] == "qwen3":
            body = [f"<Instruct>: {cfg['instruction']}\n<Query>: {qs[int(q)]}\n<Document>: {t}"
                    for q, t in zip(p["query_id"], docs, strict=True)]
            enc = tok(body, add_special_tokens=False, truncation=True, max_length=cfg["max_len"] - len(pre) - len(suf))
            ids = [pre + e + suf for e in enc["input_ids"]]
        else:
            enc = tok([qs[int(q)] for q in p["query_id"]], docs, truncation="only_second", max_length=cfg["max_len"])
            ids = enc["input_ids"]
        scores = np.zeros(len(ids), dtype=np.float32)

        def score(b: list[int], ids: list = ids, scores: np.ndarray = scores) -> None:
            batch = tok.pad({"input_ids": [ids[i] for i in b]}, padding=True, return_tensors="pt").to(dev)
            try:
                with torch.inference_mode():
                    if cfg["kind"] == "qwen3":
                        # logits of the last position only: the full (batch, len, 151k vocab) tensor ran a T4 out of
                        # memory (2026-10-26)
                        logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
                        two = torch.stack([logits[:, no], logits[:, yes]], dim=1).float()
                        sc = torch.log_softmax(two, dim=1)[:, 1].exp()
                    else:
                        sc = model(**batch).logits.view(-1).float()
                scores[b] = sc.cpu().numpy()
            except torch.OutOfMemoryError:
                del batch
                torch.cuda.empty_cache()
                if len(b) == 1:
                    raise
                score(b[: len(b) // 2])
                score(b[len(b) // 2:])

        for b in make_batches([len(x) for x in ids], cfg["budget"]):
            score(b)
        res = p.assign(score=scores, has_text=[bool(t) for t in docs])
        res.to_parquet(out / f"{name}.scores.parquet", index=False)
        (out / f"{name}.done.json").write_text(json.dumps({"shard": name, "n": len(res), "s": time.time() - t0,
                                                           "no_text": int((~res["has_text"]).sum())}))
        print(f"[gpu{gpu}] {name}: {len(res)} pairs in {time.time() - t0:.0f}s "
              f"({len(res) / (time.time() - t0):.1f}/s; elapsed {time.time() - t_start:.0f}s)", flush=True)


T0 = time.time()


def main() -> None:
    cfg = load_config()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        gpu = int(sys.argv[2])
        worker(cfg, gpu, json.loads(sys.argv[3]))
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51"], check=False)
    shards = [s for s in find(cfg["pairs_glob"]) if Path(s).stem not in set(cfg["skip_shards"])]
    if cfg["part"]:
        i, n = cfg["part"]
        shards = [s for k, s in enumerate(shards) if k % n == i]
    if cfg["only_shards"]:
        shards = [s for s in shards if Path(s).stem in set(cfg["only_shards"])]
    gpus = cfg["gpus"]
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(g), json.dumps(shards[k::len(gpus)])],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "TOKENIZERS_PARALLELISM": "false",
                                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
             for k, g in enumerate(gpus)]
    codes = [p.wait() for p in procs]
    done = sorted(Path(f).stem.replace(".done", "") for f in glob.glob(str(out / "*.done.json")))
    summary = {"run": cfg["run"], "model": cfg["model"], "shards_total": len(shards), "shards_done": len(done),
               "exit_codes": codes, "elapsed_s": time.time() - T0}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
