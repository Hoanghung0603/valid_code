"""Retrieval v1 (KE_HOACH.md 11, 28.2): candidates per document language, fused per language, reranked by
bge-reranker-v2-m3, submitted as p150 chunks or the s400 sections that hold them.

    cands    per-language candidate chunk lists for each query -> <out>/cands.pkl
               vi: BM25 (syllables + bigrams) and BGE-M3 dense on Vietnamese chunks, Vietnamese query
               zh, en: BM25 (jieba / words) with the translated query plus its terms, and BGE-M3 dense with
                   the Vietnamese query and with the translation
    fuse     compares ways to merge the lists: doc hit@k overall and by doc language -> <out>/fuse.json
    rerank   cross-encoder scores for the best chunks of the top docs of each language -> <out>/rerank.pkl
    eval     docs ranked by their best reranked chunk; submissions with p150 chunks or the s400 chunks
             that hold them -> <out>/report.json
    encode   BGE-M3 dense vectors of a chunk file -> <index>/dense.npy + chunk_ids.json

Both models run on plain transformers (no FlagEmbedding, whose import pulls its training extras), with the
same pooling and scoring as FlagEmbedding: BGE-M3 dense = normalized CLS vector, reranker = raw logit.

A plain RRF over all lists lets Vietnamese chunks collect votes from both BM25 and dense while Chinese
chunks get dense only, so Chinese docs sink (baseline: hit@100 0.71 -> 0.61). Here each language is fused
on its own and the reranker, which reads query and chunk together, orders the union.

--chunks, --s400 and --index take several folders (e.g. the 40K sample and PubMed); they are concatenated
in the order given, so the chunk and index folders must be listed in the same order.

Usage (env with torch + transformers + bm25s + jieba):
    python -I retrieval/pipeline.py cands  --out D:/r2ai_data/runs/v1_p150
    python -I retrieval/pipeline.py fuse   --out D:/r2ai_data/runs/v1_p150
    python -I retrieval/pipeline.py rerank --out D:/r2ai_data/runs/v1_p150
    python -I retrieval/pipeline.py eval   --out D:/r2ai_data/runs/v1_p150
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from baseline import BGE_M3_FILES, RRF_K, doc_list, vi_tokens  # noqa: E402
from r2ai_eval import normalize  # noqa: E402
from score_submission import score_query  # noqa: E402
from translate_queries import key_of, load as load_translations  # noqa: E402

DATA = Path("D:/r2ai_data")
_WORD = re.compile(r"\w")
_WORDS = re.compile(r"\w+")
LANGS = ("vi", "zh", "en")
TRANSLATED = ("zh", "en")       # pools searched with the translated query too
# the PyTorch weights only, as for BGE-M3
RERANKER_FILES = ["config.json", "model.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json"]
# --reranker name -> (HF repo, kind). All open, <=15B, released before 01/08/2026 (checked 24/10).
RERANKERS = {
    "bge": ("BAAI/bge-reranker-v2-m3", "seqcls"),
    "gte": ("Alibaba-NLP/gte-multilingual-reranker-base", "seqcls"),
    "qwen3-0.6b": ("Qwen/Qwen3-Reranker-0.6B", "qwen3"),
    "qwen3-4b": ("Qwen/Qwen3-Reranker-4B", "qwen3"),
}
QWEN3_INSTRUCT = ("Given a Vietnamese medical question, judge whether the passage (Vietnamese, English or Chinese) "
                  "contains information that answers it")
HITS_AT = (1, 5, 10, 20, 50, 100)


# ---------------------------------------------------------------- data

def load_chunks(paths: list[Path]) -> dict[str, list]:
    cols = ["chunk_id", "doc_id", "text", "lang", "heading_path", "char_start", "char_end"]
    files = [f for p in paths for f in (sorted(p.glob("*.parquet")) if p.is_dir() else [p])]
    t = pq.read_table(files, columns=cols)
    return {c: t.column(c).to_pylist() for c in cols}


def load_index(paths: list[Path], chunk_ids: list[str]) -> np.ndarray:
    ids = [i for p in paths for i in json.loads((p / "chunk_ids.json").read_text(encoding="utf-8"))]
    assert ids == chunk_ids, "index was built from other chunk files, or listed in another order"
    return np.concatenate([np.load(p / "dense.npy") for p in paths])


def chunk_text(c: dict, i: int) -> str:
    """What the retrievers and the reranker read: heading path, then the chunk."""
    head = c["heading_path"][i] or ""
    return f"{head}\n{c['text'][i]}" if head else c["text"][i]


def zh_tokens(text: str) -> list[str]:
    import jieba
    return [t for t in jieba.lcut_for_search(normalize(text)) if _WORD.search(t)]


def en_tokens(text: str) -> list[str]:
    return _WORDS.findall(normalize(text))


TOKENIZERS = {"vi": vi_tokens, "zh": zh_tokens, "en": en_tokens}


def load_queries(args) -> tuple[list[dict], dict[int, dict]]:
    gold = {}
    for split in (["dev", "test"] if args.split == "all" else [args.split]):
        for g in json.loads((args.devset / f"{split}_gold.json").read_text(encoding="utf-8")):
            gold[g["id"]] = g
    rows = [json.loads(l) for l in (args.devset / "devset.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return [q for q in rows if q["id"] in gold], gold


def lang_query(q: dict, tr: dict, lang: str, with_terms: bool) -> str:
    """The query a pool's BM25 reads: the Vietnamese text, or the translation plus its terms."""
    if lang not in TRANSLATED:
        return q["query"]
    return tr[lang] + (" " + " ".join(tr[f"{lang}_terms"]) if with_terms else "")


# ---------------------------------------------------------------- models

class DenseEncoder:
    """BGE-M3 dense vectors: CLS token of the last layer, L2-normalized, fp16."""

    def __init__(self, device: str = "cuda"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        path = snapshot_download("BAAI/bge-m3", allow_patterns=BGE_M3_FILES)
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=torch.float16).to(device).eval()

    def encode(self, texts: list[str], batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        torch = self.torch
        order = np.argsort([-len(t) for t in texts])  # long first: padding waste stays low
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float16)
        with torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                idx = order[s:s + batch_size]
                b = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(self.device)
                v = self.model(**b).last_hidden_state[:, 0].float()
                out[idx] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        return out


class CrossEncoder:
    """Sequence-classification reranker (bge-reranker-v2-m3, gte-multilingual-reranker): raw relevance logit."""

    def __init__(self, max_length: int = 512, device: str = "cuda", repo: str = "BAAI/bge-reranker-v2-m3"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if repo == "BAAI/bge-reranker-v2-m3":
            path = snapshot_download(repo, allow_patterns=RERANKER_FILES)
        else:
            path = snapshot_download(repo, ignore_patterns=["*.bin", "*.onnx", "onnx/*", "*.gguf"])
        self.torch, self.device, self.max_length = torch, device, max_length
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, dtype=torch.float16, trust_remote_code=True).to(device).eval()

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
        torch = self.torch
        order = np.argsort([-(len(q) + len(p)) for q, p in pairs])
        out = np.empty(len(pairs), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(pairs), batch_size):
                idx = order[s:s + batch_size]
                b = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True,
                             truncation="longest_first", max_length=self.max_length, return_tensors="pt").to(self.device)
                out[idx] = self.model(**b).logits.view(-1).float().cpu().numpy()
        return out.tolist()


class Qwen3Reranker:
    """Qwen3-Reranker (causal LM): logit(yes) - logit(no) after the model card's judge prompt."""

    PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
              'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, max_length: int = 512, device: str = "cuda", repo: str = "Qwen/Qwen3-Reranker-0.6B"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = snapshot_download(repo, ignore_patterns=["*.bin", "*.gguf"])
        self.torch, self.device, self.max_length = torch, device, max_length
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok(self.PREFIX, add_special_tokens=False)["input_ids"]
        self.suf = self.tok(self.SUFFIX, add_special_tokens=False)["input_ids"]

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        torch = self.torch
        order = np.argsort([-(len(q) + len(p)) for q, p in pairs])
        out = np.empty(len(pairs), dtype=np.float32)
        room = self.max_length - len(self.pre) - len(self.suf)
        with torch.inference_mode():
            for s in range(0, len(pairs), batch_size):
                idx = order[s:s + batch_size]
                body = self.tok([f"<Instruct>: {QWEN3_INSTRUCT}\n<Query>: {pairs[i][0]}\n<Document>: {pairs[i][1]}"
                                 for i in idx], add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
                b = self.tok.pad({"input_ids": [self.pre + x + self.suf for x in body]}, padding=True,
                                 return_tensors="pt").to(self.device)
                # logits of the last position only: full-vocab logits for every position run out of GPU memory
                last = self.model(**b, logits_to_keep=1).logits[:, -1, :]
                out[idx] = (last[:, self.yes] - last[:, self.no]).float().cpu().numpy()
        return out.tolist()


def make_reranker(name: str, max_length: int):
    repo, kind = RERANKERS[name]
    return (Qwen3Reranker if kind == "qwen3" else CrossEncoder)(max_length, repo=repo)


def rerank_file(name: str) -> str:
    return "rerank.pkl" if name == "bge" else f"rerank_{name}.pkl"


# ---------------------------------------------------------------- cands

def cmd_cands(args) -> None:
    import bm25s
    import torch

    # chunk texts are read in batches and kept only as BM25 token ids: millions of chunk strings do not fit in RAM
    files = [f for p in args.chunks for f in (sorted(p.glob("*.parquet")) if p.is_dir() else [p])]
    meta = pq.read_table(files, columns=["chunk_id", "lang"])
    dense_np = load_index(args.index, meta.column("chunk_id").to_pylist())
    lang = np.array(meta.column("lang").to_pylist())
    del meta
    queries, _ = load_queries(args)
    trans = load_translations(args.translations)
    missing = [q["id"] for q in queries if key_of(q["query"]) not in trans]
    assert not missing, f"{len(missing)} queries have no translation, e.g. {missing[:5]}"
    tr = [trans[key_of(q["query"])] for q in queries]

    by_lang = {lg: np.flatnonzero(lang == lg) for lg in LANGS if (lang == lg).any()}
    t0 = time.time()
    vocab: dict[str, dict[str, int]] = {lg: {} for lg in by_lang}
    ids: dict[str, list[np.ndarray]] = {lg: [] for lg in by_lang}
    cut = args.bm25_max_tokens or None
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=20000, columns=["text", "heading_path", "lang"]):
            for text, head, lg in zip(*(b.column(k).to_pylist() for k in ("text", "heading_path", "lang"))):
                if lg not in vocab:
                    continue
                v = vocab[lg]
                toks = TOKENIZERS[lg](f"{head}\n{text}" if head else text)[:cut]
                ids[lg].append(np.fromiter((v.setdefault(t, len(v)) for t in toks), dtype=np.int32))
    bm = {}
    for lg in by_lang:
        bm[lg] = bm25s.BM25()
        bm[lg].index((ids[lg], vocab[lg]), show_progress=False)
        ids[lg] = []
    print(f"bm25 indexes {', '.join(f'{lg} {len(ix)}' for lg, ix in by_lang.items())}: {time.time() - t0:.0f}s",
          flush=True)

    model = DenseEncoder()
    enc = lambda texts: torch.from_numpy(model.encode(texts)).cuda()  # noqa: E731
    qv = {"vi": enc([q["query"] for q in queries])}
    for lg in by_lang:
        if lg in TRANSLATED:
            qv[lg] = enc([t[lg] for t in tr])
    del model
    dense = torch.from_numpy(dense_np).cuda()
    del dense_np
    sel = {lg: torch.from_numpy(ix).cuda() for lg, ix in by_lang.items()}

    k = args.depth
    out = {}
    for qi, q in enumerate(queries):
        lists = {}
        s_vi = dense @ qv["vi"][qi]
        lists["all_dense"] = torch.topk(s_vi, k).indices.cpu().numpy().astype(np.int32)
        for lg, ix in by_lang.items():
            docs, _ = bm[lg].retrieve([TOKENIZERS[lg](lang_query(q, tr[qi], lg, args.terms))],
                                      k=min(k, len(ix)), show_progress=False)
            lists[f"{lg}_bm25"] = ix[docs[0]].astype(np.int32)
            for name, s in [(f"{lg}_dense", s_vi)] + ([(f"{lg}_dense_t", dense @ qv[lg][qi])] if lg in qv and lg != "vi"
                                                      else []):
                top = torch.topk(s[sel[lg]], min(k, len(ix))).indices.cpu().numpy()
                lists[name] = ix[top].astype(np.int32)
        out[q["id"]] = lists
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "cands.pkl", "wb") as f:
        pickle.dump({"depth": k, "terms": args.terms, "lists": out}, f)
    print(f"{len(out)} queries, {time.time() - t0:.0f}s -> {args.out / 'cands.pkl'}")


# ---------------------------------------------------------------- fuse

def rrf_chunks(lists: list[np.ndarray]) -> np.ndarray:
    """RRF over chunk rankings; returns the fused chunk ranking."""
    score: dict[int, float] = {}
    for r in lists:
        for rank, i in enumerate(r.tolist(), 1):
            score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + rank)
    return np.array(sorted(score, key=score.__getitem__, reverse=True), dtype=np.int32)


def merge_docs(doc_lists: list[list[int]], weights: list[float]) -> list[int]:
    """RRF over doc rankings of disjoint language pools: with equal weights, a round robin."""
    score: dict[int, float] = {}
    for docs, w in zip(doc_lists, weights):
        for rank, d in enumerate(docs, 1):
            score[d] = score.get(d, 0.0) + w / (RRF_K + rank)
    return sorted(score, key=score.__getitem__, reverse=True)


def hit_at(gold_docs: set[int], docs: list[int]) -> dict[str, int]:
    first = next((r for r, d in enumerate(docs, 1) if d in gold_docs), None)
    return {f"@{k}": int(first is not None and first <= k) for k in HITS_AT}


def summarize(rows: list[dict], queries: list[dict]) -> dict:
    out = {}
    for name, sel in [("all", queries)] + [(lg, [q for q in queries if q["doc_lang"] == lg])
                                         for lg in sorted({q["doc_lang"] for q in queries})]:
        ids = {q["id"] for q in sel}
        rs = [r for r in rows if r["id"] in ids]
        if rs:
            out[name] = {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0] if k != "id"}
    return out


def pool_langs(lists: dict[str, np.ndarray]) -> list[str]:
    return [lg for lg in LANGS if f"{lg}_bm25" in lists]


def fused_lists(lists: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each language pool: RRF of its BM25 and dense lists."""
    return {lg: rrf_chunks([v for n, v in lists.items() if n.startswith(f"{lg}_")]) for lg in pool_langs(lists)}


def cmd_fuse(args) -> None:
    c = load_chunks(args.chunks)
    doc_of = np.array(c["doc_id"])
    queries, gold = load_queries(args)
    cands = pickle.loads((args.out / "cands.pkl").read_bytes())["lists"]
    D = 100  # docs kept per ranking

    def docs(r: np.ndarray) -> list[int]:
        return doc_list(doc_of, r, D)

    names = list(next(iter(cands.values())))
    systems = {
        "base_dense": lambda L: docs(L["all_dense"]),
        "base_rrf": lambda L: docs(rrf_chunks([L["vi_bm25"], L["all_dense"]])),
        "rrf_all": lambda L: docs(rrf_chunks([L[n] for n in names if n != "all_dense"])),
        # single lists and per-language fusions, judged on queries whose source doc is in that language
        **{f"pool:{n}": (lambda n: lambda L: docs(L[n]))(n) for n in names if n != "all_dense"},
        **{f"pool:{lg}_rrf": (lambda lg: lambda L: docs(fused_lists(L)[lg]))(lg)
           for lg in pool_langs(next(iter(cands.values())))},
        "merge_rr": lambda L: merge_docs([docs(r) for r in fused_lists(L).values()], [1.0] * len(fused_lists(L))),
    }
    report = {}
    for name, fn in systems.items():
        pool_lang = name.split(":")[1][:2] if name.startswith("pool:") else None
        sel = [q for q in queries if not pool_lang or q["doc_lang"] == pool_lang]
        if not sel:
            continue
        rows = [{"id": q["id"], **hit_at(set(gold[q["id"]]["relevant_docs"]), fn(cands[q["id"]]))} for q in sel]
        report[name] = summarize(rows, sel)
    (args.out / "fuse.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, r in report.items():
        print(f"{name:22s}", "  ".join(f"{lg}:" + " ".join(f"{r[lg][k]:.3f}" for k in ("@1", "@10", "@100"))
                                       for lg in r))


# ---------------------------------------------------------------- rerank

def rerank_pairs(L: dict[str, np.ndarray], doc_of: np.ndarray, n_docs: int, per_doc: int) -> list[int]:
    """Chunks to rerank: the best per_doc chunks of each of the top n_docs docs of each language pool."""
    out = []
    for r in fused_lists(L).values():
        keep = set(doc_list(doc_of, r, n_docs))
        taken: dict[int, int] = {}
        for i in r.tolist():
            d = int(doc_of[i])
            if d in keep and taken.get(d, 0) < per_doc:
                taken[d] = taken.get(d, 0) + 1
                out.append(i)
    return out


def cmd_rerank(args) -> None:
    files = [f for p in args.chunks for f in (sorted(p.glob("*.parquet")) if p.is_dir() else [p])]
    meta = pq.read_table(files, columns=["doc_id", "lang"])
    doc_of = meta.column("doc_id").to_numpy()
    c = {"lang": meta.column("lang").to_pylist()}  # texts are filled in below, only for the chunks in a pool
    del meta
    queries, _ = load_queries(args)
    if args.limit:
        queries = queries[:args.limit]
    trans = load_translations(args.translations)
    cands = pickle.loads((args.out / "cands.pkl").read_bytes())["lists"]
    model = make_reranker(args.reranker, args.max_length or 512)

    prev = None
    if args.from_rerank:  # cascade: rerank only the best chunks of the top docs of an earlier rerank
        prev = pickle.loads((args.out / rerank_file(args.from_rerank)).read_bytes())["scores"]

    def pool(qid: int) -> list[int]:
        if prev is None:
            return rerank_pairs(cands[qid], doc_of, args.n_docs_pool, args.per_doc_pool)
        out, taken = [], {}
        for i in ranked_chunks(prev[qid], "max"):
            d = int(doc_of[i])
            if d not in taken and len(taken) == args.top_docs:
                continue
            if taken.get(d, 0) < args.per_doc_pool:
                taken[d] = taken.get(d, 0) + 1
                out.append(i)
        return out

    pools = {q["id"]: pool(q["id"]) for q in queries}
    need = {i for p in pools.values() for i in p}
    c["text"], c["heading_path"] = {}, {}
    base = 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=50000, columns=["text", "heading_path"]):
            texts, heads = b.column("text").to_pylist(), b.column("heading_path").to_pylist()
            for j in range(len(texts)):
                if base + j in need:
                    c["text"][base + j], c["heading_path"][base + j] = texts[j], heads[j]
            base += len(texts)

    jobs, pairs = [], []  # (query id, chunk index, "vi" or "t" for the translated query) per pair
    for q in queries:
        tr = trans[key_of(q["query"])]
        for i in pools[q["id"]]:
            jobs.append((q["id"], i, "vi"))
            pairs.append((q["query"], chunk_text(c, i)))
            if args.translated_query and c["lang"][i] in TRANSLATED:
                jobs.append((q["id"], i, "t"))
                pairs.append((tr[c["lang"][i]], chunk_text(c, i)))
    print(f"{len(queries)} queries, {len(pairs)} pairs", flush=True)
    t0 = time.time()
    scores = model.score(pairs, batch_size=args.batch)
    print(f"reranked in {time.time() - t0:.0f}s ({len(pairs) / (time.time() - t0):.0f} pairs/s)", flush=True)

    out: dict[int, dict] = {}
    for (qid, i, ql), s in zip(jobs, scores):
        out.setdefault(qid, {}).setdefault(ql, {})[i] = float(s)
    with open(args.out / rerank_file(args.reranker), "wb") as f:
        pickle.dump({"n_docs": args.n_docs_pool, "per_doc": args.per_doc_pool, "reranker": args.reranker,
                     "scores": out}, f)


# ---------------------------------------------------------------- eval

def s400_map(p150: dict, s400: dict) -> np.ndarray:
    """For each p150 chunk, the s400 chunk of the same doc that overlaps it most."""
    by_doc: dict[int, list[int]] = {}
    for j, d in enumerate(s400["doc_id"]):
        by_doc.setdefault(d, []).append(j)
    out = np.full(len(p150["doc_id"]), -1, dtype=np.int64)
    for i, d in enumerate(p150["doc_id"]):
        a, b = p150["char_start"][i], p150["char_end"][i]
        best, best_ov = -1, 0
        for j in by_doc.get(d, []):
            ov = min(b, s400["char_end"][j]) - max(a, s400["char_start"][j])
            if ov > best_ov:
                best, best_ov = j, ov
        out[i] = best
    return out


def ranked_chunks(scores: dict[str, dict[int, float]], mode: str) -> list[int]:
    """Reranked chunks, best first. For chunks also scored with the translated query, mode picks the score."""
    s = dict(scores["vi"])
    if "t" in scores and mode != "vi":
        for i, z in scores["t"].items():
            s[i] = z if mode == "t" else max(s[i], z) if mode == "max" else (s[i] + z) / 2
    return sorted(s, key=s.__getitem__, reverse=True)


def build(qid: int, order: list[int], doc_of: np.ndarray, texts: list[str], n_docs: int, per_doc: int,
          to_s400: np.ndarray | None, s400_texts: list[str] | None) -> dict:
    docs = doc_list(doc_of, np.array(order), n_docs)
    keep = set(docs)
    taken: dict[int, int] = {}
    seen: set[int] = set()
    chunks = []
    for i in order:
        d = int(doc_of[i])
        if d not in keep or taken.get(d, 0) >= per_doc:
            continue
        if to_s400 is not None:
            j = int(to_s400[i])
            if j in seen:
                continue
            seen.add(j)
            text = s400_texts[j]
        else:
            text = texts[i]
        taken[d] = taken.get(d, 0) + 1
        chunks.append({"doc_id": d, "chunk_text": text})
    return {"id": qid, "relevant_docs": docs, "relevant_chunks": chunks}


def cmd_eval(args) -> None:
    p150 = load_chunks(args.chunks)
    s400 = load_chunks(args.s400)
    doc_of = np.array(p150["doc_id"])
    to_s400 = s400_map(p150, s400)
    assert (to_s400 >= 0).all(), "a p150 chunk has no s400 chunk"
    queries, gold = load_queries(args)
    rr_file = pickle.loads((args.out / rerank_file(args.reranker)).read_bytes())
    rr, n_pool = rr_file["scores"], rr_file["n_docs"]
    cands = pickle.loads((args.out / "cands.pkl").read_bytes())["lists"]
    queries = [q for q in queries if q["id"] in rr]

    orders = {}
    for mode in ["vi"] + (["t", "max", "mean"] if any("t" in v for v in rr.values()) else []):
        orders[f"rerank_{mode}"] = {q["id"]: ranked_chunks(rr[q["id"]], mode) for q in queries}
    # the same reranked chunk pool in fused order (no rerank), for comparison
    orders["fused_rr"] = {}
    for q in queries:
        pool = set(rr[q["id"]]["vi"])
        f = fused_lists(cands[q["id"]])
        rank_of = {d: r for r, d in enumerate(merge_docs([doc_list(doc_of, r, n_pool) for r in f.values()],
                                                         [1.0] * len(f)))}
        pos = {i: p for r in f.values() for p, i in enumerate(r.tolist()) if i in pool}
        orders["fused_rr"][q["id"]] = sorted(pool, key=lambda i: (rank_of[int(doc_of[i])], pos[i]))

    report = {}
    tag = "" if args.reranker == "bge" else f"_{args.reranker}"
    sub_dir = args.out / f"submissions{tag}"
    sub_dir.mkdir(exist_ok=True)
    for name, order in orders.items():
        rows = [{"id": q["id"], **hit_at(set(gold[q["id"]]["relevant_docs"]),
                                          doc_list(doc_of, np.array(order[q["id"]]), 100))} for q in queries]
        rec = {"doc_hit": summarize(rows, queries)}
        for unit in ("p150", "s400"):
            for n_docs in args.n_docs:
                for per_doc in args.per_doc:
                    sub = [build(q["id"], order[q["id"]], doc_of, p150["text"], n_docs, per_doc,
                                 to_s400 if unit == "s400" else None, s400["text"]) for q in queries]
                    s = [score_query(p, gold[p["id"]], "ws") for p in sub]
                    rec[f"{unit}_d{n_docs}_c{per_doc}"] = {
                        m: round(float(np.mean([x[m] for x in s])), 4)
                        for m in ("doc_r", "doc_f2", "chunk_p", "chunk_r", "chunk_f2")}
                    rec[f"{unit}_d{n_docs}_c{per_doc}"]["chunks_per_q"] = round(
                        float(np.mean([len(p["relevant_chunks"]) for p in sub])), 2)
                    (sub_dir / f"{name}_{unit}_d{n_docs}_c{per_doc}.json").write_text(
                        json.dumps(sub, ensure_ascii=False), encoding="utf-8")
        report[name] = rec
    report["queries"] = len(queries)
    (args.out / f"report{tag}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, rec in report.items():
        if name == "queries":
            continue
        h = rec["doc_hit"]
        print(f"{name:12s} hit@1/10/100 all " + " ".join(f"{h['all'][k]:.3f}" for k in ("@1", "@10", "@100"))
              + "  " + "  ".join(f"{lg} " + " ".join(f"{h[lg][k]:.3f}" for k in ("@1", "@10", "@100"))
                                 for lg in h if lg != "all"))
        for key, v in rec.items():
            if key != "doc_hit":
                print(f"   {key:14s} doc_f2 {v['doc_f2']:.3f}  chunk_r {v['chunk_r']:.3f}  "
                      f"chunk_f2 {v['chunk_f2']:.3f}  ({v['chunks_per_q']} chunks)")


# ---------------------------------------------------------------- encode

def cmd_encode(args) -> None:
    c = load_chunks(args.chunks)
    t0 = time.time()
    dense = DenseEncoder().encode([chunk_text(c, i) for i in range(len(c["text"]))], batch_size=args.batch,
                                  max_length=args.max_length or 1024)
    assert len(args.index) == 1, "encode writes one index folder"
    args.index[0].mkdir(parents=True, exist_ok=True)
    np.save(args.index[0] / "dense.npy", dense)
    (args.index[0] / "chunk_ids.json").write_text(json.dumps(c["chunk_id"]), encoding="utf-8")
    print(f"{len(dense)} chunks in {time.time() - t0:.0f}s -> {args.index[0]}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cands", "fuse", "rerank", "eval", "encode"])
    ap.add_argument("--out", type=Path, help="run folder (all commands but encode)")
    ap.add_argument("--chunks", type=Path, nargs="+", default=[DATA / "chunks40k/p150"])
    ap.add_argument("--s400", type=Path, nargs="+", default=[DATA / "chunks40k/s400"])
    ap.add_argument("--index", type=Path, nargs="+", default=[DATA / "index40k/p150"])
    ap.add_argument("--devset", type=Path, default=DATA / "devset/v0")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--translations", type=Path, default=DATA / "qtrans/v1/translations.jsonl")
    ap.add_argument("--depth", type=int, default=1000, help="chunks kept per candidate list")
    ap.add_argument("--terms", action=argparse.BooleanOptionalAction, default=True,
                    help="add the translated terms to the Chinese and English BM25 queries")
    ap.add_argument("--n-docs-pool", dest="n_docs_pool", type=int, default=50,
                    help="rerank: top docs taken from each language pool")
    ap.add_argument("--per-doc-pool", dest="per_doc_pool", type=int, default=3,
                    help="rerank: best chunks reranked per doc")
    ap.add_argument("--translated-query", action=argparse.BooleanOptionalAction, default=True,
                    help="rerank: also score Chinese and English chunks against the translated query")
    ap.add_argument("--max-length", type=int,
                    help="tokens; default 512 for rerank pairs, 1024 for encode (as the 40K index was built)")
    ap.add_argument("--reranker", default="bge", choices=sorted(RERANKERS),
                    help="rerank/eval: which reranker; scores go to rerank.pkl (bge) or rerank_<name>.pkl")
    ap.add_argument("--from-rerank", dest="from_rerank", choices=sorted(RERANKERS),
                    help="rerank: cascade on an earlier rerank of this run (its top --top-docs docs only)")
    ap.add_argument("--top-docs", dest="top_docs", type=int, default=150)
    ap.add_argument("--bm25-max-tokens", dest="bm25_max_tokens", type=int,
                    help="cands: index only the first N BM25 tokens of each chunk (RAM cap for millions of chunks)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-docs", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--per-doc", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    if args.cmd != "encode" and args.out is None:
        ap.error("--out is required")
    {"cands": cmd_cands, "fuse": cmd_fuse, "rerank": cmd_rerank, "eval": cmd_eval,
     "encode": cmd_encode}[args.cmd](args)


if __name__ == "__main__":
    main()
