"""Whole-doc BM25 the way the Mentor Day slides describe the organizers' baseline: Vietnamese word-segmented
(pyvi, triệu_chứng), each language searched with its own query (Vietnamese, or the EN/ZH translation), then RRF.

    python bm25w.py seg <out seg dir> <docs dirs...>      # tokenized docs, one parquet per docs file (resumable)
    python bm25w.py score <seg dir> <translations.jsonl> <query.parquet> <out pkl>

seg writes id, lang and tok (lowercase NFC tokens joined by spaces: pyvi words for vi, jieba for zh, plain words
for en) from title + text. score runs BM25 (k1 1.5, b 0.75) per language collection and keeps the top 300 docs per
query for: vi_w (pyvi words), vi_bg (syllables + syllable bigrams), en and zh (translated query).
"""
import hashlib
import json
import os
import pickle
import re
import sys
import time
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

K1, B, TOP, NF, MAXC = 1.5, 0.75, 300, 1 << 23, 30000
_WORD = re.compile(r"\w", re.U)


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").lower()


def tok(args: tuple[str, str]) -> str:
    lang, s = args
    s = norm(s)[:MAXC]
    if lang == "vi":
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(s)
    if lang == "zh":
        import jieba
        return " ".join(t for t in jieba.lcut_for_search(s) if _WORD.search(t))
    return s


def cmd_seg(out: Path, srcs: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", 16))) as pool:
        for f in [f for d in srcs for f in sorted(d.glob("*.parquet"))]:
            dst = out / f"{f.parent.name}_{f.stem}.parquet"
            if dst.exists():
                continue
            t = pq.read_table(f, columns=["id", "lang", "title", "text"]).to_pydict()
            t0 = time.time()
            toks = pool.map(tok, [(l, f"{ti or ''}\n{x or ''}") for l, ti, x in zip(t["lang"], t["title"], t["text"])],
                            chunksize=32)
            pq.write_table(pa.table({"id": t["id"], "lang": t["lang"], "tok": toks}), dst.with_suffix(".tmp"))
            dst.with_suffix(".tmp").replace(dst)
            print(f"seg {f.name}: {len(toks)} docs in {time.time() - t0:.0f}s", flush=True)


def vec(bigram: bool) -> HashingVectorizer:
    return HashingVectorizer(n_features=NF, token_pattern=r"(?u)\b\w+\b", lowercase=False, alternate_sign=False,
                             norm=None, dtype=np.float32, ngram_range=(1, 2) if bigram else (1, 1))


def bm25_top(m: sp.csr_matrix, ids: np.ndarray, Q: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """m: docs x terms counts, Q: queries x terms (binary). Top TOP doc ids and scores per query."""
    n = m.shape[0]
    df = np.bincount(m.indices, minlength=NF)
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.asarray(m.sum(1)).ravel()
    normz = (K1 * (1 - B + B * dl / dl.mean())).astype(np.float32)
    Qw = sp.csr_matrix(Q.multiply(idf.reshape(1, -1))).T.tocsr()
    best_s = np.zeros((Q.shape[0], 0), np.float32)
    best_i = np.zeros((Q.shape[0], 0), np.int64)
    for a in range(0, n, 40000):  # BM25 weights per block of rows, so RAM stays near the count matrix itself
        blk = m[a:a + 40000]
        rows = np.repeat(np.arange(blk.shape[0]), np.diff(blk.indptr))
        blk.data = (blk.data * (K1 + 1) / (blk.data + normz[a:a + 40000][rows])).astype(np.float32)
        s = (blk @ Qw).toarray().T
        del blk, rows
        k = min(TOP, s.shape[1])
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        best_s = np.concatenate([best_s, np.take_along_axis(s, top, 1)], 1)
        best_i = np.concatenate([best_i, ids[a:a + 40000][top]], 1)
        keep = np.argsort(-best_s, 1)[:, :TOP]
        best_s, best_i = np.take_along_axis(best_s, keep, 1), np.take_along_axis(best_i, keep, 1)
    return best_i, best_s


def cmd_score(seg: Path, trans: Path, qfile: Path, outp: Path) -> None:
    q = pq.read_table(qfile, columns=["id", "query"]).to_pydict()
    tr = {r["key"]: r for r in (json.loads(l) for l in trans.read_text(encoding="utf-8").splitlines() if l.strip())}
    trs = [tr[hashlib.sha1(x.encode("utf-8")).hexdigest()] for x in q["query"]]
    queries = {
        "vi_w": [tok(("vi", x)) for x in q["query"]],
        "vi_bg": [norm(x) for x in q["query"]],
        "en": [tok(("en", r["en"])) for r in trs],
        "zh": [tok(("zh", r["zh"])) for r in trs],
    }
    files = sorted(seg.glob("*.parquet"))
    out = {"qids": q["id"]}
    for name, lang, bigram in [("vi_w", "vi", False), ("en", "en", False), ("zh", "zh", False), ("vi_bg", "vi", True)]:
        hv = vec(bigram)
        parts, ids = [], []
        t0 = time.time()
        for f in files:
            t = pq.read_table(f).to_pydict()
            sel = [i for i, l in enumerate(t["lang"]) if l == lang]
            if not sel:
                continue
            txt = [t["tok"][i] for i in sel]
            if name == "vi_bg":
                txt = [x.replace("_", " ") for x in txt]
            parts.append(hv.transform(txt).tocsr())
            ids.append(np.array([t["id"][i] for i in sel], np.int64))
        if not parts:
            continue
        m = sp.vstack(parts).tocsr()
        del parts
        Q = hv.transform(queries[name]).tocsr()
        Q.data[:] = 1
        out[name] = bm25_top(m, np.concatenate(ids), Q)
        print(f"{name}: {m.shape[0]} docs, {time.time() - t0:.0f}s", flush=True)
        del m
        pickle.dump(out, open(outp, "wb"))
    print("BM25W_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "seg":
        cmd_seg(Path(sys.argv[2]), [Path(x) for x in sys.argv[3:]])
    else:
        cmd_score(*[Path(x) for x in sys.argv[2:6]])
