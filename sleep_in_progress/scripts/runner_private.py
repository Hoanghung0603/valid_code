r"""Runner dau-cuoi cho vong private (pha 1): tu query.parquet (id/so cau TU Y) + chi muc -> zip nop.

Moi stage la mot lenh rieng, resume duoc (bo qua neu report da co), ghi manifest_<stage>.json
kem sha256 moi dau vao/ra + thoi gian. Cong thuc = vi34 (xem runner_core), thay 2 phu thuoc
public bang quy tac noi tai: bo ref_chunks cf4, k_q = intrinsic_kq (tai hien c120in).

Stage b660 (CPU, deploy/runner_b660.bat, lock heavy_job.lock):
  prep          doc truy van -> queries.jsonl/qlist.json/query_run.parquet (+ mau --smoke N)
  bm25          search_cache.py -k 300 -> cache_k300.json -> bm25_top200.json
  hyde          rewrite_queries.py --mode hyde (LM Studio b660) -> hyde.jsonl + q_hyde.jsonl
  pool          pool = BM25 top-200 U dense top-200 -> pool_cands.json
  export_rr     xuat text doc pool -> docs_rr.part*.jsonl.zst (cho rerank bge)
  rr_merge      gop rerank_pool.json (GPU) -> pool.json (sap logit giam)
  fuse          rrf3d2 + fhyde (w_q8b_fhyde 2) -> cache_rrf3d2/cache_fhyde
  doclist       n_q + doclist.json + top100_bge.json
  q8b_select    cap 8B (toan doclist + 30 bien) + text + cands shard (+ bench)
  q8b_merge     gop ket qua 8B (GPU) -> q8b.json; bench Pearson neu co ref
  export_docwin text doc (doclist U top100 bge) -> docs_docwin.part*.jsonl.zst + cands_docwin.json
  chunk         chunk U900 tu docwin + k_q noi tai -> cands_chunk.json.zst + kq.json
  build         loc bad + ngoai kho -> chain vi34 (k_q noi tai) -> zip LZMA <= 99.000.000 B
  stats         so sanh voi bai tham chieu (public) + tong hop manifest
  report        bang bench Pearson + thoi gian/chi phi + ngoai suy 1.200 cau

Stage z87 (GPU du phong, deploy/runner_z87.bat, lock gpu.lock, vinmec <= 1s):
  dense         gop embedding sach (loai id nhiem) + dense/HyDE top-200 toan kho

Stage GPU Modal (deploy/modal_runner.py, deploy/runner_modal.bat): rr | docwin | q8b | bench.
Dau ra GPU duoc keo ve <out>/ truoc khi chay stage tiep theo.

Dung:
  python scripts/runner_private.py prep --tag rsmoke20 --smoke 20
  python scripts/runner_private.py build --tag rsmoke20 --delta 56
"""
import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner_core as rc
from runner_core import fingerprint, load_json, write_json, write_manifest

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
IX_DEFAULT = ROOT / "data/index/vi30"
QUERY_DEFAULT = ROOT / "data/query.parquet"
BAD_TSV = ROOT / "data/crawl/async_recovery_20261009/audit/contamination_bad_ids_20261024.tsv"
N_BAD = 16079
LINKS = ROOT / "data/sub/diag19/links_corpus_vi_en_zh.parquet"
OUT_ROOT_DEFAULT = ROOT / "data/sub/runner"
WORK_ROOT_DEFAULT = ROOT / "work/runner"
CHARS_DOC = 200000
RERANK_CHARS = 3000
Q8B_CHARS = 200000
Q8B_CAP = 1024 * 12
WINDOW_WORDS, WINDOW_STRIDE, MAX_LEN = 300, 250, 512
EMB_DIRS = ["emb", "emb_all", "emb23", "vi24/emb", "vi25/emb", "vi30/emb"]  # cu -> moi (z87 work/)
L40S_USD_H = 1.95


class P:
    """Duong dan cua mot tag chay (out b660 / work z87)."""

    def __init__(self, tag: str, out_root=None, work_root=None, ix=None, machine="b660"):
        self.tag = tag
        self.machine = machine
        self.out = Path(out_root or OUT_ROOT_DEFAULT) / tag
        self.work = Path(work_root or WORK_ROOT_DEFAULT) / tag
        self.ix = Path(ix or IX_DEFAULT)
        self.dir = self.work if machine == "z87" else self.out

    def f(self, name: str) -> Path:
        return self.dir / name


@contextmanager
def machine_lock(p: P, stage: str):
    """Lock dung cho may: z87 = gpu.lock (owned), b660 = heavy_job.lock do bat quan ly (khong giu o day)."""
    if p.machine == "z87":
        from vi24 import owned_lock
        with owned_lock(ROOT / "locks" / "gpu.lock", f"runner {p.tag} {stage} pid={os.getpid()}"):
            yield
    else:
        yield


def run_script(script: str, *args, cwd: Path | None = None):
    S = Path(__file__).resolve().parent
    cmd = [sys.executable, str(S / script), *map(str, args)]
    print("RUN", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def load_bad() -> set:
    from vi30 import read_ids_tsv
    bad = set(read_ids_tsv(BAD_TSV))
    assert len(bad) == N_BAD, len(bad)
    return bad


def load_corpus() -> set:
    import vi29 as v29
    return v29.load_corpus_ids(LINKS)


class PartWriter:
    """JSONL.zst writer don gian, xoay part theo --part-mb (byte chua nen)."""

    def __init__(self, out_base: Path, part_mb: int = 80, log=print):
        self.base = Path(out_base)
        self.limit = max(part_mb, 1) * 1_000_000
        self.log = log
        self.pi = -1
        self.written = 0
        self.n = 0
        self.paths = []
        self._open()

    def _open(self):
        import zstandard
        self.pi += 1
        p = self.base.parent / f"{self.base.name}.part{self.pi:03d}.jsonl.zst"
        self.fh = p.open("wb")
        self.zw = zstandard.ZstdCompressor(level=6).stream_writer(self.fh)
        self.paths.append(p)
        self.written = 0

    def write(self, obj):
        self.zw.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.n += 1
        self.written += 1
        # xoay theo byte nen da day xuong dia (fh.tell), khong theo so ban ghi
        if self.fh.tell() >= self.limit:
            self.close_part()
            self._open()

    def close_part(self):
        self.zw.close()
        self.fh.close()
        self.log(f"ghi {self.paths[-1].name}: {self.paths[-1].stat().st_size / 1e6:.1f}MB")

    def close(self):
        self.close_part()
        if self.n == 0:
            for p in self.paths:
                p.unlink(missing_ok=True)
            self.paths = []


def export_texts(p: P, wanted: dict, prefix: str, chars: int = CHARS_DOC, cap: int = 0,
                 no_title: bool = True, part_mb: int = 80) -> list:
    """Xuat text doc {id: idx} tu chi muc + boilerplate -> docs_<prefix>.part*.jsonl.zst
    {id, idx, text}. no_title=True: text sach cat chars (docwin/q8b/chunk); no_title=False:
    title + text cat chars = che do embed (vi23.cmd_export_new mode embed, dung cho rerank bge)."""
    from bigindex import BigIndex
    from bigpipeline import load_texts
    from boilerplate import BoilerplateFilter
    from export_docs import make_body
    t0 = time.time()
    idx = BigIndex(p.ix)
    bp = BoilerplateFilter.from_json((p.ix / "boilerplate.json").read_text(encoding="utf-8"))
    texts = load_texts(idx, wanted, bp, log=lambda m: print(m, flush=True))
    w = PartWriter(p.dir / f"docs_{prefix}", part_mb)
    empty = 0
    for d, i in sorted(wanted.items()):
        r = texts.get(d)
        if not r:
            continue
        body = make_body(r["title"], r["text"], chars, no_title=no_title)
        if cap:
            body = body[:cap]
        empty += not body.strip()
        w.write({"id": d, "idx": i, "text": body})
    w.close()
    rep = {"prefix": prefix, "yeu_cau": len(wanted), "co_text": len(texts), "empty_text": empty,
           "chars": chars, "no_title": no_title, "cap": cap,
           "parts": [str(x) for x in w.paths], "giay": round(time.time() - t0, 1)}
    write_json(p.dir / f"export_{prefix}_report.json", rep)
    return w.paths


# ---------------------------------------------------------------- stage b660

def stage_prep(p: P, a) -> None:
    t0 = time.time()
    rows = rc.load_queries(a.queries)
    ids = [i for i, _ in rows]
    assert len(set(ids)) == len(ids), "id trung trong truy van"
    sel = rc.evenly_spaced(ids, a.smoke) if a.smoke and a.smoke < len(ids) else list(ids)
    keep = {str(i): q for i, q in rows}
    sub = [(i, keep[str(i)]) for i in sel]
    write_json(p.dir / "qlist.json", [i for i, _ in sub])
    if a.smoke:
        write_json(p.dir / "smoke_ids.json", {"n": a.smoke, "ids": [i for i, _ in sub],
                                              "tong_goc": len(rows)})
    with (p.dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i, q in sub:
            f.write(json.dumps({"id": int(i), "query": q}, ensure_ascii=False) + "\n")
    import pandas as pd
    pd.DataFrame({"id": [i for i, _ in sub], "query": [q for _, q in sub]}).to_parquet(
        p.dir / "query_run.parquet")
    bad, corpus = load_bad(), load_corpus()
    write_manifest(p.dir, "prep", {"queries": str(a.queries), "smoke": a.smoke},
                   [a.queries], [p.dir / "queries.jsonl", p.dir / "qlist.json",
                                 p.dir / "query_run.parquet"], time.time() - t0,
                   extra={"so_cau": len(sub), "id_min": min(sel), "id_max": max(sel),
                          "bad_ids": len(bad), "corpus_ids": len(corpus)}, machine=p.machine)


def stage_bm25(p: P, a) -> None:
    t0 = time.time()
    out = p.dir / "cache_k300.json"
    if not out.exists():
        run_script("search_cache.py", "--index", p.ix, "--queries", p.dir / "query_run.parquet",
                   "--out", out, "-k", 300)
    cache = load_json(out)
    top = {q: [[h[0], h[1]] for h in v[:200]] for q, v in cache.items()}
    write_json(p.dir / "bm25_top200.json", top)
    write_manifest(p.dir, "bm25", {"k": 300, "top": 200, "index": str(p.ix)},
                   [p.dir / "query_run.parquet", p.ix / "ids.npy"], [out, p.dir / "bm25_top200.json"],
                   time.time() - t0, extra={"so_cau": len(top)}, machine=p.machine)


def stage_hyde(p: P, a) -> None:
    t0 = time.time()
    hy, qh = p.dir / "hyde.jsonl", p.dir / "q_hyde.jsonl"
    if not hy.exists():
        run_script("rewrite_queries.py", "--queries", p.dir / "query_run.parquet", "--out", hy,
                   "--mode", "hyde", "--model", a.hyde_model, "--workers", 4)
    rows = [json.loads(l) for l in hy.read_text(encoding="utf-8").splitlines() if l.strip()]
    rong = sum(1 for r in rows if not str(r.get("hyde", "")).strip())
    if not qh.exists() or rong == 0:
        run_script("rewrite_queries.py", "--queries", p.dir / "query_run.parquet", "--out", qh,
                   "--mode", "hyde", "--gen-file", hy)
    write_manifest(p.dir, "hyde", {"model": a.hyde_model, "mode": "hyde", "temperature": 0.3},
                   [p.dir / "query_run.parquet"], [hy, qh], time.time() - t0,
                   extra={"so_cau": len(rows), "hyde_rong": rong}, machine=p.machine)


def stage_pool(p: P, a) -> None:
    """pool_cands theo --pool-mode: base = BM25-200 U dense-200 (mac dinh, nhu smoke);
    p1 = BM25-300 U dense-200 U HyDE-200 (can bm25_top300.json va hyde_top200.json)."""
    t0 = time.time()
    mode = getattr(a, "pool_mode", "base")
    if mode == "p1":
        # BM25 top-300 lay tu cache_k300 (cung cache cua bm25; khong tim lai)
        cache300 = load_json(p.dir / "cache_k300.json")
        write_json(p.dir / "bm25_top300.json",
                   {q: [[h[0], h[1]] for h in v[:300]] for q, v in cache300.items()})
    bm300 = load_json(p.dir / "bm25_top300.json") if mode == "p1" else load_json(p.dir / "bm25_top200.json")
    dn = load_json(p.dir / "dense_top200.json")
    hy = load_json(p.dir / "hyde_top200.json") if mode == "p1" else {}
    out = {}
    for q in sorted(set(bm300) | set(dn) | set(hy), key=lambda x: int(x)):
        out[q] = rc.pool_for_mode(mode, bm300.get(q, []), dn.get(q, []), hy.get(q, []))
    write_json(p.dir / "pool_cands.json", out)
    n = max(len(out), 1)
    srcs = ["bm25_top300.json" if mode == "p1" else "bm25_top200.json", "dense_top200.json"]
    srcs += ["hyde_top200.json"] if mode == "p1" else []
    write_manifest(p.dir, "pool", {"top": 200, "nguon": "BM25 U dense", "pool_mode": mode},
                   [p.dir / s for s in srcs], [p.dir / "pool_cands.json"], time.time() - t0,
                   extra={"so_cau": len(out), "TB_pool_cau": round(sum(map(len, out.values())) / n, 2)},
                   machine=p.machine)


def stage_export_rr(p: P, a) -> None:
    t0 = time.time()
    cands = load_json(p.dir / "pool_cands.json")
    wanted = {}
    for rows in cands.values():
        for h in rows:
            wanted.setdefault(int(h[0]), int(h[1]))
    # che do embed (vi23.cmd_export_new mode embed): title + text sach, cat 3.000 ky tu — dung cho rerank bge
    paths = export_texts(p, wanted, "rr", chars=RERANK_CHARS, no_title=False)
    write_manifest(p.dir, "export_rr", {"chars": RERANK_CHARS, "no_title": False}, [p.ix / "ids.npy"],
                   paths + [p.dir / "export_rr_report.json"], time.time() - t0,
                   extra={"so_doc": len(wanted)}, machine=p.machine)


def stage_rr_merge(p: P, a) -> None:
    t0 = time.time()
    raw = {}
    for f in sorted(p.dir.glob("rerank_pool*.json")):
        raw.update(load_json(f))
    cands = load_json(p.dir / "pool_cands.json")
    pool, none = {}, 0
    for q, rows in cands.items():
        sc = {str(h[0]): h[2] for h in raw.get(q, [])}
        hits = [[h[0], h[1], sc.get(str(h[0]))] for h in rows]
        none += sum(1 for h in hits if h[2] is None)
        pool[q] = rc.sort_by_score(hits)
    write_json(p.dir / "pool.json", pool)
    write_manifest(p.dir, "rr_merge", {"sort": "logit desc"},
                   sorted(p.dir.glob("rerank_pool*.json")) + [p.dir / "pool_cands.json"],
                   [p.dir / "pool.json"], time.time() - t0,
                   extra={"so_cau": len(pool), "logit_None": none}, machine=p.machine)
    assert none == 0, f"{none} cap thieu logit bge (thieu text?)"


def stage_fuse(p: P, a) -> None:
    t0 = time.time()
    pool = load_json(p.dir / "pool.json")
    q8b = load_json(p.dir / "q8b.json") if (p.dir / "q8b.json").exists() else {}
    dense = load_json(p.dir / "dense_top200.json")
    hyde = load_json(p.dir / "hyde_top200.json")
    rrf3d2, fhyde = rc.fuse_rrf(pool, q8b, dense, hyde, rc.W_Q8B_FHYDE)
    write_json(p.dir / "cache_rrf3d2.json", rrf3d2)
    write_json(p.dir / "cache_fhyde.json", fhyde)
    write_manifest(p.dir, "fuse", {"k_rrf": rc.K_RRF, "w": [rc.W_BGE, rc.W_Q8B, rc.W_DENSE],
                                   "w_q8b_fhyde": rc.W_Q8B_FHYDE, "w_hyde": rc.W_HYDE,
                                   "q8b_hit": sum(map(len, q8b.values()))},
                   [p.dir / "pool.json", p.dir / "dense_top200.json", p.dir / "hyde_top200.json"],
                   [p.dir / "cache_rrf3d2.json", p.dir / "cache_fhyde.json"], time.time() - t0,
                   extra={"so_cau": len(rrf3d2)}, machine=p.machine)


def stage_doclist(p: P, a) -> None:
    t0 = time.time()
    pool = load_json(p.dir / "pool.json")
    fhyde = load_json(p.dir / "cache_fhyde.json")
    dl, ns, t100 = {}, [], {}
    for q in sorted(pool, key=lambda x: int(x)):
        n = rc.n_q_for(pool[q])
        dl[q] = rc.doclist_for(fhyde.get(q, []), n)
        ns.append(n)
        t100[q] = rc.top_bge(pool[q], rc.TOP_BGE)
    write_json(p.dir / "doclist.json", dl)
    write_json(p.dir / "top100_bge.json", t100)
    n = max(len(dl), 1)
    write_manifest(p.dir, "doclist", {"thr": rc.THR_N, "lo": rc.LO_N, "hi": rc.HI_N,
                                      "ref_chunks": "bo (quy tac noi tai)"},
                   [p.dir / "pool.json", p.dir / "cache_fhyde.json"],
                   [p.dir / "doclist.json", p.dir / "top100_bge.json"], time.time() - t0,
                   extra={"n_q_TB": round(sum(ns) / n, 2), "n_q_min": min(ns), "n_q_max": max(ns),
                          "TB_doc_cau": round(sum(map(len, dl.values())) / n, 2)}, machine=p.machine)


def stage_q8b_select(p: P, a) -> None:
    import numpy as np
    t0 = time.time()
    fhyde = load_json(p.dir / "cache_fhyde.json")
    pool = load_json(p.dir / "pool.json")
    dl = load_json(p.dir / "doclist.json")
    scored = load_json(p.dir / "q8b.json") if (p.dir / "q8b.json").exists() else {}
    pairs = rc.q8b_pairs_for(fhyde, pool, dl, scored, rc.Q8B_MARGIN)
    # --union-dirs: them cap cua cac thu muc pool mode khac (cung chi muc, cung diem 8B da co)
    for ud in a.union_dirs:
        e = Path(ud)
        pairs = rc.union_pairs(pairs, rc.q8b_pairs_for(load_json(e / "cache_fhyde.json"), load_json(e / "pool.json"),
                                                       load_json(e / "doclist.json"), scored, rc.Q8B_MARGIN))
    pairs = {q: v for q, v in pairs.items() if v}
    write_json(p.dir / "q8b_pairs.json", pairs)
    pos_all = {int(d): i for i, d in enumerate(np.load(p.ix / "ids.npy"))}
    wanted = {}
    for q, docs in pairs.items():
        pos = {str(h[0]): h[1] for h in pool.get(q, [])}
        for d in docs:
            wanted.setdefault(int(d), int(pos.get(str(d), pos_all.get(int(d), -1))))
    shards = [{q: [[d, wanted[int(d)]] for d in docs]
               for q, docs in list(pairs.items())[k::a.q8b_shards]} for k in range(a.q8b_shards)]
    outs = [p.dir / f"cands_q8b_shard{k}.json" for k in range(a.q8b_shards)]
    for k, s in enumerate(shards):
        write_json(p.dir / f"cands_q8b_shard{k}.json", s)
    if a.ref_q8b:
        run_ids = {str(i) for i in load_json(p.dir / "qlist.json")}
        ref = {q: v for q, v in load_json(a.ref_q8b).items() if q in run_ids}
        bench = rc.sample_cache_pairs(ref, a.bench_n, seed=31)
        bc = {}
        for q, d, _s in bench:
            bc.setdefault(q, []).append([d, pos_all.get(int(d), 0)])
            wanted.setdefault(int(d), pos_all.get(int(d), 0))
        write_json(p.dir / "cands_q8b_bench.json", bc)
        write_json(p.dir / "bench_q8b_ref.json", {f"{q}|{d}": s for q, d, s in bench})
        outs += [p.dir / "cands_q8b_bench.json", p.dir / "bench_q8b_ref.json"]
    paths = export_texts(p, wanted, "q8b", chars=Q8B_CHARS, cap=Q8B_CAP)
    n = max(len(pairs), 1)
    write_manifest(p.dir, "q8b_select", {"margin": rc.Q8B_MARGIN, "shards": a.q8b_shards,
                                         "chars": Q8B_CHARS, "cap": Q8B_CAP},
                   [p.dir / "cache_fhyde.json", p.dir / "doclist.json", p.dir / "pool.json"],
                   [p.dir / "q8b_pairs.json", *paths, *outs], time.time() - t0,
                   extra={"so_cau": len(pairs), "so_cap": sum(map(len, pairs.values())),
                          "TB_cap_cau": round(sum(map(len, pairs.values())) / n, 2)},
                   machine=p.machine)


def stage_q8b_merge(p: P, a) -> None:
    t0 = time.time()
    old = load_json(p.dir / "q8b.json") if (p.dir / "q8b.json").exists() else {}
    new = {}
    for f in sorted(p.dir.glob("q8b_shard*_out.json")):
        new.update(load_json(f))
    merged, st = rc.merge_q8b(old, new)
    write_json(p.dir / "q8b.json", merged)
    extras = {"trung_bo": st["trung_bo"], "none_bo": st["none_bo"], "hit_moi": sum(map(len, new.values())),
              "hit_tong": sum(map(len, merged.values()))}
    outs = [p.dir / "q8b.json"]
    bench_f = p.dir / "q8b_bench_out.json"
    if bench_f.exists() and (p.dir / "bench_q8b_ref.json").exists():
        got = {f"{q}|{h[0]}": h[2] for q, rows in load_json(bench_f).items() for h in rows
               if h[2] is not None}
        ref = load_json(p.dir / "bench_q8b_ref.json")
        ks = [k for k in ref if k in got]
        rep = rc.corr([ref[k] for k in ks], [got[k] for k in ks])
        rep["model"] = "Qwen/Qwen3-Reranker-8B"
        write_json(p.dir / "bench_q8b.json", rep)
        outs.append(p.dir / "bench_q8b.json")
        extras["bench_8b"] = rep
    write_manifest(p.dir, "q8b_merge", {}, sorted(p.dir.glob("q8b_shard*_out.json")) + [bench_f],
                   outs, time.time() - t0, extra=extras, machine=p.machine)


def stage_export_docwin(p: P, a) -> None:
    t0 = time.time()
    dl = load_json(p.dir / "doclist.json")
    t100 = load_json(p.dir / "top100_bge.json")
    pool = load_json(p.dir / "pool.json")
    cands, wanted = {}, {}
    for q in sorted(dl, key=lambda x: int(x)):
        pos = {str(h[0]): h[1] for h in pool.get(q, [])}
        rows, seen = [], set()
        for d in list(dl[q]) + [d for d in t100.get(q, []) if d not in set(map(str, dl[q]))]:
            s = str(d)
            if s in seen:
                continue
            seen.add(s)
            rows.append([d, int(pos.get(s, -1))])
            wanted.setdefault(int(d), int(pos.get(s, -1)))
        cands[q] = rows
    write_json(p.dir / "cands_docwin.json", cands)
    paths = export_texts(p, wanted, "docwin", chars=CHARS_DOC)
    n = max(len(cands), 1)
    write_manifest(p.dir, "export_docwin", {"chars": CHARS_DOC, "window": [WINDOW_WORDS, WINDOW_STRIDE]},
                   [p.dir / "doclist.json", p.dir / "top100_bge.json", p.ix / "ids.npy"],
                   [p.dir / "cands_docwin.json", *paths], time.time() - t0,
                   extra={"so_cau": len(cands), "so_doc": len(wanted),
                          "TB_cap_cau": round(sum(map(len, cands.values())) / n, 2)},
                   machine=p.machine)


def stage_chunk(p: P, a) -> None:
    from chunks_from_docwin import build_q_rows, select_q_rows, verify_verbatim
    from dwx_merge import write_zst_stream
    from rerank_run import load_docs
    t0 = time.time()
    wins = load_json(p.dir / "docwin.json.zst")
    cands = load_json(p.dir / "cands_docwin.json")
    t100 = load_json(p.dir / "top100_bge.json")
    dl = load_json(p.dir / "doclist.json")
    rrf3d2 = load_json(p.dir / "cache_rrf3d2.json")
    wanted = {int(h[0]) for rows in cands.values() for h in rows}
    docs = load_docs([str(x) for x in sorted(p.dir.glob("docs_docwin.part*.jsonl.zst"))], wanted=wanted)
    missing = wanted - set(docs)
    assert not missing, f"thieu text docwin: {len(missing)}"
    cache: dict = {}
    out, kq = {}, {}
    for q in sorted(cands, key=lambda x: int(x)):
        built = build_q_rows(cands[q], docs, wins.get(q, {}), 900, 500, cache)
        out[q] = select_q_rows(built, 100000)
        cand100 = [[d, i + 1] for i, d in enumerate(t100.get(q, []))]
        rows100 = build_q_rows(cand100, docs, wins.get(q, {}), 900, rc.TOP_BGE, cache)
        kq[q] = rc.intrinsic_kq(rows100, rrf3d2.get(q, []), set(map(str, dl.get(q, []))),
                                rc.KQ_CAP, rc.WD, "doc_text", 0)
    nbad = verify_verbatim(out, docs)
    assert nbad == 0, f"chunk khong nguyen van: {nbad}"
    write_zst_stream(p.dir / "cands_chunk.json.zst", out)
    write_json(p.dir / "kq.json", kq)
    n = max(len(out), 1)
    write_manifest(p.dir, "chunk", {"max_units": 900, "wd": rc.WD, "kq_cap": rc.KQ_CAP,
                                    "kq_mode": "cat 120 sau do loc doclist (c120in)"},
                   [p.dir / "docwin.json.zst", p.dir / "cands_docwin.json",
                    p.dir / "top100_bge.json", p.dir / "cache_rrf3d2.json"],
                   [p.dir / "cands_chunk.json.zst", p.dir / "kq.json"], time.time() - t0,
                   extra={"verbatim_bad": nbad, "TB_chunk_ung_vien_cau":
                          round(sum(map(len, out.values())) / n, 2),
                          "k_q_TB": round(sum(kq.values()) / n, 2),
                          "k_q_min": min(kq.values()), "k_q_max": max(kq.values())},
                   machine=p.machine)


def validate_runner(sub_path, n_queries: int) -> dict:
    """Chay that evaluate.py validate tren JSON bai. Tra {code, output} (code 0 = hop le)."""
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "evaluate.py"), "validate",
           str(sub_path), "--n-queries", str(n_queries)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"code": r.returncode, "output": (r.stdout + r.stderr).strip()[-2000:]}


def stage_build(p: P, a) -> None:
    import vi29 as v29
    t0 = time.time()
    bad = load_bad()
    corpus = load_corpus()
    excl = set(bad)
    ind = p.dir / "in1"
    ind.mkdir(exist_ok=True)
    spec = [("pool", "pool.json", "cache"), ("q8b", "q8b.json", "cache"),
            ("dense", "dense_top200.json", "cache"), ("hyde", "hyde_top200.json", "cache"),
            ("cands", "cands_chunk.json.zst", "cands")]
    loc = {}
    for name, fn, kind in spec:
        src = p.dir / fn
        if not src.exists():
            if name == "q8b":
                write_json(ind / fn, {})
                loc[fn] = {"nguon": "(khong co)", "truoc": 0, "sau": 0, "bo": 0}
                continue
            raise SystemExit(f"thieu {src}")
        obj = load_json(src)
        out = v29.filter_cache(obj, excl, corpus) if kind == "cache" else v29.filter_cands(obj, excl, corpus)
        nb = sum(map(len, obj.values()))
        na = sum(map(len, out.values()))
        (write_json if fn.endswith(".json") else _write_zst)(ind / fn, out)
        loc[fn] = {"nguon": str(src), "truoc": nb, "sau": na, "bo": nb - na}
        del obj, out
    files = {"pool": ind / "pool.json", "q8b": ind / "q8b.json", "dense": ind / "dense_top200.json",
             "hyde": ind / "hyde_top200.json", "cands": ind / "cands_chunk.json.zst",
             "kq": p.dir / "kq.json"}
    name = p.tag  # rc.chain_zip_check_runner ghi submission_<name>.json/.zip (tranh submission_submission_<tag>)
    if a.delta is not None:
        delta, attempts = a.delta, []
        r = rc.chain_zip_check_runner(p.dir, name, delta, a.max_zip, files)
        assert r["code"] == 0, r
    else:
        def attempt(d):
            return rc.chain_zip_check_runner(p.dir, f"{p.tag}p", d, a.max_zip, files)
        delta, attempts = rc.find_delta_runner(attempt, a.delta0, a.max_zip)
        r = rc.chain_zip_check_runner(p.dir, name, delta, a.max_zip, files)
        assert r["code"] == 0, r
    sub = load_json(p.dir / f"submission_{name}.json")
    qlist = load_json(p.dir / "qlist.json")
    clean = rc.assert_clean_runner(sub, bad, corpus, expect_ids=qlist)
    val = validate_runner(p.dir / f"submission_{name}.json", len(qlist))
    if val["code"] != 0:
        raise RuntimeError(f"evaluate.py validate loi (code {val['code']}): {val['output']}")
    fp = fingerprint(p.dir / f"submission_{name}.zip")
    write_json(p.dir / "build_report.json",
               {"name": name, "delta": delta, "attempts": attempts, "max_zip": a.max_zip,
                "loc_dau_vao": loc, "assert_clean": clean, "validate": "ok",
                "validate_output": val["output"],
                "dedup_mode": "doc_text", "ref_chunks": "bo", "zip": fp,
                "n_q_TB": r.get("n_q_TB"), "tong_chunk": r.get("tong_chunk")})
    write_manifest(p.dir, "build", {"delta": delta, "max_zip": a.max_zip, "find_delta": a.delta is None},
                   [files[k] for k in files] + [p.dir / "kq.json"],
                   [p.dir / f"submission_{name}.json", p.dir / f"submission_{name}.zip", p.dir / "build_report.json"],
                   time.time() - t0, extra={"assert_clean": clean, "zip": fp}, machine=p.machine)
    print(json.dumps({"build": name, "delta": delta, **clean, "zip_bytes": fp["bytes"],
                      "sha256": fp["sha256"]}, ensure_ascii=False), flush=True)


def _write_zst(path, obj):
    import zstandard
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            z.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def kq_blocked(sub: list, kq: dict, delta: int) -> int:
    """So cau ma k_q + delta bi chan boi so chunk that co trong bai (kp = min(kq+delta, |C_q|) trong chain)."""
    return sum(1 for it in sub if len(it["relevant_chunks"]) < kq.get(str(it["id"]), 0) + delta)


def doclist_cap_ref(sub: list, ref: list) -> float:
    """TB |doclist(bai) ∩ doclist(tham chieu)| tren cac cau chung id."""
    rd = {str(it["id"]): {str(d) for d in it["relevant_docs"]} for it in ref}
    inter = [len({str(d) for d in it["relevant_docs"]} & rd[str(it["id"])])
             for it in sub if str(it["id"]) in rd]
    return round(sum(inter) / max(len(inter), 1), 2)


def stage_stats(p: P, a) -> None:
    t0 = time.time()
    name = p.tag  # khop stage_build: file la submission_<tag>.json
    sub = load_json(p.dir / f"submission_{name}.json")
    rep = {"so_cau": len(sub),
           "TB_doc_cau": round(sum(len(it["relevant_docs"]) for it in sub) / max(len(sub), 1), 2),
           "TB_chunk_cau": round(sum(len(it["relevant_chunks"]) for it in sub) / max(len(sub), 1), 2),
           "TB_doc_co_chunk": round(sum(len({str(c["doc_id"]) for c in it["relevant_chunks"]})
                                        for it in sub) / max(len(sub), 1), 2)}
    if a.ref_sub:
        ref_all = load_json(a.ref_sub)
        ids = {str(it["id"]) for it in sub}
        ref = [it for it in ref_all if str(it["id"]) in ids]
        rep["so_voi_tham_chieu"] = {"nguon": str(a.ref_sub), **rc.compare_subs_subset(sub, ref),
                                    "TB_doclist_cap_ref": doclist_cap_ref(sub, ref)}
    kq = load_json(p.dir / "kq.json")
    delta = load_json(p.dir / "build_report.json")["delta"]
    rep["k_q"] = {"TB": round(sum(kq.values()) / max(len(kq), 1), 2),
                  "min": min(kq.values()), "max": max(kq.values()),
                  "delta": delta, "so_cau_bi_chan_boi_C": kq_blocked(sub, kq, delta)}
    write_json(p.dir / "stats_report.json", rep)
    write_manifest(p.dir, "stats", {"ref_sub": str(a.ref_sub)},
                   [p.dir / f"submission_{name}.json"], [p.dir / "stats_report.json"], time.time() - t0,
                   extra=rep, machine=p.machine)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def stage_bench_prep(p: P, a) -> None:
    """Mau cap public co diem cu (pool bge + docwin) -> bench_rr_in/bench_docwin_in (text nguyen
    ban) cho GPU cham lai; bench 8B dung cands_q8b_bench.json (q8b_select --ref-q8b).
    Lay query text tu --queries (toan bo cau public, khong gioi han o tap dang chay) de mau du."""
    import numpy as np
    t0 = time.time()
    qtext = {str(i): q for i, q in rc.load_queries(a.queries)}
    pos_all = {int(d): i for i, d in enumerate(np.load(p.ix / "ids.npy"))}
    rr_rows = dw_rows = []
    if a.ref_pool:
        rr_rows = rc.sample_cache_pairs(load_json(a.ref_pool), a.bench_n, seed=42)
    if a.ref_docwin:
        dw_rows = rc.sample_docwin_rows(load_json(a.ref_docwin), a.bench_n, seed=43)
    wanted = {}
    for _q, d, _s in rr_rows:
        wanted.setdefault(int(d), pos_all.get(int(d), -1))
    for _q, d, _s, _e, _l in dw_rows:
        wanted.setdefault(int(d), pos_all.get(int(d), -1))
    outs = []
    t_rr: dict = {}
    t_dw: dict = {}
    if wanted and rr_rows:
        from rerank_run import load_docs
        paths = export_texts(p, wanted, "benchrr", chars=RERANK_CHARS, no_title=False)
        t_rr = load_docs([str(x) for x in paths], wanted=set(wanted))
        outs += paths
    if wanted and dw_rows:
        from rerank_run import load_docs
        paths = export_texts(p, wanted, "benchdw", chars=CHARS_DOC, no_title=True)
        t_dw = load_docs([str(x) for x in paths], wanted=set(wanted))
        outs += paths
    if rr_rows:
        rows = [{"query": qtext[q], "text": t_rr[int(d)], "ref": s}
                for q, d, s in rr_rows if int(d) in t_rr and q in qtext]
        write_json(p.dir / "bench_rr_in.json", rows)
        outs.append(p.dir / "bench_rr_in.json")
    if dw_rows:
        rows = [{"query": qtext[q], "text": t_dw[int(d)][s:e], "ref": lg}
                for q, d, s, e, lg in dw_rows if int(d) in t_dw and q in qtext]
        write_json(p.dir / "bench_docwin_in.json", rows)
        outs.append(p.dir / "bench_docwin_in.json")
    write_manifest(p.dir, "bench_prep", {"ref_pool": str(a.ref_pool), "ref_docwin": str(a.ref_docwin),
                                         "bench_n": a.bench_n},
                   [Path(a.ref_pool), Path(a.ref_docwin)], outs, time.time() - t0,
                   extra={"rr_pairs": len(rr_rows), "docwin_rows": len(dw_rows)}, machine=p.machine)


def stage_bench(p: P, a) -> None:
    """Tinh Pearson/Spearman cua diem GPU moi vs diem cu (bench_*_in.ref) cho bge rerank + docwin."""
    t0 = time.time()
    outs = []
    for name, model in (("rr", "BAAI/bge-reranker-v2-m3"), ("docwin", "BAAI/bge-reranker-v2-m3")):
        fin, fout = p.dir / f"bench_{name}_in.json", p.dir / f"bench_{name}_out.json"
        if not (fin.exists() and fout.exists()):
            continue
        rows, sc = load_json(fin), load_json(fout)["scores"]
        assert len(rows) == len(sc), (name, len(rows), len(sc))
        rep = rc.corr([r["ref"] for r in rows], sc)
        rep.update({"model": model, "stage": name, "nguon_diem_cu": "cache public"})
        write_json(p.dir / f"bench_{name}.json", rep)
        outs.append(p.dir / f"bench_{name}.json")
    write_manifest(p.dir, "bench", {}, sorted(p.dir.glob("bench_*_out.json")), outs,
                   time.time() - t0, machine=p.machine)
    print(json.dumps({f.stem: load_json(f) for f in outs}, ensure_ascii=False), flush=True)


def stage_kq_check(p: P, a) -> None:
    """Do k_q noi tai vs k_q c120in tren toan bo 1.200 cau public (dung artefact cu, chi CPU):
    rows100 = chunk U900 cua lop docwin top-100 bge (ref-rows); rrf3d2 + doclist vi34 (ref).
    Bao: TB 2 ben, tuong quan theo cau (Pearson/Spearman), ca 2 bien the mode/dedup."""
    t0 = time.time()
    ref_kq = {str(it["id"]): len(it["relevant_chunks"]) for it in load_json(a.ref_kq)}
    rows_all = load_json(a.ref_rows)
    rrf = load_json(a.ref_rrf)
    dl = load_json(a.ref_doclist)
    ks = {0: {}, 1: {}}
    for q, rows in rows_all.items():
        if q not in ref_kq:
            continue
        allowed = set(map(str, dl.get(q, [])))
        for mode in (0, 1):
            ks[mode][q] = rc.intrinsic_kq(rows, rrf.get(q, []), allowed, rc.KQ_CAP, rc.WD,
                                          "doc_text", mode)
    rep = {}
    for mode in (0, 1):
        qs = sorted(set(ks[mode]) & set(ref_kq), key=lambda x: int(x))
        x = [ks[mode][q] for q in qs]
        y = [ref_kq[q] for q in qs]
        rep[f"mode{mode}"] = {"so_cau": len(qs), "k_q_moi_TB": round(sum(x) / max(len(x), 1), 2),
                              "k_q_c120in_TB": round(sum(y) / max(len(y), 1), 2),
                              "chenh_lech_TB": round(sum(a1 - b1 for a1, b1 in zip(x, y)) / max(len(x), 1), 2),
                              **rc.corr(x, y)}
    write_json(p.dir / "kq_check_report.json", rep)
    write_manifest(p.dir, "kq_check", {"kq_cap": rc.KQ_CAP, "wd": rc.WD,
                                       "ref_kq": str(a.ref_kq), "ref_rows": str(a.ref_rows)},
                   [a.ref_kq, a.ref_rows, a.ref_rrf, a.ref_doclist],
                   [p.dir / "kq_check_report.json"], time.time() - t0, extra=rep, machine=p.machine)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def stage_report(p: P, a) -> None:
    t0 = time.time()
    benches = {}
    for m, fn in (("bge_rerank", "bench_rr.json"), ("docwin_bge", "bench_docwin.json"),
                  ("qwen8b", "bench_q8b.json")):
        f = p.dir / fn
        if f.exists():
            benches[m] = load_json(f)
    stages = {}
    for m in sorted(p.dir.glob("manifest_*.json")):
        d = load_json(m)
        stages[d["stage"]] = {"giay": d["seconds"], "machine": d.get("machine", "")}
    n = max(len(load_json(p.dir / "qlist.json")), 1)
    gpu_pairs = {}
    for fn, key in (("rerank_pool.json", "rr_pairs"), ("docwin.json.zst", "docwin_wins"),
                    ("q8b_pairs.json", "q8b_pairs")):
        f = p.dir / fn
        if not f.exists():
            continue
        obj = load_json(f)
        if key == "docwin_wins":
            gpu_pairs[key] = sum(len(v2) for by_d in obj.values() for v2 in by_d.values())
        else:
            gpu_pairs[key] = sum(len(v) for v in obj.values())
    tu = 1200.0 / n
    wall = sum(v["giay"] for v in stages.values())
    rep = {"so_cau": n, "benches": benches, "stages": stages, "gpu_pairs": gpu_pairs,
           "tong_giay_wall": round(wall, 1),
           "ngoai_suy_1200_cau": {
               "wall_giay_toan_bo_cpu": round(wall * tu, 0),
               "gpu_pairs_1200": {k: round(v * tu) for k, v in gpu_pairs.items()},
               "ghi_chu": "GPU thoi gian lay tu job_report_*.json Modal; USD = GPU-h x %.2f" % L40S_USD_H}}
    job = {}
    for f in sorted(p.dir.glob("job_report_*.json")):
        job[f.stem.replace("job_report_", "")] = load_json(f)
    if job:
        gpu_s = sum(v.get("elapsed", 0) * v.get("gpus", 1) for v in job.values())
        rep["modal_jobs"] = job
        rep["gpu_giay_tong"] = round(gpu_s, 1)
        rep["gpu_gio"] = round(gpu_s / 3600, 3)
        rep["usd"] = round(gpu_s / 3600 * L40S_USD_H, 2)
        rep["ngoai_suy_1200_cau"]["usd"] = round(gpu_s / 3600 * L40S_USD_H * tu, 2)
        rep["ngoai_suy_1200_cau"]["gpu_gio"] = round(gpu_s / 3600 * tu, 2)
    write_json(p.dir / "report_runner.json", rep)
    write_manifest(p.dir, "report", {}, [p.dir / "qlist.json"], [p.dir / "report_runner.json"],
                   time.time() - t0, machine=p.machine)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- stage z87: dense

def _gpu(p: P, script: str, *args):
    """Chay script GPU: tren z87 qua vi24.gpu_run (probe vinmec, hook huy giua forward); khac: run thuong."""
    if p.machine == "z87":
        import vi24
        vi24.S = Path(__file__).resolve().parent
        vi24.gpu_run(script, *args)
    else:
        run_script(script, *args)


def stage_dense(p: P, a) -> None:
    import numpy as np
    t0 = time.time()
    clean = p.work / "emb_clean"
    clean.mkdir(parents=True, exist_ok=True)
    idx_ids = [int(x) for x in np.load(p.ix / "ids.npy")]
    pos = {d: i for i, d in enumerate(idx_ids)}
    if not list(clean.glob("blk_*.npz")):
        seen: dict = {}
        dirs = [ROOT / "work" / d for d in EMB_DIRS]
        for d in dirs:
            for f in sorted(d.glob("blk_*.npz")):
                z = np.load(f)
                for did, vec in zip(z["ids"], z["emb"]):
                    seen[int(did)] = vec
        keep = {d: v for d, v in seen.items() if d in pos}
        assert len(keep) == len(pos), f"emb {len(keep)} != chi muc {len(pos)}"
        bi = 0
        items = sorted(keep.items())
        for s in range(0, len(items), 8192):
            blk = items[s:s + 8192]
            np.savez(clean / f"blk_{bi:05d}.npz",
                     ids=np.asarray([d for d, _ in blk], dtype=np.int64),
                     emb=np.stack([v for _, v in blk]))
            bi += 1
        write_json(p.work / "emb_clean_report.json",
                   {"so_doc": len(keep), "blocks": bi, "nguon": [str(d) for d in dirs],
                    "bo_id_ngoai_chi_muc": len(seen) - len(keep)})
    write_json(p.work / "idx_map.json", {str(d): i for d, i in pos.items()})
    if not (p.dir / "dense_top200.json").exists():
        _gpu(p, "dense_topk.py", "--emb", clean, "--idx-from", p.work / "idx_map.json",
             "--queries", p.dir / "queries.jsonl", "--query-len", 256, "--top", 200,
             "--device", "cuda", "--out", p.dir / "dense_top200.json")
    if not (p.dir / "hyde_top200.json").exists():
        _gpu(p, "dense_topk.py", "--emb", clean, "--idx-from", p.work / "idx_map.json",
             "--queries", p.dir / "q_hyde.jsonl", "--query-len", 512, "--top", 200,
             "--device", "cuda", "--out", p.dir / "hyde_top200.json")
    extra = {}
    if a.ref_dense:
        ref = load_json(a.ref_dense)
        got = load_json(p.dir / "dense_top200.json")
        qs = sorted(set(got) & set(ref), key=lambda x: int(x))[:a.bench_n]
        ov = [len({str(h[0]) for h in got[q]} & {str(h[0]) for h in ref[q]}) for q in qs]
        extra["tai_lap_dense"] = {"so_cau_mau": len(qs), "TB_trung_top200": round(sum(ov) / max(len(ov), 1), 2),
                                  "min_trung": min(ov) if ov else 0, "ref": str(a.ref_dense)}
    write_manifest(p.dir, "dense", {"query_len": [256, 512], "top": 200, "emb_clean": str(clean)},
                   [p.dir / "queries.jsonl", p.dir / "q_hyde.jsonl"],
                   [p.dir / "dense_top200.json", p.dir / "hyde_top200.json"],
                   time.time() - t0, extra=extra, machine=p.machine)
    print(json.dumps(extra, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- main

STAGES = {"prep": stage_prep, "bm25": stage_bm25, "hyde": stage_hyde, "pool": stage_pool,
          "export_rr": stage_export_rr, "rr_merge": stage_rr_merge, "fuse": stage_fuse,
          "doclist": stage_doclist, "q8b_select": stage_q8b_select, "q8b_merge": stage_q8b_merge,
          "export_docwin": stage_export_docwin, "chunk": stage_chunk, "build": stage_build,
          "stats": stage_stats, "bench_prep": stage_bench_prep, "bench": stage_bench,
          "kq_check": stage_kq_check, "report": stage_report, "dense": stage_dense}
GPU_OUT_FILES = ["rerank_pool.json", "docwin.json.zst", "q8b_shard0_out.json", "q8b_bench_out.json"]


def done_already(p: P, st: str) -> bool:
    """Resume: manifest co du output, tham so chay khong doi, moi input van giu nguyen sha."""
    man = p.dir / f"manifest_{st}.json"
    if not man.exists():
        return False
    d = load_json(man)
    outs = d.get("outputs", [])
    if not outs or any(o.get("missing") for o in outs):
        return False
    if d.get("run_args") != rc.CURRENT_RUN_ARGS:
        return False
    for rec in d.get("inputs", []):
        if rec.get("missing"):
            continue
        cur = Path(rec["path"])
        if not cur.exists() or fingerprint(cur)["sha256"] != rec["sha256"]:
            return False
    return True


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=sorted(STAGES) + ["status"])
    ap.add_argument("stages", nargs="*", help="cac stage ke tiep (chay lan luot, dung o stage loi)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--machine", choices=["b660", "z87", "local"], default="b660")
    ap.add_argument("--queries", default=str(QUERY_DEFAULT))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--index", default=str(IX_DEFAULT))
    ap.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--work-root", default=str(WORK_ROOT_DEFAULT))
    ap.add_argument("--delta", type=int, default=None, help="delta co dinh (smoke: 56)")
    ap.add_argument("--delta0", type=int, default=56, help="khoi dau khi --find-delta")
    ap.add_argument("--max-zip", type=int, default=rc.MAX_ZIP)
    ap.add_argument("--q8b-shards", type=int, default=1)
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--ref-sub", default="")
    ap.add_argument("--ref-q8b", default="")
    ap.add_argument("--ref-dense", default="")
    ap.add_argument("--ref-pool", default=str(ROOT / "data/sub/diag30/cache_pool_v30.json"))
    ap.add_argument("--ref-docwin", default=str(ROOT / "data/sub/diag30/docwin_vi30.json.zst"))
    ap.add_argument("--ref-kq", default=str(ROOT / "data/sub/diag14/submission_c120in.json"))
    ap.add_argument("--ref-rows", default=str(ROOT / "data/sub/docwin/chunk_dw_u900.json.zst"))
    ap.add_argument("--ref-rrf", default=str(ROOT / "data/sub/vi34/cache_rrf3d2_v31.json"))
    ap.add_argument("--ref-doclist", default=str(ROOT / "data/sub/vi34/doclist_vi34.json"))
    ap.add_argument("--hyde-model", default="qwen35-9b")
    ap.add_argument("--union-dirs", nargs="*", default=[],
                    help="thu muc pool mode khac (vd full1200b) de hop cap 8B vao cung lan cham")
    ap.add_argument("--pool-mode", choices=list(rc.POOL_MODES), default="base",
                    help="base = BM25-200 U dense-200; p1 = BM25-300 U dense-200 U HyDE-200")
    a = ap.parse_args()
    rc.CURRENT_RUN_ARGS = {"delta": a.delta, "delta0": a.delta0, "max_zip": a.max_zip,
                           "smoke": a.smoke, "q8b_shards": a.q8b_shards, "bench_n": a.bench_n,
                           "queries": str(a.queries), "pool_mode": a.pool_mode}
    p = P(a.tag, a.out_root, a.work_root, a.index, a.machine)
    p.dir.mkdir(parents=True, exist_ok=True)
    if a.stage == "status":
        for m in sorted(p.dir.glob("manifest_*.json")):
            d = load_json(m)
            print(d["stage"], d["ts"], d["seconds"], "s", d.get("machine", ""))
        return
    todo = [a.stage, *a.stages]
    for st in todo:
        if done_already(p, st):
            print(f"{st}: da xong (resume): bo qua", flush=True)
            continue
        t0 = time.time()
        with machine_lock(p, st):
            STAGES[st](p, a)
        print("DONE", st, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
