"""Local launcher for the Kaggle rerank runner: pairs dataset -> push -> wait -> pull -> merge.

  python scripts/kaggle/launch_rerank.py make-pairs        # data/kaggle/rerank_pairs: vi top-200 + zh top-100 per query
  python scripts/kaggle/launch_rerank.py upload top1server/medret-rerank-pairs
  python scripts/kaggle/launch_rerank.py run configs/kaggle/rerank_qwen3-0.6b.json
Candidates (2026-10-26): Vietnamese docs in the order of the 0.3345 run (RRF of proxy g and whole-page hybrid, the
doc order of sub-vi-rrfwp-win2x10-v1), top-200; Chinese docs from the teammate's chunk-level dense ranking (tm_zh_q,
probe_recall_vi1000_en300_zh300), top-100. One shard = 50 queries. Page texts are read on Kaggle from the team snapshot,
so the dataset holds only ids. Run config: kernel, dataset_sources, accelerator, timeout_s, out_dir, runner (overrides of
DEFAULTS in rerank_runner.py). Shards already pulled into out_dir are skipped, so re-running resumes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from medret.config import PROJECT_ROOT, get_config
from medret.data.dataset import load_queries

RUNNER = PROJECT_ROOT / "scripts" / "kaggle" / "rerank_runner" / "rerank_runner.py"
PAIRS = PROJECT_ROOT / "data" / "kaggle" / "rerank_pairs"
_spec = importlib.util.spec_from_file_location("launch_embed", Path(__file__).parent / "launch_embed.py")
_le = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_le)
kaggle, status, TERMINAL = _le.kaggle, _le.status, _le.TERMINAL


def make_pairs(vi_top: int = 200, zh_top: int = 100, per_shard: int = 50, exclude: Path | None = None,
               out: Path = PAIRS) -> None:
    """exclude: a pairs folder already scored; its (query, doc) pairs are left out (deeper candidate pools)."""
    cfg = get_config()
    spec = importlib.util.spec_from_file_location("allocate_sub", PROJECT_ROOT / "scripts" / "analysis" / "allocate_sub.py")
    alloc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alloc)
    order, *_ = alloc.load(cfg, 7000, "rrf:wp_hybrid_q", top=vi_top,
                           scored=str(cfg.paths.dev_dir / "proxy_v6_6f" / "proxy_pairs_scored.parquet"))
    vi = pd.DataFrame([(q, d, r, "vi") for q, ds in order.items() for r, d in enumerate(ds)],
                      columns=["query_id", "doc_id", "src_rank", "lang"])
    zh = pd.read_parquet(cfg.paths.dev_dir / "bench" / "tm_zh_q.parquet")
    zh = zh[zh["rank"] < zh_top].rename(columns={"rank": "src_rank"}).assign(lang="zh")
    pairs = pd.concat([vi, zh[vi.columns]]).astype({"query_id": "int64", "doc_id": "int64", "src_rank": "int32"})
    pairs = pairs.drop_duplicates(["query_id", "doc_id"]).sort_values(["query_id", "lang", "src_rank"])
    if exclude is not None:
        done = pd.concat(pd.read_parquet(f, columns=["query_id", "doc_id"]) for f in sorted(exclude.glob("pairs-*.parquet")))
        pairs = pairs.merge(done.assign(_done=True), on=["query_id", "doc_id"], how="left")
        pairs = pairs[pairs["_done"].isna()].drop(columns="_done")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    qids = sorted(pairs["query_id"].unique())
    for k in range(0, len(qids), per_shard):
        part = pairs[pairs["query_id"].isin(qids[k: k + per_shard])]
        part.to_parquet(out / f"pairs-{k // per_shard:04d}.parquet", index=False)
    q = load_queries(cfg)[["id", "query"]]
    q.to_parquet(out / "queries.parquet", index=False)
    print(f"{len(pairs):,} pairs ({(pairs['lang'] == 'vi').sum():,} vi, {(pairs['lang'] == 'zh').sum():,} zh), "
          f"{len(qids)} queries in {(len(qids) + per_shard - 1) // per_shard} shards -> {out}")


def upload(slug: str, src: Path = PAIRS) -> None:
    (src / "dataset-metadata.json").write_text(json.dumps(  # isPrivate explicit: memory feedback-kaggle-private
        {"title": slug.split("/")[1], "id": slug, "licenses": [{"name": "other"}], "isPrivate": True}))
    listing = kaggle("datasets", "list", "--mine", "-s", slug.split("/")[1], "--csv", check=False)
    # exact ref match: a substring test took "...-pairs-long" for "...-pairs-long-pilot" (2026-10-28)
    exists = any(ln.split(",")[0] == slug for ln in listing.splitlines())
    args = ["datasets", "version", "-p", ".", "-m", "pairs"] if exists else ["datasets", "create", "-p", "."]
    print(kaggle(*args, cwd=src, check=False)[-400:])
    for _ in range(30):  # verify it is private once Kaggle has processed it
        time.sleep(20)
        out = kaggle("datasets", "list", "--mine", "-s", slug.split("/")[1], "--csv", check=False)
        hit = [ln for ln in out.splitlines() if ln.split(",")[0] == slug]
        if hit:
            print("listed under --mine:", hit[0][:160])
            break
    meta = Path(tempfile.mkdtemp())
    kaggle("datasets", "metadata", slug, "-p", str(meta), check=False)
    mf = meta / "dataset-metadata.json"
    m = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    print("isPrivate:", (m.get("info") or {}).get("isPrivate", "unknown (check the dataset page)"))


def render(cfg: dict, skip: list[str], dest: Path) -> None:
    runner_cfg = {**cfg.get("runner", {}), "run": cfg["run"], "skip_shards": skip}
    literal = json.dumps(json.dumps(runner_cfg))
    text = RUNNER.read_text(encoding="utf-8")
    src, n = re.subn(r'^CONFIG_JSON = "__CONFIG__".*$', lambda _: f"CONFIG_JSON = {literal}", text, count=1, flags=re.M)
    assert n == 1, "config marker not found in runner"
    (dest / "rerank_runner.py").write_text(src, encoding="utf-8")
    (dest / "kernel-metadata.json").write_text(json.dumps(dict(
        id=cfg["kernel"], title=cfg["kernel"].split("/")[1], code_file="rerank_runner.py", language="python",
        kernel_type="script", is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        dataset_sources=cfg["dataset_sources"], competition_sources=[], kernel_sources=[], model_sources=[]), indent=1))


def run(cfg_path: Path, no_push: bool = False) -> None:
    cfg = json.loads(cfg_path.read_text())
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", cfg["kernel"]):
        raise ValueError(f"kernel id {cfg['kernel']!r}: use lowercase letters, digits and '-' only")
    out_dir = PROJECT_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = sorted(p.name.replace(".done.json", "") for p in out_dir.glob("*.done.json"))
    print(f"run={cfg['run']} kernel={cfg['kernel']} already done locally: {len(skip)} shards")
    t0 = time.time()
    if not no_push:
        with tempfile.TemporaryDirectory() as td:
            render(cfg, skip, Path(td))
            print(kaggle("kernels", "push", "-p", td, "--accelerator", cfg.get("accelerator", "NvidiaTeslaT4"),
                         "-t", str(cfg.get("timeout_s", 41400))).strip()[-200:])
        time.sleep(15)
    while True:
        st = status(cfg["kernel"])
        print(f"  [{time.time() - t0:5.0f}s] {st}", flush=True)
        if st.startswith(TERMINAL):
            break
        time.sleep(60)
    _le.pull(cfg["kernel"], out_dir)
    sub = out_dir / "out"
    if sub.exists():
        for f in sub.iterdir():
            if f.is_file():
                f.replace(out_dir / f.name)
        shutil.rmtree(sub)
    summ = out_dir / "RUN_SUMMARY.json"
    print(summ.read_text() if summ.exists() else "no RUN_SUMMARY.json (kernel killed early?)")
    files = sorted(out_dir.glob("pairs-*.scores.parquet"))
    if files:
        allp = pd.concat(pd.read_parquet(f) for f in files)
        allp.to_parquet(out_dir / "scores.parquet", index=False)
        print(f"merged {len(files)} shards, {len(allp):,} pairs ({(~allp['has_text']).mean():.1%} without text) "
              f"-> {out_dir / 'scores.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("make-pairs")
    mp.add_argument("--vi-top", type=int, default=200)
    mp.add_argument("--zh-top", type=int, default=100)
    mp.add_argument("--exclude", type=Path, default=None, help="pairs folder already scored (deeper pools)")
    mp.add_argument("--out", type=Path, default=PAIRS)
    u = sub.add_parser("upload")
    u.add_argument("slug")
    u.add_argument("--src", type=Path, default=PAIRS)
    r = sub.add_parser("run")
    r.add_argument("config", type=Path)
    r.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "make-pairs":
        make_pairs(a.vi_top, a.zh_top, exclude=a.exclude, out=a.out)
    elif a.cmd == "upload":
        upload(a.slug, a.src)
    else:
        run(a.config, a.no_push)
