"""Local launcher for the Kaggle embedding runner: push -> wait -> pull -> verify.

  python scripts/kaggle/launch_embed.py make-trial-shards     # build + upload the small trial dataset
  python scripts/kaggle/launch_embed.py upload-dataset data/embed_inputs ducanger/medret-embed-inputs-v1
  python scripts/kaggle/launch_embed.py run configs/kaggle/embed_corpus_s1.json   # then embed_corpus.json to finish
  python scripts/kaggle/launch_embed.py run CONFIG --part 1/2 & ... --part 2/2   # n parallel kernels (slug-p1, -p2)

Run config (JSON): kernel (owner/slug), dataset_sources, model_sources, accelerator, timeout_s (push -t, outer safety net),
out_dir (local, relative to project root), inputs_dir (optional, local copy of the input shards: every pulled shard is
checked row by row against it), runner (overrides of DEFAULTS in embed_runner.py). Shards already present in
out_dir (done.json) are passed as `skip_shards`, so re-running the same command resumes. The Kaggle token is read from
.env (KAGGLE_API_KEY) and never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from medret.config import PROJECT_ROOT

RUNNER = PROJECT_ROOT / "scripts" / "kaggle" / "embed_runner" / "embed_runner.py"
BENCH = PROJECT_ROOT / "scripts" / "kaggle" / "datasets" / "bench-chunks" / "chunks.parquet"
TRIAL_DIR = PROJECT_ROOT / "scripts" / "kaggle" / "datasets" / "embed-trial"
TERMINAL = ("COMPLETE", "ERROR", "CANCEL")


def kaggle_env() -> dict:
    # .env may hold several accounts (KAGGLE_API_KEY, KAGGLE_API_KEY_1, ...): MEDRET_KAGGLE_KEY names the one to use
    var = os.environ.get("MEDRET_KAGGLE_KEY", "KAGGLE_API_KEY")
    token = next(ln.split("=", 1)[1].strip() for ln in (PROJECT_ROOT / ".env").read_text().splitlines()
                 if ln.startswith(f"{var}="))
    # the CLI keeps upload resume files under %TEMP%/.kaggle/uploads and fails on Windows if the folder is missing
    (Path(tempfile.gettempdir()) / ".kaggle" / "uploads").mkdir(parents=True, exist_ok=True)
    return {**os.environ, "KAGGLE_API_TOKEN": token, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def kaggle(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    # explicit utf-8 + replace: with the Windows locale codec, progress-bar bytes made the reader thread die and
    # stdout came back as None
    r = subprocess.run(["kaggle", *args], env=kaggle_env(), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", check=False, cwd=cwd)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def status(kernel: str) -> str:
    out = kaggle("kernels", "status", kernel, check=False)
    m = re.search(r'status "KernelWorkerStatus\.(\w+)"', out)
    return m.group(1) if m else out.strip()[-80:]


# ------------------------------------------------------------------------------------------ trial data
def make_trial_shards(shard_size: int = 960) -> None:
    """Bench chunks twice (unique chunk_id): chunk i and i+N have identical text, which lets us verify the index mapping."""
    df = pd.read_parquet(BENCH, columns=["text"])
    df = pd.concat([df, df], ignore_index=True)
    df.insert(0, "chunk_id", np.arange(len(df), dtype=np.int64))
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    for k, start in enumerate(range(0, len(df), shard_size)):
        df.iloc[start : start + shard_size].to_parquet(TRIAL_DIR / f"shard-{k:04d}.parquet", index=False)
    (TRIAL_DIR / "dataset-metadata.json").write_text(json.dumps(
        {"title": "medret-embed-trial", "id": "ducanger/medret-embed-trial", "licenses": [{"name": "other"}]}))
    print(f"{len(df)} chunks in {k + 1} shards -> {TRIAL_DIR}")
    verb = "version" if "ducanger/medret-embed-trial" in kaggle("datasets", "list", "--mine", "-s", "medret-embed-trial") \
        else "create"
    args = ["datasets", verb, "-p", ".", *(["-m", "trial shards"] if verb == "version" else [])]
    print(kaggle(*args, cwd=TRIAL_DIR, check=False)[-300:])


def upload_dataset(src: Path, slug: str, title: str | None = None) -> None:
    """Create (or add a version of) a private Kaggle dataset from the shard-*.parquet files in `src`."""
    files = sorted(src.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no shard-*.parquet in {src}")
    (src / "dataset-metadata.json").write_text(json.dumps(  # isPrivate explicit: see memory feedback-kaggle-private
        {"title": title or slug.split("/")[1], "id": slug, "licenses": [{"name": "other"}], "isPrivate": True}))
    exists = slug in kaggle("datasets", "list", "--mine", "-s", slug.split("/")[1], check=False)
    args = ["datasets", "version", "-p", ".", "-m", f"{len(files)} shards"] if exists else ["datasets", "create", "-p", "."]
    size = sum(f.stat().st_size for f in files) / 2**30
    print(f"{'versioning' if exists else 'creating'} private dataset {slug}: {len(files)} shards, {size:.2f} GB")
    print(kaggle(*args, cwd=src, check=False)[-400:])


# ---------------------------------------------------------------------------------------------- run
def render_kernel(cfg: dict, skip: list[str], dest: Path) -> None:
    runner_cfg = {**cfg.get("runner", {}), "run": cfg["run"], "skip_shards": skip}
    literal = json.dumps(json.dumps(runner_cfg))
    text = RUNNER.read_text(encoding="utf-8")
    src, n = re.subn(r'^CONFIG_JSON = "__CONFIG__".*$', lambda _: f"CONFIG_JSON = {literal}", text, count=1, flags=re.M)
    assert n == 1, "config marker not found in runner"
    (dest / "embed_runner.py").write_text(src, encoding="utf-8")
    slug = cfg["kernel"].split("/")[1]
    (dest / "kernel-metadata.json").write_text(json.dumps(dict(
        id=cfg["kernel"], title=slug, code_file="embed_runner.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=cfg.get("internet", True),
        dataset_sources=cfg.get("dataset_sources", []), competition_sources=[], kernel_sources=[],
        model_sources=cfg.get("model_sources", [])), indent=1))


def verify(out_dir: Path) -> bool:
    ok, rows, chunk_ids = True, 0, []
    dones = sorted(out_dir.glob("*.done.json"))
    for d in dones:
        meta = json.loads(d.read_text())
        vec, ids = np.load(out_dir / f"{meta['shard']}.vec.npy"), np.load(out_dir / f"{meta['shard']}.ids.npy")
        norms = np.linalg.norm(vec.astype(np.float32), axis=1)
        bad = (vec.shape != (meta["n"], meta["dim"])) or len(ids) != meta["n"] or not np.isfinite(vec).all() \
            or not np.allclose(norms, 1.0, atol=5e-3)
        if bad:
            ok = False
            print(f"  BAD shard {meta['shard']}: shape={vec.shape} n={meta['n']} norms[{norms.min():.3f},{norms.max():.3f}]")
        rows += meta["n"]
        chunk_ids.append(ids)
    allids = np.concatenate(chunk_ids) if chunk_ids else np.array([], dtype=np.int64)
    dup = len(allids) - len(np.unique(allids))
    verdict = "OK" if ok and dup == 0 else "BAD"
    print(f"verify: {len(dones)} shards, {rows} rows, duplicate ids={dup}, structure {verdict}")
    return ok and dup == 0


def verify_against_inputs(out_dir: Path, inputs_dir: Path, id_col: str) -> bool:
    """Every finished shard must hold exactly its input shard's ids, in input order (vec row k = input row k)."""
    bad, done = [], sorted(out_dir.glob("*.done.json"))
    for d in done:
        name = json.loads(d.read_text())["shard"]
        want = pd.read_parquet(inputs_dir / f"{name}.parquet", columns=[id_col])[id_col].to_numpy(np.int64)
        if not np.array_equal(np.load(out_dir / f"{name}.ids.npy"), want):
            bad.append(name)
    total = len(list(inputs_dir.glob("shard-*.parquet")))
    print(f"inputs check: {len(done)}/{total} shards embedded, {len(bad)} with ids not matching the input {bad[:5]}")
    return not bad


def verify_trial_pairs(out_dir: Path, n_unique: int = 11506, sample: int = 300) -> None:
    """Chunk i and i + n_unique have the same text but sit in different shards/batches: vectors must match."""
    vecs: dict[int, np.ndarray] = {}
    for ids_f in out_dir.glob("*.ids.npy"):
        ids, vec = np.load(ids_f), np.load(str(ids_f).replace(".ids.", ".vec."))
        vecs.update({int(i): vec[k] for k, i in enumerate(ids)})
    rng = np.random.default_rng(0)
    cands = [i for i in range(n_unique) if i in vecs and (i + n_unique) in vecs]
    if not cands:
        print("pair check: no complete pairs yet")
        return
    pick = rng.choice(cands, size=min(sample, len(cands)), replace=False)
    cos = [float(np.dot(vecs[i].astype(np.float32), vecs[i + n_unique].astype(np.float32))) for i in pick]
    print(f"pair check ({len(pick)} identical-text pairs, different shards): "
          f"cosine min={min(cos):.5f} mean={np.mean(cos):.5f}")


def pull(kernel: str, out_dir: Path) -> None:
    """`kernels output` lists 20 files per page by default; page through all of them (3 files per shard)."""
    token, pages = None, 0
    while True:
        args = ["kernels", "output", kernel, "-p", str(out_dir), "-o", "-q", "--page-size", "200"]
        text = kaggle(*args, *(["--page-token", token] if token else []), check=False)
        pages += 1
        m = re.search(r"Next Page Token = (\S+)", text)
        if not m or pages > 200:
            break
        token = m.group(1)
    print(f"pulled {len(list(out_dir.rglob('*.npy')))} .npy files in {pages} page(s) -> {out_dir}")


def run(cfg_path: Path, no_push: bool = False, part: str | None = None) -> None:
    """part "i/n": this kernel embeds every n-th input shard (from inputs_dir), offset i-1, under kernel slug
    <kernel>-p<i>, so n launchers in parallel split one job across n Kaggle sessions (same out_dir). The weekly
    GPU quota is spent the same; the wait is ~n times shorter if the account may run n GPU sessions at once."""
    cfg = json.loads(cfg_path.read_text())
    if part:
        i, n = (int(x) for x in part.split("/"))
        if not (cfg.get("inputs_dir") and 1 <= i <= n):
            raise ValueError("--part needs inputs_dir in the config and 1 <= i <= n")
        names = sorted(f.stem for f in (PROJECT_ROOT / cfg["inputs_dir"]).glob("shard-*.parquet"))
        other = [s for k, s in enumerate(names) if k % n != i - 1]
        cfg["kernel"] = f"{cfg['kernel']}-p{i}"
        cfg["run"] = f"{cfg['run']}-p{i}"
    else:
        other = []
    # Kaggle rewrites slugs (".", "_" -> "-"); polling the unrewritten id then fails with a permission error
    # forever (probe_qwen3-emb-0.6b, 2026-10-14), so refuse such ids up front
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", cfg["kernel"]):
        raise ValueError(f"kernel id {cfg['kernel']!r}: use lowercase letters, digits and '-' only")
    out_dir = PROJECT_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(json.loads(p.read_text())["shard"] for p in out_dir.glob("*.done.json"))
    skip = sorted(set(done) | set(other))
    print(f"run={cfg['run']} kernel={cfg['kernel']} already done locally: {len(done)} shards, "
          f"other parts: {len(other)} shards (skipped)")
    t0 = time.time()
    if not no_push:
        with tempfile.TemporaryDirectory() as td:
            render_kernel(cfg, skip, Path(td))
            print(kaggle("kernels", "push", "-p", td, "--accelerator", cfg.get("accelerator", "NvidiaTeslaT4"),
                         "-t", str(cfg.get("timeout_s", 41400))).strip()[-160:])
        time.sleep(10)
    while True:
        st = status(cfg["kernel"])
        print(f"  [{time.time() - t0:5.0f}s] {st}", flush=True)
        if st.startswith(TERMINAL):
            break
        time.sleep(20)
    pull(cfg["kernel"], out_dir)
    sub = out_dir / "out"  # outputs are written under /kaggle/working/out
    if sub.exists():
        for f in sub.iterdir():
            if f.is_file():  # claims/, ctl/, status/ are runner-internal state, not results
                f.replace(out_dir / f.name)
        shutil.rmtree(sub)
    summ = out_dir / "RUN_SUMMARY.json"
    print(summ.read_text() if summ.exists() else "no RUN_SUMMARY.json (kernel killed early?)")
    verify(out_dir)
    if cfg.get("inputs_dir"):
        verify_against_inputs(out_dir, PROJECT_ROOT / cfg["inputs_dir"], cfg.get("runner", {}).get("id_col", "chunk_id"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-trial-shards")
    u = sub.add_parser("upload-dataset")
    u.add_argument("src", type=Path)
    u.add_argument("slug", help="owner/dataset-slug")
    r = sub.add_parser("run")
    r.add_argument("config", type=Path)
    r.add_argument("--no-push", action="store_true", help="only wait for / pull the latest run")
    r.add_argument("--part", default=None, help="i/n: run part i of n in its own kernel (parallel sessions)")
    v = sub.add_parser("verify")
    v.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "make-trial-shards":
        make_trial_shards()
    elif a.cmd == "upload-dataset":
        upload_dataset(a.src, a.slug)
    elif a.cmd == "run":
        run(a.config, a.no_push, a.part)
    else:
        verify(a.out_dir) and verify_trial_pairs(a.out_dir)
    sys.exit(0)
