"""Corpus embedding runner for Kaggle 2xT4 (self-contained kernel script; rendered by scripts/kaggle/launch_embed.py).

Input : parquet shards `shard-*.parquet` with an int64 id column (`id_col`: `input_id` for the corpus export of
        `medret export-embed`, `chunk_id` for the trial shards) and `text`, the model input as given (for the corpus:
        BGE-M3 passage input "title heading_path chunk", see src/medret/index/embed_input.py; no prefix is added
        here unless `prefix` is set).
Output: per shard `<shard>.vec.npy` (fp16, N x D, L2-normalised), `<shard>.ids.npy` (int64 ids, row-aligned with
        vec), `<shard>.done.json` (written last).
Design (numbers from docs/experiments/20261010-embedding-throughput-kaggle-t4.md):
  * orchestrator supervises one worker per GPU; workers pull shards from a shared claim directory (dynamic balance);
  * resumable: finished shards (done.json) and `skip_shards` are never redone; stale claims of dead workers are released;
  * recycle: a worker exits between shards every `recycle_every_s` (frees CUDA context / allocator fragmentation / heap);
  * watchdog: RSS, system RAM, GPU memory, disk, temperature are sampled every `monitor_every_s`; soft limit => recycle,
    hard limit => kill + restart; OOM inside a batch => the batch is split and retried; non-finite vectors => shard fails;
  * deadline: no shard starts once elapsed + worst_factor * avg_shard_s passes the session budget
    (session_limit - startup - commit - safety), so everything is flushed long before the 12 h kill.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

CONFIG_JSON = "__CONFIG__"  # replaced by the launcher with a JSON string

DEFAULTS: dict = dict(
    # pool: cls | mean | last (decoder embedders such as Qwen3-Embedding)
    run="local", model="BAAI/bge-m3", pool="cls", prefix="", dtype="fp16", max_len=1024, budget=8192, id_col="chunk_id",
    input_glob="/kaggle/input/**/shard-*.parquet", out_dir="/kaggle/working/out", skip_shards=[],
    session_limit_s=43200, startup_s=180, commit_s=600, safety_s=1200, worst_factor=1.25, est_shard_s=600.0,
    target_work_s=0, max_output_gb=16.0, recycle_every_s=2400, recycle_every_shards=60,
    rss_soft_gb=8.0, rss_hard_gb=12.0, gpu_soft_gib=13.5, mem_avail_hard_gb=2.0, disk_min_free_gb=1.5,
    max_restarts=6, monitor_every_s=60, gpus=[0, 1], fault={},
)


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_JSON != "__CONFIG__":
        cfg.update(json.loads(CONFIG_JSON))
    return cfg


# ------------------------------------------------------------------------- pure logic (unit-tested)
def make_batches(lens: list[int], budget: int) -> list[list[int]]:
    """Length-sorted batches with at most `budget` padded tokens (a single longer item gets its own batch)."""
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in sorted(range(len(lens)), key=lens.__getitem__):
        if cur and (len(cur) + 1) * lens[i] > budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def start_limit_s(cfg: dict) -> float:
    """Latest elapsed time (since script start) at which a shard may still be *started*, before the shard-length margin."""
    hard = cfg["session_limit_s"] - cfg["startup_s"] - cfg["commit_s"] - cfg["safety_s"]
    return min(hard, cfg["target_work_s"]) if cfg["target_work_s"] else hard


def stop_reason(cfg, elapsed, ema_shard_s, out_gb, free_gb, since_spawn_s, shards_since_spawn, recycle_flag):
    """Why a worker must not take another shard (None = keep going). Deadline outranks recycle."""
    if out_gb >= cfg["max_output_gb"]:
        return "cap"
    if free_gb < cfg["disk_min_free_gb"]:
        return "disk"
    if elapsed + cfg["worst_factor"] * (ema_shard_s or cfg["est_shard_s"]) > start_limit_s(cfg):
        return "deadline"
    if recycle_flag or since_spawn_s >= cfg["recycle_every_s"] or shards_since_spawn >= cfg["recycle_every_shards"]:
        return "recycle"
    return None


def claim_next(names: list[str], out: Path, gpu: int) -> str | None:
    for s in names:
        if (out / f"{s}.done.json").exists() or (out / f"{s}.failed.json").exists():
            continue
        try:
            fd = os.open(out / "claims" / f"{s}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, str(gpu).encode())
        os.close(fd)
        return s
    return None


def release_claims(out: Path, gpu: int) -> list[str]:
    """Drop claims a dead worker left behind (finished/failed shards keep theirs)."""
    freed = []
    for c in (out / "claims").glob("*.claim"):
        s = c.name[: -len(".claim")]
        if c.read_text().strip() == str(gpu) and not (out / f"{s}.done.json").exists() \
                and not (out / f"{s}.failed.json").exists():
            c.unlink()
            freed.append(s)
    return freed


def load_shard(path: str, cfg: dict):
    """(ids as int64 ndarray, texts) of one input shard; the id column is configurable (`id_col`)."""
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(path, columns=[cfg["id_col"], "text"])
    return df[cfg["id_col"]].to_numpy(np.int64), [cfg["prefix"] + t for t in df["text"]]


def list_shards(cfg: dict) -> dict[str, str]:
    paths = sorted(glob.glob(cfg["input_glob"], recursive=True))
    return {Path(p).stem: p for p in paths}


def out_size_gb(out: Path) -> float:
    return sum(f.stat().st_size for f in out.glob("*.npy")) / 2**30


# ------------------------------------------------------------------------------------ monitoring
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def proc_rss_gb(pid: int) -> float:
    try:
        return next(int(x.split()[1]) for x in open(f"/proc/{pid}/status") if x.startswith("VmRSS")) / 2**20
    except (OSError, StopIteration):
        return 0.0


def mem_avail_gb() -> float:
    try:
        return next(int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("MemAvailable")) / 2**20
    except (OSError, StopIteration):
        return 1e9


def gpu_stats(phys: int) -> dict:
    q = "memory.used,temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active"
    try:
        r = subprocess.run(["nvidia-smi", "-i", str(phys), f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=False, timeout=20)
        m, t, c, p, th = [x.strip() for x in r.stdout.split(",")]
        return dict(mem_gib=float(m) / 1024, temp=float(t), mhz=float(c), watts=float(p), throttle=th)
    except Exception:  # noqa: BLE001
        return {}


# ----------------------------------------------------------------------------------------- worker
class NonFiniteError(RuntimeError):
    pass


def worker(cfg: dict, gpu: int, t0: float) -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    out = Path(cfg["out_dir"])
    shards = list_shards(cfg)
    names = [n for n in shards if n not in set(cfg["skip_shards"])]
    spawn_t, n_done = time.time(), 0
    ema_file = out / "status" / f"ema_gpu{gpu}.txt"
    ema = float(ema_file.read_text()) if ema_file.exists() else None
    fault = cfg["fault"]

    def why() -> str | None:
        return stop_reason(cfg, time.time() - t0, ema, out_size_gb(out), shutil.disk_usage(out).free / 2**30,
                           time.time() - spawn_t, n_done, (out / "ctl" / f"recycle_gpu{gpu}").exists())

    def finish(reason: str) -> None:
        (out / "status" / f"gpu{gpu}.json").write_text(json.dumps(dict(reason=reason, shards=n_done, ema=ema)))
        log(f"[gpu{gpu}] exit reason={reason} shards_this_spawn={n_done}")

    reason = why()
    if reason:  # decide before paying for a model load
        return finish(reason)

    dt = dict(fp16=torch.float16, fp32=torch.float32, bf16=torch.bfloat16)[cfg["dtype"]]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    try:
        model = AutoModel.from_pretrained(cfg["model"], dtype=dt, attn_implementation="sdpa")
    except TypeError:
        model = AutoModel.from_pretrained(cfg["model"], torch_dtype=dt, attn_implementation="sdpa")
    model = model.cuda().eval()
    dim = model.config.hidden_size

    def pool(h, mask):
        if cfg["pool"] == "cls":
            v = h[:, 0]
        elif cfg["pool"] == "last":  # decoder embedders (Qwen3-Embedding): last real token, right padding
            v = h[torch.arange(h.shape[0], device=h.device), mask.sum(1) - 1]
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(v.float(), dim=-1)

    @torch.inference_mode()
    def run_batch(enc, idx):
        try:
            marker = out / "status" / "fault_oom_done"
            if fault.get("oom_once") and not marker.exists() and len(idx) > 1:
                marker.write_text("1")
                raise torch.cuda.OutOfMemoryError("injected")
            width = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), width), tok.pad_token_id or 0, dtype=torch.long)
            am = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, : len(enc[i])] = torch.tensor(enc[i])
                am[r, : len(enc[i])] = 1
            ids, am = ids.cuda(non_blocking=True), am.cuda(non_blocking=True)
            return pool(model(input_ids=ids, attention_mask=am).last_hidden_state, am).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            log(f"[gpu{gpu}] OOM on batch of {len(idx)}: splitting")
            h = len(idx) // 2
            return np.concatenate([run_batch(enc, idx[:h]), run_batch(enc, idx[h:])])

    def process(name: str) -> dict:
        t1 = time.time()
        ids, texts = load_shard(shards[name], cfg)
        enc = tok(texts, truncation=True, max_length=cfg["max_len"])["input_ids"]
        if cfg["pool"] == "last" and tok.eos_token_id is not None:  # Qwen3-Embedding pools the closing EOS token
            enc = [e if e and e[-1] == tok.eos_token_id else (e[: cfg["max_len"] - 1] + [tok.eos_token_id]) for e in enc]
        lens = [len(x) for x in enc]
        vec = np.empty((len(enc), dim), dtype=np.float16)
        for b in make_batches(lens, cfg["budget"]):
            e = run_batch(enc, b)
            if not np.isfinite(e).all():
                raise NonFiniteError(name)
            vec[b] = e.astype(np.float16)
        secs = time.time() - t1
        for suffix, arr in (("vec", vec), ("ids", ids)):
            tmp = out / f"{name}.{suffix}.npy.tmp"
            with open(tmp, "wb") as f:
                np.save(f, arr)
            os.replace(tmp, out / f"{name}.{suffix}.npy")
        meta = dict(shard=name, n=len(enc), dim=dim, tokens=int(sum(lens)), secs=round(secs, 2),
                    tok_s=round(sum(lens) / secs), gpu=gpu, model=cfg["model"], dtype=cfg["dtype"])
        tmp = out / f"{name}.done.json.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, out / f"{name}.done.json")  # marker last: its presence means the shard is complete
        return meta

    failures = 0
    while True:
        reason = why()
        if reason:
            break
        name = claim_next(names, out, gpu)
        if name is None:
            reason = "done"
            break
        crash_marker = out / "status" / "fault_crash_done"
        if fault.get("crash_after") is not None and gpu == 0 and n_done == fault["crash_after"] \
                and not crash_marker.exists():
            crash_marker.write_text("1")
            log(f"[gpu{gpu}] injected crash while holding {name}")
            os._exit(137)
        try:
            meta = process(name)
            failures = 0
        except NonFiniteError:
            (out / f"{name}.failed.json").write_text(json.dumps(dict(shard=name, error="non-finite vectors")))
            log(f"[gpu{gpu}] shard {name} FAILED: non-finite vectors")
            failures += 1
            if failures >= 2:
                log(f"[gpu{gpu}] two consecutive failures, aborting this worker")
                sys.exit(3)
            continue
        ema = meta["secs"] if ema is None else 0.7 * ema + 0.3 * meta["secs"]
        ema_file.write_text(str(ema))
        n_done += 1
        log(f"[gpu{gpu}] {name}: {meta['n']} chunks, {meta['tokens']:,} tok, {meta['secs']}s, {meta['tok_s']:,} tok/s")
        gc.collect()
        torch.cuda.empty_cache()
    finish(reason)


# ----------------------------------------------------------------------------------- orchestrator
def orchestrate(cfg: dict) -> None:
    t0 = time.time()
    out = Path(cfg["out_dir"])
    for d in ("claims", "ctl", "status"):
        (out / d).mkdir(parents=True, exist_ok=True)
    shards = list_shards(cfg)
    todo = [n for n in shards if n not in set(cfg["skip_shards"])]
    log(f"run={cfg['run']} model={cfg['model']} shards total={len(shards)} todo={len(todo)} "
        f"start_limit={start_limit_s(cfg) / 3600:.2f}h (+{cfg['worst_factor']}x avg shard margin)")
    state = {g: dict(proc=None, restarts=0, done=False, reason=None) for g in cfg["gpus"]}
    stopping = {"v": False}

    def spawn(g: int) -> None:
        (out / "ctl" / f"recycle_gpu{g}").unlink(missing_ok=True)
        (out / "status" / f"gpu{g}.json").unlink(missing_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "TOKENIZERS_PARALLELISM": "false",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        cmd = [sys.executable, __file__, "--worker", "--gpu", str(g), "--t0", str(t0)]
        state[g]["proc"] = subprocess.Popen(cmd, env=env)

    def summarize() -> None:
        done = [json.loads(p.read_text()) for p in out.glob("*.done.json")]
        failed = [p.name.split(".")[0] for p in out.glob("*.failed.json")]
        finished = {d["shard"] for d in done} | set(failed) | set(cfg["skip_shards"])
        remaining = [n for n in shards if n not in finished]
        (out / "REMAINING.txt").write_text("\n".join(remaining))
        tokens, secs = sum(d["tokens"] for d in done), time.time() - t0
        (out / "RUN_SUMMARY.json").write_text(json.dumps(dict(
            run=cfg["run"], elapsed_s=round(secs), shards_done=len(done), shards_failed=failed, remaining=len(remaining),
            chunks=sum(d["n"] for d in done), tokens=tokens, tok_s_wall=round(tokens / max(secs, 1)),
            restarts={g: s["restarts"] for g, s in state.items()}, reasons={g: s["reason"] for g, s in state.items()},
            out_gb=round(out_size_gb(out), 2)), indent=1))

    def on_term(*_):
        stopping["v"] = True
        for s in state.values():
            if s["proc"] and s["proc"].poll() is None:
                s["proc"].terminate()

    signal.signal(signal.SIGTERM, on_term)
    for g in state:
        spawn(g)
    last_mon = 0.0
    while not all(s["done"] for s in state.values()) and not stopping["v"]:
        time.sleep(2)
        for g, s in state.items():
            if s["done"] or s["proc"].poll() is None:
                continue
            rc = s["proc"].returncode
            sf = out / "status" / f"gpu{g}.json"
            reason = json.loads(sf.read_text())["reason"] if sf.exists() else None
            if rc == 0 and reason == "recycle":
                log(f"[orch] gpu{g} recycled, respawning")
                spawn(g)
            elif rc == 0 and reason:
                s["done"], s["reason"] = True, reason
            else:
                freed = release_claims(out, g)
                s["restarts"] += 1
                log(f"[orch] gpu{g} worker died rc={rc}; released claims {freed}; "
                    f"restart {s['restarts']}/{cfg['max_restarts']}")
                if rc == 3 or s["restarts"] > cfg["max_restarts"]:
                    s["done"], s["reason"] = True, f"error rc={rc}"
                else:
                    spawn(g)
        if time.time() - last_mon >= cfg["monitor_every_s"]:
            last_mon = time.time()
            n_done = len(list(out.glob("*.done.json")))
            free = shutil.disk_usage(out).free / 2**30
            log(f"MON t={time.time() - t0:.0f}s shards={n_done}/{len(shards)} out={out_size_gb(out):.2f}G "
                f"disk_free={free:.1f}G sys_avail={mem_avail_gb():.1f}G")
            for g, s in state.items():
                if s["done"] or s["proc"].poll() is not None:
                    continue
                rss, gs = proc_rss_gb(s["proc"].pid), gpu_stats(g)
                log(f"MON gpu{g} rss={rss:.2f}G {gs}")
                if rss > cfg["rss_hard_gb"] or mem_avail_gb() < cfg["mem_avail_hard_gb"]:
                    log(f"[orch] HARD limit on gpu{g} (rss={rss:.1f}G): killing worker")
                    s["proc"].kill()
                elif rss > cfg["rss_soft_gb"] or gs.get("mem_gib", 0) > cfg["gpu_soft_gib"]:
                    log(f"[orch] soft limit on gpu{g}: recycle after current shard")
                    (out / "ctl" / f"recycle_gpu{g}").write_text("1")
    summarize()
    log(f"[orch] finished: {json.loads((out / 'RUN_SUMMARY.json').read_text())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--t0", type=float, default=0.0)
    a = ap.parse_args()
    conf = load_config()
    worker(conf, a.gpu, a.t0) if a.worker else orchestrate(conf)
