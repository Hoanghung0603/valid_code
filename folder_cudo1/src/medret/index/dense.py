"""Dense store for BGE-M3 vectors: an fp16 memmap ordered by input_id + a FAISS IVF-PQ index for candidates.

Layout under <index_dir>/dense/<run>/:
  vectors.f16.npy   N x D float16 (np.lib.format memmap), row = input_id, L2-normalised (as the runner wrote them)
  present.npy       N bool, rows already filled (sessions arrive one at a time; the store works when partial)
  ivfpq.faiss       inner-product IVF-PQ over the present rows, ids = input_id
  dense.json        run, dim, n, n_present, factory
Search = IVF-PQ top `k_coarse` (fast, compressed codes in RAM, ~70 bytes per vector) then exact re-scoring of
those candidates against the fp16 rows read from the memmap, so the final order is exact cosine. The full
vectors (~12 GB) never need to fit in RAM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

VECTORS, PRESENT, FAISS_FILE, META = "vectors.f16.npy", "present.npy", "ivfpq.faiss", "dense.json"


def assemble_vectors(emb_dir: Path, out_dir: Path, n_inputs: int, dim: int = 1024, log=print) -> int:
    """Copy every finished shard (`<shard>.done.json`) of a runner output into the memmap; idempotent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vpath, ppath = out_dir / VECTORS, out_dir / PRESENT
    if vpath.exists() and np.load(vpath, mmap_mode="r").shape[0] < n_inputs:
        _grow(vpath, ppath, n_inputs, dim, log)  # a crawl increment appended input ids
    mode = "r+" if vpath.exists() else "w+"
    vec = np.lib.format.open_memmap(vpath, mode=mode, dtype=np.float16, shape=None if mode == "r+" else (n_inputs, dim))
    if vec.shape != (n_inputs, dim):
        raise ValueError(f"{vpath} has shape {vec.shape}, expected {(n_inputs, dim)}")
    present = np.load(ppath) if ppath.exists() else np.zeros(n_inputs, dtype=bool)
    added = 0
    for done in sorted(emb_dir.glob("*.done.json")):
        name = json.loads(done.read_text())["shard"]
        ids = np.load(emb_dir / f"{name}.ids.npy")
        if present[ids].all():
            continue
        rows = np.load(emb_dir / f"{name}.vec.npy")
        if rows.shape != (len(ids), dim):
            raise ValueError(f"{name}: vec {rows.shape} vs {len(ids)} ids")
        vec[ids] = rows
        present[ids] = True
        added += len(ids)
    vec.flush()
    np.save(ppath, present)
    log(f"dense store: +{added:,} vectors, {int(present.sum()):,}/{n_inputs:,} present -> {vpath}")
    return added


def _grow_in_place(vpath: Path, n_inputs: int, dim: int) -> bool:
    """Append zero rows to a C-order .npy without copying it: rewrite the header with the new shape when it keeps
    its length (header padding absorbs the change, true while the row count keeps its digit count) and extend the
    file. Returns False when the header would change length; the caller then copies."""
    import io

    fmt = np.lib.format
    with open(vpath, "rb") as f:
        version = fmt.read_magic(f)
        if version not in ((1, 0), (2, 0)):
            return False
        read = fmt.read_array_header_1_0 if version == (1, 0) else fmt.read_array_header_2_0
        shape, fortran, dtype = read(f)
        offset = f.tell()
    if fortran or dtype != np.float16 or len(shape) != 2 or shape[1] != dim or shape[0] > n_inputs:
        return False
    # header written by hand at the file's own length: numpy's writer pads to its own (different) length
    # (128 bytes in the 2026-10 file vs 136 from numpy 2.4)
    len_bytes = 2 if version == (1, 0) else 4
    text = repr({"descr": fmt.dtype_to_descr(dtype), "fortran_order": False, "shape": (n_inputs, dim)})
    room = offset - len(fmt.magic(*version)) - len_bytes - 1  # 1: the final newline
    if len(text) > room:
        return False
    body = (text + " " * (room - len(text)) + "\n").encode("latin1")
    buf = io.BytesIO()
    buf.write(fmt.magic(*version))
    buf.write(len(body).to_bytes(len_bytes, "little"))
    buf.write(body)
    if buf.tell() != offset:
        return False
    with open(vpath, "r+b") as f:
        f.write(buf.getvalue())
        f.truncate(offset + n_inputs * dim * dtype.itemsize)  # new rows read as zeros
    return True


def _grow(vpath: Path, ppath: Path, n_inputs: int, dim: int, log, block: int = 500_000) -> None:
    """Make the memmap n_inputs rows long (new rows zero, not present); input ids only ever append.
    In place when possible: the copy needs a second full-size file (17.6 GB for 8.4M rows, 2026-10-20)."""
    if not _grow_in_place(vpath, n_inputs, dim):
        old = np.load(vpath, mmap_mode="r")
        tmp = vpath.with_suffix(".grow.npy")
        new = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n_inputs, dim))
        for s in range(0, old.shape[0], block):
            e = min(s + block, old.shape[0])
            new[s:e] = old[s:e]
        new.flush()
        del new, old
        tmp.replace(vpath)
    present = np.load(ppath) if ppath.exists() else np.zeros(0, dtype=bool)
    grown = np.zeros(n_inputs, dtype=bool)
    grown[: len(present)] = present
    np.save(ppath, grown)
    log(f"dense store grown to {n_inputs:,} rows ({int(grown.sum()):,} present)")


def build_ivfpq(out_dir: Path, factory: str = "IVF8192,PQ64", train_size: int = 400_000, batch: int = 200_000,
                seed: int = 0, run: str = "", scope_inputs: np.ndarray | None = None, log=print) -> None:
    """FAISS over the present vectors (only those in scope_inputs, if given: index/scope.py)."""
    import faiss

    vec = np.load(out_dir / VECTORS, mmap_mode="r")
    present = np.load(out_dir / PRESENT)
    ids = np.flatnonzero(present)
    if scope_inputs is not None:
        ids = np.intersect1d(ids, scope_inputs)
    if not len(ids):
        raise ValueError("no vectors present: run assemble first")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    train = np.sort(rng.choice(ids, size=min(train_size, len(ids)), replace=False))
    index = faiss.index_factory(vec.shape[1], factory, faiss.METRIC_INNER_PRODUCT)
    index.train(np.ascontiguousarray(vec[train], dtype=np.float32))
    log(f"trained {factory} on {len(train):,} vectors in {time.time() - t0:,.0f}s")
    for s in range(0, len(ids), batch):
        part = ids[s : s + batch]
        index.add_with_ids(np.ascontiguousarray(vec[part], dtype=np.float32), part.astype(np.int64))
    faiss.write_index(index, str(out_dir / FAISS_FILE))
    (out_dir / META).write_text(json.dumps({"run": run, "dim": int(vec.shape[1]), "n": int(vec.shape[0]),
                                            "n_present": int(len(ids)), "factory": factory,
                                            "scoped": scope_inputs is not None}))
    log(f"indexed {len(ids):,} vectors in {time.time() - t0:,.0f}s -> {out_dir / FAISS_FILE}")


class DenseIndex:
    def __init__(self, out_dir: Path, nprobe: int = 64):
        import faiss

        self.vec = np.load(out_dir / VECTORS, mmap_mode="r")
        self.index = faiss.read_index(str(out_dir / FAISS_FILE))
        faiss.extract_index_ivf(self.index).nprobe = nprobe
        self.meta = json.loads((out_dir / META).read_text())

    def search(self, queries: np.ndarray, k: int = 100, k_coarse: int = 1000) -> list[list[tuple[int, float]]]:
        """Per query: (input_id, exact cosine) of the top-k inputs. Queries must be L2-normalised float32."""
        q = np.ascontiguousarray(queries, dtype=np.float32)
        _, cand = self.index.search(q, max(k, k_coarse))
        out = []
        for qi, row in enumerate(cand):
            row = row[row >= 0]
            order = np.argsort(row)  # sorted reads are kinder to the memmap
            ids = row[order]
            scores = self.vec[ids].astype(np.float32) @ q[qi]
            top = np.argsort(-scores)[:k]
            out.append([(int(ids[i]), float(scores[i])) for i in top])
        return out
