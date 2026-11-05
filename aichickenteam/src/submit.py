"""Submission validation + zipping.

* ``validate(pred, query_ids=None, doc_lookup=None, raise_on_error=False) -> list[str]``
  returns a list of error strings (empty = valid).
* ``complete(pred, query_ids)`` -> new list with one well-formed entry per query id
  (missing ids get empty lists, bad fields are coerced/dropped), sorted by id.
* ``write_zip(pred, path, json_name="submission.json")`` writes a zip with exactly
  one UTF-8 JSON (``ensure_ascii=False``) at the root and re-reads it to verify.
* ``read_zip(path)``.

CLI::

    python -m src.submit check submission.zip [--queries data/ViBioMIR/query.parquet] [--docs data/docs]
    python -m src.submit zip pred.json submission.zip [--queries ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

DocLookup = Mapping[str, str] | Callable[[str], "str | None"]


def _lookup(doc_lookup: DocLookup, doc_id: str) -> str | None:
    if callable(doc_lookup):
        return doc_lookup(doc_id)
    return doc_lookup.get(doc_id)


def validate(pred: Any, query_ids: Iterable[int] | None = None, doc_lookup: DocLookup | None = None,
             raise_on_error: bool = False, max_errors: int = 200) -> list[str]:
    """Check the submission structure. Returns a list of error messages."""
    errs: list[str] = []

    def err(msg: str) -> None:
        errs.append(msg)

    if not isinstance(pred, list):
        err(f"top level must be a list, got {type(pred).__name__}")
        pred = []
    seen: set[int] = set()
    for i, e in enumerate(pred):
        where = f"entry[{i}]"
        if not isinstance(e, dict):
            err(f"{where}: not an object")
            continue
        qid = e.get("id")
        if isinstance(qid, bool) or not isinstance(qid, int):
            err(f"{where}: id must be an integer, got {qid!r}")
        else:
            where = f"id={qid}"
            if qid in seen:
                err(f"{where}: duplicate id")
            seen.add(qid)
        for field in ("relevant_docs", "relevant_chunks"):
            if field not in e:
                err(f"{where}: missing field {field}")
            elif not isinstance(e[field], list):
                err(f"{where}: {field} must be a list")
        docs = e.get("relevant_docs") if isinstance(e.get("relevant_docs"), list) else []
        for d in docs:
            if not isinstance(d, str) or not d:
                err(f"{where}: relevant_docs item must be a non-empty string, got {d!r}")
        chunks = e.get("relevant_chunks") if isinstance(e.get("relevant_chunks"), list) else []
        docset = set(docs)
        for j, c in enumerate(chunks):
            cw = f"{where} chunk[{j}]"
            if not isinstance(c, dict):
                err(f"{cw}: not an object")
                continue
            did, txt = c.get("doc_id"), c.get("chunk_text")
            if not isinstance(did, str) or not did:
                err(f"{cw}: doc_id must be a non-empty string, got {did!r}")
            if not isinstance(txt, str) or not txt.strip():
                err(f"{cw}: chunk_text must be a non-empty string")
            if isinstance(did, str) and did not in docset:
                err(f"{cw}: doc_id {did} not in relevant_docs")
            if doc_lookup is not None and isinstance(did, str) and isinstance(txt, str) and txt:
                text = _lookup(doc_lookup, did)
                if text is None:
                    err(f"{cw}: doc {did} not found in doc lookup")
                elif txt not in text:
                    err(f"{cw}: chunk_text is not a verbatim substring of doc {did}")
        if len(errs) >= max_errors:
            errs.append("... too many errors, stopping")
            break
    if query_ids is not None:
        qs = {int(q) for q in query_ids}
        missing = sorted(qs - seen)
        extra = sorted(seen - qs)
        if missing:
            err(f"{len(missing)} query ids missing, e.g. {missing[:10]}")
        if extra:
            err(f"{len(extra)} unknown query ids, e.g. {extra[:10]}")
    if raise_on_error and errs:
        raise ValueError("invalid submission:\n  " + "\n  ".join(errs[:50]))
    return errs


def complete(pred: Iterable[dict], query_ids: Iterable[int]) -> list[dict]:
    """Return one well-formed entry per query id (missing -> empty lists), sorted by id."""
    by_id: dict[int, dict] = {}
    for e in pred or []:
        try:
            by_id[int(e["id"])] = e
        except (KeyError, TypeError, ValueError):
            continue
    out = []
    for qid in sorted({int(q) for q in query_ids}):
        e = by_id.get(qid) or {}
        docs, seen = [], set()
        for d in e.get("relevant_docs") or []:
            d = str(d) if d is not None else ""
            if d and d not in seen:
                seen.add(d)
                docs.append(d)
        chunks, seen_c = [], set()
        for c in e.get("relevant_chunks") or []:
            if not isinstance(c, dict):
                continue
            did, txt = c.get("doc_id"), c.get("chunk_text")
            if did is None or not isinstance(txt, str) or not txt.strip():
                continue
            did = str(did)
            if (did, txt) in seen_c:
                continue
            seen_c.add((did, txt))
            chunks.append({"doc_id": did, "chunk_text": txt})
            if did not in seen:
                seen.add(did)
                docs.append(did)
        out.append({"id": qid, "relevant_docs": docs, "relevant_chunks": chunks})
    return out


def write_zip(pred: list[dict], path: str | Path, json_name: str = "submission.json") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_name = Path(json_name).name
    if not json_name.endswith(".json"):
        json_name += ".json"
    data = json.dumps(pred, ensure_ascii=False).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(json_name, data)
    tmp.replace(path)
    if read_zip(path) != pred:  # verify round trip
        raise RuntimeError(f"zip verification failed for {path}")
    return path


def read_zip(path: str | Path) -> Any:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        files = [n for n in names if not n.endswith("/")]
        if len(files) != 1 or "/" in files[0] or not files[0].endswith(".json"):
            raise ValueError(f"{path}: zip must contain exactly one .json at root, got {names}")
        return json.loads(zf.read(files[0]).decode("utf-8"))


# ------------------------------------------------------------------ helpers
def load_query_ids(path: str | Path) -> list[int]:
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        return [int(x) for x in pq.read_table(p, columns=["id"]).column("id").to_pylist()]
    if p.suffix == ".jsonl":
        return [int(json.loads(l)["id"]) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    return [int(e["id"]) for e in json.loads(p.read_text(encoding="utf-8"))]


def load_doc_texts(doc_ids: Iterable[str], paths: str | Path | list = "data/docs") -> dict[str, str]:
    """doc_id -> text for the requested ids only, from Doc parquet file(s)/dir(s)."""
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    want = sorted({str(d) for d in doc_ids})
    if not want:
        return {}
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    if not files:
        return {}
    dset = ds.dataset([str(f) for f in files], format="parquet")
    import pyarrow as pa
    tbl = dset.to_table(columns=["doc_id", "text"],
                        filter=pc.field("doc_id").isin(pa.array(want, pa.string())))
    return dict(zip(tbl.column("doc_id").to_pylist(), tbl.column("text").to_pylist()))


def _report(errs: list[str], pred: list) -> int:
    n_docs = sum(len(e.get("relevant_docs") or []) for e in pred if isinstance(e, dict))
    n_chunks = sum(len(e.get("relevant_chunks") or []) for e in pred if isinstance(e, dict))
    empty = sum(1 for e in pred if isinstance(e, dict) and not e.get("relevant_docs"))
    print(f"entries={len(pred)} docs={n_docs} chunks={n_chunks} empty_doc_entries={empty}")
    if errs:
        print(f"INVALID: {len(errs)} error(s)")
        for e in errs[:50]:
            print("  -", e)
        return 1
    print("OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="validate / build submission zips")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate an existing zip or json")
    c.add_argument("path")
    c.add_argument("--queries", default="data/ViBioMIR/query.parquet")
    c.add_argument("--docs", help="Doc parquet dir/file: also verify chunk_text is verbatim")
    z = sub.add_parser("zip", help="validate a json and write a zip")
    z.add_argument("json_path")
    z.add_argument("out")
    z.add_argument("--queries", default="data/ViBioMIR/query.parquet")
    z.add_argument("--complete", action="store_true", help="fill missing ids with empty lists")
    a = ap.parse_args(argv)

    qids = load_query_ids(a.queries) if a.queries and Path(a.queries).exists() else None
    if qids is None:
        print(f"[submit] warning: query file {a.queries} not found, not checking id coverage")
    if a.cmd == "check":
        if a.path.endswith(".zip"):
            try:
                pred = read_zip(a.path)
            except Exception as e:  # noqa: BLE001
                print(f"INVALID zip: {e}")
                return 1
        else:
            pred = json.loads(Path(a.path).read_text(encoding="utf-8"))
        lookup = None
        if a.docs:
            ids = {c["doc_id"] for e in pred if isinstance(e, dict)
                   for c in (e.get("relevant_chunks") or []) if isinstance(c, dict)}
            lookup = load_doc_texts(ids, a.docs)
        return _report(validate(pred, qids, lookup), pred if isinstance(pred, list) else [])
    pred = json.loads(Path(a.json_path).read_text(encoding="utf-8"))
    if a.complete and qids is not None:
        pred = complete(pred, qids)
    errs = validate(pred, qids)
    rc = _report(errs, pred)
    if rc == 0:
        write_zip(pred, a.out)
        print(f"wrote {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
