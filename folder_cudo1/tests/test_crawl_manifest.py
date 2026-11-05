import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from medret.crawl import build_aliases, build_manifest, iter_pages


def _write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [("id", pa.int64()), ("url", pa.string()), ("final_url", pa.string()), ("status", pa.int32()),
         ("label", pa.string()), ("detail", pa.string()), ("ctype", pa.string()), ("ms", pa.int32()),
         ("fetched_at", pa.int64()), ("html", pa.binary())]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _row(id_, url, t, label="ok", detail="", html=b"<p>x</p>"):
    return dict(id=id_, url=url, final_url=url, status=200, label=label, detail=detail, ctype="text/html",
                ms=1, fetched_at=t, html=html if label == "ok" else None)


def test_manifest_dedup_source_and_pages(tmp_path):
    _write_shard(tmp_path / "pages/a.vn/part-1.parquet", [
        _row(1, "https://a.vn/1", 100, html=b"old"),
        _row(2, "https://a.vn/2", 100, label="http_error:404"),
        _row(3, "https://a.vn/3", 100, detail="archive:wb:20240101 ok", html=b"wb"),
    ])
    _write_shard(tmp_path / "pages/a.vn/part-2.parquet", [
        _row(1, "https://a.vn/1", 200, html=b"new"),
        _row(3, "https://a.vn/3", 200, label="net_error:timeout"),  # later failure must not hide the ok copy
    ])
    (tmp_path / "pages/_plan.parquet").write_bytes(b"not a shard")

    m = build_manifest(tmp_path, workers=1).set_index("doc_id")
    assert list(m.index) == [1, 2, 3]
    assert m.loc[1, "shard"] == "pages/a.vn/part-2.parquet" and m.loc[1, "length"] == 3
    assert m.loc[2, "error"] == "http_error:404" and m.loc[2, "sha1"] == ""
    assert m.loc[3, "source"] == "wayback" and m.loc[3, "error"] == ""

    pages = {d: html for d, html, _ in iter_pages(tmp_path, m.reset_index())}
    assert pages == {1: b"new", 3: b"wb"}

    corpus = pd.DataFrame({"id": [1, 2, 3, 4], "url": ["https://a.vn/1", "https://a.vn/2", "https://a.vn/3",
                                                       "http://www.a.vn/1/"]})
    aliases = build_aliases(corpus, m.reset_index()).set_index("id")
    assert aliases.loc[4, "fetched_id"] == 1
    assert pd.isna(aliases.loc[2, "fetched_id"])


def test_snapshot_never_replaces_an_ok_page_of_an_earlier_pack(tmp_path):
    _write_shard(tmp_path / "pages/a.vn/part-1.parquet", [
        _row(1, "https://a.vn/1", 100, html=b"old"),
        _row(2, "https://a.vn/2", 100, label="http_error:503"),
    ])
    _write_shard(tmp_path / "pages/snap_20261012/html_vi-0000.parquet", [
        _row(1, "https://a.vn/1", 900, html=b"refetched"),  # later fetch of an already parsed page: ignored
        _row(2, "https://a.vn/2", 900, html=b"now ok"),  # failed before, ok now: taken
        _row(5, "https://a.vn/5", 900, label="net_error:timeout"),
        _row(5, "https://a.vn/5", 950, html=b"retry ok"),  # retry within the snapshot: latest ok
    ])
    m = build_manifest(tmp_path, workers=1).set_index("doc_id")
    assert m.loc[1, "shard"] == "pages/a.vn/part-1.parquet" and m.loc[1, "pack_rank"] == 0
    assert m.loc[2, "shard"].startswith("pages/snap_20261012/") and m.loc[2, "error"] == ""
    assert m.loc[5, "error"] == "" and m.loc[5, "offset"] == 3
    pages = {d: html for d, html, _ in iter_pages(tmp_path, m.reset_index())}
    assert pages == {1: b"old", 2: b"now ok", 5: b"retry ok"}
