"""Pipeline hai ngôn ngữ: truy vấn VI -> chỉ mục VI; bản dịch ZH của truy vấn -> chỉ mục ZH; hợp nhất danh sách doc.

Điểm BM25 của hai chỉ mục khác thang đo nên KHÔNG so sánh trực tiếp. Hợp nhất theo số lượng cố định
(n_vi doc tiếng Việt + n_zh doc tiếng Trung) rồi xen kẽ để không lệch về một phía. Chunk lấy riêng theo từng ngôn ngữ
(bằng truy vấn của ngôn ngữ đó), sau đó nối lại. Tham số là giá trị khởi đầu, chưa có nhãn vàng để tối ưu.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from bigindex import BigIndex
from bigpipeline import load_texts, pick_chunks, search_docs
from boilerplate import BoilerplateFilter


def load_zh_map(path: Path) -> dict[int, str]:
    """{id: bản dịch} từ JSONL của translate_queries.py; bỏ dòng trống, lấy dòng cuối cho mỗi id."""
    out: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("zh"):
            out[int(r["id"])] = r["zh"]
    return out


def from_cache(cache: dict, n: int) -> dict[int, list[tuple[int, int]]]:
    """{qid: [(doc_id, chỉ số)]} lấy n doc đầu từ kết quả của search_cache.py."""
    return {int(q): [(int(d), int(i)) for d, i, _ in hits[:n]] for q, hits in cache.items()}


def interleave(a: list[int], b: list[int]) -> list[int]:
    """Xen kẽ a, b theo thứ tự xếp hạng, bỏ trùng id (giữ lần xuất hiện đầu)."""
    out, seen = [], set()
    for i in range(max(len(a), len(b))):
        for lst in (a, b):
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i])
                out.append(lst[i])
    return out


def build_submission_bi(vi: tuple[BigIndex, BoilerplateFilter], zh: tuple[BigIndex, BoilerplateFilter],
                        queries: list[tuple[int, str]], zh_map: dict[int, str], n_vi: int = 20, n_zh: int = 20,
                        chunk_docs_vi: int = 5, chunk_docs_zh: int = 5, units_vi: int = 600, units_zh: int = 600,
                        chunks_vi: int = 2, chunks_zh: int = 2, log=print, cache_vi: dict | None = None,
                        cache_zh: dict | None = None) -> list[dict]:
    """cache_vi/cache_zh: kết quả từ search_cache.py ({qid: [[doc_id, chỉ số, điểm]...]}); nếu có thì cắt lấy n đầu, không tìm lại."""
    (ivi, bvi), (izh, bzh) = vi, zh
    t = time.time()
    # n_vi = 0 hoặc n_zh = 0: bỏ hẳn nhánh đó (dùng để đo riêng từng ngôn ngữ).
    zq = [(qid, zh_map[int(qid)]) for qid, _ in queries if int(qid) in zh_map]
    if cache_vi is not None:
        found_vi = from_cache(cache_vi, n_vi) if n_vi > 0 else {}
    else:
        found_vi = search_docs(ivi, queries, n_vi, 0.0) if n_vi > 0 else {}
    if cache_zh is not None:
        found_zh = from_cache(cache_zh, n_zh) if n_zh > 0 else {}
    else:
        found_zh = search_docs(izh, zq, n_zh, 0.0) if n_zh > 0 else {}
    log(f"pha 1: tìm doc VI ({len(queries)} câu) và ZH ({len(zq)} câu) ({time.time() - t:.0f}s)")

    tv = load_texts(ivi, {d: i for h in found_vi.values() for d, i in h[:chunk_docs_vi]}, bvi, log=log)
    tz = load_texts(izh, {d: i for h in found_zh.values() for d, i in h[:chunk_docs_zh]}, bzh, log=log)
    sub = []
    for qid, q in queries:
        qid = int(qid)
        dv = [d for d, _ in found_vi.get(qid, [])]
        dz = [d for d, _ in found_zh.get(qid, [])]
        chunks = pick_chunks(q, "vi", dv[:chunk_docs_vi], tv, units_vi, chunks_vi)
        if qid in zh_map and n_zh > 0:
            chunks += pick_chunks(zh_map[qid], "zh", dz[:chunk_docs_zh], tz, units_zh, chunks_zh)
        sub.append({"id": qid, "relevant_docs": interleave(dv, dz), "relevant_chunks": chunks})
    log(f"pha 2 xong ({time.time() - t:.0f}s)")
    return sub


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-vi", required=True)
    ap.add_argument("--index-zh", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--zh-queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-vi", type=int, default=20)
    ap.add_argument("--n-zh", type=int, default=20)
    ap.add_argument("--chunk-docs-vi", type=int, default=5)
    ap.add_argument("--chunk-docs-zh", type=int, default=5)
    ap.add_argument("--units-vi", type=int, default=600)
    ap.add_argument("--units-zh", type=int, default=600)
    ap.add_argument("--chunks-vi", type=int, default=2)
    ap.add_argument("--chunks-zh", type=int, default=2)
    ap.add_argument("--cache-vi", default=None, help="JSON từ search_cache.py cho chỉ mục VI (bỏ qua pha tìm doc VI)")
    ap.add_argument("--cache-zh", default=None, help="JSON từ search_cache.py cho chỉ mục ZH")
    a = ap.parse_args()

    import pandas as pd
    from evaluate import validate

    def load(p):
        p = Path(p)
        return BigIndex(p), BoilerplateFilter.from_json((p / "boilerplate.json").read_text(encoding="utf-8"))

    q = pd.read_parquet(a.queries)
    sub = build_submission_bi(load(a.index_vi), load(a.index_zh), list(zip(q["id"], q["query"])),
                              load_zh_map(Path(a.zh_queries)), a.n_vi, a.n_zh, a.chunk_docs_vi, a.chunk_docs_zh,
                              a.units_vi, a.units_zh, a.chunks_vi, a.chunks_zh, log=lambda m: print(m, flush=True),
                              cache_vi=json.loads(Path(a.cache_vi).read_text(encoding="utf-8")) if a.cache_vi else None,
                              cache_zh=json.loads(Path(a.cache_zh).read_text(encoding="utf-8")) if a.cache_zh else None)
    Path(a.out).write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    errs = validate(sub, len(q))
    nd = [len(s["relevant_docs"]) for s in sub]
    nc = [len(s["relevant_chunks"]) for s in sub]
    print(f"ghi {a.out}: {len(sub)} câu, {sum(1 for s in sub if not s['relevant_docs'])} câu rỗng, doc TB {sum(nd) / len(nd):.2f}, "
          f"chunk TB {sum(nc) / len(nc):.2f}, định dạng {'OK' if not errs else errs[:3]}", flush=True)


if __name__ == "__main__":
    main()
