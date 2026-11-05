"""Gộp hai bài nộp: relevant_docs lấy từ --docs-from, relevant_chunks lấy nguyên từ --chunks-from.

Với mỗi id: giữ nguyên thứ tự relevant_docs của --docs-from, rồi thêm (theo thứ tự xuất hiện lần đầu
trong relevant_chunks của --chunks-from) mọi doc_id của chunk chưa có trong danh sách (không trùng).
Chunk được giữ nguyên danh sách (cùng nội dung, cùng thứ tự). In thống kê: số câu, doc/câu
(min/TB/max), số câu >60 doc, chunk ngoài relevant_docs, doc trùng.

Dùng:
  python scripts/combine_sub.py --docs-from a.json --chunks-from b.json --out c.json
"""
import argparse
import json
import sys
from pathlib import Path


def combine_sub(docs_sub: list, chunks_sub: list) -> list:
    """Gộp theo id: docs của docs_sub + doc của chunk từ chunks_sub (nếu thiếu), chunks của chunks_sub."""
    chunks_by_id = {it["id"]: it for it in chunks_sub}
    out = []
    for item in docs_sub:
        ch_item = chunks_by_id[item["id"]]
        docs = list(item["relevant_docs"])
        seen = {str(d) for d in docs}
        for ch in ch_item["relevant_chunks"]:
            d = ch["doc_id"]
            if str(d) not in seen:
                seen.add(str(d))
                docs.append(d)
        out.append({"id": item["id"], "relevant_docs": docs, "relevant_chunks": ch_item["relevant_chunks"]})
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-from", required=True)
    ap.add_argument("--chunks-from", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    docs_sub = json.loads(Path(a.docs_from).read_text(encoding="utf-8"))
    chunks_sub = json.loads(Path(a.chunks_from).read_text(encoding="utf-8"))
    out = combine_sub(docs_sub, chunks_sub)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    nd = [len(s["relevant_docs"]) for s in out]
    n = max(len(out), 1)
    chunks_by_id = {it["id"]: it for it in chunks_sub}
    outside = sum(
        1 for s in out
        for ch in chunks_by_id[s["id"]]["relevant_chunks"]
        if str(ch.get("doc_id")) not in {str(d) for d in s["relevant_docs"]}
    )
    dup = sum(len(s["relevant_docs"]) - len({str(d) for d in s["relevant_docs"]}) for s in out)
    over60 = sum(1 for x in nd if x > 60)
    print(f"ghi {a.out}: {len(out)} câu, doc/câu min {min(nd)} TB {sum(nd) / n:.2f} max {max(nd)}, "
          f"câu >60 doc: {over60}, chunk ngoài relevant_docs: {outside}, doc trùng: {dup}", flush=True)


if __name__ == "__main__":
    main()
