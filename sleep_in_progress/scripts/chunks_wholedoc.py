"""Chunk = toàn văn doc (tiền tố nguyên văn), theo thứ tự doc, trong ngân sách ký tự mỗi câu.

Giả thuyết (08/10): overlap = LCS(c, g)/|g| chia cho độ dài chunk VÀNG, nên một chunk dài phủ mọi
chunk vàng trong doc đó (Chunks R tiến tới Docs R của các doc được đưa vào chunk).

Thứ tự doc mỗi câu:
  rr: theo --cands (thứ tự rerank mức doc, top-100 của cache_pool_rr)
  dw: theo điểm cửa sổ tốt nhất trong doc (max logit của docwin), đồng điểm giữ thứ tự cands
Mỗi doc lấy text[:doc_cap] (cắt lùi về khoảng trắng gần nhất). Cộng dồn tới --budget ký tự/câu;
doc cuối bị cắt về phần ngân sách còn lại nếu phần đó ≥ --min-chars, rồi dừng. Bỏ text rỗng/trùng.

Xuất định dạng chunk của union_sub.py --chunks-from: [{id, relevant_docs: [], relevant_chunks}].
Dùng:
  python scripts/chunks_wholedoc.py --docs p0.jsonl.zst p1.jsonl.zst --cands cands.json \
    --docwin docwin_top100.json.zst --orders rr dw --budgets 220000 270000 --doc-cap 40000 \
    --out-prefix D:/r2ai/data/sub/wd/chunks_wd
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def cut(text: str, n: int) -> str:
    """Tiền tố nguyên văn dài ≤ n ký tự, lùi về khoảng trắng gần nhất nếu có (trong 200 ký tự cuối)."""
    if len(text) <= n:
        return text
    t = text[:n]
    sp = max(t.rfind(" "), t.rfind("\n"))
    if sp >= n - 200 and sp > 0:
        t = t[:sp]
    return t


def order_docs(cand: list, wins_q: dict | None) -> list:
    """doc_id theo thứ tự: cands (wins_q None) hoặc max logit cửa sổ giảm dần; doc không có cửa sổ xuống cuối."""
    ids = [item[0] for item in cand]
    if wins_q is None:
        return ids

    def best(did):
        w = wins_q.get(str(did), wins_q.get(did))
        return max(x[2] for x in w) if w else float("-inf")

    pos = {d: i for i, d in enumerate(ids)}
    return sorted(ids, key=lambda d: (-best(d), pos[d]))


def pick_q(doc_ids: list, docs: dict, budget: int, doc_cap: int, min_chars: int = 2000) -> list[dict]:
    """Chọn chunk toàn văn theo thứ tự doc_ids trong ngân sách ký tự."""
    out, used, seen = [], 0, set()
    for did in doc_ids:
        t = docs.get(did)
        if not t or not t.strip():
            continue
        t = cut(t, doc_cap)
        if t in seen:
            continue
        room = budget - used
        if len(t) > room:
            if room >= min_chars:
                t = cut(t, room)
                out.append({"doc_id": did, "chunk_text": t})
            break
        seen.add(t)
        out.append({"doc_id": did, "chunk_text": t})
        used += len(t)
    return out


def build(cands: dict, docs: dict, budget: int, doc_cap: int, docwin: dict | None = None,
          min_chars: int = 2000) -> list[dict]:
    sub = []
    for q, cand in cands.items():
        wins_q = docwin.get(q, {}) if docwin is not None else None
        ids = order_docs(cand, wins_q)
        sub.append({"id": int(q), "relevant_docs": [], "relevant_chunks": pick_q(ids, docs, budget, doc_cap, min_chars)})
    return sub


def verify_verbatim(sub: list, docs: dict) -> int:
    """Số chunk không phải substring nguyên văn của text doc (phải là 0)."""
    return sum(1 for s in sub for c in s["relevant_chunks"] if c["chunk_text"] not in docs.get(c["doc_id"], ""))


def main() -> None:
    from chunks_from_docwin import load_json
    from rerank_run import load_docs

    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, nargs="+")
    ap.add_argument("--cands", required=True)
    ap.add_argument("--docwin", default="", help="cần khi --orders có dw")
    ap.add_argument("--orders", nargs="+", default=["rr"], choices=["rr", "dw"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[270000])
    ap.add_argument("--doc-cap", type=int, default=40000)
    ap.add_argument("--min-chars", type=int, default=2000)
    ap.add_argument("--top-docs", type=int, default=100)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cands = json.loads(Path(a.cands).read_text(encoding="utf-8"))
    cands = {q: c[: a.top_docs] for q, c in cands.items()}
    wanted = {item[0] for c in cands.values() for item in c}
    docs = load_docs(a.docs, wanted=wanted)
    lens = sorted(len(t) for t in docs.values())
    print(f"{len(cands)} câu, {len(docs)} doc, tải {time.time() - t0:.0f}s; ký tự/doc trung vị {lens[len(lens) // 2]} "
          f"p90 {lens[int(len(lens) * 0.9)]} TB {sum(lens) / len(lens):.0f}", flush=True)
    docwin = load_json(a.docwin) if "dw" in a.orders else None
    for order in a.orders:
        for b in a.budgets:
            sub = build(cands, docs, b, a.doc_cap, docwin if order == "dw" else None, a.min_chars)
            nc = [len(s["relevant_chunks"]) for s in sub]
            ch = [sum(len(c["chunk_text"]) for c in s["relevant_chunks"]) for s in sub]
            p = f"{a.out_prefix}_{order}_b{b // 1000}k.json"
            Path(p).write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
            print(f"{order} b{b}: ghi {p}; chunk/câu min {min(nc)} TB {sum(nc) / len(nc):.1f} max {max(nc)}; "
                  f"ký tự/câu TB {sum(ch) / len(ch):.0f}; không nguyên văn {verify_verbatim(sub, docs)}", flush=True)


if __name__ == "__main__":
    main()
