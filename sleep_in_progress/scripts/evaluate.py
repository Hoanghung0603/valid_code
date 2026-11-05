"""Đánh giá offline + kiểm tra định dạng bài nộp, mô phỏng theo mô tả Evaluation trên leaderboard.

- Doc level: Precision/Recall/F2 trên tập id tài liệu, macro trên câu hỏi.
- Chunk level: chunk nộp đúng nếu LCS_tokens(c, g) / |g| >= 0.4 (luật cũ, --chunk-rule gold) và
  thêm LCS_tokens(c, g) / |c| >= 0.4 (luật mới 23/10, --chunk-rule both, mặc định), với ít nhất
  một chunk tham chiếu cùng câu hỏi và cùng doc_id. Recall = số chunk tham chiếu được tìm thấy / tổng.
  Chunk rỗng không khớp; |c|, |g| tách token bằng cùng --tokenizer sau chuẩn hoá.
- Chuẩn hoá: NFKC, unescape HTML, lowercase, gộp khoảng trắng, strip.

CHƯA RÕ: BTC không nói cách tách token. Mặc định tách theo khoảng trắng; với tiếng Trung
(không có khoảng trắng) có thể BTC tách theo ký tự. Dùng --tokenizer để thử cả hai.

Dùng:
  python scripts/evaluate.py validate submission.json
  python scripts/evaluate.py score submission.json gold.json [--tokenizer ws|cjk] [--chunk-rule gold|both]
gold.json cùng định dạng bài nộp (tự gán nhãn cho tập dev nội bộ).
"""
import argparse
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

THRESHOLD = 0.4
N_QUERIES = 1200
_WS = re.compile(r"\s+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text).lower()
    return _WS.sub(" ", text).strip()


def tokenize(text: str, mode: str = "ws") -> list[str]:
    text = normalize(text)
    if mode == "cjk":  # mỗi ký tự Hán là 1 token, phần còn lại tách theo khoảng trắng
        text = _CJK.sub(lambda m: f" {m.group(0)} ", text)
    return text.split()


def lcs_len(a: list[str], b: list[str]) -> int:
    """Độ dài dãy con chung dài nhất (không cần liên tiếp), DP O(len(a)*len(b)) bộ nhớ O(len(b))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def chunk_match(ctoks: list[str], gtoks: list[str], rule: str = "both",
                threshold: float = THRESHOLD) -> bool:
    """Chunk nộp ctoks có khớp chunk vàng gtoks không.

    rule='gold': luật cũ, LCS/|g| >= threshold.
    rule='both': luật mới, thêm LCS/|c| >= threshold.
    Chunk rỗng không khớp. Cắt sớm: LCS <= min(|c|,|g|) nên nếu
    min(|c|,|g|) < threshold*max(|c|,|g|) thì chắc chắn không khớp, bỏ qua LCS.
    """
    nc, ng = len(ctoks), len(gtoks)
    if not nc or not ng:
        return False
    if rule == "both":
        if min(nc, ng) + 1e-9 < threshold * max(nc, ng):
            return False
        lcs = lcs_len(ctoks, gtoks)
        return lcs / ng >= threshold and lcs / nc >= threshold
    if nc + 1e-9 < threshold * ng:  # cắt sớm cho luật cũ: LCS <= |c|
        return False
    return lcs_len(ctoks, gtoks) / ng >= threshold


def f2(p: float, r: float) -> float:
    return 0.0 if p == 0 and r == 0 else 5 * p * r / (4 * p + r)


def prf(n_hit_pred: int, n_pred: int, n_hit_gold: int, n_gold: int) -> tuple[float, float, float]:
    p = n_hit_pred / n_pred if n_pred else 0.0
    r = n_hit_gold / n_gold if n_gold else 0.0
    return p, r, f2(p, r)


def validate(sub, n_queries: int | None = N_QUERIES) -> list[str]:
    """Trả về danh sách lỗi; rỗng nghĩa là hợp lệ."""
    errs = []
    if not isinstance(sub, list):
        return ["File phải là một JSON array"]
    ids = []
    for i, item in enumerate(sub):
        where = f"item[{i}]"
        if not isinstance(item, dict):
            errs.append(f"{where}: không phải object"); continue
        qid = item.get("id")
        if not isinstance(qid, int) or isinstance(qid, bool):
            errs.append(f"{where}: id phải là integer"); continue
        ids.append(qid)
        docs = item.get("relevant_docs")
        chunks = item.get("relevant_chunks")
        if not isinstance(docs, list) or not isinstance(chunks, list):
            errs.append(f"id={qid}: thiếu relevant_docs/relevant_chunks (dùng [] nếu rỗng)"); continue
        doc_set = {str(d) for d in docs}
        for j, ch in enumerate(chunks):
            if not isinstance(ch, dict) or "doc_id" not in ch or not isinstance(ch.get("chunk_text"), str):
                errs.append(f"id={qid} chunk[{j}]: cần doc_id và chunk_text"); continue
            if str(ch["doc_id"]) not in doc_set:
                errs.append(f"id={qid} chunk[{j}]: doc_id {ch['doc_id']} không có trong relevant_docs")
            if "chunk_order" in ch and not (isinstance(ch["chunk_order"], int) and ch["chunk_order"] >= 0):
                errs.append(f"id={qid} chunk[{j}]: chunk_order phải là số nguyên không âm")
    dup = len(ids) - len(set(ids))
    if dup:
        errs.append(f"{dup} id bị trùng")
    if n_queries is not None and len(set(ids)) != n_queries:
        errs.append(f"Có {len(set(ids))} câu hỏi, cần đúng {n_queries}")
    return errs


def score(sub, gold, tokenizer: str = "ws", chunk_rule: str = "both",
          threshold: float = THRESHOLD) -> dict:
    pred = {it["id"]: it for it in sub}
    acc = {k: 0.0 for k in ("doc_p", "doc_r", "doc_f2", "chunk_p", "chunk_r", "chunk_f2")}
    for g in gold:
        p = pred.get(g["id"], {"relevant_docs": [], "relevant_chunks": []})
        gd, pd_ = {str(d) for d in g["relevant_docs"]}, {str(d) for d in p["relevant_docs"]}
        hit = len(gd & pd_)
        dp, dr, df = prf(hit, len(pd_), hit, len(gd))

        gold_chunks = [(str(c["doc_id"]), tokenize(c["chunk_text"], tokenizer)) for c in g["relevant_chunks"]]
        found = [False] * len(gold_chunks)
        correct = 0
        for c in p["relevant_chunks"]:
            ctoks, cdoc = tokenize(c["chunk_text"], tokenizer), str(c["doc_id"])
            ok = False
            for k, (gdoc, gtoks) in enumerate(gold_chunks):
                if gdoc == cdoc and chunk_match(ctoks, gtoks, chunk_rule, threshold):
                    ok = True
                    found[k] = True
            correct += ok
        cp, cr, cf = prf(correct, len(p["relevant_chunks"]), sum(found), len(gold_chunks))

        for k, v in zip(acc, (dp, dr, df, cp, cr, cf)):
            acc[k] += v
    n = len(gold) or 1
    res = {k: round(v / n, 4) for k, v in acc.items()}
    res["final"] = round((res["doc_f2"] + res["chunk_f2"]) / 2, 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate"); v.add_argument("submission")
    v.add_argument("--n-queries", type=int, default=N_QUERIES)
    s = sp.add_parser("score"); s.add_argument("submission"); s.add_argument("gold")
    s.add_argument("--tokenizer", choices=["ws", "cjk"], default="ws")
    s.add_argument("--chunk-rule", choices=["gold", "both"], default="both",
                   help="gold: luật cũ LCS/|g|>=0.4; both: luật mới thêm LCS/|c|>=0.4 (mặc định)")
    a = ap.parse_args()

    sub = json.loads(Path(a.submission).read_text(encoding="utf-8"))
    if a.cmd == "validate":
        errs = validate(sub, a.n_queries)
        print("OK" if not errs else "\n".join(errs[:50]) + (f"\n... +{len(errs) - 50}" if len(errs) > 50 else ""))
        sys.exit(1 if errs else 0)
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    print(json.dumps(score(sub, gold, a.tokenizer, a.chunk_rule), indent=2))


if __name__ == "__main__":
    main()
