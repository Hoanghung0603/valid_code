"""Sinh HyDE / Reason cho truy vấn y khoa tiếng Việt qua LM Studio (API tương thích OpenAI, chỉ dùng thư viện chuẩn).

Hai chế độ (--mode):
- hyde: đoạn trả lời giả định 80-150 từ, văn phong bài viết y khoa phổ thông, không bịa số liệu.
- reason: suy luận 40-90 từ theo mẫu "Câu hỏi về: ...; các bệnh/nguyên nhân có thể: ...; thuật ngữ chuyên môn: ...".
Ghi JSONL {id, query, hyde|reason}; chạy lại được theo id (dòng đã có giá trị khác rỗng thì bỏ qua).
--gen-file: không gọi model, chỉ ghép truy vấn gốc + đoạn sinh -> {id, query} cho dense (query_len 512).
Chỉ sinh nội dung mô tả truy vấn, không dùng nhãn vàng.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_THINK = re.compile(r"<t" + r"hinking>.*?</t" + r"hinking>", re.S)

HYDE_SYSTEM = (
    "Bạn là chuyên gia y khoa viết bài phổ thông. Với câu hỏi sức khỏe của người dùng, hãy viết MỘT đoạn "
    "trả lời giả định (giống phần nội dung chính của bài viết y khoa trên vinmec hoặc hellobacsi): nêu bệnh "
    "hoặc nguyên nhân liên quan, triệu chứng thường gặp, cách xử trí hoặc điều trị. "
    "Yêu cầu: 80 đến 150 từ, tiếng Việt, văn phong bài viết y khoa, thành một đoạn văn liền mạch, "
    "không tiêu đề, không gạch đầu dòng. KHÔNG bịa số liệu cụ thể (liều thuốc, tỷ lệ, con số thống kê). "
    "Chỉ in đoạn văn, không giải thích, không ghi chú."
)
REASON_SYSTEM = (
    "Bạn là trợ lý truy vấn y khoa. Với câu hỏi sức khỏe của người dùng, hãy suy luận ngắn gọn theo mẫu: "
    "\"Câu hỏi về: <chủ đề/chủ thể>; các bệnh/nguyên nhân có thể: <các bệnh hoặc nguyên nhân liên quan>; "
    "thuật ngữ chuyên môn: <thuật ngữ y khoa tiếng Việt liên quan>\". "
    "Yêu cầu: 40 đến 90 từ, tiếng Việt, cô đọng, liệt kê bằng dấu chấm phẩy. "
    "Chỉ in phần suy luận, không giải thích thêm."
)
# Ví dụ tự viết (không lấy từ nhãn vàng) để model bám đúng mẫu và độ dài.
HYDE_SHOTS = [
    (
        "Bé nhà em 4 tuổi ho nhiều về đêm, sổ mũi 3 ngày thì phải làm sao ạ?",
        "Ho về đêm kèm sổ mũi kéo dài ở trẻ nhỏ thường gặp trong viêm đường hô hấp trên do virus, viêm mũi họng "
        "hoặc viêm mũi dị ứng. Trẻ có thể ho thành cơn, khò khè nhẹ, chảy nước mũi trong hoặc đục dần. Cha mẹ nên "
        "giữ ấm cho trẻ, nhỏ mũi bằng nước muối sinh lý, cho trẻ uống đủ nước và nghỉ ngơi. Nếu trẻ sốt cao, ho "
        "dữ dội, khó thở, bỏ ăn hoặc triệu chứng kéo dài quá một tuần, cần đưa trẻ đến khám để loại trừ viêm phế "
        "quản, viêm phổi hoặc hen phế quản và được xử trí phù hợp.",
    ),
    (
        "Uống thuốc tránh thai khẩn cấp xong bị ra máu thì có sao không bác sĩ?",
        "Ra máu âm đạo sau khi uống thuốc tránh thai khẩn cấp là phản ứng thường gặp do nội tiết tố thay đổi đột "
        "ngột, cũng có thể do niêm mạc tử cung bong nhẹ. Máu thường ra ít, màu hồng hoặc nâu, tự hết sau vài ngày. "
        "Người phụ nữ nên nghỉ ngơi, tránh quan hệ tình dục trong thời gian ra máu và theo dõi thêm các dấu hiệu "
        "đau bụng dữ dội, máu ra nhiều hoặc có mùi hôi. Nếu hiện tượng tái diễn sau mỗi lần dùng thuốc hoặc chậm "
        "kinh, cần đi khám phụ khoa để được tư vấn biện pháp tránh thai phù hợp hơn.",
    ),
]
REASON_SHOTS = [
    (
        "Kết quả xét nghiệm máu của em có bạch cầu 12.000 thì có bị làm sao không ạ?",
        "Câu hỏi về: kết quả xét nghiệm máu bạch cầu cao; các bệnh/nguyên nhân có thể: nhiễm khuẩn, viêm nhiễm, "
        "stress, hút thuốc, dùng thuốc corticoid, bệnh lý tủy xương; thuật ngữ chuyên môn: bạch cầu (WBC), tăng "
        "bạch cầu, công thức bạch cầu, nhiễm khuẩn.",
    ),
    (
        "Đau bụng vùng rốn âm ỉ mấy ngày nay là bệnh gì?",
        "Câu hỏi về: đau bụng vùng rốn âm ỉ kéo dài; các bệnh/nguyên nhân có thể: viêm ruột thừa giai đoạn sớm, "
        "rối loạn tiêu hóa, viêm ruột, nhiễm giun, tắc ruột nhẹ; thuật ngữ chuyên môn: đau bụng vùng rốn, viêm ruột "
        "thừa, rối loạn tiêu hóa, khám bụng.",
    ),
]
MODES = {
    "hyde": {"field": "hyde", "min_words": 80, "max_words": 150, "system": HYDE_SYSTEM, "shots": HYDE_SHOTS},
    "reason": {"field": "reason", "min_words": 40, "max_words": 90, "system": REASON_SYSTEM, "shots": REASON_SHOTS},
}


def strip_think(text: str) -> str:
    """Bỏ khối <think>...</think> rồi strip (model đôi khi vẫn suy luận dù reasoning_effort=none)."""
    return _THINK.sub("", text or "").strip()


def word_count(text: str) -> int:
    """Số từ = số cụm cách nhau bởi khoảng trắng."""
    return len(text.split())


def len_ok(text: str, min_words: int, max_words: int) -> bool:
    n = word_count(text)
    return min_words <= n <= max_words


def load_queries(path) -> list:
    """[(id, query)] từ JSONL {id, query} (trùng id lấy bản đầu) hoặc parquet (như translate_queries)."""
    p = Path(path)
    if p.suffix == ".parquet":
        import pandas as pd
        q = pd.read_parquet(p)
        return [(int(i), str(s)) for i, s in zip(q["id"].astype(int), q["query"])]
    out = []
    seen = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        i = int(r["id"])
        if i in seen or not str(r.get("query", "")).strip():
            continue
        seen.add(i)
        out.append((i, str(r["query"])))
    return out


def load_done(path, field: str) -> dict:
    """{id: row} các dòng JSONL đã sinh xong (row[field] khác rỗng) để resume; file thiếu/trả về {}."""
    p = Path(path)
    if not p.exists():
        return {}
    done = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get(field):
            done[int(r["id"])] = r
    return done


def merge_text(orig: str, gen: str) -> str:
    """Truy vấn cuối = câu hỏi gốc + xuống dòng + đoạn sinh (ghép text giữ thông tin câu hỏi)."""
    orig = orig.strip()
    gen = (gen or "").strip()
    return orig if not gen else orig + "\n" + gen


def build_combined(queries: list, gen: dict, field: str) -> list:
    """[{id, query}] ghép gốc với đoạn sinh trong gen {id: row}; câu thiếu đoạn sinh giữ nguyên gốc."""
    return [{"id": i, "query": merge_text(q, (gen.get(i) or {}).get(field, ""))} for i, q in queries]


def rewrite(query: str, mode: str, url: str, model: str, max_tokens: int,
            temperature: float = 0.3, timeout: int = 180, tries: int = 3) -> str:
    """Thử tối đa `tries` lần; lần sau nhắc rõ ràng hơn. Trả '' nếu không ra được đoạn hợp lệ."""
    cfg = MODES[mode]
    best = ""
    for k in range(tries):
        out = _rewrite_once(query, cfg, url, model, max_tokens, temperature, timeout, strict=k > 0)
        if out and len_ok(out, cfg["min_words"], cfg["max_words"]):
            return out
        if out and not best:
            best = out  # vẫn dùng được nếu các lần sau thất bại
    return best


def _rewrite_once(query: str, cfg: dict, url: str, model: str, max_tokens: int,
                  temperature: float, timeout: int, strict: bool) -> str:
    shots = [m for q, a in cfg["shots"] for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})]
    user = query
    if strict:
        user = (f"Chỉ in đoạn đúng mẫu, đủ {cfg['min_words']}-{cfg['max_words']} từ, không được suy luận "
                f"liên miên.\n{query}")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": cfg["system"]}, *shots, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # LM Studio bỏ qua chat_template_kwargs/"/no_think"; chỉ reasoning_effort=none tắt được suy luận của Qwen3.5
        "reasoning_effort": "none",
    }
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("LM_API_TOKEN")  # token do user cấp, chỉ đọc từ biến môi trường, không ghi ra file/log
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        msg = json.loads(r.read())["choices"][0]
    text = strip_think(msg["message"].get("content") or "")
    return text if msg.get("finish_reason") == "stop" else ""  # bị cắt do hết token -> coi như thất bại


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="JSONL {id, query} hoặc parquet")
    ap.add_argument("--out", required=True, help="JSONL {id, query, hyde|reason} (sinh) hoặc {id, query} (--gen-file)")
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--gen-file", default=None, help="JSONL đã sinh sẵn: chỉ ghép với --queries, không gọi model")
    ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions")
    ap.add_argument("--model", default="qwen35-9b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.3)
    a = ap.parse_args()

    rows = load_queries(a.queries)[: a.limit]
    field = MODES[a.mode]["field"]

    if a.gen_file:
        gen = load_done(a.gen_file, field)
        out = build_combined(rows, gen, field)
        Path(a.out).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
        print(f"ghép {len(out)} câu ({len(gen)} có {field}) -> {a.out}", flush=True)
        return

    done = load_done(a.out, field)
    todo = [(i, s) for i, s in rows if i not in done]
    print(f"{len(rows)} câu, đã có {len(done)}, còn {len(todo)}", flush=True)

    def work(item):
        i, s = item
        try:
            return {"id": i, "query": s, field: rewrite(s, a.mode, a.url, a.model, a.max_tokens, a.temperature)}
        except Exception as e:  # noqa: BLE001 - ghi lại và tiếp tục
            return {"id": i, "query": s, field: "", "error": repr(e)[:200]}

    t0 = time.time()
    fails = 0
    with Path(a.out).open("a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            fails += not r[field]
            if n % 25 == 0 or n == len(todo):
                print(f"[{time.time() - t0:6.0f}s] {n}/{len(todo)} rỗng/lỗi {fails}", flush=True)
    print(f"Xong {a.mode}: {len(todo)} câu, rỗng/lỗi {fails}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
