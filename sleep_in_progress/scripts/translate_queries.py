"""Dịch truy vấn tiếng Việt sang tiếng Trung giản thể (--target zh) hoặc tiếng Anh y khoa (--target en)
qua LM Studio (API tương thích OpenAI, chỉ dùng thư viện chuẩn).

Chạy lại được: kết quả ghi JSONL theo từng câu, câu đã có thì bỏ qua. Mô hình: Qwen3.5 9B (mở, <=15B).
Chỉ dịch nghĩa để tìm kiếm; không trả lời câu hỏi; không dùng nhãn vàng.
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

SYSTEM_ZH = (
    "Bạn là dịch giả y khoa. Dịch câu hỏi sức khỏe tiếng Việt của người dùng sang tiếng Trung giản thể "
    "thành MỘT câu truy vấn tìm kiếm ngắn gọn, giữ nguyên các thuật ngữ y khoa (bệnh, triệu chứng, thuốc, "
    "xét nghiệm, liều dùng, đối tượng bệnh nhân). Sửa lỗi chính tả rõ ràng nếu có. "
    "Chỉ in bản dịch, không giải thích, không ghi chú, không dùng ký hiệu đặc biệt."
)
SYSTEM_EN = (
    "You are a medical translator. Translate the user's Vietnamese health question into ONE short English "
    "medical search query, keeping medical terms (diseases, symptoms, drugs, lab tests, dosage, patient groups). "
    "Fix obvious spelling errors. Do NOT answer the question. "
    "Print only the English translation: no explanations, no notes, no special symbols."
)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_CJK = re.compile(r"[\u4e00-\u9fff]")
# Ví dụ mẫu tự viết (không lấy từ nhãn vàng) để model luôn xuất đúng ngôn ngữ đích.
SHOTS_ZH = [
    ("Kết quả của Beta-HCG là dưới 0.10 U/L thì là như thế nào ạ?", "Beta-HCG 结果低于0.10 U/L 是什么意思？"),
    ("Rang co le cham den, chay mau khi thuc day la sao?", "牙齿有黑点小洞，起床时出血是怎么回事？"),
    ("Em bị đau đầu, chóng mặt, huyết áp 160/100 thì uống thuốc gì ạ?", "头痛头晕，血压160/100，应该服用什么药？"),
]
SHOTS_EN = [
    ("Kết quả của Beta-HCG là dưới 0.10 U/L thì là như thế nào ạ?",
     "What does a Beta-HCG result below 0.10 U/L mean?"),
    ("Rang co le cham den, chay mau khi thuc day la sao?",
     "Black spots on teeth and bleeding gums on waking up, what could it be?"),
    ("Em bị đau đầu, chóng mặt, huyết áp 160/100 thì uống thuốc gì ạ?",
     "Headache and dizziness with blood pressure 160/100, what medication should I take?"),
]
STRICT_ZH = "必须输出简体中文，不要输出越南语。"
STRICT_EN = "Output English only, do not copy the Vietnamese question."


_VI_ONLY = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)


def is_chinese(text: str) -> bool:
    """Đạt nếu có chữ Hán và không còn chữ cái đặc trưng tiếng Việt (model đôi khi chép lại câu gốc).

    Không đòi tỉ lệ chữ Hán cao: câu y khoa hợp lệ có thể chứa nhiều ký hiệu Latinh (Beta-HCG, U/L, SCC).
    """
    cjk = len(_CJK.findall(text))
    return cjk >= 2 and cjk >= 0.25 * sum(c.isalpha() for c in text) and not _VI_ONLY.search(text)


def is_english(text: str) -> bool:
    """Đạt nếu là chữ Latinh, không còn dấu tiếng Việt (model đôi khi chép lại câu gốc)."""
    if not text or len(text) < 3:
        return False
    if _VI_ONLY.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.9


_TARGETS = {
    "zh": {"system": SYSTEM_ZH, "shots": SHOTS_ZH, "strict": STRICT_ZH, "check": is_chinese, "field": "zh"},
    "en": {"system": SYSTEM_EN, "shots": SHOTS_EN, "strict": STRICT_EN, "check": is_english, "field": "en"},
}


def translate(query: str, url: str, model: str, max_tokens: int, target: str = "zh",
              timeout: int = 180, tries: int = 3) -> str:
    """Thử tối đa `tries` lần; lần sau nhắc rõ hơn. Trả '' nếu không ra được ngôn ngữ đích."""
    cfg = _TARGETS[target]
    for k in range(tries):
        out = _translate_once(query, url, model, max_tokens, cfg, timeout, strict=k > 0)
        if cfg["check"](out):
            return out
    return ""


def _translate_once(query: str, url: str, model: str, max_tokens: int, cfg: dict,
                    timeout: int, strict: bool) -> str:
    shots = [m for q, z in cfg["shots"]
             for m in ({"role": "user", "content": q}, {"role": "assistant", "content": z})]
    user = query if not strict else f"{cfg['strict']}\n{query}"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": cfg["system"]}, *shots, {"role": "user", "content": user}],
        "temperature": 0,
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
    text = _THINK.sub("", msg["message"].get("content") or "").strip()
    return text if msg.get("finish_reason") == "stop" else ""  # bị cắt do hết token -> coi như thất bại


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True, help="file JSONL {id, query, <target>}")
    ap.add_argument("--target", choices=sorted(_TARGETS), default="zh", help="zh: Trung giản thể; en: Anh y khoa")
    ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions")
    ap.add_argument("--model", default="qwen35-9b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    field = _TARGETS[a.target]["field"]

    import pandas as pd
    q = pd.read_parquet(a.queries)
    rows = list(zip(q["id"].astype(int), q["query"]))[: a.limit]
    out = Path(a.out)
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get(field):
                done[r["id"]] = r
    todo = [(i, s) for i, s in rows if i not in done]
    print(f"{len(rows)} câu, đã có {len(done)}, còn {len(todo)} (target={a.target})", flush=True)

    def work(item):
        i, s = item
        try:
            return {"id": i, "query": s, field: translate(s, a.url, a.model, a.max_tokens, a.target)}
        except Exception as e:  # noqa: BLE001 - ghi lại và tiếp tục
            return {"id": i, "query": s, field: "", "error": repr(e)[:200]}

    t0 = time.time()
    fails = 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            fails += not r[field]
            if n % 25 == 0 or n == len(todo):
                print(f"[{time.time() - t0:6.0f}s] {n}/{len(todo)} thất bại {fails}", flush=True)


if __name__ == "__main__":
    main()
