"""Trích nội dung chính từ HTML và nhận diện trang bị chặn (captcha/anti-bot)."""
import json

import trafilatura

# Dấu hiệu trang verify/captcha đã gặp khi crawl thử (xem docs/CONTEXT.md)
BLOCK_URL_MARKERS = ("verify.html", "/captcha", "waf", "security-check")
BLOCK_TEXT_MARKERS = ("滑动拼图验证", "安全验证", "访问验证", "人机验证", "Just a moment", "Attention Required")
BLOCK_STATUSES = {403, 429, 503, 521, 522, 523, 524}
PERMANENT_STATUSES = {404, 410}
MIN_TEXT_LEN = 200


def extract(body: bytes, url: str) -> dict:
    """Trả về {'title', 'date', 'text'}; text rỗng nếu không trích được.

    favor_recall=True: thà lấy dư boilerplate còn hơn mất nội dung, vì chunk vàng có thể nằm ở bất kỳ đâu
    và overlap chỉ chia cho độ dài chunk vàng.
    """
    out = trafilatura.extract(
        body, url=url, output_format="json", with_metadata=True,
        include_comments=False, include_tables=True, favor_recall=True,
    )
    if not out:
        return {"title": "", "date": "", "text": ""}
    d = json.loads(out)
    return {"title": d.get("title") or "", "date": d.get("date") or "", "text": d.get("text") or ""}


def classify(status: int | None, final_url: str, text: str, error: str | None = None) -> str:
    """Phân loại kết quả fetch.

    ok        : có nội dung dùng được
    blocked   : bị chặn/captcha/rate-limit -> host cần lùi tốc độ, URL thử lại sau
    permanent : 404/410 hoặc trang không có nội dung -> không thử lại
    transient : timeout, lỗi mạng, 5xx khác -> thử lại sau
    """
    if error:
        return "transient"
    if status in BLOCK_STATUSES or any(m in (final_url or "").lower() for m in BLOCK_URL_MARKERS):
        return "blocked"
    if status in PERMANENT_STATUSES:
        return "permanent"
    if status != 200:
        return "transient"
    if len(text) < MIN_TEXT_LEN and any(m in text for m in BLOCK_TEXT_MARKERS):
        return "blocked"
    return "ok" if len(text) >= MIN_TEXT_LEN else "permanent"
