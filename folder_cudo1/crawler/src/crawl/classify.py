"""Giải mã HTML và phân loại kết quả fetch (ok / bị chặn / lỗi / trang rỗng) cho crawler lịch sự (crawl.polite)."""
import re
from urllib.parse import urlsplit

from charset_normalizer import from_bytes

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,zh-CN;q=0.8,zh;q=0.7,en;q=0.6",
}

_META_CS = re.compile(rb'<meta[^>]+charset=["\']?\s*([A-Za-z0-9_\-]+)', re.I)


def decode_html(raw: bytes, header_ct: str = "") -> tuple[str, str]:
    """bytes -> (html, encoding). Trang tiếng Trung hay là GBK/GB2312 mà header không ghi charset:
    ưu tiên <meta charset> khi header thiếu hoặc header nói utf-8 còn meta nói họ gb*."""
    cs = None
    m = re.search(r"charset=([\w\-]+)", header_ct or "", re.I)
    if m:
        cs = m.group(1)
    m2 = _META_CS.search(raw[:4096])
    meta_cs = m2.group(1).decode("ascii", "ignore") if m2 else None
    if meta_cs and (not cs or (cs.lower().replace("-", "") == "utf8" and "gb" in meta_cs.lower())):
        cs = meta_cs
    if cs and cs.lower() in ("gb2312", "gbk"):
        cs = "gb18030"  # tập cha, tránh lỗi giải mã
    if cs:
        try:
            return raw.decode(cs), cs
        except (LookupError, UnicodeDecodeError):
            pass
    best = from_bytes(raw).best()
    return (str(best), best.encoding) if best else (raw.decode("utf-8", "replace"), "utf-8?")


# Dấu hiệu trang chặn bot / captcha. Crawler chỉ nhận diện để lùi lại (backoff), không tìm cách vượt qua.
SIGS = [
    ("cloudflare_challenge", r"just a moment\.\.\.|/cdn-cgi/challenge-platform|cf-chl-|cf_chl_opt"),
    ("aliyun_waf",           r"acw_sc__v2|aliyun_waf|_waf_bd8ce2ce37|nc_1_wrapper|滑动验证"),
    ("knownsec_jsl",         r"__jsl_clearance|jsl_clearance_s"),
    ("360_wzws",             r"wzws_cid|_wzws|wzwschallenge"),
    ("yunsuo",               r"yunsuo_session|yunsuo_waf"),
    ("tencent_waf",          r"waf\.tencent-cloud\.com|tencent cloud waf|EO_Bot_Ssid"),
    ("baidu_verify",         r"wappass\.baidu\.com|百度安全验证"),
    ("generic_captcha",      r"captcha|验证码|人机验证|安全验证|are you a robot|xác minh bạn không phải"),
    ("access_denied",        r"access denied|request blocked|forbidden|访问被拒绝|您的访问.{0,10}(异常|受限)"),
]
SIGS = [(n, re.compile(p, re.I)) for n, p in SIGS]
# từ khoá chung cũng xuất hiện trong bài thật (vd bài hướng dẫn "SMS 验证码"): chỉ tin khi lỗi HTTP hoặc trang gần rỗng
WEAK_SIGS = {"generic_captcha", "access_denied"}


def classify(status: int, headers: dict, html: str, text_len: int, final_url: str, req_url: str) -> tuple[str, str]:
    """-> (label, detail); label: ok / blocked:* / http_error:* / redirect_home / js_render / soft_404_or_empty."""
    h = {k.lower(): v for k, v in (headers or {}).items()}
    head = html[:20000]
    if h.get("cf-mitigated") == "challenge":
        return "blocked:cloudflare_challenge", "cf-mitigated header"
    for name, rx in SIGS:
        limit = 600 if name in WEAK_SIGS else 1500
        if rx.search(head) and (status >= 400 or text_len < limit):
            return f"blocked:{name}", rx.pattern[:40]
    if status in (403, 429, 503, 520, 521, 522, 525):
        return f"blocked:http_{status}", h.get("server", "")
    if status >= 400:
        return f"http_error:{status}", ""
    fu, ru = urlsplit(final_url), urlsplit(req_url)
    if fu.path in ("", "/") and ru.path not in ("", "/"):  # bị chuyển về trang chủ = bài đã mất
        return "redirect_home", final_url
    if text_len < 200 and len(html) > 2000 and re.search(r"<script", html, re.I):
        return "js_render", f"html={len(html)} text={text_len}"
    if text_len < 200:
        return "soft_404_or_empty", f"text={text_len}"
    return "ok", ""


def visible_text_len(html: str) -> int:
    """Ước lượng nhanh độ dài text hiển thị (không phải nội dung chính)."""
    t = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return len(re.sub(r"\s+", " ", t).strip())
