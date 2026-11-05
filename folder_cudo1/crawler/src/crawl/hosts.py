"""Phân loại host trong links_corpus (97 host: ~83% tiếng Trung, ~17% tiếng Việt)."""
import re
from urllib.parse import urlsplit

_ZH = re.compile(r"\.cn$|39\.net|cnkang|120ask|zysjonline|a-hospital|zhongyibaodian|zydcd|wujue|iiyi|"
                 r"qihuangzhishu|pmphai|jb39|youlai|baidu")

# Thăm dò ngày 2026-10-04: các host này chưa lấy được, crawl_pages.py bỏ qua trừ khi --include-blocked.
BLOCKED = {
    "zysjonline.com": "Cloudflare thử thách JS (giả lập TLS Chrome vẫn 403, cần trình duyệt thật)",
    "bingli.iiyi.com": "HTTP 521",
}

# Chặn theo dấu vân tay TLS/HTTP2: httpx và curl thường bị 403, curl_cffi giả lập Chrome thì 200.
IMPERSONATE = {"nhathuoclongchau.com.vn", "vov.vn"}

# Giây tối thiểu giữa 2 request cùng host (ghi đè --delay nếu lớn hơn).
# nhathuoclongchau: 2026-10-04 sau ~500 trang ở 4-17 req/s thì trả 429 với Retry-After ~24h; ngưỡng thật chưa đo được.
# giadinhonline, youmed: trả 429 tạm thời ở 4 req/s, crawl lại chậm hơn thì được.
HOST_DELAY = {"nhathuoclongchau.com.vn": 2.0, "giadinhonline.vn": 1.0, "youmed.vn": 1.0}

# Host staging của Pharmacity trả 403 cho mọi client; cùng bài viết có ở site chính với cùng slug.
_PHARMACITY_BETA = "pmc-ecm-healthblog.beta.pharmacity.io"


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


def fetch_url(url: str) -> str:
    """URL thực sự dùng để tải nội dung của một URL trong corpus (khác URL gốc khi phải lấy qua site chính)."""
    parts = urlsplit(url)
    if parts.netloc.lower() == _PHARMACITY_BETA and parts.path.strip("/"):
        return f"https://www.pharmacity.vn/{parts.path.strip('/')}.htm"
    return url


def host_lang(host: str) -> str:
    return "zh" if _ZH.search(host) else "vi"
