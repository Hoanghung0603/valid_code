import os
import sys
import glob
import time
import random
import asyncio
import unicodedata
import argparse
import collections
import itertools
import threading
import re
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import pandas as pd
import trafilatura
from lxml import html as lxml_html
from tqdm.asyncio import tqdm
from curl_cffi import requests
import concurrent.futures

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ======================================================================
# 1. BỘ LỌC NỘI DUNG (URL BLACKLIST & MEDICAL KEYWORDS)
# ======================================================================
URL_BLACKLIST = [
    "/rao-vat/", "/san-pham/", "/gio-hang/", "/mua-ban/", "/cart/",
    "/checkout/", "/shop/", "/store/"
]

MEDICAL_KEYWORDS = [
    "bác sĩ", "bệnh", "triệu chứng", "điều trị", "thuốc", "sức khỏe", "ung thư", "viêm", "đau", "chẩn đoán", "huyết áp", "phẫu thuật",
    "khám", "y tế", "bệnh nhân", "phòng khám", "dược", "tim mạch", "tiểu đường", "dinh dưỡng", "xét nghiệm", "nhi khoa", "sản khoa",
    "thần kinh", "cơ xương khớp", "hô hấp", "tiêu hóa", "đông y", "nam dược", "thảo dược", "vắc-xin", "vaccine", "chữa trị",
    "phòng ngừa", "cấp cứu", "lâm sàng", "cơ địa", "miễn dịch", "nhiễm trùng", "virus", "vi khuẩn", "hội chứng", "mãn tính", "cấp tính",
    "医院", "医生", "治疗", "疾病", "症状", "药", "患者", "健康", "诊断", "手术", "科",
    "doctor", "disease", "symptom", "treatment", "medicine", "health", "cancer", "pain", "diagnosis", "blood pressure", "surgery", "clinic",
    "medical", "patient", "hospital", "pharmacy", "syndrome", "infection", "therapy"
]

# Some publishers expose valid short briefs or use markup that Trafilatura
# does not recognize. These domains are accepted once a real content node is
# found, even when the cleaned article is shorter than the global threshold.
DOMAIN_MIN_TEXT_LENGTH = {
    "laodong.vn": 10,
    "baotayninh.vn": 10,
}

DOMAIN_CONTENT_XPATHS = {
    "laodong.vn": [
        "//article[contains(concat(' ', normalize-space(@class), ' '), ' article-detail ')]",
        "//div[contains(concat(' ', normalize-space(@class), ' '), ' blk-c2-01 ')]",
    ],
    "baotayninh.vn": [
        "//*[@id='content-detail']",
        "//div[contains(concat(' ', normalize-space(@class), ' '), ' content-article ')]",
    ],
    "benhviennhitrunguong.gov.vn": [
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' entry-content ')]",
    ],
    "bachmai.gov.vn": [
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' post-detail__content ')]",
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' post-content ')]",
    ],
    "youmed.vn": [
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' article-content ')]",
    ],
}


def _domain_matches(domain: str, expected: str) -> bool:
    normalized = domain.lower().split(":", 1)[0].removeprefix("www.")
    return normalized == expected or normalized.endswith("." + expected)


def min_text_length_for_domain(domain: str, default: int = 50) -> int:
    for expected, minimum in DOMAIN_MIN_TEXT_LENGTH.items():
        if _domain_matches(domain, expected):
            return minimum
    return default


def _normalize_extracted_text(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _europe_pmc_metadata_text(item: dict) -> str:
    """Build source-faithful searchable text from Europe PMC metadata."""
    pieces = []
    for value in (item.get("title"), item.get("abstractText"), item.get("authorString")):
        if value:
            pieces.append(str(value))

    journal = item.get("journalInfo") or {}
    journal_meta = journal.get("journal") or {}
    journal_bits = [
        journal_meta.get("title"),
        journal.get("dateOfPublication"),
        item.get("pageInfo"),
        f"DOI: {item['doi']}" if item.get("doi") else None,
    ]
    journal_line = "; ".join(str(value) for value in journal_bits if value)
    if journal_line:
        pieces.append(journal_line)

    mesh = item.get("meshHeadingList") or {}
    descriptors = [
        heading.get("descriptorName")
        for heading in mesh.get("meshHeading", [])
        if isinstance(heading, dict) and heading.get("descriptorName")
    ]
    if descriptors:
        pieces.append("MeSH: " + "; ".join(dict.fromkeys(descriptors)))
    return _normalize_extracted_text("\n".join(pieces))


def _europe_pmc_full_text(item: dict) -> str:
    """Fetch article body when a PubMed record has no abstract but has a PMCID."""
    pmcid = item.get("pmcid")
    if not pmcid:
        return ""
    try:
        response = requests.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
            timeout=20,
        )
        if response.status_code != 200:
            return ""
        root = ET.fromstring(response.content)
        body = root.find(".//body")
        if body is None:
            return ""
        body_text = _normalize_extracted_text(" ".join(body.itertext()))
        title = _normalize_extracted_text(item.get("title") or "")
        return _normalize_extracted_text("\n".join(value for value in (title, body_text) if value))
    except Exception:
        return ""


def extract_text_with_fallback(html: str, domain: str) -> str:
    """Extract article text with deterministic domain-aware HTML fallbacks."""
    primary = _normalize_extracted_text(
        trafilatura.extract(html, include_links=False, include_images=False) or ""
    )

    try:
        tree = lxml_html.fromstring(html)

        # Prefer articleBody from structured data when publishers expose it.
        structured_candidates = []
        for node in tree.xpath("//script[@type='application/ld+json']/text()"):
            try:
                payload = json.loads(node)
            except Exception:
                continue
            stack = payload if isinstance(payload, list) else [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    body = item.get("articleBody")
                    if body:
                        structured_candidates.append(_normalize_extracted_text(body))
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)

        # Remove script/style noise before selector-based text_content().
        for noisy in tree.xpath("//script|//style|//noscript|//svg|//form"):
            parent = noisy.getparent()
            if parent is not None:
                parent.remove(noisy)

        selector_candidates = []
        for expected, xpaths in DOMAIN_CONTENT_XPATHS.items():
            if not _domain_matches(domain, expected):
                continue
            for xpath in xpaths:
                for element in tree.xpath(xpath):
                    text = _normalize_extracted_text(element.text_content())
                    if text:
                        selector_candidates.append(text)
            break

        if not selector_candidates:
            for element in tree.xpath("//article|//main"):
                text = _normalize_extracted_text(element.text_content())
                if text:
                    selector_candidates.append(text)

        # A few legacy articles return a metadata-only shell (HTTP 200) even
        # though the canonical page still exists. Preserve its title/summary
        # instead of classifying it as an empty article.
        metadata_candidates = []
        if _domain_matches(domain, "baotayninh.vn"):
            title = tree.xpath("string(//meta[@property='og:title']/@content)")
            description = tree.xpath("string(//meta[@property='og:description']/@content)")
            metadata = _normalize_extracted_text("\n".join(filter(None, [title, description])))
            if description and metadata:
                metadata_candidates.append(metadata)

        fallback_candidates = structured_candidates + selector_candidates + metadata_candidates
        fallback = max(fallback_candidates, key=len) if fallback_candidates else ""
    except Exception:
        fallback = ""

    minimum = min_text_length_for_domain(domain)
    valid = [text for text in (primary, fallback) if len(text) >= minimum]
    if not valid:
        return max((primary, fallback), key=len)

    # Domain selectors avoid modal/navigation boilerplate that can be longer
    # than the article returned by Trafilatura (notably laodong.vn).
    if fallback and len(fallback) >= minimum:
        return fallback
    return max(valid, key=len)


def resolve_javascript_cookie_challenge(req, url: str, headers: dict, proxies, impersonate_mode: str):
    """Retry simple publisher cookie challenges such as Lao Dong's D1N page."""
    html = req.text or ""
    match = re.search(r'document\.cookie=["\']([A-Za-z0-9_]+)=([^"\'+;]+)', html)
    if not match:
        return req
    cookie_name, cookie_value = match.groups()
    return requests.get(
        url,
        proxies=proxies,
        headers={**headers, "Referer": url},
        cookies={cookie_name: cookie_value},
        timeout=15,
        impersonate=impersonate_mode,
    )

def check_layer_1(url: str):
    u = url.lower()
    for b in URL_BLACKLIST:
        if b in u: return f"Blacklist: {b}"
    return None

def check_layer_2(text: str):
    # Đã vô hiệu hóa theo yêu cầu: Giữ 100% nội dung để bảo toàn trần Recall F2
    return None

# ======================================================================
# 2. PROXY POOL (XOAY VÒNG ROUND-ROBIN 1 HOẶC 2 PROXY)
# ======================================================================
class FastProxyPool:
    def __init__(self, proxy_list):
        self.proxies = [p for p in proxy_list if p]
        self._cycle = itertools.cycle(self.proxies) if self.proxies else None
        self._lock = threading.Lock()

    def get_proxy(self):
        if not self._cycle:
            return None
        with self._lock:
            return next(self._cycle)

    def __len__(self):
        return len(self.proxies)

# ======================================================================
# 3. DOMAIN CIRCUIT BREAKER (AUTO-SKIP DOMAINS CHẾT / BLOCK CỨNG)
# ======================================================================
class DomainTracker:
    def __init__(self, failure_threshold: int = 15):
        self.threshold = failure_threshold
        self.lock = threading.Lock()
        self.stats = collections.defaultdict(lambda: {'fail': 0, 'consecutive_fails': 0, 'success': 0, 'skipped': False})

    def get_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    def should_skip(self, domain: str) -> bool:
        if not domain or self.threshold <= 0:
            return False
        with self.lock:
            return self.stats[domain]['skipped']

    def record_success(self, domain: str):
        if not domain:
            return
        with self.lock:
            self.stats[domain]['success'] += 1
            self.stats[domain]['consecutive_fails'] = 0

    def record_failure(self, domain: str):
        if not domain or self.threshold <= 0:
            return
        with self.lock:
            data = self.stats[domain]
            if data['skipped']:
                return
            data['fail'] += 1
            data['consecutive_fails'] += 1
            if data['consecutive_fails'] >= self.threshold:
                data['skipped'] = True
                print(f"\n🚫 [Domain Circuit Breaker] Domain '{domain}' đã thất bại {data['consecutive_fails']} lần liên tiếp -> TỰ ĐỘNG SKIP TOÀN BỘ URL CÒN LẠI!")

def decode_html_bytes(content: bytes, domain: str, default_text: str = "") -> str:
    if any(d in domain for d in ['cnkang.com', 'wujue.com', 'zydcd.com', 'a-hospital.com']):
        try:
            cand = content.decode('gb18030')
            if cand.count('\ufffd') < 5:
                return cand
        except Exception:
            pass

    try:
        cand = content.decode('utf-8')
        if cand.count('\ufffd') < 5:
            return cand
    except Exception:
        pass

    try:
        cand_gb = content.decode('gb18030', errors='ignore')
        return cand_gb
    except Exception:
        pass

    return default_text

# ======================================================================
# 4. WORKER CRAWL MỘT URL
# ======================================================================
def fetch_one_sync(row: dict, proxy_pool: FastProxyPool, max_retries: int, domain_tracker: DomainTracker):
    url = row['url']
    row_id = row['id']
    res = {'id': row_id, 'url': url, 'text': None, 'status': None, 'reason': '', 'dropped_at_layer': None}

    domain = domain_tracker.get_domain(url)
    if domain_tracker.should_skip(domain):
        # Đánh dấu error để KHÔNG ghi vào file parquet và không mất URL
        res['status'] = 'error'
        res['reason'] = f"Domain Blocked/Skipped ({domain})"
        return res

    l1 = check_layer_1(url)
    if l1:
        res['status'] = 'dropped'
        res['dropped_at_layer'] = 1
        res['reason'] = l1
        return res

    # 0. Xử lý chuyên biệt cho pubmed.ncbi.nlm.nih.gov qua Europe PMC Open API
    if "pubmed.ncbi.nlm.nih.gov" in domain or "pubmed.ncbi.nlm.nih.gov" in url:
        m = re.search(r'/(\d+)/?', url)
        if m:
            pmid = m.group(1)
            api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20SRC:MED&format=json&resultType=core"
            try:
                api_req = requests.get(api_url, timeout=12)
                if api_req.status_code == 200:
                    data = api_req.json()
                    items = data.get('resultList', {}).get('result', [])
                    if items:
                        item = items[0]
                        title = item.get('title', '')
                        abstract = item.get('abstractText', '')
                        full_text = f"{title}\n\n{abstract}".strip() if abstract else _europe_pmc_full_text(item)
                        source = "Europe PMC abstract/full text"
                        if not full_text:
                            full_text = _europe_pmc_metadata_text(item)
                            source = "Europe PMC metadata"
                        if full_text and len(full_text) > 10:
                            res['text'] = full_text
                            res['status'] = 'success'
                            res['reason'] = f'Success (PubMed {source})'
                            domain_tracker.record_success(domain)
                            return res
                        else:
                            res['status'] = 'dropped'
                            res['dropped_at_layer'] = 2
                            res['reason'] = "No abstract in PubMed article"
                            return res
                    else:
                        # Secondary-index absence is not source deletion.
                        # Continue to the original PubMed URL below.
                        res['reason'] = "PMID not found in Europe PMC (retryable)"
            except Exception as e:
                res['status'] = 'error'
                res['reason'] = f"Europe PMC error: {str(e)[:40]}"
                domain_tracker.record_failure(domain)
                return res

    # Do not permanently drop a publisher merely because its staging hostname
    # was blocked previously. Exact-link API recovery is available separately
    # in recover_pharmacity_api.py; this generic path still probes the origin.

    crawl_url = url
    impersonate_mode = "chrome120"
    if _domain_matches(domain, "youmed.vn") and "/tin-tuc/" in url:
        # The canonical article route intermittently triggers Cloudflare's
        # managed challenge, while its ?amp=1 representation serves the same
        # canonical article reliably. Keep `url` unchanged in the registry.
        crawl_url = url.rstrip("/") + "/?amp=1"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
        }
        impersonate_mode = "safari17_0"
    elif _domain_matches(domain, "cnkang.com"):
        parsed = urlparse(url)
        crawl_url = f"https://m.cnkang.com{parsed.path}"
        if parsed.query:
            crawl_url += f"?{parsed.query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.5',
        }
        impersonate_mode = "safari17_0"
    elif _domain_matches(domain, "msdmanuals.cn"):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.5',
        }
        impersonate_mode = "safari17_0"
    elif "ask.39.net/question/" in url:
        # The old wapask endpoint now rate-limits aggressively; the canonical
        # desktop article contains the answer and is more reliable.
        crawl_url = url
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.5',
        }
        impersonate_mode = "chrome124"
    else:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
            
    for attempt in range(max_retries + 1):
        proxy_url = proxy_pool.get_proxy()
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        try:
            req = requests.get(crawl_url, proxies=proxies, headers=headers, timeout=15, impersonate=impersonate_mode)
            req = resolve_javascript_cookie_challenge(req, crawl_url, headers, proxies, impersonate_mode)
            html = decode_html_bytes(req.content, domain, req.text)
            
            # 1. Kiểm tra mã lỗi 404 / 410 (Trang đã bị xóa vĩnh viễn)
            if req.status_code in [404, 410] or "<title>404 Not Found</title>" in html:
                res['status'] = 'dropped'
                res['dropped_at_layer'] = 1
                res['reason'] = f"HTTP {req.status_code} Not Found"
                break

            # 2. Kiểm tra mã lỗi 429 Too Many Requests / Rate Limited
            if req.status_code == 429 or "too many requests" in html.lower() or "rate limit" in html.lower():
                res['reason'] = "429 Too Many Requests"
                if attempt < max_retries:
                    time.sleep(2.0)
                    continue
                else:
                    break

            # 3. Kiểm tra mã lỗi 403 Forbidden do Nginx/Server
            if req.status_code == 403 and "nginx" in html.lower() and "forbidden" in html.lower():
                res['reason'] = "HTTP 403 Forbidden (Nginx Block)"
                break

            # 4. Kiểm tra mã lỗi 5xx Server Error
            if req.status_code >= 500:
                res['reason'] = f"HTTP {req.status_code} Server Error"
                if attempt < max_retries:
                    time.sleep(2.0)
                    continue
                else:
                    break

            text = extract_text_with_fallback(html, domain)
            min_text_length = min_text_length_for_domain(domain)
            has_extractable_content = bool(text and len(text.strip()) >= min_text_length)

            # 5. Kiểm tra dấu hiệu Cloudflare / Anti-bot Block / Turnstile / Captcha.
            # Some publishers (notably YouMed) embed a Turnstile widget in a
            # form on otherwise valid article pages, so that marker alone is
            # only blocking when no article content can be extracted.
            is_cf = (
                "cf-browser-verification" in html or 
                "cookies must be enabled" in html.lower() or
                "www error blocked diagnostic" in html.lower() or
                "Just a moment..." in html or 
                "Attention Required!" in html or 
                "owner of this website" in html.lower() or
                "滑动拼图" in html or
                "拼图验证" in html or
                "人机验证" in html or
                (
                    not has_extractable_content
                    and (
                        "cf-turnstile" in html.lower()
                        or "challenges.cloudflare.com/turnstile" in html.lower()
                    )
                ) or
                (req.status_code in [403, 503] and ("cloudflare" in html.lower() or "turnstile" in html.lower() or "waf" in html.lower()))
            )
            
            if is_cf:
                res['reason'] = "Anti-bot / Captcha Block"
                if len(proxy_pool) > 1 and attempt < max_retries:
                    time.sleep(1.0)
                    continue
                else:
                    break
                
            if text and len(text.strip()) >= min_text_length:
                lower_text = text.lower()

                # Kiểm tra thông báo lỗi hệ thống ẩn bên trong nội dung văn bản
                if "too many requests" in lower_text or "rate limit" in lower_text:
                    res['reason'] = "429 Too Many Requests"
                    if attempt < max_retries:
                        time.sleep(2.0)
                        continue
                    else:
                        break

                if "owner of this website" in lower_text or "cf-browser-verification" in lower_text or "just a moment..." in lower_text or "attention required" in lower_text:
                    res['reason'] = "Cloudflare Block/Challenge"
                    if attempt < max_retries:
                        time.sleep(15.5 if len(proxy_pool) == 1 else 1.0)
                        continue
                    else:
                        break

                if "403 forbidden" in lower_text or "access denied" in lower_text:
                    res['reason'] = "HTTP 403 Forbidden"
                    break

                # Văn bản thực sự hợp lệ -> LƯU THÀNH CÔNG (KHÔNG LỌC TỪ KHÓA)
                res['text'] = text.strip()
                res['status'] = 'success'
                res['reason'] = 'Success'
                break
            else:
                # Text rỗng hoặc quá ngắn (< 50 ký tự)
                html_lower = html.lower()
                is_turnstile = (
                    "cf-turnstile" in html_lower
                    or "challenges.cloudflare.com/turnstile" in html_lower
                    or "cf-browser-verification" in html_lower
                    or "<title>just a moment" in html_lower
                )
                if is_turnstile:
                    res['reason'] = "Cloudflare Turnstile"
                    if attempt < max_retries:
                        if len(proxy_pool) == 1:
                            time.sleep(15.5)
                        else:
                            time.sleep(1.0)
                        continue
                    else:
                        break
                else:
                    if text and len(text.strip()) > 0:
                        # Có text nhưng vẫn dưới ngưỡng tối thiểu của domain.
                        res['status'] = 'dropped'
                        res['dropped_at_layer'] = 2
                        res['reason'] = f"Text too short ({len(text.strip())} chars < {min_text_length})"
                        break
                    else:
                        res['reason'] = 'Empty text'
                        if attempt < max_retries:
                            time.sleep(1.5)
                            continue
                        else:
                            break
        except Exception as e:
            res['reason'] = type(e).__name__
            # Fallback: nếu proxy lỗi kết nối, thử lại trực tiếp (direct) không qua proxy
            if proxies:
                try:
                    req_direct = requests.get(url, headers=headers, timeout=12, impersonate="chrome120")
                    if req_direct.status_code == 200:
                        html_d = decode_html_bytes(req_direct.content, domain, req_direct.text)
                        text_d = extract_text_with_fallback(html_d, domain)
                        if text_d and len(text_d.strip()) >= min_text_length_for_domain(domain):
                            res['text'] = text_d.strip()
                            res['status'] = 'success'
                            res['reason'] = 'Success (Direct Fallback)'
                            break
                except:
                    pass
            if attempt < max_retries:
                time.sleep(1.5)
                continue
            else:
                break

    # Nếu hết lượt retry mà vẫn chưa success/dropped -> gán status = 'error'
    if res['status'] is None:
        res['status'] = 'error'

    # Ghi nhận domain stats
    if res['status'] == 'success' or (res['status'] == 'dropped' and res['dropped_at_layer'] == 2):
        domain_tracker.record_success(domain)
    elif res['status'] == 'error' or (res['status'] == 'dropped' and '404' in res['reason']):
        domain_tracker.record_failure(domain)

    if res['status'] != 'success':
        print(f"\n[Debug] URL: {url} | Status: {res['status']} | Reason: {res['reason']}")

    return res

# ======================================================================
# 5. ASYNC BATCH RUNNER
# ======================================================================
async def crawl_batch(df_batch, sem, delay, proxy_pool, max_retries, pbar, executor, domain_tracker):
    loop = asyncio.get_running_loop()
    async def process_row(row):
        async with sem:
            domain = domain_tracker.get_domain(row['url'])
            if not domain_tracker.should_skip(domain) and delay > 0:
                await asyncio.sleep(random.uniform(delay * 0.7, delay * 1.3))
            return await loop.run_in_executor(executor, fetch_one_sync, row, proxy_pool, max_retries, domain_tracker)
    tasks = [process_row(row) for _, row in df_batch.iterrows()]
    results = []
    success_count = 0
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        if r['status'] == 'success':
            success_count += 1
        pbar.update(1)
        pbar.set_postfix({'success': success_count, 'status': r['status'] if r['status'] else r['reason']})
    return results

def get_already_crawled_ids(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.parquet"))
    crawled_ids = set()
    for f in files:
        try:
            df = pd.read_parquet(f)
            status = df['status'].fillna('').astype(str)
            reason = df.get('reason', pd.Series('', index=df.index)).fillna('').astype(str)
            unavailable = reason.str.match(
                r'^HTTP\s+(?:404|410)\s+Not Found$', case=False
            )
            safe_blacklist = reason.str.contains(r'blacklist', case=False, regex=True) & ~reason.str.contains(
                r'blacklist:\s*(?:shopee|lazada|tiki)', case=False, regex=True
            )
            permanent_drop = status.eq('dropped') & (unavailable | safe_blacklist)
            done = status.eq('success') | permanent_drop
            crawled_ids.update(df.loc[done, 'id'].tolist())
        except: pass
    return crawled_ids

# ======================================================================
# 6. MAIN CONTROLLER
# ======================================================================
async def main_async(args, proxy_pool, domain_tracker):
    print("=" * 70)
    print("🚀 FAST REQUESTS CRAWLER (ROTATING PROXY POOL)")
    print("=" * 70)
    print(f"📌 Input file:     {args.input}")
    print(f"📌 Output dir:     {args.output_dir}")
    print(f"📌 Proxy Mode:     {args.proxy_mode.upper()} ({len(proxy_pool)} proxy active)")
    for i, p in enumerate(proxy_pool.proxies, 1):
        print(f"   [{i}] {p.split('@')[-1]}")
    print(f"📌 Concurrency:    {args.concurrency}")
    print(f"📌 Delay jitter:   {args.delay}s")
    print(f"📌 Max Retries:    {args.max_retries}")
    print(f"📌 Domain Limit:   {args.domain_fail_limit}")
    print(f"📌 Batch size:     {args.batch_size}")
    if args.limit:
        print(f"📌 Limit URLs:     {args.limit:,}")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    df_rem = pd.read_parquet(args.input)
    print(f"\n1. Đang đọc danh sách URL cần cào...")
    print(f"   - Tổng URLs trong input: {len(df_rem):,}")
    if "crawled" in df_rem.columns:
        already_c = df_rem["crawled"].sum()
        df_rem = df_rem[~df_rem["crawled"]].copy()
        print(f"   - Tự động bỏ qua {already_c:,} URLs đã có trạng thái crawled=True. Còn lại: {len(df_rem):,} URLs cần cào.")

    if args.domain:
        is_target = df_rem['url'].str.contains(args.domain, case=False, na=False)
        df_target = pd.concat([df_rem[is_target], df_rem[~is_target]])
        print(f"   - Ưu tiên domain: {args.domain} ({is_target.sum():,} URLs)")
    else:
        df_target = df_rem.copy()

    crawled_ids = get_already_crawled_ids(args.output_dir)
    if crawled_ids:
        df_target = df_target[~df_target['id'].isin(crawled_ids)].copy()
        print(f"   - Bỏ qua {len(crawled_ids):,} IDs đã cào. Còn lại: {len(df_target):,} URLs")

    if args.limit and args.limit > 0 and len(df_target) > args.limit:
        df_target = df_target.head(args.limit).copy()
        print(f"   - Giới hạn lượt này: {len(df_target):,} URLs")

    if len(df_target) == 0:
        print("🎉 Không còn URL nào cần cào. Hoàn tất!")
        return

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)
    num_batches = (len(df_target) // args.batch_size) + (1 if len(df_target) % args.batch_size > 0 else 0)

    existing_parts = [
        int(os.path.basename(f).replace("local_part_", "").replace(".parquet", ""))
        for f in glob.glob(os.path.join(args.output_dir, "local_part_*.parquet"))
        if os.path.basename(f).replace("local_part_", "").replace(".parquet", "").isdigit()
    ]
    start_part = max(existing_parts) + 1 if existing_parts else 0

    total_success = 0
    total_processed = 0
    print(f"\n2. Bắt đầu cào ({num_batches} batches, từ part {start_part:04d})...")

    with tqdm(total=len(df_target), desc="Crawling", unit="URL") as pbar:
        for b_idx in range(num_batches):
            part_no = start_part + b_idx
            out_file = os.path.join(args.output_dir, f"local_part_{part_no:04d}.parquet")
            s = b_idx * args.batch_size
            e = min(s + args.batch_size, len(df_target))
            batch_df = df_target.iloc[s:e]
            
            batch_results = await crawl_batch(batch_df, sem, args.delay, proxy_pool, args.max_retries, pbar, executor, domain_tracker)
            df_res = pd.DataFrame(batch_results)
            
            # Chỉ lưu những URL thành công hoặc được phân loại là rác (dropped)
            df_valid = df_res[df_res['status'].isin(['success', 'dropped'])]
            df_errors = df_res[df_res['status'].eq('error')]
            succ = (df_res['status'] == 'success').sum()

            if len(df_valid) > 0: 
                df_valid.to_parquet(out_file, index=False)
            if len(df_errors) > 0:
                # Diagnostic-only ledger: nested files are ignored by the
                # root-level sync and already-crawled checks, so URLs retry.
                error_dir = os.path.join(args.output_dir, "errors")
                os.makedirs(error_dir, exist_ok=True)
                error_file = os.path.join(
                    error_dir,
                    f"error_part_{part_no:04d}_{time.time_ns()}.parquet",
                )
                df_errors.to_parquet(error_file, index=False)
                
            total_success += succ
            total_processed += len(df_res)
            
    executor.shutdown(wait=False)

    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH PHIÊN CÀO FAST!")
    print(f"   - Tổng URL đã xử lý: {total_processed:,}")
    print(f"   - Thành công: {total_success:,} ({total_success/total_processed*100:.1f}%)" if total_processed > 0 else "")
    print(f"   - Dữ liệu lưu tại: {args.output_dir}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Fast Asynchronous Requests Crawler (Multi-Proxy Rotation)")
    parser.add_argument("--input", default="data/remaining_urls.parquet")
    parser.add_argument("--output-dir", default="data/raw/corpus_text/local_recovered")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--proxy-mode", choices=["both", "15s", "240s", "none"], default="none",
                        help="Chế độ proxy: 'both' (xoay vòng cả 2 proxy), '15s' (chỉ proxy 15s), '240s' (chỉ proxy 240s), 'none' (Direct IP)")
    parser.add_argument("--proxy", default=None, help="Chỉ định custom proxy URL (ghi đè --proxy-mode)")
    parser.add_argument("--proxy-file", default=None, help="Đọc danh sách proxy từ file txt (1 proxy / dòng)")
    parser.add_argument("--domain-fail-limit", type=int, default=15, help="Tự động skip domain sau N lần fail liên tiếp (default: 15)")
    args = parser.parse_args()

    # Xử lý danh sách Proxy
    proxy_list = []
    if args.proxy_file and os.path.exists(args.proxy_file):
        with open(args.proxy_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
        proxy_list = list(dict.fromkeys(lines))
        print(f"🌐 [Proxy] Đã nạp {len(proxy_list)} proxy từ {args.proxy_file}")
    elif args.proxy:
        proxy_list = [args.proxy]
        print(f"🌐 [Proxy] Sử dụng Custom Proxy chỉ định: {args.proxy.split('@')[-1]}")
    elif args.proxy_mode == "none":
        print("🌐 [Proxy] Chế độ DIRECT IP (Không dùng Proxy)")
    else:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from src.config import settings
            p15 = getattr(settings, "PROXY_15S", None)
            p240 = getattr(settings, "PROXY_240S", None)

            if args.proxy_mode == "both":
                if p15: proxy_list.append(p15)
                if p240: proxy_list.append(p240)
            elif args.proxy_mode == "15s":
                if p15: proxy_list.append(p15)
            elif args.proxy_mode == "240s":
                if p240: proxy_list.append(p240)
        except Exception as e:
            print(f"⚠️ Không thể đọc proxy từ settings: {e}")

    proxy_pool = FastProxyPool(proxy_list)
    domain_tracker = DomainTracker(failure_threshold=args.domain_fail_limit)

    asyncio.run(main_async(args, proxy_pool, domain_tracker))

if __name__ == "__main__":
    main()
