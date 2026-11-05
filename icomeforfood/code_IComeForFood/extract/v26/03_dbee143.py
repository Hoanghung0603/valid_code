# %% CELL 4
import importlib
import subprocess
import sys

COLAB = 'google.colab' in sys.modules
# pip distribution name -> Python import name (they are not always identical).
required_packages = [
    ('pandas>=2.0', 'pandas'),
    ('pyarrow>=14', 'pyarrow'),
    ('httpx[http2]>=0.25.0', 'httpx'),
    ('charset-normalizer', 'charset_normalizer'),
    ('selectolax', 'selectolax'),
    ('trafilatura', 'trafilatura'),
    ('resiliparse', 'resiliparse'),
    ('python-dotenv', 'dotenv'),
    ('tqdm', 'tqdm'),
    ('lxml', 'lxml'),
    ('zstandard', 'zstandard'),
    ('duckdb>=1.5', 'duckdb'),
    ('h2', 'h2'),
]

def _missing_imports():
    missing = []
    for distribution, module in required_packages:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append((distribution, module))
    return missing

missing = _missing_imports()
if missing:
    packages_to_install = [distribution for distribution, _ in missing]
    print(f"Đang cài {len(packages_to_install)} thư viện: {packages_to_install}")
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', *packages_to_install],
        check=True,
    )
    still_missing = _missing_imports()
    if still_missing:
        details = ', '.join(f'{dist} (import {module})' for dist, module in still_missing)
        raise ImportError(f"Cài xong nhưng vẫn không import được: {details}. Hãy khởi động lại runtime rồi chạy lại cell này.")
    print('Đã cài xong và kiểm tra import thành công các thư viện Gold.')
else:
    print('Môi trường đã đủ thư viện Gold.')


# %% CELL 5
import os, sys, re, json, hashlib, time, warnings
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter, defaultdict
import logging

import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

# ---- CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU NGOÀI REPO ----
# Trên Colab, chọn đúng tài khoản Google Drive huylenhat1701 (hoặc dùng shortcut tới đúng r2ai_data).
# Không tự tạo thư mục nguồn; cell đối chiếu phải xác nhận dữ liệu đã có trước khi clean.
if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = Path('/content/drive/MyDrive/r2ai_data')
else:
    DATA_ROOT = Path('D:/r2ai_data')


RAW_DIR = DATA_ROOT / 'raw'
CRAWL_DIR = DATA_ROOT / 'crawl'
HTML_DIR = CRAWL_DIR / 'html'
PUBMED_DIR = CRAWL_DIR / 'pubmed'
DOCS_DIR = DATA_ROOT / 'docs'
REPORT_DIR = DATA_ROOT / 'reports'
AUDIT_DIR = REPORT_DIR / 'audit'
LOG_DIR = DATA_ROOT / 'logs'
print(f'DATA_ROOT = {DATA_ROOT}')

# %% CELL 6
# Repo source đã được nhúng trong notebook; hints lấy cùng thư mục dữ liệu Drive.
PROJECT_ROOT = None
CONFIG_ROOT = DATA_ROOT / 'crawl'
for _required_config in ('container_hints.csv', 'domain_plan.csv'):
    if not (CONFIG_ROOT / _required_config).is_file():
        raise FileNotFoundError(
            f'Thiếu {CONFIG_ROOT / _required_config}. Hãy đặt hai file cấu hình vào DATA_ROOT/crawl trên Drive.'
        )
print(f'DATA_ROOT = {DATA_ROOT}')
print(f'CONFIG_ROOT = {CONFIG_ROOT}')

# %% CELL 7
# ---- LOGGING: console + file xoay vòng theo ngày ----
LOG_DIR.mkdir(parents=True,exist_ok=True)
log_file = LOG_DIR / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logger = logging.getLogger('vibiomir')
logger.setLevel(logging.INFO)
for _old_handler in list(logger.handlers):
    try:
        _old_handler.flush()
        _old_handler.close()
    except Exception:
        pass
    logger.removeHandler(_old_handler)

fmt = logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = logging.FileHandler(log_file, encoding='utf-8')
fh.setFormatter(fmt)
logger.addHandler(fh)

def flush_log_handlers(sync=False):
    # Flush open log handlers; sync=True also asks the OS to persist file data.
    for _handler in logger.handlers:
        try:
            _handler.flush()
            if sync and isinstance(_handler, logging.FileHandler):
                os.fsync(_handler.stream.fileno())
        except (OSError, ValueError):
            pass

logger.info(f'Log file: {log_file}')
flush_log_handlers(sync=True)


# %% CELL 8
# ---- .env ----
from dotenv import load_dotenv

for p in [Path('.env'), Path('../.env'), DATA_ROOT / '.env']:
    if p.exists():
        load_dotenv(p)
        logger.info(f'Đọc .env từ: {p}')
        break
else:
    logger.info('Không có .env – OK với dataset public.')

HF_TOKEN = os.getenv('HF_TOKEN')
logger.info('HF_TOKEN: ' + ('đã nạp' if HF_TOKEN else 'không có (dataset public)'))


# %% CELL 10
# ---- Encoding helper ----
from charset_normalizer import from_bytes

_CHARSET_RE = re.compile(r'charset=[\"\']?([\w-]+)', re.I)

def detect_encoding(raw_bytes: bytes, content_type: str = '') -> str:
    """Content-Type → <meta charset> → charset_normalizer. gb2312/gbk → gb18030."""
    if content_type:
        m = _CHARSET_RE.search(content_type)
        if m:
            enc = m.group(1).lower()
            return 'gb18030' if enc in ('gb2312','gbk','x-gbk') else enc
    head = raw_bytes[:4096].decode('ascii', errors='replace')
    m = _CHARSET_RE.search(head)
    if m:
        enc = m.group(1).lower()
        return 'gb18030' if enc in ('gb2312','gbk','x-gbk') else enc
    res = from_bytes(raw_bytes[:32768]).best()
    return res.encoding if res else 'utf-8'

def extract_laodong_cookie(body_text: str) -> Optional[Dict[str, str]]:
    m = re.search(r'document\.cookie\s*=\s*[\"\']D1N=([^\'\"\;]+)', body_text)
    return {'D1N': m.group(1).strip()} if m else None


# %% CELL 11
# ---- Helper PubMed qua Europe PMC (Gold subset gọi helper ở cell Gold) ----
import httpx
import zstandard as zstd
import hashlib
import json
import time
from typing import List, Dict, Any, Optional

EUROPE_PMC_POST_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST'

def fetch_pubmed_batches(pmid_list: List[str], batch_size: int = 1000,
                         max_batches: Optional[int] = None) -> List[Dict[str, Any]]:
    """Lấy PubMed theo lô, cursorMark, cache từng lô và retry request lỗi."""
    pmids = [str(p).strip() for p in pmid_list if str(p).strip()]
    if batch_size <= 0:
        raise ValueError('batch_size phải lớn hơn 0')
    total_needed = len(pmids) if max_batches is None else min(len(pmids), max_batches * batch_size)
    cctx = zstd.ZstdCompressor(level=3, threads=-1)
    dctx = zstd.ZstdDecompressor()
    all_hits: List[Dict[str, Any]] = []
    delays = [1.0, 3.0, 10.0, 30.0]

    with httpx.Client(timeout=60.0, http2=True) as client:
        for b_idx, start in enumerate(tqdm(range(0, total_needed, batch_size), desc='Europe PMC')):
            chunk = pmids[start:min(start + batch_size, total_needed)]
            digest = hashlib.sha1('|'.join(chunk).encode('utf-8')).hexdigest()[:10]
            cache_path = PUBMED_DIR / f'batch_{b_idx:05d}_{digest}.json.zst'

            if cache_path.exists():
                try:
                    cached = json.loads(dctx.decompress(cache_path.read_bytes()).decode('utf-8'))
                    if isinstance(cached, list):
                        all_hits.extend(cached)
                        continue
                except Exception:
                    logger.warning(f'Cache PubMed hỏng, sẽ tải lại: {cache_path.name}')

            query_str = f'EXT_ID:({" OR ".join(chunk)}) AND SRC:MED'
            cursor_mark = '*'
            batch_hits: Dict[str, Dict[str, Any]] = {}

            while True:
                data = {
                    'query': query_str,
                    'resultType': 'core',
                    'format': 'json',
                    'pageSize': str(batch_size),
                    'cursorMark': cursor_mark,
                }
                payload = None
                last_error = None
                for req_attempt, delay in enumerate(delays):
                    try:
                        resp = client.post(EUROPE_PMC_POST_URL, data=data)
                        resp.raise_for_status()
                        payload = resp.json()
                        break
                    except Exception as e:
                        last_error = e
                        if req_attempt < len(delays) - 1:
                            logger.warning(f'PubMed batch {b_idx} request {req_attempt + 1} lỗi; retry sau {delay:g}s: {e}')
                            time.sleep(delay)

                if payload is None:
                    raise RuntimeError(f'PubMed batch {b_idx} thất bại sau {len(delays)} lần') from last_error

                hits = payload.get('resultList', {}).get('result', []) or []
                for hit in hits:
                    pmid = str(hit.get('pmid', '')).strip()
                    if pmid:
                        batch_hits[pmid] = hit

                next_cursor = payload.get('nextCursorMark')
                time.sleep(1.0)
                if not next_cursor or next_cursor == cursor_mark:
                    break
                cursor_mark = next_cursor

            batch_list = list(batch_hits.values())
            tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
            tmp_path.write_bytes(cctx.compress(json.dumps(batch_list, ensure_ascii=False).encode('utf-8')))
            os.replace(tmp_path, cache_path)
            all_hits.extend(batch_list)
            logger.info(f'PubMed batch {b_idx}: {len(batch_list):,}/{len(chunk):,} PMID')

    by_pmid = {}
    for hit in all_hits:
        pmid = str(hit.get('pmid', '')).strip()
        if pmid:
            by_pmid[pmid] = hit
    return list(by_pmid.values())

# %% CELL 13
# ---- Bộ công cụ làm sạch ----
try:
    from selectolax.lexbor import LexborHTMLParser as HTMLParser
except ImportError:
    from selectolax.parser import HTMLParser
from html import unescape as html_unescape

DOMAIN_LANG_MAP: Dict[str, str] = {}
CONTAINER_HINTS: Dict[str, Dict[str, Any]] = {}
TITLE_HINTS: Dict[str, str] = {}

df_plan = pd.read_csv(CONFIG_ROOT / 'domain_plan.csv')
for _, r in df_plan.iterrows():
    dom = str(r['domain']).strip().lower().removeprefix('www.')
    DOMAIN_LANG_MAP[dom] = str(r['lang']).strip().lower()

df_hints = pd.read_csv(CONFIG_ROOT / 'container_hints.csv')
HINTS_SOURCE = str(CONFIG_ROOT / 'container_hints.csv')
EXPECTED_CONTAINER_HINTS_SHA1 = '405ecc3099f7819cb6fb450992e1b7523334e2b3'
# Băm theo nội dung chuẩn hóa LF để file CSV có CRLF trên Windows/Drive không bị báo sai.
CONTAINER_HINTS_RAW = Path(HINTS_SOURCE).read_bytes()
CONTAINER_HINTS_RAW_SHA1 = hashlib.sha1(CONTAINER_HINTS_RAW).hexdigest()
CONTAINER_HINTS_SHA1 = hashlib.sha1(CONTAINER_HINTS_RAW.replace(b'\r\n', b'\n')).hexdigest()
if CONTAINER_HINTS_SHA1 != EXPECTED_CONTAINER_HINTS_SHA1:
    raise RuntimeError(f'container_hints.csv sai SHA-1: {CONTAINER_HINTS_SHA1}; cần {EXPECTED_CONTAINER_HINTS_SHA1}. Hãy đồng bộ file hints từ main rồi chạy lại.')

for _, r in df_hints.iterrows():
    dom = str(r['domain']).strip().lower().removeprefix('www.')
    main_c = str(r['main_container']).strip() if pd.notna(r.get('main_container')) else ''
    if dom == 'test.pmphai.com':
        if 'covers_all_content' not in df_hints.columns:
            raise ValueError('container_hints.csv thiếu cột covers_all_content cho test.pmphai.com')
        _covers = r.get('covers_all_content')
        if pd.isna(_covers) or not str(_covers).strip():
            raise ValueError('Thiếu covers_all_content cho test.pmphai.com trong container_hints.csv')
        main_c = re.sub(r'\s*\([^)]*%\)\s*$', '', str(_covers)).strip()
    also_c = str(r['also']).strip() if pd.notna(r.get('also')) else ''
    title_c = str(r['title']).strip() if pd.notna(r.get('title')) else ''
    also_sels = [re.sub(r'\(.*?\)', '', p).strip() for p in also_c.split(';')] if also_c else []
    CONTAINER_HINTS[dom] = {'main': main_c, 'also': [s for s in also_sels if s]}
    if title_c:
        TITLE_HINTS[dom] = title_c

logger.info(f'Loaded {len(CONTAINER_HINTS)} container hints from {HINTS_SOURCE}; {len(DOMAIN_LANG_MAP)} domain langs')
print(f'Container hints: {len(CONTAINER_HINTS)} từ {HINTS_SOURCE}')
_ZW_RE = re.compile(r'[\u200b\u2060\ufeff]')
def normalize_ws(text: str) -> str:
    if not text: return ''
    return re.sub(r'\s+', ' ', _ZW_RE.sub('', text)).strip()

def parse_published_date(value: Any) -> Optional[str]:
    if value is None: return None
    raw = normalize_ws(str(value))
    if not raw: return None
    for cand in (raw[:10], raw):
        for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%b %d, %Y'):
            try: return datetime.strptime(cand, fmt).date().isoformat()
            except ValueError: pass
    m = re.fullmatch(r'(\d{4})年(\d{1,2})月(\d{1,2})日?', raw)
    if m:
        try: return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date().isoformat()
        except ValueError: return None
    return None

SAFE_TAG_RE = re.compile(r'</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>')
def strip_html_tags_safe(text: str) -> str:
    return html_unescape(SAFE_TAG_RE.sub('', text))

SOFT_404_RE = re.compile(r'không tìm thấy|khong tim thay|page not found|not found|页面不存在|该问题已被删除|bài viết đã bị xóa', re.I)
def is_soft_404(text: str) -> bool:
    return bool(SOFT_404_RE.search(text[:600]))

ROLE_MAP = {
    'toc':        re.compile(r'(?<![a-z0-9])(?:muc-luc|table[-_ ]of[-_ ]contents|toc)', re.I),
    'reference':  re.compile(r'(?<![a-z0-9])(?:tai-lieu-tham-khao|references?|bibliography|nguon-tham-khao)(?![a-z0-9])', re.I),
    'disclaimer': re.compile(r'(?<![a-z0-9])(?:mien-tru-trach-nhiem|disclaimer|khuyen-cao)(?![a-z0-9])', re.I),
    'lead':       re.compile(r'(?<![a-z0-9])(?:sapo|lead|intro|mo-dau)(?![a-z0-9])', re.I),
    'faq':        re.compile(r'(?<![a-z0-9])(?:faq|cau-hoi-thuong-gap)(?![a-z0-9])', re.I),
}

def guess_role(node, text: str) -> Optional[str]:
    attrs = f"{node.attributes.get('class') or ''} {node.attributes.get('id') or ''}".lower()
    for role, pat in ROLE_MAP.items():
        if pat.search(attrs) or pat.search(normalize_ws(text).casefold()):
            return role
    return None

_SEL_PART_RE = re.compile(r'([#.])((?:\[[^\]]*\]|[^#.\[\s])+)')
def _css_fallback(selector: str) -> Optional[str]:
    """`div.max-w-[597px]` không parse được → `div[class~="max-w-[597px]"]`."""
    m = re.match(r'^([A-Za-z][A-Za-z0-9-]*)?(.*)$', selector.strip())
    tag, rest = (m.group(1) or ''), m.group(2)
    parts = _SEL_PART_RE.findall(rest)
    if not parts or ''.join(k + v for k, v in parts) != rest:
        return None
    return tag + ''.join(f'[class~="{v}"]' if k == '.' else f'[id="{v}"]' for k, v in parts)

def safe_css(node, selector):
    try:
        return node.css(selector)
    except Exception:
        alt = _css_fallback(selector)
        if alt:
            try: return node.css(alt)
            except Exception: pass
        return []

def safe_css_first(node, selector):
    m = safe_css(node, selector)
    return m[0] if m else None

CJK_RE = re.compile(r'[\u4e00-\u9fff]')
CONTAINER_BLOCK_TAGS = {'p','li','h1','h2','h3','h4','h5','h6','tr','table','blockquote','div','ul','ol','section','article'}

# =====================================================================
# Hằng số cho bộ trích xuất duyệt-DOM (custom_cleaner_v3, bản sửa)
# =====================================================================
sys.setrecursionlimit(max(sys.getrecursionlimit(), 6000))   # DOM lồng sâu

# Luôn bỏ, ở bất kỳ đâu. KHÔNG đưa <form> vào đây: nhiều CMS (.NET) bọc cả trang trong <form>.
HARD_JUNK_TAGS = {
    'script', 'style', 'noscript', 'iframe', 'svg', 'canvas', 'template', 'object', 'embed',
    'button', 'input', 'select', 'textarea', 'audio', 'video', 'map', 'datalist',
    'head', 'title', 'meta', 'link', 'base',
}
STRUCT_JUNK_TAGS = {'header', 'nav', 'footer'}
JUNK_ROLES = {'navigation', 'banner', 'contentinfo', 'complementary', 'search'}
EXACT_JUNK_CLASSES = {
    'breadcrumb', 'breadcrumbs', 'breadcrumb-item',
    'related', 'related-posts', 'related-articles', 'related-news', 'tin-lien-quan',
    'social-share', 'share-buttons', 'social',
    'comment', 'comments', 'binh-luan',
    'ads', 'adsbygoogle', 'advertisement', 'banner', 'quang-cao',
    'booking', 'dat-lich', 'hotline', 'popup',
    'author-info', 'byline', 'newsletter',
}
EXACT_JUNK_IDS = {'comments'}
# Khớp theo "từ" (ranh giới là ký tự không phải chữ/số: '-', '_', ' ') nên `content-fck-font-size` không dính.
FUZZY_JUNK_RE = re.compile(
    r'(?<![a-z0-9])(?:share|sharing|social|print|toolbar|related|lien-?quan|comments?|binh-?luan|'
    r'advert\w*|ads?|adv|banner|quang-?cao|popup|modal|newsletter|subscribe|sidebar|breadcrumb\w*|'
    r'tags?|keywords?|tu-?khoa|most-?read|doc-?nhieu|read-?more|doc-?them|xem-?them|'
    r'bai-?viet-?lien-?quan|tin-?moi|tin-?khac|cung-?chuyen-?muc|copyright|fb-)(?![a-z0-9])')
# Ranh giới "hết bài": chỉ dùng cho đoạn NỐI TIẾP phía sau container.
STOP_CLASS_RE = re.compile(
    r'(?<![a-z0-9])(?:related|lien-?quan|comments?|binh-?luan|tags?|keywords?|tu-?khoa|most-?read|'
    r'doc-?nhieu|read-?more|doc-?them|xem-?them|bai-?viet-?lien-?quan|tin-?moi|tin-?khac|'
    r'cung-?chuyen-?muc|footer|sidebar|newsletter)(?![a-z0-9])')
STOP_TEXT_RE = re.compile(
    r'^(?:từ khóa|từ khoá|tags?|bạn có thể quan tâm|có thể bạn quan tâm|tin liên quan|bài viết liên quan|'
    r'bài liên quan|tin cùng chuyên mục|cùng chuyên mục|đọc thêm|đọc nhiều|xem nhiều|xem thêm|tin mới nhất|'
    r'tin khác|bình luận|ý kiến bạn đọc|mọi phản ánh|chia sẻ bài viết|chịu trách nhiệm|bản quyền|copyright|©|'
    r'liên hệ quảng cáo|related (?:articles|posts)|read more|comments?)\b', re.I)
_HIDDEN_STYLE_RE = re.compile(r'display\s*:\s*none|visibility\s*:\s*hidden', re.I)

# Dòng rác giao diện (khớp toàn dòng).
BOILERPLATE_FULL_RE = re.compile(
    r'^(?:(?:A\s*){1,4}|(?i:cỡ chữ:?(?:\s*A\s*[+-]?)*|in bài(?: viết)?|in trang|gửi email|chia sẻ(?: bài viết)?:?|'
    r'share|tweet|facebook|zalo|twitter|messenger|copy link|sao chép liên kết|lưu bài|đọc báo giấy|'
    r'bình luận|quay lại|về đầu trang|back to top|\d+\s*bình luận)|[|•·\-–—/]+)$')
CONTACT_START_RE = re.compile(
    r'^(?:mọi phản ánh|mọi thông tin phản ánh|mọi ý kiến|ý kiến đóng góp|tin, bài cộng tác|gửi tin, bài|'
    r'liên hệ quảng cáo|liên hệ đặt báo)', re.I)
CONTACT_FOLLOW_RE = re.compile(r'^(?:email|e-mail|hotline|điện thoại|đt|tel|fax|zalo|website)\b', re.I)
LABEL_LINE_RE = re.compile(
    r'^(?:từ khóa|từ khoá|tags?|xem thêm|đọc thêm|tin liên quan|bài liên quan|bài viết liên quan|xem tiếp|đọc tiếp)\b',
    re.I)
DATESTAMP_RE = re.compile(
    r'^\d{1,2}[/.-]\d{1,2}[/.-]\d{4}(?:\s*[-–|,]?\s*\d{1,2}[:h]\d{2}(?:\s*\(?GMT[+-]?\d+\)?)?)?$')
BAODANANG_PREFIX_RE = re.compile(r'^(?:[A-Z]\.){1,3}[A-Z]?\s*[•·]?\s*\d{2}/\d{2}/\d{4}(?:\s+\d{2}:\d{2})?\s*')

HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
LEAF_TEXT_TAGS = {'p', 'pre', 'figcaption', 'dd', 'dt', 'caption', 'summary', 'address'}
CONTAINER_TAGS = {
    'div', 'section', 'article', 'main', 'ul', 'ol', 'dl', 'figure', 'blockquote', 'details', 'center',
    'body', 'html', 'form', 'fieldset', 'tbody', 'thead', 'tfoot', 'tr', 'td', 'th', 'hgroup', 'menu',
    'aside', 'header', 'footer', 'nav',
}
SPACE_TAGS = CONTAINER_TAGS | LEAF_TEXT_TAGS | HEADING_TAGS | {'li', 'table', 'br', 'hr'}

GENERIC_CONTAINER_SELECTORS = [
    'article', 'main', '.detail-content', '.article-content', '.post-content', '.content-detail',
    '#content', '#main-content', '.main-content', '.entry-content',
]
FALLBACK_CONTAINER_SELECTORS = [
    '[itemprop="articleBody"]', '.article-body', '.detail-content', '.article-content', '.post-content',
    '.content-detail', '.entry-content', '.detail__content', '#content', '#main-content',
    '.main-content', 'article', 'main',
]
CONTINUE_AFTER_CONTAINER = True     # nối tiếp phần bài bị HTML lỗi đẩy ra ngoài container
CONTINUE_MAX_ANCESTOR_LEVELS = 3
CONTINUE_MAX_BLOCKS = 400
NO_CONTINUE_DOMAINS: set = set()    # domain muốn tắt nối tiếp: {'example.com'}
INTEGRITY_AUDIT = True              # thêm cờ 'possible_truncation' (tốn thêm 1 lượt duyệt DOM/trang)


# ---- Postfilter nhúng: extract/postfilter.py v1.12 ----
POSTFILTER_EMBEDDED_SOURCE = r'''"""Shared junk filter applied AFTER extraction (DAC_TA section 4, "loại bỏ hoàn toàn").

Drops blocks that are template junk on every page of a site: layout duplicates, author/source
lines, "related" link lists, product cards, UI button labels, footers, image credits, copyright
notices, asker/doctor profile lines on Q&A sites. It never edits a kept block, so kept text stays
verbatim, and it rebuilds `text` and char offsets. Extractors must not dedup on their own: this
module owns deduplication (see dedup_blocks for which repeats are real content).

Validated on the gold set (eval/score_extraction.py): it must not lower recall on any page.
Every rule change is also checked on real docs by sampling the blocks it newly drops.

Usage:
    from extract.postfilter import filter_doc
    doc = filter_doc(doc)              # doc = {"text", "blocks", ...} in schema v2
    python -I extract/postfilter.py <in_dir> <out_dir>   # filter a folder of <id>.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# exact UI labels / section titles that are never content
UI_LABELS = {
    # vi
    "Xem tất cả", "Kích thước chữ", "Mặc định", "Lớn hơn", "Có thể bạn quan tâm", "Chia sẻ", "Đã kiểm duyệt",
    "Không có từ khóa theo chữ cái đã chọn", "☰ Mục lục", "Tham vấn y khoa:", "Bình luận", "Gửi bình luận",
    "Xem thêm:", "Xem thêm", "Đọc thêm", "Tin liên quan", "TIN LIÊN QUAN", "Bài viết liên quan", "Cùng chuyên mục",
    "Các bài viết liên quan", "NỘI DUNG LIÊN QUAN", "Bài viết có hữu ích với bạn?", "Cần tư vấn từ bác sĩ",
    # zh
    "隐藏详情", "我要咨询", "找医生", "举报", "举报/反馈", "搜索", "文", "没有更多了", "更多", "相关推荐", "医生作品",
    "推荐搜索", "优惠推荐", "您最近浏览的手术", "精彩推荐", "手术图文", "最受欢迎育儿热文", "其他人还在看",
    "相关问题", "相关文章推荐", "向医生提问", "栏目关注：", "分享到：", "疾病百科", "投诉", "其他类似问题",
    "回复追问：", "取消 提交",
    # en
    "Test your KnowledgeTake a Quiz!",
}
# a label that also takes the next block with it (label + value)
PAIR_LABELS = {"作者": 1, "评论": 1, "Đã kiểm duyệt nội dung": 1}
# after these markers, following short blocks are a link list until a long block appears
LIST_MARKERS = {"最受欢迎育儿热文", "其他人还在看", "Có thể bạn quan tâm", "Xem thêm:", "Xem thêm", "Tin liên quan",
                "TIN LIÊN QUAN", "Bài viết liên quan", "Cùng chuyên mục", "相关推荐", "推荐搜索", "精彩推荐", "相关文章推荐",
                "Các bài viết liên quan", "NỘI DUNG LIÊN QUAN", "其他类似问题",
                "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH"}   # tamanhhospital address/hotline box at the end of each post
# reference sections are content even though their items look like links
REF_HEADING = re.compile(r"^(nguồn|tài liệu)\s+tham\s+khảo|^tham\s+khảo$|^参考(文献|资料)|^references?$", re.I)

# Layout duplicates (DAC_TA 4): drop a later run of >= DUP_RUN consecutive blocks that repeats an earlier
# run, or a single long block (>= DUP_SINGLE chars) seen before. Short repeats are real content in the gold
# set (title restated as question / first heading, TOC entry vs heading, "Ảnh minh họa." captions).
DUP_RUN, DUP_RUN_CHARS, DUP_SINGLE = 3, 100, 100
QA_SITES = ("ask.39.net", "120ask.com")         # askers repeat themselves; the gold keeps every copy

# Source / author lines: only checked in the last TAIL_BLOCKS blocks of a document, because
# "Theo dõi ...", "Theo y học ..." are ordinary Vietnamese content (checked on 32K real docs).
TAIL_BLOCKS = 4
NOT_SOURCE = re.compile(r"^\(?Theo\s+(dõi|chân|đó|như|y học|Y học|Đông y|quan niệm|nghiên cứu|thống kê|khuyến cáo|"
                        r"hướng dẫn|các|một|nhiều|ý kiến)\b|[:：]$")
TAIL_PATTERNS = [re.compile(p) for p in (
    r"^\(?Theo\s[^.!?]{1,90}\)?$",
    r"^[^\s!?]{1,20}(\s[^\s.!?]{1,20}){0,4}\s\(Theo [^)]{1,60}\)$",   # "T. Linh (Theo Aboluowang)"
    r"^(CTV|PV|BTV)\b[^!?]{0,60}$",                            # "CTV Lương Trâm/VOV.VN (Biên dịch)"
)]
AUTHOR = re.compile(
    r"^([A-ZĐ][A-ZĐ.]{0,5}"                                   # PV, A.T, TTXVN
    r"|[A-ZĐÀ-Ỹ][\wÀ-ỹ.]*(\s[A-ZĐÀ-Ỹ][\wÀ-ỹ.]*){1,3}"         # Doãn Phong, Ngọc Lam, Trần Thị Thu
    r"|[\w.-]+\.(vn|com|net|org)"                               # Baoquangbinh.vn
    r"|Ảnh:\s.{1,30})$")

PATTERNS = [re.compile(p) for p in (
    r"^-?>{1,2}\s",                         # "-> ..." / ">> ..." inline related links
    r"^→\s",
    # dates / bylines glued to dates
    r"^•?\s*\d{1,2}/\d{1,2}/\d{4}(\s+\d{1,2}:\d{2})?$",
    r"^Cập nhật( lần cuối)?:?\s*\d",
    r"(?i)nguồn ưu tiên trên Google",       # "Chọn Bệnh viện ... làm nguồn ưu tiên trên Google tìm kiếm" (tamanh, tuoitre, vnexpress)
    r"^(?!Ngày (tham khảo|truy cập))[^\d\s][^\d]{1,30}\d{1,2}/\d{1,2}/\d{4}$",   # a reference's access date stays
    r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?$",
    r"^\w?\*{3,}\w?\s+\d{4}-\d{2}-\d{2}",       # ask.39 asker "a******5 2014-03-11 14:16"
    r"^(www\.)?[\w-]+(\.[\w-]+)+\s+\d{4}-\d{1,2}-\d{1,2}",  # "www.cnkang.com 2015-8-8 互联网"
    # VN footer / booking / reviewer boxes. Masthead labels only as "Label: value" or alone:
    # "Giám đốc Bệnh viện K cho biết...", "Giấy phép nhập khẩu ... được gia hạn" are content
    r"^(Cơ quan chủ quản|Giám đốc|Tổng [Bb]iên tập|Phó [Tt]ổng [Bb]iên tập|Giấy phép|Đường dây nóng|"
    r"Trụ sở( chính)?)\s*([:：]|$)",
    r"^Giấy phép\b.{0,40}\d+/GP",          # "Giấy phép hoạt động số 101/GP-BTTTT ..."
    r"^(@ Bản quyền|© |Bản quyền|Thông tin tòa soạn|Liên hệ quảng cáo)",
    r"(Để đặt lịch khám|vui lòng bấm số HOTLINE|Liên hệ hotline \d)",
    r"^(Bác sĩ duyệt bài|Dược sĩ Đại học|Nội dung được kiểm duyệt)",
    r"^(GP thiết lập|Không được sao chép|Người chịu trách nhiệm nội dung)",
    r"^Chủ đề\s*:",                         # tag line "Chủ đề:Gạo lứtCaloGiảm cân" (TV, huyPrime1)
    r"^\d+ lượt (xem|chơi|thích)$",
    r"^/\s?[^\s\d]{1,8}$",                  # "/ Lọ", "/ Hộp" unit next to a price
    r"^(Tác giả|Chuyên khoa|Ngày đăng)\s*:.{0,60}$",   # youmed byline box
    r'^\[[a-z_]+(\s+[a-z_]+="[^"]*")*\]$',  # unrendered WordPress shortcode '[wpcode id="293453"]'
    # zh doctor profile, credits, copyright, wiki chrome
    r"^\S{2,10}\s+(主任医师|副主任医师|主治医师|医师|住院医师)(\s|$)",   # "白云山汉方 医师 广州…"
    r"^疾病百科[•\-]",                       # 39.net / a-hospital nav "疾病百科•出血倾向"
    r"^擅长：",
    r"^图片由.{1,40}提供。?$",
    r"^39健康网\(www\.39\.net\)专稿",
    r"^（以上内容仅授权.{0,40}）$",
    r"^39健康网专业医疗保健信息平台",
    r"^中国领先的健康门户网站",
    r"未经(书面)?授权请勿转载。?$",
    r"^跳转到：",
    r"^文章录入[：:]",                        # wujue.com editor credit "文章录入：管理员 责任编辑：…"
    r"^\d+个分类:",
    r"^发表于\s*\d{4}-\d{1,2}-\d{1,2}",          # 120ask follow-up question timestamp
    r"^\d+条追问$",                           # ask.39 follow-up counter; the follow-up itself ("追问： …") stays
    r"^\S{1,12}医生推荐$",                    # youlai box "眼科医生推荐" after the article
    r"^出自A\+医学百科",
    r"^关于“.{1,60}”的留言",
    r"^如您发现信息有误",
)]
# boilerplate longer than the 200-char limit above; anchored, because older extractors glue a
# whole article into one block that contains it
LONG_PATTERNS = [re.compile(p) for p in (
    r"^Để đặt lịch khám tại .{0,40}Quý khách vui lòng",   # vinmec booking box, 200-250 chars
)]
# app promo box (youmed), matched with whitespace removed because older extractors glue its lines
PROMO = re.compile(r"^(Đặtkhámtiệnlợicùng|Đặtkhámkhôngchờđợi)|^((\+\d+(Bệnhviện|Bácsĩ|Phòngkhám))+|Nhắntinvớibácsĩ|"
                   r"Gọivideovớibácsĩ|Muasảnphẩmytếgiátốt|Lưutrữhồsơytế|Đọctinytếchínhthống|TảiứngdụngYouMed)$")
# ask.39: a row of space-separated keywords is a tag or related-search row. Site-bound, because on
# wiki/TCM sites the same shape is an herb list or a table row.
ZH_TAG_ROW = re.compile(r"^(?=.*[一-鿿])[一-鿿A-Za-z0-9]{1,8}( [一-鿿A-Za-z0-9]{1,8}){2,}$")
TAG_ROW_SITES = ("ask.39.net",)
TERMINAL = re.compile(r"[.。!！;；:：]$")
TEASER_LEAD = re.compile(r"\s(\.\.\.|…)$")   # lead cut mid-sentence: "... thu hồi toàn bộ lô mỹ phẩm mặt ..."
# a product card: heading, a few short lines, then a price ("390.000đ") or "contact us" instead of a price
PRICE = re.compile(r"^(\d{1,3}(\.\d{3})+\s?(đ|₫|VNĐ)|Cần tư vấn từ bác sĩ)$")


def is_junk(text: str) -> bool:
    t = text.strip()
    if t in UI_LABELS:
        return True
    if len(t) <= 200 and any(p.search(t) for p in PATTERNS):
        return True
    if len(t) <= 400 and any(p.search(t) for p in LONG_PATTERNS):
        return True
    if len(t) <= 200 and PROMO.search(re.sub(r"\s+", "", t)):
        return True
    return False


def _is_author_tail(blocks: list[dict], i: int) -> bool:
    """Source or author line near the end: 'Theo GDTĐ', 'PV', 'A.T', 'Doãn Phong', 'Baoquangbinh.vn'."""
    b = blocks[i]
    if b.get("type") != "paragraph" or i < len(blocks) - TAIL_BLOCKS:
        return False
    t = b["text"].strip()
    if len(t) > 120 or NOT_SOURCE.search(t):
        return False
    if any(p.search(t) for p in TAIL_PATTERNS):
        return True
    return len(t) <= 30 and not re.search(r"[一-鿿\d()]", t) and bool(AUTHOR.match(t))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedup_blocks(blocks: list[dict], site: str = "") -> list[dict]:
    t = [_norm(b.get("text", "")) for b in blocks]
    seen: dict[str, list[int]] = {}
    drop, i = set(), 0
    while i < len(t):
        run = 0                                            # longest earlier run that block i starts repeating
        for j in seen.get(t[i], []):
            k = 0
            while i + k < len(t) and j + k < i and t[j + k] == t[i + k]:
                k += 1
            run = max(run, k)
        if run >= DUP_RUN and sum(map(len, t[i:i + run])) >= DUP_RUN_CHARS:
            drop.update(range(i, i + run))
            i += run
            continue
        adjacent = bool(t[i]) and t[i] in seen and seen[t[i]][-1] == i - 1
        # the title emitted twice (page title + h1): adjacent identical headings go at any length
        if adjacent and blocks[i].get("type") == blocks[i - 1].get("type") == "heading":
            drop.add(i)
        # a long title restated as a later heading stays (gold keeps both); its adjacent copy goes
        elif (not site.endswith(QA_SITES) and len(t[i]) >= DUP_SINGLE and t[i] in seen
                and (blocks[i].get("type") != "heading" or adjacent)):
            drop.add(i)
        seen.setdefault(t[i], []).append(i)
        i += 1
    return [b for k, b in enumerate(blocks) if k not in drop]


def _is_content(b: dict) -> bool:
    t = b.get("text", "").strip()
    return len(t) > 100 or bool(re.search(r"[.。!！]$", t))


def _is_linklike(b: dict) -> bool:
    t = re.sub(r"(\.\.\.|…)$", "", b.get("text", "").strip())     # "Trái na: Ngăn ngừa ... tóc và da..."
    return len(t) <= 120 and not TERMINAL.search(t) and not REF_HEADING.search(t)


def _card_blocks(blocks: list[dict]) -> set[int]:
    """Product cards: the heading before a price line, the short lines between, unit/origin lines after."""
    drop = set()
    for i, b in enumerate(blocks):
        if not PRICE.match(b.get("text", "").strip()):
            continue
        for j in range(i - 1, max(i - 4, 0) - 1, -1):
            if len(blocks[j].get("text", "").strip()) > 30:
                break
            if blocks[j].get("type") == "heading":
                drop.update(range(j, i + 1))
                break
        drop.add(i)
        k = i + 1
        while k < len(blocks) and blocks[k].get("type") != "heading" and len(blocks[k].get("text", "").strip()) <= 10:
            drop.add(k)
            k += 1
    return drop


def _cut_tail(blocks: list[dict]) -> list[dict]:
    """Related-article lists at the end: a source line followed only by links, or >= 4 bare headings
    that read like article titles (short section headings with a lost body are kept), or >= 2
    teaser pairs (title + lead cut off with "...") of an inline related box."""
    k = len(blocks)
    while (k >= 2 and blocks[k - 1].get("type") == blocks[k - 2].get("type") == "paragraph"
           and len(blocks[k - 1].get("text", "").strip()) <= 300 and TEASER_LEAD.search(blocks[k - 1].get("text", "").strip())
           and _is_linklike(blocks[k - 2])):
        k -= 2
    if len(blocks) - k >= 4:
        blocks = blocks[:k]
    n = len(blocks)
    for s in range(3, n - 3):                          # related lists can outnumber a short article's blocks
        t = blocks[s].get("text", "").strip()
        # short and comma-free, so "Theo bác sĩ X, người bệnh cần ..." before a closing list is never cut
        if (blocks[s].get("type") == "paragraph" and len(t) <= 50 and "," not in t and not NOT_SOURCE.search(t)
                and any(p.search(t) for p in TAIL_PATTERNS) and all(map(_is_linklike, blocks[s + 1:]))):
            return blocks[:s]
    k = n
    while k > 1 and blocks[k - 1].get("type") == "heading" and _is_linklike(blocks[k - 1]):
        k -= 1
    tail, body = blocks[k:], sum(b.get("type") != "heading" for b in blocks[:k])
    # body >= 5: a page whose paragraphs were lost keeps its section headings
    if len(tail) >= 4 and body >= 5 and sum(len(b.get("text", "").strip()) for b in tail) / len(tail) >= 45:
        return blocks[:k]
    return blocks


def filter_blocks(blocks: list[dict], site: str = "") -> list[dict]:
    keep, skip_next, in_list = [], 0, False
    cards = _card_blocks(blocks)
    for i, b in enumerate(blocks):
        t = b.get("text", "").strip()
        if skip_next:
            skip_next -= 1
            continue
        if b.get("type") == "heading" and i == 0:          # never drop the title
            keep.append(b)
            continue
        if i in cards:
            continue
        if t in PAIR_LABELS:
            skip_next = PAIR_LABELS[t]
            continue
        if t in LIST_MARKERS:
            in_list = True
            continue
        if in_list:
            # the link list ends at a reference heading ("NGUỒN THAM KHẢO" after "Có thể bạn quan tâm"),
            # at real text, or at a heading followed by real text; link titles are often headings themselves
            nxt = blocks[i + 1] if i + 1 < len(blocks) else {}
            if b.get("type") == "heading":
                if not (REF_HEADING.search(t) or _is_content(nxt)):
                    continue
            elif len(t) <= 100 and not re.search(r"[.。]$", t):
                continue                                   # still inside the link list
            in_list = False
        if is_junk(t) or _is_author_tail(blocks, i):
            continue
        if site.endswith(TAG_ROW_SITES) and ZH_TAG_ROW.match(t):
            continue
        keep.append(b)
    return _cut_tail(keep)


def rebuild(blocks: list[dict]) -> tuple[str, list[dict]]:
    text, out, pos = "", [], 0
    for b in blocks:
        if out:
            text += "\n\n"
            pos += 2
        nb = dict(b, char_start=pos, char_end=pos + len(b["text"]))
        out.append(nb)
        text += b["text"]
        pos += len(b["text"])
    return text, out


def filter_doc(doc: dict) -> dict:
    blocks = dedup_blocks(list(doc.get("blocks", [])), doc.get("site") or "")
    for _ in range(3):                 # cutting a related list can expose an author line ("PV") at the new tail
        kept = filter_blocks(blocks, doc.get("site") or "")
        if len(kept) == len(blocks):
            break
        blocks = kept
    text, blocks = rebuild(blocks)
    out = dict(doc, text=text, blocks=blocks)
    if "n_chars" in doc:
        out["n_chars"] = len(text)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        (dst / f.name).write_text(json.dumps(filter_doc(doc), ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
'''
_POSTFILTER_NAMESPACE = {'__name__': '_gold_postfilter_embedded'}
exec(compile(POSTFILTER_EMBEDDED_SOURCE, 'extract/postfilter.py', 'exec'), _POSTFILTER_NAMESPACE)
filter_doc = _POSTFILTER_NAMESPACE['filter_doc']
POSTFILTER_SOURCE = 'extract/postfilter.py v1.12'
EXPECTED_POSTFILTER_SHA1 = 'b949f9851a94fe894e62cd45be8c1dce4da0ef58'
POSTFILTER_SHA1 = hashlib.sha1(POSTFILTER_EMBEDDED_SOURCE.encode('utf-8')).hexdigest()
if POSTFILTER_SHA1 != EXPECTED_POSTFILTER_SHA1:
    raise RuntimeError(f'postfilter nhúng sai SHA-1: {POSTFILTER_SHA1}; cần {EXPECTED_POSTFILTER_SHA1}. Hãy chép lại source từ main rồi chạy lại.')
print(f'Postfilter: {POSTFILTER_SOURCE}')
print(f'POSTFILTER_SHA1={POSTFILTER_SHA1}')
print(f'CONTAINER_HINTS_SHA1={CONTAINER_HINTS_SHA1} (expected {EXPECTED_CONTAINER_HINTS_SHA1})')


# %% CELL 14
# ---- Trích xuất PubMed + HTML ----
def extract_pubmed_doc(pm_rec: Dict[str, Any], doc_id: int, url: str) -> Dict[str, Any]:
    title = normalize_ws(strip_html_tags_safe(pm_rec.get('title', '') or '').rstrip('.'))
    blocks, text_parts = [], []
    char_cursor = 0
    def add_block(btype, text, level=None, role=None):
        nonlocal char_cursor
        cs, ce = char_cursor, char_cursor + len(text)
        char_cursor = ce + 2
        blocks.append({'type': btype, 'level': level, 'role': role, 'text': text, 'char_start': cs, 'char_end': ce})
        text_parts.append(text)
    if title: add_block('heading', title, level=1)
    abs_text = pm_rec.get('abstractText') or pm_rec.get('abstract') or ''
    if isinstance(abs_text, str) and abs_text:
        if '<' in abs_text:
            parts = re.split(r'<h4[^>]*>(.*?)</h4>', abs_text)
            p0 = normalize_ws(strip_html_tags_safe(parts[0]))
            if p0: add_block('paragraph', p0)
            for lbl_raw, cnt_raw in zip(parts[1::2], parts[2::2]):
                lbl = normalize_ws(strip_html_tags_safe(lbl_raw)).rstrip(':')
                cnt = normalize_ws(strip_html_tags_safe(cnt_raw))
                if cnt: add_block('paragraph', f'{lbl}: {cnt}' if lbl else cnt)
        else:
            add_block('paragraph', normalize_ws(abs_text))
    kws = pm_rec.get('keywordList', {}).get('keyword', [])
    if kws: add_block('paragraph', 'Keywords: ' + '; '.join(str(k) for k in kws) + '.')
    full_text = '\n\n'.join(text_parts)
    flags = [] if any(b['type']=='paragraph' for b in blocks) else ['no_abstract']
    return {
        'id': doc_id, 'url': url, 'final_url': url,
        'lang': 'en', 'lang_conf': 1.0,
        'title': title, 'published_date': parse_published_date(pm_rec.get('firstPublicationDate')),
        'site': 'pubmed.ncbi.nlm.nih.gov',
        'category': ((pm_rec.get('journalInfo') or {}).get('journal') or {}).get('title'),
        'text': full_text, 'blocks': blocks, 'n_chars': len(full_text),
        'sha1_text': hashlib.sha1(full_text.encode('utf-8')).hexdigest(),
        'dup_group': None, 'extractor': 'pubmed_europepmc_v3',
        'quality_flags': flags, 'schema_version': 2,
    }


# =====================================================================
# Bộ duyệt DOM: mỗi node được thăm đúng 1 lần, theo thứ tự tài liệu.
# Không dùng `seen`/id(node): wrapper tạm của selectolax tái sử dụng địa chỉ nên id() va chạm
# và làm rơi paragraph một cách ngẫu nhiên.
# =====================================================================
class _Run:
    """Dòng chữ inline đang gom (text trực tiếp + <span>/<a>/<strong>...) của một khối."""
    __slots__ = ('parts', 'link', 'link_depth', 'node')
    def __init__(self, node=None):
        self.parts: List[str] = []
        self.link = 0
        self.link_depth = 0
        self.node = node
    def add(self, s: str):
        if s:
            self.parts.append(s)
            if self.link_depth:
                self.link += len(s.strip())
    def take(self):
        text, link = ''.join(self.parts), self.link
        self.parts, self.link = [], 0
        return text, link


def _raw_stats(n) -> Tuple[int, int]:
    """(tổng ký tự, ký tự nằm trong <a>) của subtree, chỉ bỏ HARD_JUNK_TAGS."""
    total = link = 0
    stack = [(n, n.tag == 'a')]
    while stack:
        cur, inlink = stack.pop()
        c = cur.child
        while c is not None:
            t = c.tag
            if t == '-text':
                k = len(c.text(deep=False, strip=True))
                total += k
                if inlink: link += k
            elif t in HARD_JUNK_TAGS or t.startswith('-') or t == '!doctype':
                pass
            else:
                stack.append((c, inlink or t == 'a'))
            c = c.next
    return total, link


# `toc` phải bắt đầu token; tránh PhotoCMS_Caption và các class autocomplete.
TOC_REGION_RE = re.compile(r'(?<![a-z0-9])toc|table[-_\s]?of[-_\s]?contents|muc[-_\s]?luc', re.I)

def _is_toc_region(node):
    attrs = node.attributes or {}
    haystack = f"{attrs.get('class') or ''} {attrs.get('id') or ''} {attrs.get('aria-label') or ''}"
    return bool(TOC_REGION_RE.search(haystack))

REFERENCE_ATTR_RE = re.compile(r'reference|bibliograph|citation|tai-lieu-tham-khao|nguon-tham-khao', re.I)
REFERENCE_HEADING_RE = re.compile(r'^(?:tài liệu tham khảo|nguồn tham khảo|references?|bibliography|参考文献|参考资料)(?:\s*[/|–—-].*)?\s*[:：]?$', re.I)
RELATED_AUTHOR_RE = re.compile(r'(?:relate|author)', re.I)
RELATED_LINK_TAGS = CONTAINER_TAGS | {'table', 'li'}


def _subtree_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children, cur = [], node.child
        while cur is not None:
            children.append(cur)
            cur = cur.next
        stack.extend(reversed(children))


def _reference_marked(node):
    attrs = node.attributes or {}
    return bool(REFERENCE_ATTR_RE.search(
        f"{attrs.get('class') or ''} {attrs.get('id') or ''} {attrs.get('role') or ''}"))


def _has_reference_heading(node):
    for child in _subtree_nodes(node):
        if child.tag in HEADING_TAGS | {'summary'}:
            text = normalize_ws(child.text(deep=True, strip=False))
            if REFERENCE_HEADING_RE.match(text):
                return True
    return False


def _has_preceding_reference_heading(node):
    """Nhận ra khối tài liệu tham khảo đứng ngay sau heading trong DOM."""
    parent = node.parent
    if parent is None:
        return False
    previous_element = None
    child = parent.child
    while child is not None and child.mem_id != node.mem_id:
        if not child.tag.startswith('-') and child.tag != '!doctype':
            previous_element = child
        child = child.next
    return previous_element is not None and _has_reference_heading(previous_element)

def _is_reference_region(node):
    if _reference_marked(node):
        return True
    if node.tag == 'details' and _has_reference_heading(node):
        return True
    return _has_preceding_reference_heading(node)


def _related_article_card_stats(node):
    """Return count and characters covered by repeated article-card structures."""
    count = chars = 0
    for child in _subtree_nodes(node):
        if child.mem_id == node.mem_id:
            continue
        attrs = child.attributes or {}
        classes = (attrs.get('class') or '').split()
        if child.tag != 'article' and 'article' not in classes:
            continue
        total, linked = _raw_stats(child)
        if not 100 <= total <= 1800 or linked / total < 0.08:
            continue
        if child.css_first('h1,h2,h3,h4,h5,h6') is not None and child.css_first('p') is not None:
            count += 1
            chars += total
    return count, chars


def _related_article_card_count(node):
    return _related_article_card_stats(node)[0]

def _is_related_link_box(node):
    """Loại hộp link/card theo cấu trúc DOM; bảo vệ heading và tài liệu tham khảo."""
    if node.tag not in RELATED_LINK_TAGS or node.tag in HEADING_TAGS:
        return False
    if _is_reference_region(node) or _has_reference_heading(node):
        return False
    repeated_cards, repeated_chars = _related_article_card_stats(node)
    total, linked = _raw_stats(node)
    if repeated_cards >= 3 and total <= 6000 and repeated_chars >= 0.35 * total:
        return True
    if total < 80:
        return False
    attrs = node.attributes or {}
    structure = f"{attrs.get('class') or ''} {attrs.get('id') or ''}"
    summary_class_re = re.compile(r'(?<![a-z0-9])(?:sapo|summary|description|excerpt|intro|lead)(?![a-z0-9])', re.I)
    if node.tag == 'article' and total <= 1500 and linked / total >= 0.12:
        has_summary = any(
            summary_class_re.search(f"{(child.attributes or {}).get('class') or ''} {(child.attributes or {}).get('id') or ''}")
            and 30 <= len(normalize_ws(child.text(deep=True, strip=False))) <= 900
            for child in _subtree_nodes(node) if child.tag in {'div', 'span', 'p', 'small'}
        )
        has_linked_title = any(
            child.tag == 'a' and 12 <= len(normalize_ws(child.text(deep=True, strip=False))) <= 260
            for child in _subtree_nodes(node)
        )
        if has_summary and has_linked_title:
            return True

    if linked < 0.6 * total:
        return False
    has_title = has_sapo = False
    for child in _subtree_nodes(node):
        tag = child.tag
        if tag.startswith('-') or tag in HARD_JUNK_TAGS:
            continue
        text = normalize_ws(child.text(deep=True, strip=False))
        if tag in HEADING_TAGS and 8 <= len(text) <= 260:
            has_title = True
        elif tag == 'a' and 12 <= len(text) <= 260:
            has_title = True
        elif tag == 'p' and 30 <= len(text) <= 500:
            has_sapo = True
        elif tag in {'div', 'span', 'small'} and 30 <= len(text) <= 500:
            child_attrs = child.attributes or {}
            child_structure = f"{child_attrs.get('class') or ''} {child_attrs.get('id') or ''}"
            if summary_class_re.search(child_structure):
                has_sapo = True
        if has_title and has_sapo:
            return True
    return False


def _container_suspect_reasons(tree, main_node, items):
    """Flag list/news columns, requiring list structure so short/wiki pages do not trip it."""
    if main_node is None:
        return []
    headings = [normalize_ws(i.get('text', '')) for i in items if i.get('type') == 'heading']
    short_headings = [h for h in headings if 0 < len(h) <= 100]
    dates = [i.get('text', '') for i in items if re.search(
        r'\b(?:19|20)\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}|\b\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)\d{2}\b',
        i.get('text', ''))]
    paragraphs = [i.get('text', '') for i in items if i.get('type') == 'paragraph' and i.get('text')]
    long_ratio = (sum(len(t) >= 120 for t in paragraphs) / len(paragraphs)) if paragraphs else 1.0
    total, linked = _raw_stats(main_node)
    link_ratio = linked / total if total else 0.0
    repeated_cards = _related_article_card_count(main_node)

    reasons = []
    if len(short_headings) >= 6 and len(dates) >= 3:
        reasons.append('many_short_headings_with_dates')
    if len(short_headings) >= 4 and len(items) >= 12 and link_ratio >= 0.30:
        reasons.append('link_heavy_heading_list')
    if repeated_cards >= 3:
        reasons.append(f'repeated_article_cards:{repeated_cards}')
    if len(paragraphs) >= 6 and long_ratio <= 0.20:
        reasons.append('few_long_paragraphs')

    extracted_len = sum(len(i.get('text', '')) for i in items)
    og_description_node = tree.css_first('meta[property="og:description"]') or tree.css_first('meta[name="description"]')
    og_description = normalize_ws(og_description_node.attributes.get('content', '')) if og_description_node else ''
    paragraph_text = normalize_ws(' '.join(p.text(deep=True, strip=False) for p in tree.css('p')))
    if extracted_len < 500 and (len(og_description) >= 700 or (len(paragraph_text) >= 1500 and len(paragraph_text) >= max(1500, extracted_len * 4))):
        reasons.append('too_short_vs_page_text')

    frame_text = normalize_ws(main_node.text(deep=True, strip=False)).casefold()
    h1 = tree.css_first('h1')
    page_title = normalize_ws(h1.text(deep=True, strip=False)) if h1 is not None else ''
    if not page_title:
        og = tree.css_first('meta[property="og:title"]')
        page_title = normalize_ws(og.attributes.get('content', '')) if og else ''
    if page_title and page_title.casefold() not in frame_text:
        reasons.append('page_title_outside_container')

    list_reasons = {'many_short_headings_with_dates', 'link_heavy_heading_list'}
    has_list_shape = bool(list_reasons.intersection(reasons)) or repeated_cards >= 3
    corroborated = repeated_cards >= 3 or len(list_reasons.intersection(reasons)) >= 2 or any(
        reason in reasons for reason in ('few_long_paragraphs', 'page_title_outside_container'))
    return reasons if 'too_short_vs_page_text' in reasons or (has_list_shape and corroborated) else []

QA_A_RE = re.compile(r'answer|reply|doctor|anscont|sele', re.I)
QA_Q_RE = re.compile(r'ask|question', re.I)

def _qa_type(node, max_up: int = 4):
    """qa_q / qa_a from the nearest element (leaf or ancestor) with a role in class/id."""
    cur = node
    for _ in range(max_up + 1):
        if cur is None or cur.tag in ('body', 'html'):
            break
        cid = f"{cur.attributes.get('class') or ''} {cur.attributes.get('id') or ''}"
        if QA_A_RE.search(cid):
            return 'qa_a'
        if QA_Q_RE.search(cid):
            return 'qa_q'
        cur = cur.parent
    return None

class _Extractor:
    def __init__(self, is_qa: bool = False, dom_clean: str = ''):
        self.items: List[Dict[str, Any]] = []
        self.is_qa = is_qa
        self.dom_clean = dom_clean
        self.stopped = False
        self.tail = False
        self.n_tail = 0
        self.reference_depth = 0
        self.toc_depth = 0

    # ---------- phân loại rác ----------
    def _junk(self, n) -> Optional[str]:
        """None = giữ; 'skip' = bỏ subtree; 'stop' = (chỉ ở phần nối tiếp) hết bài."""
        tag = n.tag
        if self.dom_clean == 'jbk.39.net' and tag == 'aside':
            return 'skip'
        if tag in HARD_JUNK_TAGS:
            return 'skip'
        attrs = n.attributes
        cls = idv = role = ''
        if attrs:
            if 'hidden' in attrs or attrs.get('aria-hidden') == 'true':
                return 'skip'
            style = attrs.get('style')
            if style and _HIDDEN_STYLE_RE.search(style):
                return 'skip'
            cls = (attrs.get('class') or '').lower()
            idv = (attrs.get('id') or '').lower()
            role = (attrs.get('role') or '').lower()
        is_reference = self.reference_depth > 0 or _is_reference_region(n)
        is_toc = self.toc_depth > 0 or _is_toc_region(n)
        if is_toc:
            return None
        fuzzy_idv = re.sub(r'(?:\.[0-9a-f]{2})+', ' ', idv)
        if (not is_reference and tag not in {'a', 'span', 'em', 'strong', 'b', 'i', 'u', 'small', 'font', 'abbr', 'cite', 'mark', 'sup', 'sub', 'label'}
                and RELATED_AUTHOR_RE.search(f'{cls} {idv}')):
            if self.tail and re.search(r'relate', f'{cls} {idv}', re.I):
                return 'stop'
            return 'skip'
        if not is_reference and not self.tail and _is_related_link_box(n):
            return 'skip'
        kind = None
        if tag in STRUCT_JUNK_TAGS or role in JUNK_ROLES:
            kind = 'struct'
        elif (cls and not EXACT_JUNK_CLASSES.isdisjoint(cls.split())) or idv in EXACT_JUNK_IDS:
            kind = 'exact'
        elif (cls or idv) and FUZZY_JUNK_RE.search(f'{cls} {fuzzy_idv}'):
            kind = 'fuzzy'
        if kind is None:
            return None
        if self.tail:
            if tag == 'footer' or STOP_CLASS_RE.search(f'{cls} {idv}'):
                return 'stop'
            return 'skip'
        if kind == 'fuzzy' and not is_reference:
            # Khớp mờ theo tên class: nếu subtree toàn chữ bài (ít link) thì đó là nội dung, không phải rác.
            total, link = _raw_stats(n)
            if total - link >= 200 and link <= 0.3 * total:
                return None
        return 'skip'

    # ---------- phát block ----------
    def emit(self, btype, text, level=None, role=None, link=0):
        text = normalize_ws(text)
        if not text:
            return
        if role is None and self.toc_depth:
            role = 'toc'
        if role is None and self.reference_depth:
            role = 'reference'
        if self.tail:
            if STOP_TEXT_RE.match(text) and (len(text) <= 120 or btype == 'heading'):
                self.stopped = True
                return
            # Heading phải giữ nguyên kể cả khi tiêu đề chứa liên kết; mật độ link chỉ cắt phần tail không phải heading.
            if btype != 'heading' and len(text) >= 3 and link >= 0.6 * len(text):
                self.stopped = True
                return
            if BOILERPLATE_FULL_RE.match(text):
                return
        self.items.append({'type': btype, 'level': level, 'role': role, 'text': text,
                           'link': link, 'tail': self.tail})
        if self.tail:
            self.n_tail += 1
            if self.n_tail >= CONTINUE_MAX_BLOCKS:
                self.stopped = True

    def flush(self, run: _Run, kind, level):
        if not run.parts:
            return
        text, link = run.take()
        btype = kind or 'paragraph'
        if self.is_qa and btype == 'paragraph':
            btype = _qa_type(run.node) or btype
        role = guess_role(run.node, text) if run.node is not None else None
        self.emit(btype, text, level if btype == 'list_item' else None, role, link)

    # ---------- gom text của 1 khối lá (p, h1, td...) ----------
    def collect(self, n, run: _Run):
        c = n.child
        while c is not None:
            t = c.tag
            if t == '-text':
                run.add(c.text(deep=False, strip=False))
            elif t.startswith('-') or t == '!doctype':
                pass
            elif self._junk(c):
                pass
            elif t == 'br':
                run.add(' ')       # lexbor nối chữ hai bên <br> không có khoảng trắng
            else:
                spaced = t in SPACE_TAGS
                if spaced: run.add(' ')
                if t == 'a': run.link_depth += 1
                self.collect(c, run)
                if t == 'a': run.link_depth -= 1
                if spaced: run.add(' ')
            c = c.next

    def _leaf(self, n):
        run = _Run(n)
        self.collect(n, run)
        return run.take()

    # ---------- duyệt ----------
    def children(self, n, run, kind, level):
        c = n.child
        while c is not None and not self.stopped:
            self.visit(c, run, kind, level)
            c = c.next

    def visit(self, n, run: _Run, kind, level):
        if self.stopped:
            return
        t = n.tag
        if t == '-text':
            run.add(n.text(deep=False, strip=False))
            return
        if t.startswith('-') or t == '!doctype':
            return
        j = self._junk(n)
        if j == 'stop':
            self.stopped = True
            return
        if j:
            return
        if t == 'div' and self.dom_clean == 'test.pmphai.com' and 'gu_det_left_con_h1' in (n.attributes.get('class') or '').split():
            self.flush(run, kind, level)
            text, link = self._leaf(n)
            self.emit('heading', text, level=2, role=guess_role(n, text), link=link)
        elif t in HEADING_TAGS:
            self.flush(run, kind, level)
            text, link = self._leaf(n)
            self.emit('heading', text, level=int(t[1]), role=guess_role(n, text), link=link)
        elif t == 'li':
            self.flush(run, kind, level)
            depth, cur = 0, n.parent
            while cur is not None:
                if cur.tag in ('ul', 'ol'): depth += 1
                cur = cur.parent
            depth = max(depth, 1)
            sub = _Run(n)
            self.children(n, sub, 'list_item', depth)
            self.flush(sub, 'list_item', depth)
        elif t in LEAF_TEXT_TAGS:
            self.flush(run, kind, level)
            text, link = self._leaf(n)
            btype = kind or 'paragraph'
            role = guess_role(n, text)
            if self.is_qa and btype == 'paragraph':
                btype = _qa_type(n) or btype
            self.emit(btype, text, level if btype == 'list_item' else None, role, link)
        elif t == 'table':
            self.flush(run, kind, level)
            self._table(n, kind, level)
        elif t in ('br', 'hr'):
            self.flush(run, kind, level)
        elif t in CONTAINER_TAGS:
            self.flush(run, kind, level)
            is_reference_context = _is_reference_region(n)
            is_toc_context = _is_toc_region(n)
            if is_reference_context:
                self.reference_depth += 1
            if is_toc_context:
                self.toc_depth += 1
            try:
                sub = _Run(n)
                self.children(n, sub, kind, level)
                self.flush(sub, kind, level)
            finally:
                if is_toc_context:
                    self.toc_depth -= 1
                if is_reference_context:
                    self.reference_depth -= 1
        else:                                   # inline hoặc tag lạ: gộp vào run hiện tại
            if t == 'a': run.link_depth += 1
            self.children(n, run, kind, level)
            if t == 'a': run.link_depth -= 1

    def _table(self, n, kind, level):
        rows, captions = [], []
        def gather(x):
            c = x.child
            while c is not None:
                if c.tag == 'tr': rows.append(c)
                elif c.tag in ('thead', 'tbody', 'tfoot'): gather(c)
                elif c.tag == 'caption': captions.append(c)
                c = c.next
        gather(n)
        for cap in captions:
            self.visit(cap, _Run(n), kind, level)
        grid = []
        for tr in rows:
            cells, c = [], tr.child
            while c is not None:
                if c.tag in ('td', 'th'): cells.append(c)
                c = c.next
            grid.append(cells)
        # Bảng "bố cục" (mỗi hàng ≤1 ô có chữ: ảnh+chú thích, khung chia sẻ, cả bài nằm trong 1 ô):
        # đi vào từng ô như container để giữ cấu trúc đoạn thay vì dồn thành một dòng.
        layout = all(sum(1 for c in cells if _raw_stats(c)[0] > 0) <= 1 for cells in grid)
        for cells in grid:
            if self.stopped: return
            if layout:
                for cell in cells:
                    sub = _Run(cell)
                    self.children(cell, sub, kind, level)
                    self.flush(sub, kind, level)
            else:
                texts = [normalize_ws(self._leaf(c)[0]) for c in cells]
                if any(texts):
                    self.emit('table', ' '.join(texts))

    def walk_region(self, c):
        """Duyệt nội dung bên trong một container (không kiểm tra rác của chính container)."""
        is_reference_context = _is_reference_region(c)
        if is_reference_context:
            self.reference_depth += 1
        try:
            if c.tag == 'table':
                self._table(c, None, None)
                return
            run = _Run(c)
            self.children(c, run, None, None)
            self.flush(run, None, None)
        finally:
            if is_reference_context:
                self.reference_depth -= 1

    def continue_after(self, anchor, max_up: int = CONTINUE_MAX_ANCESTOR_LEVELS):
        """Đi tiếp theo thứ tự tài liệu sau container (anh em, rồi anh em của cha...) cho tới
        ranh giới 'hết bài'. Cứu phần bài bị HTML lỗi (thừa </div>...) đẩy ra ngoài container."""
        self.tail = True
        cur, up = anchor, 0
        while cur is not None and up <= max_up and not self.stopped:
            parent = cur.parent
            if parent is None or parent.tag in ('body', 'html', '-document'):
                break
            run = _Run(parent)
            s = cur.next
            while s is not None and not self.stopped:
                self.visit(s, run, None, None)
                s = s.next
            self.flush(run, None, None)
            cur, up = parent, up + 1


# ---------- chọn container ----------
def _doc_order(tree, nodes):
    if len(nodes) < 2:
        return nodes
    want = {n.mem_id: n for n in nodes}
    out = []
    for x in tree.root.traverse():
        if x.mem_id in want:
            out.append(want[x.mem_id])
            if len(out) == len(want): break
    return out

def _dedupe_nested(nodes):
    ids = {n.mem_id for n in nodes}
    keep, seen = [], set()
    for n in nodes:
        if n.mem_id in seen: continue
        seen.add(n.mem_id)
        p, nested = n.parent, False
        while p is not None:
            if p.mem_id in ids: nested = True; break
            p = p.parent
        if not nested: keep.append(n)
    return keep

def _select_containers(tree, hint):
    nodes, main_nodes = [], []
    if hint:
        for sel in hint.get('also', []): nodes.extend(safe_css(tree, sel))
        if hint.get('main'):
            main_nodes = safe_css(tree, hint['main'])
            nodes.extend(main_nodes)
    if not nodes:
        for sel in GENERIC_CONTAINER_SELECTORS:
            n = safe_css_first(tree, sel)
            if n is not None:
                nodes, main_nodes = [n], [n]
                break
    if not nodes:
        return [tree.body or tree.root], None
    nodes = _doc_order(tree, _dedupe_nested(nodes))
    main = max(main_nodes, key=lambda x: _raw_stats(x)[0]) if main_nodes else None
    return nodes, main


# ---------- hoàn thiện block: lọc dòng metadata, tính offset (không khử lặp) ----------
def _finalize(items, title, dom_clean):
    kept, skip_contact = [], False
    for idx, it in enumerate(items):
        text = it['text']
        if skip_contact:
            if CONTACT_FOLLOW_RE.match(text) and len(text) <= 120:
                continue
            skip_contact = False
        if BOILERPLATE_FULL_RE.match(text):
            continue
        if CONTACT_START_RE.match(text):
            skip_contact = True
            continue
        if LABEL_LINE_RE.match(text) and len(text) <= 250 and (len(text) <= 40 or it['link'] >= 0.4 * len(text)):
            continue
        if len(kept) < 4 and DATESTAMP_RE.match(text):      # ngày đã có ở metadata
            continue
        kept.append(dict(it))
    if dom_clean == 'baodanang.vn':
        for it in kept:
            if it['type'] != 'heading':
                it['text'] = BAODANANG_PREFIX_RE.sub('', it['text'], count=1).strip()
                break
        kept = [i for i in kept if i['text']]
    blocks, parts, cursor = [], [], 0
    if title:
        blocks.append({'type': 'heading', 'level': 1, 'role': None, 'text': title,
                       'char_start': 0, 'char_end': len(title)})
        parts.append(title)
        cursor = len(title) + 2
    n_tail = 0
    for it in kept:
        s, e = cursor, cursor + len(it['text'])
        cursor = e + 2
        blocks.append({'type': it['type'], 'level': it['level'], 'role': it['role'],
                       'text': it['text'], 'char_start': s, 'char_end': e})
        parts.append(it['text'])
        n_tail += 1 if it['tail'] else 0
    return blocks, '\n\n'.join(parts), n_tail


def _extract_title(tree, dom, dom_clean):
    title = None
    t_hint = TITLE_HINTS.get(dom) or TITLE_HINTS.get(dom_clean)
    if t_hint:
        try:
            m = re.match(r'\s*(\S+)\s+in\s+([^\s,]+)', t_hint)
            el, sel = (m.group(1), m.group(2)) if m else (None, t_hint)
            tn = None
            if el and el.lower() in HEADING_TAGS and el.lower() != sel.split('.')[0].split('#')[0].lower():
                tn = safe_css_first(tree, f'{sel} {el}')     # "h1 in div.detail-title" → lấy h1 bên trong
            tn = tn or safe_css_first(tree, sel)
            if tn is not None:
                title = normalize_ws(tn.text(deep=True, strip=False))
        except Exception:
            pass
    if not title:
        tn = tree.css_first('h1') or tree.css_first('meta[property="og:title"]') or tree.css_first('title')
        if tn:
            title = normalize_ws(tn.attributes.get('content') if tn.tag == 'meta' else tn.text(deep=True, strip=False))
    return title or None


def extract_html_doc(html_bytes: bytes, encoding: str, url: str, doc_id: int) -> Dict[str, Any]:
    dom = urlparse(url).netloc.lower()
    dom_clean = re.sub(r'^www\.', '', dom).split(':')[0]
    domain_expected_lang = DOMAIN_LANG_MAP.get(dom) or DOMAIN_LANG_MAP.get(dom_clean)
    _encoding = (encoding or '').strip()
    try:
        html_str = html_bytes.decode(_encoding or 'utf-8', errors='replace')
    except (LookupError, UnicodeDecodeError):
        html_str = html_bytes.decode('utf-8', errors='replace')
        _encoding = 'utf-8'
    if not _encoding or '\ufffd' in html_str:
        _detected = detect_encoding(html_bytes)
        if _detected:
            try:
                _reddecoded = html_bytes.decode(_detected, errors='replace')
                if not _encoding or _reddecoded.count('\ufffd') < html_str.count('\ufffd'):
                    html_str, _encoding = _reddecoded, _detected
            except (LookupError, UnicodeDecodeError):
                pass
    if '\ufffd' in html_str:
        logger.warning(f'ID {doc_id} ({dom_clean}): còn ký tự thay thế sau khi dò lại encoding {_encoding!r}')
    tree = HTMLParser(html_str)

    title = _extract_title(tree, dom, dom_clean)

    # ngày
    pub_date = None
    for sel in ['meta[property="article:published_time"]','meta[name="pubdate"]','meta[name="publishdate"]','meta[itemprop="datePublished"]','meta[name="date"]']:
        m = safe_css_first(tree, sel)
        if m and m.attributes.get('content'):
            pub_date = parse_published_date(m.attributes['content'])
            if pub_date: break
    if not pub_date:
        def _find(v):
            if isinstance(v, dict):
                if v.get('datePublished'): return v['datePublished']
                for c in v.values():
                    r = _find(c)
                    if r: return r
            elif isinstance(v, list):
                for c in v:
                    r = _find(c)
                    if r: return r
            return None
        for sc in safe_css(tree, 'script[type="application/ld+json"]'):
            try:
                pub_date = parse_published_date(_find(json.loads(sc.text(deep=True, strip=True))))
                if pub_date: break
            except (ValueError, TypeError): pass
    if not pub_date:
        for sel in ['time[datetime]','time','[itemprop="datePublished"]']:
            n = safe_css_first(tree, sel)
            if n:
                pub_date = parse_published_date(n.attributes.get('datetime') or n.attributes.get('content') or n.text(deep=True, strip=True))
                if pub_date: break

    # category từ breadcrumb
    category = None
    bc = tree.css_first('.breadcrumb, .breadcrumbs, .breadcrumb-item')
    if bc: category = normalize_ws(bc.text(deep=True, strip=False))

    # ---- nội dung ----
    hint = CONTAINER_HINTS.get(dom) or CONTAINER_HINTS.get(dom_clean)
    # Fallback for an uncovered pilot domain; prefer the site's article-body wrapper.
    if not hint and dom_clean == 'jb39.com':
        hint = {'main': 'div.post-body', 'also': []}
        logger.info(f'ID {doc_id} (jb39.com): dùng fallback selector div.post-body vì chưa có hint cấu hình.')
    is_qa = any(k in dom for k in ['120ask', 'ask.39.net'])
    containers, main_node = _select_containers(tree, hint)
    # YouMed đôi khi đặt nguồn tham khảo trong <details> ngoài khung bài chính.
    if dom_clean == 'youmed.vn':
        selected_ids = {node.mem_id for root in containers for node in _subtree_nodes(root)}
        for details_node in safe_css(tree, 'details'):
            if _is_reference_region(details_node) and details_node.mem_id not in selected_ids:
                containers.append(details_node)
                selected_ids.update(node.mem_id for node in _subtree_nodes(details_node))

    ex = _Extractor(is_qa, dom_clean)
    for c in containers:
        ex.walk_region(c)
    container_suspect_reasons = _container_suspect_reasons(tree, main_node, ex.items)
    if container_suspect_reasons:
        logger.warning(f'ID {doc_id} ({dom_clean}): container có dấu hiệu chọn sai: {container_suspect_reasons}')
    if CONTINUE_AFTER_CONTAINER and main_node is not None and dom_clean not in NO_CONTINUE_DOMAINS:
        ex.continue_after(main_node)
    blocks, full_text, n_tail = _finalize(ex.items, title, dom_clean)
    _block_text_len = len(full_text)

    container_text_fallback = False
    if len(full_text) < 200:
        # Quá ngắn: thử các vùng bài viết ngữ nghĩa khác (vẫn dùng bộ duyệt có lọc rác, giữ cấu trúc đoạn).
        sels = ([hint['main']] if hint and hint.get('main') else []) + FALLBACK_CONTAINER_SELECTORS
        best = None
        for sel in dict.fromkeys(sels):
            cands = safe_css(tree, sel)
            if not cands: continue
            ex2 = _Extractor(is_qa, dom_clean)
            for c in _dedupe_nested(cands): ex2.walk_region(c)
            b2, t2, _ = _finalize(ex2.items, title, dom_clean)
            if best is None or len(t2) > len(best[1]):
                best = (b2, t2, sel)
            if len(t2) >= 200:
                best = (b2, t2, sel)
                break
        if best and len(best[1]) > len(full_text):
            blocks, full_text, n_tail = best[0], best[1], 0
            container_text_fallback = True
            logger.warning(f'ID {doc_id} ({dom_clean}): lấy toàn văn từ container {best[2]!r} '
                           f'({len(full_text):,} chars); container chính chỉ thu được {_block_text_len:,} chars.')
        elif len(full_text) < 200:
            logger.warning(f'ID {doc_id} ({dom_clean}) vẫn ngắn: {_block_text_len:,} chars; '
                           f'hint={(hint or {}).get("main")!r}. Kiểm tra HTML thô trong shard nguồn.')

    # ---- cờ chất lượng ----
    flags = []
    if container_text_fallback: flags.append('container_text_fallback')
    if container_suspect_reasons: flags.append('container_suspect')
    if n_tail: flags.append('tail_recovered')          # có đoạn nằm NGOÀI container được nối lại
    if len(full_text) < 200: flags.append('short')
    if is_soft_404(full_text): flags.append('soft_404')
    if (INTEGRITY_AUDIT and main_node is not None and not container_text_fallback
            and main_node.parent is not None and main_node.parent.tag not in ('body', 'html', '-document')):
        # Kiểm tra toàn vẹn: cha của container (đã lọc rác) dài hơn hẳn kết quả → nghi cụt bài, cần xem tay.
        exp = _Extractor(is_qa, dom_clean)
        exp.walk_region(main_node.parent)
        _, par_text, _ = _finalize(exp.items, title, dom_clean)
        if len(par_text) > 1.5 * len(full_text) and len(par_text) - len(full_text) > 400:
            flags.append('possible_truncation')

    if 'soft_404' in flags:
        full_text, blocks = '', []

    cjk_count = len(CJK_RE.findall(full_text[:1000]))
    if cjk_count > 10:
        lang, lang_conf = 'zh', 0.99
    elif domain_expected_lang:
        lang, lang_conf = domain_expected_lang, 0.95
    elif any(dom.endswith(s) for s in ('.cn', '.com.cn')):
        lang, lang_conf = 'zh', 0.95
    elif 'pubmed' in dom:
        lang, lang_conf = 'en', 1.0
    else:
        lang, lang_conf = 'vi', 0.95

    return {
        'id': doc_id, 'url': url, 'final_url': url,
        'lang': lang, 'lang_conf': lang_conf,
        'title': title, 'published_date': pub_date,
        'site': dom, 'category': category,
        'text': full_text, 'blocks': blocks, 'n_chars': len(full_text),
        'sha1_text': hashlib.sha1(full_text.encode('utf-8')).hexdigest(),
        'dup_group': None, 'extractor': 'custom_cleaner_v3',
        'quality_flags': flags, 'schema_version': 2,
    }


# %% CELL 17
# ---- Kiểm thử nhanh cleaner (offline, không cần ZIP) ----
import logging
# Mô phỏng trang baoangiang: bài đủ 10 đoạn + ảnh chú thích + nguồn; có rác (chia sẻ, "A A", từ khóa,
# tin liên quan, liên hệ tòa soạn, footer). Ca 'stray_div' có thừa </div> làm container đóng sớm.
def _selftest_cleaner():
    # Regression mục 7: toc đúng phải match; PhotoCMS/autocomplete không được coi là mục lục.
    for _class_name in ('ez-toc-container', 'toc', 'toctitle', 'toclevel-1'):
        _node = HTMLParser(f'<div class="{_class_name}"></div>').css_first('div')
        assert _is_toc_region(_node), f'không nhận ra TOC class {_class_name}'
    for _class_name in ('PhotoCMS_Caption', 'easy-autocomplete', 'search-autocomplete', 'skipToContentContainer'):
        _node = HTMLParser(f'<div class="{_class_name}"></div>').css_first('div')
        assert not _is_toc_region(_node), f'bắt nhầm TOC class {_class_name}'
    _reference_test = HTMLParser('<section><div class="ref-heading"><h2>NGUỒN THAM KHẢO</h2></div>'
                                '<div class="reference-links"><a>Resource link</a></div></section>')
    _reference_links = _reference_test.css_first('div.reference-links')
    assert _is_reference_region(_reference_links), 'không nhận vùng tham khảo có heading anh em đứng trước'
    assert not _is_related_link_box(_reference_links), 'gán nhầm vùng tham khảo thành hộp liên quan'
    print('OK TOC boundary + heading tham khảo đứng trước link box')
    TITLE = 'Bộ Y tế phân bổ 2 triệu liều vắc xin Moderna, đề nghị không tiêm trộn'
    LEAD = 'Bộ Y tế tiếp tục phân bổ hơn 2 triệu liều vắc xin Moderna cho các tỉnh và lực lượng quân đội, trong đó Hà Nội và TP.HCM nhận nhiều nhất.'
    PARAS = ['Bộ Y tế phân bổ 2.000.040 liều vắc xin Moderna cho 53 tỉnh, thành phố, lực lượng quân đội, công an và 22 bệnh viện, viện, trường đại học. Đây là đợt phân bổ vắc xin thứ 11.', 'Tại khu vực miền Bắc, Bộ Y tế phân bổ 870.240 liều vắc xin cho 28 tỉnh, thành phố: Hà Nội, Thanh Hóa, Nghệ An, Hải Dương, Thái Bình, Hải Phòng, Bắc Giang và Bắc Kạn.', 'Trong đó, Hà Nội được phân bổ nhiều nhất với 120.960 liều, tiếp đến là tỉnh Hải Dương được phân bổ 43.680 liều; Nghệ An, Thanh Hoá mỗi tỉnh nhận gần 37.000 liều.', 'Các tỉnh Vĩnh Phúc, Sơn La, Hưng Yên, Thái Nguyên, Hà Tĩnh sẽ nhận từ 28.000 – 31.000 liều. Bắc Kạn là địa phương nhận ít nhất với 6.720 liều.', 'Tại miền Trung, 11 tỉnh được phân bổ 309.000 liều, trong đó Khánh Hoà nhiều nhất vời 42.000 liều; Quảng Ngãi nhận gần 32.000 liều.', '4 tỉnh Tây Nguyên: Kon Tum, Đắk Nông, Gia Lai và Đắk Lắk được phân bổ 80.640 liều.', '10 tỉnh, thành phố miền Nam được phân bổ 505.680 liều, trong đó TP.HCM được phân bổ nhiều nhất với 235.200 liều.', 'Ngoài ra, Bộ Y tế phân bổ 42.000 liều cho lực lượng quân đội và 33.600 liều cho lực lượng công an.', 'Bộ Y tế nói rõ, vắc xin Moderna sau khi xuất kho lạnh phải được bảo quản ở nhiệt độ 2-8 độ C và phải sử dụng hết tối đa trong 31 ngày.', 'Đặc biệt, Bộ Y tế lưu ý, không tiêm trộn vắc xin Moderna với các vắc xin khác. Các đơn vị phải xây dựng kế hoạch, đảm bảo mỗi đối tượng tiêm chủng được tiêm đủ 2 mũi vắc xin Moderna.']
    CAPTION = 'Moderna là một trong 2 loại vắc xin sử dụng công nghệ mRNA có hiệu quả bảo vệ cao nhất hiện nay'
    SRC = 'Theo THÚY HẠNH (Vietnamnet)'
    SHARE = ('<table class="share"><tr><td><a href="#"><img src="g.png"></a></td><td>'
             '<iframe src="https://www.facebook.com/plugins/share_button.php"></iframe></td></tr></table>'
             '<div class="fontsize">A A</div>')
    CONTACT = ('<p>Mọi phản ánh, ý kiến, tin, bài và hình ảnh cộng tác của độc giả có thể gửi đến Báo An Giang '
               'theo địa chỉ:</p><p>Email: baoangiangonline@gmail.com</p><p>Hotline: 02963.89.8888</p>')
    head = (f'<!doctype html><html><head><meta charset="utf-8"><title>{TITLE} - Báo An Giang Online</title>'
            f'<script>var x = "<p>junk in script</p>";</script></head><body>'
            '<header><ul><li><a href="/">Trang chủ</a></li></ul></header>'
            '<div class="breadcrumb"><a>Trang chủ</a> <a>Sức khỏe</a></div>'
            f'<div class="detail-title"><h1>{TITLE}</h1></div><div class="date">14/07/2021 - 14:24</div>'
            '<div class="detail-content">')
    tail = ('<div class="tags"><b>Từ khóa</b> <a>Bộ Y tế</a><a>Moderna</a></div>'
            '<div class="related"><h3>Bạn có thể quan tâm</h3><ul><li><a>Quán bún mắm</a></li></ul></div></div>'
            '<div class="read-more"><h2><a>Đọc nhiều</a></h2></div>'
            '<footer><p>© 2008 - 2017 Bản quyền thuộc về Báo và phát thanh, truyền hình An Giang.</p></footer></body></html>')
    ps = lambda xs: ''.join(f'<p>{x}</p>' for x in xs)
    cap = f'<p style="text-align:center"><em>{CAPTION}</em></p>'
    pages = {
        'well_formed': head + f'<div class="content-fck-font-size"><p>{LEAD}</p>{SHARE}' + ps(PARAS[:4]) + cap
                       + ps(PARAS[4:]) + f'<p><strong>{SRC}</strong></p>' + CONTACT + '</div>' + tail,
        'stray_div':   head + f'<div class="content-fck-font-size"><p>{LEAD}</p>{SHARE}' + ps(PARAS[:2]) + '</div>'
                       + ps(PARAS[2:4]) + cap + ps(PARAS[4:]) + f'<p><strong>{SRC}</strong></p>' + CONTACT + tail,
    }
    must = [LEAD] + PARAS[:4] + [CAPTION] + PARAS[4:] + [SRC]      # đúng thứ tự trong bài
    junk = ['Mọi phản ánh', 'Email:', 'Hotline', 'Từ khóa', 'Bạn có thể quan tâm', 'Đọc nhiều', '©',
            'A A', 'junk in script', 'Quán bún']
    for name, html in pages.items():
        d = extract_html_doc(html.encode('utf-8'), 'utf-8', 'https://baoangiang.com.vn/x-a1.html', 0)
        miss = [t[:50] for t in must if t not in d['text']]
        dirty = [j for j in junk if j in d['text']]
        order = [d['text'].find(t) for t in must]
        assert not miss, f'{name}: thiếu {miss}'
        assert not dirty, f'{name}: còn rác {dirty}'
        assert order == sorted(order), f'{name}: sai thứ tự'
        assert d['text'].count(TITLE) == 1, f'{name}: tiêu đề lặp'
        assert all(d['text'][b['char_start']:b['char_end']] == b['text'] for b in d['blocks']), f'{name}: lệch offset'
        print(f"OK {name:12s} chars={d['n_chars']:,} blocks={len(d['blocks'])} flags={d['quality_flags']}")
    # inline + <br>: không dính chữ, không mất <strong>, giữ thứ tự
    h = ('<html><body><div class="detail-content"><div class="content-fck-font-size">Hà Nội<br>TP.HCM '
         '<strong>đậm</strong> thường<p>sau</p><strong>cuối</strong></div></div></body></html>')
    _lvl = logger.level; logger.setLevel(logging.ERROR)      # trang mẫu cố ý ngắn, tránh cảnh báo 'short'
    try: d = extract_html_doc(h.encode(), 'utf-8', 'https://baoangiang.com.vn/y-a2.html', 0)
    finally: logger.setLevel(_lvl)
    assert [b['text'] for b in d['blocks'] if b['type'] != 'heading'] == ['Hà Nội', 'TP.HCM đậm thường', 'sau', 'cuối'], d['blocks']
    print('OK inline_br     ', [b['text'] for b in d['blocks']])

if 'extract_html_doc' not in globals():
    raise RuntimeError(
        'Chưa có extract_html_doc trong kernel. Hãy chạy cell Trích xuất PubMed + HTML phía trên, '
        'sau đó chạy lại cell tự kiểm tra cleaner.'
    )

_selftest_cleaner()


# %% CELL 19
# ---- Trích xuất + so sánh Gold 1, Gold 2 và Gold 3 ----
# Chạy sau cells 2–12; không mở/ghi crawl_status và không khởi chạy crawler.
from zipfile import ZipFile
from collections import Counter, defaultdict
import subprocess
import sys
import time
import shutil
import xml.etree.ElementTree as ET

def _reset_latest_dir(path):
    root = Path(REPORT_DIR).resolve()
    target = Path(path).resolve()
    if target.parent != root:
        raise ValueError(f'Refusing to overwrite outside REPORT_DIR: {target}')
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

GOLD_NOTEBOOK_BUILD = '2026-10-20-v26-shard-html-selection-postfilter-v1.12'
print(f"GOLD_NOTEBOOK_BUILD={GOLD_NOTEBOOK_BUILD} | POSTFILTER_SOURCE={POSTFILTER_SOURCE}")
logger.info(f'GOLD_COMPARE build={GOLD_NOTEBOOK_BUILD} postfilter_sha1={POSTFILTER_SHA1} hints_sha1={CONTAINER_HINTS_SHA1}')
flush_log_handlers(sync=True)
print(f"POSTFILTER_SHA1={POSTFILTER_SHA1}")
print(f"CONTAINER_HINTS_SHA1={CONTAINER_HINTS_SHA1} | expected={EXPECTED_CONTAINER_HINTS_SHA1}")
CLEANER_POSTFILTER_NOTEBOOK_SHA1 = hashlib.sha1(extract_html_doc.__code__.co_code + _Extractor._junk.__code__.co_code + POSTFILTER_EMBEDDED_SOURCE.encode('utf-8')).hexdigest()
print(f"CLEANER_POSTFILTER_NOTEBOOK_SHA1={CLEANER_POSTFILTER_NOTEBOOK_SHA1}")
HINTS_SHA1 = CONTAINER_HINTS_SHA1
print(f"HINTS_SHA1={HINTS_SHA1} | raw_sha1={CONTAINER_HINTS_RAW_SHA1} | expected={EXPECTED_CONTAINER_HINTS_SHA1} | DOMAIN_PLAN_SHA1={hashlib.sha1((CONFIG_ROOT / 'domain_plan.csv').read_bytes()).hexdigest()}")


def _find_gold_archive(archive_name: str) -> Path:
    candidates = [
        Path.cwd() / archive_name,
        Path('/content') / archive_name,
        Path.home() / 'Downloads' / archive_name,
        Path(DATA_ROOT) / archive_name,
        Path('/content/drive/MyDrive/r2ai_data') / archive_name,
        Path('/content/drive/MyDrive') / archive_name,
    ]
    found = next((candidate for candidate in candidates if candidate.is_file()), None)
    if found is None:
        raise FileNotFoundError(f'Không tìm thấy {archive_name}; đặt file cạnh notebook, trong Downloads hoặc DATA_ROOT.')
    return found

def run_gold_archive(gold_set_name: str, archive_path: Path) -> Dict[str, Any]:
    global GOLD_HTML_ZIP, GOLD_OUTPUT_ROOT, _gold_engine_dirs, _gold_manifest
    global _gold_timings, _gold_errors, _gold_dropped_blocks, _html_rows, _gold_pm_rows
    global _score_out, _score_success, _pages_csv, _domain_target_path, _domain_targets
    GOLD_HTML_ZIP = Path(archive_path)
    logger.info(f'GOLD_SET_START set={gold_set_name} archive={GOLD_HTML_ZIP} build={GOLD_NOTEBOOK_BUILD} postfilter_sha1={POSTFILTER_SHA1} hints_sha1={CONTAINER_HINTS_SHA1}')
    flush_log_handlers(sync=True)
    if not GOLD_HTML_ZIP.is_file():
        raise FileNotFoundError(GOLD_HTML_ZIP)
    try:
        import trafilatura as _trafilatura
    except ImportError:
        _trafilatura = None
        logger.warning("Thiếu trafilatura; bỏ qua extractor baseline này.")
    try:
        from resiliparse.extract.html2text import extract_plain_text as _resiliparse_text
    except ImportError:
        _resiliparse_text = None
        logger.warning("Thiếu resiliparse; bỏ qua extractor baseline này.")

    GOLD_OUTPUT_ROOT = REPORT_DIR / f'extract_{gold_set_name}_latest'
    _reset_latest_dir(GOLD_OUTPUT_ROOT)
    _gold_engine_dirs = {"custom_cleaner_v3": GOLD_OUTPUT_ROOT / "custom_cleaner_v3"}
    if _trafilatura is not None:
        _gold_engine_dirs["trafilatura"] = GOLD_OUTPUT_ROOT / "trafilatura"
    if _resiliparse_text is not None:
        _gold_engine_dirs["resiliparse"] = GOLD_OUTPUT_ROOT / "resiliparse"
    for _dir in _gold_engine_dirs.values():
        _dir.mkdir(parents=True, exist_ok=True)

    def _gold_write_doc(engine: str, doc: Dict[str, Any]) -> None:
        out_path = _gold_engine_dirs[engine] / f"{int(doc['id'])}.json"
        out_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _gold_make_doc(doc_id: int, url: str, engine: str, items: List[Tuple[str, str]]) -> Dict[str, Any]:
        blocks, parts, cursor = [], [], 0
        for block_type, raw_text in items:
            text = normalize_ws(str(raw_text or ""))
            if not text:
                continue
            start, end = cursor, cursor + len(text)
            blocks.append({
                "type": block_type, "level": None, "role": None,
                "text": text, "char_start": start, "char_end": end,
            })
            parts.append(text)
            cursor = end + 2
        full_text = "\n\n".join(parts)
        return {
            "id": int(doc_id), "url": url, "final_url": url,
            "lang": "unknown", "lang_conf": 0.0,
            "title": None, "published_date": None,
            "site": urlparse(url).netloc.lower(), "category": None,
            "text": full_text, "blocks": blocks, "n_chars": len(full_text),
            "sha1_text": hashlib.sha1(full_text.encode("utf-8")).hexdigest(),
            "dup_group": None, "extractor": engine,
            "quality_flags": ["gold_baseline"], "schema_version": 2,
        }

    _gold_timings = defaultdict(list)
    _gold_errors = defaultdict(int)
    _gold_dropped_blocks = []
    with ZipFile(GOLD_HTML_ZIP, "r") as _gold_zip:
        _gold_names = set(_gold_zip.namelist())
        _manifest_name = next((n for n in _gold_names if n == "manifest.csv" or n.endswith("/manifest.csv")), None)
        if _manifest_name is None:
            raise ValueError(f"{GOLD_HTML_ZIP} không có manifest.csv ở gốc hoặc thư mục con")
        _gold_prefix = _manifest_name[:-len("manifest.csv")]
        _gold_manifest = pd.read_csv(_gold_zip.open(_manifest_name), dtype={"id": "int64"})
        _required_cols = {"id", "lang", "domain", "url"}
        if not _required_cols.issubset(_gold_manifest.columns):
            raise ValueError(f"manifest.csv cần các cột: {sorted(_required_cols)}")

        for _gold_row in _gold_manifest.itertuples(index=False):
            _gold_domain = str(_gold_row.domain).strip().lower().removeprefix("www.")
            _gold_lang = str(_gold_row.lang).strip().lower()
            if _gold_domain and _gold_lang:
                DOMAIN_LANG_MAP.setdefault(_gold_domain, _gold_lang)
        _gold_html_domains = set(
            _gold_manifest.loc[
                ~_gold_manifest["domain"].astype(str).str.lower().eq("pubmed.ncbi.nlm.nih.gov"), "domain"
            ].astype(str).str.lower().str.removeprefix("www.")
        )
        _domains_without_hint = sorted(_gold_html_domains - set(CONTAINER_HINTS))
        print(f"Gold domains có hint: {len(_gold_html_domains - set(_domains_without_hint))}/{len(_gold_html_domains)}")
        if _domains_without_hint:
            logger.warning(f"Gold domains chưa có hint, dùng fallback: {_domains_without_hint}")

        _html_rows = _gold_manifest[
            ~_gold_manifest["domain"].astype(str).str.lower().eq("pubmed.ncbi.nlm.nih.gov")
        ]
        for _row in tqdm(_html_rows.itertuples(index=False), total=len(_html_rows), desc="Gold HTML"):
            _doc_id, _url = int(_row.id), str(_row.url)
            _gold_domain = str(_row.domain).strip().lower().removeprefix("www.")
            _html_name = f"{_gold_prefix}html/{_doc_id}.html"
            _meta_name = f"{_gold_prefix}html/{_doc_id}.meta.json"
            if _html_name not in _gold_names:
                logger.warning(f"Gold thiếu {_html_name}; bỏ qua ID {_doc_id}")
                continue
            _html_bytes = _gold_zip.read(_html_name)
            _meta = json.loads(_gold_zip.read(_meta_name).decode("utf-8")) if _meta_name in _gold_names else {}
            _encoding = detect_encoding(_html_bytes, _meta.get("content_type", ""))
            _html_text = _html_bytes.decode(_encoding, errors="replace")

            _t0 = time.perf_counter()
            try:
                _doc = extract_html_doc(_html_bytes, _encoding, _url, _doc_id)
                _before_filter = list(_doc.get('blocks', []))
                _doc = filter_doc(_doc)
                # filter_doc rebuilds text/offsets; refresh the content hash as well.
                _doc['sha1_text'] = hashlib.sha1(_doc['text'].encode('utf-8')).hexdigest()
                _remaining = Counter((b.get('type'), b.get('text')) for b in _doc.get('blocks', []))
                for _block in _before_filter:
                    _key = (_block.get('type'), _block.get('text'))
                    if _remaining[_key]:
                        _remaining[_key] -= 1
                    else:
                        _gold_dropped_blocks.append({
                            'id': _doc_id, 'domain': _gold_domain,
                            'block_type': _block.get('type'), 'text': _block.get('text', ''),
                        })
                _gold_write_doc("custom_cleaner_v3", _doc)
            except Exception:
                _gold_errors["custom_cleaner_v3"] += 1
                logger.exception(f"custom_cleaner_v3 lỗi ID {_doc_id}")
            finally:
                _gold_timings["custom_cleaner_v3"].append(time.perf_counter() - _t0)

            if _trafilatura is not None:
                _t0 = time.perf_counter()
                try:
                    _xml = _trafilatura.extract(
                        _html_text, url=_url, output_format="xml",
                        include_comments=False, include_tables=True,
                        include_formatting=True, favor_recall=True,
                    )
                    _items = []
                    if _xml:
                        _root = ET.fromstring(_xml)
                        _main = _root.find(".//main")
                        _main = _main if _main is not None else _root
                        _block_tags = {"head", "p", "quote", "item", "cell"}
                        for _node in _main.iter():
                            _tag = _node.tag.rsplit("}", 1)[-1] if isinstance(_node.tag, str) else ""
                            if _tag not in _block_tags:
                                continue
                            _desc = list(_node.iter())[1:]
                            if any((d.tag.rsplit("}", 1)[-1] if isinstance(d.tag, str) else "") in _block_tags for d in _desc):
                                continue
                            _text = "".join(_node.itertext())
                            _items.append(("heading" if _tag == "head" else "paragraph", _text))
                    _gold_write_doc("trafilatura", _gold_make_doc(_doc_id, _url, "trafilatura", _items))
                except Exception:
                    _gold_errors["trafilatura"] += 1
                    logger.exception(f"trafilatura lỗi ID {_doc_id}")
                finally:
                    _gold_timings["trafilatura"].append(time.perf_counter() - _t0)

            if _resiliparse_text is not None:
                _t0 = time.perf_counter()
                try:
                    _plain = _resiliparse_text(
                        _html_text, preserve_formatting=True, main_content=True,
                        list_bullets=False, alt_texts=False, links=False,
                    ) or ""
                    _items = [
                        ("paragraph", _part)
                        for _part in re.split(r"\n\s*\n+", _plain.strip())
                        if _part.strip()
                    ]
                    _gold_write_doc("resiliparse", _gold_make_doc(_doc_id, _url, "resiliparse", _items))
                except Exception:
                    _gold_errors["resiliparse"] += 1
                    logger.exception(f"resiliparse lỗi ID {_doc_id}")
                finally:
                    _gold_timings["resiliparse"].append(time.perf_counter() - _t0)

    # Xuất toàn bộ block đã loại và một mẫu ngẫu nhiên phủ đủ domain để kiểm tra thủ công.
    _dropped_df = pd.DataFrame(_gold_dropped_blocks, columns=['id', 'domain', 'block_type', 'text'])
    _dropped_all_path = GOLD_OUTPUT_ROOT / 'gold_postfilter_dropped_blocks.csv'
    _dropped_sample_path = GOLD_OUTPUT_ROOT / 'gold_postfilter_review_sample.csv'
    _dropped_df.to_csv(_dropped_all_path, index=False, encoding='utf-8-sig')
    if not _dropped_df.empty:
        _first_per_domain = _dropped_df.groupby('domain', group_keys=False).sample(n=1, random_state=1017)
        _need = min(200, len(_dropped_df)) - len(_first_per_domain)
        if _need > 0:
            _rest = _dropped_df.drop(index=_first_per_domain.index).sample(n=_need, random_state=1017)
            _dropped_sample = pd.concat([_first_per_domain, _rest]).sample(frac=1, random_state=1017)
        else:
            _dropped_sample = _first_per_domain.sample(frac=1, random_state=1017)
    else:
        _dropped_sample = _dropped_df
    _dropped_sample.to_csv(_dropped_sample_path, index=False, encoding='utf-8-sig')
    print(f'Postfilter đã bỏ {len(_dropped_df):,} block trên { _dropped_df.domain.nunique() if not _dropped_df.empty else 0} domain.')
    print(f'Toàn bộ block bị bỏ: {_dropped_all_path}; mẫu rà soát: {_dropped_sample_path}')
    if not _dropped_sample.empty:
        display(_dropped_sample)

    # 8 PMID trong Gold phải đi qua Europe PMC API, không dùng bộ trích HTML.
    _gold_pm_rows = _gold_manifest[
        _gold_manifest["domain"].astype(str).str.lower().eq("pubmed.ncbi.nlm.nih.gov")
    ]
    if len(_gold_pm_rows):
        if "fetch_pubmed_batches" not in globals():
            logger.warning("Chưa có fetch_pubmed_batches; chạy cell 24 rồi chạy lại cell Gold.")
        else:
            _gold_pmids = [
                str(urlparse(str(u)).path.rstrip("/").split("/")[-1])
                for u in _gold_pm_rows["url"]
            ]
            _t0 = time.perf_counter()
            try:
                _gold_hits = fetch_pubmed_batches(_gold_pmids, batch_size=max(1, len(_gold_pmids)), max_batches=1)
                _gold_by_pmid = {str(h.get("pmid", "")).strip(): h for h in _gold_hits}
                for _row in _gold_pm_rows.itertuples(index=False):
                    _pmid = str(urlparse(str(_row.url)).path.rstrip("/").split("/")[-1])
                    _hit = _gold_by_pmid.get(_pmid)
                    if _hit is None:
                        logger.warning(f"Europe PMC không trả PMID {_pmid}")
                        continue
                    _doc = extract_pubmed_doc(_hit, int(_row.id), str(_row.url))
                    for _engine in _gold_engine_dirs:
                        _gold_write_doc(_engine, _doc)
                _gold_timings["europepmc_api"].append(time.perf_counter() - _t0)
            except Exception:
                logger.exception("Không xử lý được PubMed subset trong Gold.")

    print(f"Gold archive: {GOLD_HTML_ZIP}")
    print(f"HTML pages: {len(_html_rows):,}; PubMed pages: {len(_gold_pm_rows):,}")
    print(f"Prediction files: {GOLD_OUTPUT_ROOT}")
    for _engine in _gold_engine_dirs:
        _times = _gold_timings.get(_engine, [])
        _avg = (sum(_times) / len(_times)) if _times else 0.0
        print(f"  {_engine}: avg={_avg:.3f}s/page | errors={_gold_errors[_engine]}")

    return {
        'gold_set': gold_set_name, 'archive': str(GOLD_HTML_ZIP),
        'output_root': str(GOLD_OUTPUT_ROOT), 'html_pages': int(len(_html_rows)),
        'pubmed_pages': int(len(_gold_pm_rows)),
        'custom_docs': len(list(_gold_engine_dirs['custom_cleaner_v3'].glob('*.json'))),
        'custom_errors': int(_gold_errors['custom_cleaner_v3']),
        'build': GOLD_NOTEBOOK_BUILD, 'postfilter_sha1': POSTFILTER_SHA1,
        'hints_sha1': CONTAINER_HINTS_SHA1,

    }

_GOLD_ARCHIVE_PATHS = {
    'gold1': _find_gold_archive('gold_html.zip'),
    'gold2': _find_gold_archive('gold2_html.zip'),
    'gold3': _find_gold_archive('gold3_html.zip'),
}
_gold_run_summaries = [
    run_gold_archive(_gold_set, _archive_path)
    for _gold_set, _archive_path in _GOLD_ARCHIVE_PATHS.items()
]

# Extraction inventory; approval is checked by check_gate before 40K.
_comparison_dir = REPORT_DIR / 'gold_comparison_latest'
_reset_latest_dir(_comparison_dir)
pd.DataFrame(_gold_run_summaries).to_csv(_comparison_dir / 'gold_comparison_summary.csv', index=False, encoding='utf-8-sig')
display(pd.DataFrame(_gold_run_summaries))
flush_log_handlers(sync=True)


# %% CELL 21
# ---- Kiểm tra toàn vẹn sau khi chạy Gold ----
# So custom_cleaner với trafilatura/resiliparse trên cùng trang: ca nào custom NGẮN HƠN rõ rệt, hoặc thiếu
# đoạn dài mà baseline có, hoặc có cờ chất lượng đáng ngờ thì liệt kê để xem tay.
def _gold_load(engine):
    out = {}
    d = _gold_engine_dirs.get(engine)
    if d is None: return out
    for p in d.glob('*.json'):
        try: out[int(p.stem)] = json.loads(p.read_text(encoding='utf-8'))
        except Exception: pass
    return out

_norm = lambda s: re.sub(r'\W+', '', (s or '').lower())
_cust = _gold_load('custom_cleaner_v3')
_refs = [r for r in (_gold_load('trafilatura'), _gold_load('resiliparse')) if r]
_SUSPECT_FLAGS = {'short', 'possible_truncation', 'tail_recovered', 'container_text_fallback', 'container_suspect', 'soft_404'}

def _missing_paragraphs(doc_id, min_len=80):
    """Đoạn dài (>=min_len ký tự) mà baseline có nhưng custom không chứa."""
    ct = _norm(_cust[doc_id]['text'])
    seen, out = set(), []
    for ref in _refs:
        for b in (ref.get(doc_id) or {}).get('blocks', []):
            k = _norm(b['text'])
            if len(b['text']) >= min_len and k not in ct and k not in seen:
                seen.add(k); out.append(b['text'])
    return out

_rows = []
for _id, _d in sorted(_cust.items()):
    _ref_len = max([len(r[_id]['text']) for r in _refs if _id in r] or [0])
    _miss = _missing_paragraphs(_id)
    _flags = set(_d['quality_flags'])
    _rows.append({
        'id': _id, 'site': _d['site'], 'n_chars': _d['n_chars'], 'ref_max_chars': _ref_len,
        'ratio': round(_d['n_chars'] / _ref_len, 2) if _ref_len else None,
        'n_missing_vs_baseline': len(_miss), 'flags': ','.join(sorted(_flags)),
        'suspect': bool(_flags & _SUSPECT_FLAGS) or bool(_miss) or (_ref_len and _d['n_chars'] < 0.7 * _ref_len),
    })
audit_df = pd.DataFrame(_rows)
audit_df.to_csv(GOLD_OUTPUT_ROOT / 'gold_integrity_audit.csv', index=False, encoding='utf-8-sig')
print(f"{len(audit_df)} trang | nghi vấn: {int(audit_df['suspect'].sum())} | "
      f"trang ngắn (<200): {int((audit_df['n_chars'] < 200).sum())} | CSV: {GOLD_OUTPUT_ROOT / 'gold_integrity_audit.csv'}")
try:
    display(audit_df[audit_df['suspect']].sort_values(['ratio', 'n_chars']).head(30))
except NameError:
    print(audit_df[audit_df['suspect']].sort_values(['ratio', 'n_chars']).head(30).to_string())

def show_doc(doc_id, max_missing=5):
    """Xem từng block của 1 trang + các đoạn baseline có mà custom thiếu."""
    d = _cust[doc_id]
    print(f"ID {doc_id} {d['url']}\nflags={d['quality_flags']} chars={d['n_chars']}")
    for b in d['blocks']:
        print(f"  [{b['type']:9s}] {b['text'][:110]}")
    for t in _missing_paragraphs(doc_id)[:max_missing]:
        print('  MISSING vs baseline >>', t[:160])


# %% CELL 23
# Canonical 40K manifest: crawl/pilot2_sample.csv in this project.
PROJECT_PILOT2_MANIFEST_SHA256 = '131b0f19cbd299b7c79e41c43b601211ed2a8127d368ba1efa0047e770f5f782'


def validate_project_pilot2_sample(sample):
    if len(sample) != 40000 or sample['id'].nunique() != 40000:
        raise ValueError('Cần đúng 40.000 ID của crawl/pilot2_sample.csv trong dự án.')
    rows = [[int(r.id), str(r.url)] for r in sample[['id', 'url']].sort_values('id').itertuples(index=False)]
    actual = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    if actual != PROJECT_PILOT2_MANIFEST_SHA256:
        raise ValueError('Mẫu 40K khác crawl/pilot2_sample.csv của dự án. Đồng bộ đúng file này lên DATA_ROOT/crawl; không tạo mẫu mới.')
    return actual


def _shard_record_order(rec, path_index, line_no):
    """Prefer crawl timestamp; legacy records use deterministic shard/append order."""
    stamp = pd.to_datetime(rec.get('fetched_at'), errors='coerce', utc=True)
    return (0 if pd.isna(stamp) else 1,
            0 if pd.isna(stamp) else stamp.value, path_index, line_no)


def _host_key(url):
    return (urlparse(url).hostname or '').lower().removeprefix('www.')

def _is_crawl_challenge(final_url, url='', body_size=None):
    parts = urlparse(final_url or '')
    if 'verify.html' in parts.path.lower():
        return True
    # Review 3.1: a tiny response on another host is a challenge, not an article.
    return bool(body_size is not None and body_size < 3000 and _host_key(url)
                and _host_key(final_url) and _host_key(url) != _host_key(final_url))

def _empty_doc(doc_id, url, final_url, flags):
    site = _host_key(url)
    return {'id': int(doc_id), 'url': url, 'final_url': final_url or url,
            'lang': DOMAIN_LANG_MAP.get(site, 'unknown'), 'lang_conf': 0.0,
            'title': '', 'published_date': None, 'site': site, 'category': None,
            'text': '', 'blocks': [], 'n_chars': 0,
            'sha1_text': hashlib.sha1(b'').hexdigest(), 'dup_group': None,
            'extractor': 'custom_cleaner_v3', 'quality_flags': list(dict.fromkeys(flags)),
            'schema_version': 2}

def _clear_unextractable_doc(doc):
    if {'soft_404', 'blocked'}.intersection(doc.get('quality_flags') or []):
        doc.update(text='', blocks=[], n_chars=0, sha1_text=hashlib.sha1(b'').hexdigest())
    return doc

def _extract_shard_html_doc(body, encoding, url, final_url, doc_id, homepage_titles=None, dropped_sink=None):
    if _is_crawl_challenge(final_url, url, len(body)):
        logger.warning(f'BLOCKED id={doc_id} source_url={url} final_url={final_url}')
        return _empty_doc(doc_id, url, final_url, ['blocked', 'crawl_challenge'])
    raw_doc = extract_html_doc(body, encoding, url, doc_id)
    filtered = filter_doc(raw_doc)
    if dropped_sink is not None:
        kept = Counter((str(b.get('type') or ''), str(b.get('text') or '')) for b in filtered.get('blocks', []))
        for block in raw_doc.get('blocks', []):
            key = (str(block.get('type') or ''), str(block.get('text') or ''))
            if kept[key]:
                kept[key] -= 1
            else:
                dropped_sink.append({'id': doc_id, 'url': url, 'site': _host_key(url), 'type': key[0], 'text': key[1]})
    doc = filtered
    doc['sha1_text'] = hashlib.sha1(doc['text'].encode('utf-8')).hexdigest()
    doc['url'], doc['final_url'] = url, final_url
    source_path, final_path = urlparse(url).path, urlparse(final_url).path
    title = normalize_ws(str(doc.get('title') or '')).casefold()
    home_title = (homepage_titles or {}).get(_host_key(url))
    if source_path not in ('', '/') and (final_path in ('', '/') or (title and title == home_title)):
        doc['quality_flags'] = list(dict.fromkeys(doc.get('quality_flags', []) + ['soft_404']))
    return _clear_unextractable_doc(doc)

def _write_parquet_replace(frame, target):
    target = Path(target)
    temporary = target.with_name(target.name + '.tmp')
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    finally:
        if temporary.exists(): temporary.unlink()

def _write_jsonl_zst_replace(records, target, level=6):
    """Atomically write one extracted document per compressed JSONL line."""
    import zstandard as zstd
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    try:
        compressor = zstd.ZstdCompressor(level=level)
        with temporary.open('wb') as raw_out:
            with compressor.stream_writer(raw_out) as writer:
                for record in records:
                    line = json.dumps(record, ensure_ascii=False, separators=(',', ':'), default=str)
                    writer.write((line + '\n').encode('utf-8'))
        temporary.replace(target)
    finally:
        if temporary.exists(): temporary.unlink()

def validate_shard_gold_parity():
    from zipfile import ZipFile
    fields = ('id','url','final_url','title','text','blocks','n_chars','sha1_text','quality_flags')
    results=[]
    for gold_set, archive in _GOLD_ARCHIVE_PATHS.items():
        run=next(x for x in _gold_run_summaries if x['gold_set']==gold_set)
        expected_root=Path(run['output_root'])/'custom_cleaner_v3'
        mismatches=[]; missing_html=[]; checked=0
        with ZipFile(archive) as zf:
            names=set(zf.namelist())
            manifest_name=next(x for x in names if x=='manifest.csv' or x.endswith('/manifest.csv'))
            prefix=manifest_name[:-len('manifest.csv')]
            manifest=pd.read_csv(zf.open(manifest_name))
            rows=manifest[~manifest['domain'].astype(str).str.lower().eq('pubmed.ncbi.nlm.nih.gov')]
            for row in rows.itertuples(index=False):
                doc_id,url=int(row.id),str(row.url)
                html_name=f'{prefix}html/{doc_id}.html'
                if html_name not in names:
                    missing_html.append(doc_id)
                    continue
                body=zf.read(html_name); meta_name=f'{prefix}html/{doc_id}.meta.json'
                meta=json.loads(zf.read(meta_name).decode('utf-8')) if meta_name in names else {}
                enc=detect_encoding(body,meta.get('content_type',''))
                doc=_extract_shard_html_doc(body,enc,url,url,doc_id)
                expected=json.loads((expected_root/f'{doc_id}.json').read_text(encoding='utf-8'))
                checked+=1
                if any(doc.get(k)!=expected.get(k) for k in fields): mismatches.append(doc_id)
        results.append({'gold_set':gold_set,'expected_html':len(rows),'checked_html':checked,'missing_html':len(missing_html),'missing_examples':missing_html[:5],'mismatches':len(mismatches),'examples':mismatches[:5], 'build':GOLD_NOTEBOOK_BUILD, 'postfilter_sha1':POSTFILTER_SHA1, 'hints_sha1':CONTAINER_HINTS_SHA1})
    result=pd.DataFrame(results)
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    result.to_csv(REPORT_DIR/'shard_gold_parity.csv',index=False,encoding='utf-8-sig')
    logger.info(f"SHARD_GOLD_PARITY checked={int(result['checked_html'].sum())} mismatches={int(result['mismatches'].sum())} report={REPORT_DIR/'shard_gold_parity.csv'} build={GOLD_NOTEBOOK_BUILD} postfilter_sha1={POSTFILTER_SHA1} hints_sha1={CONTAINER_HINTS_SHA1}")
    flush_log_handlers(sync=True)
    print(result.to_string(index=False))
    print(f"Saved parity report: {REPORT_DIR/'shard_gold_parity.csv'}")
    if result['mismatches'].sum() or result['missing_html'].sum(): raise AssertionError('Shard route differs from Gold output or one/more Gold HTML files were not checked.')
    return result

def run_sample2_extraction(shard_paths=None, sample_csv=None, resume=False, sample_limit=None, status_db=None):
    """Reclean existing pilot shards; preserve coverage and report every unextractable ID."""
    import base64
    import io
    import zstandard as zstd
    require_review_gold_pass()
    paths = resolve_existing_pilot_shards(shard_paths)
    sample_csv = Path(sample_csv) if sample_csv else CRAWL_DIR / 'pilot2_sample.csv'
    sample = pd.read_csv(sample_csv, usecols=['id', 'url'])
    sample['id'] = sample['id'].astype('int64')
    if sample['id'].duplicated().any():
        raise ValueError('pilot2_sample.csv contains duplicate IDs')
    if sample['url'].isna().any() or sample['url'].astype(str).str.strip().eq('').any():
        raise ValueError('pilot2_sample.csv contains missing URLs')
    manifest_sha256 = validate_project_pilot2_sample(sample)
    total_sample_ids = len(sample)
    if sample_limit is not None:
        if int(sample_limit) <= 0:
            raise ValueError('sample_limit must be positive')
        sample = sample.sort_values('id').head(int(sample_limit))
    elif total_sample_ids != 40000:
        raise ValueError(f'Pilot phải có 40.000 ID; hiện có {total_sample_ids:,}.')
    sample_ids = set(sample['id'])
    url_by_id = dict(zip(sample['id'], sample['url'].astype(str)))
    latest = _load_latest_status(sample_ids, status_db)
    status_by_id = latest.to_dict(orient='index')
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = '' if sample_limit is None else f'_{len(sample_ids)}'
    output = DOCS_DIR / f'docs_sample2{suffix}.parquet'
    checkpoint = DOCS_DIR / f'sample2{suffix}_checkpoint.json'
    database = Path(status_db) if status_db else CRAWL_DIR / 'status.duckdb'
    db_stat = database.stat()
    signature = {'project_manifest_sha256': manifest_sha256, 'build': GOLD_NOTEBOOK_BUILD, 'postfilter_sha1': POSTFILTER_SHA1,
                 'hints_sha1': CONTAINER_HINTS_SHA1, 'sample_ids_sha1': _sample_id_signature(sample_ids),
                 'source_inventory_sha1': _source_inventory_signature(paths),
                 'html_selection_policy': 'latest-shard-http200-v1',
                 'status_db_size': db_stat.st_size, 'status_db_mtime_ns': db_stat.st_mtime_ns}
    if resume and output.is_file() and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        summary_path = REPORT_DIR / f'sample2{suffix}_extraction_summary.csv'
        if (all(saved.get(k) == v for k, v in signature.items()) and saved.get('finished')
                and summary_path.is_file() and (REPORT_DIR / f'sample2{suffix}_missing.csv').is_file()):
            print(f'Lượt này đã xong và khớp build/DB/shard: {output}')
            return pd.read_csv(summary_path)
        raise RuntimeError('Checkpoint không khớp. Chạy resume=False để clean lại bằng build hiện tại.')

    dctx = zstd.ZstdDecompressor()
    homepage_titles = {}
    selected_records = {}
    # Reuse the four files for home-title detection; no 83-shard scan or staging.
    for path_index, path in enumerate(paths):
        with path.open('rb') as compressed, dctx.stream_reader(compressed) as reader, io.TextIOWrapper(reader, encoding='utf-8') as lines:
            for line_no, line in enumerate(lines, 1):
                try:
                    rec = json.loads(line)
                    rid = int(rec['id'])
                    ct = str(rec.get('content_type') or '').lower()
                    if (rec.get('http_code') != 200 or not rec.get('body_b64')
                            or (ct and 'html' not in ct and 'xhtml' not in ct)):
                        continue
                    body = base64.b64decode(rec['body_b64'], validate=True)
                    if not body:
                        continue
                    order = _shard_record_order(rec, path_index, line_no)
                    if rid in sample_ids and (rid not in selected_records or order > selected_records[rid][0]):
                        selected_records[rid] = (order, str(path.resolve()), line_no)
                    url = url_by_id.get(rid) or str(rec.get('url') or '')
                    final_url = str(rec.get('final_url') or url)
                    home_url = final_url if urlparse(final_url).path in ('', '/') else url
                    if home_url and urlparse(home_url).path in ('', '/') and rec.get('body_b64'):
                        body = base64.b64decode(rec['body_b64'], validate=True)
                        if _is_crawl_challenge(final_url, url, len(body)):
                            continue
                        home = extract_html_doc(body, str(rec.get('encoding') or ''), home_url, rid)
                        title = normalize_ws(str(home.get('title') or '')).casefold()
                        if title:
                            homepage_titles[_host_key(home_url)] = title
                except (ValueError, KeyError, TypeError):
                    continue

    docs, failures, occurrences, source_by_id = {}, {}, Counter(), {}
    dropped, shard_summaries = [], []
    for path in paths:
        found = matched = errors = malformed = 0
        with path.open('rb') as compressed, dctx.stream_reader(compressed) as reader, io.TextIOWrapper(reader, encoding='utf-8') as lines:
            for line_no, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                rid = None
                try:
                    rec = json.loads(line)
                    rid = int(rec['id'])
                    if rid not in sample_ids:
                        continue
                    occurrences[rid] += 1
                    found += 1
                    selected = selected_records.get(rid)
                    if selected is None or selected[1:] != (str(path.resolve()), line_no):
                        continue
                    status = status_by_id.get(rid, {})
                    body = base64.b64decode(rec['body_b64'], validate=True)
                    actual_sha = hashlib.sha1(body).hexdigest()
                    db_mismatch = (status.get('status') != 'ok'
                                   or str(status.get('sha1_raw') or '') != actual_sha)
                    matched += 1
                    ct = str(rec.get('content_type') or '').lower()
                    if ct and 'html' not in ct and 'xhtml' not in ct:
                        failures[rid] = 'non_html'
                        continue
                    url = url_by_id[rid]
                    final_url = str(rec.get('final_url') or url)
                    doc = _extract_shard_html_doc(body, str(rec.get('encoding') or ''), url, final_url,
                                                  rid, homepage_titles, dropped)
                    if db_mismatch:
                        doc['quality_flags'] = list(dict.fromkeys(doc.get('quality_flags', []) + ['db_status_mismatch']))
                    doc.update(selected_source=f'{path.name}:{line_no}', selected_sha1_raw=actual_sha,
                               selected_fetched_at=str(rec.get('fetched_at') or ''), db_status_mismatch=db_mismatch)
                    docs[rid] = doc
                    source_by_id[rid] = f'{path.name}:{line_no}'
                    failures.pop(rid, None)
                    if not doc.get('text') and not {'blocked', 'soft_404'}.intersection(doc.get('quality_flags') or []):
                        failures[rid] = 'empty_extraction'
                except Exception:
                    errors += 1
                    if rid in sample_ids:
                        failures[rid] = 'extraction_error'
                    else:
                        malformed += 1
                    logger.exception(f'SAMPLE2_ROW_ERROR shard={path.name} line={line_no} id={rid}')
        shard_summaries.append({'shard': path.name, 'sample_records': found, 'selected_http200': matched,
                                'errors': errors, 'malformed_rows': malformed})
        logger.info(f'SAMPLE2_SHARD shard={path.name} sample_records={found} selected_http200={matched} errors={errors} '
                    f'docs={len(docs)} build={GOLD_NOTEBOOK_BUILD} postfilter_sha1={POSTFILTER_SHA1} hints_sha1={CONTAINER_HINTS_SHA1}')
        flush_log_handlers(sync=True)

    missing_rows = []
    for rid in sorted(sample_ids):
        status = status_by_id.get(rid, {})
        state = str(status.get('status') or '')
        code = status.get('http_code')
        code = int(code) if pd.notna(code) else None
        reason = failures.get(rid)
        doc = docs.get(rid)
        if doc is None:
            if not status:
                reason = 'not_crawled'
            elif state != 'ok':
                reason = f'http_{code}' if code is not None and code >= 400 else (state or 'not_crawled')
            elif not occurrences[rid]:
                reason = 'status_ok_missing_shard'
            else:
                reason = reason or 'no_usable_html_in_shard'
            flags = [reason]
            if state == 'blocked' or code in (401, 403, 429):
                flags.append('blocked')
            doc = _empty_doc(rid, url_by_id[rid], str(status.get('final_url') or url_by_id[rid]), flags)
            docs[rid] = doc
        flags = doc.get('quality_flags') or []
        if 'blocked' in flags:
            reason = reason or 'blocked'
        elif 'soft_404' in flags:
            reason = 'soft_404'
        elif not doc.get('text'):
            reason = reason or 'empty_extraction'
        if reason:
            missing_rows.append({'id': rid, 'url': url_by_id[rid], 'final_url': doc['final_url'],
                                 'reason': reason, 'status': state or 'not_seen', 'http_code': code,
                                 'db_shard': status.get('shard'), 'sha1_raw': status.get('sha1_raw', ''),
                                 'occurrences': occurrences[rid], 'selected_source': source_by_id.get(rid, ''),
                                 'quality_flags': doc['quality_flags']})
    frame = pd.DataFrame([docs[k] for k in sorted(docs)])
    frame['db_status_mismatch'] = frame.get('db_status_mismatch', pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    mismatch_rows = []
    for rid, doc in docs.items():
        if doc.get('db_status_mismatch'):
            status = status_by_id.get(rid, {})
            mismatch_rows.append({'id': rid, 'url': doc['url'], 'db_status': status.get('status'),
                'db_http_code': status.get('http_code'), 'db_sha1_raw': status.get('sha1_raw'),
                'selected_sha1_raw': doc['selected_sha1_raw'], 'selected_source': doc['selected_source'],
                'selected_fetched_at': doc['selected_fetched_at'], 'n_chars': doc['n_chars']})
    pd.DataFrame(mismatch_rows, columns=['id', 'url', 'db_status', 'db_http_code', 'db_sha1_raw',
        'selected_sha1_raw', 'selected_source', 'selected_fetched_at', 'n_chars']).to_csv(
        REPORT_DIR / f'sample2{suffix}_db_status_mismatch.csv', index=False, encoding='utf-8-sig')
    _write_parquet_replace(frame, output)
    missing_columns = ['id', 'url', 'final_url', 'reason', 'status', 'http_code', 'db_shard', 'sha1_raw',
                       'occurrences', 'selected_source', 'quality_flags']
    missing = pd.DataFrame(missing_rows, columns=missing_columns)
    missing.to_csv(REPORT_DIR / f'sample2{suffix}_missing.csv', index=False, encoding='utf-8-sig')
    missing.to_csv(REPORT_DIR / f'sample2{suffix}_retry_queue.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(shard_summaries).to_csv(REPORT_DIR / f'sample2{suffix}_shard_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(dropped, columns=['id', 'url', 'site', 'type', 'text']).to_csv(
        REPORT_DIR / f'sample2{suffix}_postfilter_dropped_blocks.csv', index=False, encoding='utf-8-sig')
    frame['__short300'] = frame['n_chars'].lt(300)
    for flag in ['container_suspect', 'soft_404', 'possible_truncation', 'blocked', 'crawl_challenge']:
        frame['__' + flag] = frame['quality_flags'].apply(lambda value: flag in value)
    frame['__qa_q'] = frame['blocks'].apply(lambda blocks: sum(b.get('type') == 'qa_q' for b in blocks))
    frame['__qa_a'] = frame['blocks'].apply(lambda blocks: sum(b.get('type') == 'qa_a' for b in blocks))
    domain = frame.groupby('site', dropna=False).agg(
        documents=('id', 'count'), median_chars=('n_chars', 'median'), under_300_rate=('__short300', 'mean'),
        container_suspect_rate=('__container_suspect', 'mean'), soft_404_rate=('__soft_404', 'mean'),
        possible_truncation_rate=('__possible_truncation', 'mean'), blocked_rate=('__blocked', 'mean'),
        crawl_challenge_rate=('__crawl_challenge', 'mean'), qa_q_blocks=('__qa_q', 'sum'), qa_a_blocks=('__qa_a', 'sum')).reset_index()
    domain.to_csv(REPORT_DIR / f'sample2{suffix}_domain_quality.csv', index=False, encoding='utf-8-sig')
    detail_cols = ['id', 'url', 'final_url', 'n_chars', 'quality_flags', 'text']
    frame[frame['n_chars'].lt(200)][detail_cols].to_csv(REPORT_DIR / f'sample2{suffix}_short_docs.csv', index=False, encoding='utf-8-sig')
    frame[frame['__crawl_challenge']][detail_cols].to_csv(REPORT_DIR / f'sample2{suffix}_challenge_pages.csv', index=False, encoding='utf-8-sig')
    # Empty placeholder rows are coverage records, not duplicate article content.
    duplicates = (frame[frame['n_chars'].gt(0)].groupby('sha1_text')
        .agg(documents=('id', 'count'), urls=('url', 'nunique'), sample_urls=('url', lambda urls: ' | '.join(list(urls)[:5]))).reset_index())
    duplicates[(duplicates['documents'] > 1) & (duplicates['urls'] > 1)].to_csv(
        REPORT_DIR / f'sample2{suffix}_duplicate_text_groups.csv', index=False, encoding='utf-8-sig')
    offset_errors = []
    for row in frame[['id', 'url', 'text', 'blocks']].itertuples(index=False):
        for block in row.blocks:
            a, b = block.get('char_start'), block.get('char_end')
            if a is None or b is None or row.text[a:b] != block.get('text'):
                offset_errors.append({'id': row.id, 'url': row.url, 'char_start': a, 'char_end': b, 'block_text': block.get('text')})
    pd.DataFrame(offset_errors, columns=['id', 'url', 'char_start', 'char_end', 'block_text']).to_csv(
        REPORT_DIR / f'sample2{suffix}_offset_errors.csv', index=False, encoding='utf-8-sig')
    review = pd.concat([group.sample(n=min(5, len(group)), random_state=20261020)
                        for _, group in frame.groupby('site', dropna=False)])
    review[['id', 'url', 'title', 'n_chars', 'quality_flags', 'text']].to_csv(
        REPORT_DIR / f'sample2{suffix}_review_samples.csv', index=False, encoding='utf-8-sig')
    report = {'pilot2_total_ids': total_sample_ids, 'expected_ids': len(sample_ids), 'coverage_rows': len(frame),
              'db_status_mismatch_ids': int(frame['db_status_mismatch'].sum()),
              'expected_content_ids_review': 30224,
              'extracted_ids': int(frame['n_chars'].gt(0).sum()), 'missing_ids': len(missing),
              'status_ok_missing_ids': int(missing['status'].eq('ok').sum()),
              'short_ids': int(frame['n_chars'].lt(200).sum()), 'challenge_ids': int(frame['__crawl_challenge'].sum()),
              'blocked_ids': int(frame['__blocked'].sum()), 'soft_404_ids': int(frame['__soft_404'].sum()),
              'retry_queue_ids': len(missing), 'complete': missing.empty, 'coverage_complete': len(frame) == len(sample_ids),
              'bad_offsets': len(offset_errors), 'shards_scanned': len(paths), 'output': str(output), **signature}
    summary = pd.DataFrame([report])
    summary.to_csv(REPORT_DIR / f'sample2{suffix}_extraction_summary.csv', index=False, encoding='utf-8-sig')
    checkpoint.write_text(json.dumps({**signature, 'finished': True, 'coverage_rows': len(frame)}, indent=2), encoding='utf-8')
    logger.info(f'SAMPLE2_DONE {json.dumps(report, ensure_ascii=False)}')
    flush_log_handlers(sync=True)
    print(f"Mẫu 2: {len(frame):,} dòng phủ ID; {report['extracted_ids']:,} doc có nội dung; {len(missing):,} doc cần xem lý do.")
    return summary


def audit_pharmacity_access(status_db=None):
    """Summarize crawler access to Pharmacity and write blocked-domain evidence."""
    import duckdb
    database=Path(status_db) if status_db else CRAWL_DIR/'status.duckdb'
    if not database.is_file(): raise FileNotFoundError(f'Missing crawl status database: {database}')
    con=duckdb.connect(str(database),read_only=True)
    try:
        rows=con.execute("""SELECT id,url,final_url,status,http_code FROM crawl_status
            WHERE lower(url) LIKE '%pmc-ecm-healthblog.beta.pharmacity.io%'
               OR lower(coalesce(final_url,'')) LIKE '%pmc-ecm-healthblog.beta.pharmacity.io%'""").df()
    finally: con.close()
    code403=int(rows['http_code'].eq(403).sum()) if len(rows) else 0
    blocked=int(rows['status'].astype(str).str.lower().eq('blocked').sum()) if len(rows) else 0
    is_blocked=bool(len(rows) and code403==len(rows))
    result=pd.DataFrame([{
        'domain':'pmc-ecm-healthblog.beta.pharmacity.io','expected_urls':5333,
        'attempted_urls':len(rows),'http_403':code403,'status_blocked':blocked,
        'access_state':'blocked_403' if is_blocked else ('not_seen' if rows.empty else 'accessible_or_mixed'),
    }])
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    result.to_csv(REPORT_DIR/'pharmacity_access.csv',index=False,encoding='utf-8-sig')
    blocked_path=REPORT_DIR/'blocked_domains.csv'
    if is_blocked:
        existing=pd.read_csv(blocked_path) if blocked_path.is_file() else pd.DataFrame(columns=['domain','reason','attempted_urls','http_403'])
        existing=existing[existing['domain'].astype(str).ne('pmc-ecm-healthblog.beta.pharmacity.io')]
        existing=pd.concat([existing,pd.DataFrame([{'domain':'pmc-ecm-healthblog.beta.pharmacity.io','reason':'Crawler received HTTP 403 for every recorded URL','attempted_urls':len(rows),'http_403':code403}])],ignore_index=True)
        existing.to_csv(blocked_path,index=False,encoding='utf-8-sig')
    logger.info(f"PHARMACITY_ACCESS state={result.iloc[0]['access_state']} attempted={len(rows)} http_403={code403}")
    flush_log_handlers(sync=True)
    return result

def run_shard_extraction(shard_paths=None, sample_csv=None, overwrite=True, nt_approved=False, status_db=None):
    """Extract all available crawl shards after NT approval, selecting latest status=ok HTML once per ID."""
    if not nt_approved:
        raise PermissionError('Chỉ chạy full shards sau khi NT duyệt mẫu 2; truyền nt_approved=True để xác nhận.')
    import base64,io,zstandard as zstd
    database=Path(status_db) if status_db else CRAWL_DIR/'status.duckdb'
    latest_ok=_load_latest_ok_status(None,database)
    if latest_ok.empty: raise RuntimeError(f'No successful crawl_status rows in {database}')
    shard_paths=_ordered_source_shards(shard_paths if shard_paths is not None else HTML_DIR.glob('shard_*.jsonl.zst'))
    if not shard_paths: raise FileNotFoundError(f'No crawl HTML shard files found in {HTML_DIR}')
    DOCS_DIR.mkdir(parents=True,exist_ok=True); REPORT_DIR.mkdir(parents=True,exist_ok=True)
    output_paths=[DOCS_DIR/f"docs_{int(re.search(r'shard_([0-9]+)',p.name).group(1)):05d}.parquet" for p in shard_paths]
    existing=[p for p in output_paths if p.exists()]
    if existing and not overwrite: raise FileExistsError(f'Output parquet exists; set overwrite=True after review: {existing[:3]}')
    expected_sha=latest_ok['sha1_raw'].to_dict()
    url_by_id=latest_ok['url'].fillna('').astype(str).to_dict()
    final_url_by_id=latest_ok['final_url'].fillna('').astype(str).to_dict()
    dctx=zstd.ZstdDecompressor(); homepage_titles={}
    for source in shard_paths:
        with source.open('rb') as compressed,dctx.stream_reader(compressed) as reader,io.TextIOWrapper(reader,encoding='utf-8') as lines:
            for line in lines:
                try:
                    rec=json.loads(line); doc_id=int(rec['id']); url=url_by_id.get(doc_id,'')
                    if url and urlparse(url).path in ('','/') and rec.get('body_b64'):
                        body=base64.b64decode(rec['body_b64']); home=extract_html_doc(body,str(rec.get('encoding') or ''),url,doc_id)
                        title=normalize_ws(str(home.get('title') or '')).casefold()
                        if title: homepage_titles[_host_key(url)]=title
                except Exception: continue
    seen=set(); summaries=[]; challenge_rows=[]; total_errors=0
    for source,out_path in zip(shard_paths,output_paths):
        docs=[]; matched=errors=challenge_count=0
        with source.open('rb') as compressed,dctx.stream_reader(compressed) as reader,io.TextIOWrapper(reader,encoding='utf-8') as lines:
            for line_no,line in enumerate(lines,1):
                if not line.strip(): continue
                try:
                    rec=json.loads(line); doc_id=int(rec['id'])
                    if doc_id in seen or doc_id not in expected_sha or not rec.get('body_b64'): continue
                    body=base64.b64decode(rec['body_b64']); status_sha=str(expected_sha.get(doc_id) or '').lower()
                    if not status_sha or hashlib.sha1(body).hexdigest()!=status_sha: continue
                    seen.add(doc_id); matched+=1
                    content_type=str(rec.get('content_type') or '').lower()
                    if content_type and 'html' not in content_type and 'xhtml' not in content_type: continue
                    url=url_by_id.get(doc_id,''); final_url=final_url_by_id.get(doc_id,'') or str(rec.get('final_url') or url)
                    doc=_extract_shard_html_doc(body,str(rec.get('encoding') or ''),url,final_url,doc_id,homepage_titles)
                    if 'crawl_challenge' in (doc.get('quality_flags') or []):
                        challenge_rows.append(doc); challenge_count+=1
                    docs.append(doc)
                except Exception:
                    errors+=1; logger.exception(f'SHARD_ROW_ERROR shard={source.name} line={line_no}')
        _write_parquet_replace(pd.DataFrame(docs),out_path)
        total_errors+=errors
        shard_id=re.search(r'shard_(\d+)',source.name)
        logger.info(f"SHARD_DONE shard={shard_id.group(1) if shard_id else source.name} selected_latest_ok={matched} docs={len(docs)} challenge={challenge_count} errors={errors} build={GOLD_NOTEBOOK_BUILD} postfilter_sha1={POSTFILTER_SHA1} hints_sha1={CONTAINER_HINTS_SHA1}")
        flush_log_handlers(sync=True)
        summaries.append({'source_shard':source.name,'selected_latest_ok':matched,'docs':len(docs),'errors':errors,'output':str(out_path)})
    missing=set(latest_ok.index.astype('int64'))-seen
    summary=pd.DataFrame(summaries)
    summary.to_csv(REPORT_DIR/'shard_extraction_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'id':int(doc_id),'url':url_by_id.get(int(doc_id),''),'reason':'status_ok_html_missing_or_sha1_mismatch'} for doc_id in sorted(missing)]).to_csv(REPORT_DIR/'shard_extraction_unmatched_status_ok.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(challenge_rows).reindex(columns=['id','url','final_url','n_chars','quality_flags','text']).to_csv(REPORT_DIR/'shard_challenge_pages.csv',index=False,encoding='utf-8-sig')
    logger.info(f"SHARD_EXTRACTION_DONE files={len(shard_paths)} status_ok={len(latest_ok)} selected={len(seen)} unmatched={len(missing)} errors={total_errors} build={GOLD_NOTEBOOK_BUILD} postfilter_sha1={POSTFILTER_SHA1} hints_sha1={CONTAINER_HINTS_SHA1}")
    flush_log_handlers(sync=True)
    return summary


# %% CELL 25
def _sample_id_signature(sample_ids):
    return hashlib.sha1(','.join(map(str, sorted(sample_ids))).encode()).hexdigest()

def _ordered_source_shards(source_paths):
    return sorted((Path(x) for x in source_paths), key=lambda p: str(p.resolve()))

def _source_inventory_signature(paths):
    inventory = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha1(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()

def _load_latest_status(sample_ids=None, status_db=None):
    """Latest state per ID, including failed retries; never resurrect an older ok row."""
    import duckdb
    database = Path(status_db) if status_db else CRAWL_DIR / 'status.duckdb'
    if not database.is_file():
        raise FileNotFoundError(f'Missing crawl status database: {database}')
    con = duckdb.connect(str(database), read_only=True)
    try:
        required = {'id', 'url', 'final_url', 'status', 'http_code', 'shard', 'sha1_raw', 'fetched_at'}
        columns = {row[0] for row in con.execute('DESCRIBE crawl_status').fetchall()}
        if required - columns:
            raise ValueError(f'crawl_status missing columns: {sorted(required - columns)}')
        query = 'SELECT cs.id,cs.url,cs.final_url,cs.status,cs.http_code,cs.shard,cs.sha1_raw,cs.fetched_at FROM crawl_status cs'
        if sample_ids is not None:
            con.register('_pilot_ids', pd.DataFrame({'id': sorted(sample_ids)}).astype({'id': 'int64'}))
            query += ' INNER JOIN _pilot_ids p USING(id)'
        rows = con.execute(query).df()
    finally:
        con.close()
    rows['fetched_at'] = pd.to_datetime(rows['fetched_at'], errors='coerce', utc=True)
    for col in ['url', 'final_url', 'status', 'sha1_raw']:
        rows[col] = rows[col].fillna('').astype(str)
    rows['status'] = rows['status'].str.lower().str.strip()
    rows['sha1_raw'] = rows['sha1_raw'].str.lower().str.strip()
    return (rows.sort_values(['id', 'fetched_at'], ascending=[True, False], na_position='last', kind='stable')
            .drop_duplicates('id', keep='first').set_index('id', drop=False))

def _load_latest_ok_status(sample_ids, status_db=None):
    rows = _load_latest_status(sample_ids, status_db)
    return rows[rows['status'].eq('ok')]

GOLD_GATE_SOURCE = '"""Check a gold-set extraction run against NT\'s approval (reviews/gold_gate.json).\n\nTV never receives the gold answers, so the notebook cannot score itself.\nInstead NT scores a run, records a digest of its outputs, and the notebook\nproves it reproduces exactly that run before cleaning the 40K sample.\n\nThe digest covers what scoring depends on: text, block types and offsets,\nquality flags. JSON formatting does not matter.\n\nUsage:\n    python eval/gold_gate.py reviews/gold_gate.json \\\n        gold1=<run>/extract_gold1_latest gold2=... gold3=...\nIn a notebook:\n    from gold_gate import check_gate\n    check_gate(gate_json, {"gold1": dir1, ...}, postfilter_sha1, hints_sha1)\n"""\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nimport sys\nfrom pathlib import Path\n\nEXTRACTOR = "custom_cleaner_v3"\n\n\ndef outputs_digest(run_dir: Path) -> tuple[str, int]:\n    """sha1 over every <id>.json of the extractor, ordered by id."""\n    files = sorted((Path(run_dir) / EXTRACTOR).glob("*.json"), key=lambda p: int(p.stem))\n    h = hashlib.sha1()\n    for f in files:\n        doc = json.loads(f.read_text(encoding="utf-8"))\n        blocks = ",".join(f"{b.get(\'type\')}:{b.get(\'char_start\')}:{b.get(\'char_end\')}"\n                          for b in doc.get("blocks") or [])\n        flags = ",".join(sorted(doc.get("quality_flags") or []))\n        text_sha1 = hashlib.sha1((doc.get("text") or "").encode("utf-8")).hexdigest()\n        h.update(f"{f.stem}\\t{text_sha1}\\t{blocks}\\t{flags}\\n".encode("utf-8"))\n    return h.hexdigest(), len(files)\n\n\ndef check_gate(gate_path, run_dirs: dict, postfilter_sha1: str, hints_sha1: str) -> list[str]:\n    """Return a list of problems; empty means the run matches NT\'s approved run."""\n    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))\n    problems = []\n    if postfilter_sha1 != gate["postfilter_sha1"]:\n        problems.append(f"postfilter_sha1 {postfilter_sha1} != {gate[\'postfilter_sha1\']}")\n    if hints_sha1 != gate["container_hints_sha1"]:\n        problems.append(f"container_hints_sha1 {hints_sha1} != {gate[\'container_hints_sha1\']}")\n    for name, approved in gate["gold_sets"].items():\n        if name not in run_dirs:\n            problems.append(f"{name}: no run directory given")\n            continue\n        digest, n = outputs_digest(run_dirs[name])\n        if n != approved["docs"] or digest != approved["outputs_digest"]:\n            problems.append(f"{name}: {n} docs, digest {digest} "\n                            f"!= approved {approved[\'docs\']} docs, {approved[\'outputs_digest\']}")\n    return problems\n\n\ndef main() -> None:\n    sys.stdout.reconfigure(encoding="utf-8")\n    if len(sys.argv) < 2:\n        sys.exit(__doc__)\n    approved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["gold_sets"]\n    runs = dict(a.split("=", 1) for a in sys.argv[2:])\n    for name, d in runs.items():\n        digest, n = outputs_digest(Path(d))\n        same = name in approved and approved[name]["outputs_digest"] == digest\n        print(f"{name}: docs={n} outputs_digest={digest} {\'MATCH\' if same else \'DIFFERENT\'}")\n\n\nif __name__ == "__main__":\n    main()\n'
GOLD_GATE_APPROVAL_JSON = '{\n  "approved_by": "NT",\n  "approved_at": "2026-10-20",\n  "ready_for_40k": true,\n  "build": "2026-10-20-v25-gold-inheritance-audit-postfilter-v1.12",\n  "checked_run": "Drive reports/extract_gold{1,2,3}_latest, uploaded 2026-10-20 00:36 (same outputs as 00:24)",\n  "postfilter_sha1": "b949f9851a94fe894e62cd45be8c1dce4da0ef58",\n  "container_hints_sha1": "405ecc3099f7819cb6fb450992e1b7523334e2b3",\n  "rule": "every page of a domain: recall >= 0.99, junk <= 0.10, heading >= 0.9, required = 1.0; gold2 does not count verbatim author repeats (reviews/2026-10-18_v3_gold2.md section 3); pubmed is scored separately",\n  "gold_sets": {\n    "gold1": {"domains_passed": 32, "domains": 32, "docs": 43, "outputs_digest": "97ca02b70721334f0dab0f6102b2304d98195183"},\n    "gold2": {"domains_passed": 26, "domains": 26, "docs": 36, "outputs_digest": "fa9d8497e4bfe8322d1f4864d88314837e5902b0"},\n    "gold3": {"domains_passed": 24, "domains": 24, "docs": 24, "outputs_digest": "96efaf2ec257a596dc7c308e02098dcdc2421853"}\n  },\n  "digest": "eval/gold_gate.py outputs_digest(): sha1 over custom_cleaner_v3/<id>.json ordered by id, one line per doc: id, sha1(text), block type:char_start:char_end, sorted quality_flags",\n  "how_to_use": "Before cleaning the 40K sample, call eval/gold_gate.py check_gate(this file, {gold1: dir, gold2: dir, gold3: dir}, POSTFILTER_SHA1, CONTAINER_HINTS_SHA1). An empty list means the run reproduces the approved one. If the extractor output changes, NT re-scores and updates this file."\n}\n'
_gold_gate_namespace = {'__name__': '_notebook_gold_gate'}
exec(compile(GOLD_GATE_SOURCE, 'eval/gold_gate.py', 'exec'), _gold_gate_namespace)
check_gate = _gold_gate_namespace['check_gate']

def require_review_gold_pass():
    """Recompute approved output digests on every call; no gold answers needed."""
    candidates = [Path('reviews/gold_gate.json'), DATA_ROOT / 'reviews/gold_gate.json']
    gate_path = next((p for p in candidates if p.is_file()), None)
    if gate_path is None:
        gate_path = REPORT_DIR / '_embedded_gold_gate.json'
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(GOLD_GATE_APPROVAL_JSON, encoding='utf-8')
    summaries = {r['gold_set']: r for r in globals().get('_gold_run_summaries', [])}
    run_dirs = {name: Path(summaries.get(name, {}).get('output_root') or REPORT_DIR / f'extract_{name}_latest')
                for name in ('gold1', 'gold2', 'gold3')}
    try:
        problems = check_gate(gate_path, run_dirs, POSTFILTER_SHA1, CONTAINER_HINTS_SHA1)
    except Exception as exc:
        problems = [f'Cannot check gold gate: {type(exc).__name__}: {exc}']
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / 'gold_review_gate.json').write_text(json.dumps({
        'gate_path': str(gate_path), 'run_dirs': {k: str(v) for k,v in run_dirs.items()},
        'postfilter_sha1': POSTFILTER_SHA1, 'container_hints_sha1': CONTAINER_HINTS_SHA1,
        'ready_for_40k': not problems, 'problems': problems,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    if problems:
        raise RuntimeError('Gold khác bản NT duyệt 00h36; dừng 40K, báo NT chấm lại.\n' + '\n'.join(problems))
    print('Gold khớp bản NT duyệt 00h36: 32/32, 26/26, 24/24. Cho phép chạy 40K.')
    return problems


def resolve_existing_pilot_shards(shard_paths=None):
    """Choose four existing pilot shards; only inspect filenames, never regroup raw records."""
    if shard_paths:
        paths = _ordered_source_shards(shard_paths)
    else:
        candidates = []
        roots = {CRAWL_DIR, DOCS_DIR, DATA_ROOT}
        for root in roots:
            if not root.is_dir():
                continue
            for directory in root.iterdir():
                if directory.is_dir() and any(word in directory.name.lower() for word in ('pilot2', 'sample2')):
                    candidates.append(directory)
        if HTML_DIR.is_dir():
            candidates.append(HTML_DIR)
        groups = {}
        for directory in candidates:
            files = _ordered_source_shards(directory.glob('*.jsonl.zst'))
            if len(files) == 4:
                groups[tuple(str(p.resolve()) for p in files)] = files
        if len(groups) != 1:
            raise FileNotFoundError('Không tìm được duy nhất một nhóm bốn shard pilot đã gom. Điền PILOT2_SHARD_PATHS bằng đúng bốn file có sẵn; không cần gom lại shard.')
        paths = next(iter(groups.values()))
    if len(paths) != 4 or len({p.resolve() for p in paths}) != 4:
        raise ValueError('Cần đúng bốn shard pilot khác nhau đã gom sẵn.')
    for path in paths:
        if not path.is_file() or not path.name.endswith('.jsonl.zst'):
            raise FileNotFoundError(f'Shard pilot không hợp lệ: {path}')
    return paths

def regroup_pilot_with_120ask(source_paths, sample_csv, status_db):
    """Group latest approved raw bodies into four new pilot shards, including cloud 120ask."""
    import base64, io, zstandard as zstd
    require_review_gold_pass()
    sample = pd.read_csv(sample_csv)
    if len(sample) != 40000 or sample['id'].nunique() != 40000:
        raise ValueError('pilot2_sample.csv must contain 40,000 unique IDs')
    ids = set(sample['id'].astype(int))
    latest = _load_latest_status(ids, status_db)
    cloud_ids = set(sample.loc[sample['url'].map(lambda u: (urlparse(str(u)).hostname or '').endswith('120ask.com')), 'id'].astype(int))
    pending_cloud = [rid for rid in cloud_ids if rid not in latest.index or latest.loc[rid, 'status'] == 'pending']
    if pending_cloud:
        raise RuntimeError(f'120ask chưa crawl xong/DB chưa hợp nhất: {len(pending_cloud)} ID. Đồng bộ raw shard và status từ VPS lên Drive trước.')
    wanted = latest[latest['status'].eq('ok')].to_dict(orient='index')
    paths = _ordered_source_shards(source_paths)
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError('Thiếu source shard; thêm cả shard 120ask từ VPS trong PILOT2_SOURCE_PATHS.')
    target = CRAWL_DIR / ('pilot2_40k_' + str(time.time_ns()))
    staging = target.with_name(target.name + '.tmp')
    staging.mkdir(parents=True, exist_ok=False)
    seen = set()
    from contextlib import ExitStack
    with ExitStack() as stack:
        writers = [stack.enter_context(zstd.ZstdCompressor().stream_writer(
            stack.enter_context((staging / f'shard_{i:05d}.jsonl.zst').open('wb')))) for i in range(4)]
        for path in paths:
            with path.open('rb') as compressed, zstd.ZstdDecompressor().stream_reader(compressed) as reader, io.TextIOWrapper(reader, encoding='utf-8') as lines:
                for line in lines:
                    rec = json.loads(line)
                    rid = int(rec['id'])
                    if rid not in wanted or rid in seen or not rec.get('body_b64'):
                        continue
                    body = base64.b64decode(rec['body_b64'], validate=True)
                    if hashlib.sha1(body).hexdigest() != wanted[rid]['sha1_raw']:
                        continue
                    writers[rid % 4].write((json.dumps(rec, ensure_ascii=False) + '\n').encode('utf-8'))
                    seen.add(rid)
    missing = sorted(set(wanted) - seen)
    pd.DataFrame({'id': missing, 'reason': ['latest_ok_body_missing'] * len(missing)}).to_csv(staging / 'missing_latest_ok.csv', index=False)
    if missing:
        raise RuntimeError(f'Thiếu {len(missing)} latest-ok body (có thể 120ask chưa lên Drive); xem {staging}. Dừng trước 40K.')
    (staging / 'inventory.json').write_text(json.dumps({'sources': [str(p) for p in paths], 'matched': len(seen), '120ask_matched': len(cloud_ids & seen)}, indent=2), encoding='utf-8')
    staging.rename(target)
    return sorted(target.glob('shard_*.jsonl.zst'))


# %% CELL 26
# CHUẨN BỊ — chạy trước 40K bằng bốn shard pilot đã có, chưa chờ 120ask.
# Nếu không tự tìm được đúng thư mục, điền bốn đường dẫn tuyệt đối ở đây.
PILOT2_SHARD_PATHS = []
sample_csv = CRAWL_DIR / 'pilot2_sample.csv'
status_db = CRAWL_DIR / 'status.duckdb'
if not status_db.is_file():
    raise FileNotFoundError(f'Thiếu {status_db}; cần DB để giải thích ID thiếu và đối chiếu SHA-1.')
if not sample_csv.is_file():
    raise FileNotFoundError(sample_csv)

project_sample = pd.read_csv(sample_csv, usecols=['id', 'url'])
project_sample['id'] = project_sample['id'].astype('int64')
print(f'Project 40K manifest SHA256: {validate_project_pilot2_sample(project_sample)}')

gold_review_status = require_review_gold_pass()
shard_gold_parity = validate_shard_gold_parity()
# Mặc định dùng shard pilot cũ; đặt True khi đã đồng bộ status và shard 120ask từ VPS.
# Có thể thêm các đường dẫn shard cloud ngoài HTML_DIR vào danh sách này.
REGROUP_WITH_120ASK = False
PILOT2_SOURCE_PATHS = sorted(HTML_DIR.glob('shard_*.jsonl.zst'))
pilot_shards = (regroup_pilot_with_120ask(PILOT2_SOURCE_PATHS, sample_csv, status_db)
                if REGROUP_WITH_120ASK else resolve_existing_pilot_shards(PILOT2_SHARD_PATHS))
print(f'DATA_ROOT: {DATA_ROOT.resolve()}')
print('Chế độ: gom lại gồm 120ask' if REGROUP_WITH_120ASK else 'Chế độ: chạy 40K với shard pilot có sẵn; chưa chờ 120ask. ID thiếu vẫn có dòng rỗng và lý do trong sample2_missing.csv.')
for path in pilot_shards:
    print(f'  {path.resolve()} ({path.stat().st_size:,} bytes)')


# %% CELL 27
# CLEAN 40K — chạy sau Gold đạt và cell chuẩn bị; không tạo shard trung gian.
require_review_gold_pass()
if 'pilot_shards' not in globals():
    raise RuntimeError('Chạy cell chuẩn bị để chọn bốn shard pilot có sẵn trước.')
result_sample2 = run_sample2_extraction(
    shard_paths=pilot_shards, sample_csv=sample_csv, status_db=status_db,
    sample_limit=None, resume=False,
)
print(f'Parquet: {(DOCS_DIR / "docs_sample2.parquet").resolve()}')
print(f'Lý do thiếu: {(REPORT_DIR / "sample2_missing.csv").resolve()}')
print(f'Tóm tắt: {(REPORT_DIR / "sample2_extraction_summary.csv").resolve()}')
print('Gửi reports/extract_gold{1,2,3}_latest, gold_comparison_latest và sample2_* trên Drive cho NT duyệt.')
