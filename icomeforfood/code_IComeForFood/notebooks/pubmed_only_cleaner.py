"""PubMed-only cleaner embedded in the standalone Colab notebook.

The output follows the same schema-v2 contract as notebook 03.
"""

import hashlib
import re
from datetime import datetime
from html import unescape as html_unescape
from typing import Any, Dict, Optional


_ZW_RE = re.compile(r'[\u200b\u2060\ufeff]')
SAFE_TAG_RE = re.compile(r'</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>')


def normalize_ws(text: str) -> str:
    if not text:
        return ''
    return re.sub(r'\s+', ' ', _ZW_RE.sub('', text)).strip()


def parse_published_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = normalize_ws(str(value))
    if not raw:
        return None
    for cand in (raw[:10], raw):
        for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y',
                    '%m/%d/%Y', '%B %d, %Y', '%b %d, %Y'):
            try:
                return datetime.strptime(cand, fmt).date().isoformat()
            except ValueError:
                pass
    match = re.fullmatch(r'(\d{4})年(\d{1,2})月(\d{1,2})日?', raw)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)),
                            int(match.group(3))).date().isoformat()
        except ValueError:
            return None
    return None


def strip_html_tags_safe(text: str) -> str:
    return html_unescape(SAFE_TAG_RE.sub('', text))


def extract_pubmed_doc(pm_rec: Dict[str, Any], doc_id: int, url: str) -> Dict[str, Any]:
    title = normalize_ws(strip_html_tags_safe(pm_rec.get('title', '') or ''))
    if title.endswith('.'):
        title = title[:-1]
    blocks, text_parts = [], []
    char_cursor = 0

    def add_block(btype, text, level=None, role=None):
        nonlocal char_cursor
        cs, ce = char_cursor, char_cursor + len(text)
        char_cursor = ce + 2
        blocks.append({'type': btype, 'level': level, 'role': role, 'text': text,
                       'char_start': cs, 'char_end': ce})
        text_parts.append(text)

    if title:
        add_block('heading', title, level=1)
    abs_text = pm_rec.get('abstractText') or pm_rec.get('abstract') or ''
    if isinstance(abs_text, str) and abs_text:
        if '<' in abs_text:
            parts = re.split(r'<h4[^>]*>(.*?)</h4>', abs_text)
            p0 = normalize_ws(strip_html_tags_safe(parts[0]))
            if p0:
                add_block('paragraph', p0)
            for lbl_raw, cnt_raw in zip(parts[1::2], parts[2::2]):
                lbl = normalize_ws(strip_html_tags_safe(lbl_raw)).rstrip(':')
                cnt = normalize_ws(strip_html_tags_safe(cnt_raw))
                if cnt:
                    add_block('paragraph', f'{lbl}: {cnt}' if lbl else cnt)
        else:
            add_block('paragraph', normalize_ws(abs_text))
    keyword_list = pm_rec.get('keywordList') or {}
    kws = keyword_list.get('keyword', []) if isinstance(keyword_list, dict) else []
    if kws:
        add_block('paragraph', 'Keywords: ' + '; '.join(str(k) for k in kws) + '.')
    full_text = '\n\n'.join(text_parts)
    has_abstract = isinstance(abs_text, str) and bool(normalize_ws(strip_html_tags_safe(abs_text)))
    flags = [] if has_abstract else ['no_abstract']
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
