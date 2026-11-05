import json
import gzip
from pathlib import Path
import sys

import pandas as pd
import pytest
from lxml import html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from recover_pharmacity_api import matched_records
from recover_publisher_frontends import parse_flight, public_article, resolve_text, text_from_html
from crawl_local_fast import get_already_crawled_ids
from recover_commoncrawl_hits import decode_warc


def test_exact_link_match_does_not_remap_another_publishers_article():
    link = "https://pmc-ecm-healthblog.beta.pharmacity.io/topic/"
    post = dict(id=123, link=link, slug="topic", status="publish",
                title={"rendered": "Original title"}, content={"rendered": "<p>" + "source evidence " * 20 + "</p>"})
    assert matched_records([post], {link: 42})[link]["wordpress_id"] == 123
    assert matched_records([{**post, "link": "https://other.example/topic/"}], {link: 42}) == {}
    assert matched_records([{**post, "slug": "different"}], {link: 42}) == {}
    assert matched_records([{**post, "status": "draft"}], {link: 42}) == {}
    assert matched_records([{**post, "content": {"rendered": '<img src="image.webp">'}}], {link: 42}) == {}


def test_flight_utf8_byte_lengths_and_references_keep_chinese_and_vietnamese():
    content = "<p>Đây là nội dung 中文\nkhông được cắt sai byte.</p>"
    stream = f"a:T{len(content.encode('utf-8')):x}," + content + '\nb:{"content":{"rendered":"$a"}}\n'
    tree = html.fromstring("<html><script>self.__next_f.push(" + json.dumps([1, stream]) + ")</script></html>")
    records = parse_flight(tree)
    assert resolve_text(records["b"]["content"]["rendered"], records) == content


def test_generic_redirect_cannot_pass_publisher_identity_check():
    source = "https://pmc-ecm-healthblog.beta.pharmacity.io/topic/"
    tree = html.fromstring('<html><link rel="canonical" href="https://www.pharmacity.vn/topic.htm">'
                           '<h1>Topic</h1><div id="pmc-content-blog-detail">' + "article " * 20 + '</div></html>')
    assert public_article(tree, source, "https://www.pharmacity.vn/")[0] == ""
    assert public_article(tree, source, "https://www.pharmacity.vn/topic.htm")[0]


def test_old_dropped_api_miss_and_staging_flags_are_retryable(tmp_path):
    pd.DataFrame({"id": [1, 2, 3, 4], "status": ["dropped", "dropped", "dropped", "success"],
                  "reason": ["PMID not found in Europe PMC", "Server Staging/Beta Shut Down (HTTP 403 Nginx)",
                             "HTTP 404 Not Found", "Success"]}).to_parquet(tmp_path / "part.parquet")
    assert get_already_crawled_ids(str(tmp_path)) == {3, 4}


def test_malformed_image_attribute_cannot_hide_the_following_article():
    value = '<p><img alt=" height="800" src="image.webp" /></p><p><span new=" roman=" times=">Đoạn nội dung nguồn phải giữ nguyên.</span></p>'
    assert text_from_html(value) == 'Đoạn nội dung nguồn phải giữ nguyên.'


def test_warc_origin_identity_and_http_response_are_required():
    body = '<article>原始文章 nội dung gốc</article>'.encode()
    url = 'https://article.iiyi.com/detail/1.html'
    http = b'HTTP/1.1 200 OK\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
    record = b'WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: ' + url.encode() + b'\r\n\r\n' + http
    text, fields = decode_warc(gzip.compress(record), url, 'utf-8')
    assert text.encode() == body and fields['WARC-Target-URI'] == url
    with pytest.raises(ValueError, match='identity'):
        decode_warc(gzip.compress(record), 'https://article.iiyi.com/detail/2.html', 'utf-8')
    with pytest.raises(ValueError, match='HTTP 200'):
        decode_warc(gzip.compress(record.replace(b'200 OK', b'403 Forbidden')), url, 'utf-8')
