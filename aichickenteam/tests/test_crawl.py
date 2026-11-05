"""Offline tests for src/crawl.py and src/extractors (local HTTP server, no internet)."""
from __future__ import annotations

import asyncio
import http.server
import socketserver
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import crawl
from src.extractors import extract, extract_html
from src.extractors.generic import html_to_paras
from src.extractors.text import decode_html, detect_lang, join_paras, norm_para
from selectolax.parser import HTMLParser

ZH = "高血压患者应该低盐饮食，并且定期监测血压。如果出现头晕、胸闷等症状，应及时就医。" * 3


# ----------------------------------------------------------------------------- text utils

def test_decode_gbk_meta_and_header():
    html = f'<html><head><meta http-equiv="Content-Type" content="text/html; charset=gb2312"></head><body><p>{ZH}</p></body></html>'
    body = html.encode("gbk")
    txt, enc = decode_html(body)                     # meta only
    assert ZH in txt and enc == "gb18030"
    txt, enc = decode_html(body, "ISO-8859-1")       # wrong latin-1 default header is ignored
    assert ZH in txt
    txt, enc = decode_html(body, "gbk")
    assert ZH in txt


def test_decode_no_charset_falls_back():
    body = f"<html><body><p>{ZH}</p></body></html>".encode("gb18030")
    txt, _ = decode_html(body)
    assert ZH in txt
    vi = "<p>Bệnh tăng huyết áp cần được điều trị sớm.</p>".encode("utf-8")
    assert "huyết áp" in decode_html(vi)[0]


def test_norm_and_join():
    # decomposed Vietnamese -> NFC, whitespace collapsed, zero-width removed
    s = "Bệnh  ​ viêm\n\tgan"
    assert norm_para(s) == "Bệnh viêm gan"
    assert join_paras(["a  b", "", "a b", "c"]) == "a b\n\nc"


def test_detect_lang():
    assert detect_lang(ZH, "www.120ask.com") == "zh"
    assert detect_lang("Bệnh tăng huyết áp là gì và điều trị thế nào?", "x.com") == "vi"
    assert detect_lang("", "suckhoedoisong.vn") == "vi"
    assert detect_lang("", "www.familydoctor.com.cn") == "zh"


def test_registered_domain():
    assert crawl.registered_domain("baby.39.net") == "39.net"
    assert crawl.registered_domain("qy.familydoctor.com.cn") == "familydoctor.com.cn"
    assert crawl.registered_domain("phuyen.baodaklak.vn") == "baodaklak.vn"
    assert crawl.registered_domain("nhathuoclongchau.com.vn") == "nhathuoclongchau.com.vn"


def test_html_to_paras():
    t = HTMLParser('<div>a<b>bold</b> x<br>y<p>para <a>link</a></p><table><tr><td>k</td><td>v</td></tr></table>'
                   '<span style="display:none">hidden</span><script>var x</script></div>')
    assert html_to_paras(t.css_first("div")) == ["abold x", "y", "para link", "k | v"]


# ----------------------------------------------------------------------------- extractors

def test_120ask_keeps_question_and_answers():
    html = f"""<html><body><div class="top_nav">男科 妇科 菜单</div>
    <div class="b_askti"><h1>肾为什么疼</h1></div>
    <div class="b_askcont"><p class="crazy_new">健康咨询描述： 发病两年了</p></div>
    <div class="b_answerli"><div class="b_answertl">张医生 主治医师</div>
      <div class="crazy_new">病情分析：<br>{ZH}<br>指导意见：<br>多喝水。</div></div>
    <div class="b_answerli"><div class="crazy_new">&lt;P&gt;第二位医生的回答。&lt;/P&gt;</div></div>
    <div class="b_hotbox">其他类似问题 肾结石</div></body></html>"""
    r = extract_html(html, "http://www.120ask.com/question/1.htm")
    assert r["method"] == "site:ask120" and r["title"] == "肾为什么疼"
    assert "发病两年了" in r["text"] and "多喝水。" in r["text"] and "第二位医生的回答。" in r["text"]
    assert "菜单" not in r["text"] and "其他类似问题" not in r["text"] and "<P>" not in r["text"]
    assert r["lang"] == "zh"


def test_ask39_answers():
    html = """<html><body><div class="ask_cont"><h1 class="ask_tit">乳核散结片副作用大吗</h1>
    <div class="ask_hid"><p class="txt_ms">吃了一个月</p></div></div>
    <div class="selected"><div class="sele_all"><div class="doctor_all">柯怀 副主任医师</div>
    <p class="sele_txt"></p><p>副作用小。</p><p>注意肝功能。</p>
    <div class="doc_t_strip"><p class="doc_time">2025-12-20 11:48</p>举报</div></div></div>
    <div class="relevant_ask">相关问题 xxx</div></body></html>"""
    r = extract_html(html, "https://ask.39.net/question/1.html")
    assert r["method"] == "site:ask39"
    assert r["text"] == "吃了一个月\n\n副作用小。\n\n注意肝功能。"


def test_generic_trafilatura_article():
    paras = [f"第{i}段。" + ZH for i in range(5)]
    html = ("<html><head><title>测试文章_某某健康网</title></head><body><nav>首页 新闻 疾病</nav><article>"
            + "".join(f"<p>{p}</p>" for p in paras) + "</article><footer>版权所有</footer></body></html>")
    r = extract(html.encode("utf-8"), "https://example.cn/a/1.html")
    assert r["title"] == "测试文章"
    assert r["text"].split("\n\n") == paras


# ----------------------------------------------------------------------------- storage

def test_shard_writer_and_load_docs(tmp_path: Path):
    state = crawl.State(tmp_path / "s.sqlite")
    w = crawl.ShardWriter(tmp_path / "docs", flush_every=2, on_flush=state.put_many)
    for i in range(5):
        doc = {"doc_id": str(i), "source": "corpus", "lang": "zh", "url": f"u{i}", "title": "t",
               "text": "x" * 40, "fetched_at": "2026-10-01T00:00:00+00:00"}
        w.add(doc, {"id": i, "url": f"u{i}", "host": "h", "status": "ok", "n_chars": 40})
    w.add(None, {"id": 99, "url": "u99", "host": "h", "status": "net_error", "error": "timeout"})
    assert len(list((tmp_path / "docs").glob("corpus-*.parquet"))) == 2   # 2 auto flushes
    w.flush()
    files = sorted((tmp_path / "docs").glob("corpus-*.parquet"))
    assert [f.name for f in files] == ["corpus-00000.parquet", "corpus-00001.parquet", "corpus-00002.parquet"]
    assert list(crawl.load_docs(files[1], columns=["text"]))[0].keys() == {"doc_id", "text"}
    assert pq.read_schema(files[0]).names == list(crawl.DOC_COLS)
    docs = list(crawl.load_docs(tmp_path / "docs"))
    assert [d["doc_id"] for d in docs] == ["0", "1", "2", "3", "4"]
    # a re-crawled doc (newer shard) wins over the old copy
    w.add(dict(doc, doc_id="1", text="new" * 20), {"id": 1, "url": "u1", "host": "h", "status": "ok"})
    w.flush()
    docs = {d["doc_id"]: d for d in crawl.load_docs(tmp_path / "docs")}
    assert len(docs) == 5 and docs["1"]["text"] == "new" * 20
    # done ids are skipped, transient failures retried until max_runs
    assert state.skip_ids(max_runs=3, retry_failed=False) == {0, 1, 2, 3, 4}
    state.put_many([{"id": 99, "url": "u99", "host": "h", "status": "net_error"}] * 2)
    assert 99 in state.skip_ids(max_runs=3, retry_failed=False)
    assert 99 not in state.skip_ids(max_runs=3, retry_failed=True)


def _corpus(tmp_path: Path) -> dict:
    ids = list(range(1, 41))
    urls = [f"https://www.120ask.com/q/{i}.htm" if i % 2 else f"https://suckhoedoisong.vn/a-{i}.htm" for i in ids]
    pq.write_table(pa.table({"id": ids, "url": urls}), tmp_path / "corpus.parquet")
    cfg = dict(crawl.DEFAULTS)
    cfg["corpus"] = str(tmp_path / "corpus.parquet")
    return cfg


def test_select_targets(tmp_path: Path):
    cfg = _corpus(tmp_path)
    t = crawl.select_targets(cfg, priority="vi-first")
    assert len(t) == 40 and t[0][2] == "suckhoedoisong.vn"
    t = crawl.select_targets(cfg, sample_per_domain=3, seed=1)
    assert len(t) == 6
    t = crawl.select_targets(cfg, domains="120ask.com", shard="1/4")
    assert t and all(i % 4 == 1 and "120ask" in u for i, u, _ in t)
    t = crawl.select_targets(cfg, limit=4, priority="none")
    assert len({h for _, _, h in t}) == 2                                   # interleaved across hosts
    (tmp_path / "ids.txt").write_text("7\n3\n999\n")
    t = crawl.select_targets(cfg, ids_file=str(tmp_path / "ids.txt"), priority="none")
    assert [i for i, _, _ in t] == [7, 3]
    t = crawl.select_targets(cfg, skip={1, 3, 5}, domains="120ask.com")
    assert 1 not in {i for i, _, _ in t}


# ----------------------------------------------------------------------------- end-to-end (local server)

GBK_PAGE = ('<html><head><meta charset="gbk"><title>高血压饮食_健康网</title></head><body><nav>导航</nav>'
            f'<div class="content"><h1>高血压饮食</h1>{"".join(f"<p>{ZH}{i}</p>" for i in range(4))}</div></body></html>'
            ).encode("gbk")


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/gbk"):
            self._send(200, GBK_PAGE, "text/html")            # no charset in header
        elif self.path.startswith("/cookie"):
            if "D1N=abc" in (self.headers.get("Cookie") or ""):
                self._send(200, f"<html><body><article><p>{ZH}</p><p>{ZH}</p></article></body></html>".encode(), "text/html; charset=utf-8")
            else:
                self._send(200, b'<html><body><script>document.cookie="D1N=abc"+"; path=/";window.location.reload(true);</script></body></html>', "text/html")
        elif self.path.startswith("/pdf"):
            self._send(200, b"%PDF-1.4", "application/pdf")
        else:
            self._send(404, b"not found", "text/html")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("backend", ["curl_cffi", "httpx"])
def test_crawler_end_to_end(tmp_path: Path, server: str, backend: str):
    cfg = dict(crawl.DEFAULTS, backend=backend, per_domain_delay=0.0, retries=1, backoff_base=0.01,
               extract_workers=1, out_dir=str(tmp_path / "docs"), state_db=str(tmp_path / "s.sqlite"))
    targets = [(1, f"{server}/gbk/1", "127.0.0.1"), (2, f"{server}/cookie/2", "127.0.0.1"),
               (3, f"{server}/missing", "127.0.0.1"), (4, f"{server}/pdf", "127.0.0.1")]
    state = crawl.State(Path(cfg["state_db"]))
    writer = crawl.ShardWriter(Path(cfg["out_dir"]), 100, on_flush=state.put_many)
    info = asyncio.run(crawl.Crawler(cfg, targets, state, writer).run())
    assert info["done"] == 4
    st = dict(state.db.execute("SELECT id, status FROM crawl"))
    assert st == {1: "ok", 2: "ok", 3: "gone", 4: "not_html"}
    docs = {d["doc_id"]: d for d in crawl.load_docs(Path(cfg["out_dir"]))}
    assert set(docs) == {"1", "2"}
    d = docs["1"]
    assert d["title"] == "高血压饮食" and d["lang"] == "zh" and d["source"] == "corpus"
    assert d["text"].count("\n\n") == 3 and ZH in d["text"] and "导航" not in d["text"]
    assert ZH in docs["2"]["text"]
