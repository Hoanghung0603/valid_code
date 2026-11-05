"""Lấy text PubMed từ NCBI cho 240.000 URL pubmed trong links_corpus_vi_en_zh.

Hai backend (cùng API NCBI chính thức, cùng tool/email/nhịp <=2 request/giây, resume theo part):
- efetch (mặc định): E-utilities efetch POST, 200 PMID/lô, parse XML PubmedArticle.
- ctxp: Literature Citation Export API https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/?format=ris,
  GET batch ~100 PMID/lô, parse RIS (T1/AB/LA/Y1/KW) - dùng khi eutils bị chặn IP (misuse redirect).

- map: trích PMID từ URL (vd https://pubmed.ncbi.nlm.nih.gov/12345678/), ghi pmid_map.jsonl {id, pmid},
  báo số URL không trích được PMID.
- fetch: retry backoff với 429/5xx; ghi part theo lô -> resume được (part đã có thì bỏ qua).
  NCBI trả lỗi liên tục thì dừng và báo.
- merge: nối các part -> pubmed_text.jsonl.zst {id, pmid, title, abstract, lang, year, mesh} + thống kê.

Chạy trên b660 (Python 3.10): D:\\r2ai\\.venv\\Scripts\\python.exe scripts/pubmed_fetch.py run --backend ctxp ...
"""
import argparse
import http.client
import json
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EUTILS_HOST = "eutils.ncbi.nlm.nih.gov"
EUTILS_PATH = "/entrez/eutils/efetch.fcgi"
CTXP_HOST = "api.ncbi.nlm.nih.gov"
CTXP_PATH = "/lit/ctxp/v1/pubmed/"
TOOL = "r2ai_vibiomir"
EMAIL = __import__("os").environ.get("NCBI_EMAIL", "<NCBI_EMAIL>")
BATCH = 200
BATCH_CTXP = 100
MIN_INTERVAL = 0.55  # giây giữa 2 request bắt đầu -> <= 2 request/giây
RETRY_BACKOFF = [2.0, 4.0, 8.0, 16.0, 30.0]
MAX_BATCH_FAILS = 3  # số lô liên tiếp thất bại -> coi là NCBI chặn, dừng

_PUBMED_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_YEAR = re.compile(r"(19|20)\d{2}")


def extract_pmid(url: str):
    """Trích PMID từ URL PubMed; None nếu không khớp."""
    m = _PUBMED_URL.search(url or "")
    return int(m.group(1)) if m else None


def join_abstract(parts: list) -> str:
    """parts: [(Label, text)] -> abstract; đoạn có Label được tiền tố 'Label: '."""
    out = []
    for label, text in parts:
        t = (text or "").strip()
        if not t:
            continue
        out.append(f"{label.strip()}: {t}" if label and label.strip() else t)
    return "\n".join(out)


def _text(el) -> str:
    return "".join(el.itertext()) if el is not None else ""


def _find_year(cit) -> str:
    """Năm PubDate (Year, MedlineDate '2021 Jan-Feb', ArticleDate); '' nếu không có."""
    for path in ("Article/Journal/JournalIssue/PubDate/Year",
                 "Article/ArticleDate/Year"):
        el = cit.find(path)
        if el is not None and (el.text or "").strip():
            return el.text.strip()
    for path in ("Article/Journal/JournalIssue/PubDate/MedlineDate",):
        el = cit.find(path)
        if el is not None:
            m = _YEAR.search(el.text or "")
            if m:
                return m.group(0)
    return ""


def parse_pubmed_xml(data: bytes) -> dict:
    """XML efetch -> {pmid: {pmid, title, abstract, lang, year, mesh}}."""
    root = ET.fromstring(data)
    out = {}
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        pmid_el = cit.find("PMID")
        if pmid_el is None or not (pmid_el.text or "").strip():
            continue
        pmid = int(pmid_el.text.strip())
        article = cit.find("Article")
        title = _text(article.find("ArticleTitle")).strip() if article is not None else ""
        parts = []
        lang = ""
        if article is not None:
            ab = article.find("Abstract")
            if ab is not None:
                for at in ab.findall("AbstractText"):
                    parts.append((at.get("Label") or "", _text(at)))
            lg = article.find("Language")
            if lg is not None:
                lang = (lg.text or "").strip()
        mesh = [m for m in (_text(d).strip() for d in cit.iter("DescriptorName")) if m]
        out[pmid] = {
            "pmid": pmid,
            "title": title,
            "abstract": join_abstract(parts),
            "lang": lang,
            "year": _find_year(cit),
            "mesh": mesh,
        }
    return out


_RIS_TAG = re.compile(r"^([A-Z][A-Z0-9])  - (.*)$")


def parse_ris(data: bytes) -> dict:
    """RIS của ctxp (format=ris) -> {pmid: {pmid, title, abstract, lang, year, mesh}}.

    T1=title (TT=tiêu đề dịch, bỏ), AB=abstract (nối nhiều đoạn), LA=ngôn ngữ, Y1=năm, KW=MeSH, AN=PMID.
    """
    text = data.decode("utf-8", errors="replace")
    out = {}
    cur = None
    cur_tag = None
    for line in text.splitlines():
        m = _RIS_TAG.match(line)
        if m:
            tag, val = m.group(1), m.group(2).strip()
            cur_tag = tag
            if tag == "TY":
                cur = {"pmid": None, "title": "", "abstract": "", "lang": "", "year": "", "mesh": [],
                       "_ab": []}
            elif cur is None:
                continue
            elif tag in ("T1", "BT") and not cur["title"]:
                cur["title"] = val
            elif tag == "AB":
                cur["_ab"].append(val)
            elif tag == "LA" and not cur["lang"]:
                cur["lang"] = val
            elif tag == "Y1" and not cur["year"]:
                m2 = _YEAR.search(val)
                cur["year"] = m2.group(0) if m2 else val[:4]
            elif tag == "KW" and val:
                cur["mesh"].append(val)
            elif tag == "AN" and val.isdigit():
                cur["pmid"] = int(val)
            elif tag == "ER":
                if cur is not None and cur["pmid"] is not None:
                    cur["abstract"] = "\n".join(cur.pop("_ab"))
                    out[cur["pmid"]] = cur
                cur = None
                cur_tag = None
        elif cur is not None and cur_tag in ("AB", "T1") and line.startswith("      "):
            if cur_tag == "AB":  # dòng ngắt của giá trị dài
                cur["_ab"].append(line.strip())
            elif not cur["title"]:
                cur["title"] = line.strip()
    return out


class EfetchClient:
    """1 kết nối HTTPS tới E-utilities; POST efetch theo lô PMID; retry backoff 429/5xx."""

    def __init__(self, host: str = EUTILS_HOST, path: str = EUTILS_PATH,
                 tool: str = TOOL, email: str = EMAIL, opener=None, retry_backoff=None):
        self.host, self.path, self.tool, self.email = host, path, tool, email
        self.opener = opener  # inject cho test: callable(request) -> bytes
        self.retry_backoff = RETRY_BACKOFF if retry_backoff is None else retry_backoff
        self.conn = None
        self.last_req = 0.0
        self.n_req = 0

    def _body(self, pmids) -> bytes:
        return urllib.parse.urlencode({
            "db": "pubmed", "id": ",".join(str(p) for p in pmids),
            "retmode": "xml", "tool": self.tool, "email": self.email,
        }).encode("ascii")

    def _post_once(self, pmids) -> bytes:
        if self.opener is not None:
            return self.opener(self._body(pmids))
        if self.conn is None:
            self.conn = http.client.HTTPSConnection(self.host, timeout=120)
        headers = {"Content-Type": "application/x-www-form-urlencoded",
                   "User-Agent": f"{self.tool}/1.0"}
        self.conn.request("POST", self.path, self._body(pmids), headers)
        resp = self.conn.getresponse()
        status, data = resp.status, resp.read()
        if status == 429 or status >= 500:
            raise TransientHTTP(status)
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {data[:200]!r}")
        return data

    def fetch(self, pmids) -> bytes:
        """POST một lô; giữ nhịp <= MIN_INTERVAL giây/request; retry TransientHTTP/lỗi mạng."""
        last_err = None
        for attempt, backoff in enumerate([0.0] + list(self.retry_backoff)):
            if backoff:
                print(f"  retry sau {backoff:.0f}s ({last_err})", flush=True)
                time.sleep(backoff)
            wait = MIN_INTERVAL - (time.monotonic() - self.last_req)
            if wait > 0:
                time.sleep(wait)
            self.last_req = time.monotonic()
            self.n_req += 1
            try:
                return self._post_once(pmids)
            except (TransientHTTP, OSError, http.client.HTTPException) as e:
                last_err = e
                if self.conn is not None:
                    try:
                        self.conn.close()
                    except OSError:
                        pass
                    self.conn = None
        raise RuntimeError(f"efetch thất bại sau {len(self.retry_backoff) + 1} lần: {last_err}")

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None


class TransientHTTP(Exception):
    """429/5xx - thử lại được."""


class CtxpClient:
    """GET Literature Citation Export API (api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/?format=ris).

    Batch nhiều PMID (phẩy); giữ nhịp <= MIN_INTERVAL; retry TransientHTTP/lỗi mạng; cùng thông số với EfetchClient.
    """

    def __init__(self, host: str = CTXP_HOST, path: str = CTXP_PATH, opener=None, retry_backoff=None):
        self.host, self.path = host, path
        self.opener = opener
        self.retry_backoff = RETRY_BACKOFF if retry_backoff is None else retry_backoff
        self.conn = None
        self.last_req = 0.0
        self.n_req = 0

    def _url(self, pmids) -> str:
        return f"{self.path}?format=ris&id=" + ",".join(str(p) for p in pmids)

    def _get_once(self, pmids) -> bytes:
        if self.opener is not None:
            return self.opener(self._url(pmids))
        if self.conn is None:
            self.conn = http.client.HTTPSConnection(self.host, timeout=120)
        self.conn.request("GET", self._url(pmids), headers={"User-Agent": f"{TOOL}/1.0"})
        resp = self.conn.getresponse()
        status, data = resp.status, resp.read()
        if status == 429 or status >= 500:
            raise TransientHTTP(status)
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {data[:200]!r}")
        return data

    def fetch(self, pmids) -> bytes:
        last_err = None
        for backoff in [0.0] + list(self.retry_backoff):
            if backoff:
                print(f"  retry sau {backoff:.0f}s ({last_err})", flush=True)
                time.sleep(backoff)
            wait = MIN_INTERVAL - (time.monotonic() - self.last_req)
            if wait > 0:
                time.sleep(wait)
            self.last_req = time.monotonic()
            self.n_req += 1
            try:
                return self._get_once(pmids)
            except (TransientHTTP, OSError, http.client.HTTPException) as e:
                last_err = e
                if self.conn is not None:
                    try:
                        self.conn.close()
                    except OSError:
                        pass
                    self.conn = None
        raise RuntimeError(f"ctxp thất bại sau {len(self.retry_backoff) + 1} lần: {last_err}")

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None


def write_map(links_path: str, outdir: Path) -> tuple:
    """Đọc parquet (id, url) -> pmid_map.jsonl {id, pmid}; trả (n_url, n_pm, n_fail)."""
    import pandas as pd
    df = pd.read_parquet(links_path)
    outdir.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    rows = []
    for _i, r in df.iterrows():
        pm = extract_pmid(r["url"])
        if pm is None:
            n_fail += 1
            continue
        rows.append({"id": int(r["id"]), "pmid": pm})
    rows.sort(key=lambda x: x["id"])
    with (outdir / "pmid_map.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"map: {len(df)} URL, {len(rows)} trích được PMID, {n_fail} không trích được", flush=True)
    return len(df), len(rows), n_fail


def load_map(outdir: Path) -> list:
    rows = []
    with (outdir / "pmid_map.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def part_file(parts_dir: Path, bi: int) -> Path:
    return parts_dir / f"batch_{bi:05d}.jsonl"


def fetch_all(outdir: Path, batch: int = BATCH, limit_batches: int = 0,
              client=None, parser=None) -> dict:
    """Fetch mọi lô (resume theo part); trả thống kê. Dừng nếu NCBI trả lỗi liên tục."""
    rows = load_map(outdir)
    parts_dir = outdir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_batches = (len(rows) + batch - 1) // batch
    if limit_batches:
        n_batches = min(n_batches, limit_batches)
    own = client is None
    if own:
        client = EfetchClient()
    if parser is None:
        parser = parse_pubmed_xml
    t0 = time.time()
    stats = {"n_pm": len(rows), "n_batches": n_batches, "done": 0, "skipped": 0,
             "returned": 0, "missing": 0, "fail_stopped": False}
    streak = 0
    try:
        for bi in range(n_batches):
            pf = part_file(parts_dir, bi)
            chunk = rows[bi * batch:(bi + 1) * batch]
            if pf.exists():
                stats["skipped"] += 1
                continue
            pmids = [r["pmid"] for r in chunk]
            try:
                data = client.fetch(pmids)
            except RuntimeError as e:
                streak += 1
                print(f"lô {bi}: {e}", flush=True)
                if streak >= MAX_BATCH_FAILS:
                    stats["fail_stopped"] = True
                    print(f"DỪNG: {streak} lô liên tiếp thất bại (nhiều khả năng NCBI chặn)", flush=True)
                    break
                continue
            streak = 0
            recs = parser(data)
            got = 0
            with pf.open("w", encoding="utf-8") as f:
                for r in chunk:
                    rec = recs.get(r["pmid"])
                    if rec is None:
                        stats["missing"] += 1
                        f.write(json.dumps({"id": r["id"], "pmid": r["pmid"], "title": "",
                                            "abstract": "", "lang": "", "year": "", "mesh": [],
                                            "returned": False}) + "\n")
                    else:
                        got += 1
                        stats["returned"] += 1
                        rec["id"] = r["id"]
                        rec["returned"] = True
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["done"] += 1
            el = time.time() - t0
            if stats["done"] % 10 == 0 or stats["done"] == n_batches:
                print(f"lô {stats['done']}/{n_batches} (bỏ {stats['skipped']}), "
                      f"{stats['returned']} bài, {stats['missing']} thiếu, {el:.0f}s, "
                      f"{client.n_req} request", flush=True)
    finally:
        if own:
            client.close()
    stats["elapsed"] = round(time.time() - t0, 1)
    return stats


def merge_parts(outdir: Path) -> dict:
    """Nối parts -> pubmed_text.jsonl.zst; trả thống kê."""
    import zstandard
    parts_dir = outdir / "parts"
    files = sorted(parts_dir.glob("batch_*.jsonl"))
    n = with_ab = without = missing = 0
    len_ab = 0
    t0 = time.time()
    out_path = outdir / "pubmed_text.jsonl.zst"
    cctx = zstandard.ZstdCompressor(level=10)
    with out_path.open("wb") as raw:
        with cctx.stream_writer(raw) as w:
            for pf in files:
                for line in pf.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    n += 1
                    if not r.get("returned", True):
                        missing += 1
                    ab = r.get("abstract") or ""
                    if ab.strip():
                        with_ab += 1
                        len_ab += len(ab)
                    else:
                        without += 1
                    rec = {"id": r["id"], "pmid": r["pmid"], "title": r.get("title") or "",
                           "abstract": ab, "lang": r.get("lang") or "",
                           "year": r.get("year") or "", "mesh": r.get("mesh") or []}
                    w.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    stats = {"n_parts": len(files), "n_docs": n, "returned": n - missing, "missing": missing,
             "with_abstract": with_ab, "without_abstract": without,
             "avg_abstract_chars": round(len_ab / with_ab, 1) if with_ab else 0,
             "elapsed": round(time.time() - t0, 1), "out": str(out_path),
             "out_bytes": out_path.stat().st_size}
    (outdir / "fetch_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"merge: {n} bản ghi ({len(files)} part), {with_ab} có abstract, {without} không abstract, "
          f"{missing} PMID không trả về, TB abstract {stats['avg_abstract_chars']} ký tự, "
          f"{stats['elapsed']}s -> {out_path}", flush=True)
    return stats


def make_backend(name: str):
    """(client, parser, batch_mặc_định) theo backend."""
    if name == "ctxp":
        return CtxpClient(), parse_ris, BATCH_CTXP
    return EfetchClient(), parse_pubmed_xml, BATCH


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("map")
    m.add_argument("--links", required=True)
    m.add_argument("--outdir", required=True)
    f = sp.add_parser("fetch")
    f.add_argument("--outdir", required=True)
    f.add_argument("--batch", type=int, default=0, help="0 = theo backend (efetch 200, ctxp 100)")
    f.add_argument("--limit-batches", type=int, default=0)
    f.add_argument("--backend", choices=["efetch", "ctxp"], default="efetch")
    mg = sp.add_parser("merge")
    mg.add_argument("--outdir", required=True)
    r = sp.add_parser("run", help="map + fetch + merge")
    r.add_argument("--links", required=True)
    r.add_argument("--outdir", required=True)
    r.add_argument("--batch", type=int, default=0)
    r.add_argument("--limit-batches", type=int, default=0)
    r.add_argument("--backend", choices=["efetch", "ctxp"], default="efetch")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    t0 = time.time()
    if a.cmd in ("map", "run"):
        write_map(a.links, outdir)
    if a.cmd in ("fetch", "run"):
        client, parser, batch_def = make_backend(a.backend)
        st = fetch_all(outdir, a.batch or batch_def, a.limit_batches, client, parser)
        client.close()
        print(f"fetch ({a.backend}): {st}", flush=True)
    if a.cmd in ("merge", "run"):
        merge_parts(outdir)
    print(f"tổng {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
