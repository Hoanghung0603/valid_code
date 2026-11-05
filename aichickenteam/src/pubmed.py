"""English branch: PubMed candidate retrieval (NCBI E-utilities + Europe PMC) with sqlite cache.

Public API (see docs/INTERFACES.md):
    search(queries, retmax=100) -> list[str]          fused (RRF) PMID ranking over queries/sources
    fetch(pmids) -> list[Doc]                           title + abstract docs (cached, batched EFetch)
    candidates_for_plan(plan) -> list[Doc]              uses plan["pubmed"] + plan["en"]
    asearch / afetch / acandidates_for_plan             asyncio wrappers (run in worker threads)

CLI (prefetch all plans, resumable):
    python -m src.pubmed --plans data/cache/query_plans.jsonl --out data/docs/pubmed.parquet

Env: NCBI_API_KEY (10 req/s instead of 3), NCBI_EMAIL (sent as `email` param if set).
Notes on query writing / relaxation: docs/pubmed_notes.md
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import httpx

log = logging.getLogger("pubmed")

ROOT = Path(__file__).resolve().parent.parent
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULTS: dict[str, Any] = {
    "cache_db": "data/cache/pubmed.sqlite",
    "tool": "r2ai2026",
    "sources": ["ncbi", "europepmc"],  # candidate sources used by search()
    "retmax": 100,                       # per query per source
    "esearch_page": 20,                  # Best Match is only reliable for small pages (see esearch())
    "esearch_max_depth": 60,             # max PMIDs per ESearch query (pages of 20 -> 3 requests)
    "relax_depth": 20,                   # PMIDs per relaxed variant (1 request each)
    "min_hits": 20,                      # relax an ESearch query when count < min_hits
    "relax": True,
    "preclean": True,                    # strip stopwords/punctuation/measurements from NL queries first
    "max_relax_requests": 8,             # extra ESearch calls per query at most
    "epmc_from": ["en"],                 # plan fields sent to Europe PMC (boolean PubMed syntax is not portable)
    "epmc_has_abstract": True,
    "rrf_k": 60,
    "weights": {"ncbi": 1.0, "europepmc": 0.7, "elink": 0.5},
    "max_candidates": 300,               # docs returned by candidates_for_plan
    "fetch_batch": 200,
    "keep_no_abstract": False,
    "elink": False,                      # related-articles expansion
    "elink_top": 5,                      # expand from top-N fused hits
    "elink_per_hit": 20,
    "timeout": 60,
    "max_retries": 6,
    "inner_workers": 6,                  # parallel requests inside one call
    "workers": 4,                        # CLI: plans processed concurrently
    "epmc_rps": 8.0,
    "ncbi_rps": None,                    # None -> 2.0 (no key) / 7.5 (NCBI_API_KEY); margin avoids 429s
}


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    try:
        import yaml  # type: ignore

        with open(p, encoding="utf-8") as f:
            block = (yaml.safe_load(f) or {}).get("pubmed") or {}
        for k, v in block.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except FileNotFoundError:
        pass
    except Exception as e:  # pragma: no cover
        log.warning("config load failed (%s); using defaults", e)
    return cfg


# --------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Thread-safe evenly spaced limiter: each acquire() reserves the next free slot."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalize(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


# --------------------------------------------------------------------------- cache
class Cache:
    def __init__(self, path: str | os.PathLike):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._lock = threading.Lock()
        self.con = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript(
            """
            CREATE TABLE IF NOT EXISTS search(key TEXT PRIMARY KEY, retmax INT, count INT, pmids TEXT, extra TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS docs(pmid TEXT PRIMARY KEY, has_abstract INT, doc TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS missing(pmid TEXT PRIMARY KEY, ts REAL);
            CREATE TABLE IF NOT EXISTS elink(pmid TEXT PRIMARY KEY, related TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS plans(plan_id TEXT PRIMARY KEY, phash TEXT, pmids TEXT, scores TEXT, ts REAL);
            """
        )
        self.con.commit()

    def _q(self, sql: str, args: tuple = ()) -> list:
        with self._lock:
            return self.con.execute(sql, args).fetchall()

    def _w(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self._lock:
            self.con.executemany(sql, rows)
            self.con.commit()

    # search
    def get_search(self, key: str, retmax: int):
        r = self._q("SELECT retmax, count, pmids, extra FROM search WHERE key=?", (key,))
        if not r:
            return None
        smax, count, pmids, extra = r[0]
        pmids = json.loads(pmids)
        if smax >= retmax or len(pmids) >= count:
            return pmids[:retmax], count, json.loads(extra or "{}")
        return None

    def put_search(self, key: str, retmax: int, count: int, pmids: list[str], extra: dict | None = None):
        self._w("INSERT OR REPLACE INTO search VALUES (?,?,?,?,?,?)",
                [(key, retmax, count, json.dumps(pmids), json.dumps(extra or {}), time.time())])

    # docs
    def get_docs(self, pmids: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i in range(0, len(pmids), 900):
            chunk = pmids[i:i + 900]
            q = ",".join("?" * len(chunk))
            for pmid, doc in self._q(f"SELECT pmid, doc FROM docs WHERE pmid IN ({q})", tuple(chunk)):
                out[pmid] = json.loads(doc)
        return out

    def get_missing(self, pmids: list[str]) -> set[str]:
        out: set[str] = set()
        for i in range(0, len(pmids), 900):
            chunk = pmids[i:i + 900]
            q = ",".join("?" * len(chunk))
            out.update(r[0] for r in self._q(f"SELECT pmid FROM missing WHERE pmid IN ({q})", tuple(chunk)))
        return out

    def put_docs(self, docs: list[dict]):
        now = time.time()
        self._w("INSERT OR REPLACE INTO docs VALUES (?,?,?,?)",
                [(d["doc_id"], int(d.get("has_abstract", False)), json.dumps(d, ensure_ascii=False), now) for d in docs])

    def put_missing(self, pmids: Iterable[str]):
        now = time.time()
        self._w("INSERT OR REPLACE INTO missing VALUES (?,?)", [(p, now) for p in pmids])

    # elink
    def get_elink(self, pmids: list[str]) -> dict[str, list]:
        out = {}
        for p in pmids:
            r = self._q("SELECT related FROM elink WHERE pmid=?", (p,))
            if r:
                out[p] = json.loads(r[0][0])
        return out

    def put_elink(self, rel: dict[str, list]):
        now = time.time()
        self._w("INSERT OR REPLACE INTO elink VALUES (?,?,?)", [(k, json.dumps(v), now) for k, v in rel.items()])

    # plans (CLI resumability)
    def get_plan(self, plan_id: str, phash: str):
        r = self._q("SELECT pmids, scores FROM plans WHERE plan_id=? AND phash=?", (plan_id, phash))
        return (json.loads(r[0][0]), json.loads(r[0][1])) if r else None

    def put_plan(self, plan_id: str, phash: str, pmids: list[str], scores: list[float]):
        self._w("INSERT OR REPLACE INTO plans VALUES (?,?,?,?,?)",
                [(plan_id, phash, json.dumps(pmids), json.dumps(scores), time.time())])

    def all_docs(self) -> Iterable[dict]:
        for (doc,) in self._q("SELECT doc FROM docs"):
            yield json.loads(doc)


# --------------------------------------------------------------------------- XML parsing
def _clean(s: str) -> str:
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s or "")).strip()


def _itertext(el: ET.Element | None) -> str:
    return _clean("".join(el.itertext())) if el is not None else ""


def _year(root: ET.Element) -> int | None:
    for path in (".//PubDate/Year", ".//ArticleDate/Year", ".//PubDate/MedlineDate",
                 ".//DateCompleted/Year", ".//PubMedPubDate/Year"):
        el = root.find(path)
        if el is not None and el.text:
            m = re.search(r"(1[89]|20)\d\d", el.text)
            if m:
                return int(m.group(0))
    return None


def parse_pubmed_xml(xml_text: str | bytes) -> list[dict]:
    """Parse EFetch PubmedArticleSet XML into Doc dicts (incl. PubmedBookArticle, e.g. StatPearls)."""
    root = ET.fromstring(xml_text)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    docs = []
    for art in list(root):
        if art.tag == "PubmedArticle":
            base = art.find("MedlineCitation")
            if base is None:
                continue
            a = base.find("Article")
            pmid_el = base.find("PMID")
            title = _itertext(a.find("ArticleTitle")) if a is not None else ""
            if not title and a is not None:
                title = _itertext(a.find("VernacularTitle"))
            abs_el = a.find("Abstract") if a is not None else None
            journal = _itertext(a.find("Journal/Title")) if a is not None else ""
            ptypes = [_itertext(x) for x in (a.findall("PublicationTypeList/PublicationType") if a is not None else [])]
            mesh = [_itertext(x) for x in base.findall("MeshHeadingList/MeshHeading/DescriptorName")]
            langs = [_itertext(x) for x in (a.findall("Language") if a is not None else [])]
        elif art.tag == "PubmedBookArticle":
            base = art.find("BookDocument")
            if base is None:
                continue
            pmid_el = base.find("PMID")
            title = _itertext(base.find("ArticleTitle")) or _itertext(base.find("Book/BookTitle"))
            abs_el = base.find("Abstract")
            journal = _itertext(base.find("Book/BookTitle"))
            ptypes = [_itertext(x) for x in base.findall("PublicationType")] or ["Book Chapter"]
            mesh = [_itertext(x) for x in base.findall(".//MeshHeading/DescriptorName")]
            langs = [_itertext(x) for x in base.findall("Language")]
        else:
            continue
        if pmid_el is None or not (pmid_el.text or "").strip():
            continue
        pmid = pmid_el.text.strip()
        paras = []
        if abs_el is not None:
            for at in abs_el.findall("AbstractText"):
                t = _itertext(at)
                if not t:
                    continue
                label = _clean(at.get("Label") or "")
                paras.append(f"{label}: {t}" if label else t)
        abstract = "\n\n".join(paras)
        text = title + ("\n\n" + abstract if abstract else "")
        docs.append({
            "doc_id": pmid,
            "source": "pubmed",
            "lang": "en",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "title": title,
            "text": text,
            "fetched_at": now,
            "has_abstract": bool(abstract),
            "year": _year(art),
            "journal": journal,
            "pub_types": [p for p in ptypes if p],
            "mesh_terms": [m for m in mesh if m],
            "article_lang": ",".join(l for l in langs if l),
        })
    return docs


# --------------------------------------------------------------------------- query handling
_BOOL_RE = re.compile(r'\b(AND|OR|NOT)\b|\[[^\]]*\]|"')
_TAG_RE = re.compile(r"\[[^\]]*\]")
STOPWORDS = set("""
a an the of in on at to for from by with without about into over under between and or not is are was were be been
being do does did doing done have has had having can could should would will shall may might must what which who whom
whose when where why how that this these those it its i me my we our you your he she his her they them their there
here than then so as if but also any some all each other such very much many more most less least just only still
again ever often really please help question questions ask asked know want need tell explain explanation okay ok etc
vs versus regarding concerning someone something thing things get got make made""".split())
# generic words dropped only when relaxing an ESearch query with too few hits
GENERIC = set("""meaning mean means normal interpretation interpret result results value values number level levels
test tested check patient patients use used using take taking taken after before during while because due cause
caused below above slightly slight bit little somewhat high higher low lower elevated increased decreased raised
abnormal""".split())
UNITS = set("""u/l iu/l miu/ml mlu/ml ng/ml pg/ml mg/dl mmol/l umol/l µmol/l g/dl g/l mg/l iu ui mg ml mcg µg g kg
cm mm mmhg bpm % u l ng pg mmol""".split())


def is_boolean(q: str) -> bool:
    return bool(_BOOL_RE.search(q))


def clean_terms(q: str, drop_generic: bool = False) -> list[str]:
    """Natural-language query -> content terms (stopwords, measurements and units removed).

    Small integers directly after a content word are kept ("type 2 diabetes", "HPV 16", "stage 3")."""
    q = _TAG_RE.sub(" ", q)
    q = re.sub(r'\b(AND|OR|NOT)\b|[()"?,;:!]', " ", q)
    out: list[str] = []
    prev = ""
    for tok in q.split():
        tok = tok.strip(".'")
        t = tok.lower()
        if not t or t in STOPWORDS or t in UNITS or (drop_generic and t in GENERIC):
            prev = ""
            continue
        if re.fullmatch(r"[<>=≤≥~+\-]*[\d.,/:%x×\-]+[a-zµ/%]{0,6}", t) and not re.search(r"[a-z]{2,}\d|\d[a-z]{3,}", t):
            if re.fullmatch(r"\d{1,2}", t) and re.fullmatch(r"[a-z][a-z\-]*", prev or "-") and prev not in GENERIC:
                out.append(tok)  # type 2, stage 3, HPV 16
            prev = ""
            continue  # measurements like 1.52, 0.10, 5mg, 120/80
        out.append(tok)
        prev = t
    return out


def split_clauses(q: str) -> list[str]:
    """Split a (boolean or plain) query into top-level conjunctive clauses.

    Units are parenthesised groups, quoted phrases or words (with an optional [field] tag); explicit or
    implicit AND separates clauses, a top-level OR merges its neighbours into one "(a OR b)" clause."""
    units, i, n = [], 0, len(q)
    while i < n:
        ch = q[i]
        if ch.isspace():
            i += 1
            continue
        j = i
        if ch == "(":
            depth, inq = 0, False
            while j < n:
                if q[j] == '"':
                    inq = not inq
                elif not inq and q[j] == "(":
                    depth += 1
                elif not inq and q[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
        elif ch == '"':
            k = q.find('"', i + 1)
            j = n if k < 0 else k + 1
        else:
            while j < n and not q[j].isspace() and q[j] not in '()"':
                j += 1
                if j < n and q[j] == "[":
                    k = q.find("]", j)
                    j = n if k < 0 else k + 1
        if j < n and q[j] == "[":  # tag after a group / phrase
            k = q.find("]", j)
            j = n if k < 0 else k + 1
        units.append(q[i:j].strip())
        i = max(j, i + 1)
    clauses: list[str] = []
    pending_or = pending_not = False
    for u in units:
        if u == "AND":
            continue
        if u == "OR":
            pending_or = bool(clauses)
            continue
        if u == "NOT":
            pending_not = True
            continue
        if pending_not:
            u = f"NOT {u}"
            pending_not = False
        if pending_or:
            prev = clauses.pop()
            prev = prev[1:-1] if prev.startswith("(") and prev.endswith(")") and " OR " in prev else prev
            clauses.append(f"({prev} OR {u})")
            pending_or = False
        else:
            clauses.append(u)
    return [c for c in clauses if c and c not in ("()",)]


def _is_noise(clause: str, drop_generic: bool) -> bool:
    """Bare word clause that carries no search value (stopword, number, unit)."""
    if any(c in clause for c in '()"[') or clause.startswith("NOT "):
        return False
    return not clean_terms(clause, drop_generic=drop_generic)


def relaxations(q: str, max_n: int) -> list[tuple[str, bool]]:
    """Progressively looser variants of q (most specific first) as (variant, is_or_fallback):
    cleaned (stopwords/numbers removed) -> core (generic words removed) -> drop-one-clause (or 4-clause
    windows for long queries) -> OR of clauses."""
    out: list[tuple[str, bool]] = []
    units = split_clauses(q)
    def keep(c: str, prev: str, generic: bool) -> bool:
        if re.fullmatch(r"\d{1,2}", c):  # "type 2", "HPV 16"
            return bool(re.fullmatch(r"[A-Za-z][A-Za-z\-]*", prev)) and prev.lower() not in STOPWORDS | GENERIC
        return not _is_noise(c, generic)

    units = [c.strip(".,;:?!'") for c in units]
    cleaned = [c for k, c in enumerate(units) if c and keep(c, units[k - 1] if k else "", False)]
    core = [c for k, c in enumerate(cleaned) if keep(c, cleaned[k - 1] if k else "", True)] or cleaned
    if cleaned:
        out.append((" ".join(cleaned), False))
    if core:
        out.append((" ".join(core), False))
    if len(core) > 5:  # long NL query: contiguous windows keep multi-word concepts together
        for i in range(0, len(core) - 3, 2):
            out.append((" ".join(core[i:i + 4]), False))
    elif len(core) >= 3:
        for i in range(len(core) - 1, -1, -1):  # drop-one, last clauses first
            out.append((" ".join(c for j, c in enumerate(core) if j != i), False))
    if 2 <= len(core) <= 5:  # OR of many clauses is pure noise
        out.append((" OR ".join(core), True))
    seen, res = {q.strip(), " ".join(q.split())}, []
    for v, is_or in out:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            res.append((v, is_or))
    ands = [x for x in res if not x[1]][: max(0, max_n - 1)]
    return ands + [x for x in res if x[1]][:1]


def rrf(lists: list[tuple[list[str], float]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ids, w in lists:
        for r, pid in enumerate(ids):
            scores[pid] = scores.get(pid, 0.0) + w / (k + r + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def _dedupe(xs: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        x = (x or "").strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


# --------------------------------------------------------------------------- client
class PubMedClient:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.api_key = os.environ.get("NCBI_API_KEY") or None
        self.email = os.environ.get("NCBI_EMAIL") or None
        rps = self.cfg.get("ncbi_rps") or (7.5 if self.api_key else 2.0)  # NCBI: 3/s, 10/s with key
        self.ncbi_limiter = RateLimiter(float(rps))
        self.epmc_limiter = RateLimiter(float(self.cfg["epmc_rps"]))
        db = Path(self.cfg["cache_db"])
        self.cache = Cache(db if db.is_absolute() else ROOT / db)
        self.http = httpx.Client(timeout=self.cfg["timeout"], follow_redirects=True,
                                 headers={"User-Agent": f"{self.cfg['tool']}/1.0 (research; python-httpx)"},
                                 limits=httpx.Limits(max_connections=32, max_keepalive_connections=16))
        self.pool = ThreadPoolExecutor(max_workers=int(self.cfg["inner_workers"]))
        # separate pool for relaxation sub-queries: they are spawned from tasks running in self.pool
        self.relax_pool = ThreadPoolExecutor(max_workers=int(self.cfg["inner_workers"]))
        self.stats = {"ncbi_requests": 0, "epmc_requests": 0, "retries": 0, "cache_hits": 0}
        self._stats_lock = threading.Lock()

    def _stat(self, k: str, n: int = 1):
        with self._stats_lock:
            self.stats[k] += n

    # ---- HTTP with retry
    def _request(self, method: str, url: str, *, params=None, data=None, ncbi=True) -> httpx.Response:
        limiter = self.ncbi_limiter if ncbi else self.epmc_limiter
        if ncbi:
            extra = {"tool": self.cfg["tool"]}
            if self.email:
                extra["email"] = self.email
            if self.api_key:
                extra["api_key"] = self.api_key
            if isinstance(data, list):
                data = data + list(extra.items())
            elif data is not None:
                data = {**data, **extra}
            else:
                params = {**(params or {}), **extra}
        content, headers = None, None
        if isinstance(data, list):  # repeated keys (e.g. ELink id=..&id=..) -> encode manually
            content = urlencode(data).encode()
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = None
        last: Exception | None = None
        for attempt in range(int(self.cfg["max_retries"]) + 1):
            limiter.acquire()
            self._stat("ncbi_requests" if ncbi else "epmc_requests")
            try:
                r = self.http.request(method, url, params=params, data=data, content=content, headers=headers)
                body_err = ncbi and r.status_code == 200 and len(r.content) < 300 and b"rate limit" in r.content.lower()
                if r.status_code == 429 or r.status_code >= 500 or body_err:
                    ra = r.headers.get("Retry-After")
                    wait = float(ra) if ra and ra.replace(".", "").isdigit() else min(30, 1.5 * 2 ** attempt)
                    raise _Retry(f"HTTP {r.status_code}", wait)
                r.raise_for_status()
                return r
            except (_Retry, httpx.TransportError) as e:
                last = e
                wait = e.wait if isinstance(e, _Retry) else min(30, 1.5 * 2 ** attempt)
                wait *= 1 + 0.25 * random.random()
                self._stat("retries")
                log.debug("retry %s %s in %.1fs (%s)", method, url, wait, e)
                limiter.penalize(wait if isinstance(e, _Retry) and str(e).startswith("HTTP 429") else 0)
                time.sleep(wait)
        raise RuntimeError(f"request failed after retries: {url}: {last}")

    # ---- ESearch
    def esearch(self, term: str, retmax: int = 100) -> tuple[list[str], int, str]:
        """Best Match ESearch. NOTE: with sort=relevance only small pages are Best-Match ranked and
        deterministic (retmax<=~40; retstart+retmax<=~80). A single retmax=100 call returns a different,
        non-deterministic order that usually misses the true top-20 -> we page in `esearch_page` steps
        up to `esearch_max_depth` (see docs/pubmed_notes.md)."""
        term = term.strip()
        page = int(self.cfg["esearch_page"])
        retmax = max(1, min(int(retmax), int(self.cfg["esearch_max_depth"])))
        key = f"esearch\t{term}"
        hit = self.cache.get_search(key, retmax)
        if hit is not None:
            self._stat("cache_hits")
            return hit[0], hit[1], hit[2].get("qt", "")
        pmids: list[str] = []
        count, qt, start = 0, "", 0
        while start < retmax:
            params = {"db": "pubmed", "term": term, "retmax": min(page, retmax - start), "retstart": start,
                      "retmode": "json", "sort": "relevance"}
            r = self._request("POST", f"{EUTILS}/esearch.fcgi", data=params)
            try:
                res = r.json().get("esearchresult", {})
            except ValueError:
                res = {}
            if "ERROR" in res:
                log.debug("esearch error for %r: %s", term, res["ERROR"])
            ids = [str(x) for x in res.get("idlist", [])]
            count = int(res.get("count", 0) or 0)
            qt = qt or res.get("querytranslation", "")
            pmids += ids
            start += page
            if len(ids) < params["retmax"] or start >= count:
                break
        pmids = _dedupe(pmids)
        self.cache.put_search(key, retmax, count, pmids, {"qt": qt})
        return pmids, count, qt

    def esearch_relaxed(self, term: str, retmax: int = 100) -> list[str]:
        """ESearch; when the hit count is small, append results of progressively relaxed variants."""
        if self.cfg["preclean"] and not is_boolean(term):
            # NL sentences: '?', ':' and measurements ("0.10 U/L") make ATM return 0 hits; stopwords are
            # ignored by PubMed anyway -> search the cleaned string first (no extra request)
            term = " ".join(clean_terms(term)) or term
        pmids, count, _ = self.esearch(term, retmax)
        min_hits = int(self.cfg["min_hits"])
        if not self.cfg["relax"] or count >= min_hits:
            return pmids
        variants = relaxations(term, int(self.cfg["max_relax_requests"]))
        if not variants:
            return pmids
        # cleaned / drop-one variants are tried in parallel; OR variant only as a last resort
        main = [v for v, is_or in variants if not is_or]
        ors = [v for v, is_or in variants if is_or]
        rdepth = min(retmax, int(self.cfg["relax_depth"]))
        results = list(self.relax_pool.map(lambda v: self.esearch(v, rdepth), main))
        lists = [(ids, 1.0) for ids, c, _ in results if ids]
        total = sum(c for _, c, _ in results)
        if total < min_hits and ors:
            for v in ors:
                ids, c, _ = self.esearch(v, rdepth)
                if ids:
                    lists.append((ids, 0.5))
        fused = [p for p, _ in rrf(lists, int(self.cfg["rrf_k"]))]
        out = list(pmids)
        seen = set(out)
        for p in fused:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out[:retmax]

    # ---- Europe PMC
    def epmc_search(self, query: str, retmax: int = 100) -> list[str]:
        q = _TAG_RE.sub("", query).strip()
        if not is_boolean(query):
            terms = clean_terms(q)
            q = " ".join(terms) or q
        if not q:
            return []
        full = f"({q}) AND SRC:MED" + (" AND HAS_ABSTRACT:y" if self.cfg["epmc_has_abstract"] else "")
        key = f"epmc\t{full}"
        hit = self.cache.get_search(key, retmax)
        if hit is not None:
            self._stat("cache_hits")
            return hit[0]
        pmids: list[str] = []
        cursor, count = "*", 0
        while len(pmids) < retmax:
            params = {"query": full, "format": "json", "resultType": "idlist",
                      "pageSize": min(1000, retmax - len(pmids)), "cursorMark": cursor}
            j = self._request("GET", EPMC, params=params, ncbi=False).json()
            count = int(j.get("hitCount", 0) or 0)
            res = j.get("resultList", {}).get("result", [])
            pmids += [str(x["pmid"]) for x in res if x.get("pmid")]
            nxt = j.get("nextCursorMark")
            if not res or not nxt or nxt == cursor:
                break
            cursor = nxt
        pmids = _dedupe(pmids)[:retmax]
        self.cache.put_search(key, retmax, count, pmids)
        if count < int(self.cfg["min_hits"]) and self.cfg["relax"] and not is_boolean(query):
            terms = clean_terms(q)
            if len(terms) >= 2:
                extra = self.epmc_search(" OR ".join(terms), retmax)
                pmids = _dedupe(pmids + extra)[:retmax]
        return pmids

    # ---- ELink related articles
    def elink(self, pmids: list[str], per_hit: int = 20) -> dict[str, list[str]]:
        pmids = [str(p) for p in pmids]
        out = self.cache.get_elink(pmids)
        todo = [p for p in pmids if p not in out]
        for i in range(0, len(todo), 50):
            batch = todo[i:i + 50]
            data = [("dbfrom", "pubmed"), ("db", "pubmed"), ("linkname", "pubmed_pubmed"),
                    ("cmd", "neighbor_score"), ("retmode", "json")] + [("id", p) for p in batch]
            # repeated id params => one linkset per input PMID
            try:
                r = self._request("POST", f"{EUTILS}/elink.fcgi", data=data)
            except (RuntimeError, httpx.HTTPError) as e:  # expansion is optional: never fatal
                log.warning("elink failed: %s", e)
                r = None
            rel: dict[str, list] = {p: [] for p in batch}
            if r is not None and r.status_code == 200:
                for ls in r.json().get("linksets", []):
                    src = str(ls.get("ids", [""])[0])
                    for db in ls.get("linksetdbs", []):
                        if db.get("linkname") == "pubmed_pubmed":
                            rel[src] = [str(x["id"]) if isinstance(x, dict) else str(x)
                                        for x in db.get("links", [])][:200]
                self.cache.put_elink(rel)
            out.update(rel)
        return {p: [x for x in out.get(p, []) if x != p][:per_hit] for p in pmids}

    # ---- public API
    def search(self, queries: list[str], retmax: int = 100, sources: list[str] | None = None,
               epmc_queries: list[str] | None = None, return_scores: bool = False):
        """Fused PMID ranking (weighted RRF) over all queries and sources."""
        sources = sources or self.cfg["sources"]
        queries = _dedupe(queries)
        epmc_queries = _dedupe(epmc_queries if epmc_queries is not None else queries)
        w = self.cfg["weights"]
        jobs = []
        if "ncbi" in sources:
            jobs += [("ncbi", q) for q in queries]
        if "europepmc" in sources:
            jobs += [("europepmc", q) for q in epmc_queries]

        def run(job):
            src, q = job
            try:
                ids = self.esearch_relaxed(q, retmax) if src == "ncbi" else self.epmc_search(q, retmax)
            except Exception as e:
                log.warning("%s search failed for %r: %s", src, q, e)
                ids = []
            return src, ids

        lists = [(ids, float(w.get(src, 1.0))) for src, ids in self.pool.map(run, jobs)]
        fused = rrf(lists, int(self.cfg["rrf_k"]))
        return fused if return_scores else [p for p, _ in fused]

    def fetch(self, pmids: Iterable[str], keep_no_abstract: bool | None = None) -> list[dict]:
        """PMIDs -> Docs (input order kept). Cached; missing ones fetched by EFetch in batches."""
        keep = self.cfg["keep_no_abstract"] if keep_no_abstract is None else keep_no_abstract
        pmids = _dedupe(str(p) for p in pmids)
        have = self.cache.get_docs(pmids)
        self._stat("cache_hits", len(have))
        todo = [p for p in pmids if p not in have]
        if todo:
            gone = self.cache.get_missing(todo)
            todo = [p for p in todo if p not in gone]
        bs = int(self.cfg["fetch_batch"])
        batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]

        def efetch(batch: list[str]) -> list[dict]:
            r = self._request("POST", f"{EUTILS}/efetch.fcgi",
                              data={"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
            docs = parse_pubmed_xml(r.content)
            self.cache.put_docs(docs)
            got = {d["doc_id"] for d in docs}
            self.cache.put_missing(p for p in batch if p not in got)
            return docs

        for docs in self.pool.map(efetch, batches):
            for d in docs:
                have[d["doc_id"]] = d
        out = [have[p] for p in pmids if p in have]
        return out if keep else [d for d in out if d.get("has_abstract")]

    def candidate_pmids(self, plan: dict, retmax: int | None = None, elink: bool | None = None) -> list[tuple[str, float]]:
        retmax = int(retmax or self.cfg["retmax"])
        ncbi_q = _dedupe(list(plan.get("pubmed") or []) + list(plan.get("en") or []))
        epmc_q: list[str] = []
        for f in self.cfg["epmc_from"]:
            epmc_q += list(plan.get(f) or [])
        if not ncbi_q and not epmc_q:
            return []
        fused = self.search(ncbi_q, retmax, epmc_queries=_dedupe(epmc_q), return_scores=True)
        if self.cfg["elink"] if elink is None else elink:
            top = [p for p, _ in fused[: int(self.cfg["elink_top"])]]
            rel = self.elink(top, int(self.cfg["elink_per_hit"]))
            k = int(self.cfg["rrf_k"])
            # related lists enter the fused scores with a lower weight
            scores = dict(fused)
            for ids in rel.values():
                for r, p in enumerate(ids):
                    scores[p] = scores.get(p, 0.0) + float(self.cfg["weights"].get("elink", 0.5)) / (k + r + 1)
            fused = sorted(scores.items(), key=lambda x: -x[1])
        return fused

    def candidates_for_plan(self, plan: dict, max_candidates: int | None = None, **kw) -> list[dict]:
        n = int(max_candidates or self.cfg["max_candidates"])
        fused = self.candidate_pmids(plan, **kw)
        # fetch a bit more than n so that dropping abstract-less articles still leaves ~n docs
        head = fused[: int(n * 1.3) + 10]
        score = dict(head)
        docs = self.fetch([p for p, _ in head])[:n]
        return [{**d, "pm_rank": i, "pm_score": score.get(d["doc_id"], 0.0)} for i, d in enumerate(docs)]


class _Retry(Exception):
    def __init__(self, msg: str, wait: float):
        super().__init__(msg)
        self.wait = wait


# --------------------------------------------------------------------------- module-level API
_client: PubMedClient | None = None
_client_lock = threading.Lock()


def get_client(cfg: dict | None = None) -> PubMedClient:
    global _client
    with _client_lock:
        if _client is None or cfg is not None:
            _client = PubMedClient(cfg)
        return _client


def search(queries: list[str], retmax: int = 100) -> list[str]:
    return get_client().search(queries, retmax)


def fetch(pmids: Iterable[str]) -> list[dict]:
    return get_client().fetch(pmids)


def candidates_for_plan(plan: dict) -> list[dict]:
    return get_client().candidates_for_plan(plan)


async def asearch(queries: list[str], retmax: int = 100) -> list[str]:
    return await asyncio.to_thread(search, queries, retmax)


async def afetch(pmids: Iterable[str]) -> list[dict]:
    return await asyncio.to_thread(fetch, list(pmids))


async def acandidates_for_plan(plan: dict) -> list[dict]:
    return await asyncio.to_thread(candidates_for_plan, plan)


# --------------------------------------------------------------------------- CLI
DOC_COLUMNS = ["doc_id", "source", "lang", "url", "title", "text", "fetched_at",
               "has_abstract", "year", "journal", "pub_types", "mesh_terms", "article_lang"]


def write_docs_parquet(docs: list[dict], out: str | os.PathLike) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("doc_id", pa.string()), ("source", pa.string()), ("lang", pa.string()), ("url", pa.string()),
        ("title", pa.string()), ("text", pa.string()), ("fetched_at", pa.string()),
        ("has_abstract", pa.bool_()), ("year", pa.int32()), ("journal", pa.string()),
        ("pub_types", pa.list_(pa.string())), ("mesh_terms", pa.list_(pa.string())), ("article_lang", pa.string()),
    ])
    cols = {c: [d.get(c) for d in docs] for c in DOC_COLUMNS}
    table = pa.table(cols, schema=schema)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out)


def _plan_hash(plan: dict, cfg: dict) -> str:
    keyobj = {"pubmed": plan.get("pubmed"), "en": plan.get("en"),
              "cfg": {k: cfg[k] for k in ("sources", "retmax", "min_hits", "relax", "epmc_from", "elink",
                                          "elink_top", "elink_per_hit", "weights")}}
    return hashlib.sha1(json.dumps(keyobj, sort_keys=True).encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prefetch PubMed candidates for all query plans")
    ap.add_argument("--plans", default="data/cache/query_plans.jsonl")
    ap.add_argument("--out", default="data/docs/pubmed.parquet")
    ap.add_argument("--cands-out", default="data/cache/pubmed_candidates.jsonl",
                    help="per-query ranked PMIDs (jsonl: id, pmids, scores)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache", default=None, help="sqlite cache path (default: pubmed.cache_db)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--retmax", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--elink", action="store_true", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.retmax:
        cfg["retmax"] = args.retmax
    if args.cache:
        cfg["cache_db"] = os.path.abspath(args.cache)
    if args.elink:
        cfg["elink"] = True
    if args.max_candidates:
        cfg["max_candidates"] = args.max_candidates
    client = get_client(cfg)
    n_cand = int(cfg["max_candidates"])

    plans = []
    with open(args.plans, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                plans.append(json.loads(line))
    if args.limit:
        plans = plans[: args.limit]
    log.info("plans: %d | api_key=%s | cache=%s", len(plans), bool(client.api_key), client.cache.path)

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x

    t0 = time.time()
    results: dict[str, tuple[list[str], list[float]]] = {}
    todo = []
    for p in plans:
        pid, ph = str(p["id"]), _plan_hash(p, cfg)
        hit = client.cache.get_plan(pid, ph)
        if hit is not None:
            results[pid] = hit
        else:
            todo.append((pid, ph, p))
    log.info("resume: %d plans cached, %d to search", len(results), len(todo))

    def work(item):
        pid, ph, p = item
        fused = client.candidate_pmids(p)
        head = fused[: int(n_cand * 1.3) + 10]
        docs = client.fetch([x for x, _ in head])[:n_cand]
        sc = dict(head)
        pm = [d["doc_id"] for d in docs]
        ss = [round(sc[x], 6) for x in pm]
        client.cache.put_plan(pid, ph, pm, ss)
        return pid, pm, ss

    with ThreadPoolExecutor(max_workers=int(args.workers or cfg["workers"])) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for fu in tqdm(futs, total=len(futs), desc="plans", unit="q"):
            try:
                pid, pm, ss = fu.result()
                results[pid] = (pm, ss)
            except Exception as e:
                log.error("plan failed: %s", e)
    log.info("search+fetch done in %.1fs; stats=%s", time.time() - t0, client.stats)

    Path(args.cands_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cands_out, "w", encoding="utf-8") as f:
        for p in plans:
            pid = str(p["id"])
            if pid in results:
                pm, ss = results[pid]
                f.write(json.dumps({"id": p["id"], "pmids": pm, "scores": ss}) + "\n")
    all_pmids = _dedupe(x for pm, _ in results.values() for x in pm)
    docs = client.fetch(all_pmids)
    write_docs_parquet(docs, args.out)
    log.info("wrote %d docs -> %s ; candidates -> %s (%d/%d plans) in %.1fs",
             len(docs), args.out, args.cands_out, len(results), len(plans), time.time() - t0)


if __name__ == "__main__":
    main()
