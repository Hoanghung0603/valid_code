"""Minimal static proxy probe: zysjonline.com only, max 3 corpus URLs, LOCAL only.

Scope (parent-bounded): host zysjonline.com never runs elsewhere in this stage;
tier-4/never-bulk. max 3 corpus URLs selected from read-only corpus. LOCAL execution
through one static proxy (index 0, fixed, no rotation, no other IP for host).

Secrets: Webshare_100_proxies.txt is parsed ONLY at runtime by this script
(programmatic read from workspace file). Accepted per-line formats:
  ip:port:user:pass   OR   http(s)://[user:pass@]ip:port
Never list/print raw credentials or env. stdout = redacted counts + schemes only.
No host/IP needed in output. No credentials written anywhere (no manifest/logs/repo).
Proxy auth failure (HTTP 407) -> stop batch immediately, report, no secret logs.

Fetch: requests only (already installed locally). Sequential (concurrency 1),
delay >= 3s, connect timeout 20s, total timeout 30s, body cap 3MB (read chunks),
total cap 10MB. Mbps measured from read chunks + time but capped at 20.
No direct-origin fallback if proxy fails. No index switch if blocked.
Extract via local scripts.extract (trafilatura). Only status=final actual `ok`
records written (schema-9 jsonl) to local temp path; audit = tiny redacted stats
(no html/json raw, errors redacted).

Blocked target (403/429) -> stop host, do not try other routes.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

DEPLOY_DIR = Path(__file__).resolve().parent
ROOT = DEPLOY_DIR.parent  # r2ai/
sys.path.insert(0, str(ROOT / "scripts"))
from extract import classify, extract  # noqa: E402

ALLOWED_HOST = "zysjonline.com"
MAX_URLS = 3
MAX_BODY = 3 * 1024 * 1024
MAX_TOTAL = 10 * 1024 * 1024
CONNECT_TIMEOUT = 20
TOTAL_TIMEOUT = 30
MIN_DELAY = 3.0
MAX_MBPS = 20
SCHEMA = ("id", "url", "host", "final_url", "status", "title", "date", "text", "ts")
PROXY_FILE_CANDIDATES = (ROOT / "Webshare_100_proxies.txt",)


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def redact(text: str) -> str:
    """Redact anything credential-like from logs/errors. Conservative."""
    if not text:
        return ""
    s = str(text)
    # proxy URLs with userinfo
    s = re.sub(r"(https?://)[^/\s@]+@", r"\1***@", s)
    # ip:port:user:pass patterns
    s = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+\s*:\s*[^:\s]+\s*:\s*\S+", "***:***:***:***", s)
    # user/pass key=value fragments
    s = re.sub(r"(?i)(user(name)?|pass(word)?|pwd|token)\s*[:=]\s*\S+", r"\1=***", s)
    return s[:300]


def parse_proxy_line(line: str):
    """Return sanitized requests-style proxy URL or None. Never logs secrets."""
    line = (line or "").strip()
    if not line or line.startswith("#"):
        return None
    try:
        if "://" in line:
            p = urlparse(line)
            if p.scheme not in ("http", "https"):
                return None
            if not p.hostname or not p.port:
                return None
            # rebuild without ever exposing; requests needs full URL internally
            return f"{p.scheme}://{p.username + ':' + p.password + '@' if p.username else ''}{p.hostname}:{p.port}"
        parts = line.split(":")
        if len(parts) == 4 and parts[0] and parts[1].isdigit():
            ip, port, user, pwd = [x.strip() for x in parts]
            if not ip or not user or not pwd:
                return None
            return f"http://{user}:{pwd}@{ip}:{port}"
    except Exception:
        return None
    return None


def load_proxy_index0(path: Path):
    """Runtime-only parse of secret file; index 0 fixed. Returns (proxy_url, stats).

    stats are redacted counts only: {lines, valid, scheme}. Never raw content.
    """
    lines, valid, first, scheme = 0, 0, None, ""
    try:
        fh = path.open("r", encoding="utf-8", errors="ignore")
    except OSError:
        return None, {"lines": 0, "valid": 0, "scheme": ""}
    with fh:
        for raw in fh:
            s = (raw or "").strip()
            if not s or s.startswith("#"):
                continue
            lines += 1
            pu = parse_proxy_line(s)
            if pu:
                valid += 1
                if first is None:
                    first = pu
                    try:
                        scheme = urlparse(pu).scheme or ""
                    except Exception:
                        scheme = ""
    return first, {"lines": lines, "valid": valid, "scheme": scheme}


def select_candidates(corpus: Path, n: int = MAX_URLS):
    """Read-only parquet -> first n zysjonline id/url rows (never attempted upstream).

    No origin fetch here. Raises if fewer than 1 candidate.
    """
    import pandas as pd

    df = pd.read_parquet(corpus, columns=["id", "url"])
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("corpus IDs must be non-null and unique")
    df["host"] = df["url"].map(host_of)
    sub = df[df["host"] == ALLOWED_HOST].sort_values("id").head(n)
    if len(sub) == 0:
        raise ValueError("no valid candidates for allowed host")
    out = [{"id": int(i), "url": str(u)} for i, u in
           zip(sub["id"].tolist(), sub["url"].tolist())]
    # scope guard
    if len(out) > MAX_URLS:
        raise ValueError("candidate scope exceeds max 3")
    for c in out:
        if host_of(c["url"]) != ALLOWED_HOST or not isinstance(c["id"], int):
            raise ValueError("candidate scope violation")
    return out


def fetch_via_proxy(url: str, proxy_url: str):
    """Single GET through proxy. Returns dict with redacted-safe fields.

    Never raises credential content; errors are redacted strings.
    """
    import requests

    proxies = {"http": proxy_url, "https": proxy_url}
    t0 = time.time()
    nbytes = 0
    chunks = []
    status = None
    final_url = url
    err = None
    try:
        with requests.get(url, proxies=proxies, timeout=(CONNECT_TIMEOUT, TOTAL_TIMEOUT),
                           stream=True,
                           headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"}) as r:
            status = r.status_code
            final_url = str(getattr(r, "url", url) or url)
            # 407: proxy auth failure -> caller stops batch
            if status == 407:
                return {"status": 407, "final_url": final_url, "body": b"",
                        "nbytes": 0, "elapsed": time.time() - t0, "error": "proxy-auth-required"}
            for ch in r.iter_content(chunk_size=64 * 1024):
                if not ch:
                    continue
                nbytes += len(ch)
                if nbytes <= MAX_BODY:
                    chunks.append(ch)
                if nbytes >= MAX_BODY:
                    break
    except Exception as e:
        err = redact(f"{type(e).__name__}")
    elapsed = max(time.time() - t0, 1e-6)
    mbps = (nbytes * 8 / 1e6 / elapsed) if elapsed else 0.0
    return {"status": status, "final_url": final_url, "body": b"".join(chunks)[:MAX_BODY],
            "nbytes": nbytes, "elapsed": elapsed, "mbps": min(mbps, float("inf")),
            "error": err}


def route_hash_nonsecret(scheme: str) -> str:
    """Optional stable non-secret route hash (scheme + host only, no creds/IP)."""
    h = hashlib.sha256(f"{scheme}://{ALLOWED_HOST}".encode()).hexdigest()
    return h[:12]


def run(corpus: Path, out: Path, proxy_file: Path, delay: float = MIN_DELAY,
        session_get=None):
    """Execute bounded probe. session_get(url, proxy) injectable for tests.

    Returns (rc, audit_redacted). Only `ok` records written; files created with 'x'.
    rc: 0 done, 1 no-valid-proxy, 2 usage/output collision, 3 proxy-407 stop,
        4 target-blocked stop.
    """
    if delay < MIN_DELAY:
        raise ValueError("delay must be >= 3.0s")
    if out.suffix != "":
        rec_path = out.with_suffix(".records.jsonl")
        audit_path = out.with_suffix(".audit.jsonl")
    else:
        rec_path = out.parent / (out.name + ".records.jsonl")
        audit_path = out.parent / (out.name + ".audit.jsonl")
    if rec_path.exists() or audit_path.exists():
        return 2, {"ok": False, "error": "output-collision"}
    proxy_url, pstat = load_proxy_index0(proxy_file)
    if not proxy_url:
        audit = {"ok": False, "error": "no valid proxy",
                 "proxy_lines": pstat["lines"], "proxy_valid": pstat["valid"],
                 "scheme": pstat.get("scheme", ""), "route": route_hash_nonsecret(pstat.get("scheme", ""))}
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("x", encoding="utf-8") as af:
                af.write(json.dumps(audit, ensure_ascii=False) + "\n")
        except Exception:
            pass
        print(json.dumps({"attempted": 0, "proxy_lines": pstat["lines"],
                          "proxy_valid": pstat["valid"], "scheme": pstat.get("scheme", ""),
                          "error": "no valid proxy"}))
        return 1, audit
    try:
        cands = select_candidates(corpus, MAX_URLS)
    except Exception as e:
        audit = {"ok": False, "error": redact(str(e))[:120]}
        return 2, audit
    fetch = session_get or (lambda u, p: fetch_via_proxy(u, p))
    total_bytes = 0
    attempted = 0
    ok_n = 0
    outcomes = []
    scheme = pstat.get("scheme", "")
    route = route_hash_nonsecret(scheme)
    rf = rec_path.open("x", encoding="utf-8")
    af = audit_path.open("x", encoding="utf-8")
    rc = 0
    stop_reason = ""
    with rf, af:
        for c in cands:
            cid, url = c["id"], c["url"]
            if host_of(url) != ALLOWED_HOST:  # scope guard
                continue
            attempted += 1
            try:
                fr = fetch(url, proxy_url)
            except Exception as e:
                af.write(json.dumps({"id": cid, "ok": False,
                                     "error": redact(str(e))}) + "\n")
                break
            status = fr.get("status")
            final_url = fr.get("final_url") or url
            body = fr.get("body") or b""
            nbytes = fr.get("nbytes", len(body))
            total_bytes += nbytes
            # 407 proxy-auth failure -> stop batch, no secret logs
            if status == 407:
                audit = {"id": cid, "ok": False, "status": 407,
                         "class": "proxy-auth", "nbytes": nbytes}
                af.write(json.dumps(audit) + "\n")
                print(json.dumps({"attempted": attempted, "ok": ok_n, "scheme": scheme,
                                  "route": route, "error": "proxy-auth-407-stop"}))
                rc, stop_reason = 3, "proxy-auth-407-stop"
                break
            ferr = fr.get("error")
            ex = extract(body, final_url) if (status == 200 and body and not ferr) else \
                {"title": "", "date": "", "text": ""}
            cls = classify(status, final_url or "", ex.get("text", ""), ferr)
            outcomes.append(cls)
            audit = {"id": cid, "status": status, "class": cls,
                     "nbytes": nbytes, "text_len": len(ex.get("text", "")),
                     "title_len": len(ex.get("title", "")),
                     "elapsed": round(fr.get("elapsed", 0.0), 2)}
            if status in (403, 429) or cls == "blocked":
                audit["ok"] = False
                af.write(json.dumps(audit) + "\n")
                print(json.dumps({"attempted": attempted, "ok": ok_n, "scheme": scheme,
                                  "route": route, "error": "target-blocked-stop",
                                  "status": status, "class": cls}))
                rc, stop_reason = 4, "target-blocked-stop"
                break
            if cls == "ok" and status == 200:
                rf.write(json.dumps({"id": cid, "url": url, "host": ALLOWED_HOST,
                                     "final_url": final_url, "status": status,
                                     "title": ex.get("title", ""), "date": ex.get("date", ""),
                                     "text": ex.get("text", ""), "ts": int(time.time())},
                                    ensure_ascii=False) + "\n")
                audit["ok"] = True
                af.write(json.dumps(audit) + "\n")
                ok_n += 1
            else:
                audit["ok"] = False
                # never persist raw html/json; transient errors redacted
                if ferr:
                    audit["error"] = redact(ferr)
                af.write(json.dumps(audit) + "\n")
            if total_bytes >= MAX_TOTAL:
                break
            time.sleep(delay)
    if rc == 0:
        print(json.dumps({"attempted": attempted, "ok": ok_n, "scheme": scheme,
                          "route": route, "bytes": total_bytes}))
    return rc, {"attempted": attempted, "ok": ok_n, "scheme": scheme,
                "route": route, "stop": stop_reason, "outcomes": outcomes,
                "bytes": total_bytes}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "links_corpus.parquet"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--proxy-file", default=str(ROOT / "Webshare_100_proxies.txt"))
    ap.add_argument("--delay", type=float, default=MIN_DELAY)
    a = ap.parse_args(argv)
    rc, _ = run(Path(a.corpus), Path(a.out), Path(a.proxy_file), a.delay)
    return rc


if __name__ == "__main__":
    sys.exit(main())
