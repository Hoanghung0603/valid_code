"""Bounded nodriver pilot for known JSON-failure IDs on b660 origin IP.

- Queue must be a non-empty SUBSET of QUEUE_IDS (1881723, 1900274), cnkang.com
  only, sequential, delay>=3s. The _d run targets 1900274 alone (1881723
  already has 1 valid record; no re-fetch).
- Subclasses/imports deploy/crawl_browser_probe.runbrowser wiring; does NOT
  modify that file. Adds: fresh nodriver browser PER URL (no session leak),
  CDP 20Mbps emulation BEFORE navigation (fail-closed), main-frame typed
  events (frame_id==root + loader tracking), genuine network bytes
  (LoadingFinished/DataReceived, NOT DOM len), 10MB session cap with abort,
  txt-fallback on JSONDecodeError only, DOM title via evaluate, schema-9
  closed shard in NEW data/crawl/browser_retry_<suffix>/, audit with source
  id/url + method/provenance, no original changes.
- No proxy, no captcha solving, no threshold lowering, no raw-HTML persistence.

Exit codes: 0 done; 1 runtime/invariant failure; 2 usage/validation error;
3 lock/STOP refusal; 4 existing-output refusal.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import crawl_browser_probe as probe  # noqa: E402

NODRIVER_PIN = "nodriver==0.50.6"
QUEUE_IDS = (1881723, 1900274)
SCHEMA = ("id", "url", "host", "final_url", "status", "title", "date", "text", "ts")
BODY_CAP = 3 * 1024 * 1024
SESSION_BYTE_CAP = 10 * 1024 * 1024
# 20 Mbps download cap via CDP Network.emulate_network_conditions (bytes/s).
CDP_THROUGHPUT_BPS = 2500000
CDP_UPLOAD_BPS = 125000
TIMEOUT_S = 45.0


DEFAULT_LOCK = r"D:\r2ai\locks\heavy_job.lock"


def lock_hit(lock: Path) -> bool:
    try:
        return lock is not None and lock.exists()
    except OSError:
        return False


def stop_hit(crawl_root: Path) -> bool:
    try:
        return (crawl_root / "STOP").exists()
    except OSError:
        return False


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(10)
    except Exception as e:
        print(f"[pilot] WARN lower_priority: {e}", flush=True)


def fallback_text(html: bytes, url: str) -> dict:
    """trafilatura txt fallback when JSON output path raises JSONDecodeError."""
    import trafilatura
    try:
        txt = trafilatura.extract(html, url=url, output_format="txt",
                                  with_metadata=False, include_comments=False,
                                  include_tables=True, favor_recall=True)
    except Exception as e:
        return {"title": "", "date": "", "text": "", "fallback_error": f"{type(e).__name__}"}
    return {"title": "", "date": "", "text": txt or ""}


def _mk_cb(store: list, root_frame: str | None = None, loader: dict | None = None):
    # nodriver 0.50.6 dispatches handlers as callback(event, tab); a one-arg
    # sync function raises TypeError inside process_event and is retried as
    # callback(event) — so accept both arities explicitly.
    # Main-frame gate: typed ResponseReceived with frame_id == root frame and
    # loader_id == current navigation loader. If root/loader unknown, record
    # nothing (fail-closed: caller reports status unknown, never fakes 200).
    def _cb(event, _tab=None):
        try:
            resp = getattr(event, "response", event)
            if isinstance(resp, dict):
                url = resp.get("url", "")
                status = resp.get("status")
                rtype = resp.get("type", "Document")
                fid = resp.get("frame_id")
                lid = resp.get("loader_id")
            else:
                url = getattr(resp, "url", "")
                status = getattr(resp, "status", None)
                t = getattr(event, "type_", None)
                rtype = getattr(t, "value", t) or "Document"
                fid = getattr(event, "frame_id", None)
                fobj = getattr(fid, "value", fid)
                fid = str(fobj) if fobj is not None else None
                lobj = getattr(event, "loader_id", None)
                lid = str(getattr(lobj, "value", lobj)) if lobj is not None else None
            if rtype is not None and str(rtype) != "Document":
                return
            if root_frame is not None and fid != root_frame:
                return
            if loader is not None and loader.get("id") is not None and lid != loader["id"]:
                return
            store.append({"url": url, "status": status, "type": "Document",
                          "frame_id": fid, "loader_id": lid,
                          "frame_main": bool(root_frame is not None and fid == root_frame)})
        except Exception:
            pass
    return _cb


def _mk_bytes_cb(counter: dict, cap: int, abort: dict, ename_fixed: str | None = None):
    # Genuine network bytes: LoadingFinished.encoded_data_length per request
    # for the main frame; DataReceived streams the running total so the 10MB
    # session/page cap can abort the main load mid-flight.
    def _cb(event, _tab=None):
        try:
            if isinstance(event, dict):
                ename = ename_fixed or event.get("__event", "")
                if ename in ("LoadingFinished", "DataReceived") or "__event__" in event:
                    counter["bytes"] += int(event.get("encoded_data_length", 0) or 0)
                else:
                    return
            else:
                ename = type(event).__name__
                if ename == "LoadingFinished":
                    counter["bytes"] += int(getattr(event, "encoded_data_length", 0) or 0)
                elif ename == "DataReceived":
                    counter["bytes"] += int(getattr(event, "encoded_data_length", 0) or 0)
                else:
                    return
            if counter["bytes"] > cap:
                abort["flag"] = True
        except Exception:
            pass
    return _cb


async def _cap_session(page, store: dict, cap: int = SESSION_BYTE_CAP):
    """Apply CDP 20Mbps emulation via nodriver generator API; record config.

    Imports nodriver lazily so unit tests can inject a stub module; the real
    package lives in tools/browser_deps on b660 (never in the venv).
    cdp/network handler registration uses page.add_handler on the Tab object
    (nodriver 0.50.6: add_handler lives on Connection/Tab, NOT on a separate
    CDP session object). Returns True ONLY when emulation applied.
    """
    cfg = {"offline": False, "latency": 0,
           "download_throughput": CDP_THROUGHPUT_BPS,
           "upload_throughput": CDP_UPLOAD_BPS}
    try:
        import nodriver
        cdp = nodriver.cdp
        await page.send(cdp.network.enable())
        await page.send(cdp.network.emulate_network_conditions(
            offline=False, latency=0,
            download_throughput=CDP_THROUGHPUT_BPS,
            upload_throughput=CDP_UPLOAD_BPS))
        store["cdp_emulation"] = dict(cfg, applied=True)
        return True
    except Exception as e:
        store["cdp_emulation"] = dict(cfg, applied=False,
                                      error=f"{type(e).__name__}: {e}"[:160])
        return False
    finally:
        store["session_cap"] = cap
        store["session_bytes"] = 0


async def _open_blank_tab(browser, timeout_s: float):
    """Start from a blank tab (about:blank); never navigate before wiring."""
    page = await asyncio.wait_for(browser.get("about:blank"), timeout=timeout_s)
    await asyncio.wait_for(page.sleep(0.5), timeout=timeout_s)
    return page


async def _root_frame_id(page) -> str | None:
    """Actual root frame id via Page.get_frame_tree; None on failure."""
    try:
        import nodriver
        tree = await page.send(nodriver.cdp.page.get_frame_tree())
        frame = getattr(tree, "frame", None)
        fid = getattr(frame, "id_", None) or getattr(frame, "id", None)
        return str(getattr(fid, "value", fid)) if fid is not None else None
    except Exception:
        return None


async def _dom_title(page, timeout_s: float = 10.0) -> str:
    """Title via evaluate('document.title'); nodriver Tab has no get_title."""
    try:
        res = await asyncio.wait_for(page.evaluate("document.title"), timeout=timeout_s)
        return str(res or "")[:200]
    except Exception:
        return ""


def strict_scan(path: Path) -> dict:
    """Strict zstd scan: decompressobj + .eof per frame, schema/status/text/dup."""
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    obj = None
    frames = 0
    out_bytes = 0
    zstd_err = None
    buf = bytearray()
    pos = 0
    records = 0
    seen: set = set()
    dups = 0
    schema_bad = 0
    content_bad = 0

    def feed(out: bytes):
        nonlocal pos, records, dups, schema_bad, content_bad
        if not out:
            return
        buf.extend(out)
        while True:
            nl = buf.find(10, pos)
            if nl < 0:
                break
            raw = bytes(buf[pos:nl])
            pos = nl + 1
            records += 1
            try:
                rec = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                schema_bad += 1
                continue
            if (not isinstance(rec, dict) or tuple(rec) != SCHEMA
                    and (set(rec) != set(SCHEMA))):
                schema_bad += 1
                continue
            if [k for k in SCHEMA if k not in rec] or [k for k in rec if k not in SCHEMA]:
                schema_bad += 1
                continue
            rid = rec.get("id")
            if not isinstance(rid, int) or isinstance(rid, bool):
                schema_bad += 1
                continue
            if rid in seen:
                dups += 1
            else:
                seen.add(rid)
            text = rec.get("text")
            if rec.get("status") != 200 or not isinstance(text, str) or len(text) < 200:
                content_bad += 1
        if pos:
            del buf[:pos]
            pos = 0

    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                data = chunk
                while data:
                    if obj is None:
                        obj = dctx.decompressobj()
                    out = obj.decompress(data)
                    out_bytes += len(out)
                    feed(out)
                    if obj.eof:
                        frames += 1
                        data = obj.unused_data
                        obj = None
                    else:
                        data = b""
        if obj is not None:
            out = obj.flush()
            out_bytes += len(out)
            feed(out)
    except zstd.ZstdError as e:
        zstd_err = f"{type(e).__name__}: {e}"[:200]
    if zstd_err is None and obj is not None and not obj.eof:
        zstd_err = "empty_or_no_frame" if (frames == 0 and out_bytes == 0) else "truncated_final_frame"
    if zstd_err is None and frames == 0:
        zstd_err = "empty_or_no_frame"
    tail = bytes(buf)
    ok = (zstd_err is None and dups == 0 and schema_bad == 0 and content_bad == 0)
    return {"files": 1, "records": records, "unique": len(seen), "duplicates": dups,
            "schema_bad": schema_bad, "content_bad": content_bad,
            "frames": frames, "tail_bytes": len(tail),
            "zstd_error": zstd_err, "ok": ok, "ids": sorted(seen)}


async def _start_browser(crawl_root: Path, chrome_exe: str | None, timeout_s: float):
    """Fresh nodriver browser per URL: isolated start, caller must stop it."""
    sys.path.insert(0, str(crawl_root / "tools" / "browser_deps"))
    import nodriver as nd
    kw = {"headless": True,
          "browser_args": ["--blink-settings=imagesEnabled=false",
                           "--disable-features=MediaPreloading",
                           "--disable-features=Translate",
                           "--blink-settings=fontsEnabled=false"]}
    if chrome_exe:
        kw["browser_executable_path"] = chrome_exe
    return await asyncio.wait_for(nd.start(**kw), timeout=timeout_s)


async def _stop_browser(browser) -> None:
    """Close ONLY our own pilot browser instance (never user Chrome)."""
    try:
        if browser is None:
            return
        res = browser.stop()
        if asyncio.iscoroutine(res):
            await res
    except Exception:
        pass


async def run_pilot(items: list, out_dir: Path, crawl_root: Path,
                    delay: float = 3.0, timeout_s: float = TIMEOUT_S,
                    browser_factory=None, chrome_exe: str | None = None,
                    lock_path: Path | str | None = None):
    t0 = time.time()
    lock = Path(lock_path) if lock_path is not None else Path(DEFAULT_LOCK)
    if delay < 3.0:
        raise ValueError("delay must be >= 3.0s")
    ids = [i.get("id") for i in items]
    if not ids or len(set(ids)) != len(ids) or any(i not in QUEUE_IDS for i in ids):
        raise ValueError("pilot queue must be a non-empty deduped subset of (1881723, 1900274)")
    if any(i.get("host") != "cnkang.com" for i in items):
        raise ValueError("pilot hosts must be cnkang.com only")
    pages, audit = out_dir / "pages", out_dir / "audit"
    if out_dir.exists() or out_dir.is_symlink():
        # A prior crashed attempt may leave an empty dir or a 0-byte shard
        # (crash before any write). Reuse the same VN-date dir only when it
        # holds no audit artifacts and no non-empty shard yet; otherwise
        # refuse to avoid duplicate work.
        nonempty = [p for p in out_dir.rglob("*")
                    if p.is_file() and p.stat().st_size > 0]
        if nonempty:
            raise FileExistsError(f"Refuse existing output: {out_dir}")
    pages.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    if stop_hit(crawl_root) or lock_hit(lock):
        (audit / "run.json").write_text(json.dumps(
            {"exit": 3, "stop": "stop_or_lock_at_startup",
             "lock": str(lock)}, ensure_ascii=False), encoding="utf-8")
        return 3
    lower_priority()
    # Import existing extraction for parity/fallback chain.
    from extract import classify  # noqa: E402
    import trafilatura  # noqa: E402

    session_bytes = 0
    aborted = False
    results = []
    nbytes_net = 0
    shard_path = pages / "browser-00000.jsonl.zst"
    if shard_path.exists() and shard_path.stat().st_size > 0:
        raise FileExistsError(f"Refuse existing shard: {shard_path}")
    if shard_path.exists():
        shard_path.unlink()
    fh = shard_path.open("xb")
    rec_count = 0
    browsers_started = 0
    browsers_stopped = 0
    try:
        zw = zstd.ZstdCompressor(level=6).stream_writer(fh)
        use_factory = browser_factory is not None
        if not use_factory:
            try:
                sys.path.insert(0, str(crawl_root / "tools" / "browser_deps"))
                import nodriver as nd  # noqa: F401 (version pin check)
                from importlib.metadata import version as _pkg_version
                if _pkg_version("nodriver") != NODRIVER_PIN.split("==")[1]:
                    raise ValueError("nodriver pin mismatch")
            except Exception as e:
                (audit / "run.json").write_text(json.dumps(
                    {"exit": 2, "error": f"nodriver unavailable pinned {NODRIVER_PIN}: {e}"},
                    ensure_ascii=False), encoding="utf-8")
                fh.close()
                return 2
        try:
            for item in items:
                if stop_hit(crawl_root) or lock_hit(lock):
                    results.append({"id": item["id"], "skipped": True, "cause": "stop_or_lock"})
                    aborted = True
                    break
                url, _id = item["url"], item["id"]
                events: list = []
                net = {"bytes": 0}
                abort = {"flag": False}
                capinfo: dict = {}
                t_start = time.time()
                page = None
                browser = None
                try:
                    # Fresh browser PER URL: the _c crash was a dead session
                    # reused for the second page (ProtocolException -32001).
                    if use_factory:
                        browser = browser_factory()
                    else:
                        browser = await _start_browser(crawl_root, chrome_exe, timeout_s)
                        browsers_started += 1
                    # (2) blank tab FIRST; wire typed handlers + cap BEFORE nav.
                    page = await asyncio.wait_for(_open_blank_tab(browser, timeout_s), timeout=timeout_s)
                    import nodriver
                    cdp = nodriver.cdp
                    cap_ok = await _cap_session(page, capinfo)
                    if not cap_ok:
                        results.append({"id": _id, "url": url, "ok": False,
                                        "error": "cdp-cap-not-applied-fail-closed",
                                        "cdp": capinfo})
                        aborted = True
                        break
                    root_frame = await _root_frame_id(page)
                    loader = {"id": None}
                    # Register data handlers FIRST so no byte event is missed,
                    # then the typed main-frame gate.
                    page.add_handler(cdp.network.LoadingFinished,
                                     _mk_bytes_cb(net, SESSION_BYTE_CAP, abort, "LoadingFinished"))
                    page.add_handler(cdp.network.DataReceived,
                                     _mk_bytes_cb(net, SESSION_BYTE_CAP, abort, "DataReceived"))
                    page.add_handler(cdp.network.ResponseReceived,
                                     _mk_cb(events, root_frame, loader))
                    await asyncio.sleep(1.0)
                    # Navigate the SAME tab (no auto new-tab background nav).
                    nav = await asyncio.wait_for(page.send(cdp.page.navigate(url)), timeout=timeout_s)
                    try:
                        lid = getattr(nav, "loader_id", None)
                        loader["id"] = str(getattr(lid, "value", lid)) if lid is not None else None
                    except Exception:
                        loader["id"] = None
                    await asyncio.wait_for(page.sleep(2), timeout=timeout_s)
                    if abort["flag"] or session_bytes + net["bytes"] > SESSION_BYTE_CAP:
                        try:
                            await page.send(cdp.page.stop_loading())
                        except Exception:
                            pass
                        try:
                            await page.close()
                        except Exception:
                            pass
                        results.append({"id": _id, "url": url, "ok": False,
                                        "error": "session-byte-cap-exceeded",
                                        "cdp": capinfo})
                        aborted = True
                        break
                    html = await asyncio.wait_for(page.get_content(), timeout=timeout_s)
                    raw = (html or "").encode("utf-8", "ignore")[:BODY_CAP]
                    dom_bytes = len(raw)
                    session_bytes += net["bytes"]
                    nbytes_net += net["bytes"]
                    status, final_url = probe.pick_main_response(events, url)
                    cdp_present = bool(events)
                    if not cdp_present or status is None:
                        # (4) fail-closed: no typed main-frame evidence -> unknown, never fake 200.
                        entry = {"id": _id, "url": url, "host": "cnkang.com",
                                 "source": {"id": _id, "url": url,
                                            "old_error": "extract:JSONDecodeError"},
                                 "method": "nodriver-pilot+unknown-no-cdp",
                                 "provenance": "browser_retry pilot b660 origin IP, no proxy",
                                 "final_url": url, "status": None,
                                 "class": "transient", "dom_bytes": dom_bytes,
                                 "nbytes_network": net["bytes"],
                                 "text_len": 0, "title": "",
                                 "cdp": capinfo,
                                 "nav_evidence": [dict(e) for e in events],
                                 "elapsed": round(time.time() - t_start, 2),
                                 "ok": False, "error": "cdp-main-frame-absent"}
                        results.append(entry)
                        try:
                            await page.close()
                        except Exception:
                            pass
                        await asyncio.sleep(delay)
                        continue
                    # (5) DOM title via evaluate, never get_title/get_cdp_session.
                    title = await _dom_title(page)
                    method = "json"
                    fb_err = None
                    try:
                        ex = probe.parse_main_html(raw, final_url or url)
                    except json.JSONDecodeError as e:
                        # (6) only JSONDecodeError triggers txt fallback.
                        method = "txt-fallback"
                        fb_err = f"JSONDecodeError: {e}"[:160]
                        fb = fallback_text(raw, final_url or url)
                        ex = {"title": fb.get("title", ""), "date": "",
                              "text": fb.get("text", "")}
                        title = title or fb.get("title", "")
                    text = ex.get("text", "")
                    if len(text) < 200 and method == "json":
                        fb = fallback_text(raw, final_url or url)
                        if len(fb.get("text", "")) > len(text):
                            method = "txt-fallback"
                            text = fb.get("text", "")
                            title = title or fb.get("title", "")
                            fb_err = (fb.get("fallback_error")
                                      or "short-json-text-fallback")
                    cls = classify(status, final_url or "", text)
                    entry = {"id": _id, "url": url, "host": "cnkang.com",
                             "source": {"id": _id, "url": url,
                                        "old_error": "extract:JSONDecodeError"},
                             "method": f"nodriver-pilot+{method}",
                             "provenance": "browser_retry pilot b660 origin IP, no proxy",
                             "final_url": final_url, "status": status,
                             "class": cls, "dom_bytes": dom_bytes,
                             "nbytes_network": net["bytes"],
                             "text_len": len(text), "title": title[:80],
                             "cdp": capinfo,
                             "nav_evidence": [dict(e) for e in events],
                             "elapsed": round(time.time() - t_start, 2)}
                    if fb_err:
                        entry["source_error_brief"] = fb_err
                    if cls == "blocked":
                        entry["ok"] = False
                        results.append(entry)
                        break
                    if cls != "ok" or status != 200:
                        entry["ok"] = False
                        results.append(entry)
                    else:
                        rec = {"id": _id, "url": url, "host": "cnkang.com",
                               "final_url": final_url, "status": status,
                               "title": title, "date": "", "text": text,
                               "ts": int(time.time())}
                        if tuple(rec) != SCHEMA:
                            raise ValueError("schema drift")
                        zw.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
                        rec_count += 1
                        entry["ok"] = True
                        results.append(entry)
                except Exception as e:
                    results.append({"id": _id, "url": url, "ok": False,
                                    "error": f"{type(e).__name__}: {e}"[:200],
                                    "cdp": capinfo,
                                    "nav_evidence": [dict(e2) for e2 in events]})
                    break
                finally:
                    try:
                        if page is not None:
                            await page.close()
                    except Exception:
                        pass
                    # Per-URL browser cleanup: no leaked Chrome, own instance only.
                    try:
                        if browser is not None:
                            await _stop_browser(browser)
                            browsers_stopped += 1
                    except Exception:
                        pass
                await asyncio.sleep(delay)
        finally:
            pass
        zw.flush(zstd.FLUSH_FRAME)
        zw.close()
    except Exception:
        try:
            fh.close()
        except Exception:
            pass
        raise
    with (audit / "results.jsonl").open("x", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = audit / "manifest.sha256"
    with manifest.open("x", encoding="utf-8") as f:
        f.write(f"{sha256_of(shard_path)}  {shard_path.name}\n")
    # Strict re-scan: decompressobj + eof, schema + 200 + text>=200, dup check.
    strict = strict_scan(shard_path)
    ok_ids = [r["id"] for r in results if r.get("ok")]
    exit_code = 0 if (strict["ok"] and not aborted) else 1
    (audit / "run.json").write_text(json.dumps({
        "exit": exit_code, "batch": out_dir.name, "queue": [i.get("id") for i in items],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        "config": {"delay": delay, "timeout": timeout_s, "body_cap": BODY_CAP,
                   "session_cap": SESSION_BYTE_CAP,
                   "cdp_download_bps": CDP_THROUGHPUT_BPS,
                   "cdp_upload_bps": CDP_UPLOAD_BPS,
                   "nodriver": NODRIVER_PIN, "chrome_exe": chrome_exe,
                   "proxy": False, "headless": True},
        "results": len(results), "ok": sum(1 for r in results if r.get("ok")),
        "records": rec_count, "strict": strict, "ok_ids": ok_ids,
        "nbytes_network": nbytes_net,
        "browsers": {"started": browsers_started, "stopped": browsers_stopped},
        "lock": str(lock),
        "manifest_sha256": sha256_of(manifest),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"exit": exit_code, "ok": sum(1 for r in results if r.get("ok")),
                      "records": rec_count, "strict_ok": strict["ok"]}, ensure_ascii=False))
    return exit_code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", required=True, help="JSONL with id/url/host for the 2 pilot IDs")
    ap.add_argument("--out-root", type=Path, default=Path(r"D:\r2ai\data\crawl"))
    ap.add_argument("--batch", default=None)
    ap.add_argument("--crawl-root", type=Path, default=Path(r"D:\r2ai\data\crawl"))
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--chrome-exe", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    ap.add_argument("--lock-path", default=DEFAULT_LOCK)
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    items = [json.loads(x) for x in Path(a.urls).read_text(encoding="utf-8").splitlines() if x.strip()]
    batch = a.batch or time.strftime("%Y%m%d", time.localtime())
    out_dir = a.out_root / f"browser_retry_{batch}"
    if sys.platform == "win32":
        # ProactorEventLoop is REQUIRED for asyncio subprocesses (nodriver
        # launches Chrome). WindowsSelectorEventLoopPolicy breaks this with
        # NotImplementedError on subprocess_exec.
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        return asyncio.run(run_pilot(items, out_dir, a.crawl_root, a.delay, a.timeout,
                                     chrome_exe=a.chrome_exe or None,
                                     lock_path=a.lock_path))
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"[pilot] error: {exc}", flush=True)
        return 2
    except KeyboardInterrupt:
        print("[pilot] interrupted", flush=True)
        return 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
