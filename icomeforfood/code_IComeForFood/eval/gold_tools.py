"""Build the extraction gold set (PHAN_CONG.md section 8) without using any extractor.

The gold text must not come from trafilatura or any other extractor we later grade,
otherwise we would grade a tool against its own output. Instead:

    fetch   download the raw HTML of every page in manifest.csv
    dump    split each page into ALL visible text blocks (menus, footers included),
            numbered, plus a compact review view for a human to read
    write   a reviewer picks block ranges per page in selection.json; this step copies
            those blocks verbatim into <id>.txt in the format score_extraction.py reads

selection.json:
    {"<id>": {"status": "ok" | "dead" | "js_required" | ...,
              "keep": [[first, last], ...],      # inclusive block index ranges
              "drop": [i, ...],                  # blocks inside the ranges to leave out
              "headings": [i, ...], "required": [i, ...],
              "note": "free text"}}

Usage:
    python -I eval/gold_tools.py fetch --gold D:/r2ai_data/gold
    python -I eval/gold_tools.py fetch --gold D:/r2ai_data/gold --browser --only 708925
        (headless Chrome, for pages that answer 403/203 to plain requests: vov.vn, PubMed)
    python -I eval/gold_tools.py dump  --gold D:/r2ai_data/gold
    python -I eval/gold_tools.py write --gold D:/r2ai_data/gold
"""
from __future__ import annotations

import argparse
import csv
import html as htmllib
import json
import re
import sys
import tempfile
import time
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "head",
             "select", "option", "button", "input", "textarea", "canvas", "video",
             "audio", "picture", "img", "map", "object", "embed"}
BLOCK_TAGS = {"address", "article", "aside", "blockquote", "body", "br", "caption",
              "center", "dd", "details", "dialog", "div", "dl", "dt", "fieldset",
              "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
              "h6", "header", "hr", "html", "li", "main", "menu", "nav", "ol", "p",
              "pre", "section", "summary", "table", "tbody", "td", "tfoot", "th",
              "thead", "tr", "ul"}
_WS = re.compile(r"\s+")
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?([\w-]+)""", re.I)


def read_manifest(gold: Path) -> list[dict]:
    with open(gold / "manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# laodong.vn and similar: a tiny page that only sets a cookie and reloads
_COOKIE_JS = re.compile(rb'document\.cookie\s*=\s*"([\w-]+)=([^";]+)')
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")


def fetch_browser(url: str, profile: Path) -> bytes:
    """Rendered DOM via headless Chrome (passes cookie/JS checks such as PubMed's)."""
    import subprocess

    r = subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--no-first-run",
                        f"--user-data-dir={profile}", "--virtual-time-budget=15000",
                        "--dump-dom", url], capture_output=True, timeout=120)
    return r.stdout


def cmd_fetch(gold: Path, only: set[str] | None, delay: float, browser: bool) -> None:
    import requests

    out = gold / "html"
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA,
                      "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"})
    for m in read_manifest(gold):
        if only and m["id"] not in only:
            continue
        meta = {"id": m["id"], "url": m["url"]}
        try:
            if browser:
                # profile outside the gold folder: it holds cookies and is not data
                body = fetch_browser(m["url"], Path(tempfile.gettempdir()) / "r2ai_chrome_profile")
                (out / f"{m['id']}.html").write_bytes(body)
                meta.update(status="browser", final_url=m["url"],
                            content_type="text/html; charset=utf-8", n_bytes=len(body))
            else:
                r = s.get(m["url"], timeout=30, allow_redirects=True)
                ck = _COOKIE_JS.search(r.content) if len(r.content) < 2000 else None
                if ck:
                    s.cookies.set(ck.group(1).decode(), ck.group(2).decode())
                    r = s.get(m["url"], timeout=30, allow_redirects=True)
                    meta["cookie_challenge"] = ck.group(1).decode()
                (out / f"{m['id']}.html").write_bytes(r.content)
                meta.update(status=r.status_code, final_url=r.url,
                            content_type=r.headers.get("content-type"),
                            n_bytes=len(r.content),
                            redirects=[h.url for h in r.history])
        except Exception as e:  # noqa: BLE001 - record any network failure
            meta.update(status=None, error=repr(e))
        (out / f"{m['id']}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(m["id"], meta.get("status"), meta.get("n_bytes"), meta.get("final_url"),
              meta.get("error", ""))
        time.sleep(delay)


def decode(raw: bytes, content_type: str | None) -> tuple[str, str]:
    enc = None
    if content_type and "charset=" in content_type.lower():
        enc = content_type.lower().split("charset=")[-1].split(";")[0].strip(" \"'")
    if not enc:
        m = _META_CHARSET.search(raw[:4096])
        enc = m.group(1).decode("ascii", "ignore").lower() if m else None
    if not enc:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        enc = best.encoding if best else "utf-8"
    if enc.replace("-", "").replace("_", "") in {"gb2312", "gbk", "gb18030", "cp936"}:
        enc = "gb18030"
    return raw.decode(enc, errors="replace"), enc


def hidden(el) -> bool:
    style = (el.get("style") or "").replace(" ", "").lower()
    return ("display:none" in style or "visibility:hidden" in style
            or el.get("hidden") is not None or el.get("aria-hidden") == "true")


def label(el) -> str:
    parts = []
    while el is not None and len(parts) < 3:
        ident = el.get("id")
        cls = (el.get("class") or "").split()
        if ident or cls:
            parts.append(el.tag + (f"#{ident}" if ident else "") +
                         (f".{cls[0]}" if cls else ""))
        el = el.getparent()
    return " < ".join(parts)


def blocks_of(page: str) -> list[dict]:
    """All visible text blocks in document order (no main-content guessing)."""
    import lxml.html

    page = re.sub(r"^\s*<\?xml[^>]*\?>", "", page)  # lxml rejects str with encoding decl
    # a stray </html> before <body> (baotayninh.vn) makes lxml drop the whole page; browsers ignore it
    body = re.search(r"<body[\s>]", page, re.I)
    if body and re.search(r"</(html|body)\s*>", page[:body.start()], re.I):
        page = re.sub(r"</(html|body)\s*>", "", page, flags=re.I)
    root = lxml.html.document_fromstring(page)
    blocks: list[dict] = []
    buf: list[tuple[str, bool]] = []  # (text piece, inside <a>)
    state = {"el": root}

    def flush():
        text = _WS.sub(" ", "".join(t for t, _ in buf)).strip()
        if text:
            link_chars = sum(len(t.strip()) for t, a in buf if a)
            blocks.append({"i": len(blocks), "tag": state["el"].tag,
                           "where": label(state["el"]), "text": htmllib.unescape(text),
                           "link": round(link_chars / max(1, len(text)), 2)})
        buf.clear()

    def walk(el, in_a: bool):
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in SKIP_TAGS or not tag or hidden(el):
            if el.tail:
                buf.append((el.tail, in_a))
            return
        is_block = tag in BLOCK_TAGS
        if is_block:
            flush()
            state["el"] = el
        if el.text:
            buf.append((el.text, in_a or tag == "a"))
        for child in el:
            walk(child, in_a or tag == "a")
        if is_block:
            flush()
            state["el"] = el.getparent() if el.getparent() is not None else el
        if el.tail:
            buf.append((el.tail, in_a))

    walk(root, False)
    flush()
    return blocks


def review_view(blocks: list[dict], width: int) -> str:
    """Compact listing; runs of >= 4 link-only blocks collapse into one line."""
    lines, run = [], []

    def close_run():
        if len(run) >= 4:
            sample = " | ".join(b["text"][:20] for b in run[:4])
            lines.append(f"[{run[0]['i']}-{run[-1]['i']}] ~{len(run)} link blocks: {sample} ...")
        else:
            lines.extend(fmt(b) for b in run)
        run.clear()

    def fmt(b):
        t = b["text"] if len(b["text"]) <= width else b["text"][:width] + f"…(+{len(b['text']) - width})"
        return f"{b['i']:4d} {b['tag']:<10s} {b['where'][:40]:<40s} {t}"

    for b in blocks:
        if b["link"] >= 0.9 and len(b["text"]) < 80:
            run.append(b)
            continue
        close_run()
        lines.append(fmt(b))
    close_run()
    return "\n".join(lines)


def cmd_dump(gold: Path, only: set[str] | None, width: int) -> None:
    out = gold / "blocks"
    out.mkdir(parents=True, exist_ok=True)
    for m in read_manifest(gold):
        if only and m["id"] not in only:
            continue
        hp = gold / "html" / f"{m['id']}.html"
        if not hp.exists() or hp.stat().st_size == 0:
            print(m["id"], "no html")
            continue
        meta = json.loads((gold / "html" / f"{m['id']}.meta.json").read_text(encoding="utf-8"))
        page, enc = decode(hp.read_bytes(), meta.get("content_type"))
        blocks = blocks_of(page)
        (out / f"{m['id']}.json").write_text(
            json.dumps({"id": m["id"], "encoding": enc, "blocks": blocks},
                       ensure_ascii=False, indent=0), encoding="utf-8")
        header = (f"# {m['id']} {m['domain']} status={meta.get('status')} enc={enc} "
                  f"final={meta.get('final_url')} n_blocks={len(blocks)}\n")
        (out / f"{m['id']}.review.txt").write_text(header + review_view(blocks, width),
                                                   encoding="utf-8")
        print(m["id"], enc, len(blocks), "blocks", "bad_chars=", page.count("�"))


def cmd_write(gold: Path) -> None:
    sel = json.loads((gold / "selection.json").read_text(encoding="utf-8"))
    for pid, s in sel.items():
        if pid.startswith("_"):
            continue
        txt = gold / f"{pid}.txt"
        if s.get("status", "ok") != "ok":
            txt.unlink(missing_ok=True)
            print(pid, "skipped:", s.get("status"), s.get("note", ""))
            continue
        blocks = json.loads((gold / "blocks" / f"{pid}.json").read_text(encoding="utf-8"))["blocks"]
        drop, heads, req = set(s.get("drop", [])), set(s.get("headings", [])), set(s.get("required", []))
        parts = []
        for a, b in s["keep"]:
            for i in range(a, b + 1):
                if i in drop:
                    continue
                t = blocks[i]["text"]
                if t.startswith(("#", "!! ")):
                    raise ValueError(f"{pid} block {i} starts with a marker character")
                if i in heads:
                    t = "# " + t
                if i in req:
                    t = "!! " + t
                parts.append(t)
        missing = (heads | req) - {i for a, b in s["keep"] for i in range(a, b + 1)} - drop
        if missing:
            raise ValueError(f"{pid}: heading/required blocks outside keep ranges: {sorted(missing)}")
        txt.write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
        print(pid, len(parts), "blocks,", sum(len(p) for p in parts), "chars")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "dump", "write"])
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--browser", action="store_true", help="fetch via headless Chrome")
    ap.add_argument("--width", type=int, default=110, help="chars per block in review view")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    if args.cmd == "fetch":
        cmd_fetch(args.gold, only, args.delay, args.browser)
    elif args.cmd == "dump":
        cmd_dump(args.gold, only, args.width)
    else:
        cmd_write(args.gold)


if __name__ == "__main__":
    main()
