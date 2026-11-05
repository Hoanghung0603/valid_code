"""Build docs for zysjonline.com from the site's official data dump instead of crawling.

The site blocks automated clients with a Cloudflare managed challenge, but publishes its
whole database at https://zysjonline.com/download/ (zysj-data.zip, MySQL dump). The dump
tables are first converted to parquet (one file per table) in --dump-dir.

URL -> row mapping (checked on 14/10: 6,591 / 6,591 corpus URLs found):
    /books/<folder>/<id>/    zysjllsj.ID == id, zysjcell(Cell_ID == TypeID) folder == <folder>
    /columns/<folder>/<id>/  same as books (columns live in zysjllsj too)
    /articles/misc/<id>/     zysjzhsj.ID == id
    /formulas/<pinyin>/      zysjyj.PYFile == pinyin

Content markup -> blocks (mimics what the page shows):
    one line of NeiRong (\r\n, [br] or <br>) = one block; a line that is only [b]...[/b]
    = heading level 2; each HTML <tr> = one `table` block, cells joined by a tab;
    [imgz]file[alt]caption[/alt][/img] = caption block (alt text; dropped if no alt);
    markup [b] [i] [u] [XB] (subscript) [SB] (superscript) [size=..] and HTML tags are
    removed, entities decoded, whitespace collapsed. Other [X] are kept: they are content
    (chemical concentrations such as [S], [Na]).

Usage:
    python -I extract/zysjonline_dump.py --corpus D:/r2ai_data/raw/links_corpus_vi_en_zh.parquet \
        --dump-dir D:/r2ai_data/external/zysjonline --out D:/r2ai_data/docs/docs_zysjonline.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
from pathlib import Path

import pandas as pd

EXTRACTOR = "zysj_dump_v1"
SHORT_CHARS = 200  # DAC_TA section 7 starting value
_WS = re.compile(r"\s+")
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")
_BR = re.compile(r"<br\s*/?>|\[br\]", re.I)
_IMG = re.compile(r"\[img[a-z]*\](.*?)\[/img\]", re.I | re.S)
_ALT = re.compile(r"\[alt\](.*?)\[/alt\]", re.I | re.S)
# Only real markup. Many other [X] are content, e.g. concentrations [S], [Na], [E].
_BB = re.compile(r"\[/?(?:b|i|u|xb|sb|imgz?|alt)\]|\[size=[^\]]*\]|\[/size\]", re.I)
_BOLD_LINE = re.compile(r"^\s*\[b\](.*?)\[/b\]\s*$", re.I | re.S)
_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.I | re.S)
CELL_SEP = "\x1f"  # marks a table-row line until blocks are built


def clean(s: str) -> str:
    s = _BR.sub(" ", s)
    s = _BB.sub("", s)
    s = _HTML_TAG.sub("", s)
    return _WS.sub(" ", html.unescape(s)).strip()


def _row_line(m: re.Match) -> str:
    cells = [clean(c) for c in _CELL.findall(m.group(1))]
    return "\n" + CELL_SEP.join(cells) + "\n" if any(cells) else "\n"


def content_blocks(raw: str | None) -> list[tuple[str, int | None, str | None, str]]:
    """(type, level, role, text) for one NeiRong field."""
    if not raw:
        return []
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = _TR.sub(_row_line, raw)  # before [br] -> newline: a <br> inside a cell stays in the cell
    raw = _BR.sub("\n", raw)
    out = []
    for line in raw.split("\n"):
        if CELL_SEP in line:  # one table row = one block, cells joined by a tab (DAC_TA section 3)
            out.append(("table", None, None, "\t".join(line.split(CELL_SEP))))
            continue
        # images may sit inside a line of text: split them out in order
        pos = 0
        for m in _IMG.finditer(line):
            before = line[pos:m.start()]
            if clean(before):
                out.append(("paragraph", None, None, clean(before)))
            alt = _ALT.search(m.group(1))
            if alt and clean(alt.group(1)):
                out.append(("paragraph", None, "caption", clean(alt.group(1))))
            pos = m.end()
        rest = line[pos:]
        bold = _BOLD_LINE.match(rest)
        if bold and clean(bold.group(1)):
            out.append(("heading", 2, None, clean(bold.group(1))))
        elif clean(rest):
            out.append(("paragraph", None, None, clean(rest)))
    return out


def assemble(blocks: list[tuple[str, int | None, str | None, str]]) -> tuple[str, list[dict]]:
    text, bl, pos = "", [], 0
    for typ, level, role, t in blocks:
        if bl:
            text += "\n\n"
            pos += 2
        bl.append({"type": typ, "level": level, "role": role, "text": t,
                   "char_start": pos, "char_end": pos + len(t)})
        text += t
        pos += len(t)
    return text, bl


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    c = pd.read_parquet(args.corpus)
    z = c[c.url.str.contains("zysjonline.com", regex=False)].copy()
    parts = z.url.str.replace(r"^https?://[^/]+/", "", regex=True).str.strip("/").str.split("/")
    z["kind"], z["slug"] = parts.str[0], parts.str[1]
    z["pid"] = pd.to_numeric(parts.str[2], errors="coerce").astype("Int64")

    d = args.dump_dir
    cell = pd.read_parquet(d / "zysjcell.parquet")
    cell["folder"] = cell.Cell_WenJianJia.str.strip("/").str.split("/").str[-1]
    cell = cell.set_index("Cell_ID")
    ll = pd.read_parquet(d / "zysjllsj.parquet").set_index("ID")
    zh = pd.read_parquet(d / "zysjzhsj.parquet").set_index("ID")
    yj = pd.read_parquet(d / "zysjyj.parquet")

    rows, problems = [], []
    for r in z.itertuples():
        title, body, category, flags = None, [], None, []
        if r.kind in ("books", "columns") and r.pid in ll.index:
            row = ll.loc[r.pid]
            book = cell.loc[row.TypeID] if row.TypeID in cell.index else None
            if book is None or book.folder != r.slug:
                problems.append((r.id, r.url, "folder mismatch"))
            title, body = row.BiaoTi, content_blocks(row.NeiRong)
            category = None if book is None else book.Cell_BiaoTi
        elif r.kind == "articles" and r.pid in zh.index:
            row = zh.loc[r.pid]
            title, body, category = row.BiaoTi, content_blocks(row.NeiRong), "杂集"
        elif r.kind == "formulas" and (yj.PYFile == r.slug).any():
            row = yj[yj.PYFile == r.slug].iloc[0]
            title = row.MingCheng
            skip = {"TypeID", "ID", "NumberID", "PYFile", "TuPian", "MingCheng", "PinYin", "PinYinZT"}
            for col in yj.columns:
                if col not in skip and isinstance(row[col], str) and clean(row[col]):
                    body += content_blocks(row[col])
            flags.append("fields_without_labels")  # page labels (【组成】…) are not in the dump
        else:
            problems.append((r.id, r.url, "not in dump"))
            flags.append("not_in_dump")
        blocks = ([("heading", 1, None, clean(title))] if title and clean(title) else []) + body
        text, bl = assemble(blocks)
        if len(text) < SHORT_CHARS:
            flags.append("short")
        rows.append({
            "id": int(r.id), "url": r.url, "final_url": r.url, "lang": "zh", "lang_conf": None,
            "title": clean(title) if title else None, "published_date": None,
            "site": "zysjonline.com", "category": category, "text": text, "blocks": bl,
            "n_chars": len(text), "sha1_text": hashlib.sha1(text.encode("utf-8")).hexdigest(),
            "dup_group": None, "extractor": EXTRACTOR, "quality_flags": flags,
            "schema_version": 2,
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    bad = sum(t[c0:c1] != b for t, bl in zip(df.text, df.blocks)
              for b, c0, c1 in ((x["text"], x["char_start"], x["char_end"]) for x in bl))
    print(f"docs: {len(df)}  problems: {len(problems)}  bad offsets: {bad}")
    print("flags:", pd.Series([f for fl in df.quality_flags for f in fl]).value_counts().to_dict())
    print("n_chars median:", int(df.n_chars.median()), " total MB:", round(df.n_chars.sum() / 1e6, 1))
    for p in problems[:10]:
        print("  problem:", p)


if __name__ == "__main__":
    main()
