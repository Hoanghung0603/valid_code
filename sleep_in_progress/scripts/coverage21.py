"""Độ phủ text của kho URL mới (links_corpus_vi_en_zh.parquet, commit 12/10) trên các nguồn text ta có.

Nguồn id có text (chỉ ĐỌC):
  - chỉ mục VI: ids.npy của `--vi-index-ids`
  - kho ZH đã embed: id trong các part JSONL.zst {id, idx, text} của `--zh-emb-parts`
    (mặc định quét *.jsonl.zst trong thư mục; đối chiếu với ids.npy của chỉ mục ZH nếu có)
  - shard text `--corpus-dir` (*.jsonl.zst): bản ghi status 200 & text >= --min-chars ký tự

Phân loại id kho mới theo host (URL): pubmed.ncbi -> en; host ZH (*.39.net, *.cn, familydoctor*,
120ask, cnkang, zysjonline, a-hospital, zhongyibaodian, youlai, wujue, iiyi, jb39, zydcd, baidu,
qihuangzhishu, test.pmphai.com) -> zh; còn lại -> vi. Cách phân loại này cho đúng
740.598 vi / 240.000 zh / 240.000 en như README; đối chiếu thêm vote detect_lang từ text shard.

Kết quả (--out-dir):
  coverage.json            tổng/có text/thiếu theo vi/zh/en, top host thiếu, id VI có text nhưng
                           không nằm trong chỉ mục VI, vote ngôn ngữ theo host, sanity bài nộp
  missing_vi_by_host.json  {host: [id còn thiếu]} cho id VI
  missing_zh_by_host.json  {host: [id còn thiếu]} cho id ZH
  new_zh_ids.json          danh sách id ZH kho mới (đầu vào cho zhnew_dense.py)

Dạng:
  python coverage21.py --new NEW.parquet --corpus-dir DIR --vi-index-ids ids.npy \
      --zh-emb-parts DIR [--zh-index-ids ids.npy] [--sub SUB.json] --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pyarrow.parquet as pq

from chunking import detect_lang
from shards import read_shard

_ZH_EXACT = {
    "120ask.com", "cnkang.com", "zysjonline.com", "a-hospital.com",
    "zhongyibaodian.net", "youlai.cn", "wujue.com", "zydcd.com",
    "baidu.com", "qihuangzhishu.com", "jb39.com", "msdmanuals.cn",
    "health.people.com.cn", "test.pmphai.com",
}
_ZH_PARTS = ("39.net", "iiyi.com", "familydoctor.")
_ID_RE = re.compile(r'^\{"id":\s*(\d+),')


def classify_host(host: str) -> str:
    """'en' | 'zh' | 'vi' theo host URL (netloc)."""
    h = (host or "").lower()
    if "pubmed.ncbi" in h:
        return "en"
    if h.endswith(".cn") or h.endswith(".com.cn"):
        return "zh"
    if any(p in h for p in _ZH_PARTS):
        return "zh"
    if h.removeprefix("www.") in _ZH_EXACT:
        return "zh"
    return "vi"


def split_corpus(ids: list, urls: list) -> dict[str, dict[int, str]]:
    """{lang: {id: url}} theo classify_host."""
    out: dict[str, dict[int, str]] = {"vi": {}, "zh": {}, "en": {}}
    for i, u in zip(ids, urls):
        out[classify_host(urllib.parse.urlparse(u).netloc)][int(i)] = u
    return out


def host_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def scan_records(records, min_chars: int = 200) -> dict:
    """Quét bản ghi shard: tập id có text, vote ngôn ngữ theo host, số bản ghi.

    text_ok: status 200 & len(text) >= min_chars; text_any: status 200 & len(text) >= 1.
    """
    text_ok: set[int] = set()
    text_any: set[int] = set()
    votes: dict[str, Counter] = defaultdict(Counter)
    n = 0
    for r in records:
        did = r.get("id")
        if did is None:
            continue
        n += 1
        text = r.get("text") or ""
        if r.get("status") != 200 or not text:
            continue
        text_any.add(int(did))
        if len(text) >= min_chars:
            text_ok.add(int(did))
        votes[r.get("host") or ""][detect_lang(text)] += 1
    return {"text_ok": text_ok, "text_any": text_any,
            "votes": {h: dict(c) for h, c in votes.items()}, "records": n}


def iter_lines_zst(path: Path):
    """Yield từng dòng text của file zstd JSONL (không parse JSON)."""
    import zstandard
    with path.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as r:
            buf = b""
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    yield ln.decode("utf-8", errors="ignore")
            if buf.strip():
                yield buf.decode("utf-8", errors="ignore")


def zh_emb_ids(parts_dir: Path, pattern: str = "docs_zh_all.part*.jsonl.zst") -> tuple[set[int], int]:
    """(tập id, số dòng) đọc id từ các part JSONL.zst {id, idx, text} của kho ZH đã embed."""
    ids: set[int] = set()
    n = 0
    for p in sorted(parts_dir.glob(pattern)):
        for ln in iter_lines_zst(p):
            m = _ID_RE.match(ln)
            if m:
                ids.add(int(m.group(1)))
                n += 1
    return ids, n


def load_ids_npy(path: Path):
    import numpy as np
    return {int(x) for x in np.load(path)}


def missing_by_host(id_urls: dict[int, str], have: set[int]) -> dict[str, list[int]]:
    """{host: [id còn thiếu]} (host theo URL kho mới), id tăng dần trong từng host."""
    out: dict[str, list[int]] = defaultdict(list)
    for i, u in id_urls.items():
        if i not in have:
            out[host_of(u)].append(i)
    return {h: sorted(v) for h, v in sorted(out.items(), key=lambda kv: -len(kv[1]))}


def host_table(id_urls: dict[int, str], have: set[int], top: int = 15) -> dict:
    """Bảng theo host: tổng / có text / thiếu, xếp theo số thiếu giảm dần."""
    tot: Counter = Counter()
    ok: Counter = Counter()
    for i, u in id_urls.items():
        h = host_of(u)
        tot[h] += 1
        if i in have:
            ok[h] += 1
    rows = [{"host": h, "total": tot[h], "has_text": ok[h], "missing": tot[h] - ok[h]} for h in tot]
    rows.sort(key=lambda r: (-r["missing"], r["host"]))
    return {"top_missing": rows[:top], "top_covered": sorted(rows, key=lambda r: -r["has_text"])[:top],
            "n_hosts": len(rows), "all": rows}


def lang_block(id_urls: dict[int, str], text_ok: set[int], text_any: set[int],
               index_ids: set[int] | None) -> dict:
    n = len(id_urls)
    have = set(id_urls) & text_ok
    have_any = set(id_urls) & text_any
    b = {
        "total": n,
        "has_text": len(have),
        "has_text_frac": round(len(have) / n, 4) if n else 0.0,
        "has_any_text": len(have_any),
        "missing": n - len(have),
    }
    if index_ids is not None:
        b["in_index"] = len(set(id_urls) & index_ids)
        b["text_not_in_index"] = len(have - index_ids)
    return b


def sub_text_fraction(sub: list, text_ok: set[int]) -> dict:
    """Tỉ lệ doc của bài nộp thuộc tập có text (unique + theo lần nhắc), TB/câu."""
    n_ment = n_in = 0
    uniq: set[int] = set()
    per_q = []
    for r in sub:
        docs = [int(d) for d in r.get("relevant_docs", [])]
        uniq.update(docs)
        k = sum(1 for d in docs if d in text_ok)
        n_ment += len(docs)
        n_in += k
        per_q.append(k / len(docs) if docs else 1.0)
    return {
        "n_queries": len(sub),
        "n_doc_mentions": n_ment,
        "n_doc_unique": len(uniq),
        "frac_mentions_in_text": round(n_in / n_ment, 6) if n_ment else 0.0,
        "frac_unique_in_text": round(len(uniq & text_ok) / len(uniq), 6) if uniq else 0.0,
        "macro_frac_in_text": round(sum(per_q) / len(per_q), 6) if per_q else 0.0,
    }


def coverage_report(new: dict[str, dict[int, str]], text_ok: set[int], text_any: set[int],
                    vi_index: set[int] | None, zh_index: set[int] | None,
                    zh_emb: set[int] | None, votes: dict[str, dict[str, int]],
                    sub: list | None) -> dict:
    rep: dict = {
        "new_corpus": {
            "total": sum(len(v) for v in new.values()),
            "by_lang": {k: len(v) for k, v in new.items()},
            "readme_expected": {"vi": 740598, "zh": 240000, "en": 240000},
        },
        "text_sets": {
            "text_ok": len(text_ok),
            "text_any": len(text_any),
            "vi_index": len(vi_index) if vi_index is not None else None,
            "zh_index": len(zh_index) if zh_index is not None else None,
            "zh_embedded": len(zh_emb) if zh_emb is not None else None,
        },
        "vi": lang_block(new["vi"], text_ok, text_any, vi_index),
        "zh": lang_block(new["zh"], text_ok, text_any, zh_index),
        "en": lang_block(new["en"], text_ok, text_any, None),
    }
    exp = rep["new_corpus"]["readme_expected"]
    rep["new_corpus"]["match_readme"] = (rep["new_corpus"]["by_lang"] == exp)
    rep["vi"]["hosts"] = host_table(new["vi"], text_ok)
    rep["zh"]["hosts"] = host_table(new["zh"], text_ok)
    if zh_emb is not None:
        inter = set(new["zh"]) & zh_emb
        rep["zh"]["in_zh_embedded"] = len(inter)
        rep["zh"]["embedded_not_in_new_zh"] = len(zh_emb - set(new["zh"]))
        rep["zh_intersection_embedded"] = len(inter)

    vi_not_idx = {i: u for i, u in new["vi"].items() if i in text_ok and (vi_index is None or i not in vi_index)}
    rep["vi_text_not_in_index"] = {
        "n": len(vi_not_idx),
        "by_host": {h: ids[:500] for h, ids in missing_by_host(vi_not_idx, set()).items()},
        "by_host_n": {h: len(ids) for h, ids in missing_by_host(vi_not_idx, set()).items()},
    }

    vote_rows = []
    for h in sorted(set(votes) & ({host_of(u) for u in new["vi"].values()} | {host_of(u) for u in new["zh"].values()})):
        c = votes[h]
        vi_v, zh_v = c.get("vi", 0), c.get("zh", 0)
        vote_rows.append({"host": h, "rule": classify_host(h), "vote_vi": vi_v, "vote_zh": zh_v,
                          "vote": "zh" if zh_v > vi_v else "vi"})
    agree = sum(1 for r in vote_rows if r["rule"] == r["vote"] or (r["vote_zh"] == 0 and r["vote_vi"] == 0))
    rep["host_lang_votes"] = {"n_hosts": len(vote_rows), "agree_with_rule": agree, "rows": vote_rows}

    if sub is not None:
        rep["sub_sanity"] = sub_text_fraction(sub, text_ok)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="links_corpus_vi_en_zh.parquet (id, url)")
    ap.add_argument("--corpus-dir", required=True, help="thư mục shard *.jsonl.zst (chỉ đọc)")
    ap.add_argument("--vi-index-ids", required=True)
    ap.add_argument("--zh-emb-parts", required=True, help="thư mục docs_zh_all.part*.jsonl.zst")
    ap.add_argument("--zh-index-ids", default="")
    ap.add_argument("--sub", default="", help="bài nộp để sanity tỉ lệ doc có text")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-chars", type=int, default=200)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(a.new, columns=["id", "url"]).to_pydict()
    new = split_corpus(t["id"], t["url"])
    print(f"kho mới: {sum(len(v) for v in new.values())} id "
          f"(vi {len(new['vi'])}, zh {len(new['zh'])}, en {len(new['en'])})", flush=True)

    vi_index = load_ids_npy(Path(a.vi_index_ids))
    zh_index = load_ids_npy(Path(a.zh_index_ids)) if a.zh_index_ids else None
    zh_emb, n_lines = zh_emb_ids(Path(a.zh_emb_parts))
    print(f"vi_index {len(vi_index)}, zh_index {len(zh_index) if zh_index is not None else '-'}, "
          f"zh_emb {len(zh_emb)} ({n_lines} dòng)", flush=True)

    scan = {"text_ok": set(), "text_any": set(), "votes": {}, "records": 0}
    shards = sorted(Path(a.corpus_dir).glob("*.jsonl.zst"))
    for k, p in enumerate(shards, 1):
        s = scan_records(read_shard(p), a.min_chars)
        scan["text_ok"] |= s["text_ok"]
        scan["text_any"] |= s["text_any"]
        scan["records"] += s["records"]
        for h, c in s["votes"].items():
            scan["votes"].setdefault(h, Counter()).update(c)
        print(f"shard {k}/{len(shards)} {p.name}: text_ok tổng {len(scan['text_ok'])}", flush=True)
    print(f"shard: {scan['records']} bản ghi, text_ok {len(scan['text_ok'])}, text_any {len(scan['text_any'])}",
          flush=True)

    sub = None
    if a.sub:
        sub = json.loads(Path(a.sub).read_text(encoding="utf-8"))
    rep = coverage_report(new, scan["text_ok"], scan["text_any"], vi_index, zh_index, zh_emb,
                          scan["votes"], sub)
    rep["text_sets"]["corpus_records"] = scan["records"]
    rep["text_sets"]["min_chars"] = a.min_chars

    (out_dir / "coverage.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "missing_vi_by_host.json").write_text(
        json.dumps(missing_by_host(new["vi"], scan["text_ok"]), ensure_ascii=False), encoding="utf-8")
    (out_dir / "missing_zh_by_host.json").write_text(
        json.dumps(missing_by_host(new["zh"], scan["text_ok"]), ensure_ascii=False), encoding="utf-8")
    (out_dir / "new_zh_ids.json").write_text(json.dumps(sorted(new["zh"])), encoding="utf-8")

    for lang in ("vi", "zh", "en"):
        b = rep[lang]
        print(f"{lang}: tổng {b['total']}, có text {b['has_text']} ({b['has_text_frac']}), thiếu {b['missing']}"
              + (f", trong index {b['in_index']}, text∉index {b['text_not_in_index']}" if "in_index" in b else ""),
              flush=True)
    print(f"zh giao với đã embed: {rep.get('zh_intersection_embedded')}", flush=True)
    if sub is not None:
        print(f"sanity #5532: {rep['sub_sanity']}", flush=True)
    print(f"ghi {out_dir}", flush=True)


if __name__ == "__main__":
    main()
