"""trim_chunks: cat gon chunk quanh cua so docwin tot nhat -> submission_cxtrim (W mac dinh 450 tu).

Moi chunk (cx130cov va ung vien chunk_dw_u900_ext) duoc dinh vi nguyen van [cs, ce) trong text doc
(text.find); lay cua so (qid, doc_id) giao chunk, chon cua so logit cao nhat co >= 50% ky tu trong
chunk; cat chunk thanh doan W tu can giua cua so do (chan bien chunk thi don sang ben kia), cat o
ranh gioi khoang trang; ket qua PHAI la chuoi con nguyen van cua text doc va cua chunk goc.
Chunk <= W tu giu nguyen; khong tim duoc text / khong co cua so thoa -> giu nguyen.

Dung do tiet kiem de them chunk: k'_q = min(k_q + delta, |C_q|), delta > 57, ung vien cung duoc cat
theo cung quy tac; chon delta lon nhat voi uoc zip LZMA <= max-zip (ti le JSON/zip cua cx130cov
592.098.628 / 92.767.156), nen that LZMA, vuot thi giam delta.

measure  do nhanh W in {350, 450, 550} tren chunk cua cx130cov: ti le chunk bi cat + byte JSON.
repro    kiem tai lap delta = 57 KHONG cat tren ung vien == submission_cx130cov (1200/1200 cau).
build    kiem tai lap + chon delta + dung submission_cxtrim.json + report.json.

Dung:
  python scripts/trim_chunks.py measure --ref diag17\\submission_cx130cov.zip
      --docwin diag10\\docwin_top100.json.zst --docwin docwin_ext\\docwin_ext.json.zst
      --docs ...\\chunkdocs100.part000.jsonl.zst --docs ... --outdir trim
  python scripts/trim_chunks.py build --cands chunk_dw_u900_ext.json.zst
      --split diag12\\submission_split.json --fused diag10\\cache_rrf3d2.json
      --ref-c120in diag14\\submission_c120in.json --ref-cx130cov diag17\\submission_cx130cov.zip
      --docwin ... --docs ... --outdir trim --w 450
"""
import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from cx110 import hist_from_pairs, kprime, norm_pairs, total_chunks
from cx93 import B_COV, WD, apply_bonus, filter_base
from diag14 import order_q

NAME = "cxtrim"
DELTA_REF = 57
CX130_JSON_BYTES = 592_098_628
CX130_ZIP_BYTES = 92_767_156
MAX_ZIP = 93_000_000
W_LIST = (350, 450, 550)
WS_RE = re.compile(r"\S+")


def word_spans(s: str) -> list:
    """[(start, end)] cua tung tu (chay khong khoang trang) trong s."""
    return [(m.start(), m.end()) for m in WS_RE.finditer(s)]


def trim_around(chunk: str, win_s: int, win_e: int, w_words: int) -> str:
    """Cat chunk thanh doan w_words tu can giua cua so [win_s, win_e) (toa do trong chunk).

    Mo rong deu hai ben; chan bien chunk thi don sang ben kia; cat o ranh gioi khoang trang.
    Tra chuoi con NGUYEN VAN cua chunk. Chunk <= w_words (hoac w_words <= 0) giu nguyen.
    """
    if w_words <= 0:
        return chunk
    words = word_spans(chunk)
    n = len(words)
    if n <= w_words:
        return chunk
    hit = [i for i, (s, e) in enumerate(words) if s < win_e and e > win_s]
    if not hit:
        return chunk
    mid = (hit[0] + hit[-1] + 1) / 2.0
    a = int(mid - w_words / 2.0 + 0.5)
    a = max(0, min(a, n - w_words))
    b = a + w_words
    return chunk[words[a][0]:words[b - 1][1]]


def overlap_chars(ws: int, we: int, cs: int, ce: int) -> int:
    """So ky tu giao nhau cua [ws, we) va [cs, ce)."""
    return max(0, min(we, ce) - max(ws, cs))


def best_window(wins: list, cs: int, ce: int):
    """Cua so logit cao nhat co >= 50% ky tu nam trong chunk [cs, ce). None neu khong co."""
    best = None
    for w in wins:
        ws, we, lg = w[0], w[1], w[2]
        if we <= ws:
            continue
        ov = overlap_chars(ws, we, cs, ce)
        if ov * 2 >= we - ws and (best is None or lg > best[0]):
            best = (lg, ws, we)
    return None if best is None else (best[1], best[2], best[0])


def locate_chunk(text: str, chunk: str) -> list:
    """Tat ca vi tri text.find cua chunk trong text."""
    out, i = [], text.find(chunk)
    while i >= 0:
        out.append(i)
        i = text.find(chunk, i + 1)
    return out


def cut_chunk(text: str, chunk: str, wins: list, w_words: int):
    """Cat chunk (tim vi tri trong text) quanh cua so tot nhat.

    Tra (text_cat, meta): meta = {tim_duoc, co_cua_so, cat, vi_tri, logit}.
    Khong tim duoc text / khong co cua so thoa -> giu nguyen chunk.
    """
    locs = locate_chunk(text, chunk)
    if not locs:
        return chunk, {"tim_duoc": False, "co_cua_so": False, "cat": False, "vi_tri": 0,
                       "logit": None, "win_s": 0, "win_e": 0}
    best = None
    for cs in locs:
        bw = best_window(wins, cs, cs + len(chunk))
        if bw is not None and (best is None or bw[2] > best[1][2]):
            best = (cs, bw)
    if best is None:
        return chunk, {"tim_duoc": True, "co_cua_so": False, "cat": False, "vi_tri": locs[0],
                       "logit": None, "win_s": 0, "win_e": 0}
    cs, (ws, we, lg) = best
    out = trim_around(chunk, max(ws, cs) - cs, min(we, cs + len(chunk)) - cs, w_words)
    return out, {"tim_duoc": True, "co_cua_so": True, "cat": out != chunk, "vi_tri": cs,
                 "logit": lg, "win_s": ws, "win_e": we}


# ---------- nap du lieu ----------

def read_json_text(path) -> str:
    """Doc file JSON (.json/.json.zst/zip) thanh str (zip: entry submission.json)."""
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            return z.read("submission.json").decode("utf-8")
    if p.name.endswith(".zst"):
        import zstandard
        with p.open("rb") as f:
            return zstandard.ZstdDecompressor().stream_reader(f).read().decode("utf-8")
    return p.read_text(encoding="utf-8")


def iter_top_dict(text: str):
    """Yield (key, value) cua dict JSON tren cap do lon, khong nap toan bo value cung luc."""
    dec = json.JSONDecoder()
    n, i = len(text), text.find("{") + 1
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "}":
            break
        key, i = dec.raw_decode(text, i)
        while i < n and text[i] in " \t\r\n:":
            i += 1
        val, i = dec.raw_decode(text, i)
        yield key, val


def iter_top_list(text: str):
    """Yield tung phan tu cua list JSON tren cap do lon, khong nap toan bo."""
    dec = json.JSONDecoder()
    n, i = len(text), text.find("[") + 1
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        val, i = dec.raw_decode(text, i)
        yield val


def load_docwin(paths: list) -> dict:
    """Gop docwin: {qid: {doc_id: [[s, e, logit], ...]}} (nhieu file -> noi danh sach cua so)."""
    out: dict = {}
    for p in paths:
        d = json.loads(read_json_text(p))
        for q, docs in d.items():
            oq = out.setdefault(str(q), {})
            for did, wins in docs.items():
                oq.setdefault(str(did), []).extend(wins)
    return out


def load_texts(paths: list) -> dict:
    """{str(doc_id): text} tu cac file jsonl(.zst) {id, idx, text}; nguon truoc ghi truoc.

    Text chung giua cac nguon da kiem giong het -> giu ban dau tien la du.
    """
    import io
    import zstandard
    docs: dict = {}
    for p in paths:
        with open(p, "rb") as f:
            r = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
            for line in r:
                o = json.loads(line)
                docs.setdefault(str(o["id"]), o["text"])
    return docs


def load_ref(path) -> dict:
    """{str(id): relevant_chunks} tu submission (.json/.json.zst/zip)."""
    return {str(it["id"]): it["relevant_chunks"] for it in iter_top_list(read_json_text(path))}


def load_split_rank(split_path: str):
    """(split, base, rank) giong cx93.load_inputs; doc theo stream, chi giu id + relevant_docs."""
    split = [{"id": it["id"], "relevant_docs": it["relevant_docs"]}
             for it in iter_top_list(read_json_text(split_path))]
    base = {str(it["id"]): {str(d) for d in it["relevant_docs"]} for it in split}
    rank = {str(it["id"]): {str(d): i + 1 for i, d in enumerate(it["relevant_docs"])}
            for it in split}
    return split, base, rank


def load_kq(ref_path: str) -> dict:
    """{str(id): so relevant_chunks} tu submission tham chieu (doc stream, khong giu text)."""
    out = {}
    for it in iter_top_list(read_json_text(ref_path)):
        out[str(it["id"])] = len(it["relevant_chunks"])
    return out


def ob_of_q(rows: list, fused_hits: list, base_docs: set) -> list:
    """order_q + filter_base + apply_bonus (wd, b_cov mac dinh giong cx130)."""
    return apply_bonus(filter_base(order_q(rows, fused_hits, WD), base_docs), B_COV)


# ---------- do dai JSON ----------

def entry_len(doc_id, text: str) -> int:
    return len(json.dumps({"doc_id": doc_id, "chunk_text": text}, ensure_ascii=False))


def item_len(qid, docs_list: list, sum_entry: int, n_entry: int) -> int:
    """Do dai json.dumps(item, ensure_ascii=False) dung het (item = {id, relevant_docs, chunks})."""
    pre = json.dumps({"id": qid, "relevant_docs": docs_list, "relevant_chunks": []},
                     ensure_ascii=False)
    if n_entry == 0:
        return len(pre)
    return len(pre) + sum_entry + 2 * (n_entry - 1)


def list_json_len(item_lens: list) -> int:
    """Do dai json.dumps(list_item, ensure_ascii=False) = '[' + item + ', ' + ... + ']'."""
    return 2 + sum(item_lens) + 2 * max(len(item_lens) - 1, 0)


def uoc_zip(json_bytes: float) -> float:
    """Uoc zip LZMA theo ti le JSON/zip cua cx130cov (592.098.628 / 92.767.156)."""
    return json_bytes * CX130_ZIP_BYTES / CX130_JSON_BYTES


# ---------- xu ly theo cau ----------

def prep_stream(cands_text: str, fused: dict, base: dict, docs: dict, dw: dict,
                w_use: int, kq_of: dict, w_list: tuple = W_LIST):
    """Duyet ung vien theo cau: yield (qid, ob, lens, k57).

    ob: [{doc_id, chunk_text, t (cat theo w_use), logit, s, tim_duoc, co_cua_so, cat, win_s/e}]
    lens: {W: [tong entry_len cua prefix k57, so chunk, so chunk cat]} cho bang W.
    k57 = min(k_q + 57, |C_q|).
    """
    for q, rows in iter_top_dict(cands_text):
        q = str(q)
        ob = ob_of_q(rows, fused.get(q, []), base[q])
        wins_q = dw.get(q, {})
        k57 = kprime(kq_of.get(q, 0), len(ob), DELTA_REF)
        lens = {w: [0, 0, 0] for w in w_list}
        for i, c in enumerate(ob):
            txt = c["chunk_text"]
            t = docs.get(str(c["doc_id"]), "")
            cut, meta = cut_chunk(t, txt, wins_q.get(str(c["doc_id"]), []), w_use)
            c["t"] = cut
            c.update(meta)
            if i < k57:
                for w in w_list:
                    tw = cut if w == w_use else (
                        trim_around(txt, c["win_s"] - c["vi_tri"], c["win_e"] - c["vi_tri"], w)
                        if c["co_cua_so"] else txt)
                    lens[w][0] += entry_len(c["doc_id"], tw)
                    lens[w][1] += 1
                    lens[w][2] += (tw != txt)
        yield q, ob, lens, k57


# ---------- lenh measure ----------

def measure(ref_path: str, docwin_paths: list, docs_paths: list, outdir: str,
            w_list: tuple = W_LIST) -> dict:
    """Do nhanh tren chunk cua cx130cov: moi W -> ti le bi cat, byte JSON that (ghi file tam)."""
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    dw = load_docwin(docwin_paths)
    docs = load_texts(docs_paths)
    ref_text = read_json_text(ref_path)
    st = {w: {"byte_json": 0, "chunk": 0, "cat": 0, "co_cua_so": 0, "tim_duoc": 0}
          for w in w_list}
    n_q = 0
    tmp = {}
    for w in w_list:
        tp = out / f"_measure_w{w}.json"
        tmp[w] = tp.open("w", encoding="utf-8")
        tmp[w].write("[")
    first = True
    for it in iter_top_list(ref_text):
        n_q += 1
        q = str(it["id"])
        wins_q = dw.get(q, {})
        outs = {}
        for w in w_list:
            sel = []
            for c in it["relevant_chunks"]:
                d, txt = str(c["doc_id"]), c["chunk_text"]
                t = docs.get(d, "")
                cut, meta = (txt, {"tim_duoc": False, "co_cua_so": False, "cat": False}) if not t \
                    else cut_chunk(t, txt, wins_q.get(d, []), w)
                st[w]["chunk"] += 1
                st[w]["tim_duoc"] += meta["tim_duoc"]
                st[w]["co_cua_so"] += meta["co_cua_so"]
                st[w]["cat"] += meta["cat"]
                sel.append({"doc_id": c["doc_id"], "chunk_text": cut})
            outs[w] = {"id": it["id"], "relevant_docs": it["relevant_docs"],
                       "relevant_chunks": sel}
        for w in w_list:
            s = json.dumps(outs[w], ensure_ascii=False)
            tmp[w].write(("" if first else ",") + s)
        first = False
        if n_q % 200 == 0:
            print(f"measure {n_q} cau {time.time() - t0:.0f}s", flush=True)
    bang = {}
    for w in w_list:
        tmp[w].write("]")
        tmp[w].close()
        jb = tmp[w].name
        jb = Path(jb).stat().st_size
        s = st[w]
        bang[str(w)] = {
            "ty_le_chunk_bi_cat": s["cat"] / max(s["chunk"], 1),
            "byte_json": jb,
            "uoc_zip": uoc_zip(jb),
            "chunk": s["chunk"], "cat": s["cat"],
            "ty_le_tim_duoc_text": s["tim_duoc"] / max(s["chunk"], 1),
            "ty_le_co_cua_so": s["co_cua_so"] / max(s["chunk"], 1),
        }
    rep = {"lenh": "measure", "ref": str(ref_path), "cau": n_q, "bang_W": bang,
           "ty_le_json_zip_cx130": CX130_JSON_BYTES / CX130_ZIP_BYTES,
           "thoi_gian_giay": round(time.time() - t0, 1)}
    (out / "report_measure.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return rep


# ---------- lenh repro / build ----------

def run(cands: str, split_path: str, fused_path: str, ref_c120in_path: str,
        ref_cx130cov_path: str, docwin_paths: list, docs_paths: list, outdir: str,
        w_use: int = 450, delta: int | None = None, max_zip: int = MAX_ZIP,
        repro_only: bool = False) -> dict:
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    split, base, _rank = load_split_rank(split_path)
    fused = {str(q): v for q, v in
             json.loads(Path(fused_path).read_text(encoding="utf-8")).items()}
    kq_of = load_kq(ref_c120in_path)
    ref130 = load_ref(ref_cx130cov_path)
    dw = load_docwin(docwin_paths)
    docs = {} if repro_only else load_texts(docs_paths)
    cands_text = read_json_text(cands)
    print(f"nap xong {time.time() - t0:.0f}s", flush=True)

    prep, pairs = {}, []
    n_lech = 0
    vi_du = []
    n_q = 0
    tot_lens = {w: [0, 0, 0] for w in W_LIST}
    item_lens57 = {w: [] for w in W_LIST}
    for q, ob, lens, k57 in prep_stream(cands_text, fused, base, docs, dw, w_use, kq_of):
        n_q += 1
        kq, cap = kq_of.get(q, 0), len(ob)
        pairs.append((kq, cap))
        want = ref130.get(q, [])
        got = [{"doc_id": c["doc_id"], "chunk_text": c["chunk_text"]} for c in ob[:k57]]
        w, g = norm_pairs(want), norm_pairs(got)
        if g != w:
            n_lech += 1
            if len(vi_du) < 5:
                i = next((j for j in range(max(len(g), len(w))) if g[j:j + 1] != w[j:j + 1]), None)
                vi_du.append({"id": q, "k130": len(w), "got_n": len(g), "vi_tri_lech": i,
                              "got": (g[i][0], g[i][1][:60]) if i is not None and i < len(g) else None,
                              "want": (w[i][0], w[i][1][:60]) if i is not None and i < len(w) else None})
        if not repro_only:
            s_item = split[n_q - 1]
            for w_ in W_LIST:
                tot_lens[w_][0] += lens[w_][0]
                tot_lens[w_][1] += lens[w_][1]
                tot_lens[w_][2] += lens[w_][2]
                item_lens57[w_].append(item_len(s_item["id"], s_item["relevant_docs"],
                                                lens[w_][0], lens[w_][1]))
        for c in ob:
            c.pop("chunk_text", None)
        prep[q] = {"kq": kq, "cap": cap, "ob": ob}
        if n_q % 200 == 0:
            print(f"prep {n_q} cau {time.time() - t0:.0f}s lech {n_lech}", flush=True)
    del cands_text
    repro = {"cau": n_q, "cau_khop": n_q - n_lech, "ty_le_cau_khop": (n_q - n_lech) / max(n_q, 1),
             "ok": n_lech == 0 and n_q == len(ref130) == 1200, "vi_du_lech": vi_du}
    print(f"tai lap delta=57 khong cat: {json.dumps(repro, ensure_ascii=False)}", flush=True)

    bang = {}
    if not repro_only:
        for w_ in W_LIST:
            jb = list_json_len(item_lens57[w_])
            s0, n0, c0 = tot_lens[w_]
            bang[str(w_)] = {"byte_json": jb, "uoc_zip": uoc_zip(jb),
                             "ty_le_chunk_bi_cat": c0 / max(n0, 1), "chunk": n0, "cat": c0}
    report = {"lenh": "repro" if repro_only else "build",
              "quy_uoc": {
                  "giong": "cx130cov: doc nen = relevant_docs split (0 doc them), ung vien "
                           "chunk_dw_u900_ext.json.zst, s = zc + 4*zd (diag14.order_q), b = 1,0",
                  "khac": f"cat chunk quanh cua so docwin ({w_use} tu); k' = min(k_q + delta, |C_q|)",
                  "cat": "cua so logit cao nhat >= 50% ky tu trong chunk; doan W tu can giua "
                         "(chan bien chunk thi don sang ben kia); chuoi con nguyen van text doc",
                  "uoc_zip": "ti le JSON/zip cua cx130cov (592.098.628 / 92.767.156)",
                  "zip": "zip_submission.py --method lzma, entry submission.json"},
              "w": w_use, "cau": n_q,
              "tai_lap_delta57_khong_cat": repro, "bang_W": bang,
              "thoi_gian_giay": round(time.time() - t0, 1)}
    if repro_only:
        (out / "report_repro.json").write_text(json.dumps(report, ensure_ascii=False),
                                               encoding="utf-8")
        return report

    hist = hist_from_pairs(pairs)
    n57 = total_chunks(hist, DELTA_REF)
    d_max = max((cap - k for k, cap in pairs), default=0)
    print(f"tong chunk delta=57: {n57}, d_max {d_max}", flush=True)

    def build_at(d: int):
        sub = []
        for it in split:
            q = str(it["id"])
            p = prep[q]
            kp = kprime(p["kq"], p["cap"], d)
            seen, sel = set(), []
            for c in p["ob"][:kp]:
                t = c["t"]
                if t in seen:
                    continue
                seen.add(t)
                sel.append({"doc_id": c["doc_id"], "chunk_text": t})
            sub.append({"id": it["id"], "relevant_docs": list(it["relevant_docs"]),
                        "relevant_chunks": sel})
        return sub

    lo, hi = DELTA_REF, max(d_max, DELTA_REF)
    d_best, json_best = DELTA_REF, None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        sub = build_at(mid)
        jb = len(json.dumps(sub, ensure_ascii=False))
        del sub
        est = uoc_zip(jb)
        print(f"thu delta {mid}: json {jb}, uoc zip {est:.0f} <= {max_zip}: {est <= max_zip}",
              flush=True)
        if est <= max_zip:
            lo, d_best, json_best = mid, mid, jb
        else:
            hi = mid - 1
    if delta is not None:
        d_best = int(delta)
    sub = build_at(d_best)
    fp = out / f"submission_{NAME}.json"
    with fp.open("w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    jb = fp.stat().st_size
    del sub

    n = max(len(split), 1)
    kps = [min(p["kq"] + d_best, p["cap"]) for p in prep.values()]
    blocked = sum(1 for p in prep.values() if p["cap"] < p["kq"] + d_best)
    n_doc_q, chars, tot = [], 0, 0
    win_found = cut_n = found_n = 0
    seen_all = 0
    for it in split:
        q = str(it["id"])
        p = prep[q]
        kp = kprime(p["kq"], p["cap"], d_best)
        seen, nd = set(), set()
        for c in p["ob"][:kp]:
            t = c["t"]
            if t in seen:
                seen_all += 1
                continue
            seen.add(t)
            nd.add(str(c["doc_id"]))
            chars += len(t)
            tot += 1
            win_found += bool(c.get("co_cua_so"))
            found_n += bool(c.get("tim_duoc"))
            cut_n += bool(c.get("cat"))
        n_doc_q.append(len({str(c["doc_id"]) for c in p["ob"][:kp]}))
    report.update({
        "delta": {"ref_57": DELTA_REF, "dung": d_best, "d_max": d_max,
                  "uoc_zip_json_best": uoc_zip(json_best) if json_best else None},
        "kprime": {"TB": sum(kps) / n, "min": min(kps, default=0), "max": max(kps, default=0),
                   "tong": sum(kps)},
        "cau_chan_boi_ung_vien": blocked,
        "TB_chunk_them_vs_cx130cov_cau": (sum(kps) - n57) / n,
        "TB_doc_co_chunk_cau": sum(n_doc_q) / n,
        "TB_doc_cau": sum(len(it["relevant_docs"]) for it in split) / n,
        "TB_ky_tu_chunk": chars / max(tot, 1),
        "tong_chunk": tot, "chunk_trung_sau_cat": seen_all,
        "ty_le_chunk_tim_duoc_text": found_n / max(tot, 1),
        "ty_le_chunk_co_cua_so": win_found / max(tot, 1),
        "ty_le_chunk_bi_cat": cut_n / max(tot, 1),
        "json_bytes": jb, "uoc_zip": uoc_zip(jb), "hist_kcap": hist,
        "thoi_gian_giay": round(time.time() - t0, 1),
    })
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"delta {d_best}: json {jb}, uoc zip {uoc_zip(jb):.0f}; ghi {fp}", flush=True)
    if not repro["ok"]:
        print("KHONG tai lap duoc cx130cov voi delta=57, khong dung bai", flush=True)
        raise SystemExit(2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    ms = sp.add_parser("measure")
    ms.add_argument("--ref", required=True)
    ms.add_argument("--docwin", action="append", required=True)
    ms.add_argument("--docs", action="append", required=True)
    ms.add_argument("--outdir", required=True)
    rp = sp.add_parser("repro")
    bd = sp.add_parser("build")
    for p in (rp, bd):
        p.add_argument("--cands", required=True)
        p.add_argument("--split", required=True)
        p.add_argument("--fused", required=True)
        p.add_argument("--ref-c120in", required=True)
        p.add_argument("--ref-cx130cov", required=True)
        p.add_argument("--docwin", action="append", required=True)
        p.add_argument("--docs", action="append", required=True)
        p.add_argument("--outdir", required=True)
        p.add_argument("--w", type=int, default=450)
        p.add_argument("--max-zip", type=int, default=MAX_ZIP)
    bd.add_argument("--delta", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "measure":
        measure(a.ref, a.docwin, a.docs, a.outdir)
        return
    rep = run(a.cands, a.split, a.fused, a.ref_c120in, a.ref_cx130cov, a.docwin, a.docs,
              a.outdir, a.w, getattr(a, "delta", None), a.max_zip, repro_only=(a.cmd == "repro"))
    if a.cmd == "repro":
        raise SystemExit(0 if rep["tai_lap_delta57_khong_cat"]["ok"] else 2)
    raise SystemExit(0 if rep["uoc_zip"] <= a.max_zip else 3)


if __name__ == "__main__":
    main()
