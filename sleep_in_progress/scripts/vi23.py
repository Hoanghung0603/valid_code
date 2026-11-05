"""vi23: dua 54.164 doc VI moi (handoff_vi_20261017) vao he truy hoi, dung bai theo cong thuc #5532.

Boi canh: goi VI b660 D:\\r2ai\\data\\crawl\\handoff_vi_20261017\\vi-recov-0000{0,1,2}.jsonl.zst
(54.164 ban ghi, CHI DOC) - tat ca deu thuoc 136.809 id VI thieu text (diag21). Muc tieu: dua vao
BM25 + dense + pool + docwin, dung lai bai theo cong thuc #5532 (cx130cov) sao cho doc moi canh
tranh cong bang voi doc cu, KHONG nop bai.

QUY UOC "CHI DOI MOT BIEN" (pool_v23) - khac ke hoach ban dau theo chi dan cua user 17/10:
  - pool_v23 = tap doc cua #5532 (cache_pool_rr.json, GIU NGUYEN doc cu + logit cu + thoi tu cu)
    UNION {doc MOI nam trong BM25 top-200 cua chi muc gop vi23 hoac dense top-200 da gop}.
    Doc cu KHONG vao/ra khoi pool vi IDF thay doi. Hang BM25 cua doc moi chi dung de chon ung vien.
  - Nguon dense/HyDE trong RRF dung top-200 DA GOP theo cosine (doc cu giu nguyen diem, co the bi
    doc moi day xuong hang - he qua dung, chap nhan).
  - Rerank bge CHI cho cap (q, doc MOI) trong pool_v23.
  - Doc khong co diem Qwen3-8B (toan bo doc moi; 8B khong chay duoc Beam) -> nguon 8B khong dong
    gop RRF cho doc do (giong diag18/pool3).

Cong thuc phat sinh bai (trung #5532):
  rrf3d2 = RRF k60 (bge 1, 8B 1, dense 2) tren pool_v23; fhyde = rrf3d2 + HyDE x1 (khe thu 4);
  n_q = so doc logit bge > -2,0 kẹp [40,400] (union_sub.adaptive_n tren pool_v23);
  doc nen = n_q doc dau theo fhyde, + doc nguon cua chunk trong ref-chunks (full8b_sub --top-ens
    1e6, cung luat voi submission_split; ref-chunks = diag11/chunks_rrf3d2_cf4.json);
  chunk = cx130cov: s = zc + 4*zd (zd rrf3d2), thuong b = 1 cho chunk dau moi doc,
    k'_q = min(k_q + delta, |C_q|) voi k_q = so relevant_chunks cua submission_c120in.json,
    delta = 57 (giam neu zip LZMA vuot ngan sach); CHI chunk tu doc da nop.

Cac lenh (xem tests/test_vi23.py):
  manifest     kiem sha256 cua goi handoff voi manifest.sha256
  prep         loc doc moi (status/text/ngon ngu/trung) + ghep boilerplate + kiem id -> new_ids.json
  verify-idx   kiem idx vi23 giu nguyen idx cu, ghi new_idx.json {doc_id: idx}
  bm25-diff    do BM25 vi23 top-200 (doc cu) vs cache_vi_k300 top-200 (chi tham khao)
  dense-merge  gop top-k doc moi vao top-200 cu theo cosine (giu diem cu)
  pool-new     doc MOI vao pool_v23 (BM25_new-200 | dense_gop-200) - pool cu -> cap can rerank
  pool-merge   gop logit rerank moi vao pool_v23 -> cache_pool_v23.json (logit giam dan)
  chain        dung bai (che do khong-moi de sanity tai lap #5532, hoac that)
  newonly30    bai phu 30 doc moi/cau (khong chunk) de do precision doc moi
  stats        thong ke buoc 8

Dung:
  python scripts/vi23.py prep --shards D:\\r2ai\\data\\sub\\diag23\\shards ^
      --old-bp D:\\r2ai\\data\\index\\vi\\boilerplate.json --missing D:\\r2ai\\data\\sub\\diag21\\missing_vi_by_host.json ^
      --old-ids D:\\r2ai\\data\\index\\vi\\ids.npy --out-dir D:\\r2ai\\data\\sub\\diag23 --bp-out D:\\r2ai\\data\\index\\vi23
  python scripts/vi23.py chain --mode khong-moi --pool D:\\r2ai\\data\\sub\\cache_pool_rr.json ...
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NAME = "vi23"
MIN_CHARS = 200
THR_N = -2.0
LO_N, HI_N = 40, 400
WD = 4.0
B_COV = 1.0
DELTA = 57
K_RRF = 60
W_BGE, W_Q8B, W_DENSE, W_HYDE = 1.0, 1.0, 2.0, 1.0
W_Q8B_FHYDE = None   # vi33: trong so 8B rieng cho fhyde (None = W_Q8B)
NQ_FUSED = None      # vi35: (method, thr) chon n_q theo diem fused; None = adaptive_n bge
MAX_ZIP = 93_000_000
TOP_POOL = 200
N_NEWONLY = 30
BP_MIN_DOCS, BP_MAX_SHARE, BP_MAX_FIT = 50, 0.2, 3000


# ---------------------------------------------------------------- tien ich cache

def load_cache(path) -> dict:
    """Doc cache {qid: [[doc_id, idx, score], ...]} -> {str(qid): hits}."""
    return {str(q): v for q, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def load_json(path):
    p = Path(path)
    if p.name.endswith(".zst"):
        import zstandard
        with p.open("rb") as f:
            return json.loads(zstandard.ZstdDecompressor().stream_reader(f).read())
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path, obj) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, ensure_ascii=False)
    p.write_text(raw, encoding="utf-8")
    return len(raw.encode("utf-8"))


def hit_ids(hits: list, limit: int | None = None) -> list:
    """doc_id theo th? t? hit, b? tr?ng l?n xu?t hi?n d?u."""
    out, seen = [], set()
    for h in (hits[:limit] if limit else hits):
        d = h[0]
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


# ---------------------------------------------------------------- manifest + prep

def check_manifest(shard_dir, manifest_name: str = "manifest.sha256") -> dict:
    """Kiem sha256 cac file trong shard_dir voi manifest (dang 'hash  ten')."""
    man = {}
    for line in (Path(shard_dir) / manifest_name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        h, _, name = line.partition("  ")
        man[name.strip()] = h.strip().lower()
    rows, ok = [], 0
    for name, want in sorted(man.items()):
        p = Path(shard_dir) / name
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        good = got == want
        ok += good
        rows.append({"file": name, "sha256": got, "want": want, "ok": good,
                     "bytes": p.stat().st_size if p.exists() else None})
    return {"file": len(man), "ok": ok, "tat_ca_ok": ok == len(man) and bool(man), "cac_file": rows}


def filter_records(records: list, seen_ids: set | None = None) -> tuple:
    """Loc ban ghi nhu shard_docs cua bigindex: status 200, text >= 200 ky tu, detect_lang vi,
    loai trung id (trong records va voi seen_ids). Tra (kept, stats, ids_kept).

    kept giu nguyen ban ghi goc (id/url/final_url/host/status/title/text/date/ts)."""
    from chunking import detect_lang

    seen = set(seen_ids or ())
    kept, ids = [], []
    st = Counter()
    by_host = Counter()
    for r in records:
        st["tong"] += 1
        did = r.get("id")
        if did is None:
            st["thieu_id"] += 1
            continue
        if did in seen:
            st["trung_id"] += 1
            continue
        text = r.get("text") or ""
        if r.get("status") != 200:
            st["khong_200"] += 1
            continue
        if len(text) < MIN_CHARS:
            st["text_ngan"] += 1
            continue
        if detect_lang(text) != "vi":
            st["khong_phai_vi"] += 1
            continue
        seen.add(did)
        kept.append(r)
        ids.append(int(did))
        by_host[r.get("host", "")] += 1
        st["giu"] += 1
    stats = {"tong": st["tong"], "giu": st["giu"], "loai": {
        "khong_200": st["khong_200"], "text_ngan": st["text_ngan"],
        "khong_phai_vi": st["khong_phai_vi"], "trung_id": st["trung_id"], "thieu_id": st["thieu_id"]},
        "theo_host": dict(by_host.most_common())}
    return kept, stats, ids


def fit_new_hosts(old_bp: dict, records: list) -> tuple:
    """Ghep boilerplate: GIU NGUYEN moi muc cua old_bp (khong refit host cu), fit rieng cho host
    CHUA co trong old_bp bang BoilerplateFilter tren doc moi. Tra (merged_dict, report).

    Host moi qua it mau (< BP_MIN_DOCS) khong co muc nao (ghi nhan)."""
    from boilerplate import BoilerplateFilter

    new_hosts = sorted({r.get("host", "") for r in records} - set(old_bp))
    docs = ({"host": r.get("host", ""), "text": r.get("text") or ""}
            for r in records if r.get("host", "") in set(new_hosts))
    cnt = Counter(r.get("host", "") for r in records)
    filt = BoilerplateFilter(min_docs=BP_MIN_DOCS, max_share=BP_MAX_SHARE, max_fit_docs=BP_MAX_FIT)
    fitted = filt.fit(docs)._lines
    merged = {h: list(v) for h, v in old_bp.items()}
    for h, s in fitted.items():
        merged[h] = sorted(s)
    it_mau = [h for h in new_hosts if cnt[h] < BP_MIN_DOCS]
    khong_lap = [h for h in new_hosts if cnt[h] >= BP_MIN_DOCS and h not in fitted]
    rep = {"host_cu": len(old_bp), "host_moi_trong_goi": len(new_hosts),
           "host_moi_duoc_fit": sorted(fitted),
           "host_moi_it_mau_duoi_50": it_mau,
           "host_moi_khong_co_dong_lap": khong_lap,
           "host_moi_so_muc": {h: len(fitted[h]) for h in sorted(fitted)},
           "host_moi_so_doc": dict(sorted(((h, cnt[h]) for h in new_hosts), key=lambda x: -x[1]))}
    return merged, rep


def cmd_manifest(a) -> int:
    rep = check_manifest(a.shards)
    write_json(a.report, rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "cac_file"}, ensure_ascii=False), flush=True)
    return 0 if rep["tat_ca_ok"] else 2


def read_shard_records(path) -> list:
    """Doc JSONL.zst -> list ban ghi."""
    import zstandard
    with Path(path).open("rb") as f:
        raw = zstandard.ZstdDecompressor().stream_reader(f).read()
    return [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]


def cmd_prep(a) -> int:
    t0 = time.time()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    old_bp = json.loads(Path(a.old_bp).read_text(encoding="utf-8"))
    missing = json.loads(Path(a.missing).read_text(encoding="utf-8"))
    missing_ids = {int(i) for ids in missing.values() for i in ids}
    old_ids = {int(x) for x in np.load(a.old_ids)} if a.old_ids else set()
    all_rows, per_file = [], []
    for p in sorted(Path(a.shards).glob(a.pattern)):
        rows = read_shard_records(p)
        _kept, st, _ids = filter_records(rows)
        per_file.append({"file": p.name, "tong": st["tong"], "giu": st["giu"], "loai": st["loai"]})
        all_rows.extend(rows)
    kept, st_all, ids = filter_records(all_rows, set())
    ids = [int(x) for x in ids]
    id_set = set(ids)
    trung_cu = sorted(id_set & old_ids)
    ngoai_missing = sorted(id_set - missing_ids)
    if trung_cu or ngoai_missing:
        drop = set(trung_cu) | set(ngoai_missing)
        kept = [r for r in kept if int(r["id"]) not in drop]
        ids = [int(x) for x in ids if x not in drop]
        st_all["loai"]["trung_voi_chi_muc_cu"] = len(trung_cu)
        st_all["loai"]["ngoai_danh_sach_thieu"] = len(ngoai_missing)
        st_all["giu"] = len(kept)
        st_all["theo_host"] = dict(Counter(r.get("host", "") for r in kept).most_common())
    bp, bp_rep = fit_new_hosts(old_bp, kept)
    if a.bp_out:
        Path(a.bp_out).mkdir(parents=True, exist_ok=True)
        write_json(Path(a.bp_out) / "boilerplate.json", bp)
    write_json(out / "new_ids.json", sorted(ids))
    write_json(out / "new_host.json", {str(int(r["id"])): r.get("host", "") for r in kept})
    rep = {"quy_uoc": "prep: status 200, text >= 200 ky tu, detect_lang vi, loai trung id; "
                      "id phai nam trong missing_vi_by_host va khong trung ids.npy chi muc cu; "
                      "boilerplate: giu nguyen host cu, fit rieng host moi (min_docs 50, max_share 0.2, max_fit 3000)",
           "shards": a.shards, "file": per_file,
           "tong_sau_loc": st_all["tong"], "giu": st_all["giu"], "loai": st_all["loai"],
           "theo_host": st_all["theo_host"],
           "id_thieu_trong_missing": len(id_set - missing_ids),
           "id_trung_chi_muc_cu": len(trung_cu),
           "missing_tong": len(missing_ids),
           "boilerplate": bp_rep,
           "ghi": [str(out / "new_ids.json"), str(out / "new_host.json"),
                   str(Path(a.bp_out) / "boilerplate.json") if a.bp_out else None],
           "thoi_gian_giay": round(time.time() - t0, 1)}
    write_json(out / "prep_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("theo_host", "file")},
                     ensure_ascii=False), flush=True)
    return 0


def cmd_verify_idx(a) -> int:
    import numpy as np
    old = np.load(a.index_old_ids)
    new = np.load(a.index_new_ids)
    n_old = len(old)
    giong = bool(np.array_equal(old, new[:n_old]))
    new_ids = [int(x) for x in json.loads(Path(a.new_ids).read_text(encoding="utf-8"))]
    idx_of = {int(d): i for i, d in enumerate(new)}
    thieu = [d for d in new_ids if d not in idx_of]
    trung = len(new) - len({int(x) for x in new})
    write_json(a.out, {str(d): idx_of[d] for d in new_ids if d in idx_of})
    rep = {"so_doc_cu": n_old, "so_doc_moi_index": int(len(new)),
           "idx_cu_giu_nguyen": giong, "id_moi": len(new_ids), "id_moi_thieu_idx": thieu[:20],
           "id_trong_index": trung, "ok": bool(giong and not thieu and not trung)}
    write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0 if rep["ok"] else 2


# ---------------------------------------------------------------- bm25 diff

def bm25_diff(old_cache: dict, new_cache: dict, new_ids: set | None = None, top: int = 200) -> dict:
    """So sanh top-`top` BM25: doc cu khac giua hai chi muc + so doc moi xuat hien (tham khao)."""
    n = 0
    mat_tb = trung_tb = moi_tb = 0.0
    for q, hits in new_cache.items():
        old200 = [str(x) for x in hit_ids(old_cache.get(q, []), top)]
        new200 = [str(x) for x in hit_ids(hits, top)]
        a, b = set(old200), set(new200)
        n += 1
        mat_tb += len(a - b)
        trung_tb += len(a & b)
        if new_ids is not None:
            moi_tb += sum(1 for d in b if d in new_ids)
    n = max(n, 1)
    return {"so_cau": n, "top": top,
            "TB_doc_cu_mat_khoi_top200_vi23": mat_tb / n,
            "TB_doc_cu_trung_top200": trung_tb / n,
            "TB_doc_moi_trong_top200_vi23": moi_tb / n if new_ids is not None else None}


def cmd_bm25_diff(a) -> int:
    new_ids = ({str(i) for i in json.loads(Path(a.new_ids).read_text(encoding="utf-8"))}
               if a.new_ids else None)
    rep = bm25_diff(load_cache(a.cache_old), load_cache(a.cache_new), new_ids, a.top)
    rep["note"] = "chi tham khao: pool_v23 giu nguyen doc cu, BM25 vi23 chi dung chon doc moi"
    write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- dense merge

def merge_topk(old_hits: list, new_hits: list, top: int = 200) -> tuple:
    """Gop top-k theo cosine: hit cu GIU NGUYEN diem, trung diem thi hit cu len truoc.

    Tra (hits_moi, stats). Doc trung doc_id: lay hit cu."""
    seen, rows = set(), []
    for src, hits in (("cu", old_hits), ("moi", new_hits)):
        for h in hits:
            d = h[0]
            if d in seen:
                continue
            seen.add(d)
            rows.append((h, src))

    def key(i):
        sc = rows[i][0][2]
        return (sc is None, 0.0 if sc is None else -sc, i)

    order = sorted(range(len(rows)), key=key)
    sel = [rows[i] for i in order[:top]]
    out = [list(h) for h, _ in sel]
    st = {"top": top, "so_cu": sum(1 for _, s in sel if s == "cu"),
          "so_moi": sum(1 for _, s in sel if s == "moi"),
          "day_het": len(out) == top,
          "truoc_gop": {"cu": len(old_hits), "moi": len(new_hits)}}
    return out, st


def cmd_dense_merge(a) -> int:
    old, new = load_cache(a.cache_old), load_cache(a.cache_new)
    out, sts = {}, {}
    for q in sorted(set(old) | set(new), key=lambda x: int(x)):
        hits, st = merge_topk(old.get(q, []), new.get(q, []), a.top)
        out[q] = hits
        sts[q] = st
    n = max(len(out), 1)
    rep = {"top": a.top, "so_cau": len(out),
           "TB_doc_moi_lot": sum(s["so_moi"] for s in sts.values()) / n,
           "TB_doc_cu_giu_lai": sum(s["so_cu"] for s in sts.values()) / n,
           "cau_khong_day_top": sum(1 for s in sts.values() if not s["day_het"])}
    write_json(a.out, out)
    if a.report:
        rep["ghi"] = a.out
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- pool

def new_pool_docs(bm25_hits: list, dense_hits: list, old_pool_ids: set,
                  new_ids: set, idx_of: dict) -> tuple:
    """Doc MOI vao pool cua 1 cau: (BM25_new top-200 | dense_gop top-200) - pool cu.

    Tra (rows [[doc_id, idx, None], ...] theo thu tu: bm25 truoc, dense sau, bo trung, stats)."""
    dense200 = {str(h[0]) for h in dense_hits[:TOP_POOL]}
    rows, seen, nguon = [], set(), Counter()
    for name, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
        for h in hits[:TOP_POOL]:
            d = h[0]
            s = str(d)
            if s in old_pool_ids or s in seen:
                continue
            if s not in new_ids:
                continue
            i = idx_of.get(s, h[1])
            seen.add(s)
            rows.append([d, i, None])
            nguon[name] += 1
            if name == "bm25" and s in dense200:
                nguon["bm25+dense"] += 1
    return rows, {"bm25": nguon["bm25"], "dense": nguon["dense"], "bm25+dense": nguon["bm25+dense"]}


def cmd_pool_new(a) -> int:
    t0 = time.time()
    pool = load_cache(a.pool)
    bm25 = load_cache(a.bm25_new)
    dense = load_cache(a.dense_new)
    new_ids = {str(i) for i in json.loads(Path(a.new_ids).read_text(encoding="utf-8"))}
    idx_of = ({str(k): v for k, v in json.loads(Path(a.idx).read_text(encoding="utf-8")).items()}
              if a.idx else {})
    cands, nguon = {}, Counter()
    n_moi = []
    for q in sorted(pool, key=lambda x: int(x)):
        old_ids = {str(h[0]) for h in pool.get(q, [])}
        rows, st = new_pool_docs(bm25.get(q, []), dense.get(q, []), old_ids, new_ids, idx_of)
        for k, v in st.items():
            nguon[k] += v
        if rows:
            cands[q] = [[r[0], r[1]] for r in rows]
        n_moi.append(len(rows))
    n = max(len(cands), 1)
    tong = sum(n_moi)
    write_json(a.out_cands, cands)
    rep = {"so_cau": len(pool), "cap_can_rerank": tong, "TB_doc_moi_pool_cau": tong / max(len(pool), 1),
           "max_doc_moi_cau": max(n_moi, default=0), "cau_co_doc_moi": sum(1 for x in n_moi if x),
           "nguon_doc_moi": dict(nguon), "ghi": [a.out_cands],
           "thoi_gian_giay": round(time.time() - t0, 1)}
    if a.out_new_hits:
        write_json(a.out_new_hits, {q: [[r[0], r[1], None] for r in rows] for q, rows in cands.items()})
        rep["ghi"].append(a.out_new_hits)
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


def build_pool_v23(old_pool: dict, new_hits: dict, logits: dict) -> dict:
    """pool_v23 = hit cu (giu nguyen) + hit moi co logit, sort_by_score (logit giam dan, None cuoi).

    logits = {qid: {doc_id: logit}}; hit moi khong logit -> [d, idx, None] (xuong cuoi).
    Trung doc_id voi pool cu: bo (nguon cu thang)."""
    from fuse import sort_by_score
    out = {}
    for q in sorted(set(old_pool) | set(new_hits), key=lambda x: int(x)):
        old = [list(h) for h in old_pool.get(q, [])]
        seen = {str(h[0]) for h in old}
        lg = logits.get(q, {})
        for h in new_hits.get(q, []):
            s = str(h[0])
            if s in seen:
                continue
            seen.add(s)
            out_lg = lg.get(s, lg.get(h[0], h[2] if len(h) > 2 else None))
            old.append([h[0], h[1], out_lg])
        out[q] = sort_by_score(old)
    return out


def cmd_pool_merge(a) -> int:
    old = load_cache(a.pool_old)
    new = load_cache(a.new_hits) if a.new_hits else {}
    rr = load_cache(a.rerank) if a.rerank else {}
    logits = {q: {str(h[0]): h[2] for h in hits} for q, hits in rr.items()}
    out = build_pool_v23(old, new, logits)
    n = max(len(out), 1)
    n_moi = sum(len(out[q]) - len(old.get(q, [])) for q in out)
    n_none = sum(1 for q in out for h in out[q] if h[2] is None)
    loi_cu = 0
    for q, hits in old.items():
        gm = {str(h[0]): h for h in out.get(q, [])}
        for h in hits:
            g = gm.get(str(h[0]))
            if g is None or g[1] != h[1] or g[2] != h[2]:
                loi_cu += 1
    rep = {"so_cau": len(out), "TB_pool_cau": sum(len(v) for v in out.values()) / n,
           "TB_doc_moi_cau": n_moi / n, "hit_logit_None": n_none,
           "hit_cu_lech_logit_idx": loi_cu, "pool_cu_giu_nguyen": loi_cu == 0}
    write_json(a.out, out)
    rep["ghi"] = a.out
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- chain

def first_n_docs(hits: list, n: int) -> list:
    """n doc dau (id goc) theo th? t? hits, khong trung."""
    return hit_ids(hits, None)[:n] if n >= 0 else []


def full8b_docs(base: list, ref_chunks: list) -> list:
    """doc nen + doc nguon chunk chua co (full8b_sub --top-ens 1e6), giu th? t? xuat hien."""
    docs = list(base)
    seen = {str(d) for d in docs}
    for ch in ref_chunks:
        d = ch["doc_id"]
        if str(d) not in seen:
            seen.add(str(d))
            docs.append(d)
    return docs


def chain_build(pool: dict, q8b: dict, dense: dict, hyde: dict, cands: dict, kq: dict,
                ref_chunks: dict, thr: float = THR_N, lo: int = LO_N, hi: int = HI_N,
                wd: float = WD, b_cov: float = B_COV, delta: int = DELTA) -> tuple:
    """Toan bo chuoi (khong ghi file): fuse -> doclist -> chon chunk cx130cov.

    Tra (rrf3d2, fhyde, sub, report). sub = [{id, relevant_docs, relevant_chunks}].
    Luat doclist = full8b_sub: n_q = adaptive_n(pool[q]); nen = n_q doc dau fhyde;
    them doc nguon cua ref_chunks. Chunk: diag14.order_q (zd rrf3d2), b = b_cov, k' = min(k_q+delta, |C_q|)."""
    from cx130 import kprime
    from cx93 import apply_bonus, filter_base
    from diag14 import order_q, strip_meta
    from fuse3 import fuse3 as fuse3_fn
    from union_sub import adaptive_n

    rrf3d2 = fuse3_fn(pool, q8b, dense, (W_BGE, W_Q8B, W_DENSE, 0.0), K_RRF, None, None)
    fhyde = fuse3_fn(pool, q8b, dense, (W_BGE, W_Q8B, W_DENSE, W_HYDE), K_RRF, hyde, None)
    sub, ns, kps, caps, pairs_hist = [], [], [], [], []
    for q in sorted(pool, key=lambda x: int(x)):
        n = adaptive_n(pool[q], thr, lo, hi)
        base = first_n_docs(fhyde.get(q, []), n)
        docs = full8b_docs(base, ref_chunks.get(q, []))
        allowed = {str(d) for d in docs}
        cin = filter_base(order_q(cands.get(q, []), rrf3d2.get(q, []), wd), allowed)
        kp = kprime(kq.get(q, 0), len(cin), delta)
        sel = strip_meta(apply_bonus(cin, b_cov), kp)
        for c in sel:
            assert str(c["doc_id"]) in allowed
        sub.append({"id": int(q) if str(q).isdigit() else q,
                    "relevant_docs": docs, "relevant_chunks": sel})
        ns.append(n)
        kps.append(kp)
        caps.append(len(cin))
        pairs_hist.append((kq.get(q, 0), len(cin)))
    n_q = max(len(sub), 1)
    from cx110 import hist_from_pairs
    rep = {"n_q_TB": sum(ns) / n_q, "n_q_min": min(ns), "n_q_max": max(ns),
           "TB_doc_cau": sum(len(s["relevant_docs"]) for s in sub) / n_q,
           "TB_chunk_cau": sum(kps) / n_q, "tong_chunk": sum(kps),
           "min_chunk": min(kps), "max_chunk": max(kps),
           "TB_C_q": sum(caps) / n_q, "cau_chan_boi_C_q": sum(1 for k, c in zip(kps, caps) if c < k),
           "delta": delta, "thr": thr, "lo": lo, "hi": hi, "wd": wd, "b_cov": b_cov,
           "hist_kcap": hist_from_pairs(pairs_hist)}
    return rrf3d2, fhyde, sub, rep


def _ref_chunks_map(path: str) -> dict:
    return {str(it["id"]): it["relevant_chunks"] for it in load_json(path)}


def cmd_chain(a) -> int:
    t0 = time.time()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_cache(a.pool)
    q8b = load_cache(a.q8b)
    dense = load_cache(a.dense)
    hyde = load_cache(a.hyde) if a.hyde else {}
    cands = {str(q): v for q, v in load_json(a.cands).items()}
    ref = load_json(a.ref_chunks) if a.ref_chunks else []
    kq = {str(it["id"]): len(it["relevant_chunks"]) for it in load_json(a.ref_kq)}
    ref_map = {str(it["id"]): it["relevant_chunks"] for it in ref} if ref else {
        str(it["id"]): it["relevant_chunks"] for it in load_json(a.ref_kq)}
    rrf3d2, fhyde, sub, rep = chain_build(pool, q8b, dense, hyde, cands, kq, ref_map,
                                          a.thr, a.lo, a.hi, a.wd, a.b_cov, a.delta)
    write_json(out / f"cache_rrf3d2_{a.mode.replace('-', '')}.json", rrf3d2)
    write_json(out / f"cache_fhyde_{a.mode.replace('-', '')}.json", fhyde)
    fp = out / f"submission_{a.name}.json"
    write_json(fp, sub)
    write_json(out / f"doclist_{a.name}.json", {str(it["id"]): it["relevant_docs"] for it in sub})
    rep.update({"mode": a.mode, "name": a.name, "json_bytes": fp.stat().st_size,
                "cands": a.cands, "pool": a.pool, "thoi_gian_giay": round(time.time() - t0, 1)})
    code = 0
    if a.mode == "khong-moi":
        cmp_rep = compare_subs(sub, a.ref_sub) if a.ref_sub else {}
        rep["so_voi_bai_cu"] = cmp_rep
        print(f"sanity khong-moi vs {a.ref_sub}: {json.dumps(cmp_rep, ensure_ascii=False)}", flush=True)
        code = 0 if cmp_rep.get("ok") else 2
    write_json(out / f"report_chain_{a.name}.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "so_voi_bai_cu"}, ensure_ascii=False), flush=True)
    return code


def compare_subs(sub: list, ref_path: str) -> dict:
    """So sanh doc + chunk voi bai tham chieu (khop tuyet doi theo cau)."""
    ref = {str(it["id"]): it for it in load_json(ref_path)}
    n = n_doc = n_chunk = n_pos = 0
    vi_du = []
    for it in sub:
        q = str(it["id"])
        r = ref.get(q)
        n += 1
        if r is None:
            continue
        ok_d = [str(x) for x in it["relevant_docs"]] == [str(x) for x in r["relevant_docs"]]
        got = [(str(c["doc_id"]), c["chunk_text"]) for c in it["relevant_chunks"]]
        want = [(str(c["doc_id"]), c["chunk_text"]) for c in r["relevant_chunks"]]
        ok_c = got == want
        n_doc += ok_d
        n_chunk += ok_c
        n_pos += sum(1 for x, y in zip(got, want) if x == y)
        if (not ok_d or not ok_c) and len(vi_du) < 5:
            vi_du.append({"id": q, "doc_ok": ok_d, "chunk_ok": ok_c,
                          "n_doc": len(it["relevant_docs"]), "n_doc_ref": len(r["relevant_docs"]),
                          "n_chunk": len(got), "n_chunk_ref": len(want)})
    return {"so_cau": n, "cau_khop_doc": n_doc, "cau_khop_chunk": n_chunk,
            "ty_le_khop_doc": n_doc / max(n, 1), "ty_le_khop_chunk": n_chunk / max(n, 1),
            "vi_tri_chunk_khop": n_pos, "ok": n_doc == n and n_chunk == n, "vi_du_lech": vi_du}


# ---------------------------------------------------------------- fuse

def cmd_fuse(a) -> int:
    """Tinh rieng rrf3d2 (bge 1, 8B 1, dense 2) va fhyde (+ HyDE x1) - dung cho doclist truoc khi dung bai."""
    from fuse3 import fuse3 as fuse3_fn
    t0 = time.time()
    pool = load_cache(a.pool)
    q8b = load_cache(a.q8b)
    dense = load_cache(a.dense)
    hyde = load_cache(a.hyde) if a.hyde else {}
    rrf3d2 = fuse3_fn(pool, q8b, dense, (W_BGE, W_Q8B, W_DENSE, 0.0), K_RRF, None, None)
    w8f = W_Q8B if a.w_q8b_fhyde is None else a.w_q8b_fhyde   # vi33: 8B rieng cho fhyde
    fhyde = fuse3_fn(pool, q8b, dense, (W_BGE, w8f, W_DENSE, W_HYDE), K_RRF, hyde, None)
    write_json(a.out_rrf3d2, rrf3d2)
    write_json(a.out_fhyde, fhyde)
    rep = {"so_cau": len(pool), "rrf3d2": a.out_rrf3d2, "fhyde": a.out_fhyde,
           "TB_ung_vien_cau": sum(len(v) for v in fhyde.values()) / max(len(fhyde), 1),
           "thoi_gian_giay": round(time.time() - t0, 1)}
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- doclist (full8b)

def cmd_doclist(a) -> int:
    """Doclist theo luat full8b_sub --top-ens 1e6: n_q = adaptive_n(pool), nen = n_q doc dau fhyde,
    them doc nguon cua ref-chunks (truoc khi chon chunk - giong submission_split).
    vi35: --nq-fused-method ratio -> n_q = pick_nq_fused(fhyde[q], --nq-fused-thr, lo, hi, method)."""
    from union_sub import adaptive_n
    t0 = time.time()
    pool = load_cache(a.pool)
    fhyde = load_cache(a.fhyde)
    ref = load_json(a.ref_chunks) if a.ref_chunks else []
    method = getattr(a, "nq_fused_method", None)
    fthr = getattr(a, "nq_fused_thr", None)
    if method and fthr is None:
        raise SystemExit("--nq-fused-method can --nq-fused-thr")
    pick_nq_fused = None
    if method:
        from diag13 import pick_nq_fused
    out, ns, sizes = {}, [], []
    for q in sorted(pool, key=lambda x: int(x)):
        if method:
            n = pick_nq_fused(fhyde.get(q, []), fthr, a.lo, a.hi, method)
        else:
            n = adaptive_n(pool[q], a.thr, a.lo, a.hi)
        base = first_n_docs(fhyde.get(q, []), n)
        docs = full8b_docs(base, ref_map_chunks(ref).get(q, []))
        out[q] = docs
        ns.append(n)
        sizes.append(len(docs))
    write_json(a.out, out)
    n = max(len(out), 1)
    rep = {"n_q_TB": sum(ns) / n, "n_q_min": min(ns), "n_q_max": max(ns),
           "TB_doc_cau": sum(sizes) / n, "min_doc": min(sizes), "max_doc": max(sizes),
           "thr": a.thr, "lo": a.lo, "hi": a.hi,
           "nq_mode": "bge" if not method else f"fused_{method}", "nq_fused_thr": fthr,
           "thoi_gian_giay": round(time.time() - t0, 1), "ghi": a.out}
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


def ref_map_chunks(ref: list) -> dict:
    return {str(it["id"]): it["relevant_chunks"] for it in ref}


# ---------------------------------------------------------------- newonly30 + stats

def cmd_sanity_pairs(a) -> int:
    """Lay mau cap (q, doc) DA CO logit trong pool lam tham chieu de cham lai tren z87 (sanity logit)."""
    pool = load_cache(a.pool)
    pairs, ref = {}, {}
    for q in sorted(pool, key=lambda x: int(x))[: a.n_q]:
        rows = [h for h in pool[q] if h[2] is not None][: a.per]
        if not rows:
            continue
        pairs[q] = [[h[0], h[1]] for h in rows]
        ref[q] = {str(h[0]): h[2] for h in rows}
    write_json(a.out_pairs, pairs)
    write_json(a.out_ref, ref)
    rep = {"so_cau": len(pairs), "so_cap": sum(len(v) for v in pairs.values()),
           "ghi": [a.out_pairs, a.out_ref]}
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


def cmplogit_local(got: dict, ref: dict, tol: float) -> dict:
    """So logit theo (qid, doc_id) cho cap co o ca hai; ref = {qid: {str(doc_id): logit}}."""
    diffs, n_match, n_absent = [], 0, 0
    max_pair = None
    for q, rm in ref.items():
        gh = {str(h[0]): h[2] for h in got.get(q, [])}
        for d, sc in rm.items():
            g = gh.get(str(d))
            if g is None:
                n_absent += 1
                continue
            n_match += 1
            df = abs(float(g) - float(sc))
            diffs.append(df)
            if max_pair is None or df > max_pair[2]:
                max_pair = [q, d, df]
    return {"tol": tol, "so_cap_khop": n_match, "so_cap_thieu_trong_got": n_absent,
            "max_sai": max(diffs) if diffs else None,
            "TB_sai": sum(diffs) / len(diffs) if diffs else None,
            "cap_sai_nhat": max_pair,
            "ok": bool(diffs) and max(diffs) <= tol}


def cmd_cmplogit(a) -> int:
    ref_raw = load_cache(a.ref)
    ref = {}
    for q, hits in ref_raw.items():
        if isinstance(hits, dict):
            ref[q] = {str(k): v for k, v in hits.items()}
        else:
            ref[q] = {str(h[0]): h[2] for h in hits}
    rep = cmplogit_local(load_cache(a.got), ref, a.tol)
    write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0 if rep["ok"] else 2


def cmp_exact(a_cache: dict, b_cache: dict) -> dict:
    """Hai cache {qid: hits} bang nhau y nguyen (khoi + trinh tu + gia tri)."""
    ks_a, ks_b = set(a_cache), set(b_cache)
    n = n_eq = 0
    vi_du = []
    for q in sorted(ks_a & ks_b, key=lambda x: int(x)):
        n += 1
        if a_cache[q] == b_cache[q]:
            n_eq += 1
        elif len(vi_du) < 3:
            i = next((j for j in range(max(len(a_cache[q]), len(b_cache[q])))
                      if a_cache[q][j:j + 1] != b_cache[q][j:j + 1]), None)
            vi_du.append({"id": q, "vi_tri_lech": i,
                          "a": a_cache[q][i] if i is not None and i < len(a_cache[q]) else None,
                          "b": b_cache[q][i] if i is not None and i < len(b_cache[q]) else None})
    return {"cau_a": len(ks_a), "cau_b": len(ks_b), "cau_khop_y_nguyen": n_eq, "so_cau": n,
            "thieu_a": sorted(ks_b - ks_a, key=int)[:5], "thieu_b": sorted(ks_a - ks_b, key=int)[:5],
            "ok": n_eq == n and ks_a == ks_b and bool(n), "vi_du_lech": vi_du}


def cmd_cmp_exact(a) -> int:
    rep = cmp_exact(load_cache(a.a), load_cache(a.b))
    write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0 if rep["ok"] else 2


def cmd_newonly30(a) -> int:
    fhyde = load_cache(a.fhyde)
    new_ids = {str(i) for i in json.loads(Path(a.new_ids).read_text(encoding="utf-8"))}
    sub, n = [], []
    for q in sorted(fhyde, key=lambda x: int(x)):
        docs = []
        for d in hit_ids(fhyde[q]):
            if str(d) in new_ids:
                docs.append(d)
                if len(docs) >= a.n:
                    break
        n.append(len(docs))
        sub.append({"id": int(q) if str(q).isdigit() else q,
                    "relevant_docs": docs, "relevant_chunks": []})
    write_json(a.out, sub)
    rep = {"so_cau": len(sub), "n": a.n, "TB_doc_moi_cau": sum(n) / max(len(n), 1),
           "min_doc": min(n), "max_doc": max(n), "cau_rong": sum(1 for x in n if x == 0)}
    if a.report:
        rep["ghi"] = a.out
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


def host_of_doc(doc_id, links: dict) -> str:
    return links.get(str(doc_id), links.get(doc_id, ""))


def cmd_stats(a) -> int:
    sub = load_json(a.submission)
    fhyde = load_cache(a.fhyde) if a.fhyde else {}
    fhyde_old = load_cache(a.fhyde_old) if a.fhyde_old else {}
    pool = load_cache(a.pool) if a.pool else {}
    new_ids = {str(i) for i in json.loads(Path(a.new_ids).read_text(encoding="utf-8"))} if a.new_ids else set()
    n = max(len(sub), 1)
    n_doc = n_chunk = n_new_doc = n_new_chunk = 0
    top30_trung = 0.0
    host_new = Counter()
    links = json.loads(Path(a.links).read_text(encoding="utf-8")) if a.links else {}
    for it in sub:
        q = str(it["id"])
        docs = [str(d) for d in it["relevant_docs"]]
        chunks = it["relevant_chunks"]
        n_doc += len(docs)
        n_chunk += len(chunks)
        nd = sum(1 for d in docs if d in new_ids)
        nc = sum(1 for c in chunks if str(c["doc_id"]) in new_ids)
        n_new_doc += nd
        n_new_chunk += nc
        for d in docs:
            if d in new_ids:
                host_new[host_of_doc(d, links)] += 1
        if q in fhyde and q in fhyde_old:
            a30 = hit_ids(fhyde[q], 30)
            b30 = hit_ids(fhyde_old[q], 30)
            top30_trung += len({str(x) for x in a30} & {str(x) for x in b30}) / 30.0
    rep = {"so_cau": len(sub),
           "TB_doc_cau": n_doc / n, "TB_chunk_cau": n_chunk / n,
           "TB_doc_moi_cau": n_new_doc / n, "TB_chunk_tu_doc_moi_cau": n_new_chunk / n,
           "tong_doc_moi": n_new_doc, "tong_chunk_tu_doc_moi": n_new_chunk,
           "top30_trung_fhyde_cu_TB": top30_trung / n,
           "phan_bo_host_doc_moi_trong_danh_sach": dict(host_new.most_common()),
           "TB_doc_moi_trong_pool_cau": (sum(sum(1 for h in pool.get(q, []) if str(h[0]) in new_ids)
                                             for q in pool) / max(len(pool), 1)) if pool else None}
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- export text doc moi

def make_body(title: str, text: str, chars: int, no_title: bool = False) -> str:
    """Giong export_docs.make_body: title + '\\n' + text sach, cat chars; no_title thi chi text."""
    if no_title:
        return text[:chars]
    return (title + "\n" + text).strip()[:chars]


def cmd_export_new(a) -> int:
    """Ghi JSONL.zst {id, idx, text} cho doc MOI (bp23.clean) - giong export_docs tren chi muc cu.

    mode embed: title + text sach cat 3.000 ky tu (nhu docs_vi_all cho embed/rerank).
    mode docwin: text sach cat 200.000 ky tu, KHONG title (nhu chunkdocs/docwin_ext)."""
    import zstandard
    from boilerplate import BoilerplateFilter

    t0 = time.time()
    bp = BoilerplateFilter.from_json(Path(a.bp).read_text(encoding="utf-8"))
    idx_of = ({str(k): v for k, v in json.loads(Path(a.idx).read_text(encoding="utf-8")).items()}
              if a.idx else {})
    chars = a.chars if a.chars else (3000 if a.mode == "embed" else 200000)
    no_title = a.mode == "docwin"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = miss_idx = 0
    with out.open("wb") as fh, zstandard.ZstdCompressor(level=6).stream_writer(fh) as w:
        for p in sorted(Path(a.shards).glob(a.pattern)):
            for r in read_shard_records(p):
                did = int(r["id"])
                i = idx_of.get(str(did))
                if i is None:
                    miss_idx += 1
                    continue
                body = make_body(r.get("title") or "", bp.clean(r.get("host", ""), r.get("text") or ""),
                                 chars, no_title)
                w.write((json.dumps({"id": did, "idx": i, "text": body}, ensure_ascii=False) + "\n").encode("utf-8"))
                n += 1
    rep = {"mode": a.mode, "chars": chars, "no_title": no_title, "doc": n, "thieu_idx": miss_idx,
           "out": str(out), "bytes": out.stat().st_size, "thoi_gian_giay": round(time.time() - t0, 1)}
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- docwin cands

def ext_coverage(pool_hits: list, split_docs: list, top_n: int = 100) -> set:
    """Doc da co ung vien chunk (chunk_dw_u900_ext): top-`top_n` cache_pool_rr + relevant_docs split."""
    return {str(h[0]) for h in (pool_hits or [])[:top_n]} | {str(d) for d in (split_docs or [])}


def cmd_docwin_cands(a) -> int:
    """Cap (q, doc) trong doclist chua co ung vien chunk -> can docwin; tach khoi cap da chon docwin_p3."""
    t0 = time.time()
    doclist = {str(k): v for k, v in load_json(a.doclist).items()}
    pool = load_cache(a.pool)
    split = {str(it["id"]): it["relevant_docs"] for it in load_json(a.split)}
    p3_wins = ({str(q): v for q, v in load_json(a.p3_docwin).items()} if a.p3_docwin else {})
    p3_cands = ({str(q): [str(x[0]) for x in v] for q, v in load_json(a.p3_cands).items()}
                if a.p3_cands else {})
    new_ids = {str(i) for i in json.loads(Path(a.new_ids).read_text(encoding="utf-8"))} if a.new_ids else set()
    idx_of = {}
    for p in a.idx_from:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        for k, v in raw.items():
            if isinstance(v, list):
                for h in v:
                    if len(h) > 1 and h[1] is not None:
                        idx_of.setdefault(str(h[0]), int(h[1]))
            elif v is not None:
                idx_of.setdefault(str(k), int(v))
    c_reuse, c_new, k_old, k_new = {}, {}, {}, {}
    n_reuse = n_new = n_old = n_thieu_idx = 0
    for q in sorted(doclist, key=lambda x: int(x)):
        cov = ext_coverage(pool.get(q, []), split.get(q, []), a.top_n)
        for d in doclist[q]:
            s = str(d)
            if s in cov:
                continue
            i = idx_of.get(s)
            if i is None:
                n_thieu_idx += 1
                continue
            have_p3 = s in {str(x) for x in p3_cands.get(q, [])} and s in {str(k) for k in p3_wins.get(q, {})}
            if have_p3:
                c_reuse.setdefault(q, []).append([d, i])
                n_reuse += 1
            else:
                c_new.setdefault(q, []).append([d, i])
                if s in new_ids:
                    k_new.setdefault(q, []).append([d, i, 0.0])
                    n_new += 1
                else:
                    k_old.setdefault(q, []).append([d, i, 0.0])
                    n_old += 1
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "cands_docwin_p3reuse.json", c_reuse)
    write_json(out / "cands_docwin_vi23.json", c_new)
    write_json(out / "cache_docwin_old.json", k_old)
    write_json(out / "cache_docwin_new.json", k_new)
    rep = {"doclist": a.doclist, "cap_p3_tai_dung": n_reuse, "cap_can_docwin_moi": n_new + n_old,
           "cap_doc_moi": n_new, "cap_doc_cu": n_old, "cap_thieu_idx": n_thieu_idx,
           "TB_cap_moi_cau": (n_new + n_old) / max(len(doclist), 1),
           "TB_p3_cau": n_reuse / max(len(doclist), 1),
           "so_cau_co_cap_moi": sum(1 for v in c_new.values() if v),
           "ghi": [str(out / "cands_docwin_p3reuse.json"), str(out / "cands_docwin_vi23.json"),
                   str(out / "cache_docwin_old.json"), str(out / "cache_docwin_new.json")],
           "thoi_gian_giay": round(time.time() - t0, 1)}
    write_json(out / "docwin_cands_report.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


def cmd_wins_merge(a) -> int:
    """Gop cands + docwin wins cua 2 nguon -> 1 cands + 1 docwin (cho chunks_from_docwin mot lan)."""
    ca = {str(q): v for q, v in load_json(a.cands_a).items()}
    cb = {str(q): v for q, v in load_json(a.cands_b).items()}
    wa = {str(q): v for q, v in load_json(a.docwin_a).items()}
    wb = {str(q): v for q, v in load_json(a.docwin_b).items()}
    cands, wins = {}, {}
    for q in sorted(set(ca) | set(cb), key=lambda x: int(x)):
        rows, seen = [], set()
        for item in ca.get(q, []) + cb.get(q, []):
            s = str(item[0])
            if s in seen:
                continue
            seen.add(s)
            rows.append(item)
        if rows:
            cands[q] = rows
        m = {str(k): v for k, v in wa.get(q, {}).items()}
        m.update({str(k): v for k, v in wb.get(q, {}).items()})
        wins[q] = {k: v for k, v in m.items() if k in seen}
    write_json(a.out_cands, cands)
    write_json(a.out_docwin, wins)
    n = max(len(cands), 1)
    rep = {"so_cau": len(cands), "TB_cap_cau": sum(len(v) for v in cands.values()) / n,
           "so_cau_wins": len(wins), "ghi": [a.out_cands, a.out_docwin]}
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- check-zip

def check_zip(json_path: str, zip_path: str, report_path: str, name: str = "vi23",
              max_zip: int = MAX_ZIP) -> int:
    """Doc lai zip (1 entry submission.json) == JSON goc + zip <= max_zip; ghi next_delta neu vuot.

    Exit: 0 = ok; 1 = noi dung khac; 3 = vuot (ghi next_delta_<name>.txt); 4 = vuot nhung delta da 0."""
    import hashlib
    import zipfile
    from cx110 import suggest_next_delta
    from diag14 import verify_zip as verify_zip_bytes

    p, zp = Path(json_path), Path(zip_path)
    st = dict(verify_zip_bytes(str(p), str(zp)))
    zb, jb = zp.stat().st_size, p.stat().st_size
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    duoi = zb <= max_zip
    rp = Path(report_path)
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    d_used = report.get("delta")
    n_chunks = report.get("tong_chunk", 0)
    code, next_d = 0, None
    if not st["ok"]:
        code = 1
    elif not duoi:
        if not d_used or d_used <= 0:
            code = 4
        else:
            hist = report.get("hist_kcap", [])
            next_d = suggest_next_delta(hist, int(d_used), n_chunks, zb, max_zip, zb / max(n_chunks, 1))
            code = 3
    entry = {"name": name, "delta": d_used, "json_bytes": jb, "zip_bytes": zb,
             "duoi_gioi_han": duoi, "sha256": sha, "sha256_rut_gon": sha[:16],
             "bang_goc_bytes": st["bang_goc_bytes"], "bang_goc_json": st["bang_goc_json"],
             "entry": st["entry"], "next_delta": next_d, "ok": code == 0}
    report.setdefault("zip", {})[name] = entry
    rp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    if next_d is not None:
        (p.parent / f"next_delta_{name}.txt").write_text(f"--delta {next_d}", encoding="utf-8")
    print(json.dumps(entry, ensure_ascii=False), flush=True)
    return code


# ---------------------------------------------------------------- mau doc + cmp emb

def cmd_sample_docs(a) -> int:
    """Lay n ban ghi dau cua file JSONL.zst {id, idx, text} ra file moi (sanity embed lai doc cu)."""
    import zstandard
    rows = []
    for p in sorted(Path(a.docs).glob(a.pattern)):
        rows.extend(read_shard_records(p))
        if len(rows) >= a.n:
            break
    rows = rows[: a.n]
    out = Path(a.out)
    with out.open("wb") as fh, zstandard.ZstdCompressor(level=6).stream_writer(fh) as w:
        for r in rows:
            w.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    rep = {"so_doc": len(rows), "out": str(out), "bytes": out.stat().st_size}
    if a.report:
        write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


def load_blocks(dirs) -> dict:
    """{doc_id: emb vector} tu cac thu muc blk_*.npz (embed_run/embed_all)."""
    import numpy as np
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("blk_*.npz")):
            z = np.load(f)
            for did, vec in zip(z["ids"], z["emb"]):
                out[int(did)] = vec
    return out


def cmd_cmp_emb(a) -> int:
    """Sanity embed: so embedding moi tinh lai voi embedding da luu (cung doc, cung mo hinh).

    Bao: max |lech|, min/max cosine, so doc khac hon --tol."""
    import numpy as np
    new = load_blocks([a.new_dir])
    old = load_blocks(a.old_dirs)
    rows = []
    for did, vec in new.items():
        o = old.get(did)
        if o is None:
            continue
        v1 = vec.astype(np.float32)
        v2 = o.astype(np.float32)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        cos = float(v1 @ v2 / (n1 * n2)) if n1 and n2 else 0.0
        rows.append((did, cos, float(np.max(np.abs(v1 - v2)))))
    if not rows:
        rep = {"so_doc": 0, "ok": False, "loi": "khong co doc nao trung giua new va old"}
        write_json(a.report, rep)
        print(json.dumps(rep, ensure_ascii=False), flush=True)
        return 2
    coss = [r[1] for r in rows]
    diffs = [r[2] for r in rows]
    rep = {"so_doc": len(rows), "cosine_min": min(coss), "cosine_TB": sum(coss) / len(coss),
           "cosine_max": max(coss), "max_abs_lech": max(diffs),
           "doc_cosine_duoi": sum(1 for c in coss if c < a.tol),
           "ok": min(coss) >= a.tol}
    write_json(a.report, rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0 if rep["ok"] else 2


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("manifest")
    p.add_argument("--shards", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(fn=cmd_manifest)

    p = sp.add_parser("prep")
    p.add_argument("--shards", required=True, help="thu muc chua vi-recov-*.jsonl.zst")
    p.add_argument("--pattern", default="*.jsonl.zst")
    p.add_argument("--old-bp", required=True)
    p.add_argument("--missing", required=True)
    p.add_argument("--old-ids", default="", help="ids.npy cua chi muc cu")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bp-out", default="", help="thu muc ghi boilerplate.json gop")
    p.set_defaults(fn=cmd_prep)

    p = sp.add_parser("verify-idx")
    p.add_argument("--index-old-ids", required=True)
    p.add_argument("--index-new-ids", required=True)
    p.add_argument("--new-ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(fn=cmd_verify_idx)

    p = sp.add_parser("bm25-diff")
    p.add_argument("--cache-old", required=True)
    p.add_argument("--cache-new", required=True)
    p.add_argument("--new-ids", default="")
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--report", required=True)
    p.set_defaults(fn=cmd_bm25_diff)

    p = sp.add_parser("dense-merge")
    p.add_argument("--cache-old", required=True)
    p.add_argument("--cache-new", required=True)
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--out", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_dense_merge)

    p = sp.add_parser("pool-new")
    p.add_argument("--pool", required=True)
    p.add_argument("--bm25-new", required=True)
    p.add_argument("--dense-new", required=True)
    p.add_argument("--new-ids", required=True)
    p.add_argument("--idx", default="", help="new_idx.json (idx trong chi muc vi23)")
    p.add_argument("--out-cands", required=True, help="cands cho rerank_run.py")
    p.add_argument("--out-new-hits", default="", help="hit moi chua logit")
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_pool_new)

    p = sp.add_parser("pool-merge")
    p.add_argument("--pool-old", required=True)
    p.add_argument("--new-hits", default="")
    p.add_argument("--rerank", default="", help="logit rerank cua cap moi")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_pool_merge)

    p = sp.add_parser("export-new")
    p.add_argument("--shards", required=True)
    p.add_argument("--pattern", default="*.jsonl.zst")
    p.add_argument("--bp", required=True, help="boilerplate.json gop (vi23)")
    p.add_argument("--idx", default="", help="new_idx.json")
    p.add_argument("--mode", choices=["embed", "docwin"], required=True)
    p.add_argument("--chars", type=int, default=0)
    p.add_argument("--out", required=True, help=".jsonl.zst {id, idx, text}")
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_export_new)

    p = sp.add_parser("docwin-cands")
    p.add_argument("--doclist", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--p3-docwin", default="", help="docwin_p3.json.zst (cap da chon docwin)")
    p.add_argument("--p3-cands", default="", help="cands_docwin_p3.json")
    p.add_argument("--new-ids", default="")
    p.add_argument("--idx-from", nargs="*", default=[])
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_docwin_cands)

    p = sp.add_parser("wins-merge")
    p.add_argument("--cands-a", required=True)
    p.add_argument("--docwin-a", required=True)
    p.add_argument("--cands-b", required=True)
    p.add_argument("--docwin-b", required=True)
    p.add_argument("--out-cands", required=True)
    p.add_argument("--out-docwin", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_wins_merge)

    p = sp.add_parser("check-zip")
    p.add_argument("--json", required=True)
    p.add_argument("--zip", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--name", default="vi23")
    p.add_argument("--max-zip", type=int, default=MAX_ZIP)
    p.set_defaults(fn=lambda a: check_zip(a.json, a.zip, a.report, a.name, a.max_zip))

    p = sp.add_parser("chain")
    p.add_argument("--mode", choices=["khong-moi", "that"], required=True)
    p.add_argument("--name", default="vi23")
    p.add_argument("--pool", required=True)
    p.add_argument("--q8b", required=True)
    p.add_argument("--dense", required=True)
    p.add_argument("--hyde", default="")
    p.add_argument("--cands", required=True, help="ung vien chunk (.json/.json.zst)")
    p.add_argument("--ref-kq", required=True, help="submission_c120in.json (k_q)")
    p.add_argument("--ref-chunks", default="", help="ref chunk cho luat them doc (full8b)")
    p.add_argument("--ref-sub", default="", help="bai tham chieu de so sanh (che do khong-moi)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thr", type=float, default=THR_N)
    p.add_argument("--lo", type=int, default=LO_N)
    p.add_argument("--hi", type=int, default=HI_N)
    p.add_argument("--wd", type=float, default=WD)
    p.add_argument("--b-cov", type=float, default=B_COV)
    p.add_argument("--delta", type=int, default=DELTA)
    p.set_defaults(fn=cmd_chain)

    p = sp.add_parser("fuse")
    p.add_argument("--pool", required=True)
    p.add_argument("--q8b", required=True)
    p.add_argument("--dense", required=True)
    p.add_argument("--hyde", default="")
    p.add_argument("--out-rrf3d2", required=True)
    p.add_argument("--out-fhyde", required=True)
    p.add_argument("--w-q8b-fhyde", type=float, default=None,
                   help="trong so 8B chi cho fhyde (mac dinh = W_Q8B); rrf3d2 khong doi")
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_fuse)

    p = sp.add_parser("doclist")
    p.add_argument("--pool", required=True)
    p.add_argument("--fhyde", required=True)
    p.add_argument("--ref-chunks", default="")
    p.add_argument("--thr", type=float, default=THR_N)
    p.add_argument("--lo", type=int, default=LO_N)
    p.add_argument("--hi", type=int, default=HI_N)
    p.add_argument("--nq-fused-method", default=None,
                   help="chon n_q theo diem fused (ratio/zscore); mac dinh None = adaptive_n bge")
    p.add_argument("--nq-fused-thr", type=float, default=None,
                   help="nguong cho --nq-fused-method (bat buoc khi co method)")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_doclist)

    p = sp.add_parser("sanity-pairs")
    p.add_argument("--pool", required=True, help="cache_pool_rr.json (cap da co logit)")
    p.add_argument("--n-q", type=int, default=20)
    p.add_argument("--per", type=int, default=10, help="so cap/can")
    p.add_argument("--out-pairs", required=True)
    p.add_argument("--out-ref", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_sanity_pairs)

    p = sp.add_parser("cmplogit")
    p.add_argument("--got", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--report", required=True)
    p.set_defaults(fn=cmd_cmplogit)

    p = sp.add_parser("cmp-exact")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(fn=cmd_cmp_exact)

    p = sp.add_parser("sample-docs")
    p.add_argument("--docs", required=True, help="thu muc JSONL.zst {id, idx, text}")
    p.add_argument("--pattern", default="*.jsonl.zst")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_sample_docs)

    p = sp.add_parser("cmp-emb")
    p.add_argument("--new-dir", required=True)
    p.add_argument("--old-dirs", required=True, nargs="+")
    p.add_argument("--tol", type=float, default=0.999)
    p.add_argument("--report", required=True)
    p.set_defaults(fn=cmd_cmp_emb)

    p = sp.add_parser("newonly30")
    p.add_argument("--fhyde", required=True)
    p.add_argument("--new-ids", required=True)
    p.add_argument("--n", type=int, default=N_NEWONLY)
    p.add_argument("--out", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_newonly30)

    p = sp.add_parser("stats")
    p.add_argument("--submission", required=True)
    p.add_argument("--fhyde", default="")
    p.add_argument("--fhyde-old", default="")
    p.add_argument("--pool", default="")
    p.add_argument("--new-ids", default="")
    p.add_argument("--links", default="", help="json {doc_id: host}")
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_stats)

    a = ap.parse_args()
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
