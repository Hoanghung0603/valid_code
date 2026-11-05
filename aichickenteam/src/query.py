"""Query understanding: Vietnamese medical query -> multilingual QueryPlan.

QueryPlan (docs/INTERFACES.md):
    {"id", "query_vi", "vi": [...], "en": [...], "zh": [...], "pubmed": [...],
     "keywords": {"vi": [...], "en": [...], "zh": [...]}}

One JSON-output LLM prompt per query (batched, greedy). Robust parsing, one retry
with a larger token budget, then a deterministic fallback plan. Results are cached
in sqlite (data/cache/query.sqlite) keyed by (id, sha1(query), prompt_version).

CLI:
    python -m src.query --queries data/ViBioMIR/query.parquet \
        --out data/cache/query_plans.jsonl [--limit N] [--ids 1,2,3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any

from src.llm import parse_json

log = logging.getLogger("src.query")

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "v11"
PREFILL = '{"terms": [["'  # forces compact one-line JSON (fewer tokens)

DEFAULT_CFG: dict[str, Any] = {
    "model": "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
    "quant": "4bit",
    "prompt_version": PROMPT_VERSION,
    "batch_size": 10,
    "max_batch_tokens": 12000,
    "max_new_tokens": 600,
    "retry_max_new_tokens": 1000,
    "max_input_tokens": 3072,
    "cache": "data/cache/query.sqlite",
    "fewshot": True,
}

SYSTEM = """You are a medical terminologist (Vietnamese/English/Chinese) writing search queries for cross-lingual medical retrieval (Vietnamese and Chinese health websites, PubMed).
Input: a Vietnamese health question, often a long patient message with several questions.
Output ONLY a JSON object with keys in this order:
"terms": 3-6 key medical concepts (disease, symptom, test, drug, procedure, anatomy) as ["Vietnamese", "English medical term (MeSH style)", "简体中文医学术语"];
"vi": 1-3 concise Vietnamese search queries: first the main information need (condition + what is asked), then one per other distinct sub-question only if several things are really asked (never invent questions the patient did not ask). Keep key facts (age group, pregnancy week, test values, drugs); drop greetings and story details;
"en": English translations of "vi" (same order), precise medical terminology;
"zh": Simplified Chinese translations of "vi" (same order), standard Chinese medical terms;
"pubmed": 2-3 short English PubMed keyword queries, 2-4 concepts each, synonyms may be grouped as (a OR b), no field tags or quotes.
Rules: translate the exact concept written, never add details or diagnoses not stated (e.g. "tiểu đường" = diabetes mellitus, no type unless stated). Expand abbreviations, fix typos. Pitfalls: van 2 lá = mitral, van 3 lá = tricuspid; hành tá tràng = duodenal bulb; a-mi-đan = tonsils (扁桃体); ọe = retching (干呕); bướu cổ = goiter (甲状腺肿); mũi tiêm/mũi cuối = vaccine dose/last dose; buồng trứng = ovary (卵巢), viêm buồng trứng = oophoritis (卵巢炎); vòi trứng = fallopian tube (输卵管); TNTC = ectopic pregnancy; viêm bàng quang = cystitis; đồng âm/giảm âm/tăng âm = isoechoic/hypoechoic/hyperechoic; ba/bố = father; lao đa kháng = MDR-TB (耐多药结核病); Beta-HCG = β-hCG."""

EX_Q = ("Em bị lật cổ chân khi đá bóng 3 ngày nay, cổ chân sưng to và bầm tím, đi lại đau. "
        "Cho em hỏi có cần chụp X-quang không và bao lâu thì khỏi ạ?")
EX_A = {
    "terms": [["bong gân cổ chân", "ankle sprain", "踝关节扭伤"],
              ["sưng cổ chân", "ankle swelling", "踝关节肿胀"],
              ["bầm tím", "ecchymosis", "瘀斑"],
              ["chụp X-quang", "radiography", "X线检查"],
              ["thời gian hồi phục", "recovery time", "恢复时间"]],
    "vi": ["Bong gân cổ chân sưng bầm tím có cần chụp X-quang không?",
           "Bong gân cổ chân bao lâu thì khỏi?"],
    "en": ["Ankle sprain with swelling and bruising: is radiography needed?",
           "Recovery time of ankle sprain"],
    "zh": ["踝关节扭伤伴肿胀瘀斑是否需要拍X线片？",
           "踝关节扭伤多久能恢复？"],
    "pubmed": ["ankle sprain radiography indication",
               "ankle injuries Ottawa ankle rules",
               "ankle sprain recovery time (rehabilitation OR treatment)"],
}
_EX_MARKERS = {"ankle": ["co chan", "mat ca"], "sprain": ["bong gan", "lat", "trat"],
               "ottawa": ["ottawa"], "ecchymosis": ["bam"], "radiography": ["x-quang", "x quang", "chup phim"]}


FORMAT = ('\nAnswer on one line exactly like: {"terms": [["<Vietnamese term>", "<English term>", "<中文术语>"], ...], '
          '"vi": ["<main query>", "<sub-question>"], "en": ["...", "..."], "zh": ["...", "..."], '
          '"pubmed": ["<keywords>", "<keywords>"]}')


def build_messages(query: str, fewshot: bool = False) -> list[dict]:
    """Chat messages for one query. The few-shot example is deliberately off-domain
    (ankle sprain): a hypertension/diabetes example got copied into unrelated plans by
    the 4B model; leftovers are removed by the leak guard in normalize_plan."""
    if not fewshot:
        return [{"role": "system", "content": SYSTEM + FORMAT},
                {"role": "user", "content": query}]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EX_Q},
        {"role": "assistant", "content": json.dumps(EX_A, ensure_ascii=False)},
        {"role": "user", "content": query},
    ]


_KEYS = ("terms", "vi", "en", "zh", "pubmed")
_QSTR = re.compile(r'"((?:[^"\\]|\\.)*)"')


def parse_plan_json(text: str) -> dict | None:
    """parse_json, then per-key salvage when the object is malformed (e.g. a stray
    quote inside one "terms" triplet must not lose vi/en/zh/pubmed)."""
    d = parse_json(text)
    if isinstance(d, dict) and all(d.get(k) for k in ("vi", "en", "zh")):
        return d
    if not text:
        return d if isinstance(d, dict) else None
    out: dict = {}
    pos = [(m.start(), m.group(1)) for m in re.finditer(r'"(%s)"\s*:' % "|".join(_KEYS), text)]
    for n, (st, key) in enumerate(pos):
        seg = text[st:pos[n + 1][0] if n + 1 < len(pos) else len(text)]
        seg = seg[seg.find(":") + 1:]
        try:
            v = json.loads(seg.strip().rstrip(",}").strip().rstrip(","))
        except Exception:
            strs = [json.loads('"%s"' % x) if "\\" in x else x for x in _QSTR.findall(seg)]
            if key == "terms":
                v = [strs[i:i + 3] for i in range(0, len(strs) - 2, 3)]
            else:
                v = strs
        out.setdefault(key, v)
    if isinstance(d, dict):  # keep what the strict/repair parse got right
        for k, v in d.items():
            if v and not out.get(k):
                out[k] = v
    return out or None


# --------------------------------------------------------------------- utils
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def qhash(q: str) -> str:
    return hashlib.sha1(nfc(q).encode("utf-8")).hexdigest()


_CJK = re.compile(r"[一-鿿]")
_VI_DIACRITIC = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)


def _str_list(x: Any, max_n: int, max_len: int = 400) -> list[str]:
    if isinstance(x, str):
        x = [x]
    if not isinstance(x, list):
        return []
    out, seen = [], set()
    for v in x:
        if isinstance(v, dict):  # e.g. {"query": "..."}
            v = next((s for s in v.values() if isinstance(s, str)), "")
        if not isinstance(v, str):
            continue
        v = nfc(v)
        v = re.sub(r"\s+", " ", v)[:max_len].strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
        if len(out) >= max_n:
            break
    return out


def _clean_pubmed(s: str) -> str:
    s = re.sub(r"\[[^\]]*\]", " ", s)          # field tags [MeSH Terms]
    s = s.replace('"', " ").replace("“", " ").replace("”", " ")
    s = re.sub(r"\bAND\b", " ", s)            # implicit AND is enough
    s = re.sub(r"\s+", " ", s).strip()
    # balance parentheses
    if s.count("(") != s.count(")"):
        s = s.replace("(", " ").replace(")", " ")
        s = re.sub(r"\s+", " ", s).strip()
    return s


def _fold(s: str) -> str:
    """lowercase, strip Vietnamese diacritics (đ->d) for loose matching."""
    s = unicodedata.normalize("NFD", (s or "").lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _leaked_text(en: str, qn: str) -> bool:
    """English text mentions a few-shot-example concept that the query does not."""
    e = (en or "").lower()
    return any(k in e and not any(m in qn for m in ms) for k, ms in _EX_MARKERS.items())


def _leaked(term: list, qn: str) -> bool:
    return isinstance(term[1], str) and _leaked_text(term[1], qn)


def normalize_plan(raw: Any, qid: int, query: str) -> tuple[dict, list[str]]:
    """Coerce LLM JSON into a valid QueryPlan. Returns (plan, list_of_problems)."""
    problems: list[str] = []
    raw = raw if isinstance(raw, dict) else {}
    qn = _fold(query)
    vi = _str_list(raw.get("vi"), 5)
    en = _str_list(raw.get("en"), 5)
    zh = _str_list(raw.get("zh"), 5)
    bad = {i for i, s in enumerate(en) if _leaked_text(s, qn)}
    if bad:
        problems.append("leak")
        vi = [s for i, s in enumerate(vi) if i not in bad] if len(vi) == len(en) else vi
        zh = [s for i, s in enumerate(zh) if i not in bad] if len(zh) == len(en) else zh
        en = [s for i, s in enumerate(en) if i not in bad]
    en = [s for s in en if not _CJK.search(s)]
    zh = [s for s in zh if _CJK.search(s)]
    pub = [p for p in (_clean_pubmed(s) for s in _str_list(raw.get("pubmed"), 6, 200))
           if p and not _CJK.search(p) and len(p.split()) <= 14
           and not _leaked_text(p, qn)][:5]
    kw = raw.get("keywords") if isinstance(raw.get("keywords"), dict) else {}
    tv, te, tz = [], [], []
    for t in raw.get("terms") or []:
        if isinstance(t, dict):
            t = [t.get("vi"), t.get("en"), t.get("zh")]
        if isinstance(t, list):
            t = (list(t) + [None, None, None])[:3]
            if _leaked(t, qn):
                problems.append("leak")
                continue
            tv.append(t[0]); te.append(t[1]); tz.append(t[2])
    kvi = _str_list(tv + list(kw.get("vi") or []), 10, 80)
    ken = [s for s in _str_list(te + list(kw.get("en") or []), 10, 80)
           if not _CJK.search(s) and not _VI_DIACRITIC.search(s)]
    kzh = [s for s in _str_list(tz + list(kw.get("zh") or []), 10, 80)
           if not _VI_DIACRITIC.search(s)]
    for name, v in (("vi", vi), ("en", en), ("zh", zh), ("pubmed", pub),
                    ("kw.en", ken), ("kw.zh", kzh)):
        if not v:
            problems.append(f"empty:{name}")
    if en and any(_VI_DIACRITIC.search(s) for s in en):
        problems.append("vi_in_en")
    if not pub and en:  # derive from English rewrites
        pub = [" ".join(re.findall(r"[A-Za-z0-9\-]+", s)[:8]) for s in en[:3]]
    if not vi:
        vi = [nfc(query)]
    plan = {"id": int(qid), "query_vi": nfc(query), "vi": vi, "en": en, "zh": zh,
            "pubmed": pub, "keywords": {"vi": kvi, "en": ken, "zh": kzh}}
    return plan, problems


def fallback_plan(qid: int, query: str) -> dict:
    return {"id": int(qid), "query_vi": nfc(query), "vi": [nfc(query)], "en": [], "zh": [],
            "pubmed": [], "keywords": {"vi": [], "en": [], "zh": []}}


# --------------------------------------------------------------------- cache
class PlanCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("""CREATE TABLE IF NOT EXISTS plans (
            id INTEGER, qhash TEXT, pv TEXT, plan TEXT, status TEXT, raw TEXT, ts REAL,
            PRIMARY KEY (id, qhash, pv))""")
        self.db.commit()

    def get(self, qid: int, h: str, pv: str) -> dict | None:
        r = self.db.execute("SELECT plan FROM plans WHERE id=? AND qhash=? AND pv=?",
                            (int(qid), h, pv)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, qid: int, h: str, pv: str, plan: dict, status: str, raw: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO plans VALUES (?,?,?,?,?,?,?)",
                        (int(qid), h, pv, json.dumps(plan, ensure_ascii=False), status,
                         raw, time.time()))
        self.db.commit()


# --------------------------------------------------------------------- main
def load_cfg(path: str | Path | None = None) -> dict:
    cfg = dict(DEFAULT_CFG)
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    if p.exists():
        try:
            import yaml
            y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cfg.update(y.get("query") or {})
        except Exception as e:  # pragma: no cover
            log.warning("cannot read %s: %s", p, e)
    return cfg


def prompt_key(cfg: dict) -> str:
    """Cache key part: configured version + hash of the actual prompt/model."""
    h = hashlib.sha1((json.dumps(build_messages("", bool(cfg.get("fewshot"))), ensure_ascii=False)
                      + PREFILL + str(cfg.get("model"))).encode("utf-8")).hexdigest()[:8]
    return f"{cfg['prompt_version']}-{h}"


def build_plans(queries: list[dict], cfg: dict | None = None, llm=None,
                use_cache: bool = True) -> list[dict]:
    """queries: [{"id": int, "query": str}] -> list[QueryPlan] (same order)."""
    cfg = {**load_cfg(), **(cfg or {})}
    pv = prompt_key(cfg)
    cache = PlanCache(cfg["cache"]) if use_cache else None
    results: dict[int, dict] = {}
    todo = []
    for k, q in enumerate(queries):
        h = qhash(q["query"])
        hit = cache.get(q["id"], h, pv) if cache else None
        if hit is not None:
            results[k] = hit
        else:
            todo.append((k, q, h))
    log.info("query plans: %d cached, %d to generate", len(results), len(todo))
    if todo:
        if llm is None:
            from src.llm import get_llm
            llm = get_llm(cfg["model"], quant=cfg["quant"],
                          max_input_tokens=int(cfg["max_input_tokens"]))
        chunk = max(1, int(cfg["batch_size"])) * 4  # write cache progressively
        t0 = time.time()
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            _generate_part(part, llm, cfg, pv, cache, results)
            done = s + len(part)
            el = time.time() - t0
            log.info("generated %d/%d (%.2f s/query, ETA %.1f min) %s", done, len(todo),
                     el / done, el / done * (len(todo) - done) / 60,
                     getattr(llm, "stats", ""))
    return [results[k] for k in range(len(queries))]


def _generate_part(part, llm, cfg, pv, cache, results) -> None:
    msgs = [build_messages(nfc(q["query"]), bool(cfg.get("fewshot"))) for _, q, _ in part]
    outs = llm.generate(msgs, max_new_tokens=int(cfg["max_new_tokens"]), prefill=PREFILL,
                        batch_size=int(cfg["batch_size"]),
                        max_batch_tokens=int(cfg["max_batch_tokens"]))
    retry = []
    for (k, q, h), out in zip(part, outs):
        plan, probs = normalize_plan(parse_plan_json(out), q["id"], q["query"])
        if "empty:en" in probs or "empty:zh" in probs:  # parse failure / truncation
            retry.append((k, q, h, out))
            continue
        results[k] = plan
        if cache:
            cache.put(q["id"], h, pv, plan, "ok" if not probs else ",".join(probs), out)
    for k, q, h, out0 in retry:  # one-by-one retry with larger budget
        log.warning("retry id=%s", q["id"])
        out = llm.generate([build_messages(nfc(q["query"]), bool(cfg.get("fewshot")))],
                           max_new_tokens=int(cfg["retry_max_new_tokens"]), prefill=PREFILL,
                           batch_size=1)[0]
        plan, probs = normalize_plan(parse_plan_json(out), q["id"], q["query"])
        if not plan["en"] and not plan["zh"]:
            plan, probs, out = fallback_plan(q["id"], q["query"]), ["fallback"], out or out0
        results[k] = plan
        if cache:
            cache.put(q["id"], h, pv, plan, ",".join(probs) or "ok_retry", out)


def load_plans(path: str | Path = "data/cache/query_plans.jsonl") -> dict[int, dict]:
    """Read plans jsonl -> {id: plan}."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[int(d["id"])] = d
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queries", default="data/ViBioMIR/query.parquet")
    ap.add_argument("--out", default="data/cache/query_plans.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated ids subset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    qp = Path(a.queries)
    if qp.suffix == ".parquet":
        import pandas as pd
        qs = pd.read_parquet(qp)[["id", "query"]].to_dict("records")
    else:
        qs = [json.loads(l) for l in open(qp, encoding="utf-8") if l.strip()]
    qs = [{"id": int(q["id"]), "query": str(q["query"])} for q in qs]
    if a.ids:
        want = {int(x) for x in a.ids.split(",")}
        qs = [q for q in qs if q["id"] in want]
    if a.limit:
        qs = qs[:a.limit]
    cfg = load_cfg(a.config)
    if a.batch_size:
        cfg["batch_size"] = a.batch_size
    t0 = time.time()
    plans = build_plans(qs, cfg, use_cache=not a.no_cache)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in plans:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("wrote %d plans to %s in %.1fs", len(plans), out, time.time() - t0)


if __name__ == "__main__":
    main()
