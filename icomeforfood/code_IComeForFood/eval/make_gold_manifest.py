"""Pick the gold-set pages and write domain statistics.

Writes:
    <out>/gold/manifest.csv       id,lang,domain,url  (pages NT copies by hand)
    <out>/reports/domain_stats.csv domain,lang,n_urls,share,cum_share,wave

Usage:
    python eval/make_gold_manifest.py --corpus D:/r2ai_data/raw/links_corpus_vi_en_zh.parquet \
        --out D:/r2ai_data [--n-vi 18 --n-zh 14 --n-en 8 --seed 2026]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ZH_DOMAIN_HINTS = ("120ask.com", "39.net", "cnkang.com", "familydoctor", "zysjonline.com",
                   "zhongyibaodian.net", "a-hospital.com", "youlai.cn", "wujue.com",
                   "jb39.com", "iiyi.com", "zydcd.com", "qihuangzhishu.com", "baidu.com",
                   "pmphai.com", "msdmanuals.cn", "people.com.cn")


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def lang_of(domain: str) -> str:
    if domain == "pubmed.ncbi.nlm.nih.gov":
        return "en"
    if domain.endswith(".cn") or any(h in domain for h in ZH_DOMAIN_HINTS):
        return "zh"
    return "vi"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-vi", type=int, default=18)
    ap.add_argument("--n-zh", type=int, default=14)
    ap.add_argument("--n-en", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    c = pd.read_parquet(args.corpus)
    c["domain"] = c["url"].map(domain_of)
    c["lang"] = c["domain"].map(lang_of)
    # skip bare home pages; they are not articles
    c = c[c["url"].map(lambda u: len(urlparse(u).path) > 1)]

    stats = (c.groupby(["domain", "lang"]).size().rename("n_urls").reset_index()
             .sort_values("n_urls", ascending=False))
    stats["share"] = stats["n_urls"] / stats["n_urls"].sum()
    stats["cum_share"] = stats["share"].cumsum()
    rank = range(1, len(stats) + 1)
    stats["wave"] = [1 if r <= 20 else 2 if r <= 50 else 3 for r in rank]
    (args.out / "reports").mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out / "reports" / "domain_stats.csv", index=False)

    picks = []
    for lang, n in (("vi", args.n_vi), ("zh", args.n_zh), ("en", args.n_en)):
        doms = stats[stats["lang"] == lang]["domain"].tolist()
        # one page per domain in rank order, cycling until n pages are picked
        order = [doms[i % len(doms)] for i in range(n)] if doms else []
        used: set[int] = set()
        for k, d in enumerate(order):
            pool = c[(c["domain"] == d) & (~c["id"].isin(used))]
            row = pool.sample(1, random_state=args.seed + k).iloc[0]
            used.add(int(row["id"]))
            picks.append({"id": int(row["id"]), "lang": lang, "domain": d, "url": row["url"]})

    (args.out / "gold").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(picks).to_csv(args.out / "gold" / "manifest.csv", index=False)
    print(stats.groupby("lang")["n_urls"].sum().to_string())
    print(f"wave 1 (top 20) covers {stats[stats.wave == 1]['share'].sum():.1%} of URLs")
    print(f"picked {len(picks)} gold pages -> {args.out / 'gold' / 'manifest.csv'}")


if __name__ == "__main__":
    main()
