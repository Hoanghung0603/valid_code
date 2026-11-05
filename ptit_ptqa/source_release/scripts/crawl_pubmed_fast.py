"""High-Speed Batch PubMed Crawler via Europe PMC REST API with Proxy Rotation."""

import argparse
import glob
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def validate_and_load_proxies(proxy_file: str) -> List[str]:
    """Reads proxy file and filters down to responsive proxies within 3s."""
    if not os.path.exists(proxy_file):
        return []

    with open(proxy_file, "r", encoding="utf-8", errors="ignore") as f:
        raw_proxies = [line.strip() for line in f if line.strip()]
    unique_proxies = list(dict.fromkeys(raw_proxies))

    logger.info(f"Checking {len(unique_proxies)} proxies from {proxy_file}...")
    valid_proxies = []

    def check_proxy(p: str) -> Optional[str]:
        try:
            r = requests.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:11197%20SRC:MED&format=json&pageSize=1",
                proxies={"http": p, "https": p},
                timeout=3.0
            )
            if r.status_code == 200 and "resultList" in r.text:
                return p
        except Exception:
            return None
        return None

    with ThreadPoolExecutor(max_workers=min(20, len(unique_proxies))) as ex:
        for p in ex.map(check_proxy, unique_proxies):
            if p:
                valid_proxies.append(p)

    logger.info(f"Retained {len(valid_proxies)} responsive proxies.")
    return valid_proxies


class FastProxySession:
    def __init__(self, proxies: List[str]):
        self.proxies = list(proxies)
        self._lock = threading.Lock()
        self._index = 0

    def get_proxy(self) -> Optional[str]:
        with self._lock:
            if not self.proxies:
                return None
            p = self.proxies[self._index % len(self.proxies)]
            self._index += 1
            return p

    def mark_bad_proxy(self, proxy: str):
        with self._lock:
            if proxy in self.proxies:
                self.proxies.remove(proxy)
                logger.warning(f"Removed degraded proxy: {proxy} (Remaining: {len(self.proxies)})")


def fetch_batch_europe_pmc(
    pmid_chunk: List[Tuple[int, str, str]],  # [(row_id, url, pmid), ...]
    proxy_session: FastProxySession,
    timeout: int = 10,
    max_retries: int = 2
) -> List[Dict]:
    """
    Fetches a batch of up to 100 PMIDs in a single HTTP request from Europe PMC API.
    Fast failover: if proxy does not respond within 3.5s, immediately fall back to direct.
    """
    pmid_map = {pmid: (row_id, url) for row_id, url, pmid in pmid_chunk}
    pmids = list(pmid_map.keys())
    query_str = " OR ".join(pmids)
    url = f"{EUROPE_PMC_BASE}?query=EXT_ID:({query_str})%20SRC:MED&format=json&resultType=core&pageSize={len(pmids)}"

    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    success = False
    proxy = proxy_session.get_proxy()

    # 1. Try proxy first with a short timeout
    if proxy:
        proxies = {"http": proxy, "https": proxy}
        try:
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("resultList", {}).get("result", [])
                for item in items:
                    ret_pmid = str(item.get("id", ""))
                    if ret_pmid in pmid_map:
                        row_id, orig_url = pmid_map[ret_pmid]
                        title = str(item.get("title", "")).strip()
                        abstract = str(item.get("abstractText", "")).strip()
                        full_text = f"{title}\n\n{abstract}".strip() if abstract else title.strip()
                        if full_text and len(full_text) >= 30:
                            results.append({"id": int(row_id), "url": orig_url, "text": full_text})
                success = True
            elif resp.status_code == 429:
                time.sleep(1.0)
            else:
                proxy_session.mark_bad_proxy(proxy)
        except Exception:
            proxy_session.mark_bad_proxy(proxy)

    # 2. Direct fallback (Fast & Reliable)
    if not success:
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("resultList", {}).get("result", [])
                    for item in items:
                        ret_pmid = str(item.get("id", ""))
                        if ret_pmid in pmid_map:
                            row_id, orig_url = pmid_map[ret_pmid]
                            title = str(item.get("title", "")).strip()
                            abstract = str(item.get("abstractText", "")).strip()
                            full_text = f"{title}\n\n{abstract}".strip() if abstract else title.strip()
                            if full_text and len(full_text) >= 30:
                                results.append({"id": int(row_id), "url": orig_url, "text": full_text})
                    break
                elif resp.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
            except Exception:
                time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser(description="High-Speed Batch PubMed Crawler via Europe PMC")
    parser.add_argument("--input", default="data/remaining_urls.parquet", help="Path to input remaining urls parquet")
    parser.add_argument("--output-dir", default="data/raw/corpus_text/pubmed_recovered", help="Output directory")
    parser.add_argument("--proxy-file", default="data/raw/corpus_text/good_proxies.txt", help="Path to proxy list")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of PMIDs per HTTP request (default: 100)")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent request workers (default: 3)")
    parser.add_argument("--save-every", type=int, default=10000, help="Save parquet every N articles (default: 10000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Proxies with quick verification
    proxies = validate_and_load_proxies(args.proxy_file)
    proxy_session = FastProxySession(proxies)

    # 2. Check already crawled IDs to resume seamlessly
    existing_ids: Set[int] = set()
    for p_file in glob.glob(os.path.join(args.output_dir, "part_*.parquet")):
        try:
            df_part = pd.read_parquet(p_file, columns=["id"])
            existing_ids.update(df_part["id"].astype(int))
        except Exception:
            pass
    logger.info(f"Already crawled PubMed articles: {len(existing_ids):,}")

    # 3. Read input and filter PubMed URLs
    logger.info(f"Loading {args.input}...")
    df_all = pd.read_parquet(args.input)
    df_pubmed = df_all[df_all["url"].str.contains("pubmed", case=False, na=False)]
    logger.info(f"Total PubMed URLs in input: {len(df_pubmed):,}")

    # Filter uncrawled
    df_pending = df_pubmed[~df_pubmed["id"].isin(existing_ids)]
    logger.info(f"Pending PubMed URLs to crawl: {len(df_pending):,}")

    if len(df_pending) == 0:
        logger.info("All PubMed URLs have already been retrieved!")
        return

    # 4. Extract PMIDs
    tasks: List[Tuple[int, str, str]] = []
    for row in df_pending.itertuples(index=False):
        url_clean = str(row.url).strip()
        m = re.search(r"/(\d+)/?", url_clean)
        if m:
            pmid = m.group(1)
            tasks.append((int(row.id), url_clean, pmid))

    logger.info(f"Valid numeric PMIDs extracted: {len(tasks):,}")

    # Chunk into batches of size args.batch_size (e.g. 100)
    batches = [tasks[i:i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
    logger.info(f"Divided into {len(batches):,} API batch requests (batch_size={args.batch_size})")

    # 5. Process batches concurrently
    collected_articles: List[Dict] = []
    total_saved = len(existing_ids)
    part_idx = len(glob.glob(os.path.join(args.output_dir, "part_*.parquet")))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_batch_europe_pmc, batch, proxy_session): batch
            for batch in batches
        }

        with tqdm(total=len(batches), desc="Crawling PubMed", unit="batch") as pbar:
            for fut in as_completed(futures):
                res_items = fut.result()
                if res_items:
                    collected_articles.extend(res_items)

                pbar.update(1)

                # Save when reaching threshold
                if len(collected_articles) >= args.save_every:
                    part_file = os.path.join(args.output_dir, f"part_{part_idx:04d}.parquet")
                    df_save = pd.DataFrame(collected_articles)
                    df_save.to_parquet(part_file, index=False)
                    total_saved += len(df_save)
                    logger.info(f"Saved {len(df_save):,} articles to {part_file} (Total: {total_saved:,})")
                    part_idx += 1
                    collected_articles.clear()

    # Save remaining
    if collected_articles:
        part_file = os.path.join(args.output_dir, f"part_{part_idx:04d}.parquet")
        df_save = pd.DataFrame(collected_articles)
        df_save.to_parquet(part_file, index=False)
        total_saved += len(df_save)
        logger.info(f"Saved {len(df_save):,} articles to {part_file} (Total: {total_saved:,})")
        collected_articles.clear()

    elapsed = time.time() - t0
    logger.info(f"🎉 Hoàn thành cào PubMed trong {elapsed:.1f}s. Tổng bài viết thu thập: {total_saved:,}")


if __name__ == "__main__":
    main()
