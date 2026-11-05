import pandas as pd
import argparse
import asyncio
import logging
import random
from curl_cffi.requests import AsyncSession
import trafilatura
import urllib.parse
import os
import multiprocessing
import concurrent.futures
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(message)s')

# Tầng 1: Heuristic URL Blacklist (Ưu tiên Recall: chỉ cấm những từ thật sự rác)
URL_BLACKLIST = [
    '/rao-vat/', '/san-pham/', '/gio-hang/', '/mua-ban/', '/cart/', 
    '/checkout/', '/shop/', '/store/', 'shopee', 'lazada', 'tiki'
]

# Tầng 2: Medical Keywords (Ưu tiên Recall: Bộ từ khóa rộng, chỉ cần xuất hiện 1-2 từ là pass)
MEDICAL_KEYWORDS = [
    'bác sĩ', 'bệnh', 'triệu chứng', 'điều trị', 'thuốc', 'sức khỏe', 
    'ung thư', 'viêm', 'đau', 'chẩn đoán', 'huyết áp', 'phẫu thuật', 
    'khám', 'y tế', 'bệnh nhân', 'phòng khám', 'dược',
    '医院', '医生', '治疗', '疾病', '症状', '药', '患者', '健康', '诊断', '手术',
    'doctor', 'disease', 'symptom', 'treatment', 'medicine', 'health', 
    'cancer', 'pain', 'diagnosis', 'blood pressure', 'surgery', 'clinic', 
    'medical', 'patient', 'hospital', 'pharmacy', 'syndrome', 'infection', 'therapy'
]

def check_layer_1(url):
    url_lower = url.lower()
    for black_word in URL_BLACKLIST:
        if black_word in url_lower:
            return f"Chứa từ khóa rác trong URL: {black_word}"
    return None

import unicodedata

def check_layer_2(text):
    text_lower = unicodedata.normalize('NFC', text.lower())
    for kw in MEDICAL_KEYWORDS:
        kw_norm = unicodedata.normalize('NFC', kw)
        if kw_norm in text_lower:
            return None
    return "Không chứa bất kỳ từ khóa y khoa nào trong nội dung"

async def fetch_and_extract(url, session, proxy=None):
    result = {'url': url, 'status': None, 'text_content': None, 'reason': '', 'dropped_at_layer': None}
    
    l1_reason = check_layer_1(url)
    if l1_reason:
        result['status'] = 'dropped'
        result['dropped_at_layer'] = 1
        result['reason'] = l1_reason
        return result

    try:
        impersonate_targets = ["chrome110", "chrome120", "edge101", "safari15_3"]
        impersonate = random.choice(impersonate_targets)
        
        proxies_dict = {"http": proxy, "https": proxy} if proxy else None
        
        response = await session.get(url, impersonate=impersonate, proxies=proxies_dict, timeout=15)
        result['status'] = response.status_code
        
        if response.status_code == 200:
            html = response.text
            if "cf-browser-verification" in html or "Just a moment..." in html:
                result['reason'] = "Cloudflare Challenge (JS required)"
            else:
                # Chạy trafilatura (CPU-bound) trong threadpool để không block event loop
                loop = asyncio.get_event_loop()
                extracted_text = await loop.run_in_executor(None, trafilatura.extract, html)
                
                if extracted_text and len(extracted_text.strip()) > 200:
                    l2_reason = check_layer_2(extracted_text)
                    if l2_reason:
                        result['status'] = 'dropped'
                        result['dropped_at_layer'] = 2
                        result['reason'] = l2_reason
                    else:
                        result['text_content'] = extracted_text
                        result['status'] = 'success'
                        result['reason'] = "Success"
                else:
                    result['reason'] = "Empty or insufficient content"
        elif response.status_code in (403, 503):
            result['reason'] = f"HTTP_{response.status_code} (Still blocked)"
        else:
            result['reason'] = f"HTTP_{response.status_code}"
            
    except Exception as e:
        result['reason'] = f"Error: {type(e).__name__}"
        
    return result

async def worker_async(queue, session, results, proxies):
    while True:
        try:
            url = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
            
        proxy = random.choice(proxies) if proxies else None
        res = await fetch_and_extract(url, session, proxy)
        results.append(res)
        
        if len(results) % 500 == 0:
            logging.info(f"Processed {len(results)} URLs... Last status: {res['reason']}")

async def main_async(urls, concurrency=50, proxies=None):
    queue = asyncio.Queue()
    for url in urls:
        queue.put_nowait(url)
        
    results = []
    async with AsyncSession(verify=False) as session:
        tasks = [asyncio.create_task(worker_async(queue, session, results, proxies)) for _ in range(concurrency)]
        await asyncio.gather(*tasks)
        
    return results

def process_chunk(worker_id, chunk_urls, concurrency, proxies, tmp_dir):
    """Hàm chạy trên một Process độc lập"""
    import nest_asyncio
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    logging.info(f"Worker {worker_id} starting with {len(chunk_urls)} URLs.")
    results = loop.run_until_complete(main_async(chunk_urls, concurrency, proxies))
    
    res_df = pd.DataFrame(results)
    res_df['status'] = res_df['status'].astype(str)
    
    out_path = os.path.join(tmp_dir, f"part_{worker_id}.parquet")
    res_df.to_parquet(out_path, index=False)
    logging.info(f"Worker {worker_id} finished. Saved to {out_path}")
    return out_path

def load_proxies(proxy_file):
    if not proxy_file or not os.path.exists(proxy_file):
        return []
    with open(proxy_file, "r") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/corpus_text/dropped_urls_merged/dropped_network.parquet')
    parser.add_argument('--output', type=str, default='data/raw/corpus_text/dropped_urls_merged/recovered_network_cffi.parquet')
    parser.add_argument('--sample', type=int, default=0, help='Number of URLs to check (0 for all)')
    parser.add_argument('--concurrency', type=int, default=50, help='Async requests per worker')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='Number of processes')
    parser.add_argument('--proxy-file', type=str, default='', help='Đường dẫn file chứa list proxies')
    parser.add_argument('--filter-reason', type=str, default='', help='Filter reason')
    args = parser.parse_args()

    logging.info(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    
    if args.filter_reason:
        df = df[df['reason'].astype(str).str.contains(args.filter_reason, case=False, na=False)]
        logging.info(f"Filtered URLs by reason '{args.filter_reason}': {len(df)}")
    else:
        logging.info(f"Total dropped network URLs: {len(df)}")

    if args.sample > 0 and len(df) > args.sample:
        logging.info(f"Sampling {args.sample} URLs for the test run.")
        df = df.sample(args.sample, random_state=42)

    urls = df['url'].tolist()
    proxies = load_proxies(args.proxy_file)
    
    tmp_dir = "data/raw/corpus_text/dropped_urls_merged/tmp_parts"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Chia chunks
    num_workers = min(args.workers, len(urls))
    chunks = np.array_split(urls, num_workers)
    
    logging.info(f"Starting Multiprocessing with {num_workers} workers, {args.concurrency} concurrency/worker...")
    
    part_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            if len(chunk) > 0:
                futures.append(executor.submit(process_chunk, i, chunk.tolist(), args.concurrency, proxies, tmp_dir))
                
        for future in concurrent.futures.as_completed(futures):
            part_files.append(future.result())
            
    # Gộp tất cả các part lại
    logging.info("Merging all parts...")
    res_dfs = [pd.read_parquet(f) for f in part_files]
    final_res_df = pd.concat(res_dfs, ignore_index=True)
    
    print("\n--- Results Summary ---")
    print(final_res_df['reason'].value_counts())

    drop_cols = ['reason', 'text_content', 'status']
    merged = pd.merge(df.drop(columns=drop_cols, errors='ignore'), final_res_df, on='url', how='left')
    merged.to_parquet(args.output, index=False)
    logging.info(f"Saved detailed results to {args.output}")

if __name__ == "__main__":
    # Windows requires spawn for multiprocessing
    multiprocessing.freeze_support()
    main()
