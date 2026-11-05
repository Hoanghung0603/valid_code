import pandas as pd
import argparse
import asyncio
import logging
import os
import sys
import glob
import uuid
from playwright.async_api import async_playwright
import trafilatura

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def process_url(context, url):
    page = await context.new_page()
    result = {'url': url, 'status': None, 'text_content': None, 'reason': ''}
    try:
        # Navigate to the URL
        response = await page.goto(url, timeout=20000, wait_until='domcontentloaded')
        
        if response:
            result['status'] = response.status
            
            # If rate limited or standard HTTP error (except 403 which might be bot block)
            if response.status >= 400 and response.status not in (403, 429):
                result['reason'] = f"HTTP Error {response.status}"
            else:
                # Wait a little bit for JS rendering just in case
                try:
                    await page.wait_for_timeout(3000)
                except:
                    pass
                
                # Get fully rendered HTML
                html_content = await page.content()
                
                # Use Trafilatura to extract text from the rendered HTML
                extracted_text = trafilatura.extract(html_content)
                
                if extracted_text and len(extracted_text.strip()) > 500:
                    # Check for soft 404
                    lower_text = extracted_text.lower()
                    if any(kw in lower_text for kw in ["không tìm thấy", "404 error", "bài viết không tồn tại"]):
                        result['reason'] = "Soft 404"
                    else:
                        result['text_content'] = extracted_text
                        result['reason'] = "Success"
                else:
                    # Trafilatura couldn't find text even in rendered HTML, might be a real empty page or soft 404
                    title = await page.title()
                    if title and ("404" in title or "không tìm thấy" in title.lower()):
                        result['reason'] = "Soft 404 (from title)"
                    else:
                        result['reason'] = "Empty content after rendering"
        else:
            result['reason'] = "No response"
            
    except Exception as e:
        result['reason'] = f"Error: {type(e).__name__}"
    finally:
        await page.close()
        
    return result

async def worker(queue, worker_id, checkpoint_dir, recycle_every=500):
    """
    Worker sẽ xử lý tối đa `recycle_every` URL rồi khởi động lại Playwright Browser 
    để tránh memory leak (Target crashed) khi mở quá nhiều tab.
    Mỗi chu kỳ sẽ lưu 1 file parquet vào checkpoint_dir.
    """
    while not queue.empty():
        async with async_playwright() as p:
            # Khởi tạo browser
            browser = await p.chromium.launch(headless=True, channel='chrome')
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1280, 'height': 720}
            )
            
            local_results = []
            count = 0
            
            while not queue.empty() and count < recycle_every:
                try:
                    url = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                
                res = await process_url(context, url)
                local_results.append(res)
                logging.info(f"[W{worker_id}] Processed: {url} -> {res['reason']}")
                count += 1
                queue.task_done()
                
            await browser.close()
            
            # Lưu checkpoint thành từng chunk riêng để không tranh chấp file
            if local_results:
                part_path = os.path.join(checkpoint_dir, f"part_traf_{worker_id}_{uuid.uuid4().hex[:6]}.parquet")
                pd.DataFrame(local_results).to_parquet(part_path, index=False)
                logging.info(f"[W{worker_id}] Saved checkpoint with {len(local_results)} URLs to {part_path}")

async def main_async(urls, concurrency, checkpoint_dir):
    queue = asyncio.Queue()
    for url in urls:
        queue.put_nowait(url)
        
    tasks = [asyncio.create_task(worker(queue, i, checkpoint_dir, recycle_every=500)) for i in range(concurrency)]
    await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/corpus_text/dropped_urls_merged/dropped_filtered.parquet')
    parser.add_argument('--checkpoint_dir', type=str, default='data/raw/corpus_text/recovered_parts/trafilatura_parts')
    parser.add_argument('--output', type=str, default='data/raw/corpus_text/recovered_parts/recovered_trafilatura_32k.parquet')
    parser.add_argument('--sample', type=int, default=0, help='Number of URLs to check (0 for all)')
    parser.add_argument('--concurrency', type=int, default=8, help='Number of concurrent Playwright workers')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    logging.info(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)

    mask = df['reason'].str.contains('Trafilatura', na=False, case=False)
    target_df = df[mask].copy()

    logging.info(f"Total URLs dropped by Trafilatura: {len(target_df)}")

    if args.sample > 0 and len(target_df) > args.sample:
        logging.info(f"Sampling {args.sample} URLs for the test run.")
        target_df = target_df.sample(args.sample, random_state=123)

    all_urls = set(target_df['url'].tolist())
    
    # Đọc lại các URL đã xử lý từ checkpoint
    processed_urls = set()
    part_files = glob.glob(os.path.join(checkpoint_dir, "*.parquet"))
    for pf in part_files:
        try:
            pdf = pd.read_parquet(pf)
            if 'url' in pdf.columns:
                processed_urls.update(pdf['url'].tolist())
        except Exception as e:
            logging.warning(f"Could not read {pf}: {e}")
            
    urls_to_process = list(all_urls - processed_urls)
    
    logging.info(f"Already processed: {len(processed_urls)} URLs. Remaining: {len(urls_to_process)} URLs.")
    
    if len(urls_to_process) > 0:
        logging.info(f"Starting Playwright + Trafilatura async check for {len(urls_to_process)} URLs with concurrency {args.concurrency}...")
        
        import nest_asyncio
        nest_asyncio.apply()
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_async(urls_to_process, args.concurrency, checkpoint_dir))
    else:
        logging.info("No more URLs to process.")

    # Gom kết quả từ tất cả file parquet
    logging.info("Merging checkpoint parts into final result...")
    part_files = glob.glob(os.path.join(checkpoint_dir, "*.parquet"))
    if not part_files:
        logging.info("No checkpoint files found.")
        return
        
    res_dfs = [pd.read_parquet(pf) for pf in part_files]
    res_df = pd.concat(res_dfs, ignore_index=True)
    
    # Loại bỏ trùng lặp nếu có (giữ kết quả mới nhất)
    res_df = res_df.drop_duplicates(subset=['url'], keep='last')
    
    print("\n--- Results Summary ---")
    print(res_df['reason'].value_counts())
    
    print("\n--- Sample Extracted Text ---")
    success_cases = res_df[res_df['reason'] == 'Success']
    for _, row in success_cases.head(3).iterrows():
        text_snippet = (row['text_content'] or '')[:300].replace('\n', ' ')
        try:
            print(f"Content Snippet: {text_snippet}...")
        except Exception:
            print(f"Content Snippet: {text_snippet.encode('ascii', errors='replace').decode()}...")

    # Join results
    merged = pd.merge(target_df.drop(columns=['reason']), res_df, on='url', how='left')
    
    # Reorder columns
    cols = ['id', 'url', 'reason', 'status', 'text_content', 'dropped_at_layer']
    cols = [c for c in cols if c in merged.columns] + [c for c in merged.columns if c not in cols]
    merged = merged[cols]
    
    merged.to_parquet(args.output, index=False)
    logging.info(f"Saved detailed results to {args.output}")

if __name__ == "__main__":
    main()
