import pandas as pd
import argparse
import asyncio
import logging
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
import trafilatura
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

URL_BLACKLIST = [
    '/rao-vat/', '/san-pham/', '/gio-hang/', '/mua-ban/', '/cart/', 
    '/checkout/', '/shop/', '/store/', 'shopee', 'lazada', 'tiki'
]

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

def check_layer_2(text):
    text_lower = unicodedata.normalize('NFC', text.lower())
    for kw in MEDICAL_KEYWORDS:
        kw_norm = unicodedata.normalize('NFC', kw)
        if kw_norm in text_lower:
            return None
    return "Không chứa bất kỳ từ khóa y khoa nào trong nội dung"

async def fetch_and_extract_stealth(context, url):
    result = {'url': url, 'status': None, 'text_content': None, 'reason': '', 'dropped_at_layer': None}
    
    l1_reason = check_layer_1(url)
    if l1_reason:
        result['status'] = 'dropped'
        result['dropped_at_layer'] = 1
        result['reason'] = l1_reason
        return result

    page = await context.new_page()
    await Stealth().apply_stealth_async(page)
    
    try:
        response = await page.goto(url, timeout=25000, wait_until='domcontentloaded')
        
        if response:
            result['status'] = response.status
            # Wait extra time for Cloudflare JS to solve
            await page.wait_for_timeout(4000)
            
            html = await page.content()
            if "cf-browser-verification" in html or "Just a moment..." in html:
                result['reason'] = "Cloudflare Challenge (Still Blocked by Stealth)"
            else:
                extracted_text = trafilatura.extract(html)
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
        else:
            result['reason'] = "No response object"
            
    except Exception as e:
        result['reason'] = f"Error: {type(e).__name__}"
    finally:
        await page.close()
        
    return result

async def worker(queue, context, results, sem):
    while True:
        try:
            url = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
            
        async with sem:
            res = await fetch_and_extract_stealth(context, url)
            results.append(res)
            logging.info(f"Stealth processed: {url} -> {res['reason']}")

async def main_async(urls, concurrency=5):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720}
        )
        
        queue = asyncio.Queue()
        for url in urls:
            queue.put_nowait(url)
            
        results = []
        sem = asyncio.Semaphore(concurrency)
        
        tasks = [asyncio.create_task(worker(queue, context, results, sem)) for _ in range(concurrency)]
        await asyncio.gather(*tasks)
        await browser.close()
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input parquet file (thường là file kết quả failed của curl_cffi)')
    parser.add_argument('--output', type=str, required=True, help='Output parquet file')
    parser.add_argument('--sample', type=int, default=0, help='Number of URLs to check')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrent browsers (giữ thấp để tránh ngốn RAM)')
    args = parser.parse_args()

    logging.info(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    
    # Lọc ra các URL chưa Success
    if 'reason' in df.columns:
        df_target = df[df['reason'] != 'Success'].copy()
    else:
        df_target = df.copy()

    logging.info(f"Total target URLs for Stealth: {len(df_target)}")

    if args.sample > 0 and len(df_target) > args.sample:
        logging.info(f"Sampling {args.sample} URLs.")
        df_target = df_target.sample(args.sample, random_state=42)

    urls = df_target['url'].tolist()
    logging.info(f"Starting Playwright Stealth check for {len(urls)} URLs...")
    
    import nest_asyncio
    nest_asyncio.apply()
    
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main_async(urls, concurrency=args.concurrency))

    res_df = pd.DataFrame(results)
    
    print("\n--- Results Summary ---")
    print(res_df['reason'].value_counts())

    res_df['status'] = res_df['status'].astype(str)
    merged = pd.merge(df_target.drop(columns=['reason', 'text_content', 'status'], errors='ignore'), res_df, on='url', how='left')
    merged.to_parquet(args.output, index=False)
    logging.info(f"Saved Stealth results to {args.output}")

if __name__ == "__main__":
    main()
