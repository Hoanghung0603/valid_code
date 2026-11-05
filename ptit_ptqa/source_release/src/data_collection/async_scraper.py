import asyncio
import aiohttp
import trafilatura
import pandas as pd
import pyarrow.parquet as pq
import os
import argparse
import logging
import sys
import multiprocessing

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper_debug.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Tắt cảnh báo rác từ thư viện trafilatura
logging.getLogger('trafilatura').setLevel(logging.ERROR)

# Cấu hình Scraper
CONCURRENCY_LIMIT = 50 
TIMEOUT_SECONDS = 15

# Tầng 1: Heuristic URL Blacklist (Ưu tiên Recall: chỉ cấm những từ thật sự rác)
URL_BLACKLIST = [
    '/rao-vat/', '/san-pham/', '/gio-hang/', '/mua-ban/', '/cart/', 
    '/checkout/', '/shop/', '/store/', 'shopee', 'lazada', 'tiki'
]

# Tầng 2: Medical Keywords (Ưu tiên Recall: Bộ từ khóa rộng, chỉ cần xuất hiện 1-2 từ là pass)
MEDICAL_KEYWORDS = [
    # Tiếng Việt
    'bác sĩ', 'bệnh', 'triệu chứng', 'điều trị', 'thuốc', 'sức khỏe', 
    'ung thư', 'viêm', 'đau', 'chẩn đoán', 'huyết áp', 'phẫu thuật', 
    'khám', 'y tế', 'bệnh nhân', 'phòng khám', 'dược',
    # Tiếng Trung
    '医院', '医生', '治疗', '疾病', '症状', '药', '患者', '健康', '诊断', '手术',
    # Tiếng Anh
    'doctor', 'disease', 'symptom', 'treatment', 'medicine', 'health', 
    'cancer', 'pain', 'diagnosis', 'blood pressure', 'surgery', 'clinic', 
    'medical', 'patient', 'hospital', 'pharmacy', 'syndrome', 'infection', 'therapy'
]

def check_layer_1(url):
    """Lọc Tầng 1: Trả về lý do nếu bị lọc, ngược lại trả về None"""
    url_lower = url.lower()
    for black_word in URL_BLACKLIST:
        if black_word in url_lower:
            return f"Chứa từ khóa rác trong URL: {black_word}"
    return None

import unicodedata

def check_layer_2(text):
    """Lọc Tầng 2: Trả về lý do nếu bị lọc, ngược lại trả về None"""
    # Chuẩn hóa Unicode (NFC) để tránh lỗi font chữ Việt Nam (Tổ hợp vs Dựng sẵn)
    text_lower = unicodedata.normalize('NFC', text.lower())
    
    # Để ưu tiên Recall, chỉ cần tìm thấy ÍT NHẤT 1 từ khóa y khoa là cho pass
    for kw in MEDICAL_KEYWORDS:
        kw_norm = unicodedata.normalize('NFC', kw)
        if kw_norm in text_lower:
            return None
    return "Không chứa bất kỳ từ khóa y khoa nào trong nội dung"

async def fetch_and_extract(session, url, row_id, sem):
    """Cào và trích xuất nội dung từ 1 URL, đi qua 2 tầng lọc"""
    async with sem:
        # TẦNG 1: LỌC URL
        l1_reason = check_layer_1(url)
        if l1_reason:
            return {'id': row_id, 'url': url, 'text': None, 'status': 'dropped', 'layer': 1, 'reason': l1_reason}

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            async with session.get(url, headers=headers, timeout=TIMEOUT_SECONDS) as response:
                if response.status == 200:
                    html = await response.read()
                    text = trafilatura.extract(html, include_links=False, include_images=False)
                    
                    if not text:
                        return {'id': row_id, 'url': url, 'text': None, 'status': 'dropped', 'layer': 2, 'reason': 'Trafilatura không trích xuất được text (Có thể là trang trắng hoặc chặn bot)'}
                    
                    # TẦNG 2: LỌC NỘI DUNG (KEYWORD DENSITY)
                    l2_reason = check_layer_2(text)
                    if l2_reason:
                        return {'id': row_id, 'url': url, 'text': None, 'status': 'dropped', 'layer': 2, 'reason': l2_reason}
                        
                    # PASS ALL
                    return {'id': row_id, 'url': url, 'text': text, 'status': 'success', 'layer': None, 'reason': None}
                else:
                    return {'id': row_id, 'url': url, 'text': None, 'status': 'error', 'layer': 'network', 'reason': f'HTTP_{response.status}'}
        except asyncio.TimeoutError:
            return {'id': row_id, 'url': url, 'text': None, 'status': 'error', 'layer': 'network', 'reason': 'Timeout'}
        except Exception as e:
            return {'id': row_id, 'url': url, 'text': None, 'status': 'error', 'layer': 'network', 'reason': f'Error: {str(e)[:50]}'}

async def process_batch(df_batch, batch_index, output_dir):
    """Xử lý một lô (batch) URLs và lưu ra file"""
    out_file = os.path.join(output_dir, f"corpus_part_{batch_index:05d}.parquet")
    
    if os.path.exists(out_file):
        logger.info(f"Batch {batch_index} đã hoàn thành trước đó. Bỏ qua.")
        return
        
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    results = []
    
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _, row in df_batch.iterrows():
            tasks.append(fetch_and_extract(session, row['url'], row['id'], sem))
            
        for future in asyncio.as_completed(tasks):
            res = await future
            results.append(res)
            
            if len(results) % 500 == 0:
                logger.info(f"Batch {batch_index}: Đã xong {len(results)}/{len(df_batch)}")
                
    # Tách dữ liệu Thành công và Bị loại bỏ
    success_records = [{'id': r['id'], 'url': r['url'], 'text': r['text']} for r in results if r['status'] == 'success']
    dropped_records = [{'id': r['id'], 'url': r['url'], 'reason': r['reason'], 'dropped_at_layer': r['layer']} for r in results if r['status'] != 'success']
    
    # 1. Lưu các file Thành công vào Parquet
    if success_records:
        df_success = pd.DataFrame(success_records)
        df_success.to_parquet(out_file, engine='pyarrow')
        logger.info(f"Đã lưu Batch {batch_index}: {len(df_success)} URLs thành công vào {out_file}")
    else:
        logger.warning(f"Batch {batch_index} không có URL nào thành công!")
        # Vẫn tạo file parquet rỗng để đánh dấu là batch này đã chạy
        pd.DataFrame(columns=['id', 'url', 'text']).to_parquet(out_file, engine='pyarrow')

    # 2. Ghi nối (Append) các file bị loại vào Parquet riêng rẽ để tránh đụng độ tiến trình
    if dropped_records:
        df_dropped = pd.DataFrame(dropped_records)
        if 'dropped_at_layer' in df_dropped.columns:
            df_dropped['dropped_at_layer'] = df_dropped['dropped_at_layer'].astype(str)
        dropped_out_file = os.path.join(output_dir, f"dropped_urls_part_{batch_index:05d}.parquet")
        df_dropped.to_parquet(dropped_out_file, engine='pyarrow')
        logger.info(f"Batch {batch_index}: Đã ghi {len(df_dropped)} URLs bị loại vào {dropped_out_file}")

def run_batch_sync(args):
    """Hàm bọc đồng bộ để chạy asyncio.run trong từng process riêng biệt"""
    df_batch, batch_index, output_dir = args
    asyncio.run(process_batch(df_batch, batch_index, output_dir))

def main(input_file, output_dir, batch_size=10000, start_batch=0, max_batches=None, num_workers=None, specific_batches=None):
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Đang đọc file gốc: {input_file}")
    df = pd.read_parquet(input_file)
    total_records = len(df)
    total_batches = (total_records // batch_size) + (1 if total_records % batch_size > 0 else 0)
    
    logger.info(f"Tổng số URL: {total_records}. Chia thành {total_batches} batches (kích thước {batch_size})")
    
    tasks = []
    if specific_batches is not None:
        batch_list = specific_batches
    else:
        end_batch = total_batches
        if max_batches:
            end_batch = min(start_batch + max_batches, total_batches)
        batch_list = range(start_batch, end_batch)
        
    for i in batch_list:
        if i >= total_batches:
            continue
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_records)
        df_batch = df.iloc[start_idx:end_idx]
        tasks.append((df_batch, i, output_dir))
        
    num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Bắt đầu cào dữ liệu đa tiến trình với {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_batch_sync, tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async Web Scraper cho 4.4 Triệu URLs có tích hợp Lọc Phễu")
    parser.add_argument("--input", type=str, default="data/raw/btc_provided/links_corpus.parquet", help="File links gốc")
    parser.add_argument("--output", type=str, default="data/raw/corpus_text", help="Thư mục đầu ra")
    parser.add_argument("--batch-size", type=int, default=10000, help="Số lượng URL mỗi batch")
    parser.add_argument("--start-batch", type=int, default=0, help="Batch bắt đầu")
    parser.add_argument("--max-batches", type=int, default=None, help="Số lượng batch tối đa sẽ chạy")
    parser.add_argument("--workers", type=int, default=None, help="Số lượng tiến trình song song (mặc định: CPU-1)")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.batch_size, args.start_batch, args.max_batches, args.workers)
