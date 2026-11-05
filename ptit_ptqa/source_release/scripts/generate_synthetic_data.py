import os
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from groq import Groq
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để import được src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import settings từ src.config
from src.config import settings

from queue import Queue
import concurrent.futures

# Khởi tạo Groq Client
if not settings.GROQ_API_KEY:
    print("Vui lòng thêm GROQ_API_KEY vào file .env")
    sys.exit(1)

api_keys = [k.strip() for k in settings.GROQ_API_KEY.split(',') if k.strip()]

# Nếu GROQ_URL có chứa 'openai/v1' thì chứng tỏ user cấu hình nhầm làm đúp URL
base_url = settings.GROQ_URL
if base_url and "api.groq.com" in base_url:
    base_url = None # Xóa base_url để SDK tự gọi đúng chuẩn gốc

client_queue = Queue()
for key in api_keys:
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    client_queue.put(Groq(**kwargs))

print(f"Đã nạp {len(api_keys)} API Keys để chạy song song!")

# Đọc danh sách 4.4 triệu link
df_links = pq.read_table('data/raw/btc_provided/links_corpus.parquet').to_pandas()

# Lấy 500 URL ngẫu nhiên để dự phòng (bù trừ cho các URL bị hỏng hoặc bị lọc bỏ)
df_sample = df_links.sample(n=500, random_state=42)

import requests
from bs4 import BeautifulSoup

def crawl_text_from_url(url):
    try:
        # Giả mạo User-Agent để tránh bị block bởi một số trang web
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Lấy toàn bộ text từ các thẻ p (paragraph)
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Cắt bớt nếu text quá dài (tránh tốn token và tràn context LLM)
            if len(text) > 2000:
                text = text[:2000] + "..."
                
            return text if text else None
    except Exception:
        pass
    return None

def generate_query_with_groq(url, client):
    content = crawl_text_from_url(url)
    
    if not content:
        print(f"[Cào thất bại] Bỏ qua URL do Cloudflare chặn hoặc lỗi 403: {url}")
        return None
        
    context_info = f"URL: {url}\nNỘI DUNG BÀI BÁO (Trích lược):\n{content}"
        
    prompt = f"""Dựa vào tài liệu y khoa sau (có thể là tiếng Việt, tiếng Trung, hoặc ngôn ngữ khác): 
{context_info}

Hãy đóng vai một bệnh nhân bình thường ở Việt Nam. Viết ĐÚNG 1 câu hỏi bằng Tiếng Việt mang phong cách đời thường (không dùng từ ngữ học thuật chuyên sâu) để hỏi về vấn đề mà tài liệu này có thể giải đáp.

NHIỆM VỤ ĐẶC BIỆT (BỘ LỌC):
Nếu nội dung tài liệu HOÀN TOÀN KHÔNG LIÊN QUAN đến y tế, sức khỏe, bệnh tật, dược phẩm (ví dụ: bản tin giá gas, giá vàng, chứng khoán, thể thao, v.v.), bạn KHÔNG ĐƯỢC phép bịa ra câu hỏi y khoa. Thay vào đó, hãy trả về ĐÚNG 1 từ duy nhất: "INVALID".

LƯU Ý QUAN TRỌNG KHI TẠO CÂU HỎI (Nếu tài liệu hợp lệ): 
- Bắt buộc hỏi bằng Tiếng Việt (Bởi vì truy vấn của người dùng trong cuộc thi này luôn là Tiếng Việt, dùng để tìm kiếm tài liệu đa ngôn ngữ).
- Hãy sử dụng NGẪU NHIÊN và ĐA DẠNG các danh xưng khác nhau (ví dụ: "tôi", "em", "cháu", "mình", hoặc hỏi cho người thân như "bé nhà em", "mẹ tôi", "chồng em", hoặc câu nói trống không/ẩn danh xưng). Tuyệt đối KHÔNG chỉ lặp đi lặp lại một danh xưng.
- Độ dài câu hỏi nên tự nhiên, có chủ ngữ vị ngữ rõ ràng, giống như người thật đi hỏi bác sĩ. Tránh những câu cộc lốc (ví dụ: "Uống trà").

MỘT SỐ VÍ DỤ TỪ DỮ LIỆU THỰC TẾ ĐỂ LÀM MẪU VĂN PHONG:
- "Rối loạn cảm xúc lưỡng cực có điều trị dứt điểm được không?"
- "Em hiện đang mang thai 32 tuần. Tuần trước em vừa siêu âm Doppler, tuần này em đi siêu âm 4D có sao không ạ? Siêu âm nhiều có ảnh hưởng đến thai nhi không?"
- "Em thường xuyên bị nhiệt miệng kéo dài, cứ hết lần này lại đến lần khác lại bị, có phải do chức năng gan không?"
- "Làm thế nào để giảm đau khi bị chàm khô trong lòng bàn chân?"

Dựa vào tài liệu ở trên, hãy trả về chữ INVALID hoặc sinh ra duy nhất 1 câu hỏi tương tự:
"""
    import groq
    max_retries = 10 # Tăng số lần retry lên 10 để chống rate limit chung tài khoản
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                reasoning_effort="low",
                stream=False,
                stop=None
            )
            return completion.choices[0].message.content.strip().strip('"')
        except groq.RateLimitError as e:
            print(f"[Rate Limit] Đang đợi 20 giây để hệ thống xả token... (Lần {attempt + 1}/{max_retries})")
            time.sleep(20)
        except Exception as e:
            print(f"Lỗi khi gọi API cho URL {url}: {e}")
            return None
                
    print(f"Bỏ qua URL {url} do vượt quá số lần thử lại (Rate Limit).")
    return None

csv_path = 'data/dev_set/dev_qrels_synthetic_data.csv'
qrels = []
existing_urls = set()

if os.path.exists(csv_path):
    try:
        existing_df = pd.read_csv(csv_path)
        qrels = existing_df.to_dict('records')
        existing_urls = set(existing_df['url'].tolist())
        print(f"Đã load {len(qrels)} câu hỏi từ lần chạy trước. Đang tiếp tục...")
    except pd.errors.EmptyDataError:
        pass

print("Đang gọi Groq API để sinh 50 câu hỏi đa ngôn ngữ... (Vui lòng chờ vài giây)")

def process_row(row):
    if row['url'] in existing_urls:
        return None
        
    client = client_queue.get()
    try:
        query = generate_query_with_groq(row['url'], client)
        if query and query.strip().upper() == "INVALID":
            print(f"[Loại bỏ] URL không chứa nội dung y khoa: {row['url']}")
            return None
            
        if query:
            return {
                'url_id': row['id'],
                'query_text': query,
                'url': row['url']
            }
    finally:
        time.sleep(12) # Rate limit protection per thread (tăng lên 12s để giãn cách)
        client_queue.put(client)
    return None

target_count = 50
with concurrent.futures.ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
    futures = []
    url_iterator = df_sample.iterrows()
    
    # Nạp task liên tục
    def submit_more_tasks():
        while len(futures) < len(api_keys) * 2:
            try:
                idx, row = next(url_iterator)
                futures.append(executor.submit(process_row, row))
            except StopIteration:
                break

    submit_more_tasks()
    
    while futures and len(qrels) < target_count:
        done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
        
        for future in done:
            futures.remove(future)
            res = future.result()
            if res:
                qrels.append(res)
                
                # Re-index lại ID tuần tự chuẩn xác (tránh lỗi race-condition đụng số)
                for i, q in enumerate(qrels):
                    q['query_id'] = f"GROQ_{i+1:03d}"
                    
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                pd.DataFrame(qrels, columns=['query_id', 'url_id', 'query_text', 'url']).to_csv(csv_path, index=False)
                print(f"Đã hoàn thành {len(qrels)}/{target_count}: URL {res['url_id']}")
                
                if len(qrels) >= target_count:
                    break
                    
        # Nạp thêm task mới cho các thread đang rảnh
        if len(qrels) < target_count:
            submit_more_tasks()

    # Cancel các tasks dư thừa (nếu có)
    for future in futures:
        future.cancel()

df_out = pd.DataFrame(qrels, columns=['query_id', 'url_id', 'query_text', 'url'])

if df_out.empty:
    print("\nThất bại! Không có câu hỏi nào được sinh ra.")
else:
    print(f"\nĐã tạo thành công tổng cộng {len(df_out)} câu hỏi bằng LLM!")
    print(df_out[['query_id', 'query_text']].tail(10))
