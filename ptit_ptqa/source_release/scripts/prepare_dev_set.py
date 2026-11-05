import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys

# Thêm thư mục gốc vào sys.path để import được src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.chunker import SemanticChunker

def crawl_text_from_url(url):
    import time
    for attempt in range(3):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                if text:
                    return text
        except Exception:
            pass
        time.sleep(2)
    return None

def main():
    csv_path = 'data/dev_set/dev_qrels_synthetic_data.csv'
    if not os.path.exists(csv_path):
        print(f"File {csv_path} không tồn tại!")
        return
        
    df = pd.read_csv(csv_path)
    chunker = SemanticChunker(max_tokens=1024)
    
    queries = []
    ground_truth = []
    
    print("Đang tạo Ground Truth và Queries JSON từ CSV...")
    
    for idx, row in df.iterrows():
        query_id = str(row['query_id']).replace('GROQ_', '')
        try:
            # BTC yêu cầu id là integer
            query_id_int = int(query_id)
        except ValueError:
            query_id_int = idx + 1
            
        doc_id = str(row['url_id'])
        url = row['url']
        query_text = row['query_text']
        
        # 1. Cố gắng lấy text trước
        full_text = crawl_text_from_url(url)
        if not full_text:
            print(f"Bỏ qua Query {query_id_int} vì không thể truy cập URL: {url}")
            continue
            
        # 2. Tạo file queries
        queries.append({
            "id": query_id_int,
            "query": query_text
        })
        
        # Chia text thành các chunk
        chunks = chunker.chunk_document(doc_id=doc_id, text=full_text, url=url)
        
        # Vì câu hỏi được sinh ra từ 2000 ký tự đầu tiên của bài báo, 
        # nên chúng ta giả định chunk đầu tiên (hoặc các chunk bao phủ 2000 ký tự đầu) là Ground Truth.
        relevant_chunks = []
        char_count = 0
        for chunk in chunks:
            chunk_content = chunk['content']
            relevant_chunks.append({
                "doc_id": doc_id,
                "chunk_text": chunk_content
            })
            char_count += len(chunk_content)
            if char_count >= 2000:
                break
                
        # Nếu không có chunk nào thì gán mặc định
        if not relevant_chunks:
            relevant_chunks = [{
                "doc_id": doc_id,
                "chunk_text": full_text[:2000]
            }]
            
        ground_truth.append({
            "id": query_id_int,
            "relevant_docs": [doc_id],
            "relevant_chunks": relevant_chunks
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Đã xử lý {idx + 1}/{len(df)}...")
            
    # Lưu file
    os.makedirs('data/dev_set', exist_ok=True)
    
    with open('data/dev_set/dev_queries.json', 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
        
    with open('data/dev_set/dev_ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
        
    print("\nĐã tạo thành công:")
    print("1. data/dev_set/dev_queries.json (Đầu vào cho RAG)")
    print("2. data/dev_set/dev_ground_truth.json (Đáp án để Scorer chấm điểm)")

if __name__ == "__main__":
    main()
