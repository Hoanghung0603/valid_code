import os
import glob
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from chunker import Chunker

def process_file(parquet_file, output_dir, max_tokens, token_overlap):
    filename = os.path.basename(parquet_file)
    out_file = os.path.join(output_dir, filename.replace('corpus_', 'chunked_'))
    
    if os.path.exists(out_file):
        print(f"Bỏ qua {filename}, đã chunk xong trước đó.")
        return
        
    try:
        df = pd.read_parquet(parquet_file)
        chunker = Chunker(strategy="contextual", max_tokens=max_tokens, token_overlap=token_overlap)
        
        all_chunks = []
        # Chạy chunker cho từng bài báo trong file
        for _, row in df.iterrows():
            doc_id = str(row['id'])
            url = row['url']
            text = row['text']
            
            if not text or not isinstance(text, str): 
                continue
            
            chunks = chunker.chunk_document(doc_id=doc_id, text=text, url=url)
            # Gắn thêm chunk_index thực sự vào để phục vụ Parent-Child Retrieval sau này
            for idx, c in enumerate(chunks):
                c['chunk_index'] = idx
                
            all_chunks.extend(chunks)
            
        if all_chunks:
            df_chunks = pd.DataFrame(all_chunks)
            df_chunks.to_parquet(out_file, engine='pyarrow')
            print(f"[THÀNH CÔNG] {filename}: {len(df)} bài báo -> {len(df_chunks)} chunks.")
        else:
            print(f"[CẢNH BÁO] {filename} không sinh ra chunk nào (file rỗng hoặc toàn text rác).")
            # Tạo file rỗng để đánh dấu đã xử lý
            pd.DataFrame(columns=['chunk_id', 'doc_id', 'url', 'content', 'word_count', 'chunk_index']).to_parquet(out_file, engine='pyarrow')
            
    except Exception as e:
        print(f"[LỖI] Xử lý file {filename} thất bại: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch Semantic Chunker đa luồng (Quy mô hàng triệu chunks)")
    parser.add_argument('--input-dir', type=str, default='data/raw/corpus_text', help='Thư mục chứa parquet thô')
    parser.add_argument('--output-dir', type=str, default='data/processed/chunked_corpus', help='Thư mục lưu parquet đã chunk')
    # Mặc định cắt siêu nhỏ (256 tokens) để phục vụ "Small-to-Big Retrieval"
    parser.add_argument('--max-tokens', type=int, default=256, help='Giới hạn token mỗi chunk')
    parser.add_argument('--overlap', type=int, default=50, help='Độ trễ overlap giữa 2 chunk liên tiếp')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Số luồng CPU chạy song song')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(args.input_dir, 'corpus_part_*.parquet'))
    print(f"Tìm thấy {len(files)} file parquet. Bắt đầu băm nhỏ với {args.workers} luồng CPU...")
    
    # Chạy đa luồng cấp độ File (File-level Multiprocessing)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for f in files:
            executor.submit(process_file, f, args.output_dir, args.max_tokens, args.overlap)
            
    print("\nHOÀN TẤT QUÁ TRÌNH CHUNKING!")

if __name__ == '__main__':
    main()
