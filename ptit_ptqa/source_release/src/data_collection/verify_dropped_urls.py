import os
import pandas as pd
import argparse
import random
import json
import re
import time
from dotenv import load_dotenv
from groq import Groq

class GroqClientRotator:
    def __init__(self, keys_string):
        self.keys = [k.strip() for k in keys_string.split(',') if k.strip()]
        self.current_idx = 0
        
    def get_client(self):
        if not self.keys:
            raise ValueError("Không có API key nào được cấu hình!")
            
        key = self.keys[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        
        return Groq(api_key=key)

def verify_dropped_urls(file_path, sample_size=100, model_name="llama-3.1-70b-versatile"):
    """
    Lấy ngẫu nhiên sample_size URLs từ file dropped_filtered.parquet và gửi cho LLM (Groq) kiểm tra.
    Tạo output Parquet chứa đánh giá của LLM.
    """
    keys_str = os.environ.get("GROQ_API_KEY")
    if not keys_str:
        print("LỖI: Không tìm thấy GROQ_API_KEY trong biến môi trường.")
        return

    rotator = GroqClientRotator(keys_str)

    print(f"Đang đọc dữ liệu từ: {file_path}")
    df = pd.read_parquet(file_path)
    
    output_file = os.path.join(os.path.dirname(file_path), "checked_delete.parquet")
    
    while True:
        # Loại trừ những URL đã được check (nếu chạy nhiều lần)
        if os.path.exists(output_file):
            checked_df = pd.read_parquet(output_file)
            checked_ids = checked_df['id'].tolist()
            uncheck_df = df[~df['id'].isin(checked_ids)]
        else:
            uncheck_df = df
            checked_df = pd.DataFrame()

        if len(uncheck_df) == 0:
            print("Không còn URL nào để check! Đã hoàn thành toàn bộ.")
            break

        print(f"\n--- Đã check {len(checked_df)} URLs trước đó. Còn lại {len(uncheck_df)} URLs chưa check. ---")
        
        sample_df = uncheck_df.sample(n=min(sample_size, len(uncheck_df))).copy()
        
        urls_list = []
        for _, row in sample_df.iterrows():
            urls_list.append(f"ID: {row['id']} | URL: {row['url']} | Reason: {row['reason']}")
            
        urls_text = "\n".join(urls_list)
        
        prompt = f"""
Bạn là một chuyên gia rà soát dữ liệu y khoa (Medical Data Reviewer) đa ngôn ngữ (Anh, Việt, Trung).
Hệ thống cào dữ liệu của tôi đã tự động lọc bỏ các URL sau vì cho rằng chúng không chứa thông tin y khoa.

Dưới đây là danh sách {len(sample_df)} URLs đã bị lọc bỏ.
{urls_text}

NHIỆM VỤ CỦA BẠN:
Đánh giá TỪNG URL một xem việc hệ thống lọc bỏ URL này là đúng hay sai.
Lưu ý các Edge Cases: 
1. Đa ngôn ngữ (Anh, Việt, Trung).
2. URL chỉ chứa ID số nhưng Domain là y khoa (VD: ask.39.net, pubmed.ncbi.nlm.nih.gov).
3. Các chủ đề Y khoa mở rộng (dinh dưỡng, mẹ và bé, thể chất, sức khỏe tâm thần).

YÊU CẦU ĐẦU RA:
Bạn BẮT BUỘC phải trả về CHUẨN JSON (chỉ JSON, không markdown, không text dư thừa) theo cấu trúc sau:
{{
  "evaluations": [
    {{
      "id": 12345,
      "agreement": "Đồng ý", 
      "reason": "Giải thích ngắn gọn (VD: Đây là trang bán giày)"
    }},
    {{
      "id": 67890,
      "agreement": "Không đồng ý",
      "reason": "Giải thích ngắn gọn (VD: Domain ask.39.net là diễn đàn hỏi đáp y tế)"
    }}
  ]
}}
* Trường "agreement" CHỈ ĐƯỢC PHÉP điền 1 trong 3 giá trị sau:
- "Đồng ý": Bạn đồng ý rằng URL này là rác và NÊN BỊ XÓA.
- "Không đồng ý": Bạn cho rằng URL này là tài liệu y khoa hợp lệ và NÊN GIỮ LẠI.
- "Chưa chắc chắn": Bạn không đủ thông tin để kết luận từ URL.
"""
        
        print(f"Đang gửi {len(sample_df)} URLs tới mô hình {model_name} (qua API) để phân tích...")
        try:
            client = rotator.get_client()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8192
            )
            
            content = response.choices[0].message.content
            if not content:
                content = ""
            print("Nhận phản hồi từ LLM, đang xử lý JSON...")
            
            # Tìm và trích xuất JSON bằng regex (phòng khi LLM nói dông dài thêm vài câu)
            match = re.search(r'\{[\s\S]*\}', content)
            if not match:
                raise ValueError("Không tìm thấy chuỗi JSON hợp lệ trong phản hồi của LLM.")
                
            json_str = match.group(0)
            
            # Parse JSON
            result = json.loads(json_str)
            evals = result.get("evaluations", [])
            if not evals:
                raise ValueError("JSON trả về không có evaluations.")
            
            # Convert list of dicts to dataframe
            eval_df = pd.DataFrame(evals)
            
            # Ép kiểu ID về int64 để merge nếu cần
            eval_df['id'] = eval_df['id'].astype(pd.Int64Dtype())
            sample_df['id'] = sample_df['id'].astype(pd.Int64Dtype())
            
            # Merge kết quả LLM vào sample_df (chỉ lấy inner để bỏ qua các id LLM trả về thiếu/sai)
            final_df = sample_df.merge(eval_df, on='id', how='inner')
            
            # Gộp với dữ liệu đã check cũ (nếu có)
            if not checked_df.empty:
                final_df = pd.concat([checked_df, final_df], ignore_index=True)
                
            final_df.to_parquet(output_file, engine='pyarrow', index=False)
            print(f"\n[THÀNH CÔNG] Đã đánh giá và lưu {len(final_df) - len(checked_df)} kết quả mới vào {output_file}")
            
            # In tóm tắt
            counts = final_df['agreement'].value_counts(dropna=False)
            print("\n--- THỐNG KÊ TỔNG CỘNG ---")
            for k, v in counts.items():
                print(f"- {k}: {v}")
                
        except Exception as e:
            print(f"Lỗi: {e}\n\nPhản hồi thô từ LLM (nếu có):\n{content if 'content' in locals() else 'Không có'}")
            print("Tạm dừng 5 giây trước khi thử lại...")
            time.sleep(5)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Dùng LLM (Groq) kiểm tra lại các URL bị filter và lưu vào checked_delete.parquet.")
    parser.add_argument("--file", type=str, default="data/raw/corpus_text/dropped_urls_merged/dropped_filtered.parquet", help="Đường dẫn tới file parquet.")
    parser.add_argument("--sample", type=int, default=100, help="Số lượng URL ngẫu nhiên muốn LLM kiểm tra 1 lần.")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Tên model Groq muốn dùng.")
    
    args = parser.parse_args()
    verify_dropped_urls(args.file, args.sample, model_name=args.model)
