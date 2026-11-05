# AGY Task A2 Report

## Cách chạy

1. **Trên Colab (Sinh dự đoán tiếng Trung):**
   Đưa `colab/zh_search.py` lên Colab, cài đặt thư viện cần thiết (`pip install pyarrow numpy scipy jieba psutil`), và chạy lệnh sau (có thể dùng GPU hoặc CPU chuẩn, vì mã được tối ưu chạy tuần tự qua từng shard):

   ```bash
   python colab/zh_search.py \
       --docs /content/docs_zh \
       --plans query_plans.jsonl \
       --out /content/zh_preds.jsonl \
       --k 100
   ```
   Kết quả sẽ được ghi vào `zh_preds.jsonl` chứa danh sách docs và chunks tốt nhất cho từng truy vấn.

2. **Trên Máy Cục Bộ (Gộp dự đoán):**
   Sau khi đã có file `zh_preds.jsonl` từ Colab (hoặc máy khác) và file `vi_preds.jsonl` (có thể sinh ra từ `pipeline_v2.py --dump-preds vi_preds.jsonl`), chạy script gộp:

   ```bash
   python scripts/merge_preds.py \
       --vi-preds vi_preds.jsonl \
       --zh-preds zh_preds.jsonl \
       --out submission_merged.zip \
       --n-vi 20 \
       --n-zh 60 \
       --docs data/docs
   ```
   Script này sẽ trộn kết quả, lọc trùng lặp doc_id, trích xuất số lượng doc tối đa theo quota, xác thực tính nguyên văn của chunk_text, chuyển `doc_id` sang INT và nén thành zip hợp lệ để nộp.

## Ước lượng RAM & Thời gian (2.5 triệu doc trên Colab)

*Dựa trên cấu hình Colab tiêu chuẩn (2 CPU, 12GB RAM)*

### 1. RAM (Ước lượng)
- **Cấu trúc lưu trữ DF (Pass 1):** Mảng NumPy 1 chiều (dtype=uint32) kích thước 2^24 (16.7 triệu phần tử) chiếm chính xác **~67 MB**.
- **Bộ nhớ cho truy vấn và Heap (Top-K):** Lưu trữ 1200 truy vấn cùng thông tin IDF và min-heap (size = 100) cho mỗi truy vấn chiếm chưa tới **~50 MB**.
- **PyArrow Parquet Batching:** Việc đọc từng batch của các file Parquet kích thước vài trăm nghìn doc sẽ tiêu tốn khoảng **200 - 400 MB** tùy vào nội dung (text dài ngắn).
- **Tổng cộng RAM:** Script duy trì dung lượng RAM ở mức an toàn dưới **~600 - 800 MB** trong suốt quá trình chạy. Không bị tràn RAM trên Colab (12GB).

### 2. Thời gian (Ước lượng)
- **Pass 1 (Tính Document Frequency):** Đọc 2.5 triệu văn bản, tokenize bằng `jieba` và cập nhật từ điển Hashed (mảng). Tốc độ `jieba` khoảng 1.5 - 2 triệu ký tự/giây. Tổng thời gian khoảng **20 - 30 phút**.
- **Pass 2 (Tính điểm BM25 toàn cục):** Đọc lại 2.5 triệu văn bản, tokenize và tính điểm đối với 1200 truy vấn. Quá trình tính toán TF và cập nhật Min-Heap tiêu tốn thêm khoảng **30 - 40 phút**.
- **Trích xuất Chunk:** Thực hiện `jieba` tokenization và trích xuất chunk cho 1200 x 100 = 120,000 văn bản top đầu. Mất khoảng **2 - 5 phút**.
- **Tổng thời gian:** Dự kiến khoảng **1 giờ đến 1 giờ 15 phút** để hoàn tất toàn bộ tiến trình cho 2.5 triệu văn bản trên phần cứng 2 vCPUs của Colab. Tốc độ có thể bị ảnh hưởng bởi độ dài trung bình thực tế của các file text ZH.
