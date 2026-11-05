# AGY Task 1 – Pipeline v1 (BM25 tiếng Việt + query plans), sinh nhiều biến thể để nộp

Thư mục dự án: /home/phong/Desktop/hsk/test2. Đọc trước: R2AI2026_cuoc_thi.md, docs/INTERFACES.md, scripts/baseline_bm25.py.
Python: .venv/bin/python. KHÔNG cài/gỡ gói toàn cục, KHÔNG xoá dữ liệu trong data/, KHÔNG dừng các tiến trình đang chạy
(src.crawl, src.pubmed). RAM máy 31GB (crawl đang dùng một phần) → giữ tiến trình của bạn ≤ 14GB.

## Bối cảnh
- Điểm = (DOCS F2-macro + CHUNKS F2-macro)/2. Baseline BM25 của BTC: 0.3764 (doc F2 0.559, P 0.88, R 0.51; chunk F2 0.194, P 0.34, R 0.18).
  → Tài liệu đúng trùng từ vựng nhiều với câu hỏi; nhiều câu hỏi giống phần "hỏi đáp" trên vinmec/longchau/... .
- Đã có: data/docs/corpus-*.parquet (~630k trang VI + vài trăm ZH), data/cache/query_plans.jsonl (1200 plan: vi/en/zh rewrites, keywords).
- scripts/baseline_bm25.py tạo submissions/baseline_v0.zip (BM25 doc-level trên 205k doc, chunk = đoạn BM25 cao nhất).
  Vấn đề: với 630k doc thì RAM vượt (205k doc đã tốn 10.6GB); chunk hay là tiêu đề ngắn.

## Việc cần làm
1. scripts/pipeline_v1.py: BM25 tiếng Việt chạy được trên TOÀN BỘ doc VI hiện có trong RAM ≤ 14GB
   (ví dụ: hash vocab về 2^24 bucket, int32/float16, cắt text 6000 ký tự, build theo lô rồi vstack; hoặc dùng src/index.py build_chunks/build_bm25 đã có sẵn).
2. Truy vấn: dùng query_vi + plan["vi"] (biến thể) + plan["keywords"]["vi"]; thử gộp điểm (sum có trọng số / RRF).
3. Chunk: chunk ứng viên là đoạn văn (paragraph) và cửa sổ 2-3 câu; loại chunk < 80 ký tự hoặc chỉ là tiêu đề/câu hỏi
   (nếu đoạn là câu hỏi giống query thì lấy đoạn NGAY SAU nó – phần trả lời). chunk_text phải là substring nguyên văn của doc.text.
4. Không có nhãn → tự làm dev set nhỏ "pseudo-gold": các query mà văn bản query xuất hiện gần nguyên văn trong 1 doc
   (substring/ngưỡng overlap cao) → doc đó là gold doc; dùng để so sánh các cấu hình (src/evaluate.py có sẵn).
5. Xuất 4 file nộp vào submissions/ (đặt tên v1a..v1d.zip) khác nhau đúng 1 yếu tố, ví dụ:
   v1a: doc top1 + chunk top1;  v1b: doc ≤2 (ratio 0.85) + 1 chunk/doc;  v1c: doc ≤3 (ratio 0.8) + 2 chunk/doc;  v1d: như v1b nhưng chunk = cửa sổ 2-3 câu.
   Kiểm tra mỗi file bằng `.venv/bin/python -m src.submit check <zip> --docs data/docs`.
6. Ghi docs/AGY_TASK_1_REPORT.md: cách làm, RAM/thời gian, thống kê (#doc,#chunk TB), kết quả trên pseudo-gold, mô tả từng zip.
