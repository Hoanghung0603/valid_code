# AGY Task C – Dữ liệu tiếng Trung: crawl 39.net trên máy này + kiểm tra chất lượng (không nộp bài)

Đọc docs/AGY_TASK_2.md. Agent A lo nộp bài; bạn KHÔNG nộp. Máy yếu: mọi tiến trình `nice -n 15`, crawler extract_workers ≤ 3,
RAM ≤ 6GB. Không dừng tiến trình của agent khác.

1. Crawl 39.net (496k URL; trên Colab bị trả trang rỗng): bật `split_hosts: [39.net]` trong configs/default.yaml (chỉ sửa khối crawl),
   chạy nền: `nice -n 15 .venv/bin/python -m src.crawl --domains 39.net --per-domain 8 --delay 0.1 --out-dir data/docs_zh`
   Kiểm tra vài trang đầu có nội dung thật (không rỗng); nếu rỗng, xem src/extractors/sites.py và sửa extractor/charset.
2. Crawl tiếp phần tiếng Việt còn dở (~100k URL, đã có state trong data/cache/crawl.sqlite): `--lang vi --per-domain 8`, chạy sau 39.net
   hoặc song song nếu CPU cho phép (load average < 8).
3. Viết scripts/zh_index.py: BM25 tiếng Trung (jieba qua src.index.tokenize(..., "zh")) theo cách shard + hashed vocab như
   scripts/pipeline_v1.py (build/predict), đọc data/docs_zh/*.parquet (và thư mục Drive nếu người dùng tải về), truy vấn = plan["zh"] + keywords.zh.
   Hàm `search_zh(plan, k) -> [(score, doc_id)]` để agent A gọi. Test trên dữ liệu 39.net đã crawl.
4. Ghi docs/AGY_TASK_C_REPORT.md: tiến độ crawl, tỉ lệ rỗng, cách dùng zh_index.
