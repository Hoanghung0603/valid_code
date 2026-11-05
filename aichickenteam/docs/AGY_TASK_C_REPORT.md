# Báo cáo AGY Task C

1. **Crawl 39.net:**
   - Đã sửa `configs/default.yaml` thêm `split_hosts: [39.net]`.
   - Đang chạy tiến trình crawl nền với lệnh `nice -n 15 .venv/bin/python -m src.crawl --domains 39.net --per-domain 8 --delay 0.1 --out-dir data/docs_zh`.
   - **Tỉ lệ rỗng (empty):** Rất cao (trên 90%). Khi kiểm tra nguyên nhân, nhận thấy các trang trả về `text: ''` với dung lượng tải về là 2287 bytes. Đây là trang yêu cầu **Captcha (滑动拼图验证 - Sliding puzzle verification)** chứ không phải lỗi do `extractor` hay `charset` giải mã sai.
   - Một số trang thuộc `ask.39.net` hay `jbk.39.net` (dung lượng 40KB+) thì crawler vẫn trích xuất thành công và lưu với trạng thái `ok`.

2. **Crawl tiếng Việt:**
   - Đã kiểm tra `load average` của hệ thống hiện tại đang ở mức `12.08` (cao hơn ngưỡng 8). Do đó, tạm thời không khởi chạy tiến trình crawl tiếng Việt để tránh quá tải máy.

3. **Index tiếng Trung (`scripts/zh_index.py`):**
   - Đã viết script `scripts/zh_index.py` thực hiện build BM25 shard + hashed vocab tương tự `pipeline_v1.py` cho `data/docs_zh/*.parquet`.
   - Hỗ trợ thêm thư mục của Google Drive nếu người dùng tải sẵn dữ liệu (tìm trong các thư mục mặc định như `~/Drive/MyDrive/r2ai/docs_zh`).
   - Hàm `search_zh(plan, k=20)` đã được viết sẵn để import vào pipeline, nhận vào `plan` và trả về `[(score, doc_id)]`.

Agent A có thể import `from scripts.zh_index import search_zh` để bắt đầu tích hợp.
