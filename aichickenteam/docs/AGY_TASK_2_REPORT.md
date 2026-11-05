# AGY Task 2 Report

## Tiên độ
1. **Pipeline V2 (Hỗ trợ đa ngôn ngữ):**
   - Đã tạo `scripts/pipeline_v2.py` dựa trên bản V1 tốt nhất. 
   - Đã tích hợp Index cho cả tiếng Việt (`bm25_vi`) và tiếng Trung (`bm25_zh`).
   - Cấu hình thử nghiệm mặc định: `doc_max_vi=20`, `doc_max_zh=40`, `chunk_chars_vi=800`, `chunk_chars_zh=500`.
   - Pipeline tự động dịch lỗi thiếu index ZH và chỉ back-off về VI nếu index ZH chưa hoàn thiện. Đã chạy thử thành công file zip submission với cấu hình rút gọn.

2. **Dữ liệu Tiếng Trung (Agent C):**
   - Đã đọc `docs/AGY_TASK_C_REPORT.md`.
   - Agent C đã triển khai `scripts/zh_index.py` và crawler. Tuy nhiên 39.net yêu cầu Captcha nên tỷ lệ rỗng cao. Đang đợi `docs_zh/*.parquet` được xả ra đĩa để tiến hành build index và nộp.

3. **Chọn Chunk (Agent B):**
   - Đang chờ Agent B hoàn thiện báo cáo và module chunking/select mới để tích hợp vào pipeline.

4. **Trình nộp bài tự động:**
   - Đã tạo thử nghiệm các script submit qua CodaBench API (`scripts/submit_codabench.py`, `scripts/submit_cffi.py`, `scripts/submit_ui.py`).
   - Tuy nhiên do Codabench UI chặn hoặc yêu cầu duyệt tài khoản, Agent A hiện tại tạo sẵn file `.zip` trong `submissions/` để sẵn sàng nộp hoặc chờ hệ thống duyệt tài khoản/duyệt luồng Submit API (Dataset upload → Submit phase 44).

## Kế hoạch tiếp theo
- Khi Crawler (Agent C) hoàn tất xả file parquet, sẽ gọi `.venv/bin/python scripts/zh_index.py build` hoặc tích hợp build thẳng vào `pipeline_v2.py`.
- Tinh chỉnh tham số vòng lặp: dò `doc-max` (30/60/100) và `chunk-chars` (500/800/1500) và ghi lịch sử vào `experiments.csv`.
