# AGY Task B – Cải thiện chọn chunk (không nộp bài)

Đọc docs/AGY_TASK_2.md (bối cảnh + sự thật đã xác minh). Agent A lo nộp bài; bạn KHÔNG nộp, KHÔNG sửa scripts/pipeline_v1.py
(tạo file mới scripts/chunk_v2.py). Máy yếu: `nice -n 15`, RAM ≤ 8GB, không dùng GPU.

Dữ liệu chấm thật của lần nộp v1 (submissions/v1_k40.zip): per-query details nằm trong scoring_result của submission 4195
(agent A có thể tải; nếu chưa có, tự tải bằng tài khoản trong .r2ai_cred: GET /api/submissions/4195/get_details/ → scoring_result).
Mỗi prediction chunk có: doc_id, chunk_order, token_count, relevant (overlap ≥0.4), best_overlap.

Mục tiêu: trong các doc ĐÚNG, tăng tỉ lệ chunk trúng gold (hiện ~19%) mà không làm giảm precision quá nhiều.
1. Phân tích: chunk trúng vs trượt khác nhau thế nào (độ dài, vị trí trong doc, độ trùng từ với query, đoạn đầu bài/đoạn hỏi-đáp...).
2. Viết scripts/chunk_v2.py: hàm `select_chunks(query, plan, doc_text, lang, cfg) -> list[str]` (substring nguyên văn),
   hỗ trợ: chunk dài hơn, nhiều chunk/doc, cửa sổ quanh đoạn khớp, ưu tiên phần mở đầu / phần trả lời; cho cả vi và zh.
3. Đánh giá offline bằng các chunk "relevant" của v1 làm pseudo-gold (LCS/|gold| ≥0.4) và đề xuất 2–3 cấu hình cho agent A nộp.
4. Ghi docs/AGY_TASK_B_REPORT.md: phát hiện + cấu hình đề xuất + cách gọi hàm.
