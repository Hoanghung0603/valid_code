# AGY Task 2 – Tối ưu điểm leaderboard R2AI2026 (vòng lặp: làm → nộp → đọc điểm → sửa)

Thư mục: /home/phong/Desktop/hsk/test2 · Python: `.venv/bin/python` · RAM 31GB, máy yếu → mọi tiến trình nặng chạy `nice -n 15`,
giữ ≤ 12GB RAM, KHÔNG chạy model GPU trên máy này (GPU/LLM nặng → Colab). Không xoá dữ liệu trong data/.
Đọc trước: R2AI2026_cuoc_thi.md, docs/INTERFACES.md, scripts/pipeline_v1.py (bản tốt nhất hiện tại).

## Sự thật đã xác minh (đừng làm lại)
- Chấm: FINAL = (DOCS F2-macro + CHUNKS F2-macro)/2, public = 600/1200 câu (file nộp vẫn đủ 1200).
- **doc_id phải là INT** (id trong data/ViBioMIR/links_corpus.parquet). Chuỗi → bị hiểu "pubmed:<id>" → 0 điểm.
- **Chỉ doc trong links_corpus được tính. PubMed vô giá trị → bỏ.**
- Gold/câu: trung vị 60 doc (0–190), ~75 chunk; ≈1 chunk gold / doc gold.
- Chunk đúng nếu LCS_tokens(pred, gold)/|gold| ≥ 0.4 (sau NFKC, lowercase, gộp khoảng trắng), cùng doc_id; trùng ≥0.8 bị loại.
  ⇒ chunk dài dễ phủ gold hơn (mẫu số là độ dài gold). Có trường tuỳ chọn chunk_order.
- Kết quả: v0 (3 doc/câu) 0.0077; **v1 (BM25 VI, 40 doc/câu, chunk ~800 ký tự) 0.0728**
  (doc P .164 R .133; chunk P .032 R .029). Baseline BTC 0.3764 (doc P .88 R .51; chunk F2 .194).
- Số doc VI trúng gold bão hoà ~8/câu dù gold 57→170 ⇒ **phần lớn gold là trang TIẾNG TRUNG**.
  Trong doc VI đúng, chunk 800 ký tự chỉ trúng gold ~19%.

## Dữ liệu
- VI: data/docs/corpus-*.parquet (~625k doc), index BM25 sẵn ở data/cache/bm25_vi (scripts/pipeline_v1.py build/predict).
- ZH: đang crawl trên Colab → Google Drive `MyDrive/r2ai/docs_zh/corpus-*.parquet` (cùng schema Doc; ~20h).
  Lấy về máy bằng Google Drive (rclone/gdown hoặc nhờ người dùng tải) – hoặc chạy phần ZH ngay trên Colab.
  39.net bị rỗng khi crawl từ Colab → crawl 39.net trên máy này: `nice -n 15 .venv/bin/python -m src.crawl --domains 39.net --per-domain 8`
  (split_hosts: [39.net] trong configs/default.yaml), extract_workers thấp.
- Query plans (vi/en/zh rewrites, keywords): data/cache/query_plans.jsonl (1200).

## Việc cần làm (lặp đến khi vượt 0.3764)
1. Pipeline ZH: BM25 ZH (jieba, src/index.py tokenize "zh") trên doc ZH đã có, truy vấn = plan["zh"] + keywords.zh.
   Gộp với VI (RRF hoặc phân bổ số doc theo ngôn ngữ, ví dụ 20 VI + 40 ZH), chunk ZH dài ~300–600 ký tự.
2. Dò số doc/câu (30/60/100) và độ dài chunk (500/800/1500 ký tự, hoặc nhiều chunk/doc) – F2 ưu tiên recall.
3. Mỗi lần nộp: zip chứa đúng 1 file .json, đủ 1200 id, doc_id INT, chunk_text là substring nguyên văn.
   Kiểm tra: `.venv/bin/python -m src.submit check <zip>` (chú ý: validator cũ coi doc_id chuỗi; chỉ chuyển INT ở bước ghi zip như pipeline_v1).
4. Nộp: https://leaderboard.aiguru.com.vn/competitions/15/ (Codabench). Tài khoản: đọc file .r2ai_cred (R2AI_USER/R2AI_PASS, không in ra log, không copy đi nơi khác)
   Agent A là agent DUY NHẤT được nộp. Tối đa 10 lần/ngày – mỗi lần chỉ đổi 1 yếu tố.
   Sau khi chấm: GET /api/submissions/<id>/get_details/ → `scoring_result` (zip: scores.txt, evaluation.json, details.jsonl
   per-query: gold count, relevant_predictions, chunk best_overlap) → phân tích để quyết định bước sau.
5. Ghi nhật ký mọi lần nộp vào experiments.csv (ngày, file, cấu hình, điểm 7 cột) và tóm tắt trong docs/AGY_TASK_2_REPORT.md.
