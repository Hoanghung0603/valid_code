# AGY Task A2 – Tìm kiếm tiếng Trung chạy trên Colab + gộp với tiếng Việt (KHÔNG nộp bài)

Bối cảnh: docs/AGY_TASK_2.md, experiments.csv (bản tốt nhất v2 = 0.0988: scripts/pipeline_v2.py, VI top40 + scripts/chunk_v2.py).
Phân tích chấm thật: data/eval/details_4195_v1.jsonl, data/eval/details_4208_v2.jsonl.
Phần lớn gold là trang tiếng Trung. Dữ liệu ZH đang được crawl trên Colab vào /content/docs_zh/corpus-*.parquet
(schema Doc: doc_id str, lang, title, text...), sẽ có vài trăm nghìn → ~2.5 triệu doc. Colab: 2 CPU, ~12GB RAM, Python 3.13.

## Viết 2 file (tự chứa, ít phụ thuộc):
1. `colab/zh_search.py` — chạy TRÊN COLAB:
   `python colab/zh_search.py --docs /content/docs_zh --plans query_plans.jsonl --out /content/zh_preds.jsonl --k 100`
   - Không giữ toàn bộ corpus trong RAM: duyệt từng shard parquet (hoặc nhóm shard ~200k doc), build BM25 (jieba, có
     stopword, idf toàn cục tính ở lượt 1 bằng đếm df trên hashed vocab) rồi tìm cho cả 1200 truy vấn, giữ top-k toàn cục / truy vấn.
   - Truy vấn = plan["zh"] + plan["keywords"]["zh"] (+ title boost). Điểm BM25 chuẩn.
   - Với top-k doc mỗi truy vấn: chọn chunk bằng logic của scripts/chunk_v2.py (copy hàm cần thiết vào file, lang="zh",
     chunk dài ~ tương đương 1700 ký tự VI ≈ 600–900 ký tự ZH; thử để tham số).
   - Ghi jsonl: {"id": qid, "docs": [[score, doc_id], ...], "chunks": [{"doc_id": ..., "chunk_text": ..., "score": ...}]}
   - Phụ thuộc: pyarrow, numpy, scipy, jieba (pip có sẵn trên Colab trừ jieba). In tiến độ, thời gian, RAM.
2. `scripts/merge_preds.py` — chạy trên máy này: gộp VI (sinh lại bằng pipeline_v2 hoặc đọc 1 file preds VI jsonl cùng định dạng —
   thêm option `--dump-preds` cho pipeline_v2 nếu cần) với zh_preds.jsonl theo quota `--n-vi 20 --n-zh 60` hoặc RRF, ghi zip nộp
   (doc_id INT, đủ 1200 id, chunk_text là substring nguyên văn – kiểm tra với text gốc nếu có).
3. Test cả hai trên dữ liệu nhỏ tại máy: các doc lang=zh trong data/docs/corpus-00000.parquet (~200 doc) – chạy end-to-end ra zip hợp lệ.
4. Ghi docs/AGY_TASK_A2_REPORT.md: cách chạy, ước lượng RAM/thời gian cho 2.5M doc trên Colab.
Máy này yếu: nice -n 15, RAM ≤ 8GB. Không nộp bài, không đọc .r2ai_cred.
