# IComeForFood – R2AI 2026 Stage 3 (ViBioMIR): mã nguồn

Truy hồi tài liệu y khoa đa ngôn ngữ (VI/EN/ZH) với truy vấn tiếng Việt. Phương pháp, dữ liệu và mô hình được mô tả trong
`THUYET_MINH.md` (cùng thư mục). Tài liệu này mô tả mã nguồn và cách chạy lại.

## 1. Cấu trúc mã nguồn

| Thư mục / tệp | Việc |
|---|---|
| `crawl/` | Thu thập HTML: `crawl_home.py` (crawl HTTP giới hạn tốc độ, dừng khi bị chặn), `crawl_browser.py`, `longchau_*.py`, `vov.py` (crawl bằng trình duyệt), `merge_browser_output.py`, `sync_pubmed_cache.py`, … |
| `crawl/commoncrawl/` | Tra chỉ mục Common Crawl và tải đoạn WARC cho trang không crawl trực tiếp được; notebook Kaggle cho 120ask.com |
| `notebooks/` | Pipeline dữ liệu của thành viên thứ hai: `01_data_pipeline` (crawl, lưu shard), `03_gold_review_and_sample2` (bộ trích), `04_pubmed_only`, `05_pubmed_rescue` (PubMed qua Europe PMC) |
| `extract/` | Trích và làm sạch: `v26/03_dbee143.py` (bộ trích dùng cho kho, bản xuất của notebook 03), `postfilter.py`, `cc_gate.py` (chạy bộ trích trên shard HTML, cổng chất lượng, ghi parquet doc), `zysjonline_dump.py` |
| `chunking/` | `chunker.py` và cấu hình `configs/s400.json` (chunk dùng cho bài nộp) |
| `retrieval/baseline.py`, `retrieval/pipeline.py` | Tách từ, mã hóa BGE-M3 (`encode`), tìm ứng viên BM25 + dense (`cands`), rerank (`rerank`), các lớp reranker (bge, Qwen3) |
| `retrieval/translate_queries.py` | Dịch câu hỏi sang EN/ZH bằng Qwen3.5-9B qua llama.cpp |
| `retrieval/sub3/` | Các bước sau rerank và tạo bài nộp: `rerank_more.py`, `qemb.py`, `qemb_pool.py`, `prf.py`, `q8rr.py`, `combine_aux2.py`, `make_sub3.py`, `lzbig.py`, `reproduce_submission.sh`, `same_submission.py`; các tệp `job_*.sh` là lệnh đã chạy trên máy GPU |
| `eval/` | Chấm offline: chuẩn hóa và LCS theo luật BTC (`r2ai_eval.py`), chấm file nộp (`score_submission.py`), chấm bộ trích (`score_extraction.py`) |
| `DAC_TA_LAM_SACH_v1.md`, `HUONG_DAN_CRAWL.md` | Đặc tả làm sạch và hướng dẫn crawl |

## 2. Môi trường

| Môi trường | Dùng cho | Phần mềm |
|---|---|---|
| CPU | Tạo bài nộp từ kết quả trung gian, BM25 nguyên doc, chunk, chấm offline | Python 3.11; `requirements-cpu.txt` |
| GPU | Mã hóa và rerank | Python 3.11, CUDA 12.4, PyTorch 2.6.0; `requirements-gpu.txt`. Đội chạy trên 1× RTX 4090 24GB, RAM 31GB |
| Trích HTML | `extract/cc_gate.py` | Python 3.11–3.13; `requirements.txt` (pandas, pyarrow, lxml, requests, charset-normalizer, duckdb) + `zstandard` |
| Dịch câu hỏi | `translate_queries.py` | llama.cpp `llama-server` với Qwen3.5-9B Q4_K_M (GGUF) |

```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt                       # máy GPU: thêm  pip install -r requirements-gpu.txt
```

Đường dẫn dữ liệu: các script đọc biến môi trường `R2AI_DATA` (mặc định `D:/r2ai_data`), trỏ tới thư mục `r2ai_data/` tải từ Drive.

## 3. Dữ liệu và checkpoint

- **Dữ liệu:** tải thư mục `r2ai_data/` từ https://drive.google.com/drive/folders/14O6HPJAtQySitIAiEKx7NGCx0vEnExkV (kiểm tra bằng `sha256sum -c SHA256SUMS.txt`), rồi đặt `R2AI_DATA=/đường/dẫn/r2ai_data`.
- **Checkpoint:** mọi mô hình là bản gốc trên Hugging Face, không tinh chỉnh; `transformers` tự tải khi chạy. Định danh và revision ở `THUYET_MINH.md` mục 3. Tải trước nếu muốn:

```
huggingface-cli download BAAI/bge-m3 --revision 5617a9f61b028005a4858fdac845db406aefb181
huggingface-cli download BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
huggingface-cli download Qwen/Qwen3-Embedding-8B --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
huggingface-cli download Qwen/Qwen3-Reranker-8B --revision 77d193c791ed757ca307ee72715aa132723da912
```

- **Chỉ mục:** chỉ mục BM25 được dựng trong bộ nhớ mỗi lần chạy `cands`; chỉ mục dense BGE-M3 (`dense.npy`, khoảng 6GB) được tạo bằng `pipeline.py encode` (mục 4.2, bước 2). Kết quả truy hồi và chấm điểm đã có sẵn trong `r2ai_data/runs/sub3/`, nên đường 4.1 không cần dựng chỉ mục.

## 4. Chạy lại

### 4.1 Tạo lại đúng bài nộp private từ kết quả trung gian (CPU, khoảng 30 phút, RAM 16GB)

```
cd retrieval/sub3
R2AI_DATA=/path/to/r2ai_data PYTHON=python bash reproduce_submission.sh
python same_submission.py $R2AI_DATA/runs/sub3/reproduce/submission.zip $R2AI_DATA/submissions/private_P2.zip
```

Các bài private khác của đội là biến thể của bài trên: P3 `W_RR=2`, P4 `W_RR=0.5` (trọng số Qwen3-Reranker-8B), P6 `W_Q8=3 W_RR=2` (trọng số Qwen3-Embedding-8B và Qwen3-Reranker-8B), P7 `THR=-3` (ngưỡng cắt doc); đặt biến trước lệnh, ví dụ `W_RR=2 R2AI_DATA=... bash reproduce_submission.sh`, và so với tệp tương ứng.

`reproduce_submission.sh` gộp điểm của các mô hình (bge, PRF, Qwen3-Embedding-8B, Qwen3-Reranker-8B), xếp lại, cắt số doc theo từng câu, cắt đoạn trả về từ văn bản doc và nén LZMA. `same_submission.py` so từng câu: danh sách doc và đoạn trả về phải trùng với bài đã nộp. Đội đã chạy lại trên máy Windows (Python 3.11, 16GB RAM, khoảng 30 phút): kết quả trùng 1.200/1.200 câu cả doc lẫn đoạn với bài private đã nộp.

### 4.2 Chạy toàn bộ pipeline từ kho doc (GPU)

Các lệnh dưới đây là đúng các lệnh đội đã chạy; `$D` là `R2AI_DATA`, `C` là danh sách thư mục chunk theo thứ tự `chunks_sub2/s400 chunks_sub3add/s400 chunks_sub4add/s400 chunks_sub5add/s400`. Thời gian đo trên RTX 4090.

1. **Chunk** (CPU, khoảng 2 giờ): `python chunking/chunker.py chunk --config chunking/configs/s400.json --docs $D/docs_sub2 --out $D/chunks_sub2` (lặp cho `sub3add`, `sub4add`, `sub5add`).
2. **Dịch câu hỏi:** chạy `llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 16384 -np 4 --port 8080 --jinja`, rồi `python retrieval/translate_queries.py --queries $D/raw/query.parquet --out $D/qtrans/v1/translations.jsonl` (có sẵn trên Drive).
3. **Mã hóa BGE-M3** (khoảng 70 phút): `python retrieval/pipeline.py encode --chunks $D/chunks_sub2/s400 --index $D/index_sub2/s400 --batch 64 --max-length 512` (lặp cho từng thư mục chunk).
4. **Ứng viên và rerank** (`retrieval/sub3/job_sub3.sh`): `pipeline.py cands`, BM25 nguyên doc (`bm25doc.py`, `bm25list.py`, `merge_list.py`), rồi `pipeline.py rerank --n-docs-pool 300 --per-doc-pool 1` → `runs/sub3/rerank.pkl` (khoảng 75 phút).
5. **Thêm chunk cho top 200:** `make_pool.py 200 3 pool_more.pkl` → `rerank_more.py pool_more.pkl rerank_more.pkl 3 $C` (25 phút).
6. **Qwen3-Embedding-0.6B** (`job_qemb.sh`, khoảng 3 giờ): `qemb.py Qwen/Qwen3-Embedding-0.6B qemb06.pkl <thư mục docs>` → `pool_qemb.py` → `rerank_more.py ... rerank_qemb.pkl 2`.
7. **Doc mới VOV/Common Crawl** (`job_vov.sh`): mã hóa chunk mới, `newcands.py`, `rerank_more.py` → `rerank_new.pkl`.
8. **PRF chủ đề** (CPU): `make_pool_best.py 400 3 pool_best400q.pkl <các file rerank>` → `prf.py pool_best400q.pkl prf400q.pkl 10,20,40 $C -- <thư mục index>`.
9. **Qwen3-Embedding-8B** (`job_qemb8_fp16.sh`, khoảng 9 giờ): `qemb.py Qwen/Qwen3-Embedding-8B qemb8.pkl ...` → `pool_qemb8.py` → `rerank_more.py ... rerank_qemb8.pkl 2`; cosine cho top 300: `qemb_pool.py Qwen/Qwen3-Embedding-8B pool_best400q.pkl 300 q8pool300.pkl ...` (biến `QUANT=4` để chạy NF4).
10. **Qwen3-Reranker-8B** (`job_fix8.sh`, khoảng 7,5 giờ): `pool_best_lab.py 300 3 pool_q8rr.pkl $C -- <các file rerank>` → `q8rr.py pool_q8rr.pkl q8rr300.pkl Qwen/Qwen3-Reranker-8B $C`.
11. **Tạo bài nộp:** như mục 4.1.

Kho doc được tạo từ HTML bằng `extract/cc_gate.py <thư mục shard HTML> <status.csv> <thư mục ra>` (biến `R2AI_KEEP=ok,short`), rồi gộp bằng `retrieval/sub3/assemble_*.py`. Nội dung web thay đổi theo thời gian, nên crawl lại không cho kết quả giống hệt; kho doc trên Drive là bản dùng cho bài nộp.

## 5. Chấm offline

```
python eval/score_submission.py --help
```

Chấm P/R/F2 mức doc và mức chunk theo luật của BTC (chuẩn hóa NFKC, lowercase, LCS token ≥ 40% độ dài đoạn tham chiếu), dùng với tập dev tự tạo; điểm chính thức chỉ có từ máy chấm của BTC.
