# Plan làm bài – R2AI2026 Truy hồi tài liệu y khoa đa ngôn ngữ

> Chi tiết cuộc thi: [R2AI2026_cuoc_thi.md](R2AI2026_cuoc_thi.md)
> Phần cứng hiện có: RTX 3050 **8GB VRAM**, 31GB RAM, 16 CPU. Mọi lựa chọn mô hình bên dưới đều phải chạy vừa cấu hình này.

---

## 0. Nhận định chiến lược

| Đặc điểm đề | Hệ quả cho thiết kế |
|---|---|
| Chấm bằng **F2**, nặng Recall gấp 4 lần Precision | Nên trả về *hơi nhiều* kết quả. Nhưng trả quá nhiều thì Precision sụt, kéo F2 xuống, nên cần **ngưỡng cắt động theo từng truy vấn** thay vì top-k cố định. |
| Chấm **macro theo truy vấn** | Câu nào cũng có trọng số như nhau. Tránh để câu nào trả về rỗng. |
| Chấm 2 cấp **doc** và **chunk** | Hai bài toán riêng. Chunk phải được trích **nguyên văn** từ tài liệu gốc. |
| Tiếng Anh = **toàn bộ PubMed** (hơn 36 triệu bài) | Không thể tự index hết. Phải dùng **API search (ESearch / Europe PMC) để lấy ứng viên, sau đó rerank bằng mô hình**. |
| Việt/Trung = danh sách URL do BTC cho | Tự crawl và index cục bộ, quy mô nhỏ nên làm được bằng dense + sparse. |
| Mô hình mở, ≤14B, phát hành trước 01/06/2026, không dùng API đóng | Dùng BGE-M3, bge-reranker-v2-m3, họ Qwen3 (Embedding/Reranker/LLM). |
| Không có tập train có nhãn | Phải **tự tạo dev set** (gán nhãn tay + sinh tổng hợp) và tận dụng leaderboard public (10 lượt/ngày). |
| Private: 01/11–04/11, **5 bài tổng cộng** | Pipeline phải **chạy tự động end-to-end** trên bộ câu hỏi mới, kể cả phần gọi PubMed. |

## 1. Câu hỏi cần hỏi BTC ngay (trước 01/10)

1. **chunk_text được so khớp thế nào?** Khớp chính xác, chứa nhau (containment) hay theo độ trùng token/ROUGE? Câu này quyết định chiến lược chunking, và là rủi ro lớn nhất.
2. Nhãn chunk tiếng Anh lấy từ **abstract** hay **full-text** (PMC)?
3. FINAL SCORE được tính thế nào? Các cột TABLES / EXECUTION / ANSWER trên leaderboard có vẻ chép từ cuộc thi khác.
4. Hạn nộp thực tế là 07:00 (theo cấu hình hệ thống) hay 23:59 (theo trang Overview)?
5. Có giới hạn số doc/chunk tối đa cho mỗi truy vấn không? Thứ tự có ảnh hưởng không (leaderboard có cột MRR5)?
6. File danh sách URL Việt/Trung khi nào phát hành? Có kèm snapshot nội dung không, phòng khi link chết?

## 2. Kiến trúc hệ thống

```
Truy vấn tiếng Việt
   │
   ├─► [Query Understanding] LLM mở (Qwen3-8B / Qwen2.5-7B, 4-bit)
   │      • dịch sang EN (thuật ngữ MeSH) + ZH
   │      • tách truy vấn nhiều ý → sub-queries
   │      • sinh từ khoá / đồng nghĩa (HyDE tuỳ chọn)
   │
   ├─► Nhánh VI/ZH (index cục bộ)
   │      Hybrid: BM25 (pyvi/underthesea cho VI, jieba cho ZH) + dense BGE-M3
   │      → RRF fusion → top ~100 chunk
   │
   ├─► Nhánh EN (PubMed online + cache)
   │      ESearch + Europe PMC với nhiều câu query (EN, MeSH, sub-queries)
   │      → hợp nhất ~200–500 PMID → EFetch title+abstract (có cache)
   │      → chunk abstract → dense BGE-M3 lọc sơ bộ → top ~100
   │
   └─► [Rerank chung] cross-encoder đa ngữ (bge-reranker-v2-m3 / Qwen3-Reranker)
          chấm (query_vi, chunk) trực tiếp, không cần dịch chunk
          │
          ├─► Chọn chunk: ngưỡng động (tuyệt đối + tương đối so với top1), min 1, max K
          └─► Chọn doc: gộp điểm chunk theo doc (max / sum top-2) → ngưỡng động
                 → ghi JSON: relevant_docs, relevant_chunks
```

### Lựa chọn mô hình (vừa 8GB VRAM)

| Vai trò | Lựa chọn chính | Dự phòng | Ghi chú |
|---|---|---|---|
| Embedding | **BGE-M3** (568M, fp16 ~2.3GB) | Qwen3-Embedding-0.6B, multilingual-e5-large | BGE-M3 có cả dense và sparse, hỗ trợ VI/ZH/EN tốt |
| Reranker | **bge-reranker-v2-m3** | Qwen3-Reranker-0.6B / 4B (4-bit) | Kiểm tra ngày phát hành < 01/06/2026 |
| LLM (dịch/mở rộng query) | **Qwen3-8B** hoặc Qwen2.5-7B-Instruct (GGUF Q4 qua llama.cpp/Ollama) | Qwen3-4B | Chỉ xử lý truy vấn (vài trăm câu), không đụng kho dữ liệu |
| Tách từ | pyvi / underthesea (VI), jieba (ZH) | – | Cho BM25 |

> Mỗi mô hình phải ghi lại link Hugging Face và ngày phát hành để nộp kèm working notes.

## 3. Chi tiết từng thành phần

### 3.1 Thu thập & tiền xử lý (VI/ZH)
- Crawler async (httpx + asyncio), rate-limit theo domain, retry, **lưu HTML thô** để xử lý lại khi cần.
- Trích nội dung chính: `trafilatura`, có fallback `readability-lxml`. Viết rule riêng nếu vài domain chiếm phần lớn dữ liệu.
- Chuẩn hoá: Unicode NFC, bỏ menu/quảng cáo/"bài liên quan". **Giữ nguyên văn bản gốc** để `chunk_text` luôn là trích đoạn thật.
- Lưu mỗi doc: `{id, url, lang, title, text, paragraphs[]}` ở dạng JSONL/Parquet.

### 3.2 Chunking
- Đơn vị gốc: **đoạn văn / câu**, lưu offset trong văn bản gốc.
- Tạo nhiều kích thước (1 câu, 2–3 câu, 1 đoạn, sliding ~200 token) và **chọn theo kết quả dev/leaderboard**. Cách chấm chunk của BTC quyết định kích thước tối ưu (xem mục 1).
- PubMed: title và các câu/đoạn của abstract (tách theo nhãn BACKGROUND/METHODS/RESULTS/CONCLUSIONS nếu có).
- Kiểm tra bắt buộc: `chunk_text ∈ text_gốc` (substring).

### 3.3 Nhánh PubMed
- Đăng ký **NCBI API key** (10 req/s, không có key chỉ được 3 req/s) và dùng Europe PMC song song.
- Với mỗi truy vấn, sinh 3–6 câu search: bản dịch EN, bản dùng MeSH term, từng sub-query, và bản Boolean nới lỏng.
- **Cache toàn bộ** (sqlite): query→PMID, PMID→abstract. Chạy trước cho public test để có sẵn dữ liệu.
- Lọc sơ bộ: bài phải có abstract. Có thể ưu tiên review/guideline bằng publication type (kiểm chứng trên dev).
- Ước tính thời gian: khoảng 300 câu × 6 search × 0.1s, cộng EFetch theo batch 200 PMID, tổng dưới 30 phút. Vừa với khung private 3 ngày.

### 3.4 Chọn số lượng kết quả (quan trọng cho F2)
- Chunk: giữ những chunk có `score ≥ τ_abs` **và** `score ≥ α·score_top1`, tối thiểu 1, tối đa K.
- Doc: `doc_score = max(chunk_score)`, có thể cộng thêm phần thưởng khi doc có nhiều chunk tốt. Áp ngưỡng tương tự.
- Tối ưu α, τ, K trên dev set. F2 thường tối ưu khi trả về nhiều hơn số gold khoảng 1.5–2 lần.
- Bảo đảm **relevant_docs ⊇ doc_id của relevant_chunks** (nhất quán giữa hai cấp).
- Cân bằng ngôn ngữ: kiểm tra xem gold có phân bố đều VI/ZH/EN không; nếu không, thêm hệ số theo ngôn ngữ.

### 3.5 Dev set & đánh giá nội bộ
- Viết `evaluate.py` tái hiện đúng công thức P/R/F2 macro ở 2 cấp. Với chunk, thử nhiều cách khớp: exact, containment, overlap ≥ 0.5.
- **Dev tay:** khoảng 50 truy vấn tiếng Việt tự viết theo phong cách ví dụ của BTC, gán nhãn doc/chunk bằng pooling (lấy top của nhiều hệ thống rồi người duyệt).
- **Dev tổng hợp:** lấy chunk ngẫu nhiên (VI/ZH/EN), dùng LLM sinh câu hỏi tiếng Việt, chunk đó là gold. Cách này có sẵn hàng nghìn cặp để tune retriever và ngưỡng.
- Mỗi thí nghiệm log vào `experiments.csv` (config, điểm dev, điểm public nếu có nộp).

### 3.6 (Tuỳ chọn) Fine-tune
- Nếu còn thời gian: fine-tune nhẹ BGE-M3 hoặc reranker trên dữ liệu tổng hợp (query_vi, chunk đa ngữ), có hard negatives. Dùng LoRA hoặc batch nhỏ + gradient checkpointing để vừa 8GB.
- Chỉ làm khi baseline đã chạy ổn và dev set đủ tin cậy.

## 4. Lộ trình

| Thời gian | Việc | Đầu ra |
|---|---|---|
| **23/09 – 30/09** (chuẩn bị) | Hỏi BTC các câu ở mục 1. Dựng repo và môi trường; tải BGE-M3, reranker, Qwen. Viết client PubMed + cache, crawler, chunker, `evaluate.py`, writer/validator JSON + ZIP. Thử trên vài URL y khoa VI/ZH tự chọn. | Pipeline khung chạy được |
| **01/10 – 05/10** | Có data: crawl toàn bộ URL VI/ZH, build index. Baseline 1: BM25 + dịch EN + ESearch. **Nộp baseline sớm** để hiểu cách chấm. | Bài nộp đầu tiên, điểm tham chiếu |
| **06/10 – 15/10** | Hybrid BGE-M3 + rerank. Query expansion/tách ý. Dựng dev set (tay + tổng hợp). So sánh các kích thước chunk trên leaderboard. | Hệ thống v2 |
| **16/10 – 25/10** | Tune ngưỡng động, cân bằng ngôn ngữ, gộp điểm doc. Ablation có kiểm soát (thay đổi 1 yếu tố mỗi lượt nộp). | Hệ thống v3 + bảng ablation |
| **26/10 – 31/10** | Đóng băng cấu hình. Script chạy end-to-end 1 lệnh: `run.py --queries test.jsonl --out submission.zip`. Diễn tập với bộ câu hỏi mới giả lập, đo thời gian. | Pipeline sẵn sàng cho private |
| **01/11 – 04/11** (private, 5 lượt) | Lượt 1: config tốt nhất. Lượt 2–3: biến thể ngưỡng (nhiều Recall hơn / nhiều Precision hơn). Để dự phòng 1–2 lượt. **Nộp trước 07:00 ngày 04/11.** | Bài nộp chính thức |
| **05/11 – …** | Viết **working notes paper** (bắt buộc để kết quả được công nhận): phương pháp, mô hình kèm ngày phát hành, nguồn dữ liệu, ablation. | Paper |

### Chiến lược dùng lượt nộp public (10/ngày)
- Mỗi lượt chỉ đổi **1 biến** so với bản tốt nhất, và ghi log đầy đủ.
- Dùng vài lượt đầu để "dò" cách chấm chunk: cùng một tập doc, thử chunk 1 câu / 1 đoạn / cả abstract.
- Không overfit public: chỉ giữ thay đổi nào cải thiện cả dev lẫn public.

## 5. Cấu trúc repo đề xuất

```
test2/
├── R2AI2026_cuoc_thi.md
├── PLAN.md
├── data/            # raw html, docs.parquet, queries, cache.sqlite (gitignore)
├── indexes/         # BM25, FAISS/Qdrant
├── src/
│   ├── crawl.py     # VI/ZH crawler + extraction
│   ├── pubmed.py    # ESearch/EFetch/EuropePMC + cache
│   ├── chunk.py
│   ├── index.py     # BM25 + BGE-M3 dense/sparse
│   ├── query.py     # dịch/mở rộng/tách ý bằng LLM
│   ├── retrieve.py  # hybrid + RRF
│   ├── rerank.py
│   ├── select.py    # ngưỡng động doc/chunk
│   ├── evaluate.py  # P/R/F2 macro 2 cấp
│   └── submit.py    # validate JSON + zip
├── configs/         # yaml cho từng thí nghiệm
├── run.py           # end-to-end
└── experiments.csv
```

## 6. Rủi ro & phương án

| Rủi ro | Phương án |
|---|---|
| Cách khớp chunk không rõ, chunk sai kích thước làm điểm chunk ≈ 0 | Hỏi BTC; dò bằng leaderboard ngay tuần đầu |
| URL VI/ZH chết hoặc chặn crawl | Retry, đổi User-Agent, dùng Wayback Machine; crawl ngay ngày 01/10 |
| PubMed rate-limit / API lỗi trong private | API key, cache, chạy song song Europe PMC; chạy sớm ngày 01/11 |
| Gold EN nằm ngoài tập ứng viên API trả về (recall trần thấp) | Tăng số câu search, dùng MeSH, related-articles (ELink `pubmed_pubmed`) từ top hit |
| VRAM 8GB không đủ | Quantize 4-bit, batch nhỏ, tách bước (LLM chạy xong mới load reranker); nếu cần thì thuê GPU ngắn hạn |
| Vi phạm quy định mô hình | Lập bảng mô hình + ngày phát hành + số tham số trước khi dùng |

## 7. Việc làm ngay tuần này

- [ ] Gửi email hỏi BTC (mục 1)
- [ ] Đăng ký tài khoản đội và NCBI API key
- [ ] Setup môi trường: `torch`, `FlagEmbedding`, `rank_bm25`/`bm25s`, `faiss`, `trafilatura`, `httpx`, `pyvi`, `jieba`, `llama.cpp`/`ollama`
- [ ] Viết `pubmed.py` + cache, thử với truy vấn mẫu "sỏi thận tắc nghẽn đường tiết niệu"
- [ ] Viết `evaluate.py` và `submit.py` (validator định dạng)
- [ ] Prototype end-to-end trên ~10 truy vấn tự viết
