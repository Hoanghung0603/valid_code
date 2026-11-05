# R2AI 2026 Stage 3 – ViBioMIR: truy hồi y khoa đa ngôn ngữ (đội aichickenteam)

Tài liệu thuyết minh đầy đủ (kiến trúc, dữ liệu, mô hình, khai báo nguồn nhãn): **[docs/THUYET_MINH.md](docs/THUYET_MINH.md)**.
Nhật ký kỹ thuật và mọi thí nghiệm: [HANDOFF.md](HANDOFF.md), [experiments.csv](experiments.csv).

Bài nộp private: `submissions/private/P7_v76b.zip` (ID 7924). Public: 0.4077 theo luật chấm đoạn mới (23/10/2026).

---

## 1. Yêu cầu môi trường

| Thành phần | Phiên bản đã dùng |
|---|---|
| OS | Ubuntu 22.04, x86_64 |
| Python | 3.12.7 |
| GPU (chỉ cần khi tái hiện đầy đủ) | NVIDIA RTX 3050 8GB (local) + Kaggle T4×2 (kernel) |
| RAM | ≥ 16 GB (bước nén LZMA cần ~9 GB) |
| Ổ đĩa | ~10 GB cho tái hiện nhanh; ~60 GB cho tái hiện đầy đủ |

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install lightgbm==4.7.0 scikit-learn==1.9.1 ijson==3.6.0 kaggle==2.2.4
```

## 2. Dữ liệu

```bash
# Dữ liệu gốc của BTC (câu hỏi + danh sách URL)
.venv/bin/huggingface-cli download AIGuruTinix/ViBioMIR query.parquet links_corpus.parquet --repo-type dataset --local-dir data/ViBioMIR
.venv/bin/huggingface-cli download AIGuruTinix/ViBioMIR links_corpus_vi_en_zh.parquet --repo-type dataset --local-dir data/ViBioMIR_new
```

Dữ liệu đội tự thu thập và kết quả trung gian nằm trên Google Drive: **[ĐIỀN LINK GOOGLE DRIVE]**

```bash
mkdir -p data/docs data/cache/kg_zhdocs
tar xf vi_docs.tar -C data/docs                    # 735k trang VI: data/docs/corpus-*.parquet
cp zh_docs.parquet data/cache/kg_zhdocs/           # 222k trang ZH
tar xf artifacts_final.tar                         # giải nén vào data/cache/… (tái hiện nhanh)
tar xf artifacts_features.tar                      # (tuỳ chọn) đặc trưng để chạy lại LTR
```

## 3. Mô hình (checkpoint)

Tất cả tải trực tiếp từ Hugging Face (tự tải khi chạy, hoặc `huggingface-cli download <id>`):

`GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1`, `BAAI/bge-m3`, `intfloat/multilingual-e5-large-instruct`, `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507` (bản 4-bit `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`).

Ở bước tái hiện nhanh, không cần GPU hay checkpoint fine-tune: điểm số của các mô hình đã nằm trong `artifacts_final.tar`.

## 4. Tái hiện nhanh bài nộp cuối (~1 giờ, CPU)

Tái tạo `submission.zip` giống bài private P7_v76b từ các kết quả trung gian:

```bash
P="--preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy"

# 4.1 Bài VI: trộn ứng viên theo thứ tự LTR, chọn số doc tối ưu F2, chọn đoạn (≈4 GB RAM)
.venv/bin/python scripts/fuse_dense.py $P \
  --dense data/cache/dense_gn_top300_v9.json --w-dense 2.0 \
  --dense2 data/cache/dense_bgem3_chunks_v9.json --w-dense2 1.0 \
  --dense3 data/cache/dense_chunks_top_v9.json --w-dense3 2.0 \
  --adaptive 2.6 --ef2 1.9 --ef2-range 30 280 --chunk-ratio 1.0 --chunk-max 240 \
  --order data/cache/ltrx7_out/ltr_order_oof34.json --chunk-chars 8000 --out data/cache/tmp/base_vi.zip

# 4.2 Trộn tài liệu ZH theo xác suất + cấu hình đoạn cho luật chấm mới (≈4 GB RAM)
.venv/bin/python scripts/merge_zh.py --base data/cache/tmp/base_vi.zip \
  --order data/cache/ltrx7_out/ltr_order_oof34 --zh data/cache/zh_probs7.parquet \
  --zh-scale 0.9 --zh-chars 1800 --vi-chars 5000 --chunk-max 250 --out data/cache/tmp/final_deflate.zip

# 4.3 Nén ZIP-LZMA để < 100 MB (≈9 GB RAM, 2–3 giờ)
R2AI_REORDER=1 R2AI_LZMA_DICT_MB=1024 R2AI_LZMA_HC4=1 \
  .venv/bin/python scripts/zip_lzma.py data/cache/tmp/final_deflate.zip submission.zip
```

`submission.zip` chứa đúng một tệp `submission.json`, có đủ 1200 câu, mỗi câu gồm `relevant_docs` (doc_id nguyên) và `relevant_chunks` (`doc_id`, `chunk_text` trích nguyên văn).
Nếu máy ít RAM, chạy 4.2 và 4.3 trên Kaggle CPU bằng kernel `kaggle_buildzip/` (xem đầu tệp).

## 5. Tái hiện đầy đủ từ đầu

Các bước theo thứ tự. Bước có ký hiệu **[K]** chạy trên Kaggle: đẩy bằng `kaggle kernels push -p <thư mục>`, dataset đầu vào ghi trong `kernel-metadata.json` của từng thư mục, tham số ghi ở đầu mỗi tệp.

| # | Bước | Lệnh / kernel | Đầu ra |
|---|---|---|---|
| 1 | Thu thập VI | `python -m src.crawl --lang vi --per-domain 8`; phần bị chặn IP: **[K]** `kaggle_crawl*/`; Pharmacity: `scripts/pharmacity_fetch.py` | `data/docs/corpus-*.parquet` |
| 2 | Thu thập ZH (bộ 235k) | **[K]** `kaggle_zhcrawl/s0..s7`; 39.net: `scripts/fetch_39m.py` | `zh_docs.parquet` |
| 3 | Query plan | `python -m src.query` (Qwen3-4B 4-bit) | `data/cache/query_plans.jsonl` |
| 4 | BM25 VI + chunk | `scripts/pipeline_v2.py predict --doc-max-vi 300 --doc-max-zh 0 --chunk-chars-vi 1600 --dump-preds data/cache/vi_preds_300.jsonl` | `vi_preds_300.jsonl` |
| 5 | GreenNode chấm đoạn BM25 | `scripts/rerank_vi.py score --kind embed --model GreenNode/... --top 200` | `rr_gn_top200.npy` |
| 6 | Dense VI cấp doc | `scripts/dense_vi.py encode` → `search --k 300` | `dense_gn/`, `dense_gn_top300.json` |
| 7 | Dense VI cấp đoạn | **[K]** `kaggle/dense_chunks.py` (GreenNode), **[K]** `kaggle_bgem3/` (bge-m3) | `dense_chunks_top.json`, `dense_bgem3_chunks.json` |
| 8 | e5 | `colab/encode_e5.py` | `e5/` |
| 9 | Cross-encoder | `colab/rerank_bge.py`, `colab/rerank_win.py` (zero-shot); **[K]** `kaggle_ft/`, `kaggle_cewin_*/`, `kaggle_gnft_*/` (fine-tune 5 fold); **[K]** `kaggle_savemodels/` (model toàn nhãn) | `rr_bge_all`, `rr_win`, `ce_ft`, `ce_ftwin`, `gnft` |
| 10 | Gộp doc mới thu thập thêm | `scripts/newdocs_prep.py` → **[K]** `kaggle_newdocs/` → `scripts/newdocs_filter.py` → `scripts/newdocs_merge.py --ver N` (lặp v3→v9) | các cache `*_v9` |
| 11 | kNN | `scripts/doc_knn.py` → `scripts/knn_expand.py` → **[K]** `kaggle_knnpairs/` → `scripts/knn_merge.py` | `knn_exp.json` |
| 12 | CE toàn nhãn | `scripts/ce2_prep.py` → **[K]** `kaggle_ce2b/` | `ce2b_v9.parquet` |
| 13 | LTR VI | `scripts/ltr.py <đặc trưng v9> --rr5 ce2b_v9 --xq data/cache/e5_v9 --stages 1 --seeds 5 --rounds 700` (hoặc **[K]** `kaggle_ltrx/`) | `ltr_order_oof34.json` (+`_probs`) |
| 14 | Đặc trưng ZH | **[K]** `kaggle_zhfeat/` (bge-m3 dense, bge-reranker, CE toàn nhãn) | `zh_pairs.parquet`, `zh_emb.npy` |
| 15 | Dịch câu hỏi | **[K]** `kaggle_qtrans/` (Qwen2.5-7B) → `scripts/zh_qfeat.py` | `q_zh.json`, `zh_qfeat.parquet` |
| 16 | CE VI↔ZH | `scripts/cezh_prep.py` → **[K]** `kaggle_cezh/` | `cezh_v1.parquet` |
| 17 | LTR ZH | `scripts/zh_ltr3.py --feat data/cache/zhfeat3_out --extra data/cache/cezh_v1.parquet --qfeat data/cache/zh_qfeat.parquet --out data/cache/zh_probs7.parquet` | `zh_probs7.parquet` |
| 18 | Tạo bài nộp | mục 4 | `submission.zip` |

Các bước 9, 12, 13, 16, 17 cần nhãn liên quan (câu, tài liệu). Nhãn được rút từ tệp chấm chi tiết (`details.jsonl`) do hệ thống của BTC trả về trong vòng public (xem *Khai báo nguồn nhãn* trong `docs/THUYET_MINH.md`). BTC đã thu hồi các tệp này, nên đội có thể cung cấp bản lưu cho BTC khi được yêu cầu.

## 6. Nộp bài

```bash
# vòng private (phase 45); --relay: tải lên qua Kaggle khi mạng tới BTC chậm
.venv/bin/python scripts/submit_api.py submission.zip --phase 45 --relay --wait
```
Tài khoản BTC đặt trong `.r2ai_cred` (`R2AI_USER=…`, `R2AI_PASS=…`; không đưa vào Git).
