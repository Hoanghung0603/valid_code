# ViBioMIR (ROAD TO AI 2026, Stage 3): mã nguồn của bài nộp S11

Hệ thống truy hồi tài liệu y sinh: câu hỏi tiếng Việt → danh sách tài liệu (vi, zh) và các đoạn văn nguyên văn trả lời câu hỏi.

**Bài nộp được mô tả:** `sub-long-kproxy-nproxy-v1` (S11).

| | Doc F2 | Doc P / R | Chunk F2 | Chunk P / R | Final |
|---|---|---|---|---|---|
| Public leaderboard (28/10/2026) | 0,4094 | 0,2374 / 0,5768 | 0,3376 | 0,2193 / 0,4410 | **0,3735** |

Bài này cũng là lượt nộp đầu tiên của đội ở vòng private.

**Tài liệu:**
- Thuyết minh sản phẩm: tổng quan hệ thống, dữ liệu, mô hình ([docs/THUYET-MINH.md](docs/THUYET-MINH.md)).
- Tài liệu này: cài đặt, tải dữ liệu và chạy lại.

## 1. Gói nộp gồm những gì

| Thành phần | Nội dung | Dung lượng | Link |
|---|---|---|---|
| Mã nguồn | Thư mục này: package `medret`, các script của pipeline, config, test | < 1 MB | [ma-nguon.zip](https://drive.google.com/file/d/1-u843C2fn3jhaAwWxtdlgaySVURXitDC/view?usp=sharing) |
| Dữ liệu A | Đầu vào của bước ghép cuối, đủ để tái lập S11 trong khoảng 20 phút trên CPU (mục 4) | 2,9 GB | [du-lieu-A.zip](https://drive.google.com/file/d/1I-w7_QmhkTSyslAHfA0WpJOTJfLV7ueI/view?usp=sharing) |
| Dữ liệu B | Dữ liệu trung gian để dựng lại các bước trước (mục 5) | 7,6 GB | cung cấp khi BTC yêu cầu |
| Dữ liệu C (tùy chọn) | HTML gốc đã crawl (11 GB); chỉ mục dense BGE-M3 (19 GB) và BM25 (1,7 GB). Có thể dựng lại từ B | 32 GB | cung cấp khi BTC yêu cầu |

Cả ba gói dữ liệu đều giải nén vào thư mục `data/` của repo, giữ nguyên cấu trúc con. Gói B không ghi đè file nào của gói A.

Chỉ mã nguồn và gói A được đăng qua link. Đội đã chuẩn bị sẵn gói B (`du-lieu-B.zip`, 8,0 GB) và gói C, và sẽ gửi ngay khi BTC yêu cầu; các bước ở mục 5 cần đến chúng.

## 2. Yêu cầu môi trường

| Việc | Máy | Ghi chú |
|---|---|---|
| Tái lập S11 từ dữ liệu A (mục 4) | CPU, RAM ≥ 8 GB (≥ 3 GB trống khi đóng gói), ổ trống ≥ 5 GB | Đã chạy lại ngày 03/11/2026 trên Windows 10, Python 3.11.16 |
| Parse, chunk, BM25, DuckDB (mục 5.2–5.3) | CPU, RAM ≥ 16 GB, ổ trống ≥ 80 GB | |
| Embedding BGE-M3, rerank Qwen3-Reranker (mục 5.3, 5.7) | GPU: Kaggle Notebook 2 × Tesla T4 | Script đẩy kernel và kéo kết quả về tự động |
| Rerank chunk bge-reranker-v2-m3 (mục 5.4) | API FPT AI Marketplace hoặc GPU | |

- **Python:** ≥ 3.11.
- **Phiên bản thư viện:** đúng như lúc dựng S11 ghi trong [requirements-lock.txt](requirements-lock.txt).
- **Hệ điều hành:** code chạy được trên cả Windows và Linux. Script `.sh` chạy bằng bash (Git Bash trên Windows).

## 3. Cài đặt và tải dữ liệu

```bash
# tải ma-nguon.zip (link ở mục 1), giải nén vào thư mục vibiomir-s11
cd vibiomir-s11
python -m venv .venv && source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt && pip install -e . --no-deps
# hoặc theo nhu cầu: pip install -e ".[dev]" (mục 4), thêm ".[index,ml,kaggle,crawl]" cho mục 5

medret download          # dữ liệu chính thức AIGuruTinix/ViBioMIR -> data/raw/ViBioMIR/ (query.parquet, links_corpus_vi_en_zh.parquet)
# Dữ liệu A (bắt buộc cho mục 4): tải du-lieu-A.zip (link ở mục 1), giải nén tại thư mục gốc repo (các file nằm dưới data/)
# Dữ liệu B (cho mục 5): đội gửi khi BTC yêu cầu, giải nén cùng chỗ
cp .env.example .env     # chỉ cần cho mục 5: khóa Kaggle, API FPT
pytest                   # test offline
```

Cấu trúc `data/` sau khi giải nén gói A:

```
data/
  kaggle/rerank/<run>/scores.parquet   điểm Qwen3-Reranker của 7 lượt chạy Kaggle (query_id, doc_id, src_rank, lang, score, has_text)
  runs/<bài>/<bài>.zip                 9 bài nộp trước S11: danh sách gốc, bài cho ngân sách chunk, các bài dùng để fit proxy
  runs/pool-vi-m1000/reranked.parquet  điểm bge-reranker-v2-m3 của 1.000 chunk vi mỗi câu hỏi (vị trí đoạn tốt nhất)
  indexes/meta.duckdb                  text của 272.196 doc mà bước ghép cuối đọc (bản rút gọn, scripts/analysis/slim_meta.py)
  parsed/documents/*/part-gloryvu-20261023-zh.parquet   text của 97.094 doc zh lấy từ snapshot crawl của đội
  external/gloryvu_20261023/corpus_annotated.parquet    quan hệ trùng URL http/https (doc song sinh)
  reference/sub-long-kproxy-nproxy-v1/  file zip đã nộp + proxy đã fit lúc đó, dùng để đối chiếu
```

## 4. Tái lập nhanh bài S11 (dữ liệu A, CPU, khoảng 20 phút)

```bash
bash scripts/reproduce_s11.sh
# -> data/runs/sub-long-kproxy-nproxy-v1/sub-long-kproxy-nproxy-v1.zip (khoảng 94,5 MB; nộp file này)
```

| Bước | Lệnh trong script | Kết quả mong đợi |
|---|---|---|
| 1. Gộp điểm rerank | `merge_scores.py`, `merge_long_scores.py` | `qwen3-0.6b-all` 720.000 cặp; `qwen3-4b-all-tw-long` 491.220 cặp, trong đó 180.000 cặp lấy điểm đọc dài |
| 2. Fit proxy rerank | `rerank_proxy.py fit --deep --long --no-loo` | theta = [−0,386; 0,257; 0,236; −0,002; −0,029; −0,311; −0,619; −0,875; −1,081] |
| 3. Chọn k | `rerank_proxy.py k --deep --long` | Doc F2 kỳ vọng 0,4470; k trung bình 151,3 |
| 4. Ghép bài | `fused_sub.py ...` | N trung bình 133,6; 153,6 chunk/query (19,5 zh); validate OK |
| 5. Đối chiếu | `compare_submissions.py` | `1200 vs 1200 queries; 0 differ: identical` |
| 6. Đóng gói | `allocate_sub.py pack --dict-mb 256` | zip LZMA, chứa đúng 1 file `submission.json` |

- **Kết quả kiểm chứng (03/11/2026):** bước 1–3 cho file trùng từng giá trị với bản lúc dựng S11. Bước 5 xác nhận cả 1.200 query trùng với file đã nộp, cả danh sách doc, thứ tự doc lẫn từng chunk.
- **Thứ tự query:** file JSON tạo ở bước 4 khác file đã nộp theo từng byte, vì bước đóng gói sắp lại thứ tự query để nén tốt hơn.
- **Máy ít RAM:** `DICT_MB=192 bash scripts/reproduce_s11.sh`, file sẽ to hơn khoảng 2–3%.

## 5. Dựng lại từ đầu

Thứ tự các bước giống sơ đồ ở mục 1 của [THUYET-MINH.md](docs/THUYET-MINH.md).

- **Kiểm chứng:** mỗi bước đã được chạy lại và đối chiếu với kết quả gốc; mức kiểm chứng ghi ở cột cuối các bảng. Riêng các bước đánh dấu "đóng băng" dùng kết quả đã lưu trong gói B.
- **Lưu ý:** các bước cần GPU, crawl hoặc API thì không cho ra kết quả trùng từng bit khi chạy lại, vì trang web thay đổi và tính toán fp16 trên GPU có sai số.
- **Tái lập đúng S11:** dùng kết quả trung gian trong gói A/B ở bước cuối cùng mà bạn không chạy lại.

### 5.1 Thu thập dữ liệu (crawl)

```bash
pip install -e ".[crawl]"
cd crawler
python scripts/split_crawl_sets.py --sets 3                 # chia URL của corpus cho 3 máy (mỗi máy một IP)
python scripts/crawl_set.py --todo build/crawl_sets/set_1.parquet --out ../data/crawl/pages   # chạy trên từng máy
python scripts/crawl_set.py --todo build/crawl_sets/set_1.parquet --out ../data/crawl/pages --status
cd ..
```

- **Đầu ra:** HTML thô vào `data/crawl/pages/<site>/part-*.parquet` (cột `id, url, final_url, status, label, detail, ctype, ms, fetched_at, html`).
- **Crawler** (`crawler/src/crawl/polite.py`):
  - tối đa 1 request/giây mỗi site;
  - tuân thủ robots.txt;
  - bị chặn thì nghỉ và lùi dần tới 30 phút;
  - không vượt captcha.
- **Chạy tiếp:** chạy lại cùng lệnh là tiếp tục từ chỗ dừng.
- **Snapshot crawl của đội:** dùng thêm (bản 12/10, 19/10, 23/10). Xem THUYET-MINH §2.

### 5.2 Parse, chunk, nhập snapshot

```bash
pip install -e ".[index]"
medret manifest                    # data/crawl/pages -> manifest.parquet, aliases.parquet (URL trùng sau chuẩn hóa)
medret templates                   # học boilerplate theo host trên một mẫu trang
medret parse                       # HTML -> data/parsed/documents (text toàn trang, lọc nhiễu theo template của host)
python scripts/analysis/import_team_snapshot.py                              # 41.240 doc vi chỉ snapshot 19/10 có
python scripts/analysis/import_team_snapshot.py --src gloryvu_20261023 --lang zh   # 97.094 doc zh từ snapshot 23/10
medret chunk                       # data/parsed/documents -> data/chunks (pack-v2, ~256 token, tokenizer BGE-M3)
```

Gói B đã có sẵn đầu ra của bước này: `data/parsed/documents`, `data/chunks`, `data/crawl/manifest.parquet`, `aliases.parquet`.

### 5.3 Chỉ mục

```bash
medret export-embed                                         # input BGE-M3 không trùng -> data/embed_inputs
python scripts/kaggle/launch_embed.py upload-dataset data/embed_inputs <tài_khoản>/medret-embed-inputs-v1
python scripts/kaggle/launch_embed.py run configs/kaggle/embed_corpus.json     # BGE-M3 trên Kaggle 2×T4, chạy tiếp được nếu bị ngắt
medret build-meta                                           # DuckDB: documents (có text), chunks, aliases
medret build-scope                                          # giới hạn chỉ mục trong corpus v2
medret build-bm25 --lang vi                                 # Tantivy, term = âm tiết + bigram
medret build-dense                                          # vector fp16 (8.148.135 chunk) + FAISS IVF8192,PQ64
python scripts/analysis/doc_vectors.py                      # vector mức doc: chunk đầu (head) và trung bình (mean)
```

- **Config Kaggle:** `embed_corpus*.json` (lượt chính cùng các lượt bổ sung `inc1`, `inc2`, `inc4`, `s1`).
- **Đổi tài khoản:** sửa slug `kernel` và `dataset_sources` sang tài khoản của bạn.

### 5.4 Pool chunk tiếng Việt (`pool-vi-m1000`)

```bash
medret run --name pool-vi-m1000 --channels bm25_vi,dense_vi --k-per-channel 600 --n-candidates 1200 \
           --top-m 1000 --k-max 200 --tau-doc 0
```

- **Cách làm:** BM25 vi và BGE-M3 dense lấy mỗi kênh 600 chunk, gộp RRF (k = 60) thành 1.200 ứng viên. bge-reranker-v2-m3 chấm 1.000 chunk đầu mỗi câu hỏi.
- **Model rerank:** gọi qua API FPT (`FPT_API_BASE`, `FPT_API_KEY`). FPT chỉ host model mở, không có prompt.
- **Dịch câu hỏi:** run này chỉ có kênh vi nên không dịch. Lúc dựng S11, pipeline có dịch câu hỏi sang tiếng Trung để chấm khoảng 61k chunk zh; các điểm đó không đi vào S11 (THUYET-MINH §3.3). Chạy lại bây giờ thì các chunk zh đó được chấm bằng câu hỏi tiếng Việt.
- **Gói A/B:** có sẵn `reranked.parquet` và `candidates.parquet` của run gốc.

### 5.5 Các kênh mức doc, proxy g và danh sách vi

```bash
python scripts/analysis/retrieval_bench.py channels         # data/dev/bench/{chunk,head_q,mean_q,...}.parquet (top-300)
python scripts/analysis/whole_page.py channels              # wp_{dense,sparse,hybrid}_q từ vector whole-page của đội
python scripts/analysis/gold_proxy.py features              # tập ứng viên + 6 đặc trưng mỗi cặp (query, doc)
python scripts/analysis/gold_proxy.py fit --no-loo          # proxy g -> data/dev/proxy_pairs_scored.parquet
python scripts/analysis/allocate_sub.py build --name sub-vi-rrfwp-win2x10-v1 --mult 1.8 --adaptive-k --adaptive-n \
       --order rrf:wp_hybrid_q --scored data/dev/proxy_v6_6f/proxy_pairs_scored.parquet --windows 10
```

| Bước | Trạng thái |
|---|---|
| Các kênh `data/dev/bench` | Đóng băng, có trong gói B |
| `proxy_v6_6f` | Đóng băng, có trong gói B. Fit lại cần 26 bài nộp cũ trong `LB_SETS` của `gold_proxy.py` (có thể cung cấp thêm) |
| `allocate_sub.py build` từ gói B | Đã chạy lại ngày 03/11: 1.200/1.200 query trùng bài đã nộp |

- **Vector whole-page:** do teammate tính, code ở `teammate/page_embeddings/`.

### 5.6 Thêm doc tiếng Trung và bài gốc

```bash
python scripts/analysis/probe_channel.py --zip data/external/teammate_probes/probe_recall_vi1000_en300_zh300.zip \
       --start 1300 --end 1600 --name tm_zh_q                   # thứ hạng zh: BGE-M3 dense mức chunk của teammate
python scripts/analysis/add_lang_docs.py --base sub-vi-rrfwp-win2x10-v1 --channel tm_zh_q --k 30 --chunks 0 \
       --name sub-vi-rrfwp-zh30docs-v1                          # danh sách gốc của S11: + 30 doc zh mỗi câu hỏi
python scripts/analysis/add_lang_docs.py --base sub-vi-rrfwp-win2x10-v1 --channel tm_zh_q --k 30 --chunks 25 \
       --max-chars 2000 --extra-docs 'data/parsed/documents/*/part-gloryvu-20261023-zh.parquet' \
       --name sub-vi-rrfwp-zh30c25snap-v1                       # ngân sách chunk của S11
```

Cả ba lệnh đã chạy lại ngày 03/11 và trùng kết quả gốc.

### 5.7 Rerank doc bằng Qwen3-Reranker trên Kaggle

```bash
python scripts/kaggle/launch_rerank.py make-pairs --out data/kaggle/rerank_pairs                 # pool đầu
python scripts/kaggle/launch_rerank.py make-pairs --vi-top 400 --zh-top 200 \
       --exclude data/kaggle/rerank_pairs --out data/kaggle/rerank_pairs_deep                   # pool sâu
python scripts/kaggle/launch_rerank.py upload <tài_khoản>/medret-rerank-pairs --src data/kaggle/rerank_pairs
python scripts/kaggle/launch_rerank.py run configs/kaggle/rerank_qwen3-0.6b.json                # đẩy kernel, chờ, kéo kết quả, gộp shard
# sau khi có điểm 0.6B của pool sâu và điểm 4B của pool đầu:
python scripts/kaggle/make_score_pairs.py deep4b --out data/kaggle/rerank_pairs_deep4b
python scripts/kaggle/make_score_pairs.py twins --out data/kaggle/rerank_pairs_twins
# sau khi có qwen3-4b-all-tw (scripts/analysis/merge_scores.py):
python scripts/kaggle/make_long_pairs.py --top 150 --chars 4000 --out data/kaggle/rerank_pairs_long
```

| Lượt (`configs/kaggle/`) | Cặp | Model, độ dài | Đầu ra `data/kaggle/rerank/` | Thời gian (2×T4) |
|---|---|---|---|---|
| `rerank_qwen3-0.6b` | pool đầu: vi top-200 + zh top-100 mỗi câu hỏi (360k) | 0.6B; 1.600 ký tự, 512 token | `qwen3-0.6b` | 4,7 giờ |
| `rerank_qwen3-4b`, `-4b-tail-p1/-p2` | pool đầu (180k + 180k) | 4B; 1.600 ký tự, 512 token | `qwen3-4b`, `qwen3-4b-tail` | 8,9 giờ + 2 × 4 giờ |
| `rerank_qwen3-0.6b-deep` | pool sâu: vi hạng 200–399 + zh 100–199 (360k) | 0.6B | `qwen3-0.6b-deep` | 4,3 giờ |
| `rerank_qwen3-4b-deep` | top-75 mỗi câu hỏi của pool sâu theo 0.6B (90k) | 4B | `qwen3-4b-deep` | 4,6 giờ |
| `rerank_qwen3-4b-twins` | doc zh không có text, đọc qua bản http/https trùng (81k) | 4B | `qwen3-4b-twins` | 3,4 giờ |
| `rerank_qwen3-4b-long-{a..e}` | top-150 mỗi câu hỏi theo thứ tự của S10 (180k) | 4B; 4.000 ký tự, 1.024 token | `qwen3-4b-long` | 2–7 giờ mỗi kernel |

- **Kiểm chứng:** bốn bộ cặp (pool đầu, pool sâu, deep4b, twins) đã dựng lại ngày 03/11 và trùng từng dòng với bộ đã upload lên Kaggle.
- **Text của trang:** kernel đọc từ snapshot của đội (`vibiomir-crawl-snapshot`, có trong gói B ở `data/external/gloryvu_20261023`). Các bộ twins và long tự mang text theo (`texts-*.parquet`).
- **Đổi tài khoản:** sửa `kernel` và `dataset_sources` trong config sang tài khoản của bạn. Dataset luôn được tạo ở chế độ private.

### 5.8 Ghép cuối

Như mục 4 (`bash scripts/reproduce_s11.sh`).

## 6. Cấu trúc mã nguồn

Bản này chỉ gồm code thuộc luồng của S11. Các thí nghiệm không dùng cho bài nộp đã được bỏ ra.

```
src/medret/            package chính (CLI `medret`)
  config.py            config có kiểu, đọc configs/default.yaml (đường dẫn tương đối theo thư mục gốc repo)
  data/                tải dataset, chuẩn hóa URL
  crawl/manifest.py    chỉ mục các shard HTML, URL trùng
  parse/               HTML -> text toàn trang (resiliparse), lọc nhiễu theo template từng host
  chunk/               chia chunk pack-v2 (offset start/end trên text gốc)
  index/               input embedding, DuckDB, Tantivy BM25, FAISS + memmap fp16, phạm vi corpus v2
  retrieve/            truy hồi đa kênh + RRF, mã hóa câu hỏi BGE-M3 (translate.py: chặn model chỉ dùng trong dev)
  rerank/fpt_api.py    bge-reranker-v2-m3 qua API, có cache
  select/              chọn doc, cửa sổ chunk nguyên văn
  submit/              ghi, validate, đọc theo luồng và đóng gói file nộp (LZMA)
  eval/                metric local (LCS, F2 macro) để kiểm tra
  pipeline.py, cli.py  `medret run` và các lệnh
scripts/
  analysis/            các bước mức doc: kênh, proxy g, danh sách vi, thêm zh, proxy rerank, thứ tự cuối, gộp điểm, slim_meta
  kaggle/              launcher và runner trên Kaggle (embedding, rerank), tạo các bộ cặp
  reproduce_s11.sh     bước ghép cuối (mục 4); compare_submissions.py: đối chiếu 2 file nộp
  run_limited.sh       chạy job nặng trên laptop (giới hạn luồng, tự dừng khi thiếu RAM)
configs/               default.yaml, laptop-lowmem.yaml, configs/kaggle/*.json (đúng các lượt đã chạy)
crawler/               crawler lịch sự nhiều máy của đội (nhánh feat/crawl-pipeline, commit f4fe98e)
teammate/              code nhúng vector whole-page của teammate (nhánh vinh-dev, commit 2d2f623)
tests/                 test offline (pytest)
```

## 7. Tuân thủ luật

- **Model:** chỉ dùng model mở, ≤ 15B tham số, phát hành trước 01/08/2026: BGE-M3, bge-reranker-v2-m3, Qwen3-Reranker-0.6B và -4B. Không dùng LLM sinh văn bản và không dùng API của model đóng ở bất kỳ bước nào của bài nộp. Chi tiết ở THUYET-MINH §3.
- **Chunk:** `chunk_text` luôn là một đoạn cắt nguyên văn `text[lo:hi]` từ text của doc, không dịch hay viết lại.
- **Dữ liệu ngoài:** được trích dẫn đủ ở THUYET-MINH §2.1, theo quy định "dữ liệu bên ngoài phải trích dẫn rõ nguồn gốc" (overview của đề).
  - **Bản lưu công khai:** với các URL của corpus mà đội không tải trực tiếp được (bị chặn, lỗi 5xx, link chết), text lấy từ bản lưu của **chính URL đó** trên Internet Archive Wayback Machine (73.023 trang của corpus v2) và Common Crawl (36.651 trang).
  - **Không dùng thêm:** không dùng nguồn nào ngoài các URL của corpus. Không dùng MedKB hay bất kỳ công cụ tìm kiếm nào để tra câu hỏi thi.
