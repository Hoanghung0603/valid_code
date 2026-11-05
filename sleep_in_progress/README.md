# Hệ thống truy hồi y khoa R2AI 2026

Mã nguồn cho nhóm sleep_in_progress, mô tả các bài nộp private vi53, vi56 và nhóm mở rộng vi57–vi59. Tài liệu phương pháp, dữ liệu, prompt, mô hình và kết quả nằm trong tệp PDF thuyết minh gửi kèm email, không đóng trong gói mã nguồn này. Luồng chính: thu thập snapshot → làm sạch và chỉ mục → BM25/dense/HyDE và mở rộng truy vấn → rerank → RRF → doc/chunk nguyên văn (vi48 hoặc CX) → ZIP.

**Trạng thái replay:** Đã tái dựng trên CPU từ điểm số đã đóng băng, không chạy lại suy luận GPU. Bài vi48 có SHA ZIP khớp bản lịch sử `0bfc09df…281e`. Hai bài vi53 và vi56 có SHA ZIP khớp bản đã nộp (`4df40fb4…9bd9`, `1a8cae50…92fc`), SHA JSON cũng khớp, 1.200/1.200 câu khớp cả thứ tự doc và chunk; thời gian wall khoảng 699 s và 604 s. Bài vi47 dựng lại bằng bản sao launcher chỉ đổi đường dẫn; `q3e_new.json` của vi56 được chọn lại bằng `vi56.local_select` nên không phải bằng chứng SHA lịch sử. 58 test liên quan đạt. Chi tiết: `REPLAY.md`.

## 1. Môi trường

- Dựng chỉ mục vi36 bằng CPU từ shard corpus VI đúng schema; đầu ra lịch sử khoảng 5,82 GB. Chuẩn bị không gian cho snapshot, cache, part trung gian và ZIP; khuyến nghị ít nhất 32 GB RAM. Đỉnh RAM/đĩa và thời gian dựng chưa đo.
- CPU: Python 3.10.6 đã được ghi nhận ở một môi trường xử lý; Python 3.13.3 ở môi trường khác. Kiểm thử môi trường được chọn trước khi chạy đầy đủ.
- GPU: Python 3.11; PyTorch/Transformers hoặc vLLM theo từng lớp image trong `deploy/modal_*.py`, có GPU NVIDIA phù hợp và driver/CUDA tương thích. Bước 8B/14B cần GPU phù hợp bộ nhớ model và runtime; cấp phát theo cấu hình launcher và thực hiện benchmark trước khi chạy toàn bộ.
- `requirements/cpu-py310.txt` và `requirements/local-py313.txt` là ảnh chụp môi trường quan sát ngày 04/11, **không phải lockfile lịch sử đã chứng minh replay**; `requirements/modal-images.md` ghi các image. Mã crawl có `requirements-crawl.txt` với phiên bản ghim. Ví dụ PowerShell với Python 3.10, sau khi kiểm tính sẵn có của các gói:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-crawl.txt
python -m pip install -r requirements/cpu-py310.txt
```

`requirements-crawl.txt` không phải toàn bộ phụ thuộc để chạy pipeline. Các image GPU dùng Python 3.11; nhánh BGE có torch 2.6.0, transformers 4.51.3, numpy 1.26.4, zstandard 0.23.0; nhánh Qwen rerank dùng vLLM 0.15.1; nhánh embedding dùng torch 2.9.1 và transformers 4.57.6. Phiên bản driver, CUDA và toàn bộ phụ thuộc bắc cầu lịch sử chưa được ghim/đối soát; sử dụng manifest môi trường khi được cung cấp.

## 2. Dữ liệu, checkpoint và xác minh

1. Tải query và danh mục URL từ [ViBioMIR](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR) hoặc gói `r2ai_queries_links`. Chuẩn bị snapshot sạch `r2ai_corpus_vi_clean`, đầu ra HyDE/rewrite `r2ai_llm_outputs` và cache `r2ai_replay_inputs` (input vi59 nằm trong `r2ai_replay_inputs_vi59`). Liên kết của các gói nằm ở mục 1.2 của thuyết minh; các gói trên Google Drive có quyền **“Bất kỳ ai có đường liên kết đều xem được”**. Đối chiếu manifest SHA-256 trước khi dùng.
2. Tải checkpoint từ các model card chính thức trong mục 1.3 của thuyết minh. Qwen3-Embedding-8B dùng revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, Qwen3-14B dùng `40c069824f4251a91eefaf281ebe4c544efd3e18`; các revision Qwen reranker và BGE reranker **được ghim cho vi56** tại `deploy/modal_vi56.py`. Một số lớp lịch sử không ghim; cần giữ cache/manifest gốc.
3. Snapshot tài liệu được thu thập có giới hạn theo website; tái crawl phải tuân theo quyền truy cập, giảm tốc hoặc dừng khi bị chặn/captcha, không vượt anti-bot. Dùng snapshot đã bàn giao để tái lập thay vì coi web hiện tại là bất biến.
4. Các bài kết quả nằm trong gói `r2ai_submissions`, kèm `SHA256SUMS.txt`. Tham khảo SHA-256 trong thuyết minh, ví dụ:

```powershell
Get-FileHash .\submission_vi53.zip -Algorithm SHA256
Get-FileHash .\submission_vi56.zip -Algorithm SHA256
```

## 3. Dựng chỉ mục và truy hồi

Chạy các lệnh dưới đây từ thư mục gốc mã nguồn, trong môi trường riêng; thay các tham số trong dấu góc nhọn bằng đường dẫn tương ứng.

### 3.1 Thu thập và làm sạch

`scripts/crawl.py --help`, `scripts/extract.py` và `scripts/verify_shards.py` mô tả các chế độ thu thập/trích xuất và kiểm shard. Gói snapshot đã được chốt phải bao gồm danh sách shard, manifest, cấu hình làm sạch và ID bị loại. Nhóm đã loại 16.079 ID nghi nhiễm trong danh sách kiểm tra, nhập lại các bản hợp lệ và bổ sung top-up; `scripts/vi28.py`–`vi30.py` và `vi36.py` thể hiện các bước lịch sử. **Không nhập chồng** snapshot tổng hợp với các gói cũ đã được thay thế. Xác nhận ID hợp lệ và xử lý trùng liên shard trước khi index.

### 3.2 BM25 — CPU

Ví dụ lệnh có thật của `bigindex.py`, dùng shard đã được chuẩn bị, ở thư mục đầu ra mới:

```powershell
python scripts/bigindex.py build --shards <thu-muc-shard-sach> --out <thu-muc-index-moi> --lang vi --nbits 26 --workers 3 --budget 20000000
python scripts/bigindex.py query --index <thu-muc-index-moi> --q "câu hỏi y khoa" -k 200
python scripts/check_index.py --help
```

Lệnh dùng các giá trị `workers=3`, `budget=20000000` được `vi36.py` áp dụng; `bigindex.py` tự fit boilerplate nếu chưa có `boilerplate.json` trong thư mục đầu ra. Để tái tạo chỉ mục vi36 lịch sử, chuẩn bị đúng `boilerplate.json`, thứ tự shard, ID thay thế và snapshot theo manifest; `vi30.py`/`vi36.py` thực hiện thêm bước làm sạch liên shard. **Gói `r2ai_corpus_vi_clean` hiện thiếu `status`/`host` và đã gộp title vào text, nên không đưa trực tiếp vào lệnh trên:** cần shard đúng schema `bigindex.shard_docs` và đối chiếu dữ liệu lịch sử. Chỉ mục đầu ra lịch sử khoảng 5,82 GB; thời gian và đỉnh RAM/đĩa chưa đo. BM25 dùng CRC32 2^26, k1=1,2, b=0,75.

### 3.3 Dense, HyDE và mở rộng — GPU

`scripts/embed_run.py`, `embed_all.py`, `dense_topk.py` tạo vector bge-m3 và top-k; `scripts/rewrite_queries.py` sinh HyDE qua runtime tương thích; `scripts/vi41.py` cùng `deploy/modal_vi41.py`, `modal_vi41_search.py` tạo rewrite và tìm ứng viên mở rộng. Xem tham số thực bằng:

```powershell
python scripts/dense_topk.py --help
python scripts/rewrite_queries.py --help
python scripts/vi41.py --help
```

Ví dụ tìm top-200 trên GPU khi đã có embeddings/doc-export và query JSONL đúng schema:

```powershell
python scripts/dense_topk.py --emb <thu-muc-vector> --queries <queries.jsonl> --docs <docs.jsonl.zst> --out <dense-top200.json> --model BAAI/bge-m3 --top 200 --device cuda
```

vi41 dựa trên đầu ra vi36 và nhiều cache, không chỉ query/index. Không có một lệnh trên khẳng định dựng chính xác pool tích lũy. Các lệnh rewrite/retrieval thực tế phụ thuộc các part, manifest và nguồn dữ liệu có trong các ghi chú vận hành nội bộ (không kèm gói); không chạy lại với đường dẫn mặc định khi chưa chuẩn bị chúng.

### 3.4 Rerank và chunk — GPU, sau đó CPU

`deploy/modal_runner.py`/`scripts/vi37.py` tạo và hợp nhất BGE docwin/Qwen doc; `deploy/modal_vi41_score.py` chấm ứng viên mở rộng; `deploy/modal_vi45.py` chấm chunk, `deploy/modal_vi50.py` chấm d8w, `deploy/modal_vi51.py` tính vector Qwen embedding. Input phải khớp manifest và từng bước merge được kiểm trước bước kế tiếp; một số launcher yêu cầu hạ tầng GPU riêng, không thể chạy bằng ví dụ CPU đơn giản. Xem `--help`/mã của launcher và các gói dữ liệu đi kèm; cấu hình tài khoản, bí mật và đường dẫn phải được người chạy cung cấp qua môi trường an toàn, không hardcode.

Ước lượng từ các lượt đã chạy, **GPU-giờ đã ghi nhận chứ không phải dự báo toàn bộ**: vi37 0,90; mở rộng vi41 1,12; vi45 1,96; vi50 1,68; vi51 cả 0,6B và 8B 4,03. Chưa bao gồm hết các bước gốc, thời gian chờ, lỗi hoặc chi phí phát sinh. Từ các score, `vi42.py`, `vi44.py`, `vi45.py`, `vi47.py`, `vi48.py` dựng chunk theo công thức trong thuyết minh. Ví dụ giao diện vi48 — chỉ dùng khi mọi tệp input đã khớp SHA:

```powershell
python scripts/vi48.py build --vi45-dir <thu-muc-vi45-day-du> --vi45 <submission_vi45.zip> --vi47 <submission_vi47.zip> --queries <queries.jsonl> --out <thu-muc-vi48-moi>
```

### 3.5 Mở rộng pool và chunk (vi56, vi57–vi59) — GPU và CPU

`scripts/vi55.py`/`deploy/modal_vi55.py` tạo vector toàn kho và top-k; `scripts/vi56.py`/`deploy/modal_vi56.py` chọn top-100 ngoài pool và chấm phần bổ sung. Các bước yêu cầu input part, model weights và manifest đúng SHA; theo báo cáo, vi55 dùng khoảng 3,57 GPU-giờ thành công và vi56 khoảng 0,58 GPU-giờ, chưa gồm mọi lỗi/preemption/chi phí khởi tạo. Nhánh vi56 giữ chunk/count vi48 và dùng fallback hạng cuối cho dense/HyDE lịch sử thiếu cosine. Các bài mở rộng private gồm vi57 (`doc vi53 + CX`), vi58 (`doc vi56 + CX`) và vi59 (mở rộng ứng viên ra top-200 Qwen3-Embedding-8B toàn kho ngoài pool vi41c, chấm bổ sung hạng 101–200 trên H100 rồi ghép RRF + CX); chi tiết quy tắc CX, thông số đóng gói, chi phí và mã bài nộp (`#7934` vi53, `#7935` vi56, `#7964` vi57, `#7969` vi58, vi59 qua email) được trình bày tại mục 1.1.4–1.1.5 của thuyết minh.

## 4. Replay CPU từ cache và tạo ZIP

Hướng dẫn replay chi tiết nằm trong `REPLAY.md` và `PACKAGE.md` ở gốc gói `r2ai_replay_inputs`; phần vi59 xem `PACKAGE.md` của gói `r2ai_replay_inputs_vi59`. Trong mã bàn giao, đường dẫn của máy cũ đã thay bằng `<WORKDIR>` (thư mục gốc chứa `r2ai/` và `r2ai_deliver/`) và `<HIST_TMP>` (thư mục chứa input lịch sử). Chuỗi Python bắt đầu bằng hai placeholder này đọc giá trị từ biến môi trường `WORKDIR` và `HIST_TMP`. Các file `.bat`, `.ps1`, `.sh` dùng biến môi trường cùng tên.

`scripts/vi52.py` dựng vi53 với các đối số `--name vi53 --cosine <cosines_vi51b.json> --reference51 <submission_vi51b.zip>`; mặc định code còn đọc các input dưới cây `<HIST_TMP>/` (thư mục tạm chứa input lịch sử), gồm `vi50/fusion_context.json.zst`, `vi50/d8w.json`, `vi48/submission_vi48.zip`, `vi50/submission_vi50.zip`, `vi45/bad_ids.json`, `vi45/new_ids.json` và parquet links. Đây là **đường dẫn cứng lịch sử**, không phải cấu hình di động: cần hướng dẫn replay/phiên bản mã bàn giao đã điều chỉnh đường dẫn hoặc khôi phục đủ cây file theo cách an toàn được chấp thuận. Chỉ thay `--out` không đổi đường dẫn input. Nếu bật sanity đối chứng vi52 thì còn cần `--reference52` và `--sanity-cosine` tương ứng.

```powershell
python scripts/vi52.py build --name vi53 --cosine <cosines_vi51b.json> --reference51 <submission_vi51b.zip> --out <thu-muc-vi53-moi>
```

`scripts/vi56.py build --out <thu-muc-vi56-moi>` cũng đọc các cache cũ theo đường dẫn cứng trên; `--out` phải chứa `prepare_report.json`, `new_ids.json`, `q3e_new.json`, `expected_bge.json`, `expected_qwen.json`, `input_audit.json`, các `modal_input` và **đầy đủ các score part cùng completion metadata khớp SHA**. Build kiểm sanity vi53 trước khi mở pool:

```powershell
python scripts/vi56.py sanity --out <thu-muc-vi56-moi>
python scripts/vi56.py build --out <thu-muc-vi56-moi>
```

Hai lệnh build có guard chống ghi đè và cần cả bộ đầu vào/sanity hợp lệ; dùng output mới và sao lưu tệp nguồn. Đừng thay guard hoặc tự tạo score khi dữ liệu thiếu. Tốc độ lịch sử: vi53 build CPU khoảng 856 giây; vi48 khoảng 1.010 giây, tùy máy và I/O.

## 5. Kiểm tra kết quả

Khi đã giải nén ZIP ra **thư mục kiểm tra mới**:

```powershell
python scripts/evaluate.py validate <thu-muc-kiem-tra>/submission.json
python -m zipfile -t <duong-dan>/submission_vi53.zip
```

Kiểm tra thêm ZIP chỉ có `submission.json`, đủ ID 1–1200, không trùng doc hay cặp `(doc_id, chunk_text)`, mọi chunk thuộc doclist, doc thuộc corpus chuẩn, chunk nguyên văn và kích thước ≤99.000.000 byte. `scripts/vi52.py check-zip --name vi53 --out <thu-muc-output>` và `scripts/vi56.py check-zip --out <thu-muc-output>` cần `submission_vi*.json` cùng ZIP; các builder thường xóa JSON trung gian sau khi kiểm, nên phải giải nén lại vào vị trí kiểm tra riêng nếu chạy lệnh này. So SHA với bảng trong thuyết minh; nếu khác, phân biệt khác nội dung với khác thứ tự/nén/phiên bản.

## 6. Giới hạn

Các cache được tạo tăng dần; chỉ mục và điểm của từng tầng cần khớp snapshot/manifest. Tái chạy GPU có thể thay đổi kết quả số học, hạng hòa hoặc bytes ZIP; nội dung website có thể thay đổi. Nhánh CX (vi57, vi58) và vi59 chưa có điểm public; replay chỉ chạy trên CPU từ điểm số đóng băng, không chạy lại suy luận GPU. Các giới hạn về revision, GGUF, môi trường và replay được tóm tắt tại mục 1.6 của thuyết minh.
