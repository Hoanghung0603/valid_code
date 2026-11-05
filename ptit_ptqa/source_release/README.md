# R2AI Stage 3 — Exp143 technical release

Release: `stage3-exp143-20261103`  
Phạm vi: truy hồi tài liệu và đoạn bằng chứng y khoa VI/EN/ZH; **không sinh chẩn đoán**.  
Control công khai theo evaluator hai chiều mới: Final `0.3190`, Docs F2 `0.3731`, Chunks F2 `0.2648`. Điểm private chưa được công bố.

## 1. Nội dung bộ bàn giao

- `technical_report.pdf`: thuyết minh theo đúng năm mục email BTC.
- `presentation.pptx` và `presentation.pdf`: deck pitching 14 trang chính + 2 phụ lục kỹ thuật.
- `source_release.zip`: code/config/test/UI tối thiểu để verify và replay.
- `artifact_manifest.json`: tên, kích thước, SHA256 và dependency của từng gói.
- `model_registry.json`: model ID, immutable revision, vai trò và nguồn tải.
- `access_checklist.md`: checklist link Drive và quyền truy cập.
- `email_draft.md`: email nộp hồ sơ, còn một trường link cần người dùng điền sau khi upload.

Data, index và replay cache không nhét vào source ZIP. Chúng được chia sẻ riêng vì dung lượng lớn và phải đối chiếu SHA256 trong manifest. `source_release.zip` gồm mã Python, notebook, cấu hình, UI và tests của workspace tại thời điểm đóng gói; không chứa dữ liệu, model weights hoặc secret.

### Lấy và giải nén dữ liệu bàn giao

Tải **tất cả** volume `active_index.7z.001`–`.008`, `text_snapshot.7z.001`–`.002` và `qwen_secondary_index.7z.*` từ thư mục chia sẻ của đội. Giữ các volume cùng tên trong một thư mục. Đối chiếu SHA256 từng volume theo `artifact_manifest.json`; thiếu một volume thì không giải nén. Dùng 7-Zip 25.x hoặc tương đương:

```powershell
7z t active_index.7z.001
7z t text_snapshot.7z.001
7z t qwen_secondary_index.7z.001
7z x active_index.7z.001 -o. -y
7z x text_snapshot.7z.001 -o. -y
7z x qwen_secondary_index.7z.001 -o. -y
```

Giải nén từ thư mục gốc project để giữ đường dẫn `data/indices/new_dataset/index`, `data/indices/new_dataset/qwen3-0.6b/index`, `data/embedded_samples.parquet`, `data/raw/btc_provided/query.parquet` và `links_corpus_vi_en_zh.parquet`. Tải thêm `submission_private01_exp143_replay.zip` đúng SHA256 trong manifest. `replay.sqlite` là cache UI tùy chọn; có thể tái tạo bằng `build-cache`. Không dùng các archive index backup hay model cache.

Để dựng lại Exp143 từ output trung gian (không dùng ZIP private làm input), tải `exp143_rebuild_inputs.7z` và giải nén từ project root. Gói này gồm Exp142 control, P2A source windows và BGE reranker scores:

```powershell
7z t exp143_rebuild_inputs.7z
7z x exp143_rebuild_inputs.7z -o. -y
python -m scripts.build_final_sprint_exp143_short_spans
```

So sánh nội dung `submission.json` của ZIP mới với reference Exp143, không chỉ so ZIP hash vì container nén có thể khác. Đây là tái dựng từ cache inference đã lưu. Full fresh embedding/rerank từ raw corpus cần checkpoint trong `model_registry.json` và GPU; clean-room chưa xác nhận toàn bộ đường fresh inference.

### Private02–05, nếu được nộp

Các biến thể này là thí nghiệm *append-only* sau release Exp143, chưa có điểm private. Chỉ coi là bài đã nộp khi có receipt portal. Tải thêm `private_variants_inputs.7z` và bốn ZIP/manifest tương ứng nếu đội đưa chúng vào hồ sơ. Gói input chứa QuerySpec trạng thái/ID và BM25 candidate pool để chạy `scripts/build_private_bm25_additions.py`; Private03 dùng lại P2A cache trong `exp143_rebuild_inputs.7z`; Private04 kết hợp kết quả Private02/03 qua `scripts/combine_private_treatments.py`. `scripts/audit_private_four.py` kiểm tra coverage, official IDs và append-only. Các script này có trong source ZIP, nhưng không phải thành phần đã được chứng minh tăng điểm BTC.

## 2. Ba chế độ tái hiện

### Replay — đã kiểm chứng, không cần GPU

Replay chỉ hợp lệ khi query và prediction fingerprint khớp release. Đây là đường dùng cho UI review và tái tạo đúng submission đã nộp; không được mô tả là fresh inference.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-release.txt
python -m src.release.cli --config configs/release_exp143.json verify
python -m src.release.cli --config configs/release_exp143.json build-cache
python -m src.release.cli --config configs/release_exp143.json replay --output outputs/exp143_replay.zip
```

Kỳ vọng: `verify` trả `status=pass`, 1.200 query, 183.703 doc predictions, 127.271 chunk predictions; output replay có SHA256 `14687b27…e7c4ac3`.

### Run — inference thực

Các notebook GPU và worker inference nằm trong source inventory, nhưng không được bật trong UI review vì môi trường local này chưa fresh-run full 1.200 query. Để tái chạy model, tải đúng checkpoint/revision trong `model_registry.json`, dựng index theo manifest, rồi chạy notebook/worker tương ứng. Kết quả mới phải qua validator và **không** được gắn nhãn replay.

### Rebuild — từ text snapshot

1. Tải corpus mapping, text snapshot, chunk metadata và embedding artifacts theo `artifact_manifest.json`.
2. Đối chiếu SHA256 trước khi giải nén.
3. Nếu dùng index đã tải, chỉ cần `verify`. Nếu cần rebuild để kiểm tra, chạy `python -m scripts.rebuild_clean_index --no-swap` sau khi đặt input đúng các đường dẫn mã nguồn mong đợi. `--no-swap` tạo staging mà không thay active index; kiểm tra `stats.json` và hash trước khi tự quyết định swap. `scripts/streaming_index_rebuild.py` là biến thể cho dữ liệu lớn. Việc có script trong source không thay cho kiểm chứng full rebuild sạch; log clean-room hiện xác nhận replay, verify và dựng lại Exp143 từ cache.
4. Kiểm tra `stats.json`: 1.212.421 documents, 1.724.798 chunks, dense dim 1.024.

## 3. Chạy UI local

```powershell
python -m pip install -r demo/medical_retrieval/requirements.txt
python -m uvicorn demo.medical_retrieval.app:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`. Banner phải ghi rõ **Verified replay**. Chọn query → chạy replay → mở tab chunks → bấm một chunk để đối chiếu source. Export chỉ là một query review, không phải submission 1.200 query. Mục knowledge graph nằm dưới console, có nhãn **Experimental**, không sửa kết quả replay. Mọi query đều có đường nối từ replay cache; lớp concept corpus chỉ bật khi cung cấp riêng `concept_graph_v1.sqlite` và manifest tương ứng trong `data/reference/`.

## 4. Cấu hình và fingerprints

Nguồn sự thật duy nhất cho CLI và UI là `configs/release_exp143.json`. Checker fail-closed nếu thay đổi query, corpus mapping, submission hoặc prediction payload. Không có API key trong release; mọi credential phát triển nằm ngoài source.

| Artifact | SHA256 |
|---|---|
| query.parquet | `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08` |
| links_corpus_vi_en_zh.parquet | `87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7` |
| private replay ZIP | `14687b27b12a98777daddc290e54e2c25d310743917511866a06cee51e7c4ac3` |
| canonical predictions | `0cecc857fc2ee9dcdc60cf931d6e66bfb8d677e653396924850746f1bb3e0a9e` |

## 5. Môi trường đã kiểm tra

Các model của champion đều là checkpoint mã nguồn mở; `model_registry.json` ghi ID, revision bất biến, trang tải chính thức và prompt Qwen reranker. Replay không cần tải weights. Khi chạy fresh inference, cài `huggingface_hub` và tải đúng revision (HF_TOKEN chỉ cần nếu nền tảng yêu cầu, không nằm trong source):

```powershell
python -m pip install huggingface_hub
python -c "import json; from huggingface_hub import snapshot_download; d=json.load(open('model_registry.json',encoding='utf-8')); [snapshot_download(repo_id=m['id'],revision=m['revision']) for m in d['models']]"
```

Notebook Kaggle có dependency pins và batch/length riêng ngay trong cell đầu; chọn notebook đúng bước thay vì trộn môi trường release CPU với CUDA. Champion không gọi API bên ngoài, do đó không có API key/prompt API production để bàn giao.

- Windows 11; Python 3.13.12 cho release verify/replay/test.
- pandas 3.0.5, pyarrow 25.0.0, NumPy 2.4.4.
- GPU development: Python 3.10, PyTorch 2.7.1+cu128, Transformers 5.14.1, FAISS CPU 1.15.0, FlagEmbedding 1.4.2.
- UI: FastAPI 0.115.14, Uvicorn 0.34.3; chỉ bind `127.0.0.1` mặc định.
- Dung lượng tối thiểu: source nhỏ; replay ZIP ~94,5 MB; active index ~15,7 GB; text snapshot ~2,2 GB; cộng dung lượng tạm khi build.

## 6. Kiểm thử

```powershell
python -m pytest tests/test_release_core.py tests/test_private_champion_replay.py tests/test_concept_graph.py tests/test_replay_graph.py -q
```

Clean-room pass yêu cầu: import được source release; 10 test pass (gồm 4 test graph); `verify` pass; UI health trả đúng release ID; một lượt query có docs/chunks; evidence không nhận filesystem path từ browser.

## 7. Troubleshooting

- `SHA256 mismatch`: không tiếp tục; tải lại đúng artifact/revision.
- `Replay cache not found`: chạy `build-cache`, cần submission ZIP hợp lệ.
- UI evidence chậm lần đầu: resolver đang tìm `doc_id` trong 90 text parts; lần tiếp theo dùng RAM cache.
- `Exact chunk is not a contiguous substring`: UI vẫn hiện chunk nguyên văn và ghi unresolved; không highlight gần đúng.
- OOM khi inference: giảm batch, resume từ shard đã hoàn thành; không thay model revision âm thầm.
- ZIP >=100 MB: validator từ chối; không upload thử file quá giới hạn.
- Link Drive yêu cầu đăng nhập: sửa quyền và thử lại bằng cửa sổ ẩn danh trước khi gửi email.

## 8. Giới hạn đã khai báo

- Private score chưa công bố; không suy diễn từ public score.
- Replay chứng minh integrity và reproducibility của prediction, không chứng minh relevance với qrels private.
- 183 corpus IDs chưa nằm trong ba registry embedded/unembedded/remaining tại thời điểm khóa; không có ID ngoài corpus trong các registry này.
- UI không bật live inference và không lưu lịch sử/PHI/analytics. Mục knowledge graph là visualization thực nghiệm, không dùng trong bài Exp143; nếu không cài SQLite corpus graph, replay evidence graph vẫn hoạt động.
