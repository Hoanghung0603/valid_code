# Interfaces chung – R2AI2026 RAG

Mọi module phải tuân theo đặc tả này để ghép được với nhau. Đọc kèm `R2AI2026_cuoc_thi.md` và `PLAN.md`.

## Môi trường
- Python: `/home/phong/Desktop/hsk/test2/.venv/bin/python` (torch CUDA, RTX 3050 **8GB**, dùng chung giữa các tiến trình → khi test giữ đỉnh VRAM ≤ 3.5GB, batch nhỏ, giải phóng model khi xong).
- Đĩa trống ~34GB. Không lưu HTML thô hàng loạt. Dữ liệu lớn để trong `data/` (parquet zstd).
- Mã nguồn trong `src/` (package `src`, chạy bằng `python -m src.<module>`). Cấu hình mặc định ở `configs/default.yaml` (mỗi module tự thêm khối khoá của mình, không xoá khoá của module khác).
- Cache: sqlite trong `data/cache/<module>.sqlite`.
- Không dùng API LLM đóng. Mô hình phải mở trọng số, ≤14B, phát hành trước 01/06/2026; ghi link HF + ngày phát hành vào `docs/MODELS.md`.

## Dữ liệu đầu vào
- `data/ViBioMIR/query.parquet`: `id:int, query:str` (1200 câu tiếng Việt)
- `data/ViBioMIR/links_corpus.parquet`: `id:int, url:str` (4.39M URL, 97 domain, đa số Trung, phần còn lại Việt)

## Schema

### Doc (`data/docs/*.parquet`, do crawler + pubmed ghi)
| trường | kiểu | ghi chú |
|---|---|---|
| doc_id | str | corpus: `str(id)` của links_corpus; PubMed: `PMID` (chuỗi số) |
| source | str | `corpus` \| `pubmed` |
| lang | str | `vi` \| `zh` \| `en` |
| url | str | |
| title | str | |
| text | str | nội dung chính đã làm sạch, Unicode NFC. **Là văn bản gốc để trích chunk nguyên văn.** Đoạn văn cách nhau bằng `\n\n` |
| fetched_at | str | ISO time |

PubMed: `text = title + "\n\n" + abstract` (giữ nhãn mục như "BACKGROUND:" nếu có).

### Chunk (`data/chunks/*.parquet`, do `src/chunk.py` sinh)
| trường | kiểu | ghi chú |
|---|---|---|
| chunk_id | str | `f"{doc_id}#{i}"` |
| doc_id, lang, source | str | |
| start, end | int | offset ký tự trong `Doc.text`; **bất biến: `doc.text[start:end] == chunk.text`** |
| text | str | |

### QueryPlan (`src/query.py` → `data/cache/query_plans.jsonl`)
```json
{"id": 1, "query_vi": "...", "vi": ["biến thể/ý con tiếng Việt"], "en": ["english rewrites"],
 "zh": ["中文改写"], "pubmed": ["PubMed search strings (MeSH/boolean)"], "keywords": {"vi": [], "en": [], "zh": []}}
```

### Candidate (đầu ra retrieve/rerank, dạng list[dict] trong Python)
`{"chunk_id", "doc_id", "lang", "source", "text", "score": float, "stage": "bm25|dense|rrf|rerank"}`

### Submission (đề bài)
```json
[{"id": 1, "relevant_docs": ["123", "38912345"],
  "relevant_chunks": [{"doc_id": "123", "chunk_text": "nguyên văn"}]}]
```
ZIP chứa đúng 1 file `.json` ở gốc, đủ 1200 id.

## API Python mỗi module phải cung cấp
- `src/crawl.py`: CLI crawl theo shard/ưu tiên domain; `load_docs(paths=None) -> Iterator[dict]`.
- `src/pubmed.py`: `search(queries: list[str], retmax=100) -> list[str PMID]`, `fetch(pmids) -> list[Doc]` (có cache), `candidates_for_plan(plan) -> list[Doc]`.
- `src/query.py`: `build_plans(queries: list[dict]) -> list[QueryPlan]` (cache theo id+hash query).
- `src/chunk.py`: `chunk_doc(doc, cfg) -> list[Chunk]`.
- `src/index.py`: build BM25 (theo ngôn ngữ) + dense index cho chunk corpus; `src/retrieve.py`: `retrieve(plan, k) -> list[Candidate]`; `src/rerank.py`: `rerank(query_vi, candidates) -> list[Candidate]`.
- `src/select.py`: `select(candidates, cfg) -> (docs: list[str], chunks: list[dict])`.
- `src/evaluate.py`: `evaluate(pred_json, gold_json, chunk_match="exact|contain|overlap") -> dict`.
- `src/submit.py`: `validate(pred, query_ids)`, `write_zip(pred, path)`.
- `run.py`: end-to-end `--queries data/ViBioMIR/query.parquet --out submission.zip`.
