# Script entrypoints

Chỉ giữ trong thư mục này các entrypoint có thể chạy lại hoặc còn thuộc pipeline hiện hành.

## Submission và đánh giá

- `prepare_private_champion.py`: replay best theo evaluator mới (Exp143) cho private 02–04/11; chạy `python -m scripts.prepare_private_champion`. Chỉ cho phép đúng 1.200 query/input cache đã khóa, audit từng prediction bằng champion và ZIP <100 MB; chưa tự nộp portal. Đổi `--out-prefix` nếu cần build lại, không ghi đè.
- `generate_best_submission.py`: entrypoint cấu hình best **lịch sử** (defaults Exp50); không dùng defaults như best hiện hành. Xem plan/experiment log để chọn champion đúng evaluator.
- `generate_true_union_submission.py`: hợp nhất candidate/submission có kiểm soát.
- `validate_submission.py`: kiểm tra schema và invariants trước khi nộp.
- `validate_evidence_submission.py`: kiểm tra evidence/chunk coverage.
- `verify_data_state.py`: snapshot trạng thái corpus/index/registry.

## Index và embedding

- `rebuild_clean_index.py`, `streaming_index_rebuild.py`: rebuild index.
- `merge_embedded_artifact.py`, `merge_recovered_artifacts.py`: merge artifact có kiểm tra.
- `sync_embedded_zip_registry.py`: đồng bộ embedded/unembedded registry.
- `build_faiss_index.py`: FAISS build.
- `prepare_qwen3_embedding_kaggle.py`, `prepare_qwen3_doc_retrieval.py`: chuẩn bị Qwen3.
- `build_qwen3_secondary_index.py`: secondary dense index.

## Retrieval kiến trúc mới

- `p2b_localization.py`: prepare/audit/build cho paragraph localization nguyên văn trong parent P1; khóa docs và số chunks/doc, fallback parent, không local relevance scorer.
- `p2b_score_worker.py`: BGE pinned revision chấm exact candidate không truncation, overflow được đánh dấu rõ; checkpoint khóa code/input/model.
- `create_p2b_localization_notebook.py`: generate notebook 21 pilot/full trên 2×T4, chỉ output score ZIP (không tải model/cache về).
- `build_joint_qwen_docs_p2a_chunks.py`, `validate_joint_qwen_docs_p2a_chunks.py`: joint owner-release candidate và independent exact-policy audit.
- `build_best_axis_recombination.py`: recombine exact BTC-scored docs/chunks axes theo evaluator tách biệt; streaming equality audit, không rescoring.
- `prepare_qwen4b_original_chunks.py`, `qwen4b_chunk_score_worker.py`, `create_qwen4b_chunk_notebook.py`: exact original-chunk Qwen4B pilot/full scoring; non-truncated up to 4.096 tokens, overflow marked `too_long`, query-disjoint 2×T4. Throughput pilot buộc full run chia bốn part; các notebook full `part01of02`/`part02of02` là cấu hình cũ không dùng.
- `audit_qwen4b_chunk_scores.py`, `merge_qwen4b_chunk_score_parts.py`, `watch_qwen4b_chunk_full.ps1`: audit exact lineage theo partition, tự tiếp nối tối đa hai Kaggle sessions, download/audit và merge đủ bốn Qwen4B chunk-score shards.
- `build_qwen4b_chunk_filter_submission.py`, `audit_qwen4b_chunk_filter_candidate.py`: build/audit Exp139 keep90 trên original chunks; giữ docs nguyên control, bảo toàn chunk quá dài không có score và kiểm tra provenance/ZIP.
- `build_final_sprint_candidate.py`: build Exp141 pure Qwen top130 docs, giữ nguyên toàn bộ chunks Exp139; ghi manifest/hash.
- `build_final_sprint_exp142_missing_parents.py`: chọn missing chunk-owner docs theo Qwen/BGE RRF, lấy parent nguyên văn từ active text store và nối vào Exp139; kiểm tra corpus, owner policy, CRC và ZIP limit.
- `build_final_sprint_exp143_short_spans.py`: append tối đa ba P2a windows/query cho chunks dài, chỉ khi source chunk hash khớp Exp142; kiểm tra offsets/text slice, giữ nguyên toàn bộ control chunks, CRC và giới hạn ZIP.
- `build_queryspec_bm25_candidates.py`: production candidate expansion từ query gốc, canonical query, evidence atoms và EN/ZH; chuẩn hóa tổng trọng số atom, tái dùng cache BM25 VI và chỉ tạo pool để câu Việt gốc dense-rerank.
- `package_exp144_queryspec_dense_inputs.py`: đóng gói QuerySpec, candidate pool và Exp143 control thành một private Kaggle input có hash/CRC.
- `create_exp144_queryspec_dense_kaggle_notebook.py`: tạo notebook 23 chạy exact BGE-M3 trên union Exp143 + QuerySpec/BM25 bằng 1–2 T4, checkpoint mỗi 25 query và chỉ tải ranking artifacts về.
- `build_exp144_docs_candidates.py`: từ ranking Exp144, build docs-only candidates và giữ toàn bộ chunks Exp143 chính xác; treatment chính `fused130_owner_gated` chỉ đổi query có QuerySpec hợp lệ, fuse Qwen/current + exact dense rồi khôi phục chunk-owner floor.

- `run_openrouter_queryspec.py`: QuerySpec Qwen3-14B có SQLite cache/resume/retry/provenance; xuất `query_specs.parquet` và `evidence_atoms.parquet`.
- `audit_queryspec_outputs.py`: kiểm tra coverage, fallback, atom invariants và model/provider provenance trước retrieval.
- `build_bm25_document_index.py`: BM25 document index.
- `build_bm25_translation_treatment.py`: treatment dùng query dịch.
- `generate_translation_submission.py`: tạo docs-only Exp136 từ bản dịch production; khóa top-150 Exp122 và giữ nguyên chunks.
- `validate_translation_submission.py`: kiểm tra Exp136 với Exp122, status dịch, schema, CRC và dung lượng ZIP.
- `generate_sentence_passage_submission.py`: tạo Exp138 sentence-aligned chunk passages với docs và chunk counts khóa theo Exp122.
- `validate_sentence_passage_submission.py`: đối chiếu độc lập docs/order/count, ownership, max passage length và ZIP CRC của Exp138.
- `build_icd10_query_routes.py`, `build_icd10_passage_routes.py`: ICD-10 soft routing.
- `package_exp130_bm25_dense_kaggle_inputs.py`: đóng gói input cho cascade Kaggle.
- `create_translation_production_kaggle_notebook.py`: tái tạo notebook dịch production.
- `prepare_p1_reranker_pilot.py`: đóng gói pilot 120 query hoặc full 1.200 query × 216 docs Exp122 với tối đa hai semantic passages/doc; giữ source offsets, chunk-owner flags và provenance.
- `create_p1_reranker_kaggle_notebook.py`: tái tạo notebook P1 chạy BGE reranker v2 m3 và Qwen3-Reranker-4B trên hai T4 độc lập, có checkpoint theo shard và output audit.
- `audit_p1_reranker_outputs.py`: kiểm tra output Kaggle P1 theo pair/doc membership và provenance, xuất proxy diagnostics cùng annotation queue; không coi owner flags là nhãn BTC.
- `prepare_p1_annotation_review.py`: lấy mẫu theo rank Qwen (1/40/100/180), top BGE, điểm bất đồng lớn nhất và owner thấp nhất; ghép exact source passages, chia 80 development/40 holdout theo strata, ẩn split/score/rank/owner proxy và sinh `review_instructions.md` có định nghĩa nhãn + giới hạn qrels.
- `prepare_p1_compact_human_review.py`: chọn bộ rà soát mù lõi 240 dòng (Qwen rank 1 và 180 cho mỗi query) từ queue 818 dòng bằng sampling reasons, không dùng nhãn yếu; giữ annotation IDs để nạp lại vào calibration.
- `create_p1_weak_judge_kaggle_notebook.py`: tái tạo notebook Qwen3-14B 4-bit self-host trên 2×T4 để gán weak labels có exact-evidence check; không dùng synthetic dev set và không coi output là BTC ground truth.
- `prepare_p1_weak_review.py`: chọn một core row cho đủ 120 query và tối đa hai case bổ sung/query, xuất CSV review mù không lộ split, weak label, score, rank hay owner proxy.
- `calibrate_p1_reranker.py`: so BGE/Qwen trên nhãn human-only và weak/mixed theo hai luồng riêng; nhận human labels từ `judgment_id` hoặc `query_id`+`doc_id` (P1 blind CSV), có thể thêm reviewed pairs ngoài weak queue; gate chỉ cho phép candidate-pool trial, không chứng minh corpus recall hay BTC relevance vì không có qrels.
- `create_p1_full_reranker_kaggle_notebook.py`: tạo notebook full run một model đã thắng; BGE có thể chạy một partition, Qwen4B bắt buộc ít nhất bốn partition để nằm trong giới hạn phiên Kaggle.
- `merge_p1_full_reranker_parts.py`: merge query-disjoint full-score ZIPs, khóa model revision/input hash và xác nhận đủ 1.200 × 216 query-doc trước submission builder.
- `audit_p1_full_reranker_part.py`: audit từng ZIP full-score độc lập với input pairs: exact pair/source lineage, score hữu hạn, passage→doc max, rank permutation, revision/hash/CRC và partition query coverage. Chạy trước merge/submission.
- `audit_p1_docs_policy_impact.py`: preflight docs-only cutoff trên full scores, thống kê số docs thật sự bị loại sau khi bảo vệ chunk owners; chỉ là structural impact, không dùng qrels/weak labels và không dự báo điểm BTC.
- `build_p1_verified_docs_submission.py`: sau calibration, tạo docs-only P1 trial từ full reranker scores; buộc giữ chunk-owning docs, giữ chunks chính xác như control và ghi manifest/hash/provenance.
- `validate_p1_docs_submission.py`: kiểm tra độc lập một ZIP docs-only so với Exp122 theo luồng trên 1.200 query: layout/CRC/size, chunks giống hệt, docs là subsequence, không mất chunk owners; có thể lưu báo cáo JSON.

## Corpus recovery

- `crawl_local_fast.py`: crawler chính.
- `reconcile_remaining_urls.py`: đối soát remaining URLs.
- `audit_local_recovered_final.py`: audit recovery.
- `recover_commoncrawl_hits.py`, `recover_pharmacity_api.py`, `recover_publisher_frontends.py`: recovery theo nguồn.

## Quy ước

- Thí nghiệm chỉ dùng một lần phải ghi kết quả vào `docs/experiments_log/`, sau đó chuyển code sang `archive/`.
- Logic được dùng từ hai entrypoint trở lên nên chuyển vào `src/` và có test.
- Không tạo thêm script `generate_expNNN_*.py` cho sweep nhỏ; dùng một generator tham số hóa hoặc cấu hình dữ liệu.
