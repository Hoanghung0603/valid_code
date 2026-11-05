@echo off
rem p3 tren z87 (GPU, uu tien thap, resume theo phan):
rem   A) sanity rerank 200 cap da co logit -> p3\sanity_out.json (doi chieu 1e-2 o b660)
rem   B) rerank bge-reranker-v2-m3 cho cap NEW thieu logit -> p3\rerank_p3.json (resume p3\rr_p3_parts)
rem   C) dense bge-m3 cho toan bo cap NEW (truy van goc 256 + HyDE 512), embedding doc DA CO
rem      -> p3\cache_new_dense.json + p3\cache_new_hyde.json
rem Doc text = text sach da cat 3.000 ky tu trong docs_vi_all.part000..006 (giong luc tao cache_pool_rr).
rem Log: D:\r2ai\work\p3\p3.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
if not exist D:\r2ai\work\p3 mkdir D:\r2ai\work\p3
echo [%date% %time%] start p3 > work\p3\p3.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\p3\sanity_pairs.json --out D:\r2ai\work\p3\sanity_out.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 50 --resume-dir D:\r2ai\work\p3\rr_sanity_parts >> D:\r2ai\work\p3\p3.log 2>&1"
echo [%date% %time%] sanity done >> work\p3\p3.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\p3\cands_p3.json --out D:\r2ai\work\p3\rerank_p3.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 2000 --resume-dir D:\r2ai\work\p3\rr_p3_parts >> D:\r2ai\work\p3\p3.log 2>&1"
echo [%date% %time%] rerank done >> work\p3\p3.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dense_pairs.py --emb D:\r2ai\work\emb D:\r2ai\work\emb_all --pairs D:\r2ai\work\p3\cands_new_all.json --queries D:\r2ai\work\queries.jsonl --query-len 256 --queries2 D:\r2ai\work\hyde\q_hyde.jsonl --query-len2 512 --out D:\r2ai\work\p3\cache_new_dense.json --out2 D:\r2ai\work\p3\cache_new_hyde.json --report D:\r2ai\work\p3\dense_report.json >> D:\r2ai\work\p3\p3.log 2>&1"
echo [%date% %time%] dense done >> work\p3\p3.log
echo [%date% %time%] DONE >> work\p3\p3.log
