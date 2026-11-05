@echo off
rem Docwin mo rong D_q BENCHMARK 20 cau dau tren z87 (bge-reranker-v2-m3 fp16, max-len 512, batch 16,
rem cua so 300 tu buoc 250 - GIONG docwin cu). Do cap/s de uoc ETA truoc khi chay day du.
rem Out: D:\r2ai\work\docwin_ext\dwx_bench.json.zst  Resume: dwx_bench_parts. Log: dwx_bench.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
if not exist work\docwin_ext mkdir work\docwin_ext
set DOCS=
for %%f in (D:\r2ai\work\docwin_ext\docwin_ext.part*.jsonl.zst) do call set DOCS=%%DOCS%% %%f
echo [%date% %time%] start dwx_bench > work\docwin_ext\dwx_bench.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\docwin_run.py --docs %DOCS% --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\docwin_ext\cands_docwin_ext.json --out D:\r2ai\work\docwin_ext\dwx_bench.json.zst --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 500 --window-words 300 --window-stride 250 --limit-queries 20 --resume-dir D:\r2ai\work\docwin_ext\dwx_bench_parts >> work\docwin_ext\dwx_bench.log 2>&1"
echo [%date% %time%] DONE >> work\docwin_ext\dwx_bench.log
