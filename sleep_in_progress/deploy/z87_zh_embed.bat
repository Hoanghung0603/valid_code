@echo off
rem Embed 540.863 doc ZH (D:\r2ai\work\zh\docs\docs_zh_all.partNNN.jsonl.zst, 1.500 ky tu) + 1.200 cau VI bang bge-m3
rem (fp16, doc 512 / query 256 token) tren z87, GPU only, uu tien thap, resume theo khoi 4096 doc (work\zh\emb).
rem work-old = thu muc rong (khong tai dung embedding VI). Out: D:\r2ai\work\zh\zh_dense_top100.json
rem Log: D:\r2ai\work\zh\embed_zh.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai\scripts
echo [%date% %time%] start embed_zh >> D:\r2ai\work\zh\embed_zh.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\embed_all.py --docs D:\r2ai\work\zh\docs --queries D:\r2ai\work\queries.jsonl --work-old D:\r2ai\work\zh\emb_none --work D:\r2ai\work\zh\emb --out D:\r2ai\work\zh\zh_dense_top100.json --doc-len 512 --query-len 256 --batch 16 --top 100 >> D:\r2ai\work\zh\embed_zh.log 2>&1"
echo [%date% %time%] DONE >> D:\r2ai\work\zh\embed_zh.log
