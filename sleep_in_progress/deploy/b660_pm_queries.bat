@echo off
rem Chuan bi truy van VI/EN cho z87: queries_vi.jsonl {id,query} tu query.parquet, queries_en.jsonl {id,query,en} tu en_queries.jsonl.
rem Ghi vao D:\r2ai\data\pubmed\queries\ (upload sang z87 D:\r2ai\work\pm\). Log: D:\r2ai\data\pubmed\queries.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
set W=D:\r2ai\data\pubmed
if not exist %W%\queries mkdir %W%\queries
echo start %DATE% %TIME% > %W%\queries.log
%P% -c "import json,pandas as pd,pathlib; q=pd.read_parquet(r'D:\r2ai\data\query.parquet'); rows=[{'id':int(i),'query':s} for i,s in zip(q['id'],q['query'])]; pathlib.Path(r'D:\r2ai\data\pubmed\queries\queries_vi.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8'); en=[json.loads(l) for l in open(r'D:\r2ai\data\llm\en_queries.jsonl',encoding='utf-8') if l.strip()]; pathlib.Path(r'D:\r2ai\data\pubmed\queries\queries_en.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in en),encoding='utf-8'); print('vi',len(rows),'en',len(en))" >> %W%\queries.log 2>&1
if errorlevel 1 goto fail
echo end %DATE% %TIME% >> %W%\queries.log
echo exit done >> %W%\queries.log
goto :eof

:fail
echo QUERIES PREP FAILED >> %W%\queries.log
echo exit fail >> %W%\queries.log
goto :eof
