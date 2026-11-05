"""Build the standalone, one-cell PubMed completion notebook."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = HERE / '04_pubmed_only.ipynb'

INTRO = """# PubMed riêng: lấy PMID còn thiếu và xuất docs

**Chỉ chạy một cell code bên dưới.** Cell dùng `/content/drive/MyDrive/r2ai_data/crawl/status.duckdb` và các file cache `crawl/pubmed/batch_*.json.zst` đã có. Nó tìm PMID thiếu, lấy qua NCBI EFetch, lưu XML, sao lưu rồi cập nhật **chính** `status.duckdb` trên Drive, và xuất tất cả PMID thành Parquet schema v2 trong `r2ai_data/docs/pubmed_<run>/`.

Tiến trình sẽ in ở từng bước. Không cần chạy Gold hay HTML. Hãy dừng các tiến trình khác đang ghi DuckDB trước khi chạy; chạy lại cell sẽ dùng các batch NCBI đã lưu.
"""

SETUP = """# 1. Dependencies, Drive, and paths
print('PubMed only: preparing dependencies and Drive...', flush=True)
import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

requirements = {
    'duckdb': 'duckdb', 'httpx': 'httpx', 'pandas': 'pandas',
    'pyarrow': 'pyarrow', 'zstandard': 'zstandard',
}
missing = [package for module, package in requirements.items()
           if importlib.util.find_spec(module) is None]
if missing:
    print('Installing:', ', '.join(missing), flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])

from google.colab import drive
drive.mount('/content/drive')
DATA_ROOT = Path('/content/drive/MyDrive/r2ai_data')
CRAWL_DIR = DATA_ROOT / 'crawl'
STATUS_DB = CRAWL_DIR / 'status.duckdb'
PUBMED_DIR = CRAWL_DIR / 'pubmed'
DOCS_DIR = DATA_ROOT / 'docs'
REPORT_DIR = DATA_ROOT / 'reports' / (
    'pubmed_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
if not STATUS_DB.is_file():
    raise FileNotFoundError(f'Status DB not found: {STATUS_DB}')
batch_files = sorted(PUBMED_DIR.glob('batch_*.json.zst'))
if not batch_files:
    raise FileNotFoundError(f'PubMed cache batches not found: {PUBMED_DIR}')
print(f'Status DB: {STATUS_DB}', flush=True)
print(f'Europe PMC cache: {len(batch_files)} batch files in {PUBMED_DIR}', flush=True)
print(f'Docs output root: {DOCS_DIR}', flush=True)

"""

RUN = """
# 4. Fetch remaining PMIDs, sync status DB, and export docs
print('PubMed only: starting NCBI completion and docs export...', flush=True)
result_pubmed = run_pubmed_completion(
    status_db=STATUS_DB,
    pubmed_dir=PUBMED_DIR,
    docs_dir=DOCS_DIR,
    report_dir=REPORT_DIR,
    cleaner=extract_pubmed_doc,
)
print('DONE. PubMed docs:', result_pubmed['output'], flush=True)
print('Content docs:', result_pubmed['content_docs'], flush=True)
print('Still missing:', result_pubmed['placeholder_docs'], flush=True)
print('Report:', REPORT_DIR / 'pubmed_extraction_summary.json', flush=True)
"""


def make_cell(kind, cell_id, text):
    cell = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True),
            'id': cell_id}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    return cell


source = (SETUP + '# 2. PubMed-only schema-v2 cleaner\n'
          + (HERE / 'pubmed_only_cleaner.py').read_text(encoding='utf-8')
          + '\n# 3. NCBI completion, DB sync, and Parquet writer\n'
          + (HERE / 'pubmed_completion_cell.py').read_text(encoding='utf-8')
          + RUN)
notebook = {
    'cells': [make_cell('markdown', 'pubmed-only-intro', INTRO),
              make_cell('code', 'pubmed-only-run', source)],
    'metadata': {'colab': {'provenance': []},
                 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4,
    'nbformat_minor': 5,
}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Wrote {TARGET}: {len(source):,} code characters')
