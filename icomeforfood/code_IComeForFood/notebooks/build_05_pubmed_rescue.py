"""Build a one-cell Colab notebook for only the still-missing PubMed IDs."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source_notebook = json.loads((HERE / '04_pubmed_only.ipynb').read_text(encoding='utf-8'))
base = ''.join(source_notebook['cells'][1]['source'])
marker = '# 4. Fetch remaining PMIDs, sync status DB, and export docs'
assert base.count(marker) == 1
base = base.split(marker)[0]
base = base.replace(
    "print('PubMed helper loaded. Run the next cell to fetch missing PMIDs and export docs.', flush=True)",
    "print('PubMed helper loaded.', flush=True)",
)

intro = """# PubMed: thử lại riêng các PMID còn thiếu

**Chạy một cell code bên dưới.** Cell đọc các PMID PubMed chưa ở trạng thái `ok` từ chính `crawl/status.duckdb`, thử NCBI EFetch theo batch 10 PMID, rồi thử từng PMID chưa có kết quả. Mọi XML và kết quả yêu cầu đều được lưu để có thể chạy tiếp. Nếu lấy thêm được bài, cell sao lưu và cập nhật `status.duckdb`, rồi tạo **một bộ docs Parquet mới**; bộ docs cũ vẫn còn nguyên.

Cell này chỉ xử lý PMID còn thiếu. Nếu NCBI vẫn không trả nội dung cho một PMID sau hai cách thử, nó sẽ tiếp tục được liệt kê trong báo cáo, không bị gắn nhầm là đã crawl thành công. Dừng các tiến trình khác đang ghi DuckDB trước khi chạy.
"""

run = """
# 5. Retry only non-ok PubMed PMIDs and rebuild docs if any are recovered
if Path(str(STATUS_DB) + '.wal').exists():
    raise RuntimeError('Stop other processes writing status.duckdb before this retry')
source_rows = _pubmed_source_rows(STATUS_DB)
targets = sorted((pmid for pmid, row in source_rows.items() if row[2] != 'ok'), key=int)
print(f'PubMed remaining in status.duckdb: {len(targets)} PMID(s)', flush=True)
if targets:
    retry_result = retry_ncbi_missing(PUBMED_DIR, targets, REPORT_DIR, batch_size=10)
    if retry_result['recovered']:
        print('Recovered records found; rebuilding PubMed docs and syncing status.duckdb...',
              flush=True)
        result_pubmed = run_pubmed_completion(
            status_db=STATUS_DB,
            pubmed_dir=PUBMED_DIR,
            docs_dir=DOCS_DIR,
            report_dir=REPORT_DIR,
            cleaner=extract_pubmed_doc,
        )
        print('DONE. New PubMed docs:', result_pubmed['output'], flush=True)
        print('Content docs:', result_pubmed['content_docs'], flush=True)
        print('Still missing:', result_pubmed['placeholder_docs'], flush=True)
    else:
        print('No new records returned. Existing DB and docs remain as they were.', flush=True)
else:
    print('All PubMed IDs are already ok; nothing to retry.', flush=True)
"""

source = (base + '\n# 4. Targeted NCBI retry\n'
          + (HERE / 'pubmed_rescue_cell.py').read_text(encoding='utf-8') + run)


def cell(kind, cell_id, body):
    entry = {'cell_type': kind, 'metadata': {}, 'source': body.splitlines(keepends=True),
             'id': cell_id}
    if kind == 'code':
        entry.update(execution_count=None, outputs=[])
    return entry


notebook = {
    'cells': [cell('markdown', 'pubmed-rescue-intro', intro),
              cell('code', 'pubmed-rescue-run', source)],
    'metadata': source_notebook['metadata'],
    'nbformat': 4, 'nbformat_minor': 5,
}
target = HERE / '05_pubmed_rescue.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Wrote {target}: {len(source):,} code characters')
