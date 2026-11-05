# Sync existing PubMed batches into status.duckdb

The PubMed notebook stores Europe PMC records in `crawl/pubmed/batch_*.json.zst`.
The retry crawler reads `crawl_status` from DuckDB and cannot see batch files on
Google Drive. Download the existing `crawl/pubmed` folder from Drive and extract
all batch files into `D:\r2ai_data\crawl\pubmed` before running the commands.
Do not run the notebook's PubMed download again just to update DuckDB.

Stop the current `retry_crawl4ai.py run` process first. Its URL queue was built
before the DB update; an active process would still request PubMed URLs even
after those IDs become `ok`. The import command checks its run lock when
`--apply` is used.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\crawl\sync_pubmed_cache.py `
  --cache-dir 'D:\r2ai_data\crawl\pubmed' `
  --status-db 'D:\r2ai_data\crawl\status.duckdb'

.\.venv\Scripts\python.exe -X utf8 .\crawl\sync_pubmed_cache.py `
  --cache-dir 'D:\r2ai_data\crawl\pubmed' `
  --status-db 'D:\r2ai_data\crawl\status.duckdb' --apply
```

The first command is a read-only dry run. Check that `db_pubmed` is 240,000,
`matched` is close to 239,771, and `missing` is close to 229 (the 21 October
review numbers; current batches may differ). The importer refuses an apply
below 99% coverage by default. It decodes and validates every batch, matches
PMID to the DB URL, then marks only matched IDs `ok` in one transaction.
Before writing, it copies the DB and any WAL into `crawl/backups/` under the
shared lock.
`pubmed_cache_index` records each ID's batch filename and import time; the
original compressed batches remain required for later extraction. IDs absent
from the cache retain their prior status. Repeating the import is safe.
`check_retry_coverage.py` treats an indexed PubMed ID as locally available only
while its referenced batch file is present.

After applying, restart `retry_crawl4ai.py run` to rebuild its queue from the
updated DB. Handle the remaining PMID list separately through NCBI EFetch;
the Europe PMC cache import does not invent records for missing IDs.
