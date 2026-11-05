# Crawl4AI retry queue outside Long Chau, zysjonline, PubMed and VOV

`crawl/retry_crawl4ai.py` reads every `crawl_status` row whose status is not
`ok`, excluding `nhathuoclongchau.com.vn`, `tiemchunglongchau.com.vn`,
`zysjonline.com`, `pubmed.ncbi.nlm.nih.gov`, and `vov.vn`, including their
subdomains. Previously failed/blocked URLs on other hosts remain eligible.
The old 331,176-ID count from 21 October 2026 included five pending
`tiemchunglongchau.com.vn` IDs; use `plan` for the current count.
Results from this domain already saved in staging before the exclusion remain
eligible for sync into the DB; the exclusion only prevents new crawl requests.

The run stages under `D:\r2ai_data\crawl\retry_crawl4ai\`. A successful
record stores the original navigation response bytes, not Crawl4AI's rendered
HTML. Its six JSON fields (`id`, `final_url`, `http_code`, `content_type`,
`encoding`, `body_b64`) are the main crawler's raw contract. `status.csv` has
the `crawl_status` columns plus `shard_file`. A status for a saved body is
written only after its staging shard is complete. A `STOP` file causes a soft
stop. Repeating `run` skips IDs already in local `status.csv`; use
`--retry-failed` when deliberately revisiting failed attempts. A full
`run --max-urls 0` includes those failed attempts automatically so the queue
can cover non-`ok` IDs. URLs that have returned HTTP 404 at least twice are
skipped on subsequent runs; use `--retry-404` to explicitly revisit them.
This avoids retrying the same missing URL at every restart. A 404 may reflect
site-specific access handling, so the skipped URLs remain in the DB for review.

The retry browser uses the same optimized Crawl4AI profile as
`longchau_odd_fast.py`: text/light/memory-saving modes, page recycle after 100
pages, and `prefetch` with a 45-second page timeout. The retry crawler keeps
its own cross-domain queue, per-host pacing, cooldowns, and raw-response hook.

**`run` now auto-syncs by default.** After each 50-request checkpoint and at
clean exit, it validates newly saved bodies, writes canonical
`crawl/html/shard_XXXXX.jsonl.zst` files, and updates `status.duckdb` in a
transaction. The append-only cursor in `sync_state.json` makes this incremental;
starting `run` again first catches up any checkpoint saved before a crash.
Replaying a committed checkpoint does not create duplicate main shards.
The terminal prints the current queue position, ID, host, HTTP code, and running
counts after every request. The status CSV and DB commit successful HTML at
the 50-request checkpoint or at clean exit.

## Commands (PowerShell, from repository root)

```powershell
.\.venv\Scripts\python.exe -X utf8 .\crawl\retry_crawl4ai.py plan
.\.venv\Scripts\python.exe -X utf8 .\crawl\retry_crawl4ai.py run --max-urls 50
.\.venv\Scripts\python.exe -X utf8 .\crawl\retry_crawl4ai.py run --max-urls 0
New-Item -ItemType File -Path 'D:\r2ai_data\crawl\retry_crawl4ai\STOP'
Remove-Item -LiteralPath 'D:\r2ai_data\crawl\retry_crawl4ai\STOP'
.\.venv\Scripts\python.exe -X utf8 .\crawl\retry_crawl4ai.py run --max-urls 0
```

For a targeted Pharmacity recovery after the beta host returns 403:

```powershell
Remove-Item -LiteralPath 'D:\r2ai_data\crawl\retry_crawl4ai\STOP' -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -u -X utf8 .\crawl\retry_crawl4ai.py run --max-urls 0 --source-host pmc-ecm-healthblog.beta.pharmacity.io
```

This maps a one-segment beta article slug to
`https://www.pharmacity.vn/<slug>.htm` for the request. It keeps the original
source URL and ID in `crawl_status`, writes the public URL to `final_url`,
paces/cools down the public host, and auto-syncs successful raw HTML. The
`--source-host` option limits only the source queue; it can be repeated.

For Tâm Anh, the old classifier treated the word `recaptcha` inside a normal
article's JavaScript as a challenge. The retry classifier now requires stronger
challenge evidence or `captcha` in the page title. A one-page live Crawl4AI
check returned the expected article title and HTTP 200. Run that host by itself:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 .\crawl\retry_crawl4ai.py run --max-urls 0 --out 'D:\r2ai_data\crawl\retry_crawl4ai_tamanh' --source-host tamanhhospital.vn
```

On the local DB snapshot checked 23 October 2026, excluding the five requested
domains leaves 72,108 non-`ok` source IDs. A full resume (`--max-urls 0`) selects
71,853 for requests; the rest have saved `ok`/`non_html` staging output or were
skipped after repeated 404 responses. These counts change as sync/crawl runs.

The `STOP` file must be removed before a deliberate resume. No manual merge
is needed for normal `run`. `sync` is available to import an older staged
checkpoint without sending network requests; `--no-auto-sync` leaves output
in staging for a manual run. The legacy `merge` command remains available for
full validation or recovery.

`plan` sends no requests. `run` defaults to at most 50 requests; zero means
continue through the full queue. It checks robots.txt, makes one request at a
time, keeps at least five seconds between requests to the same host, and stops
requesting a host after a block. It records 401/403/429/503/521 and challenge
pages with HTTP 200 as `blocked`; an HTTP 404 page is an HTTP error even if its
text contains a challenge marker. A full run prints the next eligible time and
waits for the five-minute host cooldown instead of silently exiting with zero
requests. A bounded run exits with that explanation. Create the `STOP` file or
press Ctrl+C to end a waiting run. It does not retry or rotate identity within
a request. Edit `--delay`
only to slow down, or use an appropriate host and rate for each site.
PubMed and VOV IDs remain in the DB but receive no new requests from this queue.

Auto-sync validates every new raw record against its status row and source DB.
It reserves shard numbers already referenced by the DB, including
`home_import_shards`, and will not overwrite an existing file. Stop any other
DB writer that does not use the shared `status.duckdb.lock`; this crawler and
`longchau_odd_fast.py` share that lock. The old `merge` command still validates the complete
staging set; without `--apply` it is a dry run.

## Source-specific caveat

PubMed contributed 240,000 of the earlier 331,176-ID queue. Its Europe PMC / NCBI
collection runs separately; this browser retry queue excludes PubMed and VOV.

The staging output does not repair missing historical shards. On the checked
local machine, 83 DB-referenced main HTML shards have not yet been synced into
`D:\r2ai_data\crawl\html`; sync them separately before full extraction.
