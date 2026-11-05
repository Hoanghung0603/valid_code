# PubMed branch – notes (src/pubmed.py)

Measured 2026-10-01 from this machine, **without** an NCBI API key.

## API
```python
from src import pubmed
pubmed.search(["kidney stone urinary obstruction treatment"], retmax=100)  # -> [PMID] fused (weighted RRF)
pubmed.fetch(["30990297", ...])            # -> [Doc] (cached, EFetch batches of 200, skips no-abstract)
pubmed.candidates_for_plan(plan)           # plan["pubmed"] + plan["en"] -> [Doc] (+ pm_rank, pm_score)
await pubmed.acandidates_for_plan(plan)    # asyncio wrappers (asearch / afetch) run in worker threads
c = pubmed.get_client(); c.esearch(q), c.esearch_relaxed(q), c.epmc_search(q), c.elink(pmids)
```
CLI (resumable: finished plans are stored in the sqlite `plans` table, all HTTP results are cached):
```
python -m src.pubmed --plans data/cache/query_plans.jsonl --out data/docs/pubmed.parquet \
       [--cands-out data/cache/pubmed_candidates.jsonl] [--limit N] [--workers 4] [--elink] [--cache path]
```
Outputs: `data/docs/pubmed.parquet` (unique Docs: the INTERFACES columns plus `has_abstract, year, journal,
pub_types[], mesh_terms[], article_lang`) and `pubmed_candidates.jsonl` (`{"id", "pmids": [ranked], "scores"}` per query).
Doc text = `title + "\n\n" + abstract`, with one paragraph per `AbstractText` and the section label kept
(`"BACKGROUND: ..."`). Inline markup (`<i>`, `<sup>`) is flattened, whitespace collapsed, NFC.
`PubmedBookArticle` (StatPearls / NCBI Bookshelf chapters) is parsed as well.

Cache: `data/cache/pubmed.sqlite` (tables search / docs / missing / elink / plans). Items are never refetched.
PMIDs that EFetch does not return are stored in `missing`.
Env: `NCBI_API_KEY` (rate goes from 2.0 to 7.5 req/s), `NCBI_EMAIL` (sent as `email`), `tool=r2ai2026`.

## Findings

### 1. ESearch `sort=relevance` is only correct for small pages (important)
PubMed Best Match ordering is returned only when `retmax` is small. For `kidney stone urinary obstruction treatment`:

| call | first PMIDs | overlap with the true top-20 |
|---|---|---|
| retmax=5/20/40 | 30990297 (*Kidney Stones: Treatment and Prevention*), 33887891, 3292230 | 20/20 |
| retmax=45 | 19193412, 7557700, ... | 7 |
| retmax=60/100/200 | 19193412 ... (and it **changes between identical calls**) | 0–1 |

`retmax=20` pages with `retstart=0,20,40,60` were stable and gave the same order as retmax=10 paging. From
retstart≥80 the order was non-deterministic. A page of retmax=40 at retstart=40 did not match the 20-pages.
**So `esearch()` pages in steps of 20 up to `esearch_max_depth` (default 60, so 3 requests per string).**
A naive `retmax=100` call (as in the competition example URL) loses the Best Match top hits.

### 2. Plain-text ESearch: automatic term mapping (ATM) and zero-hit causes
- ATM maps phrases to MeSH plus all-fields synonyms (`kidney stone` becomes `"kidney calculi"[MeSH] OR ...`).
  Stopwords (what, for, is, ...) are ignored. Every remaining concept is ANDed.
- What gives **0 hits**:
  - numbers and units. `beta-hCG below 0.10 U/L interpretation` returns 0, because `"0.10"` and `"u l"` must match.
  - `?` or `:` inside sentences.
  - "to do" being parsed as the author `do, to[Author]` (`what to do for urinary obstruction ...` returns 0).
  - long natural-language sentences, where too many concepts are ANDed.
- `SCC tumor marker 1.52 elevated` returns 1 hit. Removing the number gives 476.
- Verbose questions with stopwords give the **same** count and order as the keyword version. Cleaning costs nothing.

### 3. Implemented relaxation (`esearch_relaxed`)
0. `preclean` (no extra request). For non-boolean strings, search `clean_terms(q)`, which drops stopwords,
   punctuation, measurements (`1.52`, `0.10 U/L`, `120/80 mmHg`) and units. It keeps small integers after a word
   (`type 2`, `HPV 16`).
1. If `count < min_hits` (20), try the variants below in parallel at depth 20 (≤ `max_relax_requests`):
   - **cleaned**
   - **core**: generic words dropped (normal, level, result, high/low, elevated, interpretation, ...)
   - **drop-one-clause** variants. For more than 5 clauses, sliding 4-clause windows instead.
   Clauses come from `split_clauses()`, which respects `( ... OR ... )` groups, quotes and `[tags]`. This means
   the `query.py` style `"chronic kidney disease diet (diabetes OR diabetic)"` and the full boolean MeSH style are both handled.
2. If they still return fewer than `min_hits` hits in total, search the **OR of clauses** (≤5 clauses, weight 0.5).
   Best Match ranks documents that match more clauses first.
3. The final list is the original hits first, then the RRF of the relaxed lists.

Example: `beta-hCG below 0.10 U/L interpretation` with 0 hits becomes `beta-hCG` (5352 hits). Its top results:
32599164 *Tubal Ectopic Pregnancy with Undetectable Initial Serum β-hCG*, 27630909 *Role of Serum Beta hCG in
Early Diagnosis ... Ectopic Pregnancy*. A plan string like `beta-hCG negative pregnancy test` is much better
(203 hits; 38292164 *False-negative qualitative hCG test ('hook effect')*, 21691529 *Molar Pregnancy with False
Negative β-hCG Urine*). **The LLM should rewrite to the clinical intent ("negative / undetectable hCG,
exclude pregnancy"), not copy the number.**

### 4. Europe PMC
- `resultType=idlist` + `(q) AND SRC:MED AND HAS_ABSTRACT:y` returns the PMID directly (1 request per 1000 ids, cursorMark paging).
- Its default "relevance" ranking is strongly **recency-biased**: the top hits are 2026 case reports. It handles
  natural language without zero-hit problems (`what to do for urinary obstruction caused by kidney stones` gives 1962 hits).
  It does not understand PubMed `[MeSH]` tags, so only `plan["en"]` is sent (`epmc_from`). Weight 0.7 in RRF.
  It adds recent papers that Best Match ranks low. Recall source, not precision.

### 5. ELink `pubmed_pubmed`
One POST with repeated `id=` returns one linkset per PMID (`cmd=neighbor_score`), so **1 request per plan**.
The neighbours are good. For 30990297: *Nephrolithiasis: acute management and prevention*, *Prevention of
recurrent urinary stone disease*, ... It added 89 new PMIDs for the kidney-stone plan. It is enabled in
`configs/default.yaml` (top 5 hits × 20 neighbours, weight 0.5).

### 6. Rate limits
- 6 threads at an evenly spaced 2.5 req/s still caused 3–4 HTTP 429s per 30 requests. Network jitter puts more
  than 3 requests into one second. 2.0 req/s caused 0. Defaults: 2.0 req/s without a key, 7.5 req/s with one.
  429/5xx/"API rate limit exceeded" are retried with exponential backoff (honouring Retry-After), and a 429 pauses the shared limiter.
- Sequential latency: ESearch 0.4–1.2 s; EFetch of 200 PMIDs 2–3 s, about 1.5 MB XML, about 2 KB JSON per doc.
  About 94% of retrieved PMIDs have an abstract (`HAS_ABSTRACT` filtering is done at fetch time).

## Results (live)
`tests/test_pubmed.py` (`PUBMED_LIVE=1`): 4 hand-translated ViBioMIR questions, `candidates_for_plan`
(en string only, top 50). All have 10/10 on-topic top-10 titles, 5–7 s each from a cold cache. Examples:
- sỏi thận tắc nghẽn (example query in the task description): 30990297 *Kidney Stones: Treatment and Prevention*; 42399868 *Early urinary
  decompression and mortality in septic obstructing ureteral stones*; 20063999 *Evidence-based drainage of
  infected hydronephrosis secondary to ureteric calculi*
- q5 lao đa kháng: 38523140 *Multidrug-resistant tuberculosis* (Nat Rev Dis Primers); 39693541 *Levofloxacin for the
  Prevention of MDR-TB in Vietnam*; 40992540 *Protecting households on exposure to ... MDR-TB*; 27592087
  *Preventing the spread of MDR-TB and protecting contacts*
- q4 SCC 1.52: 2591447 *Experiences with SCC antigen, a new tumor marker for cervical carcinoma*; 2790678
  *Elevated levels of SCC antigen in patients with a benign disease of the skin*; 37787289 *Role of SCC antigen in prognostication ...*
- q13 thuốc lao + viêm gan: 17052297 *Antituberculosis drugs and hepatotoxicity*; 21181011 *Antituberculosis
  drug-induced hepatotoxicity ...*
- q12 rốn ướt: 38319317 *Umbilical cord stump medication ...*; 20828455 *Umbilical separation time delayed by alcohol application*

## Timing / budget
10 realistic plans (query.py style: 1–2 `en` sentences + 2–3 `pubmed` keyword strings) with ELink, cold cache,
4 workers. This took **90 s, 171 NCBI requests (17 per plan) + 1.7 Europe PMC requests per plan**, gave ~300 docs with an
abstract per plan, and wrote 2918 unique docs. Per plan: ~9 ESearch pages, 0–8 relaxation calls, 1 ELink, 2 EFetch.
The process is rate-bound, not latency-bound.

| 1200 queries, ~5 strings each (~20 NCBI req/plan, ~24k req) | estimate |
|---|---|
| no API key (2.0 req/s) | **≈ 3.3 h** |
| NCBI_API_KEY (7.5 req/s) | **≈ 55 min** |
| `esearch_max_depth: 40` (2 pages/string) with key | ≈ 40 min |
| re-run with a warm cache | seconds |
Storage: ≈ 5 KB per doc in sqlite (incl. search cache). For 1200 plans × 300 docs, with overlap giving maybe 250–350k
unique docs, that is about 1.5 GB sqlite and 250 MB parquet.

## Recommendations
1. **Get an NCBI API key** and export `NCBI_API_KEY`/`NCBI_EMAIL`. That is 3.7× faster, which matters for the private round
   (new queries, 3-day window: prefetch on day 1).
2. Query generation (src/query.py): make `pubmed` strings short keyword strings with 2–4 concepts and MeSH-style terms, as in the
   current prompt. Rewrite test values into the clinical meaning ("undetectable beta-hCG", "elevated SCC antigen")
   instead of numbers. Include one broad string (main disease + intent) so that every plan gets ≥ 20 hits.
3. Keep Europe PMC + ELink for recall. Rely on the reranker for precision. The fused PubMed order is a weak signal,
   but `pm_score` can be used as a small prior or tie-breaker.
4. Possible filters to evaluate on dev: exclude non-human (veterinary hits appear, e.g. *Surgical removal of canine uroliths*).
   Prefer `Review`/`Practice Guideline` pub types, which fit patient-style questions. Use `article_lang == "eng"`.
5. If the gold PubMed labels are pooled from simple ESearch runs (e.g. the example URL `retmax=100`), note that the
   organisers' list order may be non-deterministic (finding 1). Consider adding a raw `retmax=100` call per main string to cover that pool.
