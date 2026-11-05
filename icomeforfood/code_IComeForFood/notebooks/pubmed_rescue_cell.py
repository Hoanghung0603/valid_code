"""Retry only PubMed PMIDs still missing after the first NCBI pass.

Run after pubmed_completion_cell.py in the same Colab notebook.
"""


def _rescue_sidecar(pubmed_dir, stem, pmids):
    """Return cached hits for a completed request, or None if it needs sending."""
    xml_file = pubmed_dir / f'{stem}.xml.zst'
    meta_file = pubmed_dir / f'{stem}.request.json'
    if not meta_file.is_file():
        return None
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    if meta.get('pmids') != pmids or meta.get('xml_file') != xml_file.name:
        raise ValueError(f'NCBI retry sidecar does not match request: {meta_file}')
    if not xml_file.is_file():
        raise FileNotFoundError(f'NCBI retry XML missing: {xml_file}')
    hits = _parse_ncbi_xml(_read_zstd(xml_file))
    if sorted(hits, key=int) != meta.get('returned'):
        raise ValueError(f'NCBI retry sidecar/XML mismatch: {meta_file}')
    return set(hits)


def _rescue_request(pubmed_dir, stem, pmids, client, api_key):
    cached = _rescue_sidecar(pubmed_dir, stem, pmids)
    if cached is not None:
        print(f'NCBI retry cached: {stem}; {len(cached)}/{len(pmids)} found', flush=True)
        return cached
    params = {'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'xml',
              'tool': 'ViBioMIR'}
    if api_key:
        params['api_key'] = api_key
    for attempt, delay in enumerate((1, 3, 10, 30), 1):
        try:
            response = client.get(NCBI_EFETCH_URL, params=params)
            response.raise_for_status()
            hits = _parse_ncbi_xml(response.content)
            if set(hits) - set(pmids):
                raise ValueError(f'NCBI returned unrequested PMID(s): {set(hits) - set(pmids)}')
            break
        except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
            if attempt == 4 or (isinstance(exc, httpx.HTTPStatusError)
                                and exc.response.status_code in (401, 403)):
                raise RuntimeError(f'NCBI retry failed for {stem}: {exc}') from exc
            print(f'NCBI retry error; waiting {delay}s before attempt {attempt + 1}: {exc}',
                  flush=True)
            time.sleep(delay)
    xml_file = pubmed_dir / f'{stem}.xml.zst'
    meta_file = pubmed_dir / f'{stem}.request.json'
    temp_xml = xml_file.with_suffix(xml_file.suffix + '.tmp')
    temp_xml.write_bytes(zstd.ZstdCompressor(level=6).compress(response.content))
    os.replace(temp_xml, xml_file)
    meta = {'pmids': pmids, 'xml_file': xml_file.name,
            'returned': sorted(hits, key=int),
            'fetched_at': datetime.now(timezone.utc).isoformat(),
            'reason': 'targeted NCBI retry after incomplete first pass'}
    temp_meta = meta_file.with_suffix(meta_file.suffix + '.tmp')
    temp_meta.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    os.replace(temp_meta, meta_file)
    print(f'NCBI retry fetched: {stem}; {len(hits)}/{len(pmids)} found', flush=True)
    time.sleep(0.12 if api_key else 0.4)
    return set(hits)


def retry_ncbi_missing(pubmed_dir, pmids, report_dir, client=None, batch_size=10):
    """Retry missing PMIDs in small batches, then individually if still absent."""
    pubmed_dir, report_dir = Path(pubmed_dir), Path(report_dir)
    targets = sorted(set(str(pmid) for pmid in pmids), key=int)
    if not 1 <= batch_size <= 20:
        raise ValueError('batch_size must be from 1 to 20')
    if len(targets) > 500:
        raise ValueError(f'{len(targets)} non-ok PubMed IDs; check DB before retry')
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f'NCBI targeted retry: {len(targets)} PMID(s), {batch_size} per batch', flush=True)
    api_key = os.environ.get('NCBI_API_KEY')
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=90.0, http2=True)
    batch_hits = set()
    single_hits = set()
    try:
        for start in range(0, len(targets), batch_size):
            chunk = targets[start:start + batch_size]
            digest = hashlib.sha1(','.join(chunk).encode('ascii')).hexdigest()[:12]
            stem = f'ncbi_efetch_retry_batch_{start // batch_size:03d}_{digest}'
            print(f'NCBI retry batch {start // batch_size + 1}: {len(chunk)} PMID(s)', flush=True)
            batch_hits.update(_rescue_request(pubmed_dir, stem, chunk, client, api_key))
        unresolved_after_batch = sorted(set(targets) - batch_hits, key=int)
        print(f'NCBI retry: {len(unresolved_after_batch)} PMID(s) need one-by-one check',
              flush=True)
        for index, pmid in enumerate(unresolved_after_batch, 1):
            print(f'NCBI retry single {index}/{len(unresolved_after_batch)}: {pmid}',
                  flush=True)
            single_hits.update(_rescue_request(
                pubmed_dir, f'ncbi_efetch_retry_single_{pmid}', [pmid], client, api_key))
    finally:
        if own_client:
            client.close()
    recovered = batch_hits | single_hits
    unresolved = sorted(set(targets) - recovered, key=int)
    result = {'target_pmids': len(targets), 'batch_hits': len(batch_hits),
              'single_hits': len(single_hits), 'recovered': len(recovered),
              'unresolved': len(unresolved), 'unresolved_pmids': unresolved,
              'report': str(report_dir / 'pubmed_targeted_retry.json')}
    Path(result['report']).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'NCBI targeted retry complete: {len(recovered)}/{len(targets)} recovered; '
          f'{len(unresolved)} unresolved. Report: {result["report"]}', flush=True)
    return result
