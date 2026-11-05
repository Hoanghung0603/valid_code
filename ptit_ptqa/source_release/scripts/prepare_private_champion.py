"""Replay Exp143 only for exactly matching, audited public/private queries.

This is cached-inference replay, NOT fresh model inference for unseen queries.
Fails closed on changed query contents, input hashes, or output predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts import build_final_sprint_exp143_short_spans as champion
from scripts.build_final_sprint_exp142_missing_parents import read_rows
from scripts.generate_translation_submission import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "submission_exp143_p2a_long_chunks_add3.zip"
CHAMPION_HASH = "039eb431209008b00eeb7b8df76b4d784c20a677c1b375444a167bf19f229649"
CONTROL_HASH = "7b545b65765c528e1abb708a89bd65be1ef57f507f04c101fc722bc094627a82"
RUN_MANIFEST = ROOT / "data/kaggle_outputs/p2a_chunk_full/p2a_chunk_full/run_manifest.json"
ORIGINAL_PAIRS = ROOT / "data/kaggle_inputs/p2a_chunk_full_v1/p2a_pairs.parquet"


def validate_queries(current: pd.DataFrame, cached: pd.DataFrame, count: int = 1200) -> list[int]:
    if set(current.columns) != {"id", "query"}:
        raise ValueError("Expected official query schema id/query")
    if len(current) != count or current.id.duplicated().any() or current.isna().any().any():
        raise ValueError("Missing/duplicate/null official queries")
    if not pd.api.types.is_integer_dtype(current.id):
        raise ValueError("Query IDs must be integers")
    if not current['query'].map(lambda q: isinstance(q, str) and bool(q.strip())).all():
        raise ValueError("Empty/non-string query")
    pairs = cached[["query_id", "query_text"]].drop_duplicates()
    if len(pairs) != count or pairs.query_id.duplicated().any():
        raise ValueError("Cache query coverage or text inconsistent")
    expected = dict(zip(current.id, current['query'], strict=True))
    observed = dict(zip(pairs.query_id, pairs.query_text, strict=True))
    if expected != observed:
        raise ValueError("Query texts/IDs differ: fresh retrieval and inference required")
    return current.id.astype(int).tolist()


def validate_row(row: dict, official: set[int]) -> tuple[int, int]:
    if set(row) != {"id", "relevant_docs", "relevant_chunks"} or type(row['id']) is not int:
        raise ValueError("Invalid submission schema")
    docs = row['relevant_docs']
    if not docs or any(type(d) is not int for d in docs) or len(docs) != len(set(docs)):
        raise ValueError("Empty, duplicate or noninteger docs")
    if not set(docs).issubset(official):
        raise ValueError("Out-of-corpus doc")
    seen = set()
    for c in row['relevant_chunks']:
        if (set(c) != {'doc_id', 'chunk_text'} or type(c['doc_id']) is not int
                or c['doc_id'] not in official or not isinstance(c['chunk_text'], str)
                or not c['chunk_text'].strip()):
            raise ValueError("Invalid chunk")
        key = (c['doc_id'], c['chunk_text'])
        if key in seen:
            raise ValueError("Duplicate chunk")
        seen.add(key)
    return len(docs), len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=ROOT/'data/raw/btc_provided/query.parquet')
    parser.add_argument('--out-prefix', type=Path, default=ROOT/'submission_private01_exp143_replay')
    args = parser.parse_args()
    output = args.out_prefix.with_suffix('.zip')
    audit_path = args.out_prefix.with_name(args.out_prefix.name+'_audit.json')
    build_manifest = args.out_prefix.with_name(args.out_prefix.name+'_build_manifest.json')
    if any(p.exists() for p in [output, audit_path, build_manifest, output.with_suffix('.json')]):
        raise FileExistsError('Refusing to overwrite private candidate or audit')

    print('Preflight: verify query and cached-input provenance', flush=True)
    manifest = json.loads(RUN_MANIFEST.read_text(encoding='utf-8'))
    query_hash = sha256_file(args.queries)
    if query_hash != manifest['input_manifest']['official_queries_sha256']:
        raise ValueError('Official query file hash changed; cache reuse not approved')
    for p, expected in [
        (CHAMPION, CHAMPION_HASH), (champion.CONTROL, CONTROL_HASH),
        (ORIGINAL_PAIRS, manifest['input_manifest']['artifacts']['p2a_pairs.parquet']['sha256']),
        (champion.SCORES, manifest['outputs']['p2a_pair_scores.parquet']),
    ]:
        if sha256_file(p) != expected:
            raise ValueError(f'Input hash mismatch: {p}')
    # The Kaggle notebook rewrites the input parquet with zstd. Its file hash
    # therefore differs from the input manifest: verify all values and dtypes
    # against the original hash-pinned input, not just the query columns.
    original_pairs = pd.read_parquet(ORIGINAL_PAIRS)
    exported_pairs = pd.read_parquet(champion.PAIRS)
    pd.testing.assert_frame_equal(original_pairs, exported_pairs, check_exact=True)
    ids = validate_queries(pd.read_parquet(args.queries), exported_pairs)
    pair_provenance = {
        'original_input_sha256': sha256_file(ORIGINAL_PAIRS),
        'kaggle_export_sha256': sha256_file(champion.PAIRS),
        'all_rows_columns_dtypes_order_equal': True,
        'serialization_note': 'Kaggle notebook rewrites input as zstd parquet',
    }
    del original_pairs, exported_pairs
    corpus = ROOT/'data/raw/btc_provided/links_corpus_vi_en_zh.parquet'
    official = set(pd.read_parquet(corpus, columns=['id']).id.astype(int))
    print('All 1200 query texts/IDs and cache hashes match; rebuilding Exp143 selection/ZIP', flush=True)
    build = champion.build(output, build_manifest)
    print('Build complete; independent exact-prediction and schema audit', flush=True)
    found = []
    docs = chunks = 0
    payload_digest = hashlib.sha256()
    for actual, reference in zip(read_rows(output), read_rows(CHAMPION), strict=True):
        if actual != reference:
            raise ValueError(f'Predictions differ from champion at query {actual.get("id")}')
        nd, nc = validate_row(actual, official)
        docs += nd; chunks += nc; found.append(actual['id'])
        payload_digest.update(json.dumps(actual, ensure_ascii=False, sort_keys=True,
                                        separators=(',', ':')).encode('utf-8') + b'\n')
    if found != ids or docs != 183703 or chunks != 127271:
        raise ValueError('Output query order/coverage or champion counts differ')
    if output.stat().st_size >= 100_000_000:
        raise ValueError('ZIP exceeds decimal 100MB')
    if sha256_file(args.queries) != query_hash:
        raise ValueError('Query file changed during replay')
    with zipfile.ZipFile(output) as archive:
        json_digest = hashlib.sha256()
        with archive.open('submission.json') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                json_digest.update(block)
    audit = {
        'schema_version': 'private_champion_replay_v1', 'audit_status': 'pass',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_mode': 'cached_inference_replay_no_new_gpu_inference',
        'queries_path': str(args.queries), 'queries_sha256': query_hash,
        'query_texts_exactly_match_cached_public_queries': True,
        'btc_private_query_reuse_confirmed_by_user': True,
        'private_deadline': '2026-11-04T23:59:00+07:00',
        'reference_experiment': 'Exp143', 'reference_public_final_score_new_evaluator': 0.3190,
        'private_score': None, 'submission_status': 'not_submitted_user_upload_required',
        'queries': len(found), 'docs': docs, 'chunks': chunks,
        'all_predictions_exactly_equal_champion': True,
        'chunk_source_validation': 'exact unchanged audited champion text; added spans rebuilt and verified by source hash/offset',
        'fresh_full_corpus_source_scan': False,
        'canonical_predictions_sha256': payload_digest.hexdigest(),
        'json_sha256': json_digest.hexdigest(),
        'reference_zip_sha256': CHAMPION_HASH,
        'output_zip': str(output), 'output_zip_sha256': sha256_file(output),
        'output_zip_bytes': output.stat().st_size,
        'corpus_sha256': sha256_file(corpus),
        'p2a_run_manifest_sha256': sha256_file(RUN_MANIFEST),
        'pairs_provenance': pair_provenance,
        'build_manifest': str(build_manifest), 'build_result': build,
        'warning': 'Public score is not a private prediction; no private slot consumed by local build.',
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['audit_status','queries','docs','chunks',
                                          'output_zip_bytes','output_zip_sha256','submission_status']}, indent=2), flush=True)
    print(f'Audit: {audit_path}', flush=True)


if __name__ == '__main__':
    main()
