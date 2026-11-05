"""Audit recovered embedding ZIPs, rebuild once, then synchronize registries.

All derived search artifacts are rebuilt in a separate staging directory.
Original index shards are hard-linked read-only; the old index is retained after
activation. Registry writes have backups/rollback and preserve compressed pages.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import merge_embedded_artifact as merge
from scripts.audit_local_recovered_final import REGISTRIES, OFFICIAL, digest, validate_partition
from scripts.sync_embedded_zip_registry import read_artifact, select_moves, commit_registry_moves
from scripts.streaming_index_rebuild import rebuild_dense, rebuild_sparse, check


def snapshot(path):
    return {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in path.rglob('*') if p.is_file()}


def audit_inputs(sources, out):
    registry_hashes = {n: digest(p) for n, p in REGISTRIES.items()}
    official_hash = digest(OFFICIAL)
    official = pd.read_parquet(OFFICIAL, columns=['id', 'url'])
    identities = {n: pd.read_parquet(p, columns=['id', 'url']) for n, p in REGISTRIES.items()}
    unembedded = pd.read_parquet(REGISTRIES['unembedded'])
    state = merge.load_active_state()
    stats, active_docs, chunk_offset, output_part, text_part, _ = state
    check(active_docs <= set(official.id), 'Active index contains IDs outside official corpus')
    check(stats['model'] == 'BAAI/bge-m3' and stats['dense_dim'] == 1024 and stats['vocab_size'] == 250005,
          'Unsupported active embedding format/model')
    plans, doc_frames, text_frames = [], [], []
    seen = set()
    for source in sources:
        sha = digest(source)
        docs, texts, source_stats = read_artifact(source)
        check(digest(source) == sha, 'ZIP changed during audit')
        source_ids = set(docs.doc_id.astype(int))
        check(not source_ids & seen, 'Duplicate doc_id across input artifacts')
        check(docs.doc_id.map(official.set_index('id').url).eq(docs.url).all(), 'Artifact outside official ID/URL mapping')
        seen.update(source_ids)
        # Inspect raw offsets/order; read_artifact already validates vectors and text.
        _, root, chunks, chunk_docs, outside, overlap = merge.audit_source(source, set(official.id), active_docs)
        check(source_ids == chunk_docs and not outside, 'Artifact document/chunk identities mismatch')
        keep = source_ids - active_docs
        plans.append(dict(source=source, sha256=sha, stats=source_stats, root=root,
                          chunks=chunks, keep=keep, overlap=overlap))
        doc_frames.append(docs)
        text_frames.append(texts)
        print(f'{source.name}: {len(docs):,} valid docs, {len(chunks):,} chunks, {len(overlap):,} already indexed', flush=True)
    docs = pd.concat(doc_frames, ignore_index=True)
    texts = pd.concat(text_frames, ignore_index=True)
    moved, pending, updated = select_moves(docs, texts, official, identities, unembedded, REGISTRIES['embedded'])
    final_docs = active_docs | seen
    check(final_docs == set(updated['embedded'].id), 'Planned embedded registry differs from final index IDs')
    docs[['doc_id', 'url', 'n_chunks']].to_parquet(out / 'artifact_documents.parquet', index=False)
    moved.to_parquet(out / 'registry_rows_to_move.parquet', index=False)
    return (official, official_hash, identities, registry_hashes, state, plans,
            moved, pending, updated, final_docs)


def verify_index(index, expected_docs, total_chunks):
    metadata = pq.read_table(index / 'chunks_metadata.parquet')
    check(len(metadata) == total_chunks and set(metadata['doc_id'].to_pylist()) == expected_docs,
          'Rebuilt metadata document set mismatch')
    ids = np.asarray(metadata['chunk_id'])
    check(np.array_equal(ids, np.arange(total_chunks)), 'Rebuilt chunk IDs not contiguous')
    check(np.array_equal(ids, np.load(index / 'chunk_ids.npy', allow_pickle=False)), 'Global chunk ID lookup mismatch')
    text_ids = []
    for p in sorted((index / 'text').glob('part_*.parquet')):
        text_ids.extend(pq.read_table(p, columns=['doc_id'])['doc_id'].to_pylist())
    check(len(text_ids) == len(expected_docs) and set(text_ids) == expected_docs,
          'Text store has missing/duplicate/outside documents')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    check(not out.exists(), 'Audit directory already exists; preserve previous evidence')
    out.mkdir(parents=True)
    sources = [p.resolve() for p in args.source]
    check(len(set(sources)) == len(sources), 'Duplicate source ZIP arguments')
    old_snapshot = snapshot(merge.ACTIVE_INDEX)
    inputs = audit_inputs(sources, out)
    official, official_hash, identities, before_hashes, state, plans, moved, pending, updated, final_docs = inputs
    stats, _, chunk_offset, output_part, text_part, _ = state
    now = datetime.now(ZoneInfo('Asia/Bangkok'))
    staging = merge.INDEX_ROOT / ('index_recovered_merge_staging_' + now.strftime('%Y%m%dT%H%M%S'))
    backup = merge.INDEX_ROOT / ('index_backup_pre_recovered_merge_' + now.strftime('%Y%m%dT%H%M%S'))
    report = dict(status='audited', created_at=now.isoformat(), staging=str(staging), backup_index=str(backup),
                  official_sha256=official_hash, before_index_stats=stats, before_registry_sha256=before_hashes,
                  before_registry_counts={n: len(f) for n, f in identities.items()},
                  after_registry_counts={n: len(f) for n, f in updated.items()},
                  registry_rows_to_move=len(moved), registry_already_embedded_docs=sum(len(p['keep']) for p in plans) - len(moved),
                  artifacts=[dict(path=str(p['source']), sha256=p['sha256'], stats=p['stats'],
                                  added_docs=len(p['keep']), skipped_indexed_docs=len(p['overlap'])) for p in plans])
    manifest = out / 'manifest.json'

    def save():
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    swapped = False
    registry_committed = False
    try:
        merge.STAGING_INDEX = staging
        merge.prepare_staging()
        for plan in plans:
            added, output_part = merge.append_source_shards(
                plan['source'], plan['root'], plan['chunks'], plan['keep'], chunk_offset,
                output_part, stats['dense_dim'], stats['vocab_size'])
            text_part = merge.append_source_text(plan['source'], plan['root'], plan['keep'], text_part)
            chunk_offset += added
            plan['added_chunks'] = added
        gc.collect()
        report['status'] = 'rebuilding'
        save()
        report['dense_rebuild'] = rebuild_dense(staging, chunk_offset, stats['dense_dim'])
        report['sparse_rebuild'] = rebuild_sparse(staging, chunk_offset, stats['vocab_size'])
        verify_index(staging, final_docs, chunk_offset)
        new_stats = {**stats, 'total_docs': len(final_docs), 'total_chunks': chunk_offset,
                     'emb_parts': output_part, 'chunks_parts': output_part, 'text_parts': text_part,
                     'updated_at': datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(),
                     'source': stats.get('source', '') + '; merged ' + ', '.join(p.name for p in sources),
                     'latest_added_valid_docs': sum(len(p['keep']) for p in plans),
                     'latest_added_valid_chunks': sum(p['added_chunks'] for p in plans),
                     'latest_filtered_outside_docs': 0,
                     'latest_skipped_overlap_docs': sum(len(p['overlap']) for p in plans)}
        report['after_index_stats'] = new_stats
        report['status'] = 'staging_verified'
        (staging / 'stats.json').write_text(json.dumps(new_stats, ensure_ascii=False, indent=2), encoding='utf-8')
        save()
        check(snapshot(merge.ACTIVE_INDEX) == old_snapshot, 'Concurrent active index change')
        check(digest(OFFICIAL) == official_hash, 'Official corpus changed')
        for p in plans:
            check(digest(p['source']) == p['sha256'], 'Source artifact changed')
        for name, path in REGISTRIES.items():
            check(digest(path) == before_hashes[name], f'Concurrent registry change: {name}')
        check(not backup.exists(), 'Backup index path already exists')
        merge.ACTIVE_INDEX.rename(backup)
        try:
            staging.rename(merge.ACTIVE_INDEX)
        except Exception:
            backup.rename(merge.ACTIVE_INDEX)
            raise
        swapped = True
        report['status'] = 'index_activated_registry_pending'
        save()
        if len(moved):
            commit_registry_moves(REGISTRIES, moved, pending, before_hashes, out)
            registry_committed = True
        final = {n: pd.read_parquet(p, columns=['id', 'url']) for n, p in REGISTRIES.items()}
        validate_partition(official, final)
        check(set(final['embedded'].id) == final_docs, 'Active index/embedded registry mismatch')
        for name in final:
            pd.testing.assert_frame_equal(final[name].reset_index(drop=True), updated[name].reset_index(drop=True),
                                          check_dtype=False, check_index_type=False, check_column_type=False)
        for name in ('remaining', 'excluded'):
            check(digest(REGISTRIES[name]) == before_hashes[name], f'{name} unexpectedly changed')
        check(snapshot(backup) == old_snapshot, 'Backup index contents changed')
        report['after_registry_sha256'] = {n: digest(p) for n, p in REGISTRIES.items()}
        report['status'] = 'complete'
        report['completed_at'] = datetime.now(ZoneInfo('Asia/Bangkok')).isoformat()
        report['cache_policy'] = 'Old retrieval/CE/submission caches were retained as history; rerun or fingerprint-check before reuse against this index.'
        save()
        shutil.copy2(manifest, merge.ACTIVE_INDEX / 'rebuild_manifest.json')
    except Exception as error:
        if registry_committed:
            for name in ('embedded', 'unembedded'):
                shutil.copy2(out / 'backups' / REGISTRIES[name].name, REGISTRIES[name])
        if swapped:
            merge.ACTIVE_INDEX.rename(staging)
            backup.rename(merge.ACTIVE_INDEX)
        report['status'] = 'failed_original_state_restored' if swapped else 'failed_before_activation'
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
