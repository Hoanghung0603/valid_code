"""Move only vector-backed ZIP documents between registries, not the active index.

Dry run by default. Appending preserves the existing compressed Parquet payload
without loading its text into RAM. Backups and a pending-index manifest are kept.
Applying requires fastparquet (tested with 2026.9.0).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_local_recovered_final import REGISTRIES, OFFICIAL, digest, validate_partition


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_artifact(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP entries")
        require(archive.testzip() is None, "ZIP CRC failure")
        roots = [n[:-len('stats.json')] for n in names if n.endswith('/stats.json')]
        require(len(roots) == 1, "Expected one artifact index root")
        root = roots[0]
        stats = json.loads(archive.read(root + 'stats.json'))

        def read_parts(kind):
            parts = sorted(n for n in names if n.startswith(root + kind + '/part_') and n.endswith('.parquet'))
            require(bool(parts), f"Missing {kind} rows")
            frames = [pd.read_parquet(io.BytesIO(archive.read(n))) for n in parts]
            frame = pd.concat(frames, ignore_index=True)
            with np.load(io.BytesIO(archive.read(root + kind + '/_index.npz')), allow_pickle=False) as lookup:
                key = 'chunk_id' if kind == 'chunks' else 'doc_id'
                expected = {(int(row[key]), part, j) for part, f in enumerate(frames)
                            for j, row in enumerate(f.to_dict('records'))}
                actual = list(zip(lookup['key'].tolist(), lookup['part'].tolist(), lookup['row'].tolist()))
                require(len(actual) == len(expected) and set(actual) == expected,
                        f"{kind} lookup mismatch")
            return frame

        docs, chunks, texts = [read_parts(kind) for kind in ('docs', 'chunks', 'text')]
        for frame, key in ((docs, 'doc_id'), (texts, 'doc_id'), (chunks, 'chunk_id')):
            require(not frame[key].isna().any() and not frame[key].duplicated().any(), f"Duplicate/null {key}")
        require(not docs.url.isna().any() and not docs.url.duplicated().any(), "Duplicate/null document URL")
        doc_ids = set(docs.doc_id)
        require(doc_ids == set(texts.doc_id) == set(chunks.doc_id), "Docs/text/chunks identity mismatch")
        require(docs.set_index('doc_id').n_chunks.to_dict() == chunks.groupby('doc_id').size().to_dict(),
                "Per-document chunk count mismatch")
        text_map = texts.set_index('doc_id').text.to_dict()
        for row in chunks.itertuples(index=False):
            text = text_map[row.doc_id]
            require(isinstance(text, str) and 0 <= row.start < row.end <= len(text), "Invalid text/span")
            require(row.end - row.start == row.n_chars and bool(text[row.start:row.end].strip()), "Empty/bad span")

        vector_ids = []
        ids_parts = sorted(n for n in names if n.startswith(root + 'emb_dense/ids_') and n.endswith('.npy'))
        require(bool(ids_parts), "Missing vectors")
        for name in ids_parts:
            stem = name.rsplit('/', 1)[-1][4:-4]
            ids = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
            dense = np.load(io.BytesIO(archive.read(root + f'emb_dense/vec_{stem}.npy')), allow_pickle=False)
            require(ids.ndim == 1 and ids.dtype.kind in 'iu', "Invalid vector IDs")
            require(dense.shape == (len(ids), 1024) and np.isfinite(dense).all(), "Invalid dense vectors")
            require((np.linalg.norm(dense.astype('float32'), axis=1) > 0).all(), "Zero dense vector")
            with np.load(io.BytesIO(archive.read(root + f'emb_dense/sp_{stem}.npz')), allow_pickle=False) as sp:
                ptr, terms, vals = sp['ptr'], sp['ids'], sp['vals']
                require(ptr.shape == (len(ids) + 1,) and ptr.dtype.kind in 'iu', "Invalid CSR pointers")
                require(terms.ndim == vals.ndim == 1 and terms.dtype.kind in 'iu', "Invalid sparse arrays")
                require(ptr[0] == 0 and ptr[-1] == len(terms) == len(vals) and (np.diff(ptr) >= 0).all(),
                        "CSR alignment mismatch")
                require(np.isfinite(vals).all() and (terms >= 0).all() and (terms < 250005).all(), "Invalid sparse values")
            vector_ids.extend(ids.tolist())
        require(len(vector_ids) == len(set(vector_ids)) and set(vector_ids) == set(chunks.chunk_id),
                "Vector/chunk IDs mismatch")
        require(stats['kept'] == stats['rows_docs'] == stats['rows_text'] == len(docs), "Stats docs mismatch")
        require(stats['chunks'] == stats['rows_chunks'] == len(chunks), "Stats chunks mismatch")
        require(stats['docs'] == stats['kept'] + stats['dropped'] + stats['failed'], "Stats input mismatch")
        return docs, texts, stats


def select_moves(docs, texts, official, identities, unembedded, embedded_path):
    validate_partition(official, identities)
    require(docs.doc_id.map(official.set_index('id').url).eq(docs.url).all(), "Artifact outside official ID/URL mapping")
    ids = set(docs.doc_id.astype(int))
    known_ids = set(identities['embedded'].id) | set(unembedded.id)
    require(ids <= known_ids, "Artifact document absent from embedded/unembedded registries")
    moved = unembedded[unembedded.id.isin(ids)].copy()
    already = ids & set(identities['embedded'].id)
    existing_parts = []
    if already:
        # Broad min/max stats in old million-row groups cannot prune these IDs.
        # Inspect ID-only groups first to avoid decoding unrelated giant text.
        with pq.ParquetFile(embedded_path) as stored:
            for i in range(stored.num_row_groups):
                group_ids = stored.read_row_group(i, columns=['id']).column('id').to_pylist()
                if already.intersection(group_ids):
                    group = stored.read_row_group(i).to_pandas()
                    existing_parts.append(group[group.id.isin(already)])
    existing = pd.concat(existing_parts, ignore_index=True) if existing_parts else moved.iloc[:0]
    require(set(existing.id) == already, "Previously embedded rows not found")
    current = pd.concat([moved, existing], ignore_index=True).set_index('id')
    for row in texts.itertuples(index=False):
        require(current.loc[row.doc_id, 'text'] == row.text, f"Artifact/registry text mismatch for {row.doc_id}")
    updated_unembedded = unembedded[~unembedded.id.isin(ids)].reset_index(drop=True)
    updated = dict(identities)
    updated['embedded'] = pd.concat([identities['embedded'], moved[['id', 'url']]], ignore_index=True)
    updated['unembedded'] = updated_unembedded[['id', 'url']]
    validate_partition(official, updated)
    return moved, updated_unembedded, updated


def commit_registry_moves(paths, moved, unembedded, before_hashes, out):
    """Stage both files, verify content, then replace with rollback on failure."""
    # Public append API: https://fastparquet.readthedocs.io/en/latest/api.html
    # Arrow iter_batches still decodes large dictionaries in the original file.
    import fastparquet

    backup = out / 'backups'
    backup.mkdir(exist_ok=False)
    for name in ('embedded', 'unembedded'):
        shutil.copy2(paths[name], backup / paths[name].name)
        require(digest(backup / paths[name].name) == before_hashes[name], "Backup hash mismatch")
    staged = {name: out / ('staged_' + paths[name].name) for name in ('embedded', 'unembedded')}
    shutil.copy2(backup / paths['embedded'].name, staged['embedded'])
    append_frame = moved.copy()
    for column in append_frame.select_dtypes(include=['str', 'string']).columns:
        append_frame[column] = append_frame[column].astype(object)
    fastparquet.write(str(staged['embedded']), append_frame, append=True, write_index=False, compression='SNAPPY')
    unembedded.to_parquet(staged['unembedded'], index=False, compression='snappy')
    # Prove all old compressed pages are byte-for-byte unchanged, excluding only
    # the old footer, which must be replaced to register the appended row group.
    with (backup / paths['embedded'].name).open('rb') as old, staged['embedded'].open('rb') as new:
        old.seek(-8, 2)
        trailer = old.read(8)
        require(trailer[4:] == b'PAR1', "Invalid original Parquet footer")
        remaining = old.tell() - 8 - int.from_bytes(trailer[:4], 'little')
        require(remaining >= 4, "Invalid footer length")
        old.seek(0)
        while remaining:
            size = min(8 * 1024 * 1024, remaining)
            require(old.read(size) == new.read(size), "Original compressed payload changed")
            remaining -= size
    with pq.ParquetFile(backup / paths['embedded'].name) as old, pq.ParquetFile(staged['embedded']) as new:
        require(new.metadata.num_rows == old.metadata.num_rows + len(moved), "Staged row count mismatch")
        require(new.num_row_groups == old.num_row_groups + 1, "Unexpected row group count")
        require(new.schema_arrow.equals(old.schema_arrow), "Original schema changed")
        for i in range(old.num_row_groups):
            require(old.metadata.row_group(i).to_dict() == new.metadata.row_group(i).to_dict(), "Old row group metadata changed")
        extra = pa.Table.from_pandas(moved, schema=new.schema_arrow, preserve_index=False)
        require(extra.equals(new.read_row_group(old.num_row_groups)), "Appended content changed")
    pd.testing.assert_frame_equal(pd.read_parquet(staged['unembedded']), unembedded,
                                  check_dtype=False, check_index_type=False, check_column_type=False)
    for name, path in paths.items():
        require(digest(path) == before_hashes[name], f"Concurrent edit: {name}")
    committed = []
    try:
        for name in ('embedded', 'unembedded'):
            os.replace(staged[name], paths[name])
            committed.append(name)
    except Exception:
        for name in reversed(committed):
            shutil.copy2(backup / paths[name].name, paths[name])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    require(not out.exists(), "Use a new audit directory to preserve previous evidence")
    docs, texts, stats = read_artifact(args.artifact)
    source_hash = digest(args.artifact)
    before = {name: digest(path) for name, path in REGISTRIES.items()}
    official_hash = digest(OFFICIAL)
    official = pd.read_parquet(OFFICIAL, columns=['id', 'url'])
    identities = {name: pd.read_parquet(path, columns=['id', 'url']) for name, path in REGISTRIES.items()}
    unembedded = pd.read_parquet(REGISTRIES['unembedded'])
    moved, pending, updated = select_moves(docs, texts, official, identities, unembedded, REGISTRIES['embedded'])
    out.mkdir(parents=True)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'validated',
        'artifact': str(args.artifact.resolve()), 'artifact_sha256': source_hash,
        'artifact_stats': stats, 'official_sha256': official_hash,
        'moved_ids': moved.id.astype(int).tolist(), 'already_registered_ids': sorted(set(docs.doc_id) - set(moved.id)),
        'before_counts': {name: len(f) for name, f in identities.items()},
        'after_counts': {name: len(f) for name, f in updated.items()},
        'before_sha256': before, 'active_index_modified': False,
        'index_merge_required_for_new_documents': True,
        'checks': ['ZIP CRC', 'official ID/URL', 'exact source text', 'dense finite/nonzero 1024D',
                   'CSR alignment', 'chunk/vector ID bijection', 'spans', 'lookup tables', 'registry partition'],
    }
    manifest = out / 'manifest.json'

    def save():
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    moved.to_parquet(out / 'moved_documents.parquet', index=False)
    save()
    if args.apply and len(moved):
        require(digest(args.artifact) == source_hash and digest(OFFICIAL) == official_hash, "Source changed")
        try:
            commit_registry_moves(REGISTRIES, moved, pending, before, out)
        except Exception as error:
            report['status'] = 'aborted_before_completion'
            report['error'] = f'{type(error).__name__}: {error}'
            save()
            raise
        try:
            final = {name: pd.read_parquet(path, columns=['id', 'url']) for name, path in REGISTRIES.items()}
            validate_partition(official, final)
            for name in final:
                pd.testing.assert_frame_equal(final[name].reset_index(drop=True), updated[name].reset_index(drop=True),
                                              check_dtype=False, check_index_type=False, check_column_type=False)
        except Exception:
            for name in ('embedded', 'unembedded'):
                shutil.copy2(out / 'backups' / REGISTRIES[name].name, REGISTRIES[name])
            raise
        report['status'] = 'applied'
        report['after_sha256'] = {name: digest(path) for name, path in REGISTRIES.items()}
        report['checks'].extend(['all original compressed pages and row-group metadata unchanged',
                                 'staged parquet roundtrip', 'post-write partition'])
        save()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
