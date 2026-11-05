"""Verify PubMed cache reconciliation without network or user data."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import zstandard as zstd

from crawl.sync_pubmed_cache import reconcile
from crawl import check_retry_coverage as coverage


class SyncPubmedCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(temporary.cleanup)
        root = self.root = Path(temporary.name)
        self.cache = root / 'pubmed'
        self.cache.mkdir()
        self.db = root / 'status.duckdb'
        self.retry = root / 'retry'
        self.retry.mkdir()
        con = duckdb.connect(str(self.db))
        con.execute('''CREATE TABLE crawl_status (
            id BIGINT PRIMARY KEY,url VARCHAR,final_url VARCHAR,status VARCHAR,
            http_code INTEGER,content_type VARCHAR,n_bytes BIGINT,
            fetched_at TIMESTAMP,shard INTEGER,sha1_raw VARCHAR,error_msg VARCHAR)''')
        con.executemany('INSERT INTO crawl_status(id,url,status) VALUES (?,?,?)', [
            (1, 'https://pubmed.ncbi.nlm.nih.gov/123/', 'pending'),
            (2, 'https://pubmed.ncbi.nlm.nih.gov/456/', 'blocked'),
            (3, 'https://pubmed.ncbi.nlm.nih.gov/789/', 'pending'),
            (4, 'https://example.org/123/', 'pending'),
        ])
        con.close()

    def batch(self, name, records):
        (self.cache / name).write_bytes(zstd.ZstdCompressor().compress(
            json.dumps(records).encode('utf-8')))

    def test_dry_run_apply_and_idempotence(self):
        self.batch('batch_00000_a.json.zst', [
            {'pmid': '123', 'title': 'One'}, {'pmid': '456', 'title': 'Two'}])
        self.batch('batch_00001_b.json.zst', [{'pmid': '123', 'title': 'One'}])
        report = reconcile(self.cache, self.db, retry_out=self.retry)
        self.assertEqual((report['matched'], report['missing'], report['duplicate_records']),
                         (2, 1, 1))
        con = duckdb.connect(str(self.db), read_only=True)
        self.assertEqual(con.execute("SELECT count(*) FROM crawl_status WHERE status='ok'").fetchone()[0], 0)
        con.close()
        with self.assertRaisesRegex(ValueError, 'Coverage below'):
            reconcile(self.cache, self.db, apply=True, retry_out=self.retry)
        report = reconcile(self.cache, self.db, apply=True, min_coverage=0.5,
                           retry_out=self.retry)
        self.assertEqual(report['to_update'], 2)
        self.assertEqual(reconcile(self.cache, self.db, apply=True, min_coverage=0.5,
                                   retry_out=self.retry)['to_update'], 0)
        con = duckdb.connect(str(self.db), read_only=True)
        try:
            self.assertEqual(con.execute("SELECT count(*) FROM crawl_status WHERE status='ok'").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT status FROM crawl_status WHERE id=3").fetchone()[0], 'pending')
            self.assertEqual(con.execute("SELECT batch_file FROM pubmed_cache_index WHERE id=1").fetchone()[0],
                             'batch_00000_a.json.zst')
            self.assertEqual(con.execute('SELECT count(*) FROM pubmed_cache_index').fetchone()[0], 2)
        finally:
            con.close()

    def test_rejects_record_without_content(self):
        self.batch('batch_00000_a.json.zst', [{'pmid': '123'}])
        with self.assertRaisesRegex(ValueError, 'Missing PMID/title/abstract'):
            reconcile(self.cache, self.db, retry_out=self.retry)

    def test_coverage_counts_indexed_api_batch_as_available(self):
        self.batch('batch_00000_a.json.zst', [
            {'pmid': '123', 'title': 'One'}, {'pmid': '456', 'title': 'Two'}])
        reconcile(self.cache, self.db, apply=True, min_coverage=0.5,
                  retry_out=self.retry)
        (self.retry / 'status.csv').write_text('id,url,status\n', encoding='utf-8')
        html = self.root / 'html'
        html.mkdir()
        output = io.StringIO()
        with patch.object(coverage, 'ROOT', self.root), \
             patch.object(coverage, 'DATABASE', self.db), \
             patch.object(coverage, 'HTML', html), \
             patch.object(coverage, 'STAGED', self.retry), \
             redirect_stdout(output):
            coverage.main()
        self.assertIn('PubMed API đã đồng bộ và có batch tại máy: 2', output.getvalue())
        self.assertIn('Còn thiếu dữ liệu crawl: 2', output.getvalue())


if __name__ == '__main__':
    unittest.main()
