"""Offline checks for notebook 03's NCBI PubMed completion cell."""

import ast
import hashlib
import html
import io
import json
import logging
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import duckdb
import httpx
import pyarrow.parquet as pq
import zstandard as zstd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks' / '03_gold_review_and_sample2.ipynb'
SOURCE = ROOT / 'notebooks' / 'pubmed_completion_cell.py'
PUBMED_ONLY = ROOT / 'notebooks' / '04_pubmed_only.ipynb'
CLEANER_SOURCE = ROOT / 'notebooks' / 'pubmed_only_cleaner.py'
RESCUE_NOTEBOOK = ROOT / 'notebooks' / '05_pubmed_rescue.ipynb'
RESCUE_SOURCE = ROOT / 'notebooks' / 'pubmed_rescue_cell.py'


def cell(notebook, name):
    return ''.join(next(item for item in notebook['cells'] if item['id'] == name)['source'])


def cleaner_from_notebook(notebook):
    tree = ast.parse(cell(notebook, 'gold-extractors'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'extract_pubmed_doc')
    safe_tags = re.compile(r'</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>')
    namespace = dict(Dict=Dict, Any=Any, hashlib=hashlib, re=re,
                     normalize_ws=lambda value: re.sub(r'\s+', ' ', value).strip(),
                     strip_html_tags_safe=lambda value: html.unescape(safe_tags.sub('', value)),
                     parse_published_date=lambda value: value)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'pubmed_cleaner.py', 'exec'), namespace)
    return namespace['extract_pubmed_doc']


class FakeNCBI:
    def __init__(self, xml):
        self.xml = xml
        self.calls = []

    def get(self, url, *, params):
        self.calls.append((url, dict(params)))
        request = httpx.Request('GET', url, params=params)
        return httpx.Response(200, content=self.xml, request=request)


class PubMedCompletionTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.helper = cell(self.notebook, 'pubmed-completion-helpers')
        self.namespace = {'__name__': 'pubmed_notebook_test'}
        exec(compile(self.helper, 'pubmed_completion_cell.py', 'exec'), self.namespace)
        self.cleaner = cleaner_from_notebook(self.notebook)

    def test_cell_source_and_syntax(self):
        self.assertEqual(self.helper, SOURCE.read_text(encoding='utf-8'))
        for item in self.notebook['cells']:
            if item['cell_type'] == 'code':
                ast.parse(''.join(item['source']))
        doc = self.cleaner({'title': 'Title.. ', 'keywordList': {'keyword': ['one']}},
                           1, 'https://pubmed.ncbi.nlm.nih.gov/1/')
        self.assertEqual(doc['title'], 'Title.')
        self.assertEqual(doc['quality_flags'], ['no_abstract'])

    def test_pubmed_only_notebook_has_no_gold_gate_and_same_cleaner(self):
        standalone = json.loads(PUBMED_ONLY.read_text(encoding='utf-8'))
        self.assertEqual([item['id'] for item in standalone['cells']],
                         ['pubmed-only-intro', 'pubmed-only-run'])
        source = cell(standalone, 'pubmed-only-run')
        ast.parse(source)
        self.assertNotIn('require_review_gold_pass', source)
        self.assertIn(CLEANER_SOURCE.read_text(encoding='utf-8'), source)
        self.assertIn(SOURCE.read_text(encoding='utf-8'), source)
        self.assertNotIn('require_review_gold_pass()', cell(self.notebook, 'pubmed-completion-run'))
        namespace = {}
        exec(compile(CLEANER_SOURCE.read_text(encoding='utf-8'),
                     'pubmed_only_cleaner.py', 'exec'), namespace)
        records = [
            {'title': 'Title.. ', 'keywordList': {'keyword': ['one']}},
            {'title': 'Study &amp; survey.', 'abstractText': '<h4>BACKGROUND</h4>Value &lt;6',
             'firstPublicationDate': '2020-10-21',
             'journalInfo': {'journal': {'title': 'Medical Journal'}}},
        ]
        for record in records:
            with self.subTest(record=record):
                self.assertEqual(namespace['extract_pubmed_doc'](record, 1, 'https://example.test/'),
                                 self.cleaner(record, 1, 'https://example.test/'))

    def test_targeted_retry_recovers_only_returned_pmids_and_resumes(self):
        rescue_nb = json.loads(RESCUE_NOTEBOOK.read_text(encoding='utf-8'))
        code = cell(rescue_nb, 'pubmed-rescue-run')
        ast.parse(code)
        self.assertNotIn('require_review_gold_pass', code)
        self.assertIn(RESCUE_SOURCE.read_text(encoding='utf-8'), code)
        exec(compile(RESCUE_SOURCE.read_text(encoding='utf-8'),
                     'pubmed_rescue_cell.py', 'exec'), self.namespace)

        class PartialNCBI:
            def __init__(self):
                self.calls = []

            def get(self, url, *, params):
                ids = params['id'].split(',')
                self.calls.append(ids)
                returned = ['1'] if len(ids) > 1 else (['2'] if ids == ['2'] else [])
                articles = ''.join(
                    f'<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID>'
                    f'<Article><ArticleTitle>Title {pmid}</ArticleTitle></Article>'
                    '</MedlineCitation></PubmedArticle>' for pmid in returned)
                request = httpx.Request('GET', url, params=params)
                return httpx.Response(200, content=(f'<PubmedArticleSet>{articles}'
                                                    '</PubmedArticleSet>').encode(), request=request)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pubmed = root / 'pubmed'
            pubmed.mkdir()
            self.namespace['time'].sleep = lambda _: None
            first = PartialNCBI()
            result = self.namespace['retry_ncbi_missing'](
                pubmed, ['1', '2', '3'], root / 'reports', client=first)
            self.assertEqual((result['recovered'], result['unresolved_pmids']), (2, ['3']))
            self.assertEqual(first.calls, [['1', '2', '3'], ['2'], ['3']])
            self.assertEqual(len(list(pubmed.glob('ncbi_efetch_retry_*.xml.zst'))), 3)
            second = PartialNCBI()
            again = self.namespace['retry_ncbi_missing'](
                pubmed, ['1', '2', '3'], root / 'reports', client=second)
            self.assertEqual(again['recovered'], 2)
            self.assertEqual(second.calls, [])

    def test_ncbi_229_pmids_use_two_requests_and_cached_results(self):
        with tempfile.TemporaryDirectory() as folder:
            pubmed = Path(folder)
            pmids = [str(i) for i in range(1, 230)]
            fake = FakeNCBI(b'<PubmedArticleSet/>')
            self.namespace['time'].sleep = lambda _: None
            self.assertEqual(self.namespace['_fetch_ncbi_missing'](pubmed, pmids, client=fake), 2)
            self.assertEqual([len(call[1]['id'].split(',')) for call in fake.calls], [200, 29])
            self.assertEqual(len(list(pubmed.glob('ncbi_efetch_*.xml.zst'))), 2)
            again = FakeNCBI(b'<PubmedArticleSet/>')
            self.assertEqual(self.namespace['_fetch_ncbi_missing'](pubmed, pmids, client=again), 0)
            self.assertEqual(again.calls, [])

    def test_fetch_sync_and_clean_all_ids(self):
        xml = b'''<PubmedArticleSet><PubmedArticle><MedlineCitation>
          <PMID>103</PMID><Article><ArticleTitle>NCBI study.</ArticleTitle>
          <Abstract><AbstractText Label="BACKGROUND">Value &lt;6 mU/l matters.</AbstractText>
          </Abstract><Journal><JournalIssue><PubDate><Year>2020</Year>
          <Month>Oct</Month><Day>21</Day></PubDate></JournalIssue>
          <Title>Medical Journal</Title></Journal></Article>
          <KeywordList><Keyword>first</Keyword><Keyword>second</Keyword></KeywordList>
          </MedlineCitation></PubmedArticle></PubmedArticleSet>'''
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pubmed = root / 'pubmed'
            pubmed.mkdir()
            docs = root / 'docs'
            report_dir = root / 'reports'
            db = root / 'status.duckdb'
            con = duckdb.connect(str(db))
            con.execute('''CREATE TABLE crawl_status (
                id BIGINT, url VARCHAR, status VARCHAR, final_url VARCHAR,
                http_code INTEGER, content_type VARCHAR, n_bytes BIGINT,
                fetched_at TIMESTAMP, shard BIGINT, sha1_raw VARCHAR,
                error_msg VARCHAR, attempts INTEGER)''')
            con.executemany('INSERT INTO crawl_status(id,url,status,attempts) VALUES (?,?,?,0)',
                            [(pmid, f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/', 'pending')
                             for pmid in (101, 102, 103, 104)])
            con.close()
            records = [
                {'pmid': '101', 'title': 'Europe PMC one.', 'abstractText': 'Abstract one.'},
                {'pmid': '102', 'title': 'Europe PMC two.', 'abstractText': ''},
            ]
            batch = pubmed / 'batch_00000_test.json.zst'
            batch.write_bytes(zstd.ZstdCompressor().compress(json.dumps(records).encode()))
            fake = FakeNCBI(xml)
            self.namespace['time'].sleep = lambda _: None
            result = self.namespace['run_pubmed_completion'](
                db, pubmed, docs, report_dir, self.cleaner, client=fake)
            self.assertEqual(result['content_docs'], 3)
            self.assertEqual(result['placeholder_docs'], 1)
            self.assertEqual(result['ncbi_new_batches'], 1)
            self.assertEqual(fake.calls[0][1]['id'], '103,104')
            self.assertTrue(Path(result['status_backup']).is_file())
            self.assertEqual(len(list(pubmed.glob('ncbi_efetch_*.xml.zst'))), 1)
            con = duckdb.connect(str(db), read_only=True)
            try:
                self.assertEqual(con.execute('SELECT id,status FROM crawl_status ORDER BY id').fetchall(),
                                 [(101, 'ok'), (102, 'ok'), (103, 'ok'), (104, 'http_error')])
                self.assertEqual(con.execute('SELECT count(*) FROM pubmed_cache_index').fetchone()[0], 3)
            finally:
                con.close()
            rows = []
            for path in Path(result['output']).glob('docs_*.parquet'):
                rows.extend(pq.read_table(path).to_pylist())
            self.assertEqual(len(rows), 4)
            by_id = {row['id']: row for row in rows}
            self.assertIn('BACKGROUND: Value <6 mU/l matters.', by_id[103]['text'])
            self.assertIn('Keywords: first; second.', by_id[103]['text'])
            self.assertEqual(by_id[103]['published_date'], '2020-10-21')
            self.assertEqual(by_id[103]['extractor'], 'pubmed_ncbi_efetch_v1')
            self.assertEqual(by_id[102]['quality_flags'], ['no_abstract'])
            self.assertEqual(by_id[104]['extractor'], 'placeholder')
            self.assertEqual(by_id[104]['text'], '')
            for row in rows:
                self.assertEqual(row['text'], '\n\n'.join(b['text'] for b in row['blocks']))
            again = FakeNCBI(xml)
            self.namespace['run_pubmed_completion'](
                db, pubmed, docs, report_dir, self.cleaner, client=again)
            self.assertEqual(again.calls, [])
            con = duckdb.connect(str(db), read_only=True)
            try:
                self.assertEqual(con.execute('SELECT attempts FROM crawl_status WHERE id=104').fetchone()[0], 1)
            finally:
                con.close()


if __name__ == '__main__':
    unittest.main()
