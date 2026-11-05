"""Independently verify a document-preserving evidence submission against source."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.utils.submission_io import iter_json_array, sha256_file, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'submission_exp74_full1212k_exp73_refresh.json')
    parser.add_argument('--index-dir',type=Path,default=ROOT/'data/indices/new_dataset/index')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    official=set(pd.read_parquet(ROOT/'data/raw/btc_provided/links_corpus_vi_en_zh.parquet',columns=['id']).id.astype(int))
    expected={int(q['id']):q['relevant_docs'] for q in iter_json_array(args.baseline)}
    needed=defaultdict(set)
    seen=set()
    total_chunks=0
    outside_docs=0
    for q in iter_json_array(args.submission):
        if type(q['id']) is not int or q['id'] in seen:raise ValueError('Invalid query ID')
        seen.add(q['id'])
        if set(q)!=set(('id','relevant_docs','relevant_chunks')):raise ValueError('Unexpected schema')
        if q['relevant_docs']!=expected[q['id']]:raise ValueError('Document axis changed')
        if any(type(d) is not int for d in q['relevant_docs']):raise ValueError('Non-integer document ID')
        if len(set(q['relevant_docs']))!=216 or not set(q['relevant_docs'])<=official:raise ValueError('Invalid documents')
        represented=set()
        for c in q['relevant_chunks']:
            if set(c)!=set(('doc_id','chunk_text')):raise ValueError('Invalid chunk schema')
            doc=c['doc_id'];text=c['chunk_text']
            if type(doc) is not int or doc not in official or doc in represented:raise ValueError('Invalid or repeated chunk document')
            if type(text) is not str or not text.strip():raise ValueError('Empty chunk')
            represented.add(doc);needed[doc].add(text);total_chunks+=1
        outside_docs+=len(represented-set(q['relevant_docs']))
    if seen!=set(expected) or len(seen)!=1200:raise ValueError('Missing queries')
    verified=set()
    for path in sorted((args.index_dir/'text').glob('part_*.parquet')):
        frame=pd.read_parquet(path,columns=['doc_id','text'])
        for doc,source in frame[frame.doc_id.isin(needed)].itertuples(index=False,name=None):
            doc=int(doc)
            if doc in verified:raise ValueError('Duplicate source document')
            for passage in needed[doc]:
                if passage not in source:raise ValueError(f'Non-verbatim passage for {doc}')
            verified.add(doc)
        if len(verified)==len(needed):break
    if verified!=set(needed):raise ValueError('Missing source text')
    archive_path=args.submission.with_suffix('.zip')
    if archive_path.stat().st_size>=100_000_000:raise ValueError('ZIP exceeds hard limit')
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist()!=['submission.json'] or archive.testzip() is not None:raise ValueError('Invalid ZIP members or CRC')
        digest=hashlib.sha256()
        with archive.open('submission.json') as handle:
            while block:=handle.read(4*1024*1024):digest.update(block)
        if digest.hexdigest()!=sha256_file(args.submission):raise ValueError('ZIP payload differs from JSON')
    report=dict(status='passed',queries=len(seen),documents_equal_exp74=True,
                chunk_document_pairs=total_chunks,unique_source_docs=len(needed),
                chunk_docs_outside_selected_docs=outside_docs,
                official_ids_only=True,unique_chunk_doc_per_query=True,all_chunks_verbatim=True,
                zip_crc_payload_passed=True,json_sha256=sha256_file(args.submission),zip_sha256=sha256_file(archive_path),
                zip_bytes=archive_path.stat().st_size)
    write_json(args.report,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
