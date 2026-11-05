"""Export untouched vi51 part order from the identical index/shard/clean pipeline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from vi48 import fingerprint, read, write
from vi51 import iterz


def prepare(root, out):
    import pyarrow.parquet as pq
    import zstandard as zstd
    from bigindex import BigIndex
    from boilerplate import BoilerplateFilter
    from dwmax_diag import safe_index
    from export_docs import iter_all_texts
    start = time.monotonic()
    original = root/'data/sub/vi51'
    prep = read(original/'prepare_report.json')
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'prepare_report.json').exists()
    inp = out/'modal_input'; inp.mkdir(exist_ok=True)
    orders = {}; all_ids = set(); before = {}
    for name, meta in sorted(prep['parts'].items()):
        p = original/'modal_input'/name
        assert fingerprint(p) == prep['files'][name]
        before[name] = fingerprint(p)
        ids = [r['id'] for r in iterz(p)]
        assert len(ids) == meta['docs'] and len(set(ids)) == len(ids)
        assert not all_ids.intersection(ids)
        all_ids.update(ids); orders[name.replace('docs51.', 'docs54.')] = ids
    assert len(orders) == 29 and len(all_ids) == 223065
    index = root/'data/index/vi36'; safe_index(index); idx = BigIndex(index)
    positions = {int(d): i for i, d in enumerate(idx.ids)}
    assert all_ids <= positions.keys()
    bp = BoilerplateFilter.from_json((index/'boilerplate.json').read_text(encoding='utf-8'))
    # Exact same helper, first status=200 record, title + clean, unlimited chars.
    texts = {}
    for d, _, text in iter_all_texts(idx, {d: positions[d] for d in all_ids}, bp, 2**31-1):
        assert d not in texts and text.strip()
        texts[d] = text
    assert texts.keys() == all_ids
    parts = {}
    for name, ids in orders.items():
        p = inp/name
        with zstd.open(p, 'wt', encoding='utf-8') as f:
            for d in ids:
                f.write(json.dumps(dict(id=d, text=texts[d]), ensure_ascii=False)+'\n')
        actual = list(iterz(p))
        assert [r['id'] for r in actual] == ids and all(r['text'].strip() for r in actual)
        parts[name] = dict(**fingerprint(p), docs=len(ids),
                           id_order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                           exact_vi51_order=True, nonempty=True)
        print('RAW_PART '+json.dumps(parts[name]), flush=True)
    qs = pq.read_table(root/'data/query.parquet').to_pylist()
    queries = [dict(id=int(q['id']), text=q['query']) for q in qs]
    assert len(queries) == 1200 and {q['id'] for q in queries} == set(range(1,1201))
    assert all(q['text'].strip() for q in queries)
    write(inp/'queries54.json', queries)
    pool = read(original/'modal_input/pool.json'); write(inp/'pool.json', pool)
    assert set(all_ids) == {d for ds in pool.values() for d in ds}
    assert all(fingerprint(original/'modal_input'/n) == m for n,m in before.items())
    report = dict(documents=len(all_ids), queries=len(qs), pairs=sum(map(len,pool.values())),
                  source_index=str(index), source='BigIndex.shard_of immutable id; first status=200',
                  cleaning='export_docs.iter_all_texts + BoilerplateFilter.clean + make_body(title,clean,2**31-1)',
                  source_query=str(root/'data/query.parquet'), parts=parts,
                  files={p.name:fingerprint(p) for p in inp.iterdir()}, vi51_unchanged=True,
                  exact_part_order=True, nonempty=True, seconds=time.monotonic()-start)
    write(out/'prepare_report.json', report)
    print('RAW_DONE '+json.dumps(report), flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('D:/r2ai'))
    ap.add_argument('--out',type=Path,default=Path('D:/r2ai/data/sub/vi54'))
    a=ap.parse_args()
    from dwmax_diag import owned_lock
    lock=a.root/'locks/heavy_job.lock'
    while lock.exists(): time.sleep(10)
    with owned_lock(lock, f'vi54 raw export pid={os.getpid()}'):
        prepare(a.root,a.out)
