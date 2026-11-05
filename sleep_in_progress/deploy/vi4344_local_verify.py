"""Independent CPU-only final local ZIP audit against downloaded b660 report."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import vi44 as v
import vi43
from vi41c import indexed, load_zip

BASE = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', choices=['vi44', 'vi43'])
    name = ap.parse_args().name
    out = BASE/name
    zp = out/f'submission_{name}.zip'
    report = json.loads((out/f'build_report_{name}.json').read_text(encoding='utf-8'))
    fp = v.v.fingerprint(zp)
    assert fp['bytes'] == report['zip_bytes'] <= v.v.LIMIT
    assert fp['sha256'] == report['sha256']
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json']
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert z.testzip() is None
        rows = json.loads(z.read('submission.json'))
    assert len(rows) == 1200
    assert set(indexed(rows)) == set(range(1, 1201))
    for row in rows:
        docs = list(map(str, row['relevant_docs']))
        assert len(docs) == len(set(docs))
        pairs = [(str(c['doc_id']), c['chunk_text']) for c in row['relevant_chunks']]
        assert len(pairs) == len(set(pairs))
        assert {d for d, t in pairs} <= set(docs)
    ref37 = load_zip(BASE/'vi37/submission_vi37.zip')
    ref42 = load_zip(BASE/'vi42/submission_vi42.zip')
    if name == 'vi44':
        bs = v.v.dw.read(out/'bridges_vi44.json.zst')
        for q, items in bs.items():
            assert items == sorted(items, key=lambda b: (-b['score'], b['rank'], b['boundary_index']))
            assert all(b['score'] >= 0 for b in items)
        top50 = {q: [b for b in items if b['rank'] <= 50] for q, items in bs.items()}
        assert top50 == v.v.dw.read(BASE/'vi42/bridges_vi42.json.zst')
        sanity = v.v.compare_by_id(v.v.append_bridges(ref37, top50), ref42)
        assert sanity == {'count': 1200, 'docs': 1200, 'chunks': 1200}
        result = v.check_relations(rows, ref37, ref42, bs, report['threshold'])
    else:
        cp = BASE/'vi41c/submission_vi41c.zip'
        assert v.v.fingerprint(cp)['sha256'] == vi43.SOURCE_SHA
        expected, result = vi43.combine(ref37, ref42, load_zip(cp))
        assert v.v.compare_by_id(rows, expected) == {'count': 1200, 'docs': 1200, 'chunks': 1200}
    result.update({**fp, 'ids': 1200, 'duplicates': 0, 'chunks_outside_docs': 0,
                   'entry': 'submission.json', 'compression': 'LZMA', 'crc': 0,
                   'b660_sha_match': True,
                   'all_chunk_verbatim': 'fully verified on b660; not repeated locally',
                   'assert_clean_bad_ids_corpus': 'fully verified on b660; not repeated locally'})
    (out/'local_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
