"""CPU-only independent final ZIP audit; read-only original vi37."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import vi42


def main():
    dest = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi42')
    path = dest/'submission_vi42.zip'
    report = json.loads((dest/'build_report_vi42.json').read_text(encoding='utf-8'))
    fp = vi42.fingerprint(path)
    assert fp['sha256'] == report['sha256'] and fp['bytes'] == report['zip_bytes'] <= vi42.LIMIT
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ['submission.json']
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert z.testzip() is None
        sub = json.loads(z.read('submission.json'))
    with zipfile.ZipFile(dest.parent/'vi37/submission_vi37.zip') as z:
        ref = json.loads(z.read('submission.json'))
    cmp = vi42.compare_by_id(sub, ref)
    assert cmp['count'] == cmp['docs'] == 1200
    old = {str(it['id']): it for it in ref}
    bridges = vi42.dw.read(dest/'bridges_vi42.json.zst')
    threshold = report['threshold']
    prefix = duplicates = 0
    for it in sub:
        before = old[str(it['id'])]['relevant_chunks']
        prefix += it['relevant_chunks'][:len(before)] == before
        rows = bridges[str(it['id'])]
        assert rows == sorted(rows, key=lambda b: (-b['score'], b['rank'], b['boundary_index']))
        assert all(b['score'] >= 0 and 1 <= b['rank'] <= 50 for b in rows)
        expected = [{'doc_id': b['doc_id'], 'chunk_text': b['chunk_text']} for b in rows
                    if threshold is not None and b['score'] >= threshold]
        assert it['relevant_chunks'][len(before):] == expected
        pairs = [(str(c['doc_id']), c['chunk_text']) for c in it['relevant_chunks']]
        duplicates += len(pairs)-len(set(pairs))
        assert all(d in set(map(str, it['relevant_docs'])) for d, t in pairs)
    assert prefix == 1200 and duplicates == 0
    assert {int(it['id']) for it in sub} == set(range(1, 1201))
    result = {**fp, 'ids': 1200, 'doclists_equal': cmp['docs'], 'original_chunk_prefix': prefix,
              'duplicates': duplicates, 'entry': 'submission.json', 'compression': 'LZMA', 'crc': 0,
              'bridge_manifest_order_and_append': 1200,
              'all_chunk_verbatim': 'verified fully on b660 text export; not repeated locally'}
    (dest/'local_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
