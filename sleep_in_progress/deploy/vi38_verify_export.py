"""Read-only post-export audit: source hashes, tokenizer caps, manifests and blind leakage."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vi38 as v


def main():
    from tokenizers import Tokenizer
    dest = v.OUT/'export'
    manifest = v.dw.read(dest/'manifest.json')
    for name, fp in manifest['files'].items():
        p = v.OUT/name if name == 'tokenizer.json' else dest/name
        assert v.fingerprint(p) == fp, name
    assert v.fingerprint(v.OLD/'submission_vi37.json') == manifest['source']
    tok = Tokenizer.from_file(str(v.OUT/'tokenizer.json'))
    rows = v.dw.read(dest/'pairs.json')
    counts = {'head_over256': 0, 'window_over512': 0, 'prompt_over4095': 0,
              'absent_owner_nonempty': 0}
    for r in rows:
        head = r['prompt'].split('Đầu bài: ', 1)[1].split('\nĐoạn bằng chứng (có thể nằm xa đầu bài): ', 1)[0]
        win = r['prompt'].split('\nĐoạn bằng chứng (có thể nằm xa đầu bài): ', 1)[1].rsplit('\nChỉ in một chữ số', 1)[0]
        counts['head_over256'] += len(tok.encode(head, add_special_tokens=False).ids) > 256
        counts['window_over512'] += len(tok.encode(win, add_special_tokens=False).ids) > 512
        counts['prompt_over4095'] += r['n_tokens'] >= 4096
        counts['absent_owner_nonempty'] += r['full_doc_tokens'] > 0 and not r['window_owner']
    blind = v.dw.read(dest/'blind.json')
    assert len(blind) == 1024 and len({r['code'] for r in blind}) == 1024
    assert all(set(r) == {'code', 'qid', 'query', 'doc_id', 'text', 'truncated'} for r in blind)
    assert all(not n for n in counts.values()), counts
    result = dict(manifest_files=len(manifest['files']), selected=len(rows), **counts,
                  blind=1024, source_unchanged=True,
                  unique_docs_selected=len({r['doc_id'] for r in rows}),
                  unique_docs_with_bench=len({r['doc_id'] for r in rows+v.dw.read(dest/'bench.json')}),
                  blind_key=v.fingerprint(v.OUT/'private/blind_key.json'),
                  manifest=v.fingerprint(dest/'manifest.json'))
    v.dw.write(v.OUT/'verify_export_report.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import os
    with v.dw.owned_lock(v.ROOT/'locks/heavy_job.lock', f'vi38 verify pid={os.getpid()}'):
        main()
