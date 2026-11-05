"""Repair only new vi38 exports after detecting Unicode boundary re-encoding overshoot.

No old dataset edits; manifest commits last. Deterministic shortened evidence, not labels.
"""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vi38 as v


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(v.OUT/'tokenizer.json'))
    dest = v.OUT/'export'
    manifest = v.dw.read(dest/'manifest.json')
    rows = v.dw.read(dest/'pairs.json')
    import json
    query = {str(r['id']): r['query'] for r in map(json.loads,
             (v.OLD/'modal_input/queries.jsonl').read_text(encoding='utf-8').splitlines())}
    fixed = {}
    for r in rows:
        prompt = r['prompt']
        head = prompt.split('Đầu bài: ', 1)[1].split('\nĐoạn bằng chứng (có thể nằm xa đầu bài): ', 1)[0]
        win = prompt.split('\nĐoạn bằng chứng (có thể nằm xa đầu bài): ', 1)[1].rsplit('\nChỉ in một chữ số', 1)[0]
        nh, nw = head, win
        while len(tok.encode(nh, add_special_tokens=False).ids) > 256:
            nh = nh[:-1]
        while len(tok.encode(nw, add_special_tokens=False).ids) > 512:
            nw = nw[:-1]
        if nh == head and nw == win:
            continue
        assert prompt == v.chat(query[r['qid']], head, win), 'Ambiguous prompt delimiter'
        r['excerpt_offsets'][0][1] -= len(head)-len(nh)
        r['excerpt_offsets'][1][1] -= len(win)-len(nw)
        r['prompt'] = v.chat(query[r['qid']], nh, nw)
        r['prompt_sha256'] = v.sha(r['prompt'])
        r['n_tokens'] = len(tok.encode(r['prompt'], add_special_tokens=False).ids)
        fixed[r['pair_id']] = r
    assert 0 < len(fixed) <= 20, len(fixed)
    # Bench checked separately: none of its prompts exceeded caps; preserve downloaded files.
    assert not set(fixed) & {r['pair_id'] for r in v.dw.read(dest/'bench.json')}
    for name in manifest['modes']['bulk']:
        part = v.dw.read(dest/name)
        if any(r['pair_id'] in fixed for r in part):
            part = [fixed.get(r['pair_id'], r) for r in part]
            v.dw.write(dest/name, part)
            manifest['files'][name] = v.fingerprint(dest/name)
    v.dw.write(dest/'pairs.json', rows)
    manifest['files']['pairs.json'] = v.fingerprint(dest/'pairs.json')
    manifest['bulk_prompt_tokens'] = sum(r['n_tokens'] for r in rows if r['status'] == 'ready')
    v.dw.write(dest/'manifest.json', manifest)
    v.dw.write(v.OUT/'export_report.json', manifest)
    v.dw.write(v.OUT/'token_cap_fix_report.json', dict(fixed=len(fixed),
                reason='Unicode shared offsets can add one token after substring re-encoding',
                bulk_tokens=manifest['bulk_prompt_tokens']))
    print('FIXED', len(fixed), 'tokens', manifest['bulk_prompt_tokens'], flush=True)


if __name__ == '__main__':
    with v.dw.owned_lock(v.ROOT/'locks/heavy_job.lock', f'vi38 capfix pid={os.getpid()}'):
        main()
