"""New blinded gate on actual vi38b swaps. No gate1 files or labels are read."""
from collections import Counter
import json
import os
from pathlib import Path
import random
import zipfile

import vi38 as v


def stratified_sample(rows, rng, n=150):
    groups = {}
    for row in rows:
        groups.setdefault(row['qid'], []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    # Round-robin query allocation; no query gets a third until every second exhausted.
    while len(selected) < n:
        queries = list(groups)
        rng.shuffle(queries)
        changed = False
        for q in queries:
            if groups[q] and len(selected) < n:
                selected.append(groups[q].pop())
                changed = True
        assert changed, 'Not enough actual swaps for new blind gate'
    return selected


def create_blind(exchanges, queries, texts, tok, excluded_pairs):
    rng = random.Random(20261030)
    selected = []
    for role, field in [('out', 'out'), ('in', 'incoming')]:
        rows = [dict(qid=q, doc_id=c[field], role=role, pair_id=f'{q}:{c[field]}')
                for q, cs in exchanges.items() for c in cs if f'{q}:{c[field]}' not in excluded_pairs]
        selected.extend(stratified_sample(rows, rng))
    rng.shuffle(selected)
    blind, key = [], {}
    for r in selected:
        code = f'G2{rng.getrandbits(128):032x}'
        assert code not in key
        text = texts[r['doc_id']]
        enc = tok.encode(text, add_special_tokens=False)
        end = enc.offsets[min(3000, len(enc.ids))-1][1] if enc.ids else 0
        while end and len(tok.encode(text[:end], add_special_tokens=False).ids) > 3000:
            end -= 1
        blind.append(dict(code=code, qid=r['qid'], query=queries[r['qid']], doc_id=r['doc_id'],
                          text=text[:end], truncated=end < len(text)))
        key[code] = r
    assert len(blind) == len({r['code'] for r in blind}) == 300
    assert len({(r['qid'], r['doc_id']) for r in blind}) == 300
    return blind, key


def main():
    from tokenizers import Tokenizer
    from rerank_run import load_docs
    base = v.OUT
    with v.dw.owned_lock(v.ROOT/'locks/heavy_job.lock', f'vi38b gate2 pid={os.getpid()}'):
        exchanges = v.dw.read(base/'exchanges_vi38b.json')
        # Exclude ALL original blind benchmark pairs (superset of gate1) without reading gate1.
        prior_blind = v.dw.read(base/'export/blind.json')
        excluded = {f"{r['qid']}:{r['doc_id']}" for r in prior_blind}
        allids = {c[k] for cs in exchanges.values() for c in cs for k in ('out', 'incoming')}
        texts = load_docs([str(p) for p in (base/'export').glob('docs.part*.jsonl.zst')], wanted=allids)
        queries = {str(r['id']): r['query'] for r in map(json.loads,
                   (v.OLD/'modal_input/queries.jsonl').read_text(encoding='utf-8').splitlines())}
        blind, key = create_blind(exchanges, queries, texts,
                                 Tokenizer.from_file(str(base/'tokenizer.json')), excluded)
        dest = base/'judge2'
        v.dw.write(dest/'gate2_blind.json', blind)
        v.dw.write(dest/'private/gate2_key.json', key)
        # Key is deliberately NOT included in public shard lists or report file map.
        for i in range(6):
            v.dw.write(dest/f'gate2_s{i+1}.json', blind[i*50:(i+1)*50])
        subset = random.Random(20261031).sample(blind, 150)
        for i in range(3):
            v.dw.write(dest/f'gate2_haiku_s{i+1}.json', subset[i*50:(i+1)*50])
        files = [dest/'gate2_blind.json', *[dest/f'gate2_s{i+1}.json' for i in range(6)],
                 *[dest/f'gate2_haiku_s{i+1}.json' for i in range(3)]]
        report = dict(n=300, haiku=150, excluded_prior_pairs=len(excluded),
                      files={p.name: v.fingerprint(p) for p in files},
                      query_counts={role: dict(Counter(r['qid'] for r in key.values() if r['role'] == role))
                                    for role in ('out', 'in')})
        v.dw.write(base/'gate2_report.json', report)
        with zipfile.ZipFile(base/'gate2_package.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in [*files, dest/'private/gate2_key.json']:
                z.write(p, str(p.relative_to(dest)))
        print('GATE2 DONE', len(blind), flush=True)


if __name__ == '__main__':
    main()
