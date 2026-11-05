"""Independent recomputation of every rule-b exchange; read-only sources."""
from collections import Counter
import json
import math
import os
from pathlib import Path

import vi38 as v

def main():
    with v.dw.owned_lock(v.ROOT/'locks/heavy_job.lock', f'vi38b independent audit pid={os.getpid()}'):
        b = v.OUT
        old = v.dw.read(v.OLD/'submission_vi37.json')
        new = v.dw.read(b/'submission_vi38b.json')
        pairs = v.dw.read(b/'export/pairs.json')
        js = v.dw.read(b/'merged.json')
        actual = v.dw.read(b/'exchanges_vi38b.json')
        byq = {}
        for r in pairs:
            byq.setdefault(r['qid'], []).append(r)
        total = eligible = 0
        for before, after in zip(old, new):
            q = str(before['id'])
            protected = {int(c['doc_id']) for c in before['relevant_chunks']}
            aa, bb = [], []
            for r in byq.get(q, []):
                j = js.get(r['pair_id'], {})
                if r['status'] != 'ready' or j.get('status') != 'ok':
                    continue
                assert all(math.isfinite(j[f'logp{i}']) for i in range(3))
                if r['role'] == 'A' and j['label'] in (0, 1):
                    c = j.get('confirmation', {})
                    assert c and j['confirm_text_sha256'] == r['text_sha256']
                    if c['label'] in (0, 1):
                        aa.append((c['score'], r['rank'], r['doc_id']))
                if r['role'] == 'B' and j['label'] == 2:
                    bb.append((-j['score'], r['rank'], r['doc_id']))
            aa.sort()
            bb.sort()
            eligible += len(aa)
            expected = list(before['relevant_docs'])
            changes = []
            for a, c in list(zip(aa, bb))[:5]:
                assert a[2] not in protected and c[2] not in before['relevant_docs']
                pos = expected.index(a[2])
                expected[pos] = c[2]
                changes.append(dict(position=pos, out=a[2], incoming=c[2]))
            assert expected == after['relevant_docs'] and actual[q] == changes
            total += len(changes)
        report = dict(queries=1200, eligible_after=eligible, swaps=total,
                      exact_exchange_recompute=True, fulltext_score_sort=True,
                      source=v.fingerprint(v.OLD/'submission_vi37.json'))
        v.dw.write(b/'independent_audit_vi38b.json', report)
        print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
