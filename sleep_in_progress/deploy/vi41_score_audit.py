"""Local audit of exact candidate IDs and finite scores; no GPU or launch."""
import hashlib
import json
import math
import sys
from vi41_local import TEMP

stage=sys.argv[1]
if stage=='rr':
    expected=json.loads((TEMP/'modal_input/new_cands.json').read_text())
    got=json.loads((TEMP/'rr_new.json').read_text())
elif stage=='q8b':
    expected=json.loads((TEMP/'modal_input/q8b_cands.json').read_text())
    got=json.loads((TEMP/'q8b_new.json').read_text())
    qreport=json.loads((TEMP/'qselect_report.json').read_text())
    assert sum(map(len,expected.values()))==qreport['pairs']
else:
    if stage!='docwin':
        raise SystemExit('rr|q8b|docwin only')
    import zstandard
    expected=json.loads((TEMP/'modal_input/new_cands.json').read_text())
    with (TEMP/'docwin_new.json.zst').open('rb') as f:
        with zstandard.ZstdDecompressor().stream_reader(f) as stream:
            got=json.loads(stream.read())
    lengths=json.loads((TEMP/'export_lengths.json').read_text())
    assert set(expected)==set(got)
    windows=0
    text_docs={}
    for p in sorted((TEMP/'modal_input').glob('docs_new.part*.jsonl.zst')):
        with p.open('rb') as f:
            with zstandard.ZstdDecompressor().stream_reader(f) as stream:
                for line in stream.read().splitlines():
                    row=json.loads(line);text_docs[str(row['id'])]=row['text']
    for q,rows in expected.items():
        nonempty={str(d) for d,i in rows if lengths[str(d)]['windows']}
        assert nonempty<=set(got[q])
        assert set(got[q])<={str(d) for d,i in rows}
        for d,wins in got[q].items():
            for s,e,score in wins:
                assert 0<=s<e<=lengths[d]['chars'] and math.isfinite(score)
                assert text_docs[d][s:e].strip(),(q,d,s,e)
            assert len(wins)==lengths[d]['windows'],(q,d,len(wins),lengths[d]['windows'])
            windows+=len(wins)
    result=dict(stage=stage,queries=len(got),windows=windows,finite=True,exact_nonempty_ids=True,span_errors=0)
    assert windows==json.loads((TEMP/'export_report.json').read_text())['windows']
    (TEMP/'docwin_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    raise SystemExit(0)
assert set(expected)==set(got)
values=[]
for q,rows in expected.items():
    assert len(rows)==len(got[q]) and {r[0] for r in rows}=={r[0] for r in got[q]}
    for d,i,s in got[q]:
        assert s is not None and math.isfinite(s),(q,d,s)
        values.append(s)
values.sort()
result=dict(stage=stage,queries=len(got),pairs=len(values),finite=True,exact_ids=True,
    mean=sum(values)/len(values),min=values[0],p10=values[int(.1*(len(values)-1))],
    median=values[len(values)//2],p90=values[int(.9*(len(values)-1))],max=values[-1],
    above_minus2=sum(s>-2 for s in values),above_zero=sum(s>0 for s in values))
(TEMP/(stage+'_audit.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result))
