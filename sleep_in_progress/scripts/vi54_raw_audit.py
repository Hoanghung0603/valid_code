"""Independent local audit of all raw input bytes, IDs/order and immutable vi51."""
import hashlib
import json
from pathlib import Path
import sys

from vi48 import fingerprint,read,write
from vi51 import iterz

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
OUT=TEMP/'vi54'


def main():
    prep=read(OUT/'prepare_report.json');old=read(TEMP/'vi51/prepare_report.json')
    seen=set();parts={}
    for n,meta in prep['files'].items():assert fingerprint(OUT/'modal_input'/n)==meta
    for n,meta in prep['parts'].items():
        p=OUT/'modal_input'/n;rows=list(iterz(p));ids=[r['id'] for r in rows]
        reference=TEMP/'vi51/modal_input'/n.replace('docs54.','docs51.')
        assert fingerprint(reference)==old['files'][reference.name]
        original=[r['id'] for r in iterz(reference)]
        assert ids==original and len(ids)==meta['docs'] and len(ids)==len(set(ids))
        assert not seen.intersection(ids);seen.update(ids)
        assert all(set(r)=={'id','text'} and r['text'].strip() for r in rows)
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==meta['id_order_sha256']
        parts[n]=dict(**fingerprint(p),documents=len(rows),exact_id_order=True,nonempty=True)
    pool=read(OUT/'modal_input/pool.json');qs=read(OUT/'modal_input/queries54.json')
    assert seen=={d for ids in pool.values() for d in ids} and len(seen)==223065
    assert len(qs)==1200 and {r['id'] for r in qs}==set(range(1,1201)) and all(r['text'].strip() for r in qs)
    report=dict(documents=len(seen),pairs=sum(map(len,pool.values())),queries=len(qs),parts=parts,
                full_sha=True,exact_id_order=True,no_empty=True,vi51_unchanged=True)
    assert report['pairs']==609662
    write(OUT/'raw_audit_local.json',report);print('RAW_AUDIT_LOCAL_PASS',flush=True)


if __name__=='__main__':main()
