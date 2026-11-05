"""Build from SHA-checked complete committed cosines while embeddings download."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi48 import fingerprint

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')
tier=sys.argv[1];out=TEMP.parent/('vi51'+tier);seen=set()
for p in TEMP.glob('state_sdpa_*.json'):
    s=json.loads(p.read_text())
    for job,r in s['calls'].items():
        if r.get('tier')!=tier or r.get('mode')!='bulk':continue
        assert r['status']=='done'
        assert not seen.intersection(r['parts']);seen.update(r['parts'])
        assert fingerprint(out/'outputs'/s['profile']/job/'cosines.json.zst')==r['result']['files']['cosines.json.zst']
assert seen==set(json.loads((TEMP/'prepare_report.json').read_text())['parts'])
assert not (out/'build_report.json').exists()
with (out/'build_intent.json').open('x') as f:json.dump(dict(time=time.time(),pid=os.getpid(),cosines_sha_pass=True),f)
for script,args in [('vi51.py',['build','--tier',tier,'--out',str(out)]),('vi51_audit.py',['--out',str(out)])]:
    subprocess.run(['python',str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
print('BUILD_AUDIT_READY '+tier,flush=True)
