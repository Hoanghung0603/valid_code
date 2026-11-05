"""Fetch small committed cosine files first; leaves embeddings collector independent."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vi51_sdpa_local import cli,TEMP
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vi48 import fingerprint


def main(tier):
    for sp in TEMP.glob('state_sdpa_*.json'):
        s=json.loads(sp.read_text())
        for job,r in s['calls'].items():
            if r.get('tier')!=tier or r.get('mode')!='bulk':continue
            assert r['status']=='done'
            dest=TEMP.parent/('vi51'+tier)/'outputs'/s['profile']/job;dest.mkdir(parents=True,exist_ok=True)
            for n in ('cosines.json.zst','bulk_report.json','function_wall.json'):
                p=dest/n;expected=r['result']['files'].get(n)
                if p.exists() and (not expected or fingerprint(p)==expected):continue
                pending=dest/(n+'.fastdownload')
                cli(s['profile'],'volume','get','--force','r2ai-vi51-data',f'/out/{tier}/{job}/{n}',str(pending))
                pending.replace(p)
                if expected:assert fingerprint(p)==expected
            print('COSINES_READY '+job,flush=True)


if __name__=='__main__':main(sys.argv[1])
