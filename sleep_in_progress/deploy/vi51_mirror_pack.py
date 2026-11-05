"""Package collected embeddings/cosines into bounded exact-byte mirror parts."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vi48 import fingerprint,write

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))


def main(tier):
    out=TEMP/('vi51'+tier);dest=TEMP/'vi51'/('mirror_'+tier);dest.mkdir(exist_ok=True)
    assert not (dest/'manifest.json').exists(),'Reuse existing mirror manifest, never rewrite'
    paths=sorted(p for p in (out/'outputs').rglob('*') if p.is_file() and not p.name.endswith('.download'))
    paths += [out/n for n in ('prebuild_audit.json','offline.json') if (out/n).exists()]
    cosine=TEMP/'vi51'/f'cosines_vi51{tier}.json'
    if cosine.exists():paths.append(cosine)
    chunks=[];current=[];size=0
    for p in paths:
        assert p.stat().st_size<98_000_000
        if current and size+p.stat().st_size>98_000_000:chunks.append(current);current=[];size=0
        current.append(p);size+=p.stat().st_size
    if current:chunks.append(current)
    parts=[]
    for i,ps in enumerate(chunks):
        path=dest/f'vi51{tier}_mirror_{i:03}.zip'
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED) as z:
            for p in ps:z.write(p,str(p.relative_to(TEMP)).replace('\\','/'))
        parts.append(dict(name=path.name,**fingerprint(path),files={str(p.relative_to(TEMP)):fingerprint(p) for p in ps}))
    write(dest/'manifest.json',dict(tier=tier,parts=parts))
    print(json.dumps(dict(directory=str(dest),parts=[dict(name=r['name'],bytes=r['bytes']) for r in parts])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('tier',choices=['a','b']);main(ap.parse_args().tier)
