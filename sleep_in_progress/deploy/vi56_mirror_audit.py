"""Small b660 CPU audit; scoped to vi56, no crawl access, no heavy-job lock mutation."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

OUT=Path('D:/r2ai/data/sub/vi56')


def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())


def main():
    dest=OUT/'final';dest.mkdir(exist_ok=True)
    # Explicit transport inventory; validate flat names and exact existing bytes before extracting.
    tp=OUT/'mirror_transport.json'
    if tp.exists():
        packs=json.loads(tp.read_text(encoding='utf-8'))
        for pack in packs:
            assert Path(pack['name']).name==pack['name']
            p=OUT/pack['name']
            assert fingerprint(p)=={k:pack[k] for k in ('bytes','sha256')}
            with zipfile.ZipFile(p) as z:
                assert z.namelist()==pack['files'] and all(Path(n).name==n for n in z.namelist())
                for name in z.namelist():
                    blob=z.read(name);target=dest/name
                    if target.exists():
                        assert target.read_bytes()==blob, ('Refuse overwrite different file',name)
                    else:target.write_bytes(blob)
    expected=json.loads((dest/'mirror_manifest.json').read_text(encoding='utf-8'))
    assert all(Path(name).name==name for name in expected)
    got={name:fingerprint(dest/name) for name in expected}
    assert got==expected
    zp=OUT/'submission_vi56.zip'
    if zp.exists():assert fingerprint(zp)==expected[zp.name]
    else:zp.write_bytes((dest/zp.name).read_bytes())
    task=subprocess.run(['schtasks','/Query','/TN','r2ai_vi56_prepare','/FO','LIST'],capture_output=True,text=True)
    lock=Path('D:/r2ai/locks/heavy_job.lock')
    result=dict(status='MIRROR_VERIFIED',files=got,prepare_task_absent=task.returncode!=0,
                heavy_lock_absent=not lock.exists(),heavy_lock_owner=lock.read_text(encoding='utf-8') if lock.exists() else None)
    (OUT/'mirror_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
