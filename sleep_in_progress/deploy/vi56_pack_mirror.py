"""Prepare vi56-only transport and exact-byte manifest; not a submission build."""
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi56 import TEMP, read, write, fingerprint


def main():
    out=TEMP/'vi56'
    assert read(out/'final_report.json')['status']=='LOCAL_COMPLETE_MIRROR_PENDING'
    names=['submission_vi56.zip','build_report.json','independent_audit.json','offline.json','cost_report.json',
           'state.json','final_report.json','input_audit.json','selection_report.json','prepare_report.json',
           'new_ids.json','new_scores.json','q3e_new.json','sanity.json','bulk_forecast.json','bulk_plan.json',
           'original_app_logs.txt','original_qwen_wall.json','original_bge_wall.json',
           'tests_final.xml','tests_chain_final.xml']
    manifest={name:fingerprint(out/name) for name in names}
    write(out/'mirror_manifest.json',manifest)
    # Keep each transport <98MB, ZIP_STORED; immutable submission ZIP stays exactly as built.
    batches=[];batch=[];size=0
    for name in names+['mirror_manifest.json']:
        n=(out/name).stat().st_size
        if batch and size+n>97_000_000:
            batches.append(batch);batch=[];size=0
        batch.append(name);size+=n
    if batch:batches.append(batch)
    packs=[]
    for i,batch in enumerate(batches):
        target=out/f'mirror_pack{i:02}.zip'
        assert not target.exists(), 'Do not overwrite transport'
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as z:
            for name in batch:z.write(out/name,name)
        assert target.stat().st_size<98_000_000
        packs.append(dict(name=target.name,**fingerprint(target),files=batch))
    write(out/'mirror_transport.json',packs)
    print('MIRROR_PACK_READY',packs,flush=True)


if __name__=='__main__':main()
