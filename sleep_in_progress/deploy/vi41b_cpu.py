"""One fixed-count CPU variant, isolated output; no model/network calls."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(os.environ.get('R2AI_ROOT','D:/r2ai'))
OUT=ROOT/'data/sub/vi41b'
SOURCE=ROOT/'data/sub/vi41'
TASK='r2ai_vi41b_cpu'
sys.path.insert(0,str(ROOT/'scripts'))
import dwmax_diag as dw
import vi41


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def status():
    result={}
    lock=ROOT/'locks/heavy_job.lock'
    result['lock']=lock.read_text() if lock.exists() else None
    result['task']=subprocess.run(['schtasks','/Query','/TN',TASK,'/FO','LIST','/V'],capture_output=True,text=True).stdout
    log=OUT/'cpu.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-4000))
            result['log_tail']=f.read().decode('utf-8',errors='replace')
        result['log_mtime']=log.stat().st_mtime
    for name in ('job_report.json','build_report.json','fixed_nq_report.json','stats_report.json','offline_report.json'):
        if (OUT/name).exists():
            result[name]=vi41.read(OUT/name)
    print(json.dumps(result,ensure_ascii=False))


def build():
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'submission_vi41b.zip').exists(), 'No automatic rebuild'
    with dw.owned_lock(ROOT/'locks/heavy_job.lock',f'vi41b cpu pid={os.getpid()}'):
        paths=['new_ids.json','new_provenance.json','export_lengths.json','export_report.json',
            'modal_input/new_cands.json','modal_input/q8b_cands.json',
            'modal_out/rr_new.json','modal_out/q8b_new.json','modal_out/docwin_new.json.zst',
            'qselect_report.json','new_cands_u900.json.zst','pre_chunks_report.json',
            'sanity_report.json','sanity_chain_report.json','input_audit_report.json',
            'preverify_report.json','verify_lengths.json']
        snapshots={str(SOURCE/p):digest(SOURCE/p) for p in paths}
        reference=ROOT/'data/sub/vi37/submission_vi37.json'
        snapshots[str(reference)]=digest(reference)
        for p in paths:
            dst=OUT/p
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/p,dst)
        vi41.OUT=OUT
        vi41.NQ_FROM=reference
        for label,fn in [('merge',vi41.merge),('chunk',vi41.chunk),('build',vi41.build),('stats',vi41.stats)]:
            print('START',label,flush=True)
            fn()
            print('DONE',label,'elapsed',time.monotonic()-start,flush=True)
        # Exact expanded-pool scores/order, no reranking difference permitted.
        assert vi41.read(OUT/'cache_fhyde_v41.json')==vi41.read(SOURCE/'cache_fhyde_v41.json')
        for source in ('pool','q8b','dense','hyde'):
            assert vi41.read(OUT/f'in1/{source}_v41.json')==vi41.read(SOURCE/f'in1/{source}_v41.json')
        assert vi41.read(OUT/'docwin_max_v41.json')==vi41.read(SOURCE/'docwin_max_v41.json')
        from vi41_offline_audit import summarize
        old=vi41.read(reference)
        new=vi41.read(OUT/'submission_vi41b.json')
        def scores(path):
            return {q:{int(h[0]):float(h[2]) for h in hs if h[2] is not None and math.isfinite(float(h[2]))}
                    for q,hs in vi41.read(path).items()}
        report=summarize(old,new,vi41.read(OUT/'new_ids.json'),vi41.read(OUT/'cache_fhyde_v41.json'),
            scores(ROOT/'data/sub/vi37/q8b_v37.json'),scores(OUT/'in1/q8b_v41.json'))
        counts={str(r['id']):len(r['relevant_docs']) for r in old}
        report['counts_equal']=sum(len(r['relevant_docs'])==counts[str(r['id'])] for r in new)
        assert report['counts_equal']==1200
        lengths=vi41.read(OUT/'export_lengths.json')
        q8=scores(OUT/'in1/q8b_v41.json')
        missing=[(str(r['id']),d) for r in new for d in r['relevant_docs'] if d not in q8.get(str(r['id']),{})]
        report['missing_8b_final']={'pairs':len(missing),'nonempty_or_old':sum(str(d) not in lengths or bool(lengths[str(d)]['windows']) for q,d in missing)}
        assert report['missing_8b_final']['nonempty_or_old']==0
        vi41.write(OUT/'offline_report.json',report)
        assert all(digest(Path(p))==h for p,h in snapshots.items())
        vi41.write(OUT/'source_sha_audit.json',{'unchanged':True,'files':snapshots})
        vi41.write(OUT/'job_report.json',dict(done=True,seconds=time.monotonic()-start,
            fixed_counts_equal=1200,rrf_and_all_scores_equal_vi41=True,source_sha_unchanged=True,
            cpu_only=True,usd=0))
        print('VI41B DONE',flush=True)


def cleanup():
    assert vi41.read(OUT/'job_report.json')['done']
    task=subprocess.run(['schtasks','/Query','/TN',TASK],capture_output=True,text=True)
    assert task.returncode!=0, 'Delete only our completed task before cleanup audit'
    assert not (ROOT/'locks/heavy_job.lock').exists()
    baseline=vi41.read(SOURCE/'original_inputs_sha.json')
    for rel,expected in baseline.items():
        p=ROOT/'data/sub'/rel
        assert p.stat().st_size==expected['bytes'] and digest(p)==expected['sha256'], rel
    report=dict(task_absent=True,lock_absent=True,zip_bytes=(OUT/'submission_vi41b.zip').stat().st_size,
                sha256=digest(OUT/'submission_vi41b.zip'),original_input_sha_unchanged=len(baseline))
    vi41.write(OUT/'cleanup_report.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    {'build':build,'status':status,'cleanup':cleanup}[sys.argv[1]]()
