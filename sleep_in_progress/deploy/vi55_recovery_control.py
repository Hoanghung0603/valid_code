"""Recovery control: one persistent call record, no automatic respawn."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,fingerprint,RATE
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55')
M=T.parent/'modalenv/Scripts/modal.exe'
MISSING=[f'docs55.part{i:03d}.jsonl.zst' for i in (16,13,21,26,32,12,23,10)]


def cli(profile,*args):
    p=subprocess.run([str(M),*args],cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'),
                     capture_output=True,text=True,encoding='utf-8',timeout=7200)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def handoff():
    lines=['# vi55 recovery pha 1 — duyệt 31/10 22:50 VN',
      '- Không pha2/nộp/browser/commit/crawl; hard stop04:00VN, mục tiêu01:30VN01/11.',
      '- Fix resume CPU: exact IDs/order + fp16/shape4096/finite/L2/SHA/input SHA; complete metadata cuối, commit từng part.',
      '- Partial không gộp; tối đa2attempt/part; test preemption dở→rerun=clean,13testPASS.',
      '- Giữ43part cũ426707doc +222997VI2048; chỉ8part thiếu80000doc mới trên30d/<MODAL_PROFILE>.',
      '- Missing: 016,013,021,026,032,012,023,010; fp16SDPA/left/LAST/L2/doc1024/query256/pin1d8ad4ca.',
      '- Query một call1200, audit full rồi dùng chung3workspace; top1000 merge disjoint/đúng cosine.',
      '- Old upper18.938931953USD; trần26USD; search/billing chưa settled; giữVolume, stop app cuối.']
    for p in sorted(T.glob('recovery_state_*.json')):
        s=read(p);lines.append(f"- {s['profile']} app={s.get('app_id')} stopped={s.get('stopped')} reserve={s.get('reserved',0):.6f}USD baseline={s.get('baseline',{}).get('metered_cost')}")
        for job,r in s['calls'].items():lines.append(f"- {job}: {r.get('call_id','INTENT')} {r['status']} {r.get('parts',[])}")
    lines.append('- State <HIST_TMP>/vi55/recovery_state_PROFILE.json; poll known ID bằng vi55_recovery_control.py wait; KHÔNG spawn lại.')
    if (T/'shared_query_audit.json').exists():
        q=read(T/'shared_query_audit.json');lines.append(f"- Queryfull {q['documents']}/shape={q['shape']}/finiteL2PASS/SHA={q['sha256']}; dùngchung mọi search.")
    if (T/'recovery_<MODAL_PROFILE_D>_gate.json').exists():
        g=read(T/'recovery_<MODAL_PROFILE_D>_gate.json');lines.append(f"- Gate50min={g['repeat_cosine_min']:.9f}/mean={g['repeat_cosine_mean']:.9f}/GPU-s={g['function_seconds']:.6f}PASS.")
    lines.append('- LỗiCPUstartup30d conditional env đãcancel terminal+fix image.env; ping_fixedPASS;0GPU wasted ởlỗiCPU.')
    lines.append('- Testfull1197PASS2skip;13vi55PASS/27chuỗiPASS; chưaF2/billingsettled.')
    assert len(lines)<=60
    (ROOT/'docs/VI55_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','spawn','wait','upload','stop','download'])
    ap.add_argument('--profile',default=__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"));ap.add_argument('--job',default='weights')
    ap.add_argument('--function',default='download_weights');ap.add_argument('--mode',default='bulk')
    ap.add_argument('--parts',nargs='*',default=[]);ap.add_argument('--watchdog',type=int,default=800)
    ap.add_argument('--remote');ap.add_argument('--local',type=Path)
    a=ap.parse_args();os.environ.update(MODAL_PROFILE=a.profile,PYTHONIOENCODING='utf-8')
    sp=T/f'recovery_state_{a.profile}.json'
    s=read(sp) if sp.exists() else dict(profile=a.profile,calls={},reserved=0,stopped=False)
    def save():write(sp,s);handoff()
    if a.action=='init':
        assert not sp.exists(),'Do not reset state'
        apps=json.loads(cli(a.profile,'app','list','--json'));write(T/f'recovery_apps_{a.profile}.json',apps)
        own=[x for x in apps if x['description']=='r2ai-vi55' and x['state']!='stopped'];assert len(own)==1
        s.update(app_id=own[0]['app_id'],baseline=json.loads(cli(a.profile,'billing','summary','--json')));save()
    elif a.action=='spawn':
        import modal
        assert a.job not in s['calls'],'Never respawn ambiguous or existing call'
        apps=json.loads(cli(a.profile,'app','list','--json'));write(T/f'recovery_apps_{a.profile}_{a.job}.json',apps)
        assert any(x['app_id']==s['app_id'] and x['state']!='stopped' for x in apps)
        reserve=(a.watchdog+12)*RATE if a.function in ('work','retrieve') else .03
        total=18.938931953320207+sum(read(p)['reserved'] for p in T.glob('recovery_state_*.json'))+reserve
        assert total<=26,total
        if a.function=='work':
            assert sum(r['status']=='running' and r['function']=='work' for r in s['calls'].values())<4
            if a.mode=='bulk':assert set(a.parts)<=set(MISSING) and a.profile==__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")
        r=dict(status='spawn_intent',function=a.function,mode=a.mode,parts=a.parts,watchdog=a.watchdog,reserve=reserve,start=time.time())
        s['calls'][a.job]=r;s['reserved']+=reserve;save()
        args=(a.mode,a.job,a.parts,a.watchdog,1024) if a.function=='work' else ((a.job,) if a.function=='retrieve' else ())
        call=modal.Function.from_name('r2ai-vi55',a.function).spawn(*args)
        r.update(call_id=call.object_id,status='running');save();print(json.dumps(r),flush=True)
    elif a.action=='wait':
        import modal
        r=s['calls'][a.job]
        if r['status']=='done':return
        assert r['status']=='running'
        try:result=modal.FunctionCall.from_id(r['call_id']).get(timeout=7200)
        except (TimeoutError,modal.exception.TimeoutError):
            raise
        except Exception as exc:
            r.update(status='failed_terminal',error=repr(exc));save();raise
        r.update(status='done',result=result,finish=time.time())
        if r['function'] in ('work','retrieve'):
            actual=(result['function_seconds']+12)*RATE
            s['reserved']+=actual-r['reserve'];r['reserve']=actual
        write(T/f'recovery_{a.profile}_{a.job}.json',result);save()
        print(json.dumps(dict(job=a.job,status='done',function_seconds=result.get('function_seconds'),documents=result.get('documents'))),flush=True)
    elif a.action=='upload':
        prep=read(T/'prepare_report.json')
        names=a.parts or (MISSING+['queries.json'])
        for name in names:
            p=T/'modal_input'/name
            if name in prep['parts']:assert fingerprint(p)=={k:prep['parts'][name][k] for k in ('bytes','sha256')}
            cli(a.profile,'volume','put','--force','r2ai-vi55-data',str(p),'/in/'+name)
        cli(a.profile,'volume','put','--force','r2ai-vi55-data',str(T/'prepare_report.json'),'/prepare_report.json')
    elif a.action=='download':
        cli(a.profile,'volume','get','--force','r2ai-vi55-data',a.remote,str(a.local))
    else:
        assert all(r['status'] in ('done','failed_terminal') for r in s['calls'].values())
        cli(a.profile,'app','stop','-y',s['app_id']);s.update(stopped=True,after=json.loads(cli(a.profile,'billing','summary','--json')));save()


if __name__=='__main__':main()
