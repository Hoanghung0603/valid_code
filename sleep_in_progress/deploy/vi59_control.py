"""Durable vi59 Modal controller. Each spawn ID is saved before any get."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi56 import read,write,fingerprint,RATE,completed_part
OUT = ROOT.parent/'r2ai_deliver/cand/vi59d'
MODAL = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe')
os.environ.update(PYTHONIOENCODING='utf-8',MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"))


def cli(*args):
    p = subprocess.run([str(MODAL),*map(str,args)],cwd=ROOT,env=os.environ,capture_output=True,
                       text=True,encoding='utf-8',timeout=3600)
    if p.returncode: raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def handoff(s):
    lines = ['# vi59d — KHÔNG nộp/browser/commit/crawl',
             '- Fixed top200 q3e ngoài pool vi41c; direct vi56.extend/vi52.fused/protect; chunks/count vi48.',
             '- Chỉ chấm101–200; vi56 reuse exactID/SHA; GPU trực tiếp vi56.gpu, bench vi56 fix/SHA gpu_code.json.',
             '- App r2ai-vi59/data r2ai-vi59-data; vi56 weights read_only; max4H100/retries0.',
             f"- App={s.get('app_id')}; stopped={s.get('stopped',False)}; upper={s.get('upper',0):.6f}USD; baseline={s.get('baseline')}",
             '- Local <WORKDIR>/r2ai_deliver/cand/vi59d; mirror D:/r2ai/data/sub/vi59; hardstop20:30VN.']
    lines += ['- Recovery approved DECISIONS15:05 cap12USD; basevi48trueSHA0bfc09df…8281e.',
              '- Former658order mismatches caused by vi53fallback; truevi48 rerun BOTH sanity exactorder1200/1200.',
              '- Input28partSHA PASS; eligible92355pairs/reuse32088/extra60267; bgedoc60267/qdoc60267/bgewin292401/d8w106030.']
    lines.append('- Recovery15:05/15:06: vi48 thật SHA0bfc09df…8281e; trước fallbackvi53 lệch658thứtự nhưng tậpđúng; với vi48 thật2sanityđúngthứtự1200/1200, không sửa merge_row.')
    for n,r in s.get('calls',{}).items(): lines.append(f"- {n}: {r.get('call_id','INTENT')} {r['status']} reserve={r.get('reserve',0):.6f}")
    for n in ('selection_report','sanity','forecast','build_report','cx_extra_report','cost_report','extra_input_audit'):
        if (OUT/(n+'.json')).exists(): lines.append('- '+n+': '+json.dumps(read(OUT/(n+'.json')),ensure_ascii=False))
    lines.append('- Chưa kiểm: gold/F2, billing settled; không đụng historical/replay/crawl.')
    assert len(lines)<=50
    (ROOT/'docs/VI59_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','upload','forecast','spawn','poll','collect','stop'])
    ap.add_argument('--job',default='ping');ap.add_argument('--function',default='ping')
    ap.add_argument('--parts',nargs='*',default=[]);ap.add_argument('--watchdog',type=int,default=240)
    ap.add_argument('--wait',type=int,default=0)
    a=ap.parse_args();import modal
    sp=OUT/'state.json';s=read(sp) if sp.exists() else dict(calls={},upper=.1,profile=__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"))
    def save():write(sp,s);handoff(s)
    if a.action=='init':
        assert not sp.exists(),'already initialized'
        apps=json.loads(cli('app','list','--json'));write(OUT/'apps_init.json',apps)
        own=[x for x in apps if x['description']=='r2ai-vi59' and x['state']!='stopped'];assert len(own)==1
        s.update(app_id=own[0]['app_id'],baseline=json.loads(cli('billing','summary','--json')),stopped=False);save()
    elif a.action=='upload':
        uploaded=read(OUT/'uploaded.json') if (OUT/'uploaded.json').exists() else {}
        prep=read(OUT/'extra/prepare_report.json')
        for p,dst in [(OUT/'extra/prepare_report.json','/prepare_report.json')]+[(OUT/'extra/modal_input'/n,'/in/'+n) for n in prep['parts']]:
            fp=fingerprint(p)
            if uploaded.get(dst)==fp:continue
            cli('volume','put','--force','r2ai-vi59-data',p,dst)
            uploaded[dst]=fp;write(OUT/'uploaded.json',uploaded);print('UPLOADED '+p.name,flush=True)
        write(OUT/'upload_complete.json',dict(parts=len(prep['parts']),complete=True))
    elif a.action=='forecast':
        prep=read(OUT/'extra/prepare_report.json');c=prep['counts']
        q=read(OUT/'inputs/vi56/outputs/bench_qwen_fix/report.json')
        b=read(OUT/'inputs/vi56/outputs/bench_bge_fix/report.json')
        qs=(c['q8b_doc']+c['d8w'])/q['pairs_per_second']*1.3+4*90
        bs=(c['bge_doc']+c['bge_win'])/b['pairs_per_second']*1.3+2*45
        report=dict(qwen_seconds=qs,bge_seconds=bs,usd_upper_total=(qs+bs)*RATE+s['upper']+.25,cap=12,
                    bench_reused=True,code=read(OUT/'gpu_code.json'))
        write(OUT/'forecast.json',report);save()
        assert report['usd_upper_total']<=12,('STOP forecast over12USD',report)
    elif a.action=='spawn':
        from datetime import datetime,timezone,timedelta
        now=datetime.now(timezone(timedelta(hours=7)))
        assert now.hour*60+now.minute < 20*60+30,'HARD STOP20:30VN'
        assert a.job not in s['calls'],'Never respawn ambiguous call'
        apps=json.loads(cli('app','list','--json'));write(OUT/f'apps_{a.job}.json',apps)
        assert any(x['app_id']==s['app_id'] and x['state']!='stopped' for x in apps)
        gpu=a.function in ('qwen','bge')
        if not gpu:assert read(OUT/'upload_complete.json')['complete']
        active=[r for r in s['calls'].values() if r['function'] in ('qwen','bge') and r['status']=='running']
        if gpu:
            assert len(active)<4 and s['calls']['ping']['status']=='done'
            assert read(OUT/'forecast.json')['usd_upper_total']<=12
            assert read(OUT/'sanity.json')['top100_vs_vi56']['set_and_count_exact']==1200
        reserve=(a.watchdog+10)*RATE if gpu else .025
        assert s['upper']+reserve<=12,('STOP reservation over cap',s['upper'],reserve)
        r=dict(status='spawn_intent',function=a.function,parts=a.parts,watchdog=a.watchdog,reserve=reserve,start=time.time())
        s['calls'][a.job]=r;s['upper']+=reserve;save()
        args=(a.job,a.parts,a.watchdog) if gpu else ()
        call=modal.Function.from_name('r2ai-vi59',a.function).spawn(*args)
        r.update(call_id=call.object_id,status='running');save();print(json.dumps(r),flush=True)
    elif a.action=='poll':
        r=s['calls'][a.job];assert r['status']=='running'
        try:result=modal.FunctionCall.from_id(r['call_id']).get(timeout=a.wait)
        except (TimeoutError,modal.exception.TimeoutError):print('RUNNING '+r['call_id']);return
        except Exception as exc:
            r['error']=repr(exc)
            if isinstance(exc,(modal.exception.RemoteError,modal.exception.FunctionTimeoutError)):r['status']='failed_terminal'
            save();raise
        r.update(status='done',finish=time.time(),result=result)
        if r['function'] in ('qwen','bge'):
            actual=(result['function_seconds']+10)*RATE;s['upper']+=actual-r['reserve'];r['reserve']=actual
        write(OUT/(a.job+'.json'),result);save();print(json.dumps(result),flush=True)
    elif a.action=='collect':
        r=s['calls'][a.job];assert r['status']=='done'
        if r['function']=='ping':return
        dest=OUT/'extra/outputs'/a.job;dest.mkdir(parents=True,exist_ok=True)
        for n in r['parts']:
            for suffix in ('.scores.json','.scores.complete.json','.scores.attempts.json'):
                p=dest/(n+suffix)
                if not p.exists():cli('volume','get','--force','r2ai-vi59-data','/out/'+a.job+'/'+n+suffix,p)
            assert completed_part(OUT/'extra/modal_input'/n,dest/(n+'.scores.json'))
        cli('volume','get','--force','r2ai-vi59-data','/out/'+a.job+'/report.json',dest/'report.json')
    else:
        assert all(r['status'] in ('done','failed_terminal') for r in s['calls'].values())
        apps=json.loads(cli('app','list','--json'));own=next(x for x in apps if x['app_id']==s['app_id'])
        if own['state']!='stopped':cli('app','stop','-y',s['app_id'])
        s.update(stopped=True,after=json.loads(cli('billing','summary','--json')));save()
        write(OUT/'apps_final.json',json.loads(cli('app','list','--json')))


if __name__=='__main__':main()
