"""Persist launch intent/call IDs before get; no implicit respawn or retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55')
MODAL=TEMP.parent/'modalenv/Scripts/modal.exe'
sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,RATE


def cli(profile,*args):
    p=subprocess.run([str(MODAL),*args],cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'),
                     capture_output=True,text=True,encoding='utf-8',timeout=3600)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def handoff():
    lines=['# vi55 — pha 1; không pha2/nộp/browser/commit/crawl',
      '- Qwen8B pin1d8ad4ca9b3dd8059ad90a75d4983776a23d44af;fp16SDPA/left/LAST/L2;queryinstruct256.',
      '- Nguồnindexvi36/iter_all_texts/BoilerplateFilter.clean/make_bodyunlimited;740598VIstandard.',
      '- Audit29part223065vi51bPASS;loại68ZHlịchsử,táidùng222997VI2048;mới506707VI1024chốtsaubench.',
      '- Trần26USD/bench≤.4USD/max4H100mỗiworkspace; mục tiêu01:30VN/hardstop05:00VN01/11.',
      '- <HIST_TMP>/vi55;mirrorD:/r2ai/data/sub/vi55;giữVolume/appstopcuối;khôngđụngvi54.']
    for p in sorted(TEMP.glob('state_*.json')):
        s=read(p);lines.append(f"- {s['profile']} app={s.get('app_id')} upper={s.get('upper',0):.6f}USD stopped={s.get('stopped',False)} baseline={s.get('baseline',{}).get('metered_cost')}")
        for n,r in s['calls'].items():
            lines.append(f"- {n} {r.get('call_id','INTENT')} {r['status']} {r.get('parts',[])}")
            if r.get('result') and r.get('mode')=='bench':lines.append('- Bench '+json.dumps(r['result']))
    p=TEMP/'decision.json'
    if p.exists():
        d=read(p);lines.append(f"- Chốt{d['doc_len']}token/forecast+30%embed{d['chosen_forecast']['usd']:.6f}USD/totalreserve{d['total_reserved_upper']:.6f}<26.")
    p=TEMP/'prepare_report.json'
    if p.exists():
        r=read(p);lines.append(f"- Kho{r['corpus_vi']}/text{r['with_text']}/thiếu{r['without_text']}/mới{r['new_documents']};token1024={r['tokens1024']},512={r['tokens512']}.")
    lines.append('- Resume: modalenv/Scripts/python.exe deploy/vi55_local.py poll --profile PROFILE --job JOB --wait 0;khôngspawnlạijob.')
    lines.append('- Collector: python deploy/vi55_collect_search.py --profile PROFILE (pythonlocalcónumpy,controltựdùngmodalenv);2collectorđangnền,khôngchạytrùng.')
    lines.append('- B660prepare435.969s/LastResult0/taskđãxoá/lockabsent;inputaudit51part/SHA/exactID/tokenPASS.')
    lines.append('- FullsuiteD1192PASS2skip;10testvi55PASS;py_compilePASS. LầnC2crawlerlow_disk(chỉmôitrường)đãkhắcphụcbằngbasetempD.')
    assert len(lines)<=60
    write(TEMP/'handoff_state.json',dict(updated=time.time()))
    (ROOT/'docs/VI55_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['init','spawn','poll','upload','stop'])
    a.add_argument('--profile',required=True);a.add_argument('--job',default='weights');a.add_argument('--function',default='download_weights')
    a.add_argument('--mode',default='bench');a.add_argument('--parts',nargs='*',default=[]);a.add_argument('--watchdog',type=int,default=230)
    a.add_argument('--doc-len',type=int,default=1024);a.add_argument('--wait',type=int,default=0);args=a.parse_args()
    os.environ.update(MODAL_PROFILE=args.profile,PYTHONIOENCODING='utf-8')
    import modal
    TEMP.mkdir(parents=True,exist_ok=True);sp=TEMP/f'state_{args.profile}.json'
    s=read(sp) if sp.exists() else dict(profile=args.profile,calls={},upper=.05)
    def save():write(sp,s);handoff()
    if args.action=='init':
        apps=json.loads(cli(args.profile,'app','list','--json'))
        own=[x for x in apps if x['description']=='r2ai-vi55' and x['state']!='stopped'];assert len(own)==1
        s.update(app_id=own[0]['app_id'],baseline=json.loads(cli(args.profile,'billing','summary','--json')),stopped=False);save()
    elif args.action=='spawn':
        assert args.job not in s['calls'],'Never respawn ambiguous call'
        apps=json.loads(cli(args.profile,'app','list','--json'));write(TEMP/f'apps_{args.profile}_{args.job}.json',apps)
        assert any(x['app_id']==s['app_id'] and x['state']!='stopped' for x in apps)
        reserve=(args.watchdog+8)*RATE if args.function in ('work','retrieve') else .02
        assert sum(read(p)['upper'] for p in TEMP.glob('state_*.json'))+reserve<=26
        if args.function in ('work','retrieve'):
            assert sum(r['status']=='running' and r.get('function')=='work' for r in s['calls'].values())<4
            if args.mode=='bench':
                used=sum(r['reserve'] for p in TEMP.glob('state_*.json') for r in read(p)['calls'].values() if r.get('mode')=='bench' and r.get('function')=='work')
                assert used+reserve<=.4
        r=dict(status='spawn_intent',function=args.function,mode=args.mode if args.function=='work' else 'cpu',
               parts=args.parts,watchdog=args.watchdog,doc_len=args.doc_len,reserve=reserve,start=time.time())
        s['calls'][args.job]=r;s['upper']+=reserve;save()
        params=(args.mode,args.job,args.parts,args.watchdog,args.doc_len) if args.function=='work' else ((args.job,) if args.function=='retrieve' else ())
        call=modal.Function.from_name('r2ai-vi55',args.function).spawn(*params)
        r.update(call_id=call.object_id,status='running');save();print(json.dumps(r),flush=True)
    elif args.action=='poll':
        r=s['calls'][args.job];assert r['status']=='running'
        try:result=modal.FunctionCall.from_id(r['call_id']).get(timeout=args.wait)
        except (TimeoutError,modal.exception.TimeoutError):print('RUNNING '+r['call_id']);return
        except Exception as exc:
            r['error']=repr(exc)
            if isinstance(exc,(subprocess.CalledProcessError,modal.exception.RemoteError,modal.exception.FunctionTimeoutError)):r['status']='failed_terminal'
            save();raise
        r.update(status='done',finish=time.time(),result=result)
        if r['function'] in ('work','retrieve'):
            actual=(result['function_seconds']+8)*RATE;s['upper']+=actual-r['reserve'];r['reserve']=actual
        write(TEMP/f'{args.profile}_{args.job}.json',result);save();print(json.dumps({k:v for k,v in result.items() if k!='ids'}),flush=True)
    elif args.action=='upload':
        for name in args.parts:
            cli(args.profile,'volume','put','--force','r2ai-vi55-data',str(TEMP/'modal_input'/name),'/in/'+name)
            print('UPLOADED '+name,flush=True)
    else:
        assert all(r['status'] in ('done','failed_terminal') for r in s['calls'].values())
        cli(args.profile,'app','stop','-y',s['app_id']);s.update(stopped=True,after=json.loads(cli(args.profile,'billing','summary','--json')));save()


if __name__=='__main__':main()
