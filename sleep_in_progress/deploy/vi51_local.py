"""Explicit launch, persisted call IDs before poll; isolated vi51 ledger."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')
MODAL=__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'
VOL='r2ai-vi51-data';APP='r2ai-vi51';RATE=4.5232/3600


def cli(profile,*args):
    p=subprocess.run([MODAL,*args],cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'),
        capture_output=True,text=True,encoding='utf-8',timeout=1800)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def handoff():
    lines=['# vi51 — không nộp/browser/commit/crawl',
        '- RRF k60×1 bổ sung trên toàn pool vi41c; LAST/L2, title+clean2048/query instruct256; chunk/count vi48+bảo vệvi41c.',
        '- Revisions a=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3; b=1d8ad4ca9b3dd8059ad90a75d4983776a23d44af.',
        '- Release 05/06/2025 theo model-card-linked Qwen blog; Apache2.0; .596B/7.567B (<15B).',
        '- App r2ai-vi51 / Volume r2ai-vi51-data; retries0; chỉ spawn sau applist, call chưaterminal KHÔNGrespawn.',
        '- Trần30USD; <MODAL_PROFILE_A> bench≤.6 + bulk a≤2; 8B bulk CHỈprofilemới; mỗiworkspace≤2H100.',
        '- Prepare lần1 lỗi assumption tokenizer thêm special tokens khác; sửa tokenize riêng mỗi tier (khôngđổi quy tắc).',
        '- Code scripts/vi51.py, deploy/modal_vi51.py, deploy/vi51_local.py; data Temp/vi51, b660 data/sub/vi51.',
        '- Profile30b/<MODAL_PROFILE> đã có16:35VN, khôngcầnlogin;30c chưa có. Không tiếpGPU sau2timeout.']
    if (TEMP/'prepare_report.json').exists():
        p=json.loads((TEMP/'prepare_report.json').read_text(encoding='utf-8'))
        lines.append(f"- PrepareDONE {p['documents']}doc/{p['pairs']}pair/{p['tokens']}token(a); {len(p['parts'])}part.")
    for p in sorted(TEMP.glob('state_*.json')):
        s=json.loads(p.read_text(encoding='utf-8'))
        lines.append(f"- {s['profile']} app={s.get('app_id')} upper=${s.get('reserved_upper',0):.6f} stopped={s.get('stopped',False)}")
        for n,r in s['calls'].items():
            lines.append(f"- {n} {r.get('call_id','INTENT')} {r['status']} {r.get('tier','')} {r.get('mode','')}; watchdog={r.get('watchdog')}")
        if s.get('stop_reason'):lines.append('- DỪNG: '+s['stop_reason'])
        if s.get('gpu_function_seconds'):lines.append(f"- GPUfunction={s['gpu_function_seconds']:.6f}s; throughput/cosine/forecast chưađo; KHÔNGbulk/ZIP.")
    assert len(lines)<=60
    (ROOT/'docs/VI51_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['adopt','deploy','ping','launch','poll','upload','stop'])
    ap.add_argument('--profile',required=True);ap.add_argument('--job',default='ping');ap.add_argument('--tier',choices=['a','b'],default='a')
    ap.add_argument('--mode',choices=['bench','bulk'],default='bench');ap.add_argument('--parts',nargs='*',default=[])
    ap.add_argument('--watchdog',type=int,default=180);ap.add_argument('--wait',type=int,default=0)
    a=ap.parse_args();TEMP.mkdir(exist_ok=True)
    os.environ['MODAL_PROFILE']=a.profile;os.environ['PYTHONIOENCODING']='utf-8'
    import modal
    sp=TEMP/f'state_{a.profile}.json'
    state=json.loads(sp.read_text(encoding='utf-8')) if sp.exists() else dict(profile=a.profile,calls={},reserved_upper=.015)
    def save():
        if sp.exists():
            fresh=json.loads(sp.read_text(encoding='utf-8'))
            for name,rec in fresh.get('calls',{}).items():
                if name not in state['calls'] or (rec['status']=='done' and state['calls'][name]['status']!='done'):
                    state['calls'][name]=rec
            state['reserved_upper']=max(state['reserved_upper'],fresh.get('reserved_upper',0))
        sp.write_text(json.dumps(state,indent=2),encoding='utf-8');handoff()
    if a.action in ('adopt','deploy'):
        apps=json.loads(cli(a.profile,'app','list','--json'))
        state['billing_before']=json.loads(cli(a.profile,'billing','summary','--json'));save()
        if a.action=='deploy':
            (TEMP/f'deploy_{a.profile}.log').write_text(cli(a.profile,'deploy','deploy/modal_vi51.py'),encoding='utf-8')
            apps=json.loads(cli(a.profile,'app','list','--json'))
        own=[x for x in apps if x['description']==APP and x['state']!='stopped']
        assert len(own)==1;state['app_id']=own[0]['app_id'];save()
    elif a.action=='upload':
        manifest=json.loads((TEMP/'prepare_report.json').read_text(encoding='utf-8'))
        from hashlib import sha256
        for n in a.parts:
            p=TEMP/'modal_input'/n
            assert sha256(p.read_bytes()).hexdigest()==manifest['files'][n]['sha256']
            cli(a.profile,'volume','put',VOL,str(p),f'/in/{n}')
            print('UPLOADED '+n,flush=True)
    elif a.action in ('ping','launch'):
        name='ping' if a.action=='ping' else a.job
        assert name not in state['calls'], 'Never repeat ambiguous call'
        apps=json.loads(cli(a.profile,'app','list','--json'))
        (TEMP/f'apps_before_{a.profile}_{name}.json').write_text(json.dumps(apps,indent=2),encoding='utf-8')
        assert any(x['app_id']==state['app_id'] and x['state']!='stopped' for x in apps)
        if a.action=='launch':
            assert state['calls']['ping']['status']=='done'
            assert sum(r['status'] in ('running','spawn_intent') and n!='ping' for n,r in state['calls'].items())<2
            reserve=(a.watchdog+8)*RATE
            allstates=[json.loads(p.read_text(encoding='utf-8')) for p in TEMP.glob('state_*.json')]
            assert sum(s['reserved_upper'] for s in allstates)+reserve<=30
            if a.profile==__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"):
                assert not (a.tier=='b' and a.mode=='bulk')
                assert state['reserved_upper']+reserve<=2.6
                if a.mode=='bench':
                    assert sum((r.get('watchdog',0)+8)*RATE for n,r in state['calls'].items() if n!='ping' and r.get('mode')=='bench')+reserve<=.58
            state['reserved_upper']+=reserve
        state['calls'][name]=dict(status='spawn_intent',tier=a.tier,mode=('cpu' if a.action=='ping' else a.mode),parts=a.parts,watchdog=a.watchdog,start=time.time());save()
        call=modal.Function.from_name(APP,'ping' if a.action=='ping' else 'work').spawn(*(() if a.action=='ping' else (a.tier,a.mode,name,a.parts,a.watchdog)))
        state['calls'][name].update(status='running',call_id=call.object_id);save()
        print(json.dumps(state['calls'][name]),flush=True)
    elif a.action=='poll':
        r=state['calls'][a.job];assert r['status']=='running'
        try:result=modal.FunctionCall.from_id(r['call_id']).get(timeout=a.wait)
        except (TimeoutError,modal.exception.TimeoutError):
            print('RUNNING '+r['call_id']);return
        except Exception as exc:
            r['last_poll_error']=repr(exc);save();raise
        r.update(status='done',result=result,finish=time.time());save()
        if a.job!='ping':
            folder=TEMP if r['mode']=='bench' else TEMP.parent/('vi51'+r['tier'])
            dest=folder/'outputs'/a.profile/a.job;dest.mkdir(parents=True,exist_ok=True)
            files=['bench_report.json','bench_embeddings.npz','function_wall.json'] if r['mode']=='bench' else list(result['files'])+['bulk_report.json','function_wall.json']
            for n in files:cli(a.profile,'volume','get','--force',VOL,f"/out/{r['tier']}/{a.job}/{n}",str(dest/n))
        print(json.dumps(result),flush=True)
    else:
        assert all(r['status']=='done' for r in state['calls'].values())
        cli(a.profile,'app','stop','--yes',state['app_id']);state['stopped']=True
        state['billing_after']=json.loads(cli(a.profile,'billing','summary','--json'));save()


if __name__=='__main__':main()
