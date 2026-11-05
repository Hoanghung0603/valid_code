"""Explicit profile/part control; saved intent and call ID before any polling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')
MODAL=__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'


def cli(profile,*args):
    p=subprocess.run([MODAL,*args],cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'),
        capture_output=True,text=True,encoding='utf-8',timeout=1800)
    if p.returncode:
        raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['deploy','ping','launch','poll','stop'])
    ap.add_argument('--profile',required=True)
    ap.add_argument('--job',default='ping')
    ap.add_argument('--parts',nargs='*')
    ap.add_argument('--watchdog',type=int,default=1250)
    ap.add_argument('--wait',type=int,default=0)
    ap.add_argument('--staged',action='store_true')
    a=ap.parse_args()
    os.environ['MODAL_PROFILE']=a.profile
    os.environ['PYTHONIOENCODING']='utf-8'
    import modal
    sp=TEMP/('bulk_state_'+a.profile+'.json')
    state=json.loads(sp.read_text()) if sp.exists() else dict(profile=a.profile,calls={})
    def save():
        if sp.exists():
            fresh=json.loads(sp.read_text())
            for name,rec in fresh.get('calls',{}).items():
                if name not in state['calls'] or (rec['status'] in ('done','failed') and state['calls'][name]['status'] not in ('done','failed')):
                    state['calls'][name]=rec
            if fresh.get('reserved_upper',0)>state.get('reserved_upper',0):
                state['reserved_upper']=fresh['reserved_upper']
        sp.write_text(json.dumps(state,indent=2),encoding='utf-8')
    if a.action=='deploy':
        apps=json.loads(cli(a.profile,'app','list','--json'))
        assert not any(x['description']=='r2ai-vi45' and x['state']!='stopped' for x in apps)
        state['billing_before']=json.loads(cli(a.profile,'billing','summary','--json'))
        state['deploy_start']=time.time()
        save()
        (TEMP/('deploy_bulk_'+a.profile+'.log')).write_text(cli(a.profile,'deploy','deploy/modal_vi45.py'),encoding='utf-8')
        own=[x for x in json.loads(cli(a.profile,'app','list','--json')) if x['description']=='r2ai-vi45' and x['state']!='stopped']
        assert len(own)==1
        state['app_id']=own[0]['app_id']
        save()
        print(json.dumps(state))
    elif a.action in ('ping','launch'):
        name='ping' if a.action=='ping' else a.job
        assert name not in state['calls'], 'Never repeat an ambiguous call'
        if a.action=='launch':
            assert state['calls']['ping']['status']=='done'
            manifest=json.loads((TEMP/'prepare_vi44_report.json').read_text())
            parts=[]
            for n in a.parts:
                p=TEMP/'modal_input'/n
                assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['files'][n]['sha256']
                parts.append(dict(name=n,sha256=manifest['files'][n]['sha256'],count=manifest['counts'][n]))
                if not a.staged:
                    cli(a.profile,'volume','put','r2ai-runner',str(p),'/in/vi45/jobs/'+name+'/'+n)
            config=TEMP/(name+'.job.json')
            config.write_text(json.dumps(dict(parts=parts)))
            if not a.staged:
                cli(a.profile,'volume','put','r2ai-runner',str(config),'/in/vi45/jobs/'+name+'/job.json')
            state.setdefault('reserved_upper',0.)
            reserve=(a.watchdog+25)*4.5232/3600
            limit=3.6 if a.profile==__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>") else 8.0
            assert state['reserved_upper']+reserve<=limit
            other=__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>") if a.profile==__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>") else __import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>")
            otherstate=TEMP/('bulk_state_'+other+'.json')
            otherreserve=json.loads(otherstate.read_text()).get('reserved_upper',0) if otherstate.exists() else 0
            assert .319106727+.3+otherreserve+state['reserved_upper']+reserve<=12., 'Total vi45 upper cap'
            state['reserved_upper']+=reserve
        apps=json.loads(cli(a.profile,'app','list','--json'))
        assert any(x['app_id']==state['app_id'] and x['state']!='stopped' for x in apps)
        state['calls'][name]=dict(status='spawn_intent',start=time.time(),parts=a.parts,watchdog=a.watchdog)
        save()
        call=modal.Function.from_name('r2ai-vi45','ping' if a.action=='ping' else 'bulk').spawn(*(() if a.action=='ping' else (name,a.watchdog)))
        state['calls'][name].update(status='running',call_id=call.object_id)
        save()
        with (ROOT/'docs/VI45_HANDOFF.md').open('a',encoding='utf-8') as f:
            f.write(f'- CALL {a.profile}/{name}: `{call.object_id}`; parts={a.parts}; running (state={sp.name}).\n')
        print(json.dumps(state['calls'][name]))
    elif a.action=='poll':
        rec=state['calls'][a.job]
        try:
            result=modal.FunctionCall.from_id(rec['call_id']).get(timeout=a.wait)
        except (TimeoutError,modal.exception.TimeoutError):
            print('RUNNING '+rec['call_id'])
            return
        except Exception as exc:
            rec.update(status='failed',error=repr(exc),finish=time.time())
            save()
            raise
        rec.update(status='done',result=result,finish=time.time())
        save()
        if a.job!='ping':
            dest=TEMP/'outputs'/a.profile/a.job
            dest.mkdir(parents=True,exist_ok=True)
            for r in result['parts']:
                n=r['part'].replace('.jsonl.zst','.logits.json')
                cli(a.profile,'volume','get','--force','r2ai-runner','/out/vi45/bulk/'+a.job+'/'+n,str(dest/n))
                assert hashlib.sha256((dest/n).read_bytes()).hexdigest()==r['output_sha256']
            cli(a.profile,'volume','get','--force','r2ai-runner','/out/vi45/bulk/'+a.job+'/function_wall.json',str(dest/'function_wall.json'))
        print(json.dumps(result))
    else:
        assert all(c['status'] in ('done','failed') for c in state['calls'].values())
        cli(a.profile,'app','stop','--yes',state['app_id'])
        state['billing_after']=json.loads(cli(a.profile,'billing','summary','--json'))
        state['apps_after']=json.loads(cli(a.profile,'app','list','--json'))
        save()
        print(json.dumps(state))


if __name__=='__main__':
    main()
