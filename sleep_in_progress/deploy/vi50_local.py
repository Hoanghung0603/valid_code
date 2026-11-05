"""Single explicit launch/poll; no automatic retry, bounded cross-workspace ledger."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi50')
MODAL=__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'


def cli(profile,*args):
    p=subprocess.run([MODAL,*args],cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'),
        capture_output=True,text=True,encoding='utf-8',timeout=1800)
    if p.returncode:
        raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def handoff():
    states=[json.loads(p.read_text(encoding='utf-8')) for p in TEMP.glob('state_*.json')]
    lines=['# vi50 — GPU bulk, không nộp/browser/commit/crawl',
        '- Trần40USD tổng; <MODAL_PROFILE_A>≤20USDupper; mỗiworkspace≤2H100; retries0; khôngbench.',
        '- Nguồn RRFthứ6 k60×1; scopevi41c underlyingvi41b topn_q+100; chunk/countvi48giữ.',
        '- Input/forecast/cid: <HIST_TMP>/vi50/prepare_report.json; code scripts/vi50.py.',
        '- ZIP mục tiêu20:00VN; hardstop21:30VN; call khôngterminal tuyệtđối khôngrespawn.']
    if (TEMP/'prepare_report.json').exists():
        p=json.loads((TEMP/'prepare_report.json').read_text(encoding='utf-8'))
        lines.append(f"- PrepareDONE {p['pairs']}U900/{p['reused_pairs']}reuse/{p['new_pairs']}new; top8={p['top8']}; forecastupper${p['forecast']['usd_upper']:.6f}.")
    if (TEMP/'independent_prepare_audit.json').exists():
        lines.append('- IndependentpreparePASS all539208U900/320960querydoc/150799doc/verbatim0;SHA16part+contextkhớp2máy.')
    lines.append('- TestCPU1080PASS2skip(gồm11testvi50/E2E1200validate/checkzip/1build);taskprepare/auditSYSTEMLastResult0đãxoá.')
    lines.append('- Rankhistoricalsanityvi41b+vi41c1200PASS;thứhạngtoànpoollàfhyde5trướcchènmandatory,khôngRRFtrêndoclistđãcắt.')
    lines.append('- Automationvi50_continue.py:refreshprofilemỗiphần;after_bulk.ps1→finish_local.py(ledger/1build/audit);lỗithìdừng.')
    for s in states:
        lines.append(f"- {s['profile']} app={s.get('app_id')} upper=${s.get('reserved_upper',0):.6f}; state_{s['profile']}.json")
        for name,r in s['calls'].items():
            lines.append(f"- {s['profile']}/{name}: {r.get('call_id','INTENT')} {r['status']}; parts={r.get('parts')}")
    assert len(lines)<=60
    (ROOT/'docs/VI50_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['deploy','ping','launch','poll','stop'])
    ap.add_argument('--profile',required=True);ap.add_argument('--job',default='ping')
    ap.add_argument('--parts',nargs='*');ap.add_argument('--watchdog',type=int,default=1500)
    ap.add_argument('--wait',type=int,default=0)
    a=ap.parse_args();TEMP.mkdir(exist_ok=True)
    os.environ['MODAL_PROFILE']=a.profile;os.environ['PYTHONIOENCODING']='utf-8'
    import modal
    sp=TEMP/f'state_{a.profile}.json'
    state=json.loads(sp.read_text(encoding='utf-8')) if sp.exists() else dict(profile=a.profile,calls={},reserved_upper=.1)
    def save():
        if sp.exists():
            fresh=json.loads(sp.read_text(encoding='utf-8'))
            for name,rec in fresh.get('calls',{}).items():
                if name not in state['calls'] or (rec['status']=='done' and state['calls'][name]['status']!='done'):
                    state['calls'][name]=rec
            state['reserved_upper']=max(state['reserved_upper'],fresh.get('reserved_upper',0))
        sp.write_text(json.dumps(state,indent=2),encoding='utf-8');handoff()
    if a.action=='deploy':
        apps=json.loads(cli(a.profile,'app','list','--json'))
        assert not any(x['description']=='r2ai-vi50' and x['state']!='stopped' for x in apps)
        state['billing_before']=json.loads(cli(a.profile,'billing','summary','--json'));save()
        modal.Volume.from_name('r2ai-runner',create_if_missing=True).hydrate()
        (TEMP/f'deploy_{a.profile}.log').write_text(cli(a.profile,'deploy','deploy/modal_vi50.py'),encoding='utf-8')
        own=[x for x in json.loads(cli(a.profile,'app','list','--json')) if x['description']=='r2ai-vi50' and x['state']!='stopped']
        assert len(own)==1;state['app_id']=own[0]['app_id'];save()
    elif a.action in ('ping','launch'):
        name='ping' if a.action=='ping' else a.job
        assert name not in state['calls'], 'Never repeat ambiguous call'
        if a.action=='launch':
            # Required fresh inventory before EACH part allocation, no token access.
            (TEMP/f'profiles_before_{a.profile}_{name}.txt').write_text(cli(a.profile,'profile','list'),encoding='utf-8')
            assert state['calls']['ping']['status']=='done'
            assert sum(r['status'] in ('running','spawn_intent') and n!='ping' for n,r in state['calls'].items())<2
            manifest=json.loads((TEMP/'prepare_report.json').read_text(encoding='utf-8'))
            parts=[]
            assigned=set()
            upper=0
            for p in TEMP.glob('state_*.json'):
                s=json.loads(p.read_text(encoding='utf-8'));upper+=s['reserved_upper']
                assigned.update(n for r in s['calls'].values() for n in (r.get('parts') or []))
            assert not set(a.parts)&assigned
            for n in a.parts:
                p=TEMP/'modal_input'/n
                assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['files'][n]['sha256']
                parts.append(dict(name=n,sha256=manifest['files'][n]['sha256'],count=manifest['counts'][n]))
                cli(a.profile,'volume','put','r2ai-runner',str(p),f'/in/vi50/jobs/{name}/{n}')
            config=TEMP/f'{name}.job.json';config.write_text(json.dumps(dict(parts=parts)),encoding='utf-8')
            cli(a.profile,'volume','put','r2ai-runner',str(config),f'/in/vi50/jobs/{name}/job.json')
            reserve=(a.watchdog+90)*4.5232/3600
            assert upper+reserve<=40
            assert state['reserved_upper']+reserve<=(20 if a.profile==__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>") else 29)
            state['reserved_upper']+=reserve
        apps=json.loads(cli(a.profile,'app','list','--json'))
        assert any(x['app_id']==state['app_id'] and x['state']!='stopped' for x in apps)
        state['calls'][name]=dict(status='spawn_intent',parts=a.parts,watchdog=a.watchdog,start=time.time());save()
        call=modal.Function.from_name('r2ai-vi50','ping' if a.action=='ping' else 'bulk').spawn(*(() if a.action=='ping' else (name,a.watchdog)))
        state['calls'][name].update(status='running',call_id=call.object_id);save()
        print(json.dumps(state['calls'][name]))
    elif a.action=='poll':
        r=state['calls'][a.job]
        assert r['status']=='running'
        try:
            result=modal.FunctionCall.from_id(r['call_id']).get(timeout=a.wait)
        except (TimeoutError,modal.exception.TimeoutError):
            print('RUNNING '+r['call_id']);return
        except Exception as exc:
            # Unknown network/server errors are NOT terminal proof.
            r['last_poll_error']=repr(exc);save();raise
        r.update(status='done',result=result,finish=time.time());save()
        if a.job!='ping':
            dest=TEMP/'outputs'/a.profile/a.job;dest.mkdir(parents=True,exist_ok=True)
            for part in result['parts']:
                n=part['part'].replace('.jsonl.zst','.logits.json')
                cli(a.profile,'volume','get','--force','r2ai-runner',f'/out/vi50/bulk/{a.job}/{n}',str(dest/n))
                assert hashlib.sha256((dest/n).read_bytes()).hexdigest()==part['output_sha256']
            cli(a.profile,'volume','get','--force','r2ai-runner',f'/out/vi50/bulk/{a.job}/function_wall.json',str(dest/'function_wall.json'))
        print(json.dumps(result))
    else:
        assert all(r['status']=='done' for r in state['calls'].values())
        cli(a.profile,'app','stop','--yes',state['app_id'])
        state['billing_after']=json.loads(cli(a.profile,'billing','summary','--json'))
        state['apps_after']=json.loads(cli(a.profile,'app','list','--json'));save()


if __name__=='__main__':
    main()
