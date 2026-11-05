"""Local audited transfers and single-launch budget watchdog. No submission/browser."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def prediction(tokens, tokens_per_second, loaded, hourly):
    seconds = tokens/tokens_per_second*1.3+loaded+120
    return dict(seconds=seconds, usd=seconds*hourly/3600)

TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/dchunk')
MODAL = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'
os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
RATE = (1.95+2*.0473+16*.008)/3600


def cmd(*args):
    p = subprocess.run([MODAL,*args],capture_output=True,text=True,encoding='utf-8',errors='replace')
    if p.returncode and not (args[:2]==('app','stop') and 'App is already stopped' in p.stdout+p.stderr):
        raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resume_forecast(manifest, plan, bench, prior_upper, seconds):
    known = {x['name']:x for x in manifest['files']}
    run = plan['run_parts']
    assert len(run)==len(set(run)) and set(run)<=set(known)
    assert set(run).isdisjoint({x['name'] for x in plan['good']})
    assert set(run)|{x['name'] for x in plan['good']}==set(known)
    tokens = sum(known[n]['tokens'] for n in run)
    pred = prediction(tokens,bench['tokens_per_second'],bench['loaded_seconds'],bench['hourly_upper'])
    assert seconds >= pred['seconds']
    assert prior_upper+pred['usd'] < 5
    assert prior_upper+seconds*bench['hourly_upper']/3600+.15 < 5
    return dict(prior_upper_usd=prior_upper,predicted_remaining_usd=pred['usd'],
                predicted_remaining_seconds=pred['seconds'],remaining_tokens=tokens,run_parts=run)


def poll_cmd(*args):
    """Read-only reconnect; never relaunch. Stop after 20 continuous minutes offline."""
    deadline = time.monotonic()+1200
    while True:
        try:
            return cmd(*args)
        except RuntimeError as exc:
            text = str(exc)
            if 'billing' in text.lower() or not any(s in text for s in ('connect','Timeout','timeout','getaddrinfo','resolve','DNS','unavailable')):
                raise
            if time.monotonic()>=deadline:
                raise TimeoutError('Offline >20 minutes; detached app may still run; NO relaunch') from exc
            time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['upload','bench','bulk','fetch','status','audit-input','forecast','recover-bench','audit-stopped-bulk','audit-partial','resume','wait-resume'])
    ap.add_argument('--seconds',type=int,default=6000)
    ap.add_argument('--attempt',type=int,default=1)
    ap.add_argument('--scope',choices=['full','expanded_pool'],default='full')
    a = ap.parse_args()
    TEMP.mkdir(parents=True,exist_ok=True)
    inp = TEMP / 'input'
    ledger = TEMP / 'ledger.json'
    history = json.loads(ledger.read_text()) if ledger.exists() else []
    def save():
        ledger.write_text(json.dumps(history,indent=2),encoding='utf-8')
    def stopped(apps):
        active = [x for x in apps if str(x.get('state',x.get('State',''))).lower() not in ('stopped','') or str(x.get('tasks','0'))!='0']
        assert not active, active
    if a.stage=='audit-partial':
        stopped(json.loads(cmd('app','list','--json')))
        before=json.loads(cmd('billing','summary','--json'))
        log=cmd('run','deploy/modal_dchunk.py','--stage','audit-partial')
        (TEMP/'partial_audit_modal.log').write_text(log,encoding='utf-8')
        for name in ('resume_plan.json','partial_samples.npz'):
            cmd('volume','get','--force','r2ai-runner','/out/dchunk/'+name,str(TEMP/name))
        after=json.loads(cmd('billing','summary','--json'))
        history.append(dict(stage='audit-partial',billing_before=before,billing_after=after,
                            snapshot_delta=float(after['metered_cost'])-float(before['metered_cost']),upper_usd=.03))
        save()
        print((TEMP/'resume_plan.json').read_text())
    elif a.stage=='resume':
        apps=json.loads(cmd('app','list','--json'))
        stopped(apps)
        previous=[x for x in history if x['stage']=='resume']
        assert a.attempt==len(previous)+1, 'No automatic relaunch'
        if previous:
            assert a.attempt>1 and previous[-1].get('apps_after_stop'), 'Explicit stopped audit required'
        plan=json.loads((TEMP/'resume_plan.json').read_text())
        assert plan['manifest_sha256']==sha(inp/'manifest.json')
        b=json.loads((TEMP/'bench_report.json').read_text())
        baseline=json.loads(cmd('billing','summary','--json'))
        prior=max(sum(max(x.get('snapshot_delta',0),x['upper_usd']) for x in history),
                  float(baseline['metered_cost'])-float(history[0]['billing_before']['metered_cost']))
        budget=resume_forecast(json.loads((inp/'manifest.json').read_text()),plan,b,prior,a.seconds)
        budget['plan_sha256']=sha(TEMP/'resume_plan.json')
        (TEMP/'resume_budget.json').write_text(json.dumps(budget,indent=2))
        cmd('volume','put','r2ai-runner',str(TEMP/'resume_budget.json'),'/out/dchunk/resume_budget.json')
        # Recheck immediately before launch, after the network transfer.
        stopped(json.loads(cmd('app','list','--json')))
        r=dict(stage='resume',attempt=a.attempt,started=time.time(),billing_before=baseline,seconds_budget=a.seconds,
               upper_usd=(a.seconds+30)*b['hourly_upper']/3600+.05,**budget)
        assert prior+r['upper_usd']<5
        history.append(r)
        save()
        log=TEMP/('resume_modal.log' if a.attempt==1 else f'resume{a.attempt}_modal.log')
        with log.open('w',encoding='utf-8') as f:
            p=subprocess.Popen([MODAL,'run','--detach','deploy/modal_dchunk.py','--stage','resume',
                                '--seconds-budget',str(a.seconds),'--gpu-kind','H100'],stdout=f,stderr=subprocess.STDOUT)
        r['client_pid']=p.pid
        import re
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            text=log.read_text(encoding='utf-8')
            ids=set(re.findall(r'\bap-[A-Za-z0-9]+',text))
            if ids:
                assert len(ids)==1
                r['app_id']=ids.pop()
                save()
                print(json.dumps(r,indent=2))
                break
            if p.poll() is not None:
                raise RuntimeError(text)
            time.sleep(2)
        else:
            raise TimeoutError('Launch identity unknown; inspect apps; NEVER relaunch')
    elif a.stage=='wait-resume':
        from datetime import datetime
        r=[x for x in history if x['stage']=='resume'][-1]
        while True:
            apps=json.loads(poll_cmd('app','list','--json'))
            app=next(x for x in apps if x['app_id']==r['app_id'])
            if app['state']=='stopped' and str(app['tasks'])=='0':
                break
            time.sleep(30)
        wall=(datetime.fromisoformat(app['stopped_at'])-datetime.fromisoformat(app['created_at'])).total_seconds()
        b=json.loads((TEMP/'bench_report.json').read_text())
        r.update(app_wall_seconds=wall,gpu_residency_seconds_upper=wall+15,
                 upper_usd=(wall+15)*b['hourly_upper']/3600+.05,apps_after_stop=apps)
        r['billing_after']=json.loads(poll_cmd('billing','summary','--json'))
        r['snapshot_delta']=float(r['billing_after']['metered_cost'])-float(r['billing_before']['metered_cost'])
        save()
        assert sum(max(x.get('snapshot_delta',0),x['upper_usd']) for x in history)<5
        for name in ('partial_cosine_report.json','embedding_complete.json','top1000.npz','bulk_report.json'):
            poll_cmd('volume','get','--force','r2ai-runner','/out/dchunk/'+name,str(TEMP/name))
        rep=json.loads((TEMP/'bulk_report.json').read_text())
        assert sha(TEMP/'top1000.npz')==rep['top1000_sha256']
        assert rep['run_parts']==r['run_parts']
        r['function_gpu_seconds']=rep['total_seconds']
        r['exit']=0
        save()
        print(json.dumps(dict(ledger=r,bulk=rep),indent=2))
    elif a.stage=='audit-stopped-bulk':
        from datetime import datetime
        r=next(x for x in history if x['stage']=='bulk')
        apps=json.loads(cmd('app','list','--json'))
        stopped(apps)
        app=next(x for x in apps if x['app_id']=='ap-FV6Y6PIVNgIogoKOkpDyS3')
        wall=(datetime.fromisoformat(app['stopped_at'])-datetime.fromisoformat(app['created_at'])).total_seconds()
        assert wall>=0
        rate=json.loads((TEMP/'bench_report.json').read_text())['hourly_upper']/3600
        r.update(exit='client DNS failure; bulk incomplete; NO relaunch',apps_after_stop=apps,
                 app_wall_seconds=wall,gpu_residency_seconds_upper=wall+15,upper_usd=(wall+15)*rate+.015)
        r['billing_after']=json.loads(cmd('billing','summary','--json'))
        r['snapshot_delta']=float(r['billing_after']['metered_cost'])-float(r['billing_before']['metered_cost'])
        vectors=json.loads(cmd('volume','ls','r2ai-runner','/out/dchunk/vectors','--json'))
        names={Path(x['filename']).name for x in vectors}
        manifest=json.loads((inp/'manifest.json').read_text())
        parts=[x for x in manifest['files'] if x['name'] in names]
        assert len(parts)==len(vectors)==3
        r['partial_vectors']=vectors
        r['persisted_chunks']=sum(x['chunks'] for x in parts)
        r['persisted_tokens']=sum(x['tokens'] for x in parts)
        r['partial_note']='Persisted files only: vectors not downloaded/independently audited; actual additional uncommitted work unknown'
        save()
        rep=dict(ledger=history,total_snapshot_delta=float(r['billing_after']['metered_cost'])-float(history[0]['billing_before']['metered_cost']),
                 total_upper_usd=sum(x['upper_usd'] for x in history),billing_note='Unsettled nonmonotonic billing snapshots; upper includes startup/cleanup and extra reserve')
        (TEMP/'stop_report.json').write_text(json.dumps(rep,indent=2))
        assert rep['total_upper_usd']<5
        print(json.dumps(rep,indent=2))
    elif a.stage=='recover-bench':
        import re
        assert len(history)==1 and history[0]['stage']=='bench' and 'upper_usd' not in history[0]
        log=(TEMP/'bench_modal.log').read_text(encoding='utf-8')
        assert 'App completed.' in log and len(re.findall('BENCH \\{',log))==2
        ids=sorted(set(re.findall(r'\bap-[A-Za-z0-9]+',log)))
        for ident in ids:
            print(cmd('app','stop','--yes',ident))
        r=history[0]
        r.update(exit=0,recovery='post-completion stop already stopped; NO relaunch',stopped_app_ids=ids,
                 elapsed_wall=time.time()-r['started'],upper_usd=105*(2.1726+4.7626)/3600+.04)
        r['billing_after']=json.loads(cmd('billing','summary','--json'))
        r['snapshot_delta']=float(r['billing_after']['metered_cost'])-float(r['billing_before']['metered_cost'])
        r['apps_after_stop']=json.loads(cmd('app','list','--json'))
        save()
        stopped(r['apps_after_stop'])
        assert r['snapshot_delta']<=.3 and r['upper_usd']<=.3
        cmd('volume','get','r2ai-runner','/out/dchunk/volume_audit.json',str(TEMP/'volume_audit.json'))
        va=json.loads((TEMP/'volume_audit.json').read_text())
        assert va['manifest_sha256']==sha(inp/'manifest.json') and va['sha_errors']==0
        assert all(va['small_sha256'][f]==sha(inp/f) for f in ('bench.npz','queries.json'))
        reports=[]
        for kind in ('L40S','H100'):
            name='bench_'+kind+'.json'
            cmd('volume','get','r2ai-runner','/out/dchunk/'+name,str(TEMP/name))
            reports.append(json.loads((TEMP/name).read_text()))
        assert all(x['manifest_sha256']==sha(inp/'manifest.json') for x in reports)
        b=min(reports,key=lambda x:x['predicted_bulk_usd'])
        (TEMP/'bench_report.json').write_text(json.dumps(dict(b,comparisons=reports),indent=2))
        cmd('volume','put','r2ai-runner',str(TEMP/'bench_report.json'),'/out/dchunk/bench_report.json')
        print(json.dumps(dict(ledger=r,selected=b),indent=2))
    elif a.stage=='forecast':
        b = json.loads((TEMP/'bench_report.json').read_text())
        ar = json.loads((TEMP/'audit_report.json').read_text())
        candidates = dict(full=ar['embedded_tokens'],**ar['alternatives'])
        candidates.pop('expanded_pool_documents',None)
        rep = {name:dict(tokens=tokens,**prediction(tokens,b['tokens_per_second'],b['loaded_seconds'],b['hourly_upper']))
               for name,tokens in candidates.items()}
        rep['note']='Token-linear estimates, +30% overhead + startup + 120s search; expanded_pool = dense-chunk trên pool mở rộng, KHÔNG phải toàn kho'
        (TEMP/'forecast.json').write_text(json.dumps(rep,indent=2))
        print(json.dumps(rep,indent=2))
    elif a.stage=='audit-input':
        import numpy as np
        manifest = json.loads((inp/'manifest.json').read_text())
        count = tokens = 0
        for part in manifest['files']:
            p = inp/part['name']
            assert sha(p)==part['sha256'] and p.stat().st_size==part['bytes']
            with np.load(p) as z:
                ids,ptr,cids = z['ids'],z['ptr'],z['chunk_id']
            assert ptr[0]==0 and ptr[-1]==len(ids)==part['tokens']
            assert np.array_equal(cids,np.arange(count,count+part['chunks']))
            assert np.diff(ptr).min()>=3 and np.diff(ptr).max()<=514
            assert np.all(ids[ptr[:-1]]==0)
            assert np.all(ids[ptr[1:]-1]==2)
            count += part['chunks']
            tokens += len(ids)
        assert count==manifest['chunks'] and tokens==manifest['embedded_tokens_with_overlap_specials']
        rep = dict(chunks=count,tokens=tokens,sha_errors=0,token_shape_errors=0,
                   manifest_sha256=sha(inp/'manifest.json'),input_bytes=manifest['bytes'])
        (TEMP/'input_audit.json').write_text(json.dumps(rep,indent=2))
        print(json.dumps(rep),flush=True)
    elif a.stage=='upload':
        manifest = json.loads((inp/'manifest.json').read_text())
        files = ['manifest.json','queries.json','bench.npz']+[x['name'] for x in manifest['files']]
        known = {x['name']:x for x in manifest['files']}
        remote = {Path(x['filename']).name:x for x in json.loads(cmd('volume','ls','r2ai-runner','/in/dchunk','--json'))}
        sent = []
        for f in files:
            p = inp/f
            if f in known:
                assert sha(p)==known[f]['sha256'] and p.stat().st_size==known[f]['bytes']
            if f in remote:
                assert remote[f]['size']==p.stat().st_size, ('Remote size mismatch',f)
                print('EXISTS',f,flush=True)
            else:
                print('PUT',f,p.stat().st_size,flush=True)
                cmd('volume','put','r2ai-runner',str(p),'/in/dchunk/'+f)
            sent.append(dict(name=f,bytes=p.stat().st_size,sha256=sha(p)))
        (TEMP/'upload_audit.json').write_text(json.dumps(sent,indent=2))
        # Read back the small inputs. Existing NPZ files have local manifest SHA,
        # remote byte equality here and will be SHA-checked inside the bulk GPU job.
        for f in ('manifest.json','queries.json','bench.npz'):
            p = TEMP/('volume_check_'+f)
            cmd('volume','get','r2ai-runner','/in/dchunk/'+f,str(p))
            assert sha(p)==sha(inp/f), f
    elif a.stage in ('bench','bulk'):
        stopped(json.loads(cmd('app','list','--json')))
        assert not any(r['stage']==a.stage for r in history), 'One launch per job; audit explicit before retry'
        if a.stage=='bench':
            assert (TEMP/'input_audit.json').exists(), 'Audit all input hashes before spending'
            assert (TEMP/'audit_report.json').exists(), 'Require b660 document sample audit'
            ar = json.loads((TEMP/'audit_report.json').read_text())
            assert ar['sampled_documents']==2000 and ar['sha_errors']==0
            assert ar['manifest_sha256']==sha(inp/'manifest.json')
            assert (TEMP/'upload_audit.json').exists()
        rate = RATE
        if a.stage=='bulk':
            b = json.loads((TEMP/'bench_report.json').read_text())
            rate = b['hourly_upper']/3600
            full_prediction = b['predicted_bulk_usd']
            if a.scope=='expanded_pool':
                assert full_prediction > 4, 'Only use fallback if full corpus exceeds 4 USD'
                scope = json.loads((TEMP/'pool_scope.json').read_text())
                ar = json.loads((TEMP/'audit_report.json').read_text())
                assert scope['tokens']==ar['alternatives']['expanded_pool_tokens']
                assert scope['documents']==ar['alternatives']['expanded_pool_documents']==212855
                assert sha(TEMP/'pool_scope.npz')==scope['sha256']
                pred = prediction(scope['tokens'],b['tokens_per_second'],b['loaded_seconds'],b['hourly_upper'])
                assert pred['usd'] <= 3, 'Expanded pool prediction exceeds 3 USD: STOP'
                b = dict(b,full_predicted_bulk_usd=full_prediction,predicted_bulk_usd=pred['usd'],predicted_bulk_seconds=pred['seconds'],scope=a.scope)
                (TEMP/'selected_report.json').write_text(json.dumps(b,indent=2))
                cmd('volume','put','r2ai-runner',str(TEMP/'pool_scope.npz'),'/in/dchunk/pool_scope.npz')
                cmd('volume','put','r2ai-runner',str(TEMP/'pool_scope.json'),'/in/dchunk/pool_scope.json')
                cmd('volume','put','r2ai-runner',str(TEMP/'selected_report.json'),'/out/dchunk/selected_report.json')
            else:
                assert full_prediction <= 4
            assert sum(x.get('upper_usd',0) for x in history)+a.seconds*rate+.15 < 5
            assert a.seconds*rate <= 4
            assert a.seconds >= b['predicted_bulk_seconds'], 'Budget must include prediction overhead'
        baseline = json.loads(cmd('billing','summary','--json'))
        assert float(baseline['metered_cost'])>=0
        r = dict(stage=a.stage,scope=a.scope,started=time.time(),billing_before=baseline,seconds_budget=a.seconds)
        history.append(r)
        save()
        # During an image build no GPU is allocated; bound billed GPU residency
        # separately (90s timeout + 15s startup/cleanup per function).
        limit = 440 if a.stage=='bench' else a.seconds+100
        log = TEMP/(a.stage+'_modal.log')
        args = [MODAL,'run','deploy/modal_dchunk.py','--stage',a.stage]
        if a.stage=='bulk':
            args += ['--seconds-budget',str(a.seconds),'--gpu-kind',b['gpu_kind'],'--scope',a.scope]
        with log.open('w',encoding='utf-8') as f:
            p = subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT)
            try:
                r['exit'] = p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                r['timeout'] = True
                p.terminate()
                p.wait(timeout=20)
        # Stop only apps this experiment created, and account failed/cancelled wall too.
        apps = json.loads(cmd('app','list','--json'))
        r['apps_before_stop'] = apps
        stopped_ids = set()
        for app in apps:
            desc = str(app.get('description',app.get('Description',app.get('Name',''))))
            if 'r2ai-dchunk' in desc and str(app.get('state',app.get('State',''))).lower() != 'stopped':
                ident = app.get('App ID',app.get('App Id',app.get('app_id')))
                assert ident, app
                print(cmd('app','stop','--yes',ident),flush=True)
                stopped_ids.add(ident)
        import re
        # Ephemeral `run` can disappear from app list immediately after returning.
        # Still issue the explicit stop for the app ID written by our single launch.
        for ident in set(re.findall(r'\bap-[A-Za-z0-9]+',log.read_text(encoding='utf-8')))-stopped_ids:
            print(cmd('app','stop','--yes',ident),flush=True)
            stopped_ids.add(ident)
        r['stopped_app_ids'] = sorted(stopped_ids)
        r['elapsed_wall'] = time.time()-r['started']
        if a.stage=='bench':
            r['upper_usd'] = 105*(2.1726+4.7626)/3600 + .04
        else:
            r['upper_usd'] = r['elapsed_wall']*rate + .015
        r['billing_after'] = json.loads(cmd('billing','summary','--json'))
        r['snapshot_delta'] = float(r['billing_after']['metered_cost'])-float(baseline['metered_cost'])
        r['apps_after_stop'] = json.loads(cmd('app','list','--json'))
        save()
        print(json.dumps(r,indent=2),flush=True)
        assert r['upper_usd'] <= (.3 if a.stage=='bench' else 4.7)
        if a.stage=='bench':
            assert r['snapshot_delta'] <= .3, 'Bench billing snapshot exceeded 0.3 USD'
        assert sum(x['upper_usd'] for x in history) < 5
        assert sum(max(x.get('snapshot_delta',0),x['upper_usd']) for x in history) < 5
        assert r.get('exit') == 0 and not r.get('timeout'), log.read_text(encoding='utf-8')[-8000:]
        stopped(r['apps_after_stop'])
        fn = a.stage+'_report.json'
        if a.stage=='bench':
            cmd('volume','get','r2ai-runner','/out/dchunk/volume_audit.json',str(TEMP/'volume_audit.json'))
            va = json.loads((TEMP/'volume_audit.json').read_text())
            assert va['manifest_sha256']==sha(inp/'manifest.json')
            assert all(va['small_sha256'][f]==sha(inp/f) for f in ('bench.npz','queries.json'))
            reports = []
            for kind in ('L40S','H100'):
                name = 'bench_'+kind+'.json'
                cmd('volume','get','r2ai-runner','/out/dchunk/'+name,str(TEMP/name))
                reports.append(json.loads((TEMP/name).read_text()))
            assert all(x['manifest_sha256']==sha(inp/'manifest.json') for x in reports)
            b = min(reports,key=lambda x:x['predicted_bulk_usd'])
            (TEMP/fn).write_text(json.dumps(dict(b,comparisons=reports),indent=2))
            cmd('volume','put','r2ai-runner',str(TEMP/fn),'/out/dchunk/'+fn)
        else:
            cmd('volume','get','r2ai-runner','/out/dchunk/'+fn,str(TEMP/fn))
    elif a.stage=='fetch':
        for f in ('top1000.npz','bulk_report.json'):
            cmd('volume','get','r2ai-runner','/out/dchunk/'+f,str(TEMP/f))
            print(f,(TEMP/f).stat().st_size,sha(TEMP/f),flush=True)
    else:
        print(cmd('app','list','--json'))
        print(cmd('billing','summary','--json'))


if __name__=='__main__':
    main()
