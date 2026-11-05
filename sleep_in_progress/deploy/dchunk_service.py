"""Durable deployed Function.spawn launcher; no local entrypoint or automatic respawn."""
import argparse
import json
import time
from pathlib import Path

from dchunk_local import TEMP, cmd, poll_cmd, sha, resume_forecast

APP = 'r2ai-dchunk-svc'
ROOT = Path(__file__).resolve().parents[1]
STATE = TEMP/'service_state.json'
RATE = 4.7626/3600


def save(obj):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temporary.replace(STATE)


def handoff(state, milestone):
    lines = ['# Dense-chunk 512/50 — deployed service', f'- Mốc: {milestone}. Không nộp/browser/commit.',
             f'- App {APP}; app_id={state.get("app_id")}; Volume r2ai-runner giữ nguyên.',
             f'- State/call IDs/chứng cứ: {STATE.as_posix()}',
             '- Chỉ poll call_id đã lưu; mạng lỗi KHÔNG spawn lại; offline >20 phút dừng.',
             f'- Prior upper ${state.get("prior_upper_usd",0):.9f}; tổng upper dự trữ ${upper(state):.9f} <5.']
    for name, call in state.get('calls',{}).items():
        lines.append(f'- {name}: call_id={call.get("call_id","UNKNOWN")}, status={call["status"]}, upper=${call["upper_usd"]:.6f}.')
    p = TEMP/'resume_plan.json'
    if p.exists() and 'audit' in state.get('calls',{}) and state['calls']['audit']['status']=='done':
        plan=json.loads(p.read_text())
        lines.append(f'- Audit: good={len(plan["good"])}, bad={len(plan["bad"])}, missing={len(plan["missing"])}; shape/dtype/finite/norm/SHA PASS cho good.')
        lines.append(f'- Run_parts: {", ".join(plan["run_parts"])}')
    for name in ('embed_report.json','bulk_report.json','build_report.json'):
        p=TEMP/name
        if p.exists():
            lines.append(f'- {name}: {p.as_posix()} SHA={sha(p)}')
    p=TEMP/'embed_report.json'
    if p.exists():
        report=json.loads(p.read_text())
        lines.append(f'- Embed16part0043–0058: GPU function thực {report["total_seconds"]:.6f}s (gồm startup/audit; không phải active kernel); đủ {report["chunks"]} chunk/{report["documents"]} doc.')
    if p.exists() and 'audit-final' in state.get('calls',{}) and state['calls']['audit-final']['status']=='done':
        plan=json.loads((TEMP/'resume_plan.json').read_text())
        new=[x for x in plan['good'] if int(Path(x['name']).stem.split('_')[1])>=43]
        lines.append(f'- Audit16 mới: {len(new)} part/{sum(x["chunks"] for x in new)} chunk; norm [{min(x["norm_min"] for x in new):.9f},{max(x["norm_max"] for x in new):.9f}], float16/finite/shape/ID/SHA PASS; CPU toàn59={plan["elapsed_seconds"]:.3f}s.')
    if 'snapshot_pre_search_usd' in state:
        lines.append(f'- Billing snapshot trước search luỹ kế ${state["snapshot_pre_search_usd"]:.9f}; dự trữ search $0.486572, upper tổng <5.')
    if 'total_snapshot_usd' in state:
        lines.append(f'- Billing cuối: snapshot luỹ kế ${state["total_snapshot_usd"]:.9f}; upper ${state["total_upper_usd"]:.9f}; snapshot có thể chưa settled.')
    p=TEMP/'build_report.json'
    if p.exists():
        report=json.loads(p.read_text())
        lines.append(f'- Build: {report["zip_bytes"]} byte SHA={report["sha256"]}; validate/check_zip/assert_clean/verbatim PASS; 1200id/1entry LZMA.')
        lines.append(f'- Offline TB trùng top30: dense={report["top30_overlap_dense"]:.6f}, fhydevi37={report["top30_overlap_fhyde_vi37"]:.6f}; ngoài pool top200={report["top200_outside_vi37_pool_total"]} (TB={report["top200_outside_vi37_pool_mean"]:.6f}).')
    lines.append('- Không đụng crawl/vi34–39; không nộp/browser/commit; chưa có điểm BTC/P@30.')
    (ROOT/'docs/DCHUNK_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def upper(state):
    return state['prior_upper_usd']+sum(c['upper_usd'] for c in state.get('calls',{}).values())


def durable_spawn(state, phase, function, args, cost, modal):
    assert phase not in state['calls'], 'No automatic relaunch, including ambiguous spawn'
    assert upper(state)+cost < 5
    # Persist intent BEFORE network call. Ambiguous response must never be retried.
    record=dict(status='spawn_intent',started=time.time(),upper_usd=cost)
    state['calls'][phase]=record
    save(state)
    call=modal.Function.from_name(APP,function).spawn(*args)
    record.update(call_id=call.object_id,status='running')
    save(state)
    handoff(state,phase+' spawned')
    return record


def poll(state, phase, modal, wait=False):
    record=state['calls'][phase]
    assert record.get('call_id'), 'Ambiguous spawn: inspect server, NEVER respawn'
    if record['status']=='done' and record.get('downloaded'):
        return
    offline=None
    while True:
        try:
            result=modal.FunctionCall.from_id(record['call_id']).get(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            offline=None
            if not wait:
                print('RUNNING',record['call_id'],flush=True)
                return
            time.sleep(20)
            continue
        except Exception as exc:
            text=str(exc)
            if 'billing' in text.lower() or not any(x in text.lower() for x in ('connect','timeout','getaddrinfo','resolve','dns','unavailable')):
                record.update(status='failed',error=repr(exc))
                save(state)
                handoff(state,phase+' FAILED')
                raise
            offline=offline or time.monotonic()
            if time.monotonic()-offline>=1200:
                handoff(state,phase+' OFFLINE >20m; NO respawn')
                raise TimeoutError('Offline >20 minutes; call may still run; NEVER respawn') from exc
            time.sleep(30)
            continue
        record.update(status='done',finished=time.time(),result=result)
        save(state)
        names={'audit':['resume_plan.json','partial_samples.npz'],
               'audit-final':['resume_plan.json','partial_samples.npz'],
               'embed':['embed_report.json','embedding_complete.json','partial_cosine_report.json'],
               'search':['bulk_report.json','top1000.npz']}[phase]
        for name in names:
            poll_cmd('volume','get','--force','r2ai-runner','/out/dchunk/'+name,str(TEMP/name))
        if phase=='search':
            assert sha(TEMP/'top1000.npz')==result['top1000_sha256']
        record['downloaded']=True
        save(state)
        handoff(state,phase+' DONE')
        print(json.dumps(dict(phase=phase,call_id=record['call_id'],result=result),indent=2),flush=True)
        return


def main():
    import modal
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['deploy','audit','audit-final','embed','search','poll','wait','stop'])
    ap.add_argument('--phase',choices=['audit','audit-final','embed','search'])
    a=ap.parse_args()
    if a.action=='deploy':
        assert not STATE.exists(), 'Existing service state: inspect, never redeploy blindly'
        baseline=json.loads(cmd('billing','summary','--json'))
        history=json.loads((TEMP/'ledger.json').read_text())
        prior=max(sum(max(r.get('snapshot_delta',0),r['upper_usd']) for r in history),
                  float(baseline['metered_cost'])-float(history[0]['billing_before']['metered_cost']))
        assert prior+.05+(930+330)*RATE+.1<5
        state=dict(app=APP,prior_upper_usd=prior,billing_before=baseline,calls={})
        save(state)
        log=cmd('deploy','deploy/modal_dchunk.py')
        (TEMP/'service_deploy.log').write_text(log,encoding='utf-8')
        apps=json.loads(cmd('app','list','--json'))
        matches=[x for x in apps if x['description']==APP and x['state']!='stopped']
        assert len(matches)==1,matches
        state['app_id']=matches[0]['app_id']
        save(state)
        handoff(state,'deployed; CPU audit next')
        print(json.dumps(state,indent=2))
        return
    state=json.loads(STATE.read_text())
    if a.action in ('poll','wait'):
        return poll(state,a.phase,modal,a.action=='wait')
    if a.action in ('audit','audit-final'):
        if a.action=='audit-final':
            assert state['calls']['embed']['status']=='done'
        durable_spawn(state,a.action,'audit_partial',(),.05,modal)
    elif a.action in ('embed','search'):
        assert state['calls']['audit']['status']=='done'
        plan=json.loads((TEMP/'resume_plan.json').read_text())
        assert plan['manifest_sha256']==sha(TEMP/'input/manifest.json')
        if a.action=='embed':
            seconds=800
            forecast=resume_forecast(json.loads((TEMP/'input/manifest.json').read_text()),plan,
                                     json.loads((TEMP/'bench_report.json').read_text()),upper(state),seconds)
            assert upper(state)+(930+330)*RATE+.1<5
            budget=dict(forecast,phase='embed',seconds_budget=seconds,plan_sha256=sha(TEMP/'resume_plan.json'))
            function,args,cost='bulk_h100',(seconds,'full',True,'embed'),930*RATE+.05
            name='resume_budget.json'
        else:
            assert state['calls']['embed']['status']=='done'
            assert state['calls']['audit-final']['status']=='done'
            assert len(plan['good'])==59 and not plan['bad'] and not plan['missing']
            snapshot=json.loads(cmd('billing','summary','--json'))
            baseline=json.loads((TEMP/'ledger.json').read_text())[0]['billing_before']
            state['billing_pre_search']=snapshot
            state['snapshot_pre_search_usd']=float(snapshot['metered_cost'])-float(baseline['metered_cost'])
            assert max(upper(state),state['snapshot_pre_search_usd'])+330*RATE+.05<5
            save(state)
            seconds=260
            budget=dict(prior_upper_usd=upper(state),predicted_remaining_usd=330*RATE+.05,
                        phase='search',seconds_budget=seconds,plan_sha256=sha(TEMP/'resume_plan.json'))
            function,args,cost='search_h100',(seconds,),330*RATE+.05
            name='search_budget.json'
        (TEMP/name).write_text(json.dumps(budget,indent=2),encoding='utf-8')
        cmd('volume','put','--force','r2ai-runner',str(TEMP/name),'/out/dchunk/'+name)
        durable_spawn(state,a.action,function,args,cost,modal)
    else:
        assert all(c['status'] in ('done','failed') for c in state['calls'].values()), 'Never stop an unknown/running call'
        cmd('app','stop','--yes',state['app_id'])
        state['apps_after_stop']=json.loads(cmd('app','list','--json'))
        app=next(x for x in state['apps_after_stop'] if x['app_id']==state['app_id'])
        assert app['state']=='stopped' and str(app['tasks'])=='0'
        state['billing_after']=json.loads(cmd('billing','summary','--json'))
        baseline=json.loads((TEMP/'ledger.json').read_text())[0]['billing_before']
        state['total_snapshot_usd']=float(state['billing_after']['metered_cost'])-float(baseline['metered_cost'])
        state['total_upper_usd']=max(upper(state),state['total_snapshot_usd'])
        assert state['total_upper_usd']<5
        save(state)
        ledger=TEMP/'ledger.json'
        history=json.loads(ledger.read_text())
        if not any(r.get('stage')=='deployed-service' and r.get('app_id')==state['app_id'] for r in history):
            history.append(dict(stage='deployed-service',app_id=state['app_id'],calls=state['calls'],
                                billing_before=state['billing_before'],billing_after=state['billing_after'],
                                snapshot_delta=float(state['billing_after']['metered_cost'])-float(state['billing_before']['metered_cost']),
                                upper_usd=sum(c['upper_usd'] for c in state['calls'].values()),
                                apps_after_stop=state['apps_after_stop']))
            ledger.write_text(json.dumps(history,indent=2),encoding='utf-8')
        handoff(state,'app STOPPED/tasks0; giữ Volume')
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':
    main()
