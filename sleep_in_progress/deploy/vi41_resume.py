"""Explicit authorized packaging recovery; persistent calls and fail-fast log monitor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from vi41_local import ROOT, TEMP, STATE, MODAL, cmd, save


def note(state, text):
    path=ROOT/'docs/VI41_HANDOFF.md'
    lines=path.read_text(encoding='utf-8').splitlines()
    lines[0]='# vi41 — tiếp tục packaging recovery (đã duyệt 30/10 20:45 VN)'
    lines += ['- Recovery: '+text]
    while len(lines)>80:
        # Completed recovery events also live durably in state; keep most recent events,
        # never truncate constraints or handoff content authored above.
        drop=next((i for i,line in enumerate(lines) if line.startswith('- Recovery:')),None)
        assert drop is not None,'Handoff requires manual consolidation; do not drop constraints'
        lines.pop(drop)
    assert len(lines)<=80
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def monitor(app_id, call_id, stop_event, logfile, started):
    env=dict(os.environ,PYTHONIOENCODING='utf-8',MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
    # Container import failures can precede call association; don't hide them with a call filter.
    process=subprocess.Popen([str(MODAL),'app','logs',app_id,'--follow',
        '--show-container-id','--show-function-id'],
        cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    def reader():
        errors=0
        with logfile.open('w',encoding='utf-8') as out:
            for line in process.stdout:
                out.write(line);out.flush()
                if any(x in line for x in ('ModuleNotFoundError:', 'ImportError:', 'VI41 INIT GUARD:')):
                    errors+=1
                    if errors>=2:
                        cmd('app','stop','--yes',app_id)
                        stop_event.set()
                        return
        code=process.wait()
        if code:
            stop_event.set()
    thread=threading.Thread(target=reader,daemon=True)
    thread.start()
    return process


def main():
    import modal
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['deploy','ping','search','score','poll','recover-ping'])
    ap.add_argument('--service',choices=['search','score'],default='search')
    ap.add_argument('--phase',default='search_v2')
    ap.add_argument('--attempt',type=int,default=2)
    ap.add_argument('--stage',choices=['bench','rr','docwin','q8b'])
    ap.add_argument('--seconds',type=int,default=300)
    ap.add_argument('--label',default='')
    a=ap.parse_args()
    state=json.loads(STATE.read_text())
    service='r2ai-vi41-'+a.service
    suffix='_v'+str(a.attempt)
    key=a.service+suffix+'_app_id'
    if a.action=='recover-ping':
        # CPU ran successfully: TorchVersion subclass alone caused LOCAL unpickling failure.
        phase=a.service+'_ping_v2'
        record=state['calls'][phase]
        assert 'DeserializationError' in record['error']
        target=TEMP/(a.service+'_ping.json')
        cmd('volume','get','--force','r2ai-runner','/out/vi41/'+target.name,str(target))
        result=json.loads(target.read_text())
        record.update(status='done',result=result,recovered_from_volume=True)
        save(state)
        note(state,phase+' imports REMOTE PASS; recovered JSON '+json.dumps(result)+'; app stopped; no GPU')
        print(json.dumps(result));return
    if a.action=='deploy':
        assert key not in state,'No blind redeploy'
        apps=json.loads(cmd('app','list','--json'))
        assert not [x for x in apps if x['description']==service and x['state']!='stopped']
        (TEMP/(a.service+'_v2_deploy.log')).write_text(cmd('deploy','deploy/modal_vi41_'+a.service+'.py'),encoding='utf-8')
        own=[x for x in json.loads(cmd('app','list','--json')) if x['description']==service and x['state']!='stopped']
        assert len(own)==1,own
        state[key]=own[0]['app_id'];save(state)
        note(state,service+' deployed self-contained '+state[key]+'; chưa GPU')
        print(state[key]);return
    if a.action in ('ping','search','score'):
        phase=(a.stage+suffix if a.action=='score' else
               a.service+'_ping'+suffix+a.label if a.action=='ping' else 'search'+suffix)
        assert phase not in state['calls'],'No automatic respawn'
        if a.action=='search':
            assert a.service=='search' and state['calls']['search_ping'+suffix]['status']=='done'
        if a.action=='score':
            assert a.service=='score' and state['calls']['score_ping'+suffix]['status']=='done'
            assert a.stage and 60<=a.seconds<=10500
            pings=[r['finished'] for k,r in state['calls'].items()
                   if k.startswith('score_ping'+suffix) and r['status']=='done' and r.get('finished')]
            previous=[r['started'] for k,r in state['calls'].items()
                      if k.startswith(('bench','rr','docwin','q8b'))]
            assert not [r for r in state['calls'].values() if r['status'] in ('running','spawn_intent')], 'Another active/ambiguous call'
            assert pings and max(pings)>max(previous,default=0),'Fresh CPU import ping before EACH GPU launch'
            assert time.time()-max(pings)<300,'CPU ping stale'
            if a.stage!='bench':
                assert state['calls']['bench'+suffix]['status']=='done'
        apps=json.loads(cmd('app','list','--json'))
        (TEMP/('apps_before_'+phase+'.json')).write_text(json.dumps(apps,indent=2))
        snapshot=json.loads(cmd('billing','summary','--json'))
        upper=max(sum(r['reserve'] for r in state['calls'].values()),
                  float(snapshot['metered_cost'])-float(state['billing_before']['metered_cost']))
        reserve=(.04 if a.action=='ping' else (a.seconds+120)*2.3952/3600+.03
                 if a.action=='score' else 750*4.7626/3600+.05)
        remaining=0
        if a.action=='score' and a.stage!='bench':
            plan=json.loads((TEMP/'score_budget.json').read_text())
            assert plan[a.stage]['seconds']<=10500,'Function hard timeout budget'
            assert a.seconds==plan[a.stage]['seconds']
            if a.stage=='q8b':
                q=json.loads((TEMP/'qselect_report.json').read_text())['pairs']
                assert a.seconds>=int(q/9*1.3)+90,'Exact 8B selection not covered by forecast'
            remaining=sum(v['reserve'] for k,v in plan.items() if k!=a.stage and k+suffix not in state['calls'])
        # Three mandatory future CPU pings are part of the total, not free overhead.
        assert upper+reserve+remaining+.12<10,(upper,reserve,remaining)
        record=dict(status='spawn_intent',started=time.time(),reserve=reserve,app_id=state[key],seconds=a.seconds)
        state['calls'][phase]=record;save(state)
        note(state,phase+' intent; reserve='+str(reserve)+'; KHÔNG respawn mơ hồ')
        call=(modal.Function.from_name(service,'bench').spawn() if a.action=='score' and a.stage=='bench' else
              modal.Function.from_name(service,'score').spawn(a.stage,a.seconds) if a.action=='score' else
              modal.Function.from_name(service,a.action).spawn())
        record.update(status='running',call_id=call.object_id);save(state)
        note(state,phase+' spawned '+call.object_id+'; ID đã lưu trước poll')
        print(json.dumps(record));return
    record=state['calls'][a.phase]
    assert record['status']=='running'
    event=threading.Event()
    proc=monitor(record['app_id'],record['call_id'],event,TEMP/(a.phase+'_server.log'),record['started'])
    try:
        deadline=record['started']+(240 if 'ping' in a.phase else
                                  record.get('seconds',600)+180 if a.phase.startswith(('bench','rr','docwin','q8b')) else 780)
        while True:
            assert proc.poll() is None,'Log monitor exited; stop app rather than run unmonitored'
            assert not event.is_set(),'Repeated initialization failure; app stopped'
            assert time.time()<deadline,'Client watchdog expired'
            try:
                result=modal.FunctionCall.from_id(record['call_id']).get(timeout=15)
                break
            except (TimeoutError,modal.exception.TimeoutError):
                continue
        assert not event.is_set(),'Initialization guard triggered while result arrived'
        record.update(status='done',finished=time.time(),result=result)
        state['total_upper_reserved_usd']=sum(r['reserve'] for r in state['calls'].values())
        state['stopped_reason']=None
        save(state)
        note(state,a.phase+' DONE '+json.dumps(result,ensure_ascii=False))
        print(json.dumps(result))
    except Exception as exc:
        cmd('app','stop','--yes',record['app_id'])
        record.update(status='failed_stopped',error=repr(exc));save(state)
        note(state,a.phase+' DỪNG/app stopped: '+repr(exc))
        raise
    finally:
        proc.terminate()


if __name__=='__main__':
    main()
