"""Persisted vi45 CPU ping/bench control. No automatic retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')
STATE=TEMP/'state.json'
MODAL=__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'


def cmd(*args):
    p=subprocess.run([MODAL,*args],cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"),PYTHONIOENCODING='utf-8'),
        capture_output=True,text=True,encoding='utf-8',timeout=1200)
    if p.returncode:
        raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def save(state):
    STATE.write_text(json.dumps(state,indent=2),encoding='utf-8')


def record_handoff(state):
    p=ROOT/'docs/VI45_HANDOFF.md'
    text=p.read_text(encoding='utf-8') if p.exists() else '# vi45\n'
    marker='\n## Modal state\n'
    text=text.split(marker)[0]
    lines=[marker.rstrip(),'- State: `<HIST_TMP>/vi45/state.json`; không respawn call mơ hồ.']
    for phase,c in state.get('calls',{}).items():
        lines.append(f'- {phase}: {c["status"]}; call `{c.get("call_id","UNKNOWN")}`.')
    p.write_text(text.rstrip()+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['sample','deploy','redeploy','ping','ping2','bench','bench2','poll','stop'])
    ap.add_argument('--phase',default='bench')
    ap.add_argument('--wait',type=int,default=0)
    a=ap.parse_args()
    TEMP.mkdir(parents=True,exist_ok=True)
    if a.action=='sample':
        import random
        import zipfile
        with zipfile.ZipFile(TEMP.parent/'vi42/submission_vi42.zip') as z:
            sub=json.loads(z.read('submission.json'))
        pairs=[dict(qid=str(it['id']),doc_id=c['doc_id'],text=c['chunk_text']) for it in sub for c in it['relevant_chunks']]
        rows=random.Random(4545).sample(pairs,512)
        import pyarrow.parquet as pq
        qt=pq.read_table(ROOT/'data/query.parquet').to_pydict()
        qt={str(q):v for q,v in zip(qt['id'],qt['query'])}
        for row in rows:
            row['query']=qt[row['qid']]
        (TEMP/'bench.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
        print(len(rows))
        return
    import modal
    if a.action=='deploy':
        assert not STATE.exists()
        apps=json.loads(cmd('app','list','--json'))
        assert not [r for r in apps if r['description']=='r2ai-vi45' and r['state']!='stopped']
        state=dict(calls={},billing_before=json.loads(cmd('billing','summary','--json')))
        save(state)
        cmd('volume','put','r2ai-runner',str(TEMP/'bench.json'),'/in/vi45/bench.json')
        (TEMP/'deploy.log').write_text(cmd('deploy','deploy/modal_vi45.py'),encoding='utf-8')
        apps=json.loads(cmd('app','list','--json'))
        own=[r for r in apps if r['description']=='r2ai-vi45' and r['state']!='stopped']
        assert len(own)==1
        state['app_id']=own[0]['app_id']
        save(state)
        record_handoff(state)
        print(json.dumps(state))
        return
    state=json.loads(STATE.read_text())
    if a.action=='redeploy':
        assert state['calls']['bench']['status']=='failed'
        assert 'bench2' not in state['calls']
        (TEMP/'apps_before_redeploy.json').write_text(cmd('app','list','--json'))
        (TEMP/'redeploy.log').write_text(cmd('deploy','deploy/modal_vi45.py'),encoding='utf-8')
        print('redeployed fixed max_model_len1025/prompt1024')
        return
    if a.action in ('ping','ping2','bench','bench2'):
        assert a.action not in state['calls']
        if a.action=='bench':
            assert state['calls']['ping']['status']=='done'
        if a.action=='bench2':
            assert state['calls']['bench']['status']=='failed' and state['calls']['ping2']['status']=='done'
        (TEMP/('apps_before_'+a.action+'.json')).write_text(cmd('app','list','--json'))
        record=dict(status='spawn_intent',start=time.time())
        state['calls'][a.action]=record
        save(state)
        record_handoff(state)
        call=modal.Function.from_name('r2ai-vi45','ping' if a.action=='ping2' else a.action).spawn()
        record.update(status='running',call_id=call.object_id)
        save(state)
        record_handoff(state)
        print(json.dumps(record))
    elif a.action=='poll':
        record=state['calls'][a.phase]
        assert record.get('call_id')
        try:
            result=modal.FunctionCall.from_id(record['call_id']).get(timeout=a.wait)
        except (TimeoutError,modal.exception.TimeoutError):
            print('RUNNING '+record['call_id'])
            return
        except Exception as exc:
            record.update(status='failed',error=repr(exc))
            save(state)
            record_handoff(state)
            raise
        record.update(status='done',result=result,finish=time.time())
        save(state)
        record_handoff(state)
        if a.phase in ('bench','bench2'):
            for name in ('bench_report.json','bench_function_wall.json'):
                cmd('volume','get','--force','r2ai-runner','/out/vi45/'+name,str(TEMP/name))
            rows=json.loads((TEMP/'bench.json').read_text(encoding='utf-8'))
            assert len(rows)==len(result['scores'])
            import hashlib
            saved=[]
            for row,value in zip(rows,result['scores']):
                ident=hashlib.sha256((row['qid']+'\0'+str(row['doc_id'])+'\0'+row['text']).encode()).hexdigest()
                saved.append(dict(qid=row['qid'],doc_id=row['doc_id'],cid=ident,logit8c=value))
            (TEMP/'chunk8b_bench_logits.json').write_text(json.dumps(saved),encoding='utf-8')
        print(json.dumps(result))
    else:
        assert all(c['status'] in ('done','failed') for c in state['calls'].values())
        cmd('app','stop','--yes',state['app_id'])
        state['billing_after']=json.loads(cmd('billing','summary','--json'))
        state['apps_after']=json.loads(cmd('app','list','--json'))
        state['billing_delta']=float(state['billing_after']['metered_cost'])-float(state['billing_before']['metered_cost'])
        save(state)
        record_handoff(state)
        print(json.dumps(state))


if __name__=='__main__':
    main()
