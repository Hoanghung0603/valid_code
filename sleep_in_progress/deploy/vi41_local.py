"""Durable vi41 control: persisted intent/call ID, conservative inclusive reservations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi41')
MODAL = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe')
STATE = TEMP/'state.json'
APP = 'r2ai-vi41'


def cmd(*args):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
    result = subprocess.run([str(MODAL), *args], env=env, cwd=ROOT, capture_output=True,
                            text=True, encoding='utf-8', timeout=1200)
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result.stdout


def save(state):
    p = STATE.with_suffix('.tmp')
    p.write_text(json.dumps(state, indent=2), encoding='utf-8')
    p.replace(STATE)


def handoff(state, milestone):
    lines = ['# vi41 — pool mở rộng', '- Mốc: '+milestone,
        '- Không nộp/browser/commit; trần cứng Modal 10 USD, gồm mọi lỗi/startup.',
        '- App riêng r2ai-vi41; giữ r2ai-runner Volume và mọi dữ liệu vi34–vi40/dchunk.',
        '- State/call ID: '+STATE.as_posix(),
        '- K=20 doc mới/nguồn/câu (dense-doc, dchunk, BM25, near-dup); tối đa80/96000cặp.',
        '- Lý do K: 8B khoảng 96000/9=10667 GPU-s ~$5,8 L40S; dành >=$4 cho rewrite/search/bge/docwin/startup.',
        '- RRF giữ1/2/2/1/1 k60; n_q logit>-2 kẹp40–400; chunk U900/zc+4zd/bcov1; Δ<=55.',
        f'- App ID: {state.get("app_id")}; upper đã đặt chỗ={sum(c["reserve"] for c in state["calls"].values()):.6f} USD.',
        '- Baseline billing metered_cost='+str(state['billing_before']['metered_cost'])+'.']
    for phase, record in state['calls'].items():
        lines.append(f'- {phase}: {record["status"]}; call_id={record.get("call_id","UNKNOWN")}; reserve=${record["reserve"]:.6f}.')
    repair=TEMP/'rewrite_repair_report.json'
    if repair.exists():
        r=json.loads(repair.read_text(encoding='utf-8'))
        lines += ['- Rewrite XONG1200/4800truyvấncon/3600HyDE; seed4141/temp0; 2JSON dấu phẩy cuối sửa CPU, không sinh lại.',
                  '- Rewrite complete SHA='+r['sha256']+'; raw2lỗi giữ rewrite_errors.json.',
                  '- Rewrite GPU-function428,387s; snapshot sau stop$0,610001; upper đặt chỗ$1,946433.']
    lines += ['- Chỉ poll call ID đã lưu; spawn_intent mơ hồ KHÔNG respawn.',
              '- Chưa xong: retrieval dense-doc/dchunk/BM25/near-dup, bge/8B/docwin, build/validate/offline/ZIP.',
              '- WP2 docs/JUDGE_CALIB.md nếu chưa kết luận dùng được: ghi "chưa có cổng".',
              '- Zip đích <HIST_TMP>/vi41/submission_vi41.zip.']
    (ROOT/'docs/VI41_HANDOFF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    import modal
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['deploy','rewrite','poll','stop','search-deploy','search',
                                     'score-deploy','bench','rr','docwin','q8b'])
    ap.add_argument('--phase', default='rewrite')
    a = ap.parse_args()
    TEMP.mkdir(parents=True, exist_ok=True)
    if a.action == 'deploy':
        assert not STATE.exists(), 'Existing state; inspect, no blind redeploy'
        apps = json.loads(cmd('app','list','--json'))
        assert not [x for x in apps if x['description']==APP and x['state']!='stopped']
        state = dict(billing_before=json.loads(cmd('billing','summary','--json')), calls={})
        save(state)
        (TEMP/'deploy.log').write_text(cmd('deploy','deploy/modal_vi41.py'), encoding='utf-8')
        apps = json.loads(cmd('app','list','--json'))
        own = [x for x in apps if x['description']==APP and x['state']!='stopped']
        assert len(own) == 1, own
        state['app_id'] = own[0]['app_id']
        save(state)
        handoff(state,'deployed, chưa GPU')
        print(json.dumps(state))
        return
    state = json.loads(STATE.read_text())
    if a.action=='score-deploy':
        assert 'score_app_id' not in state
        apps=json.loads(cmd('app','list','--json'))
        assert not [x for x in apps if x['description']=='r2ai-vi41-score' and x['state']!='stopped']
        (TEMP/'score_deploy.log').write_text(cmd('deploy','deploy/modal_vi41_score.py'),encoding='utf-8')
        apps=json.loads(cmd('app','list','--json'))
        own=[x for x in apps if x['description']=='r2ai-vi41-score' and x['state']!='stopped']
        assert len(own)==1
        state['score_app_id']=own[0]['app_id']
        save(state)
        handoff(state,'score deployed, chưa spawn')
        print(state['score_app_id'])
        return
    if a.action in ('bench','rr','docwin','q8b'):
        phase=a.action
        assert phase not in state['calls'],'No automatic retries'
        apps=json.loads(cmd('app','list','--json'))
        (TEMP/('apps_before_'+phase+'.json')).write_text(json.dumps(apps,indent=2))
        snapshot=json.loads(cmd('billing','summary','--json'))
        spent=max(sum(r['reserve'] for r in state['calls'].values()),
                  float(snapshot['metered_cost'])-float(state['billing_before']['metered_cost']))
        plan=json.loads((TEMP/'score_budget.json').read_text())
        seconds=plan[phase]['seconds']
        reserve=(seconds+120)*2.3952/3600+.03
        remaining=sum(p['reserve'] for s,p in plan.items() if s not in state['calls'] and s!=phase)
        assert spent+reserve+remaining<10,(spent,reserve,remaining)
        if phase!='bench':
            assert state['calls']['bench']['status']=='done'
        record=dict(status='spawn_intent',started=time.time(),reserve=reserve,seconds=seconds)
        state['calls'][phase]=record
        save(state)
        handoff(state,phase+' intent')
        call=(modal.Function.from_name('r2ai-vi41-score','bench').spawn() if phase=='bench' else
              modal.Function.from_name('r2ai-vi41-score','score').spawn(phase,seconds))
        record.update(status='running',call_id=call.object_id)
        save(state)
        handoff(state,phase+' spawned ID persisted')
        print(json.dumps(record))
        return
    if a.action=='search-deploy':
        assert 'search_app_id' not in state
        apps=json.loads(cmd('app','list','--json'))
        assert not [x for x in apps if x['description']=='r2ai-vi41-search' and x['state']!='stopped']
        (TEMP/'search_deploy.log').write_text(cmd('deploy','deploy/modal_vi41_search.py'),encoding='utf-8')
        apps=json.loads(cmd('app','list','--json'))
        own=[x for x in apps if x['description']=='r2ai-vi41-search' and x['state']!='stopped']
        assert len(own)==1
        state['search_app_id']=own[0]['app_id']
        save(state)
        handoff(state,'rewrite repaired; search deployed chưa spawn')
        print(state['search_app_id'])
        return
    if a.action=='search':
        assert 'search' not in state['calls']
        apps=json.loads(cmd('app','list','--json'))
        (TEMP/'apps_before_search.json').write_text(json.dumps(apps,indent=2))
        reserve=750*4.7626/3600+.05
        assert sum(x['reserve'] for x in state['calls'].values())+reserve+6.5<10
        record=dict(status='spawn_intent',started=time.time(),reserve=reserve)
        state['calls']['search']=record
        save(state)
        handoff(state,'search spawn intent; không respawn mơ hồ')
        call=modal.Function.from_name('r2ai-vi41-search','search').spawn()
        record.update(status='running',call_id=call.object_id)
        save(state)
        handoff(state,'search spawned ID persisted')
        print(json.dumps(record))
        return
    if a.action == 'rewrite':
        assert 'rewrite' not in state['calls'], 'Never automatically respawn'
        apps = json.loads(cmd('app','list','--json'))
        (TEMP/'apps_before_rewrite.json').write_text(json.dumps(apps, indent=2))
        # 1100 hard timeout + 200 startup/scaledown/CPU and fixed contingency.
        reserve = 1300*(4.54+4*.0473+48*.008)/3600+.1
        assert sum(x['reserve'] for x in state['calls'].values())+reserve+7.5 < 10
        record = dict(status='spawn_intent', started=time.time(), reserve=reserve)
        state['calls']['rewrite'] = record
        save(state)
        handoff(state,'rewrite spawn intent; mơ hồ không chạy lại')
        call = modal.Function.from_name(APP,'rewrite').spawn()
        record.update(status='running',call_id=call.object_id)
        save(state)
        handoff(state,'rewrite spawned; ID đã lưu trước poll')
        print(json.dumps(record))
    elif a.action == 'poll':
        record = state['calls'][a.phase]
        assert record.get('call_id'), 'Ambiguous spawn; never retry'
        try:
            result = modal.FunctionCall.from_id(record['call_id']).get(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            print('RUNNING '+record['call_id'])
            return
        except Exception as exc:
            record.update(status='failed',error=repr(exc))
            save(state)
            handoff(state,'DỪNG: '+repr(exc))
            raise
        record.update(status='done',finished=time.time(),result=result)
        save(state)
        names={'search':['dchunk_variants.npz','dchunk_search_report.json'],
               'rewrite':['rewrites.jsonl','rewrite_errors.json','rewrite_report.json','rewrite_function_wall.json'],
               'bench':['bench_report.json'],'rr':['rr_new.json','rr_function_report.json'],
               'docwin':['docwin_new.json.zst','docwin_function_report.json'],
               'q8b':['q8b_new.json','q8b_function_report.json']}[a.phase]
        for name in names:
            cmd('volume','get','--force','r2ai-runner','/out/vi41/'+name,str(TEMP/name))
        handoff(state,a.phase+' xong')
        print(json.dumps(result))
    else:
        assert all(r['status'] in ('done','failed') for r in state['calls'].values())
        cmd('app','stop','--yes',state['app_id'])
        if state.get('search_app_id'):
            cmd('app','stop','--yes',state['search_app_id'])
        if state.get('score_app_id'):
            cmd('app','stop','--yes',state['score_app_id'])
        apps=json.loads(cmd('app','list','--json'))
        own=next(x for x in apps if x['app_id']==state['app_id'])
        assert own['state']=='stopped' and str(own['tasks'])=='0'
        state['billing_after']=json.loads(cmd('billing','summary','--json'))
        state['snapshot_delta']=float(state['billing_after']['metered_cost'])-float(state['billing_before']['metered_cost'])
        state['apps_after_stop']=apps
        save(state)
        handoff(state,'app stopped/tasks0, giữ Volume')
        print(json.dumps(state))


if __name__=='__main__':
    main()
