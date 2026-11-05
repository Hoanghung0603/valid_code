"""Resume-safe vi54 control. Persist spawn intent + call ID before polling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi54')
MODAL=OUT.parent/'modalenv/Scripts/modal.exe'
RATE=4.5232/3600


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def cli(profile,*args,parse=False):
    result=subprocess.run([str(MODAL),*args],cwd=ROOT,
        env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8',MODAL_IMAGE_BUILDER_VERSION='2025.06'),
        capture_output=True,text=True,encoding='utf-8',timeout=3600)
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    return json.loads(result.stdout) if parse else result.stdout


def handoff():
    lines=['# vi54 — tiếp quyết định20:35VN; không nộp/browser/commit',
        '- Nemotron nvidia/Nemotron-3-Embed-8B-BF16 pin d1f2f25730bbd775b99b29185134bc86653bf2d1; release16/07/2026;7.953B/OpenMDW1.1.',
        '- Raw riêng docs54 cùng29part/tập+thứtựvi51; nguồnindexvi36/iter_all_texts/BoilerplateFilter.clean/make_body unlimited;khôngdecodeQwen/khôngsửavi51.',
        '- query: /passage: ; Nemotron256/2048; bf16 SDPA/left/attention-mask mean/L2/4096;torch2.9.1/transformers5.2.0;khôngsửarope.',
        '- Configis_causal=False; truyềnis_causal=False/use_cache=False (Ministral5.2hardcodemoduleTrue),khôngđổicheckpoint/địnhnghĩaencoder.',
        '- vi41c+d8w+RRFk60×1 dense mean z-score theoquery q3e8B/Nemotron;missingnguồncònlại;chunk+countvi48/bảovệvi41c.',
        '- Trần25USD tổng/bench≤.6USD/≤4H100 mỗiworkspace;dựbáo+30%trướcbulk. Mụctiêu03:00VN01/11/hardstop06:00.',
        '- Weights functionCPU3600s→Volume r2ai-vi54-weights; GPUoffline;input/outputVolume r2ai-vi54-data. retries0.',
        '- b660 taskSYSTEMr2ai_vi54_prepare+owned heavy_job.lock;chờlockkhác/khôngxoálockngườikhác;chỉđọcshardđúngindex.',
        '- State <HIST_TMP>/vi54/state_PROFILE.json; python(modalenv) deploy/vi54_control.py poll/collect --profile PROFILE --job JOB;khôngrespawncallmơhồ.']
    prep=OUT/'prepare_report.json'
    if prep.exists():
        p=load(prep);lines.append(f"- Raw audit {p['documents']}doc/{p['pairs']}pair/{p['queries']}query/{len(p['parts'])}part;exactorder/nonempty/SHA/vi51unchangedPASS.")
    for path in sorted(OUT.glob('state_*.json')):
        s=load(path);lines.append(f"- {s['profile']} app={s.get('app_id')} stopped={s.get('stopped',False)} upper={s.get('reserved_upper',0):.6f}USD")
        for n,r in s['calls'].items():
            lines.append(f"- {s['profile']} {n} {r.get('call_id','INTENT')} {r['status']} {r.get('parts',[])}")
            if r.get('result',{}).get('tokens_per_second'):
                b=r['result'];lines.append(f"- Bench tok/s={b['tokens_per_second']:.3f}; cos min/mean/max={b['repeat_cosine_min']:.9f}/{b['repeat_cosine_mean']:.9f}/{b['repeat_cosine_max']:.9f}; function={b['function_seconds']:.3f}s.")
        if s.get('forecast'):lines.append('- Forecast '+json.dumps(s['forecast']))
        if s.get('stop_reason'):lines.append('- STOP '+s['stop_reason'])
    for n in ('offline.json','build_report.json'):
        if (OUT/n).exists():lines.append('- '+n+' '+json.dumps(load(OUT/n),ensure_ascii=False)[:1600])
    assert len(lines)<=60
    (ROOT/'docs/VI54_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','spawn','poll','collect','stop','upload','recover-ping'])
    ap.add_argument('--profile',required=True);ap.add_argument('--job',default='weights')
    ap.add_argument('--kind',choices=['weights','ping','bench','bulk'],default='weights')
    ap.add_argument('--parts',nargs='*',default=[]);ap.add_argument('--watchdog',type=int,default=430)
    ap.add_argument('--wait',type=int,default=0);a=ap.parse_args()
    os.environ.update(MODAL_PROFILE=a.profile,PYTHONIOENCODING='utf-8')
    import modal
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/f'state_{a.profile}.json'
    s=load(path) if path.exists() else dict(profile=a.profile,calls={},reserved_upper=.1)
    def save():
        path.write_text(json.dumps(s,indent=2),encoding='utf-8');handoff()
    if a.action=='init':
        apps=cli(a.profile,'app','list','--json',parse=True)
        own=[x for x in apps if x['description']=='r2ai-vi54' and x['state']!='stopped'];assert len(own)==1
        s.update(app_id=own[0]['app_id'],stopped=False,billing_before=cli(a.profile,'billing','summary','--json',parse=True));save()
    elif a.action=='upload':
        prep=load(OUT/'prepare_report.json')
        names=a.parts or list(prep['files'])+['prepare_report.json']
        from hashlib import sha256
        existing={r['filename'].removeprefix('in/'):r['size'] for r in cli(a.profile,'volume','ls','r2ai-vi54-data','/in','--json',parse=True)}
        for n in names:
            p=OUT/n if n=='prepare_report.json' else OUT/'modal_input'/n
            if n!='prepare_report.json':assert sha256(p.read_bytes()).hexdigest()==prep['files'][n]['sha256']
            if n in existing:
                assert existing[n]==p.stat().st_size,'remote input size mismatch: '+n
                print('EXISTS (remote SHA checked by CPU ping) '+n,flush=True);continue
            cli(a.profile,'volume','put','r2ai-vi54-data',str(p),'/in/'+n)
            print('UPLOADED '+n,flush=True)
        s['uploaded']=True;save()
    elif a.action=='spawn':
        assert a.job not in s['calls'],'Never respawn ambiguous/terminal job'
        apps=cli(a.profile,'app','list','--json',parse=True)
        assert any(x['app_id']==s['app_id'] and x['state']!='stopped' for x in apps)
        reserve=0
        if a.kind in ('bench','bulk'):
            assert s['calls']['ping']['status']=='done'
            reserve=(a.watchdog+8)*RATE
            states=[load(p) for p in OUT.glob('state_*.json')]
            assert sum(x['reserved_upper'] for x in states)+reserve<=25
            if a.kind=='bench':
                assert sum(r.get('reserve',0) for x in states for r in x['calls'].values() if r['kind']=='bench')+reserve<=.6
            else:
                assert (OUT/'forecast.json').exists() and load(OUT/'forecast.json')['upper_usd']<=25
                assert (OUT/'raw_audit_local.json').exists() and load(OUT/'raw_audit_local.json')['vi51_unchanged']
            s['reserved_upper']+=reserve
        r=dict(status='spawn_intent',kind=a.kind,parts=a.parts,watchdog=a.watchdog,reserve=reserve,start=time.time())
        s['calls'][a.job]=r;save()
        fn='download_weights' if a.kind=='weights' else 'ping' if a.kind=='ping' else 'work'
        call=modal.Function.from_name('r2ai-vi54',fn).spawn(*(() if a.kind in ('weights','ping') else (a.kind,a.job,a.parts,a.watchdog)))
        r.update(call_id=call.object_id,status='running');save();print(json.dumps(r),flush=True)
    elif a.action=='poll':
        r=s['calls'][a.job];assert r['status']=='running'
        try:result=modal.FunctionCall.from_id(r['call_id']).get(timeout=a.wait)
        except (TimeoutError,modal.exception.TimeoutError):print('RUNNING '+r['call_id']);return
        except Exception as exc:
            r['error']=repr(exc)
            if isinstance(exc,(subprocess.CalledProcessError,modal.exception.RemoteError,modal.exception.FunctionTimeoutError)):
                r['status']='failed_terminal'
            save();raise
        r.update(status='done',result=result,finish=time.time())
        if r['reserve']:
            actual=(result['function_seconds']+8)*RATE
            s['reserved_upper']+=actual-r['reserve'];r['reserve']=actual
        save();print(json.dumps(result),flush=True)
    elif a.action=='recover-ping':
        r=s['calls'][a.job]
        assert r['kind']=='ping' and 'DeserializationError' in r.get('error','')
        pending=OUT/f'ping_{a.profile}.json'
        cli(a.profile,'volume','get','--force','r2ai-vi54-data','/ping.json',str(pending))
        result=load(pending)
        assert result['revision']=='d1f2f25730bbd775b99b29185134bc86653bf2d1' and result['all_input_sha'] and result['documents']==223065
        r.update(status='done',result=result,finish=time.time(),recovered_from_committed_json=True);save()
    elif a.action=='collect':
        r=s['calls'][a.job];assert r['status']=='done'
        if r['kind'] not in ('bench','bulk'):return
        dest=OUT/'outputs'/a.profile/a.job;dest.mkdir(parents=True,exist_ok=True)
        files=['bench_report.json','bench_embeddings.npz','function_wall.json'] if r['kind']=='bench' else list(r['result']['files'])+['bulk_report.json','function_wall.json']
        from hashlib import sha256
        for n in dict.fromkeys(files):
            p=dest/n;meta=r['result'].get('files',{}).get(n)
            def valid():return p.exists() and (not meta or (p.stat().st_size==meta['bytes'] and sha256(p.read_bytes()).hexdigest()==meta['sha256']))
            if valid():continue
            pending=dest/(n+'.download');cli(a.profile,'volume','get','--force','r2ai-vi54-data',f'/out/{a.job}/{n}',str(pending));pending.replace(p);assert valid()
        r['collected']=True;save()
    else:
        assert all(r['status'] in ('done','failed_terminal') for r in s['calls'].values())
        cli(a.profile,'app','stop','--yes',s['app_id'])
        s.update(stopped=True,billing_after=cli(a.profile,'billing','summary','--json',parse=True),apps_after=cli(a.profile,'app','list','--json',parse=True));save()


if __name__=='__main__':main()
