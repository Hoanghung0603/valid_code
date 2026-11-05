"""SDPA recovery control: persisted spawn IDs, explicit terminal states, no auto retry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')
MODAL=TEMP.parent/'modalenv/Scripts/modal.exe'
APP='r2ai-vi51';VOL='r2ai-vi51-data';RATE=4.5232/3600


def cli(profile,*args):
    p=subprocess.run([str(MODAL),*args],cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'),
        capture_output=True,text=True,encoding='utf-8',timeout=1800)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def handoff():
    lines=['# vi51 — SDPA recovery; không nộp/browser/commit/crawl',
        '- Quyết định17:00: transformers4.57.6/torch2.9.1+SDPA fp16/H100; left/LAST/L2; gate min>=.999 mean>=.9999; nếu fail DỪNG.',
        '- Giữ title+clean2048/query instruct256; fullpoolvi41c/RRFk60×1/chunk+countvi48/bảo vệvi41c; KHÔNG prepare lại.',
        '- Revisions a97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 b1d8ad4ca9b3dd8059ad90a75d4983776a23d44af; release05/06/2025.',
        '- Input223065doc/609662pair/288258527token/29part; inputauditPASS; parta/b SHA giống nhau toàn29part.',
        '- App r2ai-vi51/Volume r2ai-vi51-data riêng; retries0/max4H10030c+max2H10030b; KHÔNG respawn call chưa terminal; không đụng vi50.',
        '- Prior vLLM đãstopped:425.060119GPU-s/upper<MODAL_PROFILE_A> .569092+30b .015=.584092USD. Trần tổng30USD; benchmới≤.5USD.',
        '- Localshared <HIST_TMP>/vi51; output <HIST_TMP>/vi51a,b/outputs; b660 D:/r2ai/data/sub/vi51.',
        '- Collect: modalenv/Scripts/python.exe deploy/vi51_sdpa_local.py poll --profile PROFILE --job JOB --wait 0; xong dùng collect cùng args.',
         '- Không launch lại; callid lưu state_sdpa_PROFILE.json trước poll. Profileenv+PYTHONIOENCODING=utf-8 bắt buộc.',
         '- 30c deploy dùng MODAL_IMAGE_BUILDER_VERSION=2025.06 (client chưa hỗ trợ default2026.10); không đổi workspace setting.']
    for p in sorted(TEMP.glob('state_sdpa_*.json')):
        s=load(p);lines.append(f"- {s['profile']} app={s.get('app_id')} upper={s.get('reserved_upper',0):.6f}USD stopped={s.get('stopped',False)}")
        for n,r in s['calls'].items():
            lines.append(f"- {n} {r.get('call_id','INTENT')} {r['status']} tier{r.get('tier','')} {r.get('parts',[])}")
            if r.get('mode')=='bench' and r.get('result'):
                b=r['result'];lines.append(f"- Bench{r['tier']}: {b.get('dtype')} {b['tokens_per_second']:.3f}tok/s cosineMin/mean/max {b['repeat_cosine_min']:.9f}/{b.get('repeat_cosine_mean',0):.9f}/{b['repeat_cosine_max']:.9f} GPUfunction{b['function_seconds']:.3f}s.")
        if s.get('forecast'):lines.append('- Forecast '+json.dumps(s['forecast'],ensure_ascii=False))
        if s.get('stop_reason'):lines.append('- STOP '+s['stop_reason'])
        live=[r for r in s['calls'].values() if r.get('mode')=='bulk']
        lines.append(f"- {s['profile']} committedDONE {sum(len(r['parts']) for r in live if r['status']=='done')}/{sum(len(r['parts']) for r in live)}part; GPUknown "+str(sum(r.get('result',{}).get('function_seconds',0) for r in s['calls'].values()))+'s.')
    lines.extend(['- Resume a: modalenv/Scripts/python.exe deploy/vi51_resume.py --tier a --build (chỉ poll/collect call có sẵn; không spawn).',
        '- Resume b: modalenv/Scripts/python.exe deploy/vi51_resume.py --tier b --build --stop; đangnềnPID40928(parent25612), resume_b.lock chặntrùng; build_intentcấmbuildlại.',
        '- Bench fp16 PASS cả2; sanity5query a PASS; forecast+30% a4.489276USD/15.2ph, b19.043928USD/44.4ph GPU song song(+queuea); tổngupper24.857125<30.',
        '- Audit prebuild: python deploy/vi51_prebuild.py a; final: python scripts/vi51_audit.py --out <HIST_TMP>/vi51a.',
        '- Mirror: python deploy/vi51_mirror_pack.py a → Temp/vi51/mirror_a ZIP_STORED<98MB/manifest SHA; SFTP vào b660 data/sub/vi51.',
        '- Server restore a: D:/r2ai/.venv/Scripts/python.exe D:/r2ai/vi51_unpack_mirror.py --root D:/r2ai/data/sub/vi51 --manifest D:/r2ai/data/sub/vi51/mirror_a_manifest.json (đãPASS).',
        '- 1089testPASS/2skip+9testvi51PASS; statecallIDđãrestore theo logkhôngrespawn, merge stateđãsửa; CPUimportPASS2workspace.',
        '- aZIP89.699.030B SHAe01a6c52559ef212595d5afbe107dc273fe7a75232dbe40944c17575f9b082b9 local=b660; validate/check0/fullclean+independent_auditPASS.',
        '- 17:32VN knownGPU3211.088136s/upper23.526431USD: prior30.569092+30b6.163936+30c16.793403; snapshotmetered30c9.065884/30b6.394498USD(chungvi50prior).',
        '- a29/29GPU DONE2481.905371s; cosineSHA/exactpart/querySHA4callPASS; build_intent1lần vi51_build_ready.py a, embeddingscollectorđộclập. KHÔNG build lại.',
        '- a collect/prebuildaudit29part223065doc609662pairPASS; embeddings/cosine6mirrorpartđãlocal+b660/artifacts, toànSHAserverMIRROR_VERIFIED0.',
        '- Hard stop a22:30VN31/10, mục tiêu21:30; b03:00VN01/11. Chỉ người phụ trách nộp bài; tác vụ này KHÔNG nộp.'])
    for tier in 'ab':
        p=TEMP.parent/f'vi51{tier}/build_report.json'
        if p.exists():
            r=load(p);lines.append(f"- ZIPvi51{tier}: {r['bytes']}B SHA{r['sha256']}; validate/check_zip0/fullclean/CRC/1200ids/chunk+countvi48/LZMA1entry PASS.")
        p=TEMP.parent/f'vi51{tier}/offline.json'
        if p.exists():
            r=load(p);lines.append(f"- Offline{tier}: đổi{r['changed_queries']}câu/vào{r['entering_per_query']:.6f}doc-câu; rhoDense{r['spearman_q3e_dense_pooled']:.9f}/Fhyde{r['spearman_q3e_fhyde_pooled']:.9f}; overlap30 {r['top30_dense_overlap_mean']:.6f}.")
    assert len(lines)<=60
    target=ROOT/'docs/VI51_HANDOFF.md';pending=target.with_suffix('.'+str(os.getpid())+'.tmp')
    pending.write_text('\n'.join(lines)+'\n',encoding='utf-8');pending.replace(target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','ping','launch','poll','collect','stop','upload'])
    ap.add_argument('--profile',required=True);ap.add_argument('--job',default='ping');ap.add_argument('--tier',choices=['a','b'],default='a')
    ap.add_argument('--mode',choices=['bench','bulk'],default='bench');ap.add_argument('--parts',nargs='*',default=[])
    ap.add_argument('--watchdog',type=int,default=180);ap.add_argument('--wait',type=int,default=0)
    a=ap.parse_args();os.environ['MODAL_PROFILE']=a.profile;os.environ['PYTHONIOENCODING']='utf-8'
    import modal
    sp=TEMP/f'state_sdpa_{a.profile}.json'
    s=load(sp) if sp.exists() else dict(profile=a.profile,calls={},reserved_upper=.02)
    initial_reserve=s['reserved_upper']
    def save():
        nonlocal initial_reserve
        if sp.exists():
            fresh=load(sp)
            for name,record in fresh['calls'].items():
                if name!=a.job or name not in s['calls']:
                    s['calls'][name]=record
            s['reserved_upper']=fresh['reserved_upper']+(s['reserved_upper']-initial_reserve)
        initial_reserve=s['reserved_upper']
        pending=sp.with_suffix('.'+str(os.getpid())+'.tmp')
        pending.write_text(json.dumps(s,indent=2),encoding='utf-8');pending.replace(sp);handoff()
    if a.action=='init':
        apps=json.loads(cli(a.profile,'app','list','--json'))
        own=[x for x in apps if x['description']==APP and x['state']!='stopped']
        assert len(own)==1
        s.update(app_id=own[0]['app_id'],stopped=False,billing_before=json.loads(cli(a.profile,'billing','summary','--json')));save()
    elif a.action=='upload':
        from hashlib import sha256
        prep=load(TEMP/'prepare_report.json')
        for n in a.parts:
            p=TEMP/'modal_input'/n;assert sha256(p.read_bytes()).hexdigest()==prep['files'][n]['sha256']
            cli(a.profile,'volume','put',VOL,str(p),f'/in/{n}');print('UPLOADED '+n,flush=True)
    elif a.action in ('ping','launch'):
        name=a.job
        assert name not in s['calls'],'Never repeat ambiguous or terminal job'
        apps=json.loads(cli(a.profile,'app','list','--json'))
        (TEMP/f'apps_sdpa_{a.profile}_{name}.json').write_text(json.dumps(apps),encoding='utf-8')
        assert any(x['app_id']==s['app_id'] and x['state']!='stopped' for x in apps)
        if a.action=='launch':
            assert s['calls'][s.get('ping_job','ping')]['status']=='done'
            # Calls can queue; the deployed work function enforces max_containers=4
            # jointly for a+b, so b may be spawned immediately after all a calls.
            assert sum(r['status'] in ('running','spawn_intent') and r.get('mode')!='cpu' for r in s['calls'].values())<8
            reserve=(a.watchdog+8)*RATE
            allstates=[load(p) for p in TEMP.glob('state_sdpa_*.json')]
            assert .584092+sum(x['reserved_upper'] for x in allstates)+reserve<=30
            if a.mode=='bench':
                used=sum((r['watchdog']+8)*RATE for x in allstates for n,r in x['calls'].items() if r.get('mode')=='bench' and n.startswith('fp16_'))
                assert used+reserve<=.5
            if a.profile==__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"):assert s['reserved_upper']+reserve<=1
            s['reserved_upper']+=reserve
        if a.action=='ping':s['ping_job']=name
        s['calls'][name]=dict(status='spawn_intent',tier=a.tier,mode='cpu' if a.action=='ping' else a.mode,parts=a.parts,watchdog=a.watchdog,start=time.time(),reserve=reserve if a.action=='launch' else 0);save()
        call=modal.Function.from_name(APP,'ping' if a.action=='ping' else 'work').spawn(*(() if a.action=='ping' else (a.tier,a.mode,name,a.parts,a.watchdog)))
        s['calls'][name].update(status='running',call_id=call.object_id);save();print(json.dumps(s['calls'][name]),flush=True)
    elif a.action=='poll':
        r=s['calls'][a.job];assert r['status']=='running'
        original_reserve=r.get('reserve',0)
        try:result=modal.FunctionCall.from_id(r['call_id']).get(timeout=a.wait)
        except (TimeoutError,modal.exception.TimeoutError):print('RUNNING '+r['call_id']);return
        except Exception as exc:
            # Transport/auth errors do NOT prove a remote call terminal.
            terminal=isinstance(exc,(subprocess.CalledProcessError,modal.exception.RemoteError,modal.exception.FunctionTimeoutError))
            r.update(error=repr(exc),last_poll=time.time())
            if terminal:r.update(status='failed_terminal',finish=time.time())
            save();raise
        r.update(status='done',result=result,finish=time.time())
        if r.get('reserve'):
            actual=(result['function_seconds']+8)*RATE
            s['reserved_upper']+=actual-original_reserve;r['reserve']=actual
        save();print(json.dumps(result),flush=True)
    elif a.action=='collect':
        r=s['calls'][a.job];assert r['status']=='done'
        folder=TEMP if r['mode']=='bench' else TEMP.parent/('vi51'+r['tier'])
        dest=folder/'outputs'/a.profile/a.job;dest.mkdir(parents=True,exist_ok=True)
        files=['bench_report.json','bench_embeddings.npz','function_wall.json'] if r['mode']=='bench' else list(r['result']['files'])+['bulk_report.json','function_wall.json']
        from hashlib import sha256
        for n in dict.fromkeys(files):
            target=dest/n;expected=r['result'].get('files',{}).get(n)
            def valid():
                return target.exists() and (not expected or (target.stat().st_size==expected['bytes'] and sha256(target.read_bytes()).hexdigest()==expected['sha256']))
            if valid():continue
            pending=dest/(n+'.download')
            cli(a.profile,'volume','get','--force',VOL,f"/out/{r['tier']}/{a.job}/{n}",str(pending))
            pending.replace(target);assert valid(),str(target)
        r['collected']=True;save();print('COLLECTED '+str(dest),flush=True)
    else:
        assert all(r['status'] in ('done','failed_terminal') for r in s['calls'].values())
        cli(a.profile,'app','stop','--yes',s['app_id']);s['stopped']=True
        s['billing_after']=json.loads(cli(a.profile,'billing','summary','--json'));save()


if __name__=='__main__':main()
