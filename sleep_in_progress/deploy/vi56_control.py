"""vi56 durable Modal control, no respawn on ambiguous state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi56 import read,write,fingerprint,RATE
OUT=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi56')
MODAL=OUT.parent/'modalenv/Scripts/modal.exe'
PROFILE=__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")
os.environ.update(PYTHONIOENCODING='utf-8',MODAL_PROFILE=PROFILE,MODAL_IMAGE_BUILDER_VERSION='2025.06')


def cli(*args):
    p=subprocess.run([str(MODAL),*args],cwd=ROOT,env=os.environ,capture_output=True,text=True,encoding='utf-8',timeout=3600)
    if p.returncode: raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def handoff(state):
    lines=['# vi56 — private only; KHÔNG nộp/browser/commit/crawl',
      '- Top100 q3e8B ngoài pool từng câu;raw32150cặp/25757doc;loại61bad lịch sử+1bodyrỗng→32088cặp;giữchunks+countvi48.',
      '- Bge doc×1/8Bdoc×2/densebge-m3×2/HyDEdense×1/bgeDocwin×1/d8w×1/q3e8B×1;k60/stable ties.',
      '- Dense/HyDE cosine lịch sử ngoài top200 thiếu→đúngvi41.append_last_ranks;giữordercũ/khôngbịa cosine.',
      '- Mới: bge512fp16;win300/250max512;8Bdoc1024;d8wmaxmọiU900/prompt+truncationvi45;khôngtop8.',
      '- q3evi55: ngoàiunion1024/unioncũ2048/title+clean/fp16/LAST/L2;khôngembedlại.',
      '- Trần10USD/30d;forecasttrướcbulk;resumeexactID+SHA/commitpart/≤1preemptionrerun.',
      '- ZIPbuild1lần/chỉsau sanity tắt docmới=vi531200/1200;≤99000000B/reorder.',
      f"- App={state.get('app_id')};stopped={state.get('stopped',False)};upper={state.get('upper',0):.6f}USD;baseline={state.get('baseline',{}).get('metered_cost')}"]
    for name,r in state['calls'].items(): lines.append(f"- {name}: {r.get('call_id','INTENT')} {r['status']} reserve={r.get('reserve',0):.6f}USD")
    for name in ('prepare_report.json','sanity.json','build_report.json'):
        if (OUT/name).exists():
            r=read(OUT/name)
            if name=='prepare_report.json':lines.append('- Forecast '+json.dumps(r['forecast']))
            elif name=='sanity.json':lines.append(f"- Sanity={r['exact']}/1200;changed_sets={r['changed_sets']}")
            else:lines.append(f"- ZIP{r['bytes']}B sha256={r['sha256']};validate/checkzip0/chunk+count1200/1200.")
    lines+=['- Local<HIST_TMP>/vi56;mirrorD:/r2ai/data/sub/vi56;deadline04:00VN/hardstop08:00VN01/11.',
      '- Chưa kiểm:gold/F2,billingsettled. Khôngđụngvi34–55/dchunk/crawl.']
    lines[-2]='- Local<HIST_TMP>/vi56;mirrorD:/r2ai/data/sub/vi56;zip~05:30VN/hardstop08:00VN01/11.'
    lines += ['- Traceback gốc đầy đủ: <HIST_TMP>/vi56/original_app_logs.txt (2 call, đã terminal).',
      '```text', 'Traceback (most recent call last):',
      '  File "/root/scripts/vi56.py", line 362, in <module>',
      "    if __name__=='__main__': main()", '                             ^^^^^^',
      '  File "/root/scripts/vi56.py", line 349, in main', '    import modal',
      "ModuleNotFoundError: No module named 'modal'", '```',
      '- Gốc: plain Python subprocess không có SDK runtime; sửa gọi gpu trực tiếp với data.commit; CLI SDK-free.',
      '- Bench phục hồi chỉ 1 lần/kind: bench_qwen_fix/bench_bge_fix; cùng lỗi nữa DỪNG, không lần3.']
    if sdiag:=state.get('diagnosis'): lines.append('- CPU regression: '+sdiag)
    if gate:=state.get('bulk_gate'): lines.append('- Bench forecast mới +30%/startup/đã dùng: '+json.dumps(gate))
    if costs:=state.get('cost_report'): lines.append('- Chi phí cuối '+json.dumps(costs))
    if (OUT/'offline.json').exists():
        ob=read(OUT/'offline.json')
        lines.append('- Offline: new/câu '+json.dumps(ob['new_docs_per_query'])+'; zero='+str(ob['zero_new_queries'])+'; vs53='+json.dumps(ob['vs_vi53']))
        lines.append('- Hạng doc mới '+json.dumps(ob['new_final_ranks'])+'; z từng nguồn trong offline.json (không đổi trọng số).')
        lines.append('- z dense/HyDE của doc mới gồm điểm append_last_ranks fallback, KHÔNG phải cosine đo mới; missing_old có ghi trong offline.json.')
    if mirror:=state.get('mirror'): lines.append('- Mirror b660 '+json.dumps(mirror))
    if cleanup:=state.get('b660_cleanup'): lines.append('- b660 cleanup '+json.dumps(cleanup))
    assert len(lines)<=60
    (ROOT/'docs/VI56_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','spawn','poll','upload','collect','stop'])
    ap.add_argument('--job',default='weights');ap.add_argument('--function',default='download_weights')
    ap.add_argument('--parts',nargs='*',default=[]);ap.add_argument('--watchdog',type=int,default=240)
    ap.add_argument('--bench',action='store_true');ap.add_argument('--wait',type=int,default=0)
    a=ap.parse_args();import modal
    sp=OUT/'state.json';s=read(sp) if sp.exists() else dict(profile=PROFILE,calls={},upper=.1)
    def save():write(sp,s);handoff(s)
    if a.action=='init':
        apps=json.loads(cli('app','list','--json'));write(OUT/'apps_init.json',apps)
        own=[x for x in apps if x['description']=='r2ai-vi56' and x['state']!='stopped'];assert len(own)==1
        s.update(app_id=own[0]['app_id'],baseline=json.loads(cli('billing','summary','--json')),stopped=False);save()
    elif a.action=='upload':
        uploaded=read(OUT/'uploaded.json') if (OUT/'uploaded.json').exists() else {}
        prep=read(OUT/'prepare_report.json')
        files=[(OUT/'prepare_report.json','/prepare_report.json')]+[(OUT/'modal_input'/n,'/in/'+n) for n in prep['parts']]
        for p,dst in files:
            fp=fingerprint(p)
            if uploaded.get(dst)==fp:continue
            cli('volume','put','--force','r2ai-vi56-data',str(p),dst)
            uploaded[dst]=fp;write(OUT/'uploaded.json',uploaded);print('UPLOADED '+p.name,flush=True)
    elif a.action=='spawn':
        assert a.job not in s['calls'],'Never respawn ambiguous or terminal call'
        apps=json.loads(cli('app','list','--json'));write(OUT/f'apps_{a.job}.json',apps)
        assert any(x['app_id']==s['app_id'] and x['state']!='stopped' for x in apps)
        gpu=a.function in ('qwen','bge');reserve=(a.watchdog+10)*RATE if gpu else .025
        assert s['upper']+reserve<=10,('STOP over cap',s['upper'],reserve)
        if gpu and not a.bench:
            assert read(OUT/'prepare_report.json')['forecast']['usd_upper']<=10
            assert s.get('bulk_gate',{}).get('usd_upper_total',float('inf'))<=10
            assert all(s['calls'][n]['status']=='done' for n in ('bench_qwen_fix','bench_bge_fix'))
        r=dict(status='spawn_intent',function=a.function,parts=a.parts,watchdog=a.watchdog,bench=a.bench,reserve=reserve,start=time.time())
        s['calls'][a.job]=r;s['upper']+=reserve;save()
        args=(a.job,a.parts,a.watchdog,a.bench) if gpu else ()
        call=modal.Function.from_name('r2ai-vi56',a.function).spawn(*args)
        r.update(call_id=call.object_id,status='running');save();print(json.dumps(r),flush=True)
    elif a.action=='poll':
        r=s['calls'][a.job];assert r['status']=='running'
        try:result=modal.FunctionCall.from_id(r['call_id']).get(timeout=a.wait)
        except (TimeoutError,modal.exception.TimeoutError):print('RUNNING '+r['call_id']);return
        except Exception as exc:
            r['error']=repr(exc)
            if isinstance(exc,(subprocess.CalledProcessError,modal.exception.RemoteError,modal.exception.FunctionTimeoutError)):r['status']='failed_terminal'
            save();raise
        r.update(status='done',finish=time.time(),result=result)
        if r['function'] in ('qwen','bge'):
            actual=(result['function_seconds']+10)*RATE;s['upper']+=actual-r['reserve'];r['reserve']=actual
        write(OUT/(a.job+'.json'),result);save();print(json.dumps(result),flush=True)
    elif a.action=='collect':
        r=s['calls'][a.job];assert r['status']=='done'
        dest=OUT/'outputs'/a.job;dest.mkdir(parents=True,exist_ok=True)
        for name in r['parts']:
            for suffix in ('.scores.json','.scores.complete.json','.scores.attempts.json'):
                cli('volume','get','--force','r2ai-vi56-data','/out/'+a.job+'/'+name+suffix,str(dest/(name+suffix)))
        cli('volume','get','--force','r2ai-vi56-data','/out/'+a.job+'/report.json',str(dest/'report.json'))
    else:
        assert all(r['status'] in ('done','failed_terminal') for r in s['calls'].values())
        apps=json.loads(cli('app','list','--json'))
        own=next(x for x in apps if x['app_id']==s['app_id'])
        if own['state']!='stopped':cli('app','stop','-y',s['app_id'])
        s.update(stopped=True,after=json.loads(cli('billing','summary','--json')));save()
        write(OUT/'apps_final.json',json.loads(cli('app','list','--json')))


if __name__=='__main__':main()
