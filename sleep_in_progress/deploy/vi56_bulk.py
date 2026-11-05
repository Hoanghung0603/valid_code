"""Single writer: launch fixed parts, persist each ID, collect existing calls only."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi56 import read,write,fingerprint,TEMP
OUT=TEMP/'vi56';PY=TEMP/'modalenv/Scripts/python.exe'
os.environ.update(PYTHONIOENCODING='utf-8',MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"))


def control(*args):
    subprocess.run([str(PY),str(ROOT/'deploy/vi56_control.py'),*args],check=True,env=os.environ)


def main():
    state=read(OUT/'state.json')
    assert all(state['calls'][n]['status']=='done' for n in ('weights','ping_fix','bench_qwen_fix','bench_bge_fix'))
    assert state['bulk_gate']['usd_upper_total']<=10
    assert read(OUT/'input_audit.json')['all_sha']
    assert read(OUT/'sanity.json')['exact']==1200
    jobs=[('qwen00','qwen',[f'qwen.part{i:03}.jsonl.zst' for i in (0,1)],1050),
          ('qwen01','qwen',[f'qwen.part{i:03}.jsonl.zst' for i in (2,3)],1050),
          ('bge00','bge',[f'bge.part{i:03}.jsonl.zst' for i in range(4)],800),
          ('bge01','bge',[f'bge.part{i:03}.jsonl.zst' for i in range(4,8)],800),
          ('qwen02','qwen',[f'qwen.part{i:03}.jsonl.zst' for i in (4,5,6)],1400)]
    write(OUT/'bulk_plan.json',dict(jobs=jobs,upper_reserved_new=sum((j[3]+10)*4.5232/3600 for j in jobs),
          forecast=read(OUT/'prepare_report.json')['forecast']))
    assert state['upper']+sum((j[3]+10)*4.5232/3600 for j in jobs)<10
    for name,stage,parts,watchdog in jobs:
        s=read(OUT/'state.json')
        running=[n for n,r in s['calls'].items() if r['function'] in ('qwen','bge') and r['status']=='running']
        if name not in s['calls'] and len(running)>=4:
            control('poll','--job',running[0],'--wait','2700')
            assert read(OUT/'state.json')['calls'][running[0]]['status']=='done'
        s=read(OUT/'state.json')
        if name not in s['calls']:control('spawn','--job',name,'--function',stage,'--watchdog',str(watchdog),'--parts',*parts)
        else:assert s['calls'][name]['parts']==parts
    for name,stage,parts,watchdog in jobs:
        s=read(OUT/'state.json')
        if s['calls'][name]['status']=='running':control('poll','--job',name,'--wait','2700')
        assert read(OUT/'state.json')['calls'][name]['status']=='done'
        control('collect','--job',name)
    control('stop')
    print('BULK_COLLECT_STOP_DONE',flush=True)


if __name__=='__main__':main()
