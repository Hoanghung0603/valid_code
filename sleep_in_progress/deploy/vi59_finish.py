"""Run only after bulk_done: aggregate, CX package, one submission build, report costs."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'deploy'))
from vi55 import read,write,fingerprint
import vi59
import vi59_control as control
OUT=ROOT.parent/'r2ai_deliver/cand/vi59d'
CHAIN_LOG=OUT.parent/'run_cx_chain.log'
CHAIN_PID=OUT.parent/'run_cx_chain.pid'
CHAIN_DONE=('vi58 checker exit=','vi58 FAILED','vi58 skipped','vi58 RAM wait timed out',
            'vi58 blocked','vi57 did not pass','CHAIN END')


def chain_alive():
    if not CHAIN_PID.exists():return False
    pid=CHAIN_PID.read_text(encoding='utf-8').strip()
    out=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/NH'],capture_output=True,text=True).stdout
    return pid in out


def wait_cx_chain():
    """Shared RAM: build vi59d only after the CX chain has finished vi58."""
    while True:
        text=CHAIN_LOG.read_text(encoding='utf-8',errors='replace') if CHAIN_LOG.exists() else ''
        if any(s in text for s in CHAIN_DONE) or not chain_alive():return
        assert time.strftime('%H:%M')<'21:30','STOP: CX chain still on vi57/vi58 at 21:30'
        print(time.strftime('%H:%M:%S')+' WAIT_CX_CHAIN',flush=True)
        time.sleep(120)
os.environ.update(PYTHONIOENCODING='utf-8',MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"))


def main():
    assert read(OUT/'bulk_done.json')['done']
    state=read(OUT/'state.json');assert state['stopped']
    if not (OUT/'cx_extra_report.json').exists():vi59.cx_extra(OUT)
    if not (OUT/'build_report.json').exists():
        wait_cx_chain()
        vi59.build(OUT)
    calls=state['calls']
    seconds=sum(r.get('result',{}).get('function_seconds',0) for r in calls.values() if r['function'] in ('qwen','bge'))
    baseline=float(state['baseline']['metered_cost']);snapshot=float(state['after']['metered_cost'])
    cost=dict(known_gpu_function_seconds=seconds,baseline_snapshot=baseline,metered_snapshot=snapshot,
              snapshot_delta=snapshot-baseline,usd_upper=max(state['upper'],snapshot-baseline)+.1,
              billing_settled=False,app_stopped=True,volumes_retained=True)
    write(OUT/'cost_report.json',cost);assert cost['usd_upper']<=12
    control.handoff(state)
    write(OUT/'finish_done.json',dict(done=True,zip=fingerprint(OUT/'submission_vi59d.zip'),cx=fingerprint(OUT/'cx_extra.zip')))


if __name__=='__main__':main()
