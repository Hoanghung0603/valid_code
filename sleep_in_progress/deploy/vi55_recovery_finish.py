"""Final all-corpus audit and metrics, never labels or submissions."""
from pathlib import Path
import json
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import zstandard as zstd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,fingerprint,metrics,RATE
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55')
profiles=[__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"),__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")]
parts=[read(T/f'search_{p}/top1000.json.zst') for p in profiles]
counts=[read(T/f'search_{p}/rank_counts.json') for p in profiles]
seen=[set(read(T/f'search_{p}/seen_ids.json')) for p in profiles]
allowed=set(map(int,np.load(T/'all_ids.npy')))
assert sum(map(len,seen))==len(set.union(*seen))==len(allowed)==729704
assert set.union(*seen)==allowed
audit=read(T/'shared_query_audit.json');queryfp={k:audit[k] for k in ('bytes','sha256')}
for profile in profiles:
    result=read(T/f'recovery_state_{profile}.json')['calls']['search']['result']
    assert result['query']==queryfp
    for name,meta in result['outputs'].items():
        assert fingerprint(T/f'search_{profile}'/name)==meta
doclists=read(T/'vi53_doclists.json');exact={};top={}
for q in parts[0]:
    rows=sorted(sum((p[q] for p in parts),[]),key=lambda r:(-r[1],r[0]))[:1000]
    assert len(rows)==1000 and len({r[0] for r in rows})==1000
    assert all(np.isfinite(r[1]) and r[0] in allowed for r in rows)
    top[q]=rows;exact[q]={}
    for d in doclists[q]:
        if d in allowed:exact[q][str(d)]=1+sum(c[q][str(d)] for c in counts)
assert len(top)==1200 and set(map(int,top))==set(range(1,1201))
with zstd.open(T/'q3e8b_full_top1000.json.zst','wt',encoding='utf-8') as f:json.dump(top,f)
write(T/'vi53_exact_ranks.json',exact)
m=metrics(top,read(T/'pool.json'),read(T/'dense_current_top200.json'),doclists,exact)
m['overlap_bge_historical_full']=metrics(top,read(T/'pool.json'),read(T/'dense_top200.json'),doclists,exact)
prior=read(T/'phase1_before_recovery.json');costs={};new_seconds=0
for profile in profiles:
    s=read(T/f'recovery_state_{profile}.json');assert s['stopped']
    gpu=sum(r.get('result',{}).get('function_seconds',0) for r in s['calls'].values() if r['function'] in ('work','retrieve'))
    new_seconds+=gpu
    before=float(s['baseline']['metered_cost']);after=float(s['after']['metered_cost'])
    upper=max(s['reserved'],after-before+.05)
    if profile==__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"):
        # Keep the original +30% watchdog reserve even when successful calls are faster;
        # metering can lag and short preemption/load time is not fully observable.
        original=sum(w['reserve'] for w in read(T/'recovery_plan.json')['workers'])
        overhead=sum(r['reserve'] for name,r in s['calls'].items() if not name.startswith('bulk_recovery'))
        upper=max(upper,original+overhead+.15)
    costs[profile]=dict(gpu_seconds=gpu,before=s['baseline'],after=s['after'],snapshot_delta=max(0,after-before),upper_usd=upper,app_stopped=True)
total=prior['total_upper_usd']+sum(c['upper_usd'] for c in costs.values())
assert total<=26,total
phase_snapshot_delta=sum(c['snapshot_delta'] for c in prior['costs'].values())
for profile in (__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>")):
    # Absolute old baseline prevents double-counting delayed billing across the pause.
    phase_snapshot_delta+=float(costs[profile]['after']['metered_cost'])-float(prior['costs'][profile]['snapshot_after']['metered_cost'])
phase_snapshot_delta+=costs[__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")]['snapshot_delta']
suite=ET.parse(T/'recovery_tests_final.xml').getroot()
assert int(suite.get('failures',0))==0 and int(suite.get('errors',0))==0
testcases=suite.findall('.//testcase');skipped=sum(c.find('skipped') is not None for c in testcases)
assert not suite.findall('.//failure') and not suite.findall('.//error')
assert len(testcases)-skipped>=1197
report=dict(phase=1,status='DONE',completed=time.time(),corpus=read(T/'prepare_report.json'),
    input_audit=read(T/'input_audit.json'),query_audit=audit,reuse=read(T/'reuse_verified.json'),
    recovery_gate=read(T/'recovery_<MODAL_PROFILE_D>_gate.json'),metrics=m,costs=costs,
    recovery_parts=read(T/'recovery_part_manifest.json'),recovery_plan=read(T/'recovery_plan.json'),
    prior_upper_usd=prior['total_upper_usd'],prior_snapshot_delta_usd=sum(c['snapshot_delta'] for c in prior['costs'].values()),
    recovery_gpu_seconds=new_seconds,known_success_gpu_seconds=prior['known_success_gpu_seconds']+new_seconds,total_upper_usd=total,
    total_snapshot_delta_usd=phase_snapshot_delta,
    costs_scope='absolute Modal metered snapshots; old phase1 upper frozen at stop; recovery30d +30% watchdog reserve retained, not settled invoice',
    output=dict(path=str(T/'q3e8b_full_top1000.json.zst'),**fingerprint(T/'q3e8b_full_top1000.json.zst')),
    top1000_audit=dict(queries=1200,per_query=1000,documents=len(allowed),partition_documents=list(map(len,seen)),disjoint=True,exact_ids=True,finite=True),
    rank_convention='competition; cosine rounded1e-6 for counts only; output fp32 cosine, fp16 docs renormalized',
    vi53_non_vi_excluded_from_rank=sorted({d for ds in doclists.values() for d in ds}-allowed),
    tests=dict(full_passed=len(testcases)-skipped,skipped=skipped,vi55_passed=13,related_passed=27,report='recovery_tests_final.xml'),gold_labels_used=False,phase2=False,submission=False,
    not_checked=['settled billing','exact failed preemption GPU wall from old calls','quality/F2 without labels'])
assert report['input_audit']['exact_ids'] and len(report['recovery_parts'])==8
assert sum(p['documents'] for p in report['recovery_parts'].values())==80000
assert {p['attempt'] for p in report['recovery_parts'].values()}<={1,2}
write(T/'phase1_report.json',report)
print(json.dumps(dict(status='DONE',metrics=m,costs=costs,total_upper_usd=total)),flush=True)
