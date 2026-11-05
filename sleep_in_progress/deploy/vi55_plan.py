"""Freeze token limit and disjoint workspace/worker allocation after one benchmark."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,forecast,RATE
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55')
p=read(T/'prepare_report.json');b=read(T/'<MODAL_PROFILE_C>_bench1024.json')
assert not (T/'decision.json').exists()
assert b['repeat_cosine_min']>=.999 and b['repeat_cosine_mean']>=.9999
states=[read(T/f'state_{profile}.json') for profile in (__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"))]
prior=sum(s['upper'] for s in states)
f1024=forecast(p['tokens1024']+p['query_tokens']*8,b['tokens_per_second'])
# Include conservative 2 x 240-second retrieval reservations and CPU/storage reserve.
tail=2*248*RATE+.15
length=1024 if prior+f1024['usd']+tail<=26 else 512
f=forecast(p[f'tokens{length}']+p['query_tokens']*8,b['tokens_per_second'])
assert prior+f['usd']+tail<=26
workers=[dict(profile=__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>") if i<4 else __import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"),job=f'bulk{i:02d}',parts=[],tokens=0) for i in range(8)]
for name,meta in sorted(p['parts'].items(),key=lambda x:-x[1][f'tokens{length}']):
    w=min(workers,key=lambda w:w['tokens']);w['parts'].append(name);w['tokens']+=meta[f'tokens{length}']
for w in workers:
    w['watchdog']=int(1.3*(w['tokens']+p['query_tokens'])/b['tokens_per_second']+40)
    w['reserve']=(w['watchdog']+8)*RATE
assert prior+sum(w['reserve'] for w in workers)+tail<=26
decision=dict(doc_len=length,reason='1024 if forecast +30%, overhead, prior and search reserves <=26 USD; else512',
              throughput=b['tokens_per_second'],forecast1024=f1024,chosen_forecast=f,prior_upper=prior,
              tail_reserve=tail,total_reserved_upper=prior+sum(w['reserve'] for w in workers)+tail,workers=workers)
write(T/'decision.json',decision)
from vi55_local import handoff
handoff();print(decision,flush=True)
