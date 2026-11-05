import importlib.util
from pathlib import Path


def test_plan_exact_partition_and_budget():
    path=Path(__file__).resolve().parents[1]/'deploy/vi51_plan.py'
    spec=importlib.util.spec_from_file_location('vi51_plan',path)
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    prep=dict(parts={f'p{i}':dict(tokens=(i+1)*1000) for i in range(29)},query_tokens=100)
    p=v.plan(prep,dict(tokens_per_second=10000),'a',4)
    names=[n for j in p['jobs'] for n in j['parts']]
    assert sorted(names)==sorted(prep['parts']) and len(names)==len(set(names))
    assert sum(j['tokens'] for j in p['jobs'])==sum(r['tokens'] for r in prep['parts'].values())
    assert p['reserve']==sum(j['reserve'] for j in p['jobs'])
    assert all(j['watchdog']>=1.3*(j['tokens']+prep['query_tokens'])/10000+90 for j in p['jobs'])
