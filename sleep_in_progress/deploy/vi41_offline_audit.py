"""Read-only CPU diagnostics against vi37; writes only vi41, never a quality gate."""
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('R2AI_ROOT', 'D:/r2ai'))
sys.path.insert(0, str(ROOT/'scripts'))


def proxy(rows, scores):
    """Same fixed 8B>0, MAR sensitivity, G=72 as scripts/judge_calib.py."""
    ps, counts, lower, upper = [], [], [], []
    missing = total = 0
    for row in rows:
        ds = row['relevant_docs']
        hits = scores.get(str(row['id']), {})
        observed = [hits[d] for d in ds if d in hits]
        n, k = len(ds), len(observed)
        good = sum(v > 0 for v in observed)
        total += n
        missing += n-k
        ps.append(good/k if k else (0. if not n else float('nan')))
        counts.append(ps[-1]*n)
        lower.append(good/n if n else 0.)
        upper.append((good+n-k)/n if n else 0.)
    valid = all(math.isfinite(p) for p in ps)
    mean = lambda xs: sum(xs)/len(xs)
    return dict(missing=missing,total=total,missing_rate=missing/total if total else 0.,
        p_conditional=mean(ps) if valid else None,p_lower=mean(lower),p_upper=mean(upper),
        correct_per_query_adjusted=mean(counts) if valid else None,
        r_proxy=mean(counts)/72 if valid else None)


def summarize(old, new, fresh, fused, old8, new8):
    n = len(new)
    assert n == 1200 and {r['id'] for r in new} == set(range(1,1201))
    def summary(rows):
        return dict(docs_per_query=sum(len(r['relevant_docs']) for r in rows)/n,
            chunks_per_query=sum(len(r['relevant_chunks']) for r in rows)/n,
            docs_with_chunks_per_query=sum(len({c['doc_id'] for c in r['relevant_chunks']}) for r in rows)/n)
    top30 = sum(sum(int(r[0]) in set(fresh[q]) for r in hits[:30]) for q,hits in fused.items())
    in_final = sum(len(set(r['relevant_docs']) & set(fresh[str(r['id'])])) for r in new)
    pold, pnew = proxy(old,old8), proxy(new,new8)
    return dict(queries=n,old=summary(old),new=summary(new),new_pool_docs_in_top30=top30,
        new_pool_docs_in_top30_per_query=top30/n,new_pool_docs_in_doclist=in_final,
        new_pool_docs_in_doclist_per_query=in_final/n,
        proxy_8b=dict(old=pold,new=pnew,r_proxy_delta=pnew['r_proxy']-pold['r_proxy'],
            threshold=0,G_estimate=72,not_a_gate=True,
            warning='Descriptive proxy only; not relevance probability, BTC recall or predicted Final; missing-at-random sensitivity.'),
        delta=55)


def main():
    import numpy as np
    import dwmax_diag as dw
    from vi41 import read, write
    out=ROOT/'data/sub/vi41'
    with dw.owned_lock(ROOT/'locks/heavy_job.lock',f'vi41 offline_audit pid={os.getpid()}'):
        old=read(ROOT/'data/sub/vi37/submission_vi37.json')
        new=read(out/'submission_vi41.json')
        def scores(path):
            return {q:{int(h[0]):float(h[2]) for h in hs if h[2] is not None and math.isfinite(float(h[2]))}
                    for q,hs in read(path).items()}
        report=summarize(old,new,read(out/'new_ids.json'),read(out/'cache_fhyde_v41.json'),
            scores(ROOT/'data/sub/vi37/q8b_v37.json'),scores(out/'in1/q8b_v41.json'))
        q8=scores(out/'in1/q8b_v41.json')
        lengths=read(out/'export_lengths.json')
        missing=[(str(r['id']),d) for r in new for d in r['relevant_docs']
                 if d not in q8.get(str(r['id']),{})]
        report['missing_8b_final']={'pairs':len(missing),
            'empty_new_text':sum(str(d) in lengths and not lengths[str(d)]['windows'] for q,d in missing),
            'nonempty_or_old':sum(str(d) not in lengths or bool(lengths[str(d)]['windows']) for q,d in missing)}
        # Keep empty-text sentinel out of the descriptive clinical logit distribution.
        old_by_q={str(r['id']):r['relevant_docs'] for r in old}
        fresh=read(out/'new_ids.json')
        bg=scores(out/'in1/pool_v41.json')
        oldbg=scores(ROOT/'data/sub/vi36/in1/pool_v36.json')
        paired_new,paired_old=[],[]
        for row in new:
            q=str(row['id'])
            for rank,d in enumerate(row['relevant_docs']):
                if d not in set(fresh[q]) or not lengths[str(d)]['windows'] or rank>=len(old_by_q[q]):
                    continue
                paired_new.append(bg[q][d])
                paired_old.append(oldbg[q][old_by_q[q][rank]])
        report['bge_nonempty_same_rank']={k:dict(n=len(v),mean=float(np.mean(v)),
            quantiles_min_p10_p50_p90_max=np.quantile(v,[0,.1,.5,.9,1]).tolist())
            for k,v in [('new',paired_new),('vi37_same_rank',paired_old)]}
        assert read(out/'build_report.json')['delta']==report['delta']
        write(out/'offline_extra_report.json',report)
        print(json.dumps(report))


if __name__=='__main__':
    main()
