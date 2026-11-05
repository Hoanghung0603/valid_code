"""vi45 isolated CPU preparation and Qwen3 chunk scoring. Never submits."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

MODEL = 'Qwen/Qwen3-Reranker-8B'
RATE = (3.95 + 4*.0473 + 48*.008)/3600


def key(c):
    return str(c['doc_id']), c['chunk_text']


def cid(q, c):
    return hashlib.sha256((str(q)+'\0'+str(c['doc_id'])+'\0'+c['chunk_text']).encode()).hexdigest()


def rank(rows, logits=None):
    """Only scored candidates precede unscored; coverage bonus stays exactly 1."""
    from chunk_docfuse import zscores
    from cx93 import apply_bonus
    if logits is None:
        return apply_bonus(sorted(rows, key=lambda c: -c['s']), 1.)
    chosen = [c for c in rows if c['cid'] in logits]
    assert all(math.isfinite(logits[c['cid']]) for c in chosen)
    z = zscores([logits[c['cid']] for c in chosen])
    scored = [{**c, 'z8c': v, 's': c['s']+v} for c, v in zip(chosen, z)]
    first = apply_bonus(sorted(scored, key=lambda c: -c['s']), 1.)
    tail = apply_bonus(sorted([c for c in rows if c['cid'] not in logits], key=lambda c: -c['s']), 1.)
    return first+tail


def shortlist(rows, n):
    return rank(rows)[:max(2*n, 200)]


def supplement_rows(q, rows, bridges, n, mean, sd):
    seen = {key(c) for c in rows}
    added = set()
    for b in bridges:
        if key(b) in seen:
            continue
        seen.add(key(b))
        zc = (b['score']-mean)/sd if sd else 0.
        # zd belongs to the unchanged full document population, supplied by caller.
        c = dict(doc_id=b['doc_id'], chunk_text=b['chunk_text'], logit=b['score'],
                 zc=zc, zd=b['zd'], s=zc+4*b['zd'])
        c['cid'] = cid(q, c)
        rows.append(c)
        added.add(c['cid'])
    return [c for c in shortlist(rows, n) if c['cid'] in added]


def supplement():
    """Preserve all 18 prepared parts; append only newly eligible vi44 bridges."""
    import dwmax_diag as dw
    import vi37
    from chunk_docfuse import doc_z
    from fuse3 import fuse3
    from vi42 import fingerprint
    root, out = vi37.ROOT, vi37.ROOT/'data/sub/vi45'
    prior = dw.read(out/'prepare_report.json')
    assert prior['scored_candidates'] == 296973
    src = out/'modal_input'
    assert all(fingerprint(src/name) == meta for name, meta in prior['files'].items())
    ref = dw.read(root/'data/sub/vi44/submission_vi44.json')
    cache = dw.read(out/'candidates_vi45.json.zst')
    bridges = dw.read(root/'data/sub/vi44/bridges_vi44.json.zst')
    raw = dw.read(vi37.OUT/'cands_v37.json.zst')
    data = vi37.inputs()
    data['q8b'] = dw.read(vi37.OUT/'q8b_v37.json')
    fused = fuse3(data['pool'], data['q8b'], data['dense'], (1.,1.,2.,0.),60,None,None)
    del data
    qt = vi37.query_texts()
    writer = vi37.Parts(src, 'bridge44')
    added = total = order_ok = set_ok = 0
    for it in ref:
        q = str(it['id'])
        vals = [float(r[3]) for r in raw.pop(q)]
        mean, sd = statistics.mean(vals), statistics.pstdev(vals)
        zd = doc_z(fused[q])
        bs = [{**b,'zd':zd.get(str(b['doc_id']),min(zd.values(),default=0.))} for b in bridges[q]]
        before = len(cache[q])
        extra = supplement_rows(q,cache[q],bs,len(it['relevant_chunks']),mean,sd)
        added += len(cache[q])-before
        for c in extra:
            writer.add(dict(qid=q,cid=c['cid'],doc_id=c['doc_id'],query=qt[q],text=c['chunk_text']))
        total += len(extra)
        zero = [key(c) for c in rank(cache[q])[:len(it['relevant_chunks'])]]
        base = [key(c) for c in it['relevant_chunks']]
        order_ok += zero == base
        set_ok += set(zero) == set(base)
        assert set(base) <= {key(c) for c in cache[q]}
    writer.close()
    vi37.writez(out/'candidates_vi45_vi44.json.zst',cache)
    files = {**prior['files'],**{p.name:fingerprint(p) for p in writer.files}}
    counts = {}
    import zstandard as zstd
    for name in files:
        with zstd.open(src/name,'rt',encoding='utf-8') as f:
            counts[name] = sum(1 for line in f if line.strip())
    report = dict(base='vi44',queries=1200,full_candidates=sum(map(len,cache.values())),
        scored_candidates=prior['scored_candidates']+total,new_bridges=added,new_scored_bridges=total,
        prior_pairs_preserved=296973,candidate_policy='original296973 + only new vi44 bridges in K=max(2*n_vi44,200)',
        sanity_zero_order=order_ok,sanity_zero_set=set_ok,sanity_changed_sets=1200-set_ok,
        sanity_reason='vi44 appends bridges after vi37 prefix; vi45 globally ranks union; formula unchanged',
        counts=counts,files=files)
    dw.write(out/'prepare_vi44_report.json',report)
    dw.write(src/'manifest_vi44.json',report)
    print(json.dumps(report),flush=True)


def assemble(ref, cache, logits):
    out, stats = [], []
    assert len({x['id'] for x in ref}) == len(ref)
    for it in ref:
        q = str(it['id'])
        rows = cache[q]
        assert len({c['cid'] for c in rows}) == len(rows)
        assert set(logits[q]) <= {c['cid'] for c in rows}
        selected = rank(rows,logits[q])[:len(it['relevant_chunks'])]
        assert len(selected) == len(it['relevant_chunks'])
        chunks = [dict(doc_id=c['doc_id'],chunk_text=c['chunk_text']) for c in selected]
        a,b = {key(c) for c in chunks},{key(c) for c in it['relevant_chunks']}
        assert len(a)==len(chunks)
        assert {str(c['doc_id']) for c in chunks} <= set(map(str,it['relevant_docs']))
        out.append({**it,'relevant_chunks':chunks})
        stats.append(dict(qid=q,jaccard=len(a&b)/len(a|b) if a|b else 1.,replaced=len(a-b)))
    return out,stats


def preverify():
    """Verify all candidate text, including doclist documents absent from vi37 chunks."""
    import dwmax_diag as dw
    import vi37
    from vi42 import fingerprint
    from rerank_run import load_docs
    out=vi37.ROOT/'data/sub/vi45'
    p=out/'candidates_vi45_vi44.json.zst'
    cache=dw.read(p)
    wanted={int(c['doc_id']) for rows in cache.values() for c in rows}
    _,_,files=vi37.export_texts(wanted,out/'verify_candidates','docs_verify_vi45')
    docs=load_docs([str(f) for f in files],wanted=wanted)
    assert wanted<=set(docs)
    bad=sum(c['chunk_text'] not in docs[int(c['doc_id'])] for rows in cache.values() for c in rows)
    assert bad==0
    report=dict(candidate_fingerprint=fingerprint(p),candidates=sum(map(len,cache.values())),
        docs=len(wanted),verbatim_bad=bad,files={str(f):fingerprint(f) for f in files})
    dw.write(out/'preverify_report.json',report)
    print('PREVERIFY_DONE '+json.dumps(report),flush=True)


def build():
    import dwmax_diag as dw
    import vi37
    import vi44
    from vi42 import fingerprint
    from reorder_zip import write_reordered_zip
    from rerank_run import load_docs
    from scipy.stats import spearmanr
    from chunk_docfuse import zscores
    root,out=vi37.ROOT,vi37.ROOT/'data/sub/vi45'
    paths=[root/'data/sub/vi37/submission_vi37.json',root/'data/sub/vi44/submission_vi44.json',
           out/'candidates_vi45_vi44.json.zst',out/'prepare_vi44_report.json']
    scorefiles=sorted((out/'scores').glob('*.logits.json'))
    assert scorefiles
    score_manifest=out/'score_transfer_manifest.json'
    if score_manifest.exists():
        transfer=dw.read(score_manifest)
        assert {p.name:fingerprint(p) for p in scorefiles}==transfer['files']
    paths += scorefiles
    before={str(p):fingerprint(p) for p in paths}
    # Preserve original vi34–vi44 artefacts, not only the directly read references.
    protected=[]
    for version in range(34,45):
        for suffix in ('json','zip'):
            p=root/f'data/sub/vi{version}/submission_vi{version}.{suffix}'
            if p.exists():
                protected.append(p)
    protected_before={str(p):fingerprint(p) for p in protected}
    ref37,ref,cache,manifest=map(dw.read,paths[:4])
    assert {str(it['id']):it['relevant_docs'] for it in ref} == {str(it['id']):it['relevant_docs'] for it in ref37}
    expected={}
    import zstandard as zstd
    for name,meta in manifest['files'].items():
        p=out/'modal_input'/name
        assert fingerprint(p)==meta
        with zstd.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                assert r['cid'] not in expected
                expected[r['cid']]=(r['qid'],str(r['doc_id']))
    logits={str(it['id']):{} for it in ref}
    flat=[]
    for p in scorefiles:
        for r in dw.read(p):
            q=str(r['qid'])
            assert expected[r['cid']]==(q,str(r['doc_id']))
            assert r['cid'] not in logits[q]
            assert math.isfinite(r['logit8c'])
            logits[q][r['cid']]=r['logit8c']
            flat.append(r)
    assert sum(map(len,logits.values())) == len(expected) == manifest['scored_candidates']
    vi37.writez(out/'chunk8b_logits_vi45.json.zst',flat)
    del flat
    sub,perquery=assemble(ref,cache,logits)
    pooled8,pooledc,rhos=[],[],[]
    for q,rows in cache.items():
        pairs=[(logits[q][c['cid']],c['zc']) for c in rows if c['cid'] in logits[q]]
        a,b=zip(*pairs)
        pooled8.extend(zscores(list(a)))
        pooledc.extend(b)
        rho=float(spearmanr(a,b).statistic)
        if math.isfinite(rho):
            rhos.append(rho)
    del cache,logits
    zp=out/'submission_vi45.zip'
    info=write_reordered_zip(sub,zp)
    assert info['zip_bytes']<=99000000
    checks=vi44.final_checks(sub,zp,'vi45',out)
    wanted={int(c['doc_id']) for it in sub for c in it['relevant_chunks']}
    # vi45 can select U900 from any vi37 doclist document, including ones without
    # a selected vi37 chunk. Its old verify_text export is NOT sufficient.
    pre=out/'preverify_report.json'
    if pre.exists():
        verified=dw.read(pre)
        assert verified['candidate_fingerprint']==fingerprint(out/'candidates_vi45_vi44.json.zst')
        files=list(verified['files'])
        assert all(fingerprint(p)==verified['files'][p] for p in files)
    else:
        _,_,files=vi37.export_texts(wanted,out/'verify_text','docs_verify_vi45')
    docs=load_docs([str(p) for p in files],wanted=wanted)
    assert wanted<=set(docs)
    verbatim=sum(c['chunk_text'] not in docs[int(c['doc_id'])] for it in sub for c in it['relevant_chunks'])
    assert verbatim==0
    assert before=={str(p):fingerprint(p) for p in paths}
    assert protected_before=={str(p):fingerprint(p) for p in protected}
    pooled_rho=float(spearmanr(pooled8,pooledc).statistic) if len(set(pooled8))>1 and len(set(pooledc))>1 else None
    dw.write(out/'offline_vi45.json',dict(per_query=perquery,
        jaccard_macro=statistics.mean(r['jaccard'] for r in perquery),
        replaced_total=sum(r['replaced'] for r in perquery),
        changed_queries=sum(r['replaced']>0 for r in perquery),
        spearman_z8c_zc_pooled=pooled_rho if pooled_rho is not None and math.isfinite(pooled_rho) else None,
        spearman_z8c_zc_macro=statistics.mean(rhos) if rhos else None,
        spearman_finite_queries=len(rhos)))
    report=dict(**info,**checks,sha256=fingerprint(zp)['sha256'],verbatim_bad=verbatim,
        pairs=len(expected),docs_equal_vi37=1200,chunk_counts_equal_vi44=1200,input_fingerprints=before,
        protected_vi34_vi44_fingerprints=protected_before)
    dw.write(out/'build_report_vi45.json',report)
    print('DONE '+json.dumps(report),flush=True)


def prepare():
    import dwmax_diag as dw
    import vi29
    import vi37
    from fuse3 import fuse3
    from chunk_docfuse import doc_z
    root, out = vi37.ROOT, vi37.ROOT/'data/sub/vi45'
    out.mkdir(parents=True, exist_ok=True)
    ref = dw.read(root/'data/sub/vi42/submission_vi42.json')
    old = {str(x['id']): x for x in dw.read(vi37.OUT/'submission_vi37.json')}
    data = vi37.inputs()
    data['q8b'] = dw.read(vi37.OUT/'q8b_v37.json')
    fused = fuse3(data['pool'], data['q8b'], data['dense'], (1., 1., 2., 0.), 60, None, None)
    del data
    cands = dw.read(vi37.OUT/'cands_v37.json.zst')
    bridges = dw.read(root/'data/sub/vi42/bridges_vi42.json.zst')
    qt = vi37.query_texts()
    dest = out/'modal_input'
    dest.mkdir(exist_ok=True)
    writer = vi37.Parts(dest, 'pairs')
    cache = {}
    total = full = order_ok = set_ok = 0
    diffs = []
    for it in ref:
        q = str(it['id'])
        assert it['relevant_docs'] == old[q]['relevant_docs']
        raw = cands.pop(q)
        rows = vi29.order_q_docdedup(raw, fused[q], set(map(str, it['relevant_docs'])), 'doc_text', 4.)
        vals = [float(r[3]) for r in raw]
        mean, sd = statistics.mean(vals), statistics.pstdev(vals)
        zd = doc_z(fused[q])
        seen = {key(c) for c in rows}
        for b in bridges[q]:
            if key(b) in seen:
                continue
            seen.add(key(b))
            # vi42 raw bridge gate score, standardized with the unchanged U900 population.
            zc = (b['score']-mean)/sd if sd else 0.
            d = zd.get(str(b['doc_id']), min(zd.values(), default=0.))
            rows.append(dict(doc_id=b['doc_id'], chunk_text=b['chunk_text'], logit=b['score'], zc=zc, zd=d, s=zc+4*d))
        for c in rows:
            c['cid'] = cid(q, c)
        zero = rank(rows)[:len(it['relevant_chunks'])]
        a, b = [key(c) for c in zero], [key(c) for c in it['relevant_chunks']]
        order_ok += a == b
        set_ok += set(a) == set(b)
        if a != b and len(diffs) < 5:
            diffs.append(dict(qid=q, replaced=len(set(a)-set(b))))
        selected = shortlist(rows, len(it['relevant_chunks']))
        for c in selected:
            writer.add(dict(qid=q, cid=c['cid'], doc_id=c['doc_id'], query=qt[q], text=c['chunk_text']))
        cache[q] = rows
        full += len(rows)
        total += len(selected)
    writer.close()
    vi37.writez(out/'candidates_vi45.json.zst', cache)
    report = dict(full_candidates=full, scored_candidates=total, queries=len(ref),
        candidate_policy='K=max(2*n_vi42,200); rank by current zc+4zd with b_cov=1',
        bridge_zc='vi42 bridge gate raw score standardized by unchanged full U900 rows',
        sanity_zero_order=order_ok, sanity_zero_set=set_ok, sanity_examples=diffs,
        sanity_reason='vi42 appends bridges after frozen vi37 prefix; global ranking of union is not algebraically identical',
        files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in writer.files})
    dw.write(dest/'manifest.json', report)
    dw.write(out/'prepare_report.json', report)
    print(json.dumps(report), flush=True)


class TargetedLogprobWorker:
    def install_yes_no(self):
        import torch
        from transformers import AutoTokenizer
        from vllm.v1.sample.sampler import Sampler
        from vllm.v1.outputs import LogprobsTensors
        from vllm.v1.sample.ops.logprobs import batched_count_greater_than
        tok = AutoTokenizer.from_pretrained(MODEL)
        targets = [tok.convert_tokens_to_ids(t) for t in ('yes', 'no')]
        def gather(logprobs, num_logprobs, token_ids):
            assert num_logprobs == 2
            ids = torch.tensor(targets, device=logprobs.device, dtype=torch.long).unsqueeze(0).expand(logprobs.shape[0], -1)
            indices = torch.cat((token_ids.unsqueeze(-1), ids), dim=1)
            values = logprobs.gather(-1, indices)
            ranks = batched_count_greater_than(logprobs, values[:, :1])
            return LogprobsTensors(indices.to(torch.int32), values, ranks)
        Sampler.gather_logprobs = staticmethod(gather)
        return targets


def gpu(mode, src, dest):
    import vllm
    from vllm import LLM, SamplingParams
    import qwen_rerank_run as qrr
    assert vllm.__version__ == '0.15.1'
    start = time.monotonic()
    src, dest = Path(src), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # Prompt remains 1024; vLLM also needs one slot for the scored answer token.
    llm = LLM(model=MODEL, dtype='float16', max_model_len=1025, tensor_parallel_size=1,
        enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=.9,
        max_num_seqs=256, max_logprobs=2, logprobs_mode='raw_logprobs', seed=4545,
        worker_extension_cls='vi45.TargetedLogprobWorker', generation_config='vllm')
    tok = llm.get_tokenizer()
    ids = [tok.convert_tokens_to_ids(t) for t in ('yes','no')]
    assert llm.collective_rpc('install_yes_no') == [ids]
    load_wall = time.monotonic()-start
    def score(rows):
        enc = [qrr.build_input_ids(tok, qrr.DEFAULT_INSTRUCTION, r['query'], r['text'][:12288], 1024) for r in rows]
        assert all(len(p)<=1024 for p in enc)
        t = time.monotonic()
        obs = llm.generate([{'prompt_token_ids':p} for p in enc], SamplingParams(temperature=0,max_tokens=1,
            allowed_token_ids=ids,logprobs=2,seed=4545),use_tqdm=False)
        elapsed = time.monotonic()-t
        scores = []
        for ob in obs:
            probs = ob.outputs[0].logprobs[0]
            assert set(probs) == set(ids)
            value = float(probs[ids[0]].logprob-probs[ids[1]].logprob)
            assert math.isfinite(value)
            scores.append(value)
        return scores, sum(map(len,enc)), elapsed
    if mode == 'bench':
        rows = json.loads((src/'bench.json').read_text())
        values, tokens, elapsed = score(rows)
        repeat, rtokens, relapsed = score(rows[:50])
        import numpy as np
        pearson = float(np.corrcoef(values[:50],repeat)[0,1])
        assert pearson >= .999
        report = dict(pairs=len(rows)+50,unique_pairs=len(rows),input_tokens=tokens+rtokens,
            token_s=tokens/elapsed,repeat_token_s=rtokens/relapsed,pearson=pearson,
            max_abs_repeat=max(abs(a-b) for a,b in zip(values,repeat)),load_wall=load_wall,
            inference_wall=elapsed+relapsed,seconds=time.monotonic()-start,
            scores=values,model=MODEL,max_len=1024)
        (dest/'bench_report.json').write_text(json.dumps(report))
        return report
    assert mode == 'bulk'
    import zstandard as zstd
    import argparse
    config = json.loads((src/'job.json').read_text())
    reports = []
    for part in config['parts']:
        p = src/part['name']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == part['sha256']
        with zstd.open(p,'rt',encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == part['count']
        assert len({r['cid'] for r in rows}) == len(rows)
        target = dest/(p.name.replace('.jsonl.zst','.logits.json'))
        assert not target.exists(), 'Refuse duplicate scoring/output overwrite'
        values, tokens, elapsed = score(rows)
        output = [dict(qid=r['qid'],cid=r['cid'],doc_id=r['doc_id'],logit8c=v) for r,v in zip(rows,values)]
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(output),encoding='utf-8')
        tmp.replace(target)
        report = dict(part=p.name,pairs=len(rows),tokens=tokens,inference_wall=elapsed,
            output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),input_sha256=part['sha256'])
        target.with_suffix('.report.json').write_text(json.dumps(report),encoding='utf-8')
        reports.append(report)
        # Commit each completed part, not only at job end.
        import modal
        modal.Volume.from_name('r2ai-runner').commit()
        print('PART_DONE '+json.dumps(report),flush=True)
    report = dict(parts=reports,load_wall=load_wall,seconds=time.monotonic()-start,model=MODEL,max_len=1024)
    (dest/'bulk_report.json').write_text(json.dumps(report),encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage',choices=['prepare','supplement','preverify','build','gpu'])
    ap.add_argument('--mode',default='bench')
    ap.add_argument('--input')
    ap.add_argument('--output')
    a = ap.parse_args()
    if a.stage == 'gpu':
        print(json.dumps(gpu(a.mode,a.input,a.output)))
    else:
        import vi37
        from vi39 import wait_lock
        with wait_lock(vi37.ROOT/'locks/heavy_job.lock',f'vi45 pid={os.getpid()}'):
            {'supplement':supplement,'prepare':prepare,'preverify':preverify,'build':build}[a.stage]()


if __name__ == '__main__':
    main()
