"""vi56: one fixed top100 expansion, complete cross-encoder scores, vi53 RRF.

Immutable historical sources; no submission API. CPU prepare on b660 only.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

from vi55 import read, write, iterz, fingerprint

TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
RATE = 4.5232 / 3600


def select(top, pool, allowed):
    assert set(top) == set(pool)
    result = {}
    for q, hits in pool.items():
        old = {int(h[0]) for h in hits}
        assert len(top[q]) >= 100
        first = top[q][:100]
        assert len({int(h[0]) for h in first}) == 100
        result[q] = [int(d) for d, s in first if int(d) not in old and int(d) in allowed]
    return result


def local_select(out):
    import numpy as np
    ctx = read(TEMP/'vi50/fusion_context.json.zst')
    top = read(TEMP/'vi55/q3e8b_full_top1000.json.zst')
    allowed = set(map(int, np.load(TEMP/'vi55/all_ids.npy')))
    # Same mandatory contamination exclusion as the historical submission pipeline.
    bad=set(map(int,read(TEMP/'vi45/bad_ids.json')))-set(map(int,read(TEMP/'vi45/new_ids.json')))
    excluded=[(q,int(d)) for q,rs in top.items() for d,s in rs[:100]
              if int(d) in bad and int(d) not in {int(h[0]) for h in ctx['data']['pool'][q]}]
    allowed-=bad
    if (out/'prepare_report.json').exists():
        prep=read(out/'prepare_report.json')
        body_empty={int(d) for q,d in prep['empty_pairs']}
        allowed-=body_empty
    fresh = select(top, ctx['data']['pool'], allowed)
    write(out/'new_ids.json', fresh)
    q3e = {q: {str(d): s for d, s in rows[:100] if int(d) in set(fresh[q])} for q, rows in top.items()}
    write(out/'q3e_new.json', q3e)
    report = dict(new_pairs=sum(map(len, fresh.values())), unique_docs=len({d for ds in fresh.values() for d in ds}),
                  queries=len(fresh), top=100, contaminated_excluded=excluded,inputs={str(p): fingerprint(p) for p in
                  (TEMP/'vi50/fusion_context.json.zst', TEMP/'vi55/q3e8b_full_top1000.json.zst', TEMP/'vi55/all_ids.npy')})
    write(out/'selection_report.json', report)
    print(json.dumps(report), flush=True)


def forecast(qwen_pairs, bge_pairs, jobs_q=4, jobs_b=2):
    qs = qwen_pairs * 1024 / 40661.696 * 1.3 + jobs_q * 65
    bs = bge_pairs / 300 * 1.3 + jobs_b * 35
    return dict(qwen_pairs=qwen_pairs, bge_pairs=bge_pairs, qwen_seconds=qs, bge_seconds=bs,
                usd_upper=(qs+bs)*RATE + .25, margin=1.3, cap=10.)


def prepare(root, out):
    import vi37
    from chunking import chunk_document
    from docwin_run import window_spans
    from vi45 import cid
    start = time.monotonic()
    assert not (out/'prepare_report.json').exists()
    vi37.ROOT = root
    fresh = read(out/'new_ids.json')
    wanted = {d for ds in fresh.values() for d in ds}
    lengths, _, files = vi37.export_texts(wanted, out/'texts', 'docs56')
    qt = vi37.query_texts()
    reverse = defaultdict(list)
    for q, ds in fresh.items():
        for d in ds: reverse[d].append(q)
    writers = {s: vi37.Parts(out/'modal_input', s) for s in ('bge', 'qwen')}
    counts = defaultdict(int)
    expected = {s: [] for s in writers}
    empty = []
    for p in files:
        for doc in iterz(p):
            d, idx, text = doc['id'], doc['idx'], doc['text']
            if not text.strip():
                empty.extend([q, d] for q in reverse[d])
                continue
            spans = window_spans(text, 300, 250)
            chunks = chunk_document(text, 'vi', max_units=900)
            for q in reverse[d]:
                def add(stage, kind, content, identity):
                    row = dict(qid=q, doc_id=d, idx=idx, kind=kind, identity=identity,
                               query=qt[q], text=content)
                    writers[stage].add(row)
                    expected[stage].append([identity, q, d, kind])
                    counts[kind] += 1
                add('bge', 'bge_doc', text, f'{q}:{d}:bge_doc')
                add('qwen', 'q8b_doc', text, f'{q}:{d}:q8b_doc')
                for s, e in spans:
                    add('bge', 'bge_win', text[s:e], f'{q}:{d}:bge_win:{s}:{e}')
                seen = set()
                for c in chunks:
                    ident = cid(q, dict(doc_id=d, chunk_text=c.text))
                    if ident in seen: continue
                    seen.add(ident)
                    add('qwen', 'd8w', c.text, ident)
    for w in writers.values(): w.close()
    parts = {}
    for stage, w in writers.items():
        for p in w.files:
            ids = [r['identity'] for r in iterz(p)]
            assert len(set(ids)) == len(ids)
            parts[p.name] = dict(**fingerprint(p), stage=stage, count=len(ids))
        write(out/f'expected_{stage}.json', expected[stage])
    fc = forecast(counts['q8b_doc']+counts['d8w'], counts['bge_doc']+counts['bge_win'])
    report = dict(pairs=sum(map(len, fresh.values())), unique_docs=len(wanted), empty_pairs=empty,
                  counts=dict(counts), parts=parts, forecast=fc, lengths=lengths,
                  text_files={str(p): fingerprint(p) for p in files}, seconds=time.monotonic()-start,
                  sources=['bge_doc', 'q8b_doc', 'dense_bge_m3', 'hyde_dense', 'bge_docwin', 'd8w', 'q3e8b'],
                  missing='historical dense/HyDE cosine for outside-top200 docs: exact vi41.append_last_ranks fallback',
                  text_definition='vi37.export_texts: body only, cleaned, cap200000; U900 and windows300/250',
                  q3e_definition='vi55: new union docs1024 tokens, reused old-union docs2048 tokens; no reembedding')
    write(out/'prepare_report.json', report)
    assert fc['usd_upper'] <= 10, ('STOP over cap', fc)
    print('PREPARED '+json.dumps({k:v for k,v in report.items() if k not in ('lengths','parts')}), flush=True)


def completed_part(source, dest):
    meta = dest.with_suffix('.complete.json')
    if not dest.exists() or not meta.exists(): return None
    try:
        m = read(meta)
        assert m['input'] == fingerprint(source) and m['output'] == fingerprint(dest)
        rows = list(iterz(source)); got = read(dest)
        assert [r['identity'] for r in rows] == [r['identity'] for r in got]
        assert all(math.isfinite(r['score']) for r in got)
        assert all((a['qid'], a['doc_id'], a['kind']) == (b['qid'], b['doc_id'], b['kind']) for a,b in zip(rows,got))
        return m
    except (AssertionError, ValueError, OSError, KeyError): return None


def process_part(source, dest, scorer, commit=lambda: None):
    done = completed_part(source, dest)
    if done: return dict(done, skipped=True)
    attempt_path = dest.with_suffix('.attempts.json')
    attempts = read(attempt_path) if attempt_path.exists() else []
    assert len(attempts) < 2, 'STOP same part failed/preempted twice'
    attempts.append(dict(start=time.time())); write(attempt_path, attempts); commit()
    rows = list(iterz(source))
    assert len({r['identity'] for r in rows}) == len(rows)
    start = time.monotonic(); scores = scorer(rows)
    assert len(scores) == len(rows) and all(math.isfinite(s) for s in scores)
    output = [{k:r[k] for k in ('identity','qid','doc_id','idx','kind')} | dict(score=float(s)) for r,s in zip(rows,scores)]
    write(dest, output)
    report = dict(input=fingerprint(source), output=fingerprint(dest), pairs=len(rows), seconds=time.monotonic()-start)
    write(dest.with_suffix('.complete.json'), report); commit()
    assert completed_part(source, dest)
    return report


class TargetedLogprobWorker:
    def install_yes_no(self):
        import vi45
        vi45.MODEL='/weights/qwen'
        return vi45.TargetedLogprobWorker.install_yes_no(self)


def load_scorer(stage, metrics):
    if stage == 'bge':
        from rerank_run import load_model, score_pairs
        tok, model = load_model('/weights/bge')
        def score(rows):
            metrics['tokens'] += sum(map(len, tok([r['query'] for r in rows], [r['text'] for r in rows],
                                                   truncation='only_second', max_length=512)['input_ids']))
            return [round(v,4) for v in score_pairs(model,tok,[(r['query'],r['text']) for r in rows],512,16)]
    else:
        from vllm import LLM, SamplingParams
        import qwen_rerank_run as qrr
        llm = LLM(model='/weights/qwen', dtype='float16', max_model_len=1025,
                  tensor_parallel_size=1, enable_chunked_prefill=True, enable_prefix_caching=True,
                  gpu_memory_utilization=.9, max_num_seqs=256, max_logprobs=2,
                  logprobs_mode='raw_logprobs', seed=4545,
                  worker_extension_cls='vi56.TargetedLogprobWorker', generation_config='vllm')
        # vi45 extension loads its tokenizer by MODEL; point it at the same offline weights.
        tok = llm.get_tokenizer(); ids = [tok.convert_tokens_to_ids(t) for t in ('yes','no')]
        assert llm.collective_rpc('install_yes_no') == [ids]
        def score(rows):
            enc = [qrr.build_input_ids(tok,qrr.DEFAULT_INSTRUCTION,r['query'],r['text'][:12288],1024) for r in rows]
            metrics['tokens'] += sum(map(len, enc))
            obs = llm.generate([{'prompt_token_ids':p} for p in enc],SamplingParams(temperature=0,max_tokens=1,
                        allowed_token_ids=ids,logprobs=2,seed=4545),use_tqdm=False)
            result = []
            for r, ob in zip(rows,obs):
                probs = ob.outputs[0].logprobs[0]; assert set(probs) == set(ids)
                value = float(probs[ids[0]].logprob-probs[ids[1]].logprob)
                result.append(round(value,4) if r['kind']=='q8b_doc' else value)
            return result
    return score


def gpu(stage, job, parts, watchdog, src=Path('/data/in'), dest=None, commit=lambda: None, bench=False):
    start = time.monotonic()
    assert stage in ('bge', 'qwen') and parts
    assert all(Path(p).name == p and (src/p).is_file() for p in parts)
    dest = dest or Path('/data/out')/job
    dest.mkdir(parents=True, exist_ok=True)
    metrics = dict(tokens=0)
    score = load_scorer(stage, metrics)
    if bench:
        rows=list(iterz(src/parts[0]))[:512]
        scoring_start = time.monotonic()
        values=score(rows); repeat=score(rows[:50])
        scoring_seconds = time.monotonic()-scoring_start
        from dwmax_diag import pearson
        correlation=pearson(values[:50],repeat)
        result=dict(stage=stage,pairs=len(rows)+len(rows[:50]),pearson=correlation,
                    tokens=metrics['tokens'],scoring_seconds=scoring_seconds,
                    tokens_per_second=metrics['tokens']/scoring_seconds,
                    pairs_per_second=(len(rows)+len(rows[:50]))/scoring_seconds,
                    max_abs_repeat=max(abs(a-b) for a,b in zip(values,repeat)),function_seconds=time.monotonic()-start)
        write(dest/'report.json',result);commit()
        assert correlation>=.999,result
        return result
    reports = []
    for part in parts:
        assert time.monotonic()-start < watchdog
        reports.append(process_part(src/part,dest/(part+'.scores.json'),score,commit))
    result = dict(stage=stage,job=job,parts=reports,function_seconds=time.monotonic()-start)
    write(dest/'report.json',result); commit(); return result


def extend(ctx, bge, q3e, d8w, fresh, scores, lengths):
    from copy import deepcopy
    from vi41 import append_last_ranks
    data = deepcopy(ctx['data']); bge = deepcopy(bge); q3e = deepcopy(q3e); d8w = deepcopy(d8w)
    for q, ds in fresh.items():
        assert not set(ds)&{int(r[0]) for r in data['pool'][q]}
        for d in ds:
            sd = str(d); idx = lengths[sd]['idx']; s = scores[q][sd]
            data['pool'][q].append([d,idx,s.get('bge_doc',-10000.)])
            if 'q8b_doc' in s: data['q8b'].setdefault(q,[]).append([d,idx,s['q8b_doc']])
            if 'bge_win' in s: bge.setdefault(q,{})[sd] = s['bge_win']
            if 'd8w' in s: d8w.setdefault(q,{})[sd] = s['d8w']
            q3e[q][sd] = s['q3e']
        data['pool'][q].sort(key=lambda r:-r[2])
        data['q8b'].setdefault(q,[]).sort(key=lambda r:-r[2])
    for source in ('dense','hyde'):
        data[source] = append_last_ranks(data[source],fresh,lengths)
    return data,bge,q3e,d8w


def sanity(out, base=None):
    from vi41c import load_zip
    from vi52 import fused, doclists, compare, load_docs
    ctx = read(TEMP/'vi50/fusion_context.json.zst')
    if base is None: base = load_zip(TEMP/'vi48/submission_vi48.zip')
    actual = doclists(base,fused(ctx['data'],ctx['bge'],read(TEMP/'vi50/d8w.json'),read(TEMP/'vi51/cosines_vi51b.json')))
    reference = load_docs(TEMP/'vi53/submission_vi53.zip')
    result = compare(actual,reference)
    result['mismatched_ids'] = [q for q in actual if actual[q] != reference[q]]
    result['reason'] = 'same vi53 inputs, stable RRF and protect; expansion disabled'
    write(out/'sanity.json',result)
    assert result['exact'] == 1200, 'STOP before ZIP: vi53 sanity mismatch'
    print('SANITY '+json.dumps(result),flush=True)
    return ctx,base,reference


def aggregate(out):
    prep = read(out/'prepare_report.json'); fresh = read(out/'new_ids.json'); q3e = read(out/'q3e_new.json')
    expected = {}
    for stage in ('bge','qwen'):
        for identity,q,d,kind in read(out/f'expected_{stage}.json'):
            assert identity not in expected; expected[identity] = (q,d,kind)
    result = {q:{str(d):dict(q3e=q3e[q][str(d)]) for d in ds} for q,ds in fresh.items()}
    seen = set()
    for p in sorted((out/'outputs').rglob('*.scores.json')):
        source = out/'modal_input'/p.name[:-len('.scores.json')]
        assert completed_part(source,p), ('incomplete scores',p)
        for r in read(p):
            identity,q,d,kind = r['identity'],str(r['qid']),str(r['doc_id']),r['kind']
            assert identity not in seen and expected[identity] == (q,int(d),kind)
            seen.add(identity)
            if d in result.get(q,{}):
                result[q][d][kind] = max(result[q][d].get(kind,-math.inf),r['score'])
    assert seen == set(expected), ('missing scores',len(set(expected)-seen))
    empty = {(str(q),str(d)) for q,d in prep['empty_pairs']}
    for q, ds in result.items():
        for d, scores in ds.items():
            assert set(scores) == ({'q3e'} if (q,d) in empty else {'q3e','bge_doc','q8b_doc','bge_win','d8w'})
    write(out/'new_scores.json',result)
    return result


def offline(base, rows, reference, rankings, data, bge, d8w, q3e, fresh):
    from vi52 import compare
    import dwmax_diag as dw
    new_counts=[]; new_ranks=[]
    sources = {k:{q:{str(r[0]):float(r[2]) for r in rs} for q,rs in data[k].items()} for k in ('pool','q8b','dense','hyde')}
    sources.update(bge_docwin=bge,d8w=d8w,q3e=q3e)
    zs={s:{'new_in':[],'old_out':[],'missing_new_in':0,'missing_old_out':0} for s in sources}
    for row in rows:
        q=str(row['id']); ds=set(row['relevant_docs']); nd=ds&set(fresh[q]); old_out=set(reference[row['id']])-ds
        new_counts.append(len(nd)); ranks={int(h[0]):i+1 for i,h in enumerate(rankings[q])}
        new_ranks.extend(ranks[d] for d in nd)
        for source, values in sources.items():
            vals=values.get(q,{})
            mean=statistics.mean(vals.values()) if vals else 0.; sd=statistics.pstdev(vals.values()) if vals else 0.
            for label, ids in (('new_in',nd),('old_out',old_out)):
                for d in ids:
                    if str(d) not in vals: zs[source]['missing_'+label]+=1
                    else: zs[source][label].append((vals[str(d)]-mean)/sd if sd else 0.)
    return dict(vs_vi53=compare({r['id']:r['relevant_docs'] for r in rows},reference),
                new_docs_per_query=dw.dist(new_counts),zero_new_queries=sum(n==0 for n in new_counts),
                new_final_ranks=dw.dist(new_ranks),
                source_z={s:{k:dw.dist(v) if isinstance(v,list) else v for k,v in ob.items()} for s,ob in zs.items()})


def build(out):
    from vi41c import load_zip,digest
    from vi52 import fused
    from vi50 import protect
    from vi29 import assert_clean,load_corpus_ids
    from reorder_zip import write_reordered_zip
    start=time.monotonic(); zp=out/'submission_vi56.zip'; jp=out/'submission_vi56.json'; intent=out/'build_intent.json'
    assert not zp.exists() and not intent.exists() and not jp.exists(), 'ONE build only'
    before = read(out/'input_audit.json')['old_inputs']
    assert all(fingerprint(Path(p))==meta for p,meta in before.items()), 'Historical input changed'
    ctx,base,reference=sanity(out)
    fresh=read(out/'new_ids.json'); prep=read(out/'prepare_report.json'); scores=aggregate(out)
    data,bge,q3e,d8w=extend(ctx,ctx['bge'],read(TEMP/'vi51/cosines_vi51b.json'),read(TEMP/'vi50/d8w.json'),fresh,scores,prep['lengths'])
    rankings=fused(data,bge,d8w,q3e)
    rows=[protect(r,rankings[str(r['id'])])[0] for r in base]
    write(out/'offline.json',offline(base,rows,reference,rankings,data,bge,d8w,q3e,fresh))
    source=TEMP/'vi45'; bad=set(map(int,read(source/'bad_ids.json')))-set(map(int,read(source/'new_ids.json')))
    clean=assert_clean(rows,bad,load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    chunks={r['id']:digest(r['relevant_chunks']) for r in base}; counts={r['id']:len(r['relevant_docs']) for r in base}
    assert all(digest(r['relevant_chunks'])==chunks[r['id']] and len(r['relevant_docs'])==counts[r['id']] for r in rows)
    assert all(len(set(r['relevant_docs']))==len(r['relevant_docs']) for r in rows)
    hashes={r['id']:digest(r) for r in rows}
    del ctx,base,data,bge,q3e,d8w,scores,rankings; gc.collect()
    with intent.open('x',encoding='utf-8') as f: f.write('{"build_attempts":1}')
    info=write_reordered_zip(rows,zp); assert zp.stat().st_size<=99_000_000
    del rows; gc.collect()
    try:
        with zipfile.ZipFile(zp) as z:
            assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
            assert z.testzip() is None
            with z.open('submission.json') as src,jp.open('xb') as dst: shutil.copyfileobj(src,dst)
        final=load_zip(zp)
        assert {r['id']:digest(r) for r in final}==hashes and set(hashes)==set(range(1,1201))
        assert all(digest(r['relevant_chunks'])==chunks[r['id']] and len(r['relevant_docs'])==counts[r['id']] for r in final)
        finalclean=assert_clean(final,bad,load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
        assert finalclean==clean
        del final; gc.collect()
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'validate',str(jp)],check=True)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'check-zip','--out',str(out)],check=True)
        assert all(fingerprint(Path(p))==meta for p,meta in before.items()), 'Historical input changed'
        report=dict(**info,**fingerprint(zp),validate=0,check_zip=0,assert_clean=clean,crc=0,ids=1200,
                    chunks_equal_vi48=1200,doc_counts_equal_vi48=1200,build_attempts=1,
                    inputs_unchanged=True,no_duplicate_docs=True,entry='submission.json',compression='LZMA',
                    seconds=time.monotonic()-start)
        write(out/'build_report.json',report); print('BUILD_DONE '+json.dumps(report),flush=True)
    finally:
        if jp.exists(): jp.unlink()


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['select','prepare','sanity','build','check-zip','gpu'])
    ap.add_argument('--out',type=Path,default=TEMP/'vi56'); ap.add_argument('--root',type=Path,default=Path('D:/r2ai'))
    ap.add_argument('--kind',choices=['bge','qwen']);ap.add_argument('--job');ap.add_argument('--parts',nargs='*',default=[])
    ap.add_argument('--watchdog',type=int,default=240);ap.add_argument('--bench',action='store_true')
    ap.add_argument('--src',type=Path,default=Path('/data/in'))
    a=ap.parse_args(argv); a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='gpu':
        # CLI is deliberately SDK-free. The mounted Modal parent owns commits.
        gpu(a.kind,a.job,a.parts,a.watchdog,src=a.src,dest=a.out,bench=a.bench)
    elif a.stage=='select': local_select(a.out)
    elif a.stage=='prepare':
        from vi39 import wait_lock
        with wait_lock(a.root/'locks/heavy_job.lock',f'vi56 prepare pid={os.getpid()}'): prepare(a.root,a.out)
    elif a.stage=='sanity': sanity(a.out)
    elif a.stage=='build': build(a.out)
    else:
        from vi23 import check_zip
        raise SystemExit(check_zip(str(a.out/'submission_vi56.json'),str(a.out/'submission_vi56.zip'),str(a.out/'check_zip.json'),'vi56',max_zip=99_000_000))


if __name__=='__main__': main()
