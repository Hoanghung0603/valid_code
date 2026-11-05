"""vi54 Nemotron: immutable source provenance and fixed dense z-score ensemble."""
import argparse
import json
import math
import hashlib
import time
from collections import defaultdict
from pathlib import Path

MODEL = 'nvidia/Nemotron-3-Embed-8B-BF16'
REVISION = 'd1f2f25730bbd775b99b29185134bc86653bf2d1'
RELEASE = '2026-07-16'


def query_text(text):
    return 'query: ' + text


def document_text(title, text):
    return 'passage: ' + title + '\n' + text


def mean_z(pool, *sources):
    """Population z-score per query over available pool values; missing -> other."""
    result = {}
    for q, hits in pool.items():
        ids = [str(h[0]) for h in hits]
        normalized = []
        for source in sources:
            values = {d: float(source.get(q, {})[d]) for d in ids if d in source.get(q, {})}
            assert all(math.isfinite(v) for v in values.values())
            if not values:
                normalized.append({})
                continue
            mean = sum(values.values()) / len(values)
            std = math.sqrt(sum((v-mean)**2 for v in values.values()) / len(values))
            normalized.append({d: (v-mean)/std if std else 0. for d, v in values.items()})
        result[q] = {}
        for d in ids:
            available = [s[d] for s in normalized if d in s]
            assert available, (q, d, 'both cosines missing')
            result[q][d] = sum(available)/len(available)
    return result


def require_raw(rows):
    """Qwen token IDs truncated to 2048 cannot represent original Nemotron input."""
    for row in rows:
        if not isinstance(row.get('text'), str) or not row['text'].strip() or 'input_ids' in row:
            raise ValueError('INPUT_PROVENANCE: vi51 parts contain only Qwen input_ids; original title+clean text required for exact Nemotron 2048-token truncation; decoding truncated Qwen tokens is NOT equivalent')
    return rows


def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())


def iterz(path):
    import zstandard as zstd
    with zstd.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')


def tokenize(tokenizer, rows, query=False):
    require_raw(rows)
    texts=[query_text(r['text']) if query else 'passage: '+r['text'] for r in rows]
    result=[]
    for start in range(0,len(rows),128):
        ids=tokenizer(texts[start:start+128],truncation=True,max_length=256 if query else 2048)['input_ids']
        result.extend(dict(id=r['id'],input_ids=i) for r,i in zip(rows[start:start+128],ids))
    assert all(r['input_ids'] for r in result)
    return result


def token_batches(rows, budget):
    batch=[]
    for i in sorted(range(len(rows)),key=lambda i:(len(rows[i]['input_ids']),i)):
        if batch and len(rows[i]['input_ids'])*(len(batch)+1)>budget:
            yield batch;batch=[]
        batch.append(i)
    if batch:yield batch


def gpu(mode, source, dest, parts, watchdog):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    import zstandard as zstd
    start=time.monotonic();dest.mkdir(parents=True,exist_ok=True)
    assert json.loads(Path('/weights/complete.json').read_text())['revision']==REVISION
    tok=AutoTokenizer.from_pretrained('/weights/model',padding_side='left',local_files_only=True)
    model=AutoModel.from_pretrained('/weights/model',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    assert model.config.is_causal is False
    budget=16384
    def embed(rows):
        e=np.empty((len(rows),model.config.hidden_size),dtype=np.float32)
        for indices in token_batches(rows,budget):
            assert time.monotonic()-start<watchdog-5
            batch=tok.pad([dict(input_ids=rows[i]['input_ids']) for i in indices],padding=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                # Transformers5.2 Ministral attention hardcodes module.is_causal=True;
                # explicit kwarg honors this encoder checkpoint also when mask=None.
                hidden=model(**batch,is_causal=False,use_cache=False).last_hidden_state.float()
                mask=batch['attention_mask'].unsqueeze(-1).float()
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                vectors=torch.nn.functional.normalize(pooled,p=2,dim=1)
            e[indices]=vectors.cpu().numpy()
            del batch,hidden,mask,pooled,vectors
        assert np.isfinite(e).all() and np.max(np.abs(np.linalg.norm(e,axis=1)-1))<2e-4
        return e
    common=dict(model=MODEL,revision=REVISION,release=RELEASE,backend='transformers+sdpa',
                dtype='bf16',pooling='attention-mask mean/L2; left',torch=str(torch.__version__),
                transformers=transformers.__version__,token_budget=budget,hidden_size=model.config.hidden_size,
                is_causal=False,use_cache=False)
    if mode=='bench':
        rows=tokenize(tok,list(iterz(source/'docs54.part000.jsonl.zst'))[:2000])
        embed(rows[:4]);t=time.monotonic();e=embed(rows);seconds=time.monotonic()-t
        repeat=embed(rows[:50]);cos=np.sum(e[:50]*repeat,axis=1)
        report=dict(**common,documents=len(rows),tokens=sum(len(r['input_ids']) for r in rows),seconds=seconds,
                    tokens_per_second=sum(len(r['input_ids']) for r in rows)/seconds,
                    repeat_cosine_min=float(cos.min()),repeat_cosine_mean=float(cos.mean()),repeat_cosine_max=float(cos.max()))
        # Persist numbers before ANY repeat/sanity assertions.
        write(dest/'bench_report.json',report);print('REPEAT_REPORT '+json.dumps(report),flush=True)
        qs=json.loads((source/'queries54.json').read_text())[:5]
        qe=embed(tokenize(tok,qs,True));sim=qe@e.T
        report['sanity_top10']=[dict(query_id=q['id'],query=q['text'],doc_ids=[rows[j]['id'] for j in np.argsort(-sim[i])[:10]],
            cosines=sorted(map(float,sim[i]),reverse=True)[:10],std=float(np.std(sim[i]))) for i,q in enumerate(qs)]
        report['sanity_pass']=bool(np.isfinite(sim).all() and np.all(np.std(sim,axis=1)>1e-6))
        np.savez(dest/'bench_embeddings.npz',ids=np.asarray([r['id'] for r in rows]),emb=e)
        report['function_seconds']=time.monotonic()-start
        write(dest/'bench_report.json',report);print('BENCH_REPORT '+json.dumps(report),flush=True)
        assert report['repeat_cosine_min']>=.999 and report['repeat_cosine_mean']>=.9999,report
        assert report['sanity_pass'],report
        return report
    qs=json.loads((source/'queries54.json').read_text());qe=embed(tokenize(tok,qs,True))
    np.savez(dest/'query_embeddings.npz',ids=np.asarray([r['id'] for r in qs]),emb=qe)
    pool=json.loads((source/'pool.json').read_text());reverse=defaultdict(list)
    for qi,q in enumerate(qs):
        for d in pool[str(q['id'])]:reverse[d].append(qi)
    reports=[];seen=set()
    for part in parts:
        t=time.monotonic();raw=list(iterz(source/part));rows=tokenize(tok,raw)
        assert not seen.intersection(r['id'] for r in rows)
        e=embed(rows);scores={str(q['id']):{} for q in qs}
        np.savez(dest/(part+'.emb.npz'),ids=np.asarray([r['id'] for r in rows]),emb=e.astype(np.float16))
        for i,r in enumerate(rows):
            d=r['id'];seen.add(d)
            for qi in reverse[d]:scores[str(qs[qi]['id'])][str(d)]=float(e[i]@qe[qi])
        with zstd.open(dest/(part+'.cosines.json.zst'),'wt',encoding='utf-8') as f:json.dump(scores,f)
        reports.append(dict(part=part,documents=len(rows),tokens=sum(len(r['input_ids']) for r in rows),
                            pairs=sum(map(len,scores.values())),seconds=time.monotonic()-t))
        write(dest/'partial_report.json',dict(parts=reports,function_seconds=time.monotonic()-start))
        print('PART_DONE '+json.dumps(reports[-1]),flush=True)
    report=dict(**common,parts=reports,documents=len(seen),pairs=sum(r['pairs'] for r in reports),
                function_seconds=time.monotonic()-start,files={p.name:fingerprint(p) for p in dest.iterdir() if p.is_file()})
    write(dest/'bulk_report.json',report);return report


def fused(data, bge, d8w, q3e, nemotron):
    from vi52 import fused as original
    return original(data, bge, d8w, mean_z(data['pool'], q3e, nemotron))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['gpu'])
    parser.add_argument('--mode', choices=['bench', 'bulk'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watchdog', type=int, required=True)
    parser.add_argument('--parts', nargs='*', default=[])
    args = parser.parse_args()
    return gpu(args.mode,args.source,args.out,args.parts,args.watchdog)


if __name__ == '__main__':
    main()
