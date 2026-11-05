"""vi41: isolated query expansion and expanded-pool vi37. Never submits."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SEED = 4141
K_PER_SOURCE = 20
MAX_DELTA = 55
REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
ROOT = Path(os.environ.get('R2AI_ROOT', 'D:/r2ai'))
OUT = ROOT/'data/sub/vi41'
NQ_FROM = None


def fixed_doclists(fused, reference, ref_chunks):
    """Keep reference counts and mandatory ref-chunk docs; fill by unchanged RRF.

    Appended mandatory docs retain the historical full8b order. If they overflow
    the fixed count, evict the lowest-ranked nonmandatory base docs, never grow n_q.
    """
    import vi23
    counts = {str(row['id']): len(row['relevant_docs']) for row in reference}
    assert len(counts) == len(reference) and set(counts) == set(fused)
    result = {}
    for q, hits in fused.items():
        n = counts[q]
        mandatory = {str(c['doc_id']) for c in ref_chunks.get(q, [])}
        assert len(mandatory) <= n, (q, 'reference count smaller than mandatory docs')
        docs = vi23.full8b_docs(vi23.first_n_docs(hits, n), ref_chunks.get(q, []))
        excess = len(docs)-n
        removed = set()
        for d in reversed(docs):
            if excess <= 0:
                break
            if str(d) not in mandatory:
                removed.add(str(d))
                excess -= 1
        docs = [d for d in docs if str(d) not in removed]
        assert len(docs) == n and len(set(map(str, docs))) == n, (q, n)
        result[q] = docs
    return result


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parse_rewrite(text):
    """Strict schema: failures are visible, never silently fabricated/fallback."""
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    row = json.loads(text)
    assert set(row) == {'subqueries', 'hyde'}
    assert isinstance(row['subqueries'], list) and len(row['subqueries']) == 4
    assert isinstance(row['hyde'], list) and len(row['hyde']) == 3
    for key, cap in [('subqueries', 200), ('hyde', 2500)]:
        assert all(isinstance(t, str) and 4 <= len(t.strip()) <= cap for t in row[key]), key
        assert len(set(t.strip().casefold() for t in row[key])) == len(row[key]), key
        row[key] = [t.strip() for t in row[key]]
    return row


def remove_trailing_commas(text):
    """Syntax-only repair outside JSON strings; does not generate/change any text."""
    out=[]
    quoted=escaped=False
    for i,c in enumerate(text):
        if quoted:
            out.append(c)
            if escaped:
                escaped=False
            elif c=='\\':
                escaped=True
            elif c=='"':
                quoted=False
        elif c=='"':
            quoted=True
            out.append(c)
        elif c==',' and text[i+1:].lstrip().startswith((']', '}')):
            continue
        else:
            out.append(c)
    return ''.join(out)


def repair_local(directory):
    directory=Path(directory)
    rows=[json.loads(l) for l in (directory/'rewrites.jsonl').read_text(encoding='utf-8').splitlines()]
    errors=read(directory/'rewrite_errors.json')
    queries={int(r['id']):r['query'] for r in [json.loads(l) for l in
        Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi37/modal_input/queries.jsonl').read_text(encoding='utf-8').splitlines()]}
    for error in errors:
        parsed=parse_rewrite(remove_trailing_commas(error['raw']))
        rows.append(dict(id=int(error['id']),query=queries[int(error['id'])],**parsed))
    assert len(rows)==1200 and {int(r['id']) for r in rows}==set(range(1,1201))
    rows.sort(key=lambda r:int(r['id']))
    target=directory/'rewrites_complete.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    for row in rows:
        parse_rewrite(json.dumps({k:row[k] for k in ('subqueries','hyde')},ensure_ascii=False))
    write(directory/'rewrite_repair_report.json',dict(queries=1200,subqueries=4800,hyde=3600,
        repaired_syntax_only=[e['id'] for e in errors],sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        generated_new_text=False,raw_report=read(directory/'rewrite_report.json')))


def rrf_lists(lists, k=60):
    """Deduplicate inside each ranking; deterministic numeric-ID tie break."""
    scores = {}
    for rows in lists:
        seen = set()
        for r in rows:
            d = int(r[0])
            if d in seen:
                continue
            seen.add(d)
            scores[d] = scores.get(d, 0.) + 1/(k+len(seen))
    return sorted(scores.items(), key=lambda r: (-r[1], r[0]))


def select_new(source_rankings, pool, allowed, bad, k=K_PER_SOURCE):
    """Top-K outside vi37 per source; union retains source provenance."""
    selected, provenance = {}, {}
    for q, old in pool.items():
        have = {int(r[0]) for r in old}
        keep, why = [], {}
        for source, cache in source_rankings.items():
            fresh = [int(r[0]) for r in cache.get(q, [])
                     if int(r[0]) not in have and int(r[0]) in allowed and int(r[0]) not in bad]
            fresh = list(dict.fromkeys(fresh))[:k]
            for d in fresh:
                if str(d) not in why:
                    keep.append(d)
                    why[str(d)] = []
                why[str(d)].append(source)
        selected[q], provenance[q] = keep, why
    return selected, provenance


def rewrite_gpu(root):
    from vllm import LLM, SamplingParams
    start = time.monotonic()
    src, dst = root/'in/vi41', root/'out/vi41'
    dst.mkdir(parents=True, exist_ok=True)
    queries = [json.loads(l) for l in (src/'queries.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(queries) == 1200 and {int(q['id']) for q in queries} == set(range(1,1201))
    llm = LLM(model='Qwen/Qwen3-14B', revision=REVISION, dtype='bfloat16',
              max_model_len=3072, gpu_memory_utilization=.88, seed=SEED,
              enable_prefix_caching=True, enforce_eager=True)
    tok = llm.get_tokenizer()
    instruction = ('Bạn viết lại câu hỏi để tìm tài liệu y khoa tiếng Việt. '
        'Chỉ xuất một đối tượng JSON với đúng hai khóa: "subqueries" là 4 truy vấn ngắn '
        '(mỗi truy vấn 5–18 từ, tách bệnh/thuốc/triệu chứng/thủ thuật hoặc các lõi thực có); '
        '"hyde" là 3 đoạn văn khác nhau theo văn phong bài báo y khoa, mỗi đoạn 45–70 từ. '
        'Giữ thực thể và bối cảnh, không thêm bệnh/thuốc không được gợi ý. '
        'Không lời mở đầu, không markdown, không suy luận. Mọi chuỗi bằng tiếng Việt.')
    prompts = [tok.apply_chat_template([{'role':'system','content':instruction},
        {'role':'user','content':q['query']}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False) for q in queries]
    params = SamplingParams(temperature=0., seed=SEED, max_tokens=1100)
    target = dst/'rewrites.jsonl'
    assert not target.exists(), 'Existing rewrite output: inspect; no automatic retry'
    errors, generated, input_tokens, output_tokens = [], 0, 0, 0
    with target.open('w', encoding='utf-8') as stream:
        for i in range(0,1200,40):
            assert time.monotonic()-start < 940, 'VI41 rewrite budget watchdog'
            outputs = llm.generate(prompts[i:i+40], params, use_tqdm=False)
            for q, ob in zip(queries[i:i+40], outputs):
                raw = ob.outputs[0].text
                input_tokens += len(ob.prompt_token_ids)
                output_tokens += len(ob.outputs[0].token_ids)
                try:
                    parsed = parse_rewrite(raw)
                except (ValueError, AssertionError) as exc:
                    errors.append({'id':q['id'], 'error':repr(exc), 'raw':raw})
                    continue
                stream.write(json.dumps({'id':q['id'], 'query':q['query'], **parsed}, ensure_ascii=False)+'\n')
                generated += 1
            stream.flush()
            import modal
            modal.Volume.from_name('r2ai-runner').commit()
    write(dst/'rewrite_errors.json', errors)
    write(dst/'rewrite_report.json', dict(queries=generated, subqueries=generated*4, hyde=generated*3,
        input_tokens=input_tokens, output_tokens=output_tokens, errors=len(errors), seed=SEED,
        temperature=0., revision=REVISION, seconds=time.monotonic()-start,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    assert not errors, f'STOP rewrite invalid: {len(errors)}; inspect raw, never automatic regenerate'


def bm25():
    import numpy as np
    from bigindex import BigIndex
    from bm25 import tokenize
    import dwmax_diag as dw
    idx = BigIndex(ROOT/'data/index/vi36')
    dw.safe_index(ROOT/'data/index/vi36')
    rows = [json.loads(l) for l in (OUT/'rewrites.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1200
    dest = OUT/'bm25_parts'
    dest.mkdir(parents=True, exist_ok=True)
    # Immutable 5.27 GB postings fit in b660's measured >20 GB free RAM.
    # Sequential copies avoid random page-fault thrashing for 8,400 queries.
    idx.docs=np.array(idx.docs,copy=True)
    idx.scores=np.array(idx.scores,copy=True)
    print('BM25 postings in RAM',idx.docs.nbytes+idx.scores.nbytes,flush=True)
    result = {}
    for n, row in enumerate(rows):
        q = str(row['id'])
        part = dest/(q+'.json')
        if part.exists():
            result[q] = read(part)
            continue
        lists = []
        for text in row['subqueries']+row['hyde']:
            hits = idx.search(tokenize(text, 'vi'), 200)
            lists.append([[idx.doc_id(i), score] for i, score in hits])
        result[q] = [[d, s] for d, s in rrf_lists(lists)]
        write(part, result[q])
        if n % 50 == 0:
            print('BM25', n, flush=True)
    write(OUT/'bm25_rrf.json', result)
    write(OUT/'bm25_report.json', {'queries':len(result), 'variants_per_query':7,
        'pairs':sum(map(len,result.values()))})


def inspect():
    import numpy as np
    paths = [ROOT/'work/runner/rsmoke20/emb_clean', ROOT/'work/vi36/emb']
    result = []
    for directory in paths:
        for p in sorted(directory.glob('blk_*.npz')):
            with np.load(p) as z:
                result.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                   rows=len(z['ids']),shape=list(z['emb'].shape)))
    write(ROOT/'vi41_dense_inventory.json', result)
    print(json.dumps({'files':len(result),'rows':sum(x['rows'] for x in result),'bytes':sum(x['bytes'] for x in result)}))


def dense():
    """Read-only original vi36 embeddings on z87; latency watchdog every GPU block."""
    import numpy as np
    import subprocess
    import torch
    from transformers import AutoModel, AutoTokenizer
    from dense_topk import load_embeddings
    torch.set_num_threads(2)
    start = time.monotonic()
    def latency():
        r = subprocess.run(['curl.exe','-s','-o','NUL','-w','%{http_code} %{time_total}',
                            'https://www.vinmec.com/'], capture_output=True, text=True, timeout=15)
        code, seconds = r.stdout.strip().split()
        assert r.returncode == 0 and code in ('200','301','302','308') and float(seconds) <= 1., ('STOP z87 latency',r.stdout)
    latency()
    rows = [json.loads(l) for l in (OUT/'rewrites.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1200
    variants = [(str(r['id']),t) for r in rows for t in r['subqueries']+r['hyde']]
    tok = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    model = AutoModel.from_pretrained('BAAI/bge-m3', torch_dtype=torch.float16).cuda().eval()
    qe = np.zeros((len(variants),1024),dtype=np.float16)
    with torch.inference_mode():
        for i in range(0,len(variants),8):
            if i % 400 == 0:
                latency()
            x = tok([t for _,t in variants[i:i+8]],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            h = model(**x).last_hidden_state[:,0]
            qe[i:i+8] = torch.nn.functional.normalize(h.float(),dim=-1).half().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    np.savez_compressed(OUT/'variant_embeddings.npz',emb=qe,qids=np.asarray([int(q) for q,t in variants]))
    ids, emb = load_embeddings([ROOT/'work/runner/rsmoke20/emb_clean', ROOT/'work/vi36/emb'])
    assert len(ids) == 739002
    assert np.isfinite(emb).all() and np.isfinite(qe).all()
    Q = torch.from_numpy(qe).cuda().float()
    top_s = torch.full((len(qe),200),-float('inf'),device='cuda')
    top_d = torch.full((len(qe),200),-1,device='cuda',dtype=torch.int64)
    torch.backends.cuda.matmul.allow_tf32 = False
    for i in range(0,len(ids),8192):
        latency()
        E = torch.from_numpy(emb[i:i+8192]).cuda().float()
        D = torch.from_numpy(ids[i:i+8192]).cuda()
        for j in range(0,len(qe),100):
            with torch.inference_mode():
                score = Q[j:j+100] @ E.T
                vals, ix = score.topk(min(200,len(E)),dim=1)
                scores = torch.cat([top_s[j:j+100],vals],dim=1)
                docs = torch.cat([top_d[j:j+100],D[ix]],dim=1)
                vals, ix = scores.topk(200,dim=1)
                top_s[j:j+100] = vals
                top_d[j:j+100] = docs.gather(1,ix)
        del E, D
    sd, ss = top_d.cpu().numpy(),top_s.cpu().numpy()
    assert np.isfinite(ss).all() and (sd>=0).all()
    result = {}
    for i, row in enumerate(rows):
        lists = [[[int(d),float(s)] for d,s in zip(sd[j],ss[j])]
                 for j in range(i*7,i*7+7)]
        result[str(row['id'])] = [[d,s] for d,s in rrf_lists(lists)]
    write(OUT/'dense_rrf.json',result)
    write(OUT/'dense_report.json',dict(queries=1200,variants=8400,index_documents=len(ids),seconds=time.monotonic()-start,
                                    variant_embeddings_sha256=hashlib.sha256((OUT/'variant_embeddings.npz').read_bytes()).hexdigest()))


def candidates():
    import dwmax_diag as dw
    import vi29
    sources = {s:read(OUT/(s+'_rrf.json')) for s in ('dense','dchunk','bm25','neardup')}
    pool = dw.read(ROOT/'data/sub/vi36/in1/pool_v36.json')
    corpus = vi29.load_corpus_ids(ROOT/'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    bad = set(dw.read(ROOT/'data/sub/diag28/bad_ids.json'))-set(dw.read(ROOT/'data/sub/diag30/new_ids.json'))
    selected, provenance = select_new(sources,pool,corpus,bad)
    write(OUT/'new_ids.json',selected)
    write(OUT/'new_provenance.json',provenance)
    counts = {s:sum(s in p for q,ds in provenance.items() for p in ds.values())/1200 for s in sources}
    write(OUT/'candidate_report.json',dict(k_per_source=K_PER_SOURCE,new_pairs=sum(map(len,selected.values())),
        new_per_query=sum(map(len,selected.values()))/1200,by_source_per_query=counts,
        source_sha256={s:hashlib.sha256((OUT/(s+'_rrf.json')).read_bytes()).hexdigest() for s in sources},
        search_npz_sha256=hashlib.sha256((OUT/'dchunk_variants.npz').read_bytes()).hexdigest()))
    import vi37
    wanted = {d for ds in selected.values() for d in ds}
    lengths,_,files = vi37.export_texts(wanted,OUT/'modal_input','docs_new')
    cands = {q:[[d,lengths[str(d)]['idx']] for d in ds] for q,ds in selected.items() if ds}
    write(OUT/'modal_input/new_cands.json',cands)
    write(OUT/'export_lengths.json',lengths)
    (OUT/'modal_input/queries.jsonl').write_bytes((ROOT/'data/sub/vi37/modal_input/queries.jsonl').read_bytes())
    write(OUT/'export_report.json',dict(files=[str(p) for p in files],pairs=sum(map(len,cands.values())),
        windows=sum(lengths[str(d)]['windows'] for ds in selected.values() for d in ds)))
    manifest()
    audit_inputs()


def manifest():
    files=sorted((OUT/'modal_input').iterdir())
    rows=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
          for p in files if p.is_file()]
    write(OUT/'transfer_manifest.json',dict(files=rows))


def dchunk():
    import numpy as np
    import pyarrow.parquet as pq
    from dchunk import collapse
    z = np.load(OUT/'dchunk_variants.npz')
    assert z['chunk_ids'].shape == z['scores'].shape == (8400,1000)
    assert z['qids'].shape==(8400,)
    assert set(int(q) for q in z['qids'])==set(range(1,1201))
    assert np.isfinite(z['scores']).all()
    metadata = {}
    for p in sorted((ROOT/'data/sub/dchunk').glob('chunks_*.parquet')):
        for row in pq.read_table(p,columns=['chunk_id','doc_id','idx']).to_pylist():
            metadata[row['chunk_id']] = row
    grouped = {}
    for q,cs,ss in zip(z['qids'],z['chunk_ids'],z['scores']):
        grouped.setdefault(str(int(q)),[]).append(collapse(cs,ss,metadata)[:200])
    assert len(grouped)==1200 and all(len(rows)==7 for rows in grouped.values())
    write(OUT/'dchunk_rrf.json',{q:[[d,s] for d,s in rrf_lists(rows)] for q,rows in grouped.items()})
    write(OUT/'dchunk_report.json',dict(queries=1200,variants=8400,metadata_chunks=len(metadata),
        min_distinct_docs=min(len(collapse(cs,ss,metadata)) for cs,ss in zip(z['chunk_ids'],z['scores']))))


def shingle_hashes(text, width=5):
    import re
    import unicodedata
    import zlib
    words = re.findall(r'[^\W_]+',unicodedata.normalize('NFKC',text).casefold())
    return {zlib.crc32(' '.join(words[i:i+width]).encode('utf-8')) for i in range(max(0,len(words)-width+1))}


def neardup():
    """Bottom-16 shingle hashes shortlist; verified >=.8 exact hashed-shingle Jaccard.

    Anchors are top30 vi37. This is approximate candidate generation, not proof of all duplicates.
    """
    from collections import Counter, defaultdict
    import heapq
    import pyarrow.parquet as pq
    import dwmax_diag as dw
    start=time.monotonic()
    tops=dw.read(ROOT/'data/sub/vi37/cache_fhyde_v37.json')
    anchors={int(r[0]) for rows in tops.values() for r in rows[:30]}
    directory=ROOT/'data/sub/dchunk'
    signatures={}
    for p in sorted(directory.glob('docs_*.parquet')):
        for row in pq.read_table(p).to_pylist():
            if row['doc_id'] in anchors:
                signatures[row['doc_id']]=shingle_hashes(row['text'])
    reverse=defaultdict(list)
    for d,sig in signatures.items():
        if len(sig)>=20:
            for h in heapq.nsmallest(16,sig):
                reverse[h].append(d)
    neighbors=defaultdict(list)
    compared=0
    for p in sorted(directory.glob('docs_*.parquet')):
        for row in pq.read_table(p).to_pylist():
            d=row['doc_id']
            sig=shingle_hashes(row['text'])
            if len(sig)<20:
                continue
            matches=Counter(a for h in heapq.nsmallest(16,sig) for a in reverse.get(h,[]) if a!=d)
            for a,n in matches.items():
                if n<3:
                    continue
                other=signatures[a]
                if min(len(sig),len(other))/max(len(sig),len(other))<.8:
                    continue
                compared+=1
                jaccard=len(sig&other)/len(sig|other)
                if jaccard>=.8:
                    neighbors[a].append((d,jaccard))
        print('NEARDUP',p.name,'edges',sum(map(len,neighbors.values())),flush=True)
    result={}
    for q,rows in tops.items():
        scores={}
        for rank,r in enumerate(rows[:30],1):
            for d,j in neighbors[int(r[0])]:
                scores[d]=max(scores.get(d,0),j/(60+rank))
        result[q]=sorted([[d,s] for d,s in scores.items()],key=lambda r:(-r[1],r[0]))
    write(OUT/'neardup_rrf.json',result)
    write(OUT/'neardup_report.json',dict(anchors=len(anchors),anchors_with_text=len(signatures),
        verified_pairs=sum(map(len,neighbors.values())),compared_pairs=compared,threshold=.8,
        width=5,bottom_hashes=16,min_matches=3,seconds=time.monotonic()-start,
        note='Approximate bottom-k hashed shingles; candidate misses possible; no recall guarantee'))


def sanity():
    """Run vi41's unchanged vi37 fusion with no new candidates; exact doclist check."""
    import dwmax_diag as dw
    import vi37
    data=vi37.inputs()
    data['q8b']=dw.read(ROOT/'data/sub/vi37/q8b_v37.json')
    scores=dw.read(ROOT/'data/sub/diag37/docwin_max_v37.json')
    fh,dl=vi37.fused_doclist(data,scores)
    old={str(it['id']):it['relevant_docs'] for it in dw.read(ROOT/'data/sub/vi37/submission_vi37.json')}
    changed=[q for q in old if dl.get(q)!=old[q]]
    write(OUT/'sanity_report.json',dict(queries=len(old),doclist_equal=len(old)-len(changed),
        changed=changed,expanded_pool=False,method='same fused_doclist as vi41 merge'))
    assert len(old)==1200 and not changed,changed[:10]


def sanity_chain():
    """Full unexpanded vi41 build chain, including unchanged candidate chunks, no ZIP."""
    import dwmax_diag as dw
    import vi29
    import vi37
    data=vi37.inputs()
    data['q8b']=dw.read(ROOT/'data/sub/vi37/q8b_v37.json')
    scores=dw.read(ROOT/'data/sub/diag37/docwin_max_v37.json')
    candidates=dw.read(ROOT/'data/sub/vi37/cands_v37.json.zst')
    ref={str(it['id']):it['relevant_chunks'] for it in dw.read(ROOT/'data/sub/vi36/in1/ref_chunks_v36.json')}
    kq={str(it['id']):len(it['relevant_chunks']) for it in dw.read(vi29.REF_K)}
    with vi37.chain_docwin(scores):
        _,_,sub,_=vi29.chain_build_v29(data['pool'],data['q8b'],data['dense'],data['hyde'],candidates,kq,ref,
                                     delta=MAX_DELTA,dedup_mode='doc_text')
    old={str(it['id']):it for it in dw.read(ROOT/'data/sub/vi37/submission_vi37.json')}
    doc_equal=sum(it['relevant_docs']==old[str(it['id'])]['relevant_docs'] for it in sub)
    chunk_equal=sum(it['relevant_chunks']==old[str(it['id'])]['relevant_chunks'] for it in sub)
    report=dict(queries=len(sub),doclist_equal=doc_equal,chunks_equal=chunk_equal,expanded_pool=False,
                method='full vi29 chain, same vi41 build input/ref/delta55 and vi37 docwin hook')
    write(OUT/'sanity_chain_report.json',report)
    assert len(sub)==doc_equal==chunk_equal==1200,report


def append_last_ranks(old, new, indices):
    """Authorized fallback: historical cosine missing => last ranks, old order unchanged."""
    result={q:[list(r) for r in rows] for q,rows in old.items()}
    for q,ds in new.items():
        rows=result.setdefault(q,[])
        seen={int(r[0]) for r in rows}
        floor=min([float(r[2]) for r in rows]+[0.])-1.
        rows.extend([d,indices[str(d)]['idx'],floor-i*.000001]
                    for i,d in enumerate(ds) if d not in seen)
    return result


def merged_inputs():
    import dwmax_diag as dw
    import vi37
    data=vi37.inputs()
    rr=dw.read(OUT/'modal_out/rr_new.json')
    qpath=OUT/'modal_out/q8b_new.json'
    q8=dw.read(qpath) if qpath.exists() else {}
    expected=dw.read(OUT/'modal_input/new_cands.json')
    lengths=dw.read(OUT/'export_lengths.json')
    assert set(rr)==set(expected)
    # Empty text is a real no-evidence score, not an absent value passed into fusion.
    for rows in rr.values():
        for r in rows:
            if not lengths[str(r[0])]['windows']:
                r[2]=-10000.
            else:
                assert r[2] is not None, ('missing nonempty bge',r[0])
    vi37.checked_q8b(rr,expected)
    for q in rr:
        rr[q]=sorted(rr[q],key=lambda r:-r[2])
    if qpath.exists():
        vi37.checked_q8b(q8,dw.read(OUT/'modal_input/q8b_cands.json'))
    for q, rows in rr.items():
        assert not {int(r[0]) for r in rows}&{int(r[0]) for r in data['pool'][q]}
        data['pool'][q]=sorted(data['pool'][q]+rows,key=lambda r:-r[2])
    oldq=dw.read(ROOT/'data/sub/vi37/q8b_v37.json')
    for q,rows in q8.items():
        assert not {int(r[0]) for r in rows}&{int(r[0]) for r in oldq.get(q,[])}
        oldq[q]=sorted(oldq.get(q,[])+rows,key=lambda r:-r[2])
    data['q8b']=oldq
    # Authorized simple fallback: no invented historical cosine; old source order unchanged.
    for source in ('dense','hyde'):
        data[source]=append_last_ranks(data[source],read(OUT/'new_ids.json'),lengths)
    scores=dw.read(ROOT/'data/sub/diag37/docwin_max_v37.json')
    wins=dw.read(OUT/'modal_out/docwin_new.json.zst')
    for q,rows in expected.items():
        for d,i in rows:
            assert not lengths[str(d)]['windows'] or str(d) in wins.get(q,{}),(q,d,'missing docwin')
    assert sum(len(ws) for ds in wins.values() for ws in ds.values())==read(OUT/'export_report.json')['windows']
    for q,ds in dw.maxima(wins).items():
        scores.setdefault(q,{}).update(ds)
    return data,scores,wins


def qselect():
    import vi37
    data,scores,wins=merged_inputs()
    fh,dl=vi37.fused_doclist(data,scores)
    fresh=read(OUT/'new_ids.json')
    lengths=read(OUT/'export_lengths.json')
    selected={q:[[r[0],lengths[str(r[0])]['idx']] for r in rows[:len(dl[q])+30]
                 if r[0] in set(fresh[q]) and lengths[str(r[0])]['windows']
                 and r[0] not in {h[0] for h in data['q8b'].get(q,[])}] for q,rows in fh.items()}
    selected={q:rows for q,rows in selected.items() if rows}
    write(OUT/'modal_input/q8b_cands.json',selected)
    write(OUT/'qselect_report.json',dict(pairs=sum(map(len,selected.values())),
        rule='new docs in pre-8B RRF doclist+30; old 8B unchanged; no iterative old-pool supplement',
        historical_cosine='last rank fallback; old source ranks unchanged'))
    manifest()


def preaudit():
    """Pre-8B CPU fusion audit; diagnostic only, never changes selection or scores."""
    import dwmax_diag as dw
    import vi37
    data,scores,wins=merged_inputs()
    fh,dl=vi37.fused_doclist(data,scores)
    fresh=read(OUT/'new_ids.json')
    old=dw.read(ROOT/'data/sub/vi36/in1/pool_v36.json')
    q8old=dw.read(ROOT/'data/sub/vi37/q8b_v37.json')
    report={'queries':1200,'new_in_pre8b_doclist':0,'new_in_pre8b_margin':0,
            'old_missing8b_doclist':0,'old_missing8b_margin':0,'new_nq_bge_positive':0,
            'dense_old_order_errors':0,'hyde_old_order_errors':0,'old_bge_changed':0}
    sourceold=vi37.inputs()
    for q,rows in fh.items():
        nd=set(fresh[q]);known={r[0] for r in q8old.get(q,[])}
        docs=set(dl[q]);margin={r[0] for r in rows[:len(dl[q])+30]}
        report['new_in_pre8b_doclist']+=len(docs&nd)
        report['new_in_pre8b_margin']+=len(margin&nd)
        report['old_missing8b_doclist']+=len(docs-nd-known)
        report['old_missing8b_margin']+=len(margin-nd-known)
        oldlog={r[0]:r[2] for r in old[q]}
        report['old_bge_changed']+=sum(r[2]!=oldlog[r[0]] for r in data['pool'][q] if r[0] in oldlog)
        report['new_nq_bge_positive']+=sum(r[0] in nd and r[2]>-2 for r in data['pool'][q])
        for s in ('dense','hyde'):
            before=sourceold[s].get(q,[])
            after=data[s].get(q,[])[:len(before)]
            report[s+'_old_order_errors']+=before!=after
    assert report['old_bge_changed']==report['dense_old_order_errors']==report['hyde_old_order_errors']==0
    write(OUT/'preaudit_report.json',report)


def merge():
    import dwmax_diag as dw
    import vi37
    import vi36
    assert (OUT/'modal_out/q8b_new.json').exists(),'Do not finalize a pre-8B pipeline'
    data,scores,wins=merged_inputs()
    fh,dl=vi37.fused_doclist(data,scores)
    if NQ_FROM is not None:
        ref={str(it['id']):it['relevant_chunks'] for it in dw.read(ROOT/'data/sub/vi36/in1/ref_chunks_v36.json')}
        dl=fixed_doclists(fh,read(NQ_FROM),ref)
        write(OUT/'fixed_nq_report.json',dict(source=str(NQ_FROM),
            sha256=hashlib.sha256(Path(NQ_FROM).read_bytes()).hexdigest(),queries=len(dl),
            counts_equal=sum(len(dl[str(row['id'])])==len(row['relevant_docs']) for row in read(NQ_FROM)),
            ranking='unchanged vi41 RRF; mandatory ref-chunk docs retained'))
    missing=vi36.select_8b_targets(dl,fh,data['q8b'],30)
    write(OUT/'missing_8b.json',missing)
    fresh=read(OUT/'new_ids.json')
    write(OUT/'merge_report.json',dict(missing_8b_pairs=sum(map(len,missing.values())),
        missing_new_8b_pairs=sum(d in set(fresh[q]) for q,ds in missing.items() for d in ds),
        missing_8b_in_final=sum(d not in {r[0] for r in data['q8b'].get(q,[])} for q,ds in dl.items() for d in ds),
        missing_new_8b_in_final=sum(d in set(fresh[q]) and d not in {r[0] for r in data['q8b'].get(q,[])}
                                  for q,ds in dl.items() for d in ds),
        new_pairs=sum(map(len,fresh.values())),
        supplement_rule='only new docs pre-8B top n_q+30; old 8B unchanged',
        historical_cosine='last rank fallback; old dense/HyDE order preserved'))
    write(OUT/'cache_fhyde_v41.json',fh)
    write(OUT/'doclist_v41.json',dl)
    write(OUT/'docwin_max_v41.json',scores)
    for key,obj in data.items():
        write(OUT/'in1'/(key+'_v41.json'),obj)
    # Missing targets are explicit: only new pre-8B top candidates are supplemented by authorization.


def pre_chunks():
    """Cache U900 new-candidate chunks using already-exported text while 8B runs."""
    import dwmax_diag as dw
    import vi37
    from rerank_run import load_docs
    from chunks_from_docwin import build_q_rows,select_q_rows,verify_verbatim
    cands=read(OUT/'modal_input/new_cands.json')
    wanted={d for rows in cands.values() for d,i in rows}
    docs=load_docs([str(p) for p in sorted((OUT/'modal_input').glob('docs_new.part*.jsonl.zst'))],wanted=wanted)
    wins=dw.read(OUT/'modal_out/docwin_new.json.zst')
    cache={}
    new={q:select_q_rows(build_q_rows(rows,docs,wins.get(q,{}),900,1000,cache),100000) for q,rows in cands.items()}
    assert verify_verbatim(new,docs)==0
    vi37.writez(OUT/'new_cands_u900.json.zst',new)
    write(OUT/'pre_chunks_report.json',dict(pairs=sum(map(len,cands.values())),
        chunks=sum(map(len,new.values())),verbatim_bad=0,
        text_source='exact Modal inputs; same U900/docwin scoring functions as vi37'))


def merge_prechunks(old, prec, doclists):
    from vi24 import merge_chunks
    # Direct path creates new chunk candidates ONLY for final docs; extras would
    # otherwise alter zc normalization even when order_q later filters doc IDs.
    selected={}
    for q,rs in prec.items():
        have={str(h[0]) for h in old.get(q,[])}
        rank={str(d):i+1 for i,d in enumerate(d for d in doclists.get(q,[]) if str(d) not in have)}
        selected[q]=[[r[0],rank[str(r[0])],*r[2:]] for r in rs if str(r[0]) in rank]
        selected[q].sort(key=lambda r:(-float(r[3]),r[1],r[2]))
    prec=selected
    return merge_chunks(old,prec)


def preverify():
    """Read-only corpus-body export for all possible final docs; allows cheap full verbatim audit."""
    import dwmax_diag as dw
    import vi37
    old=dw.read(ROOT/'data/sub/vi36/in1/pool_v36.json')
    fresh=read(OUT/'new_ids.json')
    wanted={r[0] for rows in old.values() for r in rows}|{d for ds in fresh.values() for d in ds}
    lengths,_,files=vi37.export_texts(wanted,OUT/'verify_text','docs_verify')
    write(OUT/'verify_lengths.json',lengths)
    write(OUT/'preverify_report.json',dict(documents=len(lengths),files=[str(p) for p in files],
        bytes=sum(p.stat().st_size for p in files),text_source='same vi37 body export/no title/cap200000'))


def chunk():
    import dwmax_diag as dw
    import vi37
    from rerank_run import load_docs
    from chunks_from_docwin import build_q_rows,select_q_rows,verify_verbatim
    from vi24 import merge_chunks
    assert (OUT/'qselect_report.json').exists()
    old=dw.read(ROOT/'data/sub/vi37/cands_v37.json.zst')
    dl=read(OUT/'doclist_v41.json')
    if (OUT/'new_cands_u900.json.zst').exists():
        assert read(OUT/'pre_chunks_report.json')['verbatim_bad']==0
        # Same rule as direct chunk path: never replace pre-existing doc's candidates.
        old=merge_prechunks(old,dw.read(OUT/'new_cands_u900.json.zst'),dl)
    pool=read(OUT/'in1/pool_v41.json')
    pos={q:{str(r[0]):r[1] for r in rows} for q,rows in pool.items()}
    fresh={q:[[int(d),pos[q][str(d)]] for d in ds if str(d) not in {str(r[0]) for r in old.get(q,[])}] for q,ds in dl.items()}
    wanted={r[0] for rows in fresh.values() for r in rows}
    if not wanted:
        vi37.writez(OUT/'cands_v41.json.zst',old)
        write(OUT/'chunk_report.json',dict(new_pairs=0,new_chunks=0,verbatim_bad=0,
            note='all final doc chunk candidates already covered; precomputed U900 reused'))
        return
    _,_,files=vi37.export_texts(wanted,OUT/'chunk_text','docs_chunk')
    docs=load_docs([str(p) for p in files],wanted=wanted)
    wins=dw.read(ROOT/'data/sub/vi37/docwin_new.json.zst')
    owners=dw.read(ROOT/'data/sub/diag37/docwin_max_provenance.json')
    for name,rel,legacy in dw.layer_specs():
        obj=dw.read(ROOT/'data/sub'/rel)
        for q,rows in fresh.items():
            for d,i in rows:
                sd=str(d)
                if sd not in wins.get(q,{}) and owners.get(q,{}).get(sd)==name and sd in obj.get(q,{}):
                    wins.setdefault(q,{})[sd]=obj[q][sd]
    for q,ds in dw.read(OUT/'modal_out/docwin_new.json.zst').items():
        wins.setdefault(q,{}).update(ds)
    assert not [(q,d) for q,rs in fresh.items() for d,i in rs if docs[d].strip() and str(d) not in wins.get(q,{})]
    cache={}
    new={q:select_q_rows(build_q_rows(rows,docs,wins.get(q,{}),900,1000,cache),100000) for q,rows in fresh.items()}
    assert verify_verbatim(new,docs)==0
    vi37.writez(OUT/'cands_v41.json.zst',merge_chunks(old,new))
    write(OUT/'chunk_report.json',dict(new_pairs=sum(map(len,fresh.values())),new_chunks=sum(map(len,new.values())),verbatim_bad=0))


def build():
    import subprocess
    import zipfile
    import dwmax_diag as dw
    import vi29
    import vi37
    from reorder_zip import write_reordered_zip
    assert read(OUT/'sanity_report.json')['doclist_equal']==1200
    assert read(OUT/'sanity_chain_report.json')['chunks_equal']==1200
    assert read(OUT/'input_audit_report.json')['source_caps_ok']
    assert (OUT/'modal_out/q8b_new.json').exists(),'8B phase must complete before build'
    data={s:read(OUT/'in1'/(s+'_v41.json')) for s in ('pool','q8b','dense','hyde')}
    cands=dw.read(OUT/'cands_v41.json.zst')
    ref={str(it['id']):it['relevant_chunks'] for it in dw.read(ROOT/'data/sub/vi36/in1/ref_chunks_v36.json')}
    kq={str(it['id']):len(it['relevant_chunks']) for it in dw.read(vi29.REF_K)}
    with vi37.chain_docwin(read(OUT/'docwin_max_v41.json')):
        rrf,fh,sub,rep=vi29.chain_build_v29(data['pool'],data['q8b'],data['dense'],data['hyde'],cands,kq,ref,
            delta=MAX_DELTA,dedup_mode='doc_text',
            doclists=read(OUT/'doclist_v41.json') if NQ_FROM is not None else None)
    if NQ_FROM is not None:
        assert read(OUT/'fixed_nq_report.json')['counts_equal']==1200
        assert fh==read(OUT/'cache_fhyde_v41.json'), 'Fixed n_q must never change RRF scores/order'
        counts={str(r['id']):len(r['relevant_docs']) for r in read(NQ_FROM)}
        assert all(len(r['relevant_docs'])==counts[str(r['id'])] for r in sub)
        rep['nq_mode']='reference_doc_counts'
    assert {str(it['id']):it['relevant_docs'] for it in sub}==read(OUT/'doclist_v41.json')
    bad=set(dw.read(ROOT/'data/sub/diag28/bad_ids.json'))-set(dw.read(ROOT/'data/sub/diag30/new_ids.json'))
    corpus=vi29.load_corpus_ids(ROOT/'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    clean=vi29.assert_clean(sub,bad,corpus)
    assert all(len(it['relevant_docs'])==len(set(it['relevant_docs'])) for it in sub)
    assert {int(it['id']) for it in sub}==set(range(1,1201))
    wanted={int(c['doc_id']) for it in sub for c in it['relevant_chunks']}
    if (OUT/'preverify_report.json').exists():
        covered=set(read(OUT/'verify_lengths.json'))
        files=[Path(p) for p in read(OUT/'preverify_report.json')['files']]
        missing={d for d in wanted if str(d) not in covered}
        if missing:
            _,_,extra=vi37.export_texts(missing,OUT/'verify_extra','docs_verify_extra')
            files+=extra
    else:
        _,_,files=vi37.export_texts(wanted,OUT/'verify_text','docs_verify')
    from rerank_run import load_docs
    docs=load_docs([str(p) for p in files],wanted=wanted)
    verbatim=sum(c['chunk_text'] not in docs[int(c['doc_id'])] for it in sub for c in it['relevant_chunks'])
    assert verbatim==0
    name='vi41b' if NQ_FROM is not None else 'vi41'
    jp,zp=OUT/f'submission_{name}.json',OUT/f'submission_{name}.zip'
    write(jp,sub)
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),'validate',str(jp)],check=True)
    packed=write_reordered_zip(sub,zp)
    assert zp.stat().st_size<=99000000,'STOP ZIP too large; delta<=55 fixed, no silent parameter scan'
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        packedsub=json.loads(z.read('submission.json'))
    assert {it['id']:it for it in packedsub}=={it['id']:it for it in sub}
    # check_zip compares bytes; validate and check against the actual reordered JSON serialization.
    jp.write_text(json.dumps(packedsub,ensure_ascii=False),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),'validate',str(jp)],check=True)
    write(OUT/'report_chain_vi41.json',rep)
    check=vi29.core.check_zip(str(jp),str(zp),str(OUT/'report_chain_vi41.json'),'vi41',max_zip=99000000)
    assert check==0
    write(OUT/'build_report.json',dict(**packed,sha256=hashlib.sha256(zp.read_bytes()).hexdigest(),
        assert_clean=clean,verbatim_bad=verbatim,validate_exit=0,check_zip=check,delta=55,queries=len(sub)))


def stats():
    import numpy as np
    import dwmax_diag as dw
    old={str(it['id']):it for it in dw.read(ROOT/'data/sub/vi37/submission_vi37.json')}
    name='vi41b' if NQ_FROM is not None else 'vi41'
    new={str(it['id']):it for it in dw.read(OUT/f'submission_{name}.json')}
    fresh=read(OUT/'new_ids.json')
    q8=read(OUT/'in1/q8b_v41.json')
    bg=read(OUT/'in1/pool_v41.json')
    oldq=dw.read(ROOT/'data/sub/vi37/q8b_v37.json')
    oldbg=dw.read(ROOT/'data/sub/vi36/in1/pool_v36.json')
    a,b,j,entered=0,0,[],0
    added_counts=[]
    removed_counts=[]
    entry_rank=[]
    old_removed_rank=[]
    matched={s:{'new':[],'vi37_same_rank':[]} for s in ('8b','bge')}
    matched_missing={s:{'new':0,'vi37_same_rank':0} for s in ('8b','bge')}
    provenance=read(OUT/'new_provenance.json')
    source_counts={s:0 for s in ('dense','dchunk','bm25','neardup')}
    for q,it in new.items():
        od,nd=set(old[q]['relevant_docs']),set(it['relevant_docs'])
        a+=len(nd-od); b+=len(od-nd);j.append(len(od&nd)/len(od|nd))
        added_counts.append(len(nd-od));removed_counts.append(len(od-nd))
        entry_rank.extend(i+1 for i,d in enumerate(it['relevant_docs']) if d not in od)
        old_removed_rank.extend(i+1 for i,d in enumerate(old[q]['relevant_docs']) if d not in nd)
        for rank,d in enumerate(it['relevant_docs']):
            if d not in set(fresh[q]):
                continue
            entered+=1
            for s in provenance[q][str(d)]:
                source_counts[s]+=1
            if rank>=len(old[q]['relevant_docs']):
                continue
            odoc=old[q]['relevant_docs'][rank]
            for s,c,oc in [('8b',q8,oldq),('bge',bg,oldbg)]:
                cm={int(r[0]):r[2] for r in c[q]};om={int(r[0]):r[2] for r in oc[q]}
                if d in cm and odoc in om:
                    matched[s]['new'].append(cm[d]);matched[s]['vi37_same_rank'].append(om[odoc])
                else:
                    matched_missing[s]['new']+=d not in cm
                    matched_missing[s]['vi37_same_rank']+=odoc not in om
    def dist(xs):
        return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,
                    quantiles=np.quantile(xs,[0,.1,.5,.9,1]).tolist() if xs else [])
    write(OUT/'stats_report.json',dict(new_pool_docs_in_final_per_query=entered/1200,
        docs_in_per_query=a/1200,docs_out_per_query=b/1200,jaccard_mean=float(np.mean(j)),
        queries_docset_unchanged=sum(x==0 and y==0 for x,y in zip(added_counts,removed_counts)),
        queries_doclist_identical=sum(old[q]['relevant_docs']==it['relevant_docs'] for q,it in new.items()),
        added_per_query_distribution=dist(added_counts),removed_per_query_distribution=dist(removed_counts),
        entered_rank_distribution=dist(entry_rank),removed_rank_distribution=dist(old_removed_rank),
        docs_per_query_old=sum(len(it['relevant_docs']) for it in old.values())/1200,
        docs_per_query_new=sum(len(it['relevant_docs']) for it in new.values())/1200,
        chunks_per_query_old=sum(len(it['relevant_chunks']) for it in old.values())/1200,
        chunks_per_query_new=sum(len(it['relevant_chunks']) for it in new.values())/1200,
        new_final_by_source_per_query={s:n/1200 for s,n in source_counts.items()},
        matched_rank_logits={s:{k:dist(v) for k,v in rows.items()} for s,rows in matched.items()},
        matched_rank_missing=matched_missing,
        precision_gate='chưa có cổng; không diễn giải logit thành xác suất P',
        changes=('pool only; new docs last ranks dense/HyDE (historical cosine unavailable); no new RRF source; '
                 + ('n_q fixed to reference doc counts' if NQ_FROM is not None else 'n_q unchanged')
                 + '; delta<=55; reorder ZIP')))


def audit_inputs():
    """Read-only hashes and structural checks for actual retrieved/scored inputs."""
    import dwmax_diag as dw
    fresh=read(OUT/'new_ids.json')
    provenance=read(OUT/'new_provenance.json')
    assert set(fresh)==set(provenance)=={str(i) for i in range(1,1201)}
    assert all(len(ds)==len(set(ds)) and len(ds)<=80 for ds in fresh.values())
    old=dw.read(ROOT/'data/sub/vi36/in1/pool_v36.json')
    for q,ds in fresh.items():
        assert set(ds)=={int(d) for d in provenance[q]}
        assert not set(ds)&{int(r[0]) for r in old[q]}
        assert all(set(s)<= {'dense','dchunk','bm25','neardup'} for s in provenance[q].values())
        assert all(sum(s in labels for labels in provenance[q].values())<=20
                   for s in ('dense','dchunk','bm25','neardup'))
    paths=[OUT/'new_ids.json',OUT/'new_provenance.json',OUT/'modal_input/new_cands.json']
    paths+=sorted((OUT/'modal_input').glob('docs_new.part*.jsonl.zst'))
    report=dict(queries=1200,pairs=sum(map(len,fresh.values())),
        source_caps_ok=True,outside_old_pool=True,duplicates=0,
        files=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    write(OUT/'input_audit_report.json',report)


def main():
    global OUT, NQ_FROM
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['rewrite-gpu','repair-local','bm25','inspect','dense','candidates','dchunk','neardup',
                                     'sanity','sanity_chain','manifest','audit_inputs','qselect','preaudit','pre_chunks','preverify','merge','chunk','build','stats'])
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path, help='Output directory (fixed-count mode defaults to separate vi41b)')
    ap.add_argument('--nq-from', type=Path, help='Freeze per-query doc counts from reference submission; never alters RRF')
    args = ap.parse_args()
    NQ_FROM = args.nq_from
    OUT = args.out or (ROOT/'data/sub/vi41b' if NQ_FROM is not None else OUT)
    if NQ_FROM is not None:
        assert OUT.resolve() != (ROOT/'data/sub/vi41').resolve(), 'vi41 inputs are read-only in fixed-count mode'
    if args.stage == 'rewrite-gpu':
        rewrite_gpu(args.root)
    elif args.stage == 'repair-local':
        repair_local(args.root)
    elif args.stage == 'inspect':
        inspect()
    else:
        import dwmax_diag as dw
        OUT.mkdir(parents=True, exist_ok=True)
        with dw.owned_lock(ROOT/'locks/heavy_job.lock', f'vi41 {args.stage} pid={os.getpid()}'):
            globals()[args.stage]()


if __name__ == '__main__':
    main()
