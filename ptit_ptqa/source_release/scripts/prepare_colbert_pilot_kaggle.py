"""Package a query-disjoint BGE-M3 ColBERT throughput pilot for Kaggle."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/kaggle_outputs/p2a_chunk_full/p2a_chunk_full/p2a_pairs.parquet"
QUERY = ROOT / "data/raw/btc_provided/query.parquet"
OUT = ROOT / "data/kaggle_inputs/colbert_pilot_20261103"
KERNEL = ROOT / "data/kaggle_inputs/colbert_pilot_kernel"
NOTEBOOK = "24_bgem3_colbert_pilot_kaggle.ipynb"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    KERNEL.mkdir(parents=True, exist_ok=True)
    pairs_path = OUT / "colbert_pilot_pairs.parquet"
    manifest_path = OUT / "manifest.json"
    if pairs_path.exists() or manifest_path.exists():
        raise FileExistsError("Pilot artifact exists; refusing to overwrite")
    queries = pd.read_parquet(QUERY, columns=["id", "query"])
    queries["length"] = queries["query"].str.len()
    queries["stratum"] = pd.qcut(queries["length"].rank(method="first"), 3, labels=False)
    rng = np.random.default_rng(20261103)
    selected = sorted(int(q) for group in range(3)
                      for q in rng.choice(queries.loc[queries.stratum == group, "id"].to_numpy(), 20, replace=False))
    source = pd.read_parquet(SOURCE, columns=["pair_id", "query_id", "doc_id", "passage_rank",
                                             "chunk_ordinal", "query_text", "passage_text"])
    source = source[source.query_id.isin(selected)].copy()
    source.sort_values(["query_id", "chunk_ordinal", "passage_rank", "pair_id"], inplace=True)
    first_rank = source.groupby(["query_id", "doc_id"], sort=False)["chunk_ordinal"].transform("min")
    source["doc_first_chunk_rank"] = first_rank
    first_docs = (source[["query_id", "doc_id", "doc_first_chunk_rank"]]
                  .drop_duplicates(["query_id", "doc_id"])
                  .sort_values(["query_id", "doc_first_chunk_rank", "doc_id"]))
    first_docs["doc_pick_rank"] = first_docs.groupby("query_id").cumcount() + 1
    keep = first_docs[first_docs.doc_pick_rank <= 50][["query_id", "doc_id", "doc_pick_rank"]]
    pilot = source.merge(keep, on=["query_id", "doc_id"], how="inner", validate="many_to_one")
    pilot = pilot.groupby(["query_id", "doc_id"], sort=False).head(2).copy()
    pilot["passage_text_sha256"] = pilot.passage_text.map(lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    pilot.sort_values(["query_id", "doc_pick_rank", "passage_rank", "pair_id"], inplace=True)
    if pilot.query_id.nunique() != 60 or pilot.pair_id.duplicated().any() or pilot.passage_text.isna().any():
        raise RuntimeError("Pilot coverage, uniqueness, or text check failed")
    pilot.to_parquet(pairs_path, index=False, compression="zstd")
    manifest = {"schema":"colbert_pilot_v1", "source":"P2a full exact source windows",
                "queries":60, "query_ids":selected, "docs_per_query_max":50, "passages_per_doc_max":2,
                "pairs":len(pilot), "unique_query_docs":len(keep),
                "source_sha256":sha(SOURCE), "query_sha256":sha(QUERY),
                "pairs_sha256":sha(pairs_path), "model":"BAAI/bge-m3",
                "model_revision":"5617a9f61b028005a4858fdac845db406aefb181",
                "max_query_tokens":256, "max_passage_tokens":384,
                "purpose":"throughput and stability pilot only; no relevance labels"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    notebook = make_notebook()
    nbf.write(notebook, ROOT / "notebooks" / NOTEBOOK)
    nbf.write(notebook, KERNEL / NOTEBOOK)
    kernel_meta = {"id":"ptquanh/r2ai-bge-m3-colbert-pilot-20261103",
                   "title":"R2AI BGE-M3 ColBERT Pilot 20261103", "code_file":NOTEBOOK,
                   "language":"python", "kernel_type":"notebook", "is_private":"true",
                   "enable_gpu":"true", "enable_tpu":"false", "enable_internet":"true",
                   "machine_shape":"NvidiaTeslaT4",
                   "dataset_sources":["ptquanh/r2ai-bgem3-colbert-pilot-input-20261103"],
                   "competition_sources":[], "kernel_sources":[], "model_sources":[]}
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2), encoding="utf-8")
    dataset_meta = {"title":"R2AI BGE-M3 ColBERT pilot input 20261103",
                    "id":"ptquanh/r2ai-bgem3-colbert-pilot-input-20261103",
                    "licenses":[{"name":"other"}], "isPrivate":True}
    (OUT / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
    print(json.dumps({"pairs":len(pilot), "queries":pilot.query_id.nunique(),
                      "docs":pilot.groupby("query_id").doc_id.nunique().describe().to_dict(),
                      "input":str(OUT), "kernel":str(KERNEL)}, indent=2))


def make_notebook():
    cells = [nbf.v4.new_markdown_cell("""# BGE-M3 ColBERT late-interaction pilot

60 official queries × up to 50 existing candidate docs × 2 verbatim passage windows. This is a **throughput/stability pilot**, not a BTC submission or a relevance evaluation. No qrels are present. Exact model revision and input SHA are checked. One T4 GPU is intentional; resume from atomic checkpoint. No private keys are printed.
"""),
             nbf.v4.new_code_cell('''%pip install -q "FlagEmbedding==1.3.5" "transformers==4.51.3" "accelerate==1.6.0" "sentence-transformers==3.4.1" "huggingface-hub==0.30.2" "pyarrow>=18,<21" "pandas>=2.2,<3"'''),
             nbf.v4.new_code_cell('''import gc, hashlib, json, os, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from FlagEmbedding import BGEM3FlagModel

ROOT=Path('/kaggle/input')
inputs=list(ROOT.rglob('colbert_pilot_pairs.parquet'))
manifests=list(ROOT.rglob('manifest.json'))
if len(inputs)!=1: raise RuntimeError(f'Expected one pilot Parquet, got {len(inputs)}')
manifest_path=inputs[0].with_name('manifest.json')
if not manifest_path.is_file(): raise RuntimeError('Pilot manifest missing beside Parquet')
manifest=json.loads(manifest_path.read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
if manifest['schema']!='colbert_pilot_v1' or sha(inputs[0])!=manifest['pairs_sha256']:
    raise RuntimeError('Input fingerprint mismatch')
pairs=pd.read_parquet(inputs[0]); assert len(pairs)==manifest['pairs']
WORK=Path('/kaggle/working/colbert_pilot');WORK.mkdir(exist_ok=True)
checkpoint=WORK/'scores.parquet'
done=pd.read_parquet(checkpoint) if checkpoint.exists() else pd.DataFrame()
done_ids=set(done.pair_id) if len(done) else set()
print({'pairs':len(pairs),'remaining':len(pairs)-len(done_ids),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},flush=True)
if not torch.cuda.is_available(): raise RuntimeError('GPU required; stop without CPU fallback')'''),
             nbf.v4.new_code_cell('''t0=time.time()
MODEL='BAAI/bge-m3'
REVISION=manifest['model_revision']
try:
    from kaggle_secrets import UserSecretsClient
    token=UserSecretsClient().get_secret('HF_TOKEN')
except Exception:
    token=None
if token:
    from huggingface_hub import login
    login(token=token,add_to_git_credential=False)
from huggingface_hub import snapshot_download
model_path=snapshot_download(repo_id=MODEL,revision=REVISION,token=token)
model=BGEM3FlagModel(model_path,use_fp16=True,devices=['cuda:0'],batch_size=8,
                    query_max_length=256,passage_max_length=384)
print({'model_load_seconds':round(time.time()-t0,1),'gpu_allocated_gb':round(torch.cuda.memory_allocated()/1e9,2)},flush=True)'''),
             nbf.v4.new_code_cell('''# Query embeddings are computed once. Passage vectors are never stored long-term.
queries=pairs[['query_id','query_text']].drop_duplicates('query_id').set_index('query_id').query_text.to_dict()
qvec={}
for qid,text in queries.items():
    out=model.encode([text],batch_size=1,max_length=256,
                     return_dense=False,return_sparse=False,return_colbert_vecs=True)
    qvec[int(qid)]=out['colbert_vecs'][0]
print({'queries_encoded':len(qvec)},flush=True)'''),
             nbf.v4.new_code_cell('''# Atomic checkpoints every 100 pairs. Resume is keyed by pair_id and input hash.
if checkpoint.exists():
    state=json.loads((WORK/'state.json').read_text())
    if state['pairs_sha256']!=manifest['pairs_sha256'] or state['revision']!=REVISION:
        raise RuntimeError('Checkpoint belongs to different input/model')
records=done.to_dict('records') if len(done) else []
pending=pairs.loc[~pairs.pair_id.isin(done_ids)]
start=time.time();n_initial=len(records)
for b in range(0,len(pending),8):
    batch=pending.iloc[b:b+8]
    output=model.encode(batch.passage_text.astype(str).tolist(),batch_size=8,max_length=384,
                        return_dense=False,return_sparse=False,return_colbert_vecs=True)
    for row,pvec in zip(batch.itertuples(index=False),output['colbert_vecs']):
        score=float(model.colbert_score(qvec[int(row.query_id)],pvec))
        if not np.isfinite(score): raise RuntimeError('Non-finite ColBERT score')
        records.append({'pair_id':row.pair_id,'query_id':int(row.query_id),'doc_id':int(row.doc_id),
                        'doc_pick_rank':int(row.doc_pick_rank),'passage_rank':int(row.passage_rank),
                        'colbert_score':score})
    if (b+len(batch))%100<8 or b+len(batch)==len(pending):
        stage=WORK/'scores.building.parquet'
        pd.DataFrame(records).to_parquet(stage,index=False)
        os.replace(stage,checkpoint)
        (WORK/'state.json').write_text(json.dumps({'pairs_sha256':manifest['pairs_sha256'],'revision':REVISION,
            'complete':len(records),'elapsed_seconds':round(time.time()-start,1)}))
        elapsed=max(time.time()-start,.001); speed=(len(records)-n_initial)/elapsed
        print({'complete':len(records),'total':len(pairs),'pairs_per_second':round(speed,2),
               'eta_minutes':round((len(pairs)-len(records))/max(speed,.001)/60,1)},flush=True)
    del output
    gc.collect()
assert len(records)==len(pairs) and len(set(x['pair_id'] for x in records))==len(pairs)
report={'input_sha256':manifest['pairs_sha256'],'model':MODEL,'revision':REVISION,
        'pairs':len(records),'queries':len(qvec),'seconds_this_session':round(time.time()-start,1),
        'pairs_per_second_this_session':round((len(records)-n_initial)/max(time.time()-start,.001),2),
        'scores_sha256':sha(checkpoint),'status':'complete','btc_score':None}
(WORK/'run_manifest.json').write_text(json.dumps(report,indent=2))
archive=Path('/kaggle/working/colbert_pilot_outputs.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    z.write(checkpoint,'scores.parquet');z.write(WORK/'run_manifest.json','run_manifest.json')
print({'output':str(archive),'sha256':sha(archive),'report':report},flush=True)''')]
    for cell in cells:
        if cell.cell_type == "code" and not cell.source.startswith("%pip"):
            compile(cell.source, "<colbert-pilot>", "exec")
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})


if __name__ == "__main__":
    main()
