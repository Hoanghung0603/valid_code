"""Writes a Kaggle embedding notebook for chunk sets / whole pages (input datasets from devset/embed_export.py).

Text sent to the model (team request, Oct 23: clean + NFC on chunks AND queries): title + chunk text (pages: the page
text), then textprep.normalize (NFC, quote / bar cleanup, whitespace collapse) and textprep.bm25_normalize (one
tone-mark placement: hoà -> hòa), so the same syllable is one token id for the sparse weights. Queries get the same.

  bge-m3   dense = L2-normalised CLS vector (fp16) and sparse = lexical weights relu(sparse_linear(h)) per token id
           (max over repeats, special tokens dropped), as FlagEmbedding BGEM3FlagModel
  me5      intfloat/multilingual-e5-large: mean pooling, "passage: " / "query: " prefixes, dense only

Outputs per worker part (written every ~100k rows, so a timeout keeps what is done):
  dense-NN-PPP.npy (fp16, rows in the order of) ids-NN-PPP.parquet (doc_id[, chunk_idx])
  sparse-NN-PPP.parquet (doc_id[, chunk_idx], tok: list<int32>, w: list<float32>)       [bge-m3]
  queries_dense.npy + queries_sparse.parquet (id, tok, w) + queries.parquet (id, query, text)

  python share/kaggle/mk_embed.py --name vibiomir-emb512a --dataset ducanger/vibiomir-chunks512-a --kind chunks --maxlen 1024
  python share/kaggle/mk_embed.py --name vibiomir-embpage --dataset thanhdatdo/vibiomir-pages --kind pages --maxlen 8192
"""
import argparse, json
from pathlib import Path

WORKER = r'''
import torch, torch.nn.functional as F                        # torch before pandas / pyarrow
import sys, time, glob, re, unicodedata as ud, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
gpu, ngpu, inp, out, deadline = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5])
KIND, MODEL, MAXLEN, SHARD, NSHARD = "__KIND__", "__MODEL__", __MAXLEN__, __SHARD__, __NSHARD__
REPL = {"\n": " ", "‘": "'", "”": '"', "“": '"', "′": "'", "``": "", "''": "", "|": ""}
OLD_TONE = re.compile(r"(?<![qQ])([oOuU])([aAeEyY])([̣̀́̃̉])(?=\W|$)")
def norm(t):                                                    # textprep.normalize + textprep.bm25_normalize
    t = ud.normalize("NFC", t or "")
    for a, b in REPL.items(): t = t.replace(a, b)
    t = re.sub(r"\s+", " ", t).strip()
    t = OLD_TONE.sub(lambda m: m.group(1) + m.group(3) + m.group(2), ud.normalize("NFD", t))
    return ud.normalize("NFC", t)
def concat(title, ctx):
    title = (title or "").strip()
    if not title: return ctx or ""
    return title + (" " if title[-1] in ".!?;:,。！？；：，、" else ". ") + (ctx or "")
files = sorted(glob.glob(f"{inp}/**/{'chunks' if KIND == 'chunks' else 'pages'}-*.parquet", recursive=True))
files = files[SHARD::NSHARD]                                     # this notebook's share of the input parts
keys = ["doc_id", "chunk_idx"] if KIND == "chunks" else ["doc_id"]
cols = keys + (["title", "context", "page_title"] if KIND == "chunks" else ["text"])
tb = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
d = tb.take(pa.array(np.arange(gpu, tb.num_rows, ngpu))).to_pandas(); del tb      # this GPU's rows only
if KIND == "chunks":
    title = d.title.where(d.title.fillna("").str.strip() != "", d.page_title).fillna("")
    texts = [norm(concat(a, b)) for a, b in zip(title, d.context)]
else:
    texts = [norm(t) for t in d.text]
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(MODEL)
if MODEL.startswith("intfloat/"): texts = ["passage: " + t for t in texts]
d = d[keys].reset_index(drop=True)                                # the text columns are no longer needed
# lengths only (token id lists for ~450k texts as Python ints took ~7 GB per worker: both were OOM-killed, Oct 23);
# each batch is tokenised again when it is encoded
L = np.zeros(len(texts), np.int32)
for s in range(0, len(texts), 20000):
    L[s:s + 20000] = [len(x) for x in tok(texts[s:s + 20000], truncation=True, max_length=MAXLEN)["input_ids"]]
order = np.argsort(-L)                                            # longest first: OOM shows up at once
m = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
SP = None
if MODEL == "BAAI/bge-m3":
    from huggingface_hub import hf_hub_download
    SP = torch.nn.Linear(1024, 1); SP.load_state_dict(torch.load(hf_hub_download(MODEL, "sparse_linear.pt"), map_location="cpu"))
    SP = SP.half().cuda()
special = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
@torch.no_grad()
def encode(batch):
    n = max(len(x) for x in batch)
    ii = torch.full((len(batch), n), tok.pad_token_id); am = torch.zeros((len(batch), n), dtype=torch.long)
    for k, x in enumerate(batch): ii[k, :len(x)] = torch.tensor(x); am[k, :len(x)] = 1
    ii, am = ii.cuda(), am.cuda()
    H = m(input_ids=ii, attention_mask=am).last_hidden_state
    if SP is None:                                               # e5: mean pooling
        v = (H * am[..., None]).sum(1) / am.sum(1, keepdim=True)
        return F.normalize(v.float(), dim=-1).half().cpu().numpy(), None
    dense = F.normalize(H[:, 0].float(), dim=-1).half().cpu().numpy()
    w = torch.relu(SP(H)).squeeze(-1).float() * am
    w, ii = w.cpu().numpy(), ii.cpu().numpy()
    sp = []
    for k in range(len(batch)):
        acc = {}
        for t, x in zip(ii[k, :len(batch[k])], w[k, :len(batch[k])]):
            if t not in special and x > 0 and x > acc.get(t, 0.0): acc[int(t)] = float(x)
        sp.append((np.fromiter(acc.keys(), np.int32, len(acc)), np.fromiter(acc.values(), np.float32, len(acc))))
    return dense, sp
part, buf_d, buf_s, buf_i, t0, done = 0, [], [], [], time.time(), 0
TOK = 40000 if MAXLEN <= 1024 else 24000                         # tokens per batch (T4 16 GB, fp16, sdpa)
def flush():
    global part, buf_d, buf_s, buf_i
    if not buf_i: return
    sel = d.iloc[np.concatenate(buf_i)][keys].reset_index(drop=True)
    np.save(f"{out}-{gpu}-{part:03d}-dense.npy", np.vstack(buf_d))
    sel.to_parquet(f"{out}-{gpu}-{part:03d}-ids.parquet", index=False)
    if buf_s:
        s = [x for b in buf_s for x in b]
        t = pa.table({**{k: sel[k].to_numpy() for k in keys}, "tok": pa.array([x[0] for x in s], pa.list_(pa.int32())),
                      "w": pa.array([x[1] for x in s], pa.list_(pa.float32()))})
        pq.write_table(t, f"{out}-{gpu}-{part:03d}-sparse.parquet", compression="zstd")
    part, buf_d, buf_s, buf_i = part + 1, [], [], []
i = 0
while i < len(order) and time.time() < deadline:
    n = max(1, min(512, TOK // max(int(L[order[i]]), 16)))
    idx = order[i:i + n]
    dv, sv = encode(tok([texts[j] for j in idx], truncation=True, max_length=MAXLEN)["input_ids"])
    buf_d.append(dv); buf_i.append(idx)
    if sv is not None: buf_s.append(sv)
    i += n; done += len(idx)
    if sum(len(x) for x in buf_i) >= 100_000: flush()
    if (i // n) % 200 == 0: print(f"gpu{gpu}: {done:,}/{len(order):,} {time.time()-t0:.0f}s len {L[order[min(i, len(order)-1)]]}", flush=True)
flush()
print(f"gpu{gpu}: done {done:,}/{len(order):,} in {time.time()-t0:.0f}s", flush=True)
'''

QUERY = r'''
import torch, torch.nn.functional as F
import glob, re, unicodedata as ud, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
exec(open("/kaggle/working/w.py").read().split("files = sorted")[0].split("gpu, ngpu")[0])      # imports only
REPL = {"\n": " ", "‘": "'", "”": '"', "“": '"', "′": "'", "``": "", "''": "", "|": ""}
OLD_TONE = re.compile(r"(?<![qQ])([oOuU])([aAeEyY])([̣̀́̃̉])(?=\W|$)")
def norm(t):
    t = ud.normalize("NFC", t or "")
    for a, b in REPL.items(): t = t.replace(a, b)
    t = re.sub(r"\s+", " ", t).strip()
    t = OLD_TONE.sub(lambda m: m.group(1) + m.group(3) + m.group(2), ud.normalize("NFD", t))
    return ud.normalize("NFC", t)
MODEL = "__MODEL__"
q = pd.read_parquet(sorted(glob.glob("/kaggle/input/**/queries.parquet", recursive=True))[0])
q["text"] = [norm(t) for t in q["query"]]
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
texts = [("query: " + t) if MODEL.startswith("intfloat/") else t for t in q.text]
D, S = [], []
SP = None
if MODEL == "BAAI/bge-m3":
    from huggingface_hub import hf_hub_download
    SP = torch.nn.Linear(1024, 1); SP.load_state_dict(torch.load(hf_hub_download(MODEL, "sparse_linear.pt"), map_location="cpu")); SP = SP.half().cuda()
special = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
with torch.no_grad():
    for s in range(0, len(texts), 32):
        b = tok(texts[s:s+32], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        H = m(**b).last_hidden_state
        if SP is None:
            v = (H * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
            D.append(F.normalize(v.float(), dim=-1).half().cpu().numpy()); continue
        D.append(F.normalize(H[:, 0].float(), dim=-1).half().cpu().numpy())
        w = (torch.relu(SP(H)).squeeze(-1).float() * b["attention_mask"]).cpu().numpy(); ii = b["input_ids"].cpu().numpy()
        for k in range(len(ii)):
            acc = {}
            for t, x in zip(ii[k], w[k]):
                if t not in special and x > 0 and x > acc.get(t, 0.0): acc[int(t)] = float(x)
            S.append((list(acc.keys()), list(acc.values())))
np.save("/kaggle/working/queries_dense.npy", np.vstack(D))
q[["id", "query", "text"]].to_parquet("/kaggle/working/queries.parquet", index=False)
if S:
    pq.write_table(pa.table({"id": q["id"].to_numpy(), "tok": pa.array([x[0] for x in S], pa.list_(pa.int32())),
                             "w": pa.array([x[1] for x in S], pa.list_(pa.float32()))}), "/kaggle/working/queries_sparse.parquet")
print("queries", len(q))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="kernel slug, e.g. vibiomir-emb512a")
    ap.add_argument("--dataset", required=True, help="input dataset ref (owner/slug)")
    ap.add_argument("--kind", choices=["chunks", "pages"], default="chunks")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--shard", default="0/1", help="this notebook's share of the input parts: i/n")
    ap.add_argument("--hours", type=float, default=11.3)
    ap.add_argument("--out", default="", help="kernel folder (default: share/kaggle/kernels/<name>)")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    w = (WORKER.replace("__KIND__", a.kind).replace("__MODEL__", a.model).replace("__MAXLEN__", str(a.maxlen))
         .replace("__SHARD__", str(si)).replace("__NSHARD__", str(sn)))
    q = QUERY.replace("__MODEL__", a.model).replace(
        'exec(open("/kaggle/working/w.py").read().split("files = sorted")[0].split("gpu, ngpu")[0])      # imports only\n', "")
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    cells = [
        md(f"# {a.name}: {a.model} {a.kind} (max {a.maxlen} tokens), input parts {si}/{sn} of {a.dataset}\n\n"
           f"Generated by share/kaggle/mk_embed.py. Text: title + chunk, NFC + quote/whitespace cleanup + one tone "
           f"placement (textprep.normalize + bm25_normalize); queries the same."),
        code("WORKER = " + repr(w) + "\nopen('/kaggle/working/w.py', 'w').write(WORKER)"),
        code(f"""import os, subprocess, sys, time, torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51.0"], check=False)
from huggingface_hub import snapshot_download; snapshot_download("{a.model}")
deadline = time.time() + {a.hours} * 3600
n = torch.cuda.device_count()
ps = [subprocess.Popen([sys.executable, "/kaggle/working/w.py", str(g), str(n), "/kaggle/input", "/kaggle/working/emb", str(deadline)],
                       env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}}) for g in range(n)]
print([p.wait() for p in ps])
os.remove("/kaggle/working/w.py")"""),
        code(q),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    owner = a.dataset.split("/")[0]
    kdir = Path(a.out) if a.out else Path(__file__).with_name("kernels") / a.name
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / f"{a.name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{a.name}", "title": a.name, "code_file": f"{a.name}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "keywords": ["gpu"], "dataset_sources": [a.dataset], "kernel_sources": [], "competition_sources": [],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=1), encoding="utf-8")
    print("written", kdir)


if __name__ == "__main__":
    main()
