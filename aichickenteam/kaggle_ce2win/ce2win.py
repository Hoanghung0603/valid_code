"""Kaggle T4×2: như kaggle_ce2 nhưng văn bản = CỬA SỔ dense tốt nhất của cặp (scripts/ce2win_prep.py) – bản toàn-nhãn của ce_ftwin.
Bước 0: cặp chưa có cửa sổ → chọn cửa sổ GN tốt nhất (≤6 × ~1600 ký tự, như kaggle_knnpairs), mỗi GPU một nửa.
Mỗi GPU một tiến trình (fork): GPU0 = fold 0,1,2; GPU1 = fold 3,4,-1. Mỗi fold xong ghi /kaggle/working/ce2win_g{gpu}.parquet (qid, doc_id, s, fold).
Model fold -1 (fp16) lưu /kaggle/working/ce2win_all để chấm doc mới về sau."""
import glob
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME, ML, BS, LR = "BAAI/bge-reranker-v2-m3", 384, 16, 1e-5
W_ = "/kaggle/working"
G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
P = pd.read_parquet(G("pairs.parquet")); P["doc_id"] = P.doc_id.astype(str); P["key"] = P.qid.astype(str) + ":" + P.doc_id
Q = pd.read_parquet(G("query.parquet")).set_index("id")["query"]
_c = [p for p in glob.glob("/kaggle/input/**/modules.json", recursive=True)
      if os.path.exists(os.path.join(os.path.dirname(p), "model.safetensors")) and "/models/" not in p]
GN = os.path.dirname(_c[0]) if _c else "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"; MAX_W, TARGET = 6, 1600
tok = AutoTokenizer.from_pretrained(NAME)
AutoModelForSequenceClassification.from_pretrained(NAME)   # tải về cache trước khi fork
t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:.0f}s]", *x, flush=True)


CKPT_EVERY, STOP_AT = 3600, 11.2 * 3600   # checkpoint mỗi giờ; tự dừng trước giới hạn 12h của Kaggle để output được lưu
# tiếp tục từ lần chạy trước: output cũ (tải về → dataset r2ai-ce2win-ckpt) được chép vào /kaggle/working
for f in glob.glob("/kaggle/input/**/ce2win_g*.parquet", recursive=True) + glob.glob("/kaggle/input/**/win_g*.parquet", recursive=True) \
        + glob.glob("/kaggle/input/**/ckpt_g*.pt", recursive=True):
    if not os.path.exists(f"{W_}/{os.path.basename(f)}"):
        os.system(f"cp '{f}' {W_}/")


def batches(df, bs, dev, shuffle, seed=0, skip=0):
    idx = np.random.default_rng(seed).permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(skip * bs, len(df), bs):
        b = df.iloc[idx[i:i + bs]]
        enc = tok(Q.loc[b.qid].tolist(), [TX[k] for k in b.key], truncation="only_second", max_length=ML, padding=True, return_tensors="pt")
        yield b, {k: v.to(dev) for k, v in enc.items()}


def windows(text):
    paras, p0 = [], 0
    for part in text.split("\n\n"):
        s0 = text.find(part, p0); e = s0 + len(part); p0 = e
        if part.strip():
            paras.append((s0, e))
    out, i = [], 0
    while i < len(paras) and len(out) < MAX_W:
        s0, e = paras[i]; j = i
        while e - s0 < TARGET and j + 1 < len(paras):
            j += 1; e = paras[j][1]
        out.append(text[s0:e]); i = j + 1
    return out or [text[:TARGET]]


def win_worker(g, ng):
    from sentence_transformers import SentenceTransformer
    dev = f"cuda:{g}"; F = pd.read_parquet(G("fulldocs.parquet")); F["doc_id"] = F.doc_id.astype(str)
    if os.path.exists(f"{W_}/win_g{g}.parquet"):
        log(f"gpu{g} windows: dùng lại"); return
    M = P[P.wtext.isna()]; docs = sorted(set(M.doc_id)); docs = docs[g::ng]; M = M[M.doc_id.isin(set(docs))]
    FT = dict(zip(F.doc_id, F.text)); WT, WD = [], []
    for d in docs:
        for w in windows(FT[d]):
            WT.append(w); WD.append(d)
    m = SentenceTransformer(GN, model_kwargs={"torch_dtype": torch.float16}, device=dev); m.max_seq_length = 512
    E = torch.tensor(m.encode(WT, normalize_embeddings=True, batch_size=64, show_progress_bar=False)).half().to(dev)
    qs = sorted(set(M.qid)); QE = torch.tensor(m.encode([Q.loc[q] for q in qs], normalize_embeddings=True, batch_size=64)).half().to(dev)
    qi = {q: i for i, q in enumerate(qs)}; wi = pd.Series(range(len(WD))).groupby(np.array(WD)).apply(list).to_dict()
    best = []
    for q, d in zip(M.qid, M.doc_id):
        w = wi[d]; s = (E[w] @ QE[qi[q]]).float().cpu().numpy(); best.append(WT[w[int(s.argmax())]])
    M[["key"]].assign(wtext=best).to_parquet(f"{W_}/win_g{g}.parquet", index=False)
    log(f"gpu{g} windows: docs {len(docs)} windows {len(WT)} pairs {len(M)}")


def worker(g, folds):
    global TX, tr, sc
    TX = dict(zip(P.key, P.wtext)); TX.update(dict(pd.concat([pd.read_parquet(f) for f in glob.glob(f"{W_}/win_g*.parquet")]).values))
    tr = P[P.train]; sc = P[P.score]
    dev = torch.device(f"cuda:{g}"); torch.cuda.set_device(dev); np.random.seed(g)
    OUT = f"{W_}/ce2win_g{g}.parquet"
    done = set(pd.read_parquet(OUT).fold.unique()) if os.path.exists(OUT) else set()
    CK = f"{W_}/ckpt_g{g}.pt"
    for f in folds:
        if f in done:
            log(f"gpu{g} fold {f}: đã xong ở lần chạy trước"); continue
        t1 = time.time()
        m = AutoModelForSequenceClassification.from_pretrained(NAME).to(dev); m.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01); scaler = torch.cuda.amp.GradScaler()
        trf = tr[tr.fold != f] if f >= 0 else tr
        steps = (len(trf) + BS - 1) // BS; sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
        start = 0
        if os.path.exists(CK):
            ck = torch.load(CK, map_location="cpu")
            if ck["fold"] == f:
                m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
                start = ck["step"]; log(f"gpu{g} fold {f}: tiếp tục từ checkpoint step {start}")
            del ck
        m.train(); last = time.time()
        for i, (b, enc) in enumerate(batches(trf, BS, dev, True, seed=100 + f, skip=start), start=start):
            with torch.autocast("cuda", dtype=torch.float16):
                logit = m(**enc).logits.view(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.tensor(b.label.values, device=dev, dtype=torch.float))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if i % 500 == 0:
                log(f"gpu{g} fold {f} step {i}/{steps} loss {loss.item():.3f}")
            stop = time.time() - t0 > STOP_AT
            if time.time() - last > CKPT_EVERY or stop:
                torch.save({"fold": f, "step": i + 1, "model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict()}, CK + ".tmp")
                os.replace(CK + ".tmp", CK); last = time.time(); log(f"gpu{g} fold {f} checkpoint step {i + 1}")
                if stop:
                    log(f"gpu{g} gần hết 12h – dừng, chạy lại kernel với output này để tiếp tục"); return
        m.eval(); m.half()
        part = sc[sc.fold == f]
        part = part.iloc[np.argsort([len(TX[k]) for k in part.key])]   # gom độ dài → ít padding
        s = []
        with torch.no_grad():
            for b, enc in batches(part, 128, dev, False):
                s += m(**enc).logits.view(-1).float().cpu().tolist()
        res = part[["qid", "doc_id"]].assign(s=s, fold=f)
        prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
        pd.concat([prev, res]).to_parquet(OUT, index=False)
        if os.path.exists(CK):
            os.remove(CK)
        if f == -1:
            m.save_pretrained(f"{W_}/ce2win_all"); tok.save_pretrained(f"{W_}/ce2win_all")
        log(f"FOLD {f} done (gpu{g}): train {len(trf)} score {len(part)} in {time.time() - t1:.0f}s")
        del m, opt; torch.cuda.empty_cache()


NG = len(os.popen("nvidia-smi -L").read().strip().splitlines())   # không gọi torch.cuda trước khi fork

if __name__ == "__main__":
    log("pairs", len(P), "train", int(P.train.sum()), "score", int(P.score.sum()), "need window", int(P.wtext.isna().sum()), "gpus", NG)
    ctx = mp.get_context("fork"); ps = [ctx.Process(target=win_worker, args=(g, NG)) for g in range(NG)]
    [p.start() for p in ps]; [p.join() for p in ps]; log("windows done", [p.exitcode for p in ps])
    plan = [[0, 1, 2], [3, 4, -1]] if NG > 1 else [[0, 1, 2, 3, 4, -1]]
    ps = [ctx.Process(target=worker, args=(g, fs)) for g, fs in enumerate(plan)]
    [p.start() for p in ps]; [p.join() for p in ps]
    log("ALLDONE", [p.exitcode for p in ps])
