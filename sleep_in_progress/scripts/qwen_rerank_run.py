"""Rerank ứng viên doc bằng Qwen/Qwen3-Reranker-0.6B (causal LM, điểm = logit "yes" − logit "no") trên GPU.

Prompt theo model card chính thức (system + <Instruct>/<Query>/<Document> + assistant <think> rỗng).
Chạy lại được: kết quả từng phần (PART_SIZE câu) lưu JSON trong --resume-dir; phần đã có thì bỏ qua.
Đầu ra --out giống rerank_run: {qid: [[doc_id, idx, score], ...]} cùng thứ tự ứng viên cands[:top],
score làm tròn 4 chữ số; doc thiếu text -> None.

Watchdog (mặc định bật; --no-watchdog để bench/test local): giữa các batch, mỗi ~60s đo độ trễ
https://www.vinmec.com/ và nhiệt GPU; chậm > 1s liên tiếp 3 lần thì nghỉ 300s; > 6 lần nghỉ trong
1 giờ thì dừng (mã thoát 3, phần dở bị bỏ, lần sau resume). GPU > 83°C thì nghỉ 120s.

Dùng:
  python scripts/qwen_rerank_run.py --docs chunkdocs100.part000.jsonl.zst ... --queries queries.jsonl \
    --cands cands.json --out qwen_top100.json --resume-dir qr_parts
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from rerank_run import load_docs, merge_parts, merge_scores, part_file, plan_parts

PART_SIZE = 50  # số câu mỗi phần khi lưu resume (như rerank_run)
LOG_EVERY = 25  # in tiến độ mỗi N batch
NAN_SCORE = -1e4  # điểm khi fp32 vẫn ra NaN/inf
DOC_CHARS_PER_TOKEN = 12  # cắt thô text doc trước khi tokenize (chỉ để tiết kiệm CPU)

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = (
    "Given a Vietnamese health question, judge whether the document discusses the same clinical "
    "entity and key clinical detail asked in the question"
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class StopJob(Exception):
    """Watchdog yêu cầu dừng job (độ trễ mạng xấu kéo dài)."""


# ---------------------------------------------------------------- prompt

def _enc(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def build_input_ids(tok, instruction: str, query: str, doc: str, max_len: int,
                    max_query_tokens: int = 256) -> list[int]:
    """Id token của prompt Qwen3-Reranker cho một cặp (query, doc).

    Tokenize riêng PREFIX, SUFFIX, "<Instruct>: ...\\n<Query>: ", query, "\\n<Document>: ", doc
    (add_special_tokens=False). Query cắt còn max_query_tokens; doc cắt để tổng ≤ max_len.
    PREFIX/SUFFIX/instruct không bao giờ bị cắt (nếu riêng chúng + query đã > max_len thì doc rỗng
    và tổng có thể vượt max_len).
    """
    pre = _enc(tok, PREFIX)
    suf = _enc(tok, SUFFIX)
    head = _enc(tok, f"<Instruct>: {instruction}\n<Query>: ")
    q = _enc(tok, query)[:max_query_tokens]
    mid = _enc(tok, "\n<Document>: ")
    fixed = len(pre) + len(head) + len(q) + len(mid) + len(suf)
    d = _enc(tok, doc)[: max(max_len - fixed, 0)]
    return pre + head + q + mid + d + suf


# ---------------------------------------------------------------- watchdog

def measure_vinmec(url: str = "https://www.vinmec.com/", timeout: float = 5.0) -> float:
    """Thời gian (giây) GET url; lỗi mạng/HTTP thì raise (Watchdog tính là chậm)."""
    import urllib.request
    t = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.time() - t


def gpu_temp() -> int | None:
    """Nhiệt GPU (°C) từ nvidia-smi; lỗi thì None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return None


class Watchdog:
    """Kiểm tra độ trễ mạng + nhiệt GPU giữa các batch; mọi phụ thuộc inject được để test."""

    def __init__(self, measure, temp_fn, sleep_fn=time.sleep, time_fn=time.time, log=print,
                 interval=60, slow_s=1.0, slow_consec=3, pause_s=300, max_pauses=6,
                 window_s=3600, temp_max=83, temp_pause_s=120):
        self.measure = measure
        self.temp_fn = temp_fn
        self.sleep = sleep_fn
        self.now = time_fn
        self.log = log
        self.interval = interval
        self.slow_s = slow_s
        self.slow_consec = slow_consec
        self.pause_s = pause_s
        self.max_pauses = max_pauses
        self.window_s = window_s
        self.temp_max = temp_max
        self.temp_pause_s = temp_pause_s
        self.last = None  # thời điểm đo gần nhất
        self.consec = 0  # số lần chậm liên tiếp
        self.pauses: list[float] = []  # thời điểm các lần tạm dừng
        self.n_measures = 0

    def check(self) -> None:
        t = self.now()
        if self.last is not None and t - self.last < self.interval:
            return
        self.last = t
        self.n_measures += 1
        self._check_latency()
        self._check_temp()

    def _check_latency(self) -> None:
        try:
            lat = self.measure()
        except Exception as e:  # lỗi đo = chậm
            lat = None
            err = repr(e)
        else:
            err = None
        if lat is not None and lat <= self.slow_s:
            self.consec = 0
            return
        self.consec += 1
        desc = f"lỗi {err}" if err else ("lỗi (None)" if lat is None else f"{lat:.2f}s")
        self.log(f"watchdog: độ trễ chậm {desc} ({self.consec}/{self.slow_consec})")
        if self.consec < self.slow_consec:
            return
        t = self.now()
        self.pauses = [p for p in self.pauses if t - p < self.window_s]
        self.pauses.append(t)
        if len(self.pauses) > self.max_pauses:
            self.log(f"STOP latency: {len(self.pauses)} lần tạm dừng trong {self.window_s}s")
            raise StopJob("latency")
        self.log(f"watchdog: tạm dừng {self.pause_s}s (lần {len(self.pauses)} trong cửa sổ)")
        self.sleep(self.pause_s)
        self.consec = 0

    def _check_temp(self) -> None:
        try:
            temp = self.temp_fn()
        except Exception:
            temp = None
        if temp is not None and temp > self.temp_max:
            self.log(f"watchdog: GPU {temp}°C > {self.temp_max}°C, nghỉ {self.temp_pause_s}s")
            self.sleep(self.temp_pause_s)


# ---------------------------------------------------------------- model / scoring

def load_model(model_name: str, dtype: str = "fp16", device_map: str | None = None):
    """(tok, model, token_true_id, token_false_id); tokenizer pad trái."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    kwargs = {"torch_dtype": td, "attn_implementation": "sdpa"}
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not device_map:
        model = model.cuda()
    model = model.eval()
    return tok, model, tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")


def score_pairs(model, tok, pairs: list, instruction: str, max_len: int, batch: int,
                true_id: int, false_id: int, progress=None, between=None) -> tuple[list, dict]:
    """Điểm logit_yes − logit_no cho từng cặp (query, doc); trả (scores, stats).

    Sắp theo độ dài để giảm padding, pad trái bằng tok.pad. between() được gọi trước mỗi batch
    (watchdog). progress(done, total, stats) mỗi LOG_EVERY batch và ở batch cuối.

    NaN/inf: batch lỗi được chấm lại ở fp32 bằng cách đổi TẠM model sang fp32 tại chỗ
    (model.float()), chấm, rồi model.half() lại + empty_cache. Chọn cách này thay vì giữ một bản
    fp32 riêng vì an toàn VRAM hơn trên 8GB: đỉnh ~2,4GB trọng số thay vì 1,2GB + 2,4GB thường trực;
    fp16 -> fp32 -> fp16 là khứ hồi chính xác nên trọng số fp16 không đổi. Cặp nào fp32 vẫn NaN/inf
    nhận NAN_SCORE (-1e4). stats: {"nan_batches": số batch chấm lại, "nan_pairs": số cặp -1e4}.
    """
    import torch
    cap = max_len * DOC_CHARS_PER_TOKEN
    enc = [build_input_ids(tok, instruction, q, d[:cap], max_len) for q, d in pairs]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = [0.0] * len(enc)
    stats = {"nan_batches": 0, "nan_pairs": 0}
    _orig_dtype = next(model.parameters()).dtype
    is_half = _orig_dtype in (torch.float16, torch.bfloat16)

    def run(b):
        inp = tok.pad({"input_ids": [enc[i] for i in b]}, padding=True, return_tensors="pt")
        inp = {k: v.cuda() for k, v in inp.items()}
        logits = model(**inp).logits[:, -1, :]
        return (logits[:, true_id].float() - logits[:, false_id].float()).cpu().tolist()

    done, nb = 0, 0
    with torch.no_grad():
        for s in range(0, len(order), batch):
            if between:
                between()
            b = order[s : s + batch]
            sc = run(b)
            if not all(math.isfinite(x) for x in sc):
                stats["nan_batches"] += 1
                if is_half:
                    model.float()
                    try:
                        sc = run(b)
                    finally:
                        model.to(_orig_dtype)
                        torch.cuda.empty_cache()
                bad = [k for k, x in enumerate(sc) if not math.isfinite(x)]
                for k in bad:
                    sc[k] = NAN_SCORE
                stats["nan_pairs"] += len(bad)
            for i, x in zip(b, sc):
                out[i] = float(x)
            done += len(b)
            nb += 1
            if progress and (nb % LOG_EVERY == 0 or done == len(enc)):
                progress(done, len(enc), stats)
    return out, stats


def build_pairs(qtext: str, cand: list, docs: dict, top: int) -> tuple[list, list]:
    """Cặp (query, text doc) cho top ứng viên có text; trả (pairs, vị trí trong cand)."""
    pairs, pos = [], []
    for k, item in enumerate(cand[:top]):
        t = docs.get(item[0])
        if t is None:
            continue
        pairs.append((qtext, t))
        pos.append(k)
    return pairs, pos


def score_stats(scores: list) -> dict:
    """min, p10, p50, p90, max (bỏ None); rỗng -> {}."""
    xs = sorted(x for x in scores if x is not None)
    if not xs:
        return {}

    def pct(p):
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]

    return {"min": xs[0], "p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9), "max": xs[-1]}


# ---------------------------------------------------------------- main

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Rerank bằng Qwen3-Reranker (logit yes − no)")
    ap.add_argument("--docs", required=True, nargs="+", help="JSONL.zst {id, idx, text} (một hoặc nhiều file)")
    ap.add_argument("--queries", required=True, help="JSONL {id, query}")
    ap.add_argument("--cands", required=True, help="JSON {qid: [[doc_id, idx], ...]} theo thứ tự ứng viên")
    ap.add_argument("--top", type=int, default=100, help="rerank N ứng viên đầu mỗi câu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--device-map", default=None, help="device_map cho model (vd cuda)")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--limit-queries", type=int, default=0, help="chỉ chạy N câu đầu (0 = tất cả)")
    ap.add_argument("--no-watchdog", action="store_true", help="tắt watchdog (test/bench local)")
    a = ap.parse_args()

    resume_dir = Path(a.resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    qs = [json.loads(l) for l in Path(a.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    qtext = {str(q["id"]): q["query"] for q in qs}
    cands = json.loads(Path(a.cands).read_text(encoding="utf-8"))
    qids = [q for q in cands if q in qtext]
    if a.limit_queries > 0:
        qids = qids[: a.limit_queries]
    wanted = {item[0] for q in qids for item in cands[q][: a.top]}
    docs = load_docs(a.docs, wanted=wanted)
    print(f"{len(qids)} câu, {len(docs)} doc, tải dữ liệu {time.time() - t0:.0f}s", flush=True)

    plan = plan_parts(qids, PART_SIZE, resume_dir)
    todo = [(pi, chunk) for pi, chunk, done in plan if not done]
    for pi, chunk, done in plan:
        if done:
            print(f"phần {pi}: bỏ qua (đã có {part_file(resume_dir, pi).name})", flush=True)
    total_pairs = sum(
        len(build_pairs(qtext[q], cands[q], docs, a.top)[0]) for _, chunk in todo for q in chunk
    )
    print(f"{len(todo)} phần cần chạy, {total_pairs} cặp", flush=True)

    wd = None if a.no_watchdog else Watchdog(measure_vinmec, gpu_temp, log=lambda m: print(m, flush=True))
    pairs_done = 0
    tot_nan_b, tot_nan_p = 0, 0
    if todo:
        if wd:
            wd.check()  # đo trước khi tải model
        tok, model, true_id, false_id = load_model(a.model, a.dtype, a.device_map)
        import torch
        print(f"load_model {time.time() - t0:.1f}s vram={torch.cuda.memory_allocated() / 2**20:.0f}MB "
              f"yes={true_id} no={false_id} dtype={a.dtype}", flush=True)
        t_run = time.time()
        try:
            for pi, chunk in todo:
                pmap, plist, ppos = {}, [], []
                for q in chunk:
                    pr, pos = build_pairs(qtext[q], cands[q], docs, a.top)
                    pmap[q] = (cands[q], pos)
                    base = len(plist)
                    plist.extend(pr)
                    ppos.append((q, base, base + len(pr)))
                t1 = time.time()

                def on_batch(done_p, np_p, st, _pi=pi):
                    el = time.time() - t_run
                    done = pairs_done + done_p
                    rate = done / max(el, 1e-9)
                    eta = (total_pairs - done) / max(rate, 1e-9)
                    print(f"phần {_pi}: {done}/{total_pairs} cặp, {el:.0f}s, {rate:.1f} cặp/s, ETA {eta:.0f}s, "
                          f"vram={torch.cuda.memory_allocated() / 2**20:.0f}MB, "
                          f"NaN chấm lại {tot_nan_b + st['nan_batches']} batch", flush=True)

                sc, st = score_pairs(model, tok, plist, a.instruction, a.max_len, a.batch,
                                     true_id, false_id, progress=on_batch,
                                     between=wd.check if wd else None)
                tot_nan_b += st["nan_batches"]
                tot_nan_p += st["nan_pairs"]
                out_part = {}
                for q, lo, hi in ppos:
                    cand, pos = pmap[q]
                    nmiss = len(cand[: a.top]) - len(pos)
                    out_part[q] = merge_scores(cand, a.top, pos, sc[lo:hi])
                    if nmiss:
                        print(f"câu {q}: thiếu text {nmiss} doc", flush=True)
                part_file(resume_dir, pi).write_text(json.dumps(out_part), encoding="utf-8")
                pairs_done += len(plist)
                el = time.time() - t1
                ss = score_stats(sc)
                dist = " ".join(f"{k}={v:.3f}" for k, v in ss.items())
                print(f"phần {pi}: ghi {part_file(resume_dir, pi).name}, {len(plist)} cặp, "
                      f"{el:.0f}s, {len(plist) / max(el, 1e-9):.1f} cặp/s; NaN {st['nan_batches']} batch, "
                      f"{st['nan_pairs']} cặp -1e4; điểm {dist}", flush=True)
        except StopJob:
            print(f"STOP latency: dừng sau {pairs_done} cặp; các phần đã xong nằm trong {resume_dir}, "
                  f"phần dở sẽ chạy lại khi resume", flush=True)
            sys.exit(3)

    out = merge_parts(plan, resume_dir)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    allsc = [r[2] for v in out.values() for r in v]
    dist = " ".join(f"{k}={v:.3f}" for k, v in score_stats(allsc).items())
    print(f"ghi {a.out}: {len(out)} câu; {pairs_done} cặp mới; NaN chấm lại {tot_nan_b} batch, "
          f"{tot_nan_p} cặp -1e4; tổng {time.time() - t0:.0f}s", flush=True)
    print(f"phân bố điểm (toàn bộ đầu ra): {dist}", flush=True)


if __name__ == "__main__":
    main()
