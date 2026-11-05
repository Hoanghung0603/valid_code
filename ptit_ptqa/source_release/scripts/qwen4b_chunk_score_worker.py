"""GPU worker for exact, non-truncated Qwen4B chunk scoring."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", required=True); p.add_argument("--revision", required=True)
    p.add_argument("--gpu", type=int, required=True); p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=4096); p.add_argument("--instruction", required=True)
    a = p.parse_args()
    torch.cuda.set_device(0); device = torch.device("cuda:0")
    frame = pd.read_parquet(a.input).reset_index(drop=True)
    if "assigned_gpu" in frame.columns:
        assigned = set(frame.assigned_gpu.astype(int).unique())
        if not assigned <= set(range(a.world_size)):
            raise RuntimeError(f"Invalid assigned_gpu values: {assigned}")
        frame = frame[frame.assigned_gpu.astype(int).eq(a.gpu)].copy().reset_index(drop=True)
    else:
        qids = sorted(frame.query_id.astype(int).unique())
        owned = {qid for i, qid in enumerate(qids) if i % a.world_size == a.gpu}
        frame = frame[frame.query_id.isin(owned)].copy().reset_index(drop=True)
    a.out.mkdir(parents=True, exist_ok=True)
    checkpoint = a.out / f"qwen4b_gpu{a.gpu}.parquet"
    done = set()
    old = None
    if checkpoint.exists():
        old = pd.read_parquet(checkpoint); done = set(old.pair_id.astype(str))
    pending = frame[~frame.pair_id.astype(str).isin(done)].copy().reset_index(drop=True)
    print(json.dumps({"event":"start","gpu":a.gpu,"rows":len(frame),"pending":len(pending)}), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model_id, revision=a.revision, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        a.model_id, revision=a.revision, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    yes_id = tok.convert_tokens_to_ids("yes"); no_id = tok.convert_tokens_to_ids("no")
    prefix = tok.encode(PREFIX, add_special_tokens=False); suffix = tok.encode(SUFFIX, add_special_tokens=False)
    revision = getattr(model.config, "_commit_hash", None) or a.revision

    # Qwen3ForCausalLM otherwise projects every sequence position to the full
    # 151k-token vocabulary even though reranking only consumes the last token.
    # Verify once that last-token-only logits are numerically identical before
    # using the cheaper path for the full shard.
    if len(pending):
        probe = pending.sort_values("passage_chars").iloc[0]
        probe_text = f"<Instruct>: {a.instruction}\n<Query>: {probe.query_text}\n<Document>: {probe.passage_text}"
        probe_ids = prefix + tok.encode(probe_text, add_special_tokens=False) + suffix
        if len(probe_ids) > a.max_length:
            raise RuntimeError("Shortest optimization probe exceeds max_length")
        probe_tensor = torch.tensor([probe_ids], dtype=torch.long, device=device)
        probe_mask = torch.ones_like(probe_tensor)
        with torch.inference_mode():
            full_last = model(input_ids=probe_tensor, attention_mask=probe_mask).logits[:, -1, [no_id, yes_id]].float()
            kept_last = model(
                input_ids=probe_tensor, attention_mask=probe_mask, logits_to_keep=1
            ).logits[:, -1, [no_id, yes_id]].float()
        probe_diff = float((full_last - kept_last).abs().max().item())
        if probe_diff > 1e-5:
            raise RuntimeError(f"logits_to_keep self-check failed: {probe_diff}")
        print(json.dumps({"event":"last_token_logits_self_check","max_abs_diff":probe_diff,
                          "probe_tokens":len(probe_ids)}), flush=True)
        del full_last, kept_last, probe_tensor, probe_mask
        torch.cuda.empty_cache()

    rows = [] if old is None else old.to_dict("records")
    started = time.time(); since_save = 0
    with torch.inference_mode():
        for index, row in enumerate(pending.itertuples(index=False), 1):
            formatted = f"<Instruct>: {a.instruction}\n<Query>: {row.query_text}\n<Document>: {row.passage_text}"
            body = tok.encode(formatted, add_special_tokens=False)
            ids = prefix + body + suffix
            if len(ids) > a.max_length:
                raw = norm = None; status = "too_long"
            else:
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                logits = model(
                    input_ids=tensor, attention_mask=torch.ones_like(tensor), logits_to_keep=1
                ).logits[:, -1, :].float()
                raw = float((logits[:, yes_id] - logits[:, no_id]).item())
                norm = float(torch.sigmoid(torch.tensor(raw)).item()); status = "ok"
            rows.append({
                "pair_id": str(row.pair_id), "query_id": int(row.query_id),
                "chunk_ordinal": int(row.chunk_ordinal), "doc_id": int(row.doc_id),
                "raw_score": raw, "normalized_score": norm, "input_tokens": len(ids),
                "status": status, "model_revision": revision, "gpu_shard": a.gpu,
                "query_sha256": str(row.query_sha256), "passage_sha256": str(row.passage_sha256),
            })
            since_save += 1
            if since_save >= 128 or index == len(pending):
                pd.DataFrame(rows).to_parquet(checkpoint, index=False, compression="zstd"); since_save = 0
            if index % 100 == 0 or index == len(pending):
                elapsed = time.time() - started
                print(json.dumps({"event":"progress","gpu":a.gpu,"complete":index,"total":len(pending),
                                  "pairs_s":round(index/max(elapsed,1e-6),3),"status":status}), flush=True)
    result = pd.read_parquet(checkpoint)
    if set(result.pair_id.astype(str)) != set(frame.pair_id.astype(str)):
        raise RuntimeError("Checkpoint coverage mismatch")
    del model; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"event":"complete","gpu":a.gpu,"rows":len(result),
                      "statuses":result.status.value_counts().to_dict()}), flush=True)


if __name__ == "__main__":
    main()
