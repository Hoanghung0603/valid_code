"""Pinned BGE exact-input scorer. Run as one subprocess per Kaggle GPU."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = 'BAAI/bge-reranker-v2-m3'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    with args.input.open('rb') as f:
        while block := f.read(4*1024*1024):
            sha.update(block)
    identity = dict(input_sha256=sha.hexdigest(), model=MODEL, revision=REVISION,
                    shard=args.shard, max_tokens=1024, batch_size=args.batch_size,
                    torch_version=torch.__version__, transformers_version=transformers.__version__,
                    worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), truncation=False)
    config = args.output/'identity.json'
    if config.exists() and json.loads(config.read_text()) != identity:
        raise RuntimeError('Refusing stale checkpoints')
    config.write_text(json.dumps(identity))
    frame = pd.read_parquet(args.input)
    owned = set(sorted(frame.query_id.unique())[args.shard::2])
    frame = frame[frame.query_id.isin(owned)]
    paths = sorted(args.output.glob('part_*.parquet'))
    done = set()
    for path in paths:
        ids = pd.read_parquet(path, columns=['pair_id']).pair_id.tolist()
        if done.intersection(ids) or len(ids) != len(set(ids)):
            raise RuntimeError('Duplicate checkpoint IDs')
        done.update(ids)
    if not done <= set(frame.pair_id):
        raise RuntimeError('Unknown checkpoint IDs')
    pending = frame[~frame.pair_id.isin(done)]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16).to('cuda:0').eval()
    if model.config._commit_hash != REVISION:
        raise RuntimeError('Unexpected model revision')
    records = []
    part = len(paths)
    start_time = time.monotonic()
    for start in range(0, len(pending), args.batch_size):
        batch = pending.iloc[start:start+args.batch_size]
        encoded = tokenizer(batch.query_text.tolist(), batch.passage_text.tolist(),
                            truncation=False, padding=False)
        lengths = [len(ids) for ids in encoded['input_ids']]
        eligible = [i for i, length in enumerate(lengths) if length <= 1024]
        logits = {}
        if eligible:
            selected = [{key: values[i] for key, values in encoded.items()} for i in eligible]
            padded = tokenizer.pad(selected, padding=True, return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                raw = model(**padded).logits.float().reshape(-1).cpu().numpy()
            if not np.isfinite(raw).all():
                raise RuntimeError('Nonfinite model score')
            logits = dict(zip(eligible, raw.tolist()))
        for i, row in enumerate(batch.itertuples(index=False)):
            records.append(dict(pair_id=row.pair_id, status='ok' if i in logits else 'too_long',
                                input_tokens=lengths[i], raw_score=logits.get(i, float('nan')),
                                revision=REVISION, shard=args.shard))
        if len(records) >= 128 or start+args.batch_size >= len(pending):
            target = args.output/f'part_{part:06d}.parquet'
            temporary = target.with_suffix('.tmp')
            pd.DataFrame(records).to_parquet(temporary, index=False)
            os.replace(temporary, target)
            records = []
            part += 1
            completed = min(start+args.batch_size, len(pending))
            print(json.dumps(dict(done=len(done)+completed, total=len(frame),
                                  pairs_s=round(completed/max(time.monotonic()-start_time, .01), 2))), flush=True)
    scores = pd.concat([pd.read_parquet(p) for p in sorted(args.output.glob('part_*.parquet'))])
    if scores.pair_id.duplicated().any() or set(scores.pair_id) != set(frame.pair_id):
        raise RuntimeError('Score coverage failure')
    scores.to_parquet(args.output/'scores.parquet', index=False)


if __name__ == '__main__':
    main()
