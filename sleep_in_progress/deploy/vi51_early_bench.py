"""Benchmark 2000 fixed real docs while full CPU prepare continues."""
import json
from pathlib import Path
import random
import sys
import zstandard as zstd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi51 import iterz
from vi48 import fingerprint,write

out=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51/early_bench')
out.mkdir(parents=True,exist_ok=True)
source=out.parent/'modal_input'
rows=list(iterz(source/'docs51.part000.jsonl.zst'))
rowsb=list(iterz(source/'docs51b.part000.jsonl.zst'))
assert rows==rowsb
bench=random.Random(5151).sample(rows,2000)
for name in ('bench.jsonl.zst','bench_b.jsonl.zst'):
    assert not (out/name).exists()
    with zstd.open(out/name,'wt',encoding='utf-8') as f:
        for r in bench:f.write(json.dumps(r)+'\n')
write(out/'manifest.json',dict(documents=2000,tokens=sum(len(r['input_ids']) for r in bench),
    source={p.name:fingerprint(p) for p in (source/'docs51.part000.jsonl.zst',source/'docs51b.part000.jsonl.zst')},
    files={n:fingerprint(out/n) for n in ('bench.jsonl.zst','bench_b.jsonl.zst')}))
print((out/'manifest.json').read_text())
