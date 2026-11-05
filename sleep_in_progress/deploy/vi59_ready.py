"""Check cx_extra with the exact vi57 reader, then write READY.json atomically."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from vi57 import source_parts_vi56
from vi48 import fingerprint
OUT = ROOT.parent/'r2ai_deliver/cand/vi59d'
D8W = 106_030


def main():
    cx = OUT/'cx_extra'
    assert json.loads((OUT/'finish_done.json').read_text(encoding='utf-8'))['done']
    n = sum(1 for _ in source_parts_vi56(cx, 'extra'))
    assert n == D8W, ('d8w count', n)
    zp = OUT/'submission_vi59d.zip'
    report = json.loads((OUT/'build_report.json').read_text(encoding='utf-8'))
    fp = fingerprint(zp)
    assert fp['sha256'] == report['sha256'] and fp['bytes'] == report['bytes'], 'zip changed after build'
    ready = dict(vi59d_zip=dict(path=str(zp), **fp), cx_manifest_sha256=fingerprint(cx/'MANIFEST.json')['sha256'],
                 parts=len(list((cx/'in').glob('qwen.part*.jsonl.zst'))), d8w_pairs=n,
                 written=time.strftime('%Y-%m-%d %H:%M:%S'))
    tmp = cx/'READY.tmp'
    tmp.write_text(json.dumps(ready, indent=2), encoding='utf-8')
    os.replace(tmp, cx/'READY.json')
    print('READY '+json.dumps(ready), flush=True)


if __name__ == '__main__':
    main()
