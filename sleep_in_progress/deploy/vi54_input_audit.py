"""Read-only audit: prove whether vi51 artifacts can meet vi54 tokenization contract."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from vi51 import iterz
from vi48 import fingerprint

TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
SOURCE = TEMP/'vi51/modal_input'
OUT = TEMP/'vi54'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = sorted(SOURCE.glob('docs51.part*.jsonl.zst'))
    prep = json.loads((TEMP/'vi51/prepare_report.json').read_text())
    counts = Counter()
    keys = Counter()
    files = {}
    seen = set()
    for part in parts:
        before = fingerprint(part)
        assert before == {k: prep['files'][part.name][k] for k in before}
        for row in iterz(part):
            assert row['id'] not in seen
            seen.add(row['id'])
            counts['documents'] += 1
            keys[','.join(sorted(row))] += 1
            counts['with_raw_title_and_text'] += int('title' in row and 'text' in row)
            counts['at_qwen_2048_cap'] += int(len(row['input_ids']) == 2048)
        files[part.name] = before
    assert len(parts) == 29 and len(seen) == 223065
    queries = json.loads((SOURCE/'queries_b.json').read_text())
    pool = json.loads((SOURCE/'pool.json').read_text())
    report = dict(parts=len(parts), **counts, schemas=dict(keys),
                  query_keys=sorted(queries[0]), query_count=len(queries),
                  pairs=sum(map(len, pool.values())), source_files=files,
                  status='STOP_INPUT_PROVENANCE',
                  reason='vi51 input contains only Qwen token IDs, already truncated at 2048. Cannot apply exact Nemotron tokenizer to original title+clean text, nor recover original query from instructed/truncated query tokens. Re-export is outside explicit reuse/no-rebuild authorization. No GPU launched.')
    (OUT/'input_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_files'}), flush=True)


if __name__ == '__main__':
    main()
