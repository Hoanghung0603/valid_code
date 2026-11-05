import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

sub_file = sys.argv[1] if len(sys.argv) > 1 else 'submission_bgem3_top100.json'
if sub_file.endswith('.zip'):
    sub_file = sub_file.replace('.zip', '.json')

if not os.path.exists(sub_file):
    print(f"File {sub_file} not found!")
    sys.exit(1)

with open(sub_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Tổng số query: {len(data)}")
doc_lens = [len(x['relevant_docs']) for x in data]
chunk_lens = [len(x['relevant_chunks']) for x in data]
print(f"Docs per query: min={min(doc_lens)}, max={max(doc_lens)}, avg={sum(doc_lens)/len(doc_lens):.1f}")
print(f"Chunks per query: min={min(chunk_lens)}, max={max(chunk_lens)}, avg={sum(chunk_lens)/len(chunk_lens):.1f}")
print(f"Queries with < 100 docs: {sum(1 for l in doc_lens if l < 100)}")
print(f"Queries with < 98 chunks: {sum(1 for l in chunk_lens if l < 98)}")

# Check duplicate docs within any query
dups = sum(1 for x in data if len(set(x['relevant_docs'])) != len(x['relevant_docs']))
print(f"Queries with duplicate docs in relevant_docs: {dups}")

# Check duplicate chunks within any query
dup_chunks = sum(1 for x in data if len(set(c['chunk_text'] for c in x['relevant_chunks'])) != len(x['relevant_chunks']))
print(f"Queries with duplicate chunks in relevant_chunks: {dup_chunks}")

zip_file = sub_file.replace('.json', '.zip')
if os.path.exists(zip_file):
    sz = os.path.getsize(zip_file) / (1024 * 1024)
    print(f"Zip file size: {sz:.2f} MB (An toàn < 100 MB: {sz < 100.0})")
