"""Reorder submission rows by shared chunk bytes, then write default ZIP_LZMA."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import time
import zipfile


WINDOW = 8 << 20


def greedy_order(items, recent=8) -> list[int]:
    """Greedy sum of shared chunk bytes over recent rows; smallest index wins ties."""
    if recent < 1:
        raise ValueError('recent must be positive')
    if not items:
        return []
    keys = []
    inverted = defaultdict(list)
    for i, item in enumerate(items):
        row = {}
        for chunk in item['relevant_chunks']:
            text = chunk['chunk_text'].encode('utf-8')
            row[hashlib.sha1(text).digest()] = len(text)
        keys.append(row)
        for key in row:
            inverted[key].append(i)
    left = set(range(len(items)))
    first = min(left, key=lambda i: (-len(keys[i]), i))
    left.remove(first)
    order = [first]
    history = deque([first], maxlen=recent)
    while left:
        scores = Counter()
        for r in history:
            for key, size in keys[r].items():
                for j in inverted[key]:
                    if j in left:
                        scores[j] += size
        nxt = min(scores, key=lambda i: (-scores[i], i)) if scores else min(left)
        left.remove(nxt)
        order.append(nxt)
        history.append(nxt)
    return order


def window_stats(items, order):
    """Approximate repeated chunk bytes inside/outside an 8 MiB window, as in review."""
    last, pos, near, far = {}, 0, 0, 0
    for i in order:
        item = items[i]
        for chunk in item['relevant_chunks']:
            text = chunk['chunk_text'].encode('utf-8')
            key = hashlib.sha1(text).digest()
            if key in last:
                if pos - last[key] < WINDOW:
                    near += len(text)
                else:
                    far += len(text)
            last[key] = pos
            pos += len(text) + 40
        pos += 12 * len(item['relevant_docs'])
    return near, far


def _write(items, zip_path, order):
    # Same bytes as json.dumps(items, ensure_ascii=False), without a huge temporary buffer.
    count = 0
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_LZMA) as archive:
        with archive.open('submission.json', 'w', force_zip64=True) as output:
            def write(data):
                nonlocal count
                output.write(data)
                count += len(data)
            write(b'[')
            for n, i in enumerate(order):
                if n:
                    write(b', ')
                write(json.dumps(items[i], ensure_ascii=False).encode('utf-8'))
            write(b']')
    return {'zip_bytes': Path(zip_path).stat().st_size, 'json_bytes': count,
            'order_sha': hashlib.sha256(json.dumps(order).encode('ascii')).hexdigest()}


def write_reordered_zip(items, zip_path) -> dict:
    """Write one default-LZMA entry; order_sha is SHA-256 of JSON index array."""
    return _write(items, zip_path, greedy_order(items))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input_zip', type=Path)
    ap.add_argument('output_zip', type=Path)
    args = ap.parse_args()
    if args.input_zip.resolve() == args.output_zip.resolve():
        ap.error('input and output must be different files')
    with zipfile.ZipFile(args.input_zip) as archive:
        items = json.loads(archive.read('submission.json'))
    start = time.perf_counter()
    before = window_stats(items, range(len(items)))
    order = greedy_order(items)
    after = window_stats(items, order)
    print(f'byte lặp trước: trong={before[0]} ngoài={before[1]}', flush=True)
    print(f'byte lặp sau: trong={after[0]} ngoài={after[1]}', flush=True)
    result = _write(items, args.output_zip, order)
    print(json.dumps({**result, 'seconds': round(time.perf_counter() - start, 1)}), flush=True)


if __name__ == '__main__':
    main()
