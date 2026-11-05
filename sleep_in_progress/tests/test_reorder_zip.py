import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reorder_zip import WINDOW, greedy_order, window_stats, write_reordered_zip


def row(i, *texts):
    return {'id': i, 'relevant_docs': [2, 1], 'relevant_chunks': [
        {'doc_id': 1, 'chunk_text': text, 'chunk_order': n} for n, text in enumerate(texts)]}


def test_order_and_roundtrip(tmp_path):
    items = [row(1, 'á', 'x'), row(2, 'x'), row(3, 'á'), row(4, 'khác')]
    assert greedy_order(items) == [0, 2, 1, 3]  # UTF-8 bytes, not character count
    result = write_reordered_zip(items, tmp_path / 'out.zip')
    with zipfile.ZipFile(tmp_path / 'out.zip') as z:
        assert z.namelist() == ['submission.json']
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        raw = z.read('submission.json')
    assert raw == json.dumps([items[i] for i in greedy_order(items)], ensure_ascii=False).encode('utf-8')
    assert {x['id']: x for x in json.loads(raw)} == {x['id']: x for x in items}
    assert result['json_bytes'] == len(raw)
    assert result['zip_bytes'] == (tmp_path / 'out.zip').stat().st_size
    assert result['order_sha'] == hashlib.sha256(json.dumps(greedy_order(items)).encode('ascii')).hexdigest()


def test_ties_fallback_empty_and_seed_independence():
    items = [row(1, 'x', 'y'), row(2, 'y'), row(3, 'x'), row(4, 'z')]
    assert greedy_order(items) == [0, 1, 2, 3]
    assert greedy_order([]) == []
    with pytest.raises(ValueError):
        greedy_order(items, recent=0)
    script = "import json; from reorder_zip import greedy_order; print(greedy_order(json.loads(%r)))" % json.dumps(items)
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'scripts')}
    results = [subprocess.check_output([sys.executable, '-c', script], env={**env, 'PYTHONHASHSEED': seed})
               for seed in ('1', '17', 'random')]
    assert len(set(results)) == 1
    assert window_stats(items, range(len(items))) == (2, 0)


def test_cli_reports_stats_and_rejects_input_as_output(tmp_path):
    source = tmp_path / 'source.zip'
    dest = tmp_path / 'reordered.zip'
    items = [row(1, 'văn bản'), row(2, 'văn bản')]
    with zipfile.ZipFile(source, 'w', zipfile.ZIP_LZMA) as archive:
        archive.writestr('submission.json', json.dumps(items, ensure_ascii=False).encode('utf-8'))
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'reorder_zip.py'
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    result = subprocess.run([sys.executable, str(script), str(source), str(dest)],
                            capture_output=True, text=True, encoding='utf-8', env=env, check=True)
    assert 'byte lặp trước:' in result.stdout and 'byte lặp sau:' in result.stdout
    assert 'zip_bytes' in result.stdout and 'json_bytes' in result.stdout
    original = source.read_bytes()
    rejected = subprocess.run([sys.executable, str(script), str(source), str(source)],
                              capture_output=True, env=env)
    assert rejected.returncode != 0 and source.read_bytes() == original


def test_window_far_and_history_size():
    items = [row(1, 'x', 'y', 'z'), row(2, 'x', 'a'), row(3, 'y'), row(4, 'a')]
    assert greedy_order(items, recent=1) == [0, 1, 3, 2]
    assert greedy_order(items, recent=2) == [0, 1, 2, 3]
    assert window_stats([row(1, 'x'), row(2, 'a' * WINDOW), row(3, 'x')], range(3)) == (0, 1)
