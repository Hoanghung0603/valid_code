import importlib.util
from pathlib import Path
import sys
import types

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('dwmax_diag', SCRIPTS / 'dwmax_diag.py')
dw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dw)


def test_maxima_and_current_text_rule():
    merged, owners = {}, {}
    allowed = {'1': {'10', '20', '30'}}
    old = dw.maxima({'1': {'10': [[0, 5, 9], [4, 9, 10]], '20': [[0, 1, 8]]}})
    st, _ = dw.merge_layer(merged, owners, old, 'old', {'20'}, allowed)
    assert st['excluded'] == 1
    _, overlaps = dw.merge_layer(merged, owners, {'1': {'10': 2, '20': 3}},
                                'new', set(), allowed)
    assert merged == {'1': {'10': 2, '20': 3}}
    assert overlaps == {'old -> new': [(10, 2)]}


@pytest.mark.parametrize('n, expected', [(0, 0), (1, 1), (300, 1), (301, 2),
                                         (550, 2), (551, 3), (800, 3)])
def test_window_count(n, expected):
    assert dw.window_count(' x\n' * n) == expected


def test_tied_spearman_and_empty():
    assert dw.spearman([1, 1, 2], [2, 2, 1]) == pytest.approx(-1)
    assert dw.spearman([1, 1], [2, 3]) is None
    assert dw.dist([]) == {'n': 0, 'mean': None}
    assert dw.coverage({'1': ['1', '2']}, {'1': {'1': 3}})['pct'] == 50


def test_rrf_addition_and_refs():
    pool = {'1': [[1, 0, 3], [2, 1, 2], [3, 2, 1]]}
    docs, fused = dw.simulate(pool, {}, {}, {}, {'1': {'3': 8}}, {'1': 1}, {'1': ['2']})
    assert docs == {'1': ['3', '2']}
    assert fused['1'][0][0] == 3
    # Missing docs get no bonus; baseline 8B weight is explicitly TWO.
    docs, _ = dw.simulate(pool, {'1': [[2, 1, 4]]}, {}, {}, {}, {'1': 1}, {})
    assert docs == {'1': ['2']}


def test_exchange_controls_and_rank():
    fh = {'1': [[1, 0, 1], [2, 1, .9], [3, 2, .8]]}
    rep, pairs = dw.comparison({'1': ['1', '2']}, {'1': ['1', '3']}, fh,
                               {'ctrl': {'1': ['3']}})
    assert rep['jaccard_macro'] == pytest.approx(1/3)
    assert rep['added_per_query']['mean'] == 1
    assert rep['incoming_original_rank']['p50'] == 3
    assert rep['unlabeled_controls']['ctrl']['in']['fraction'] == 1
    assert pairs['1']['out'] == ['2']


def test_owned_lock_cleanup_and_foreign_lock(tmp_path):
    path = tmp_path / 'locks/heavy.lock'
    with pytest.raises(ValueError):
        with dw.owned_lock(path, 'ours'):
            raise ValueError('test')
    assert not path.exists()
    path.write_text('foreign')
    with pytest.raises(FileExistsError):
        with dw.owned_lock(path, 'ours'):
            pass
    assert path.read_text() == 'foreign'


def test_forbidden_crawl_guard(tmp_path):
    dw.write(tmp_path / 'meta.json', {'shards': ['D:\\r2ai\\data\\crawl\\x.zst']})
    with pytest.raises(RuntimeError, match='Forbidden'):
        dw.safe_index(tmp_path)


def test_prepare_export_is_opt_in():
    code = (SCRIPTS / 'dwmax_diag.py').read_text(encoding='utf-8')
    assert 'modal run' not in code
    assert '--prepare-export' in code


def test_correlation_intersection_and_ties():
    rep = dw.correlation_report({'1': ['1', '2', '3', '4']},
                                {'1': {'1': 5, '2': 5, '3': 1}},
                                {'1': [[1, 0, 5], [2, 1, 5], [4, 3, 8]]})
    assert rep['common_pairs'] == 2
    assert rep['queries'] == 0  # constant tied scores => undefined, not zero
    assert rep['top30_overlap_fraction_macro'] == 1


def test_offline_without_dw_reproduces_weighted_fusion():
    from fuse3 import fuse3
    pool = {'1': [[1, 0, 3], [2, 1, 2], [3, 2, 1]]}
    q8 = {'1': [[3, 2, 4], [2, 1, 3]]}
    dense = {'1': [[2, 1, .9]]}
    hyde = {'1': [[1, 0, .8]]}
    docs, fused = dw.simulate(pool, q8, dense, hyde, {}, {'1': 2}, {})
    assert fused == fuse3(pool, q8, dense, (1, 2, 2, 1), 60, hyde)
    assert docs['1'] == [str(h[0]) for h in fused['1'][:2]]


def test_prepare_export_contract(tmp_path, monkeypatch):
    import json
    import pyarrow.parquet as pq
    import zstandard
    index = tmp_path / 'data/index/vi36'
    out = tmp_path / 'data/sub/diag37'
    dw.write(index / 'meta.json', {'shards': ['D:/r2ai/data/corpus/safe.zst']})
    dw.write(index / 'boilerplate.json', {})
    dw.write(out / 'cands_docwin_missing_top250.json', {'1': [[10, 0], [20, 1]]})
    fake_index = types.ModuleType('bigindex')
    fake_index.BigIndex = lambda path: object()
    fake_bp = types.ModuleType('boilerplate')
    fake_bp.BoilerplateFilter = types.SimpleNamespace(from_json=lambda s: object())
    fake_export = types.ModuleType('export_docs')
    def rows(idx, wanted, bp, chars, no_title=False):
        assert chars == 200000 and no_title
        assert wanted == {10: 0, 20: 1}
        yield 10, 0, 'nguyên văn'
        yield 20, 1, ''
    fake_export.iter_all_texts = rows
    for name, mod in [('bigindex', fake_index), ('boilerplate', fake_bp), ('export_docs', fake_export)]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pq, 'read_table', lambda p: types.SimpleNamespace(
        to_pydict=lambda: {'id': [1], 'query': ['câu hỏi']}))
    dw.prepare_export(tmp_path, out)
    dest = out / 'modal_input'
    with (dest / 'docs_docwin.part000.jsonl.zst').open('rb') as f:
        with zstandard.ZstdDecompressor().stream_reader(f) as z:
            records = [json.loads(s) for s in z.read().decode().splitlines()]
    assert records == [{'id': 10, 'idx': 0, 'text': 'nguyên văn'}, {'id': 20, 'idx': 1, 'text': ''}]
    assert dw.read(dest / 'export_report.json')['modal_launched'] is False
    assert dw.read(dest / 'cands_docwin.json') == {'1': [[10, 0], [20, 1]]}
