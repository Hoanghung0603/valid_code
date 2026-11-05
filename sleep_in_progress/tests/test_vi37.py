import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import vi37
import vi23
import vi29
import fuse3


def test_dw_ranks_in_pool_and_missing():
    pool = {'1': [[1, 0, 3], [2, 1, 2], [3, 2, 1]]}
    hits = vi37.docwin_hits(pool, {'1': {'3': 5, '1': 5, '999': 20}})
    assert [h[0] for h in hits['1']] == [1, 3]


def test_disabled_docwin_chain_sanity_and_unchanged_zd():
    pool = {'1': [[1, 0, 3], [2, 1, 2], [3, 2, -4]]}
    q8 = {'1': [[2, 1, 3], [1, 0, 2]]}
    dense, hyde = {'1': [[1, 0, .8]]}, {'1': [[3, 2, .7]]}
    old = vi23.W_Q8B_FHYDE
    vi23.W_Q8B_FHYDE = 2.
    try:
        baseline = vi29.chain_build_v29(pool, q8, dense, hyde, {}, {'1': 0}, {'1': []},
                                        lo=1, hi=1, delta=0, dedup_mode='doc_text')
        with vi37.chain_docwin({'1': {'3': 8}}, enabled=False):
            disabled = vi29.chain_build_v29(pool, q8, dense, hyde, {}, {'1': 0}, {'1': []},
                                            lo=1, hi=1, delta=0, dedup_mode='doc_text')
        assert disabled == baseline
        with vi37.chain_docwin({'1': {'3': 8}}):
            enabled = vi29.chain_build_v29(pool, q8, dense, hyde, {}, {'1': 0}, {'1': []},
                                           lo=1, hi=1, delta=0, dedup_mode='doc_text')
        assert enabled[0] == baseline[0]  # rrf3d2 source for chunks MUST NOT change
        assert enabled[1] != baseline[1]
        assert enabled[3]['n_q_TB'] == baseline[3]['n_q_TB']
    finally:
        vi23.W_Q8B_FHYDE = old


def test_context_restores_on_error():
    original, old = fuse3.fuse3, vi23.W_Q8B_FHYDE
    with pytest.raises(ValueError):
        with vi37.chain_docwin({}):
            raise ValueError('test')
    assert fuse3.fuse3 is original and vi23.W_Q8B_FHYDE == old


def test_parts_contract(tmp_path):
    import zstandard
    import json
    w = vi37.Parts(tmp_path, 'docs_docwin')
    w.add({'id': 1, 'idx': 0, 'text': 'nguyên văn'})
    w.close()
    with w.files[0].open('rb') as f:
        with zstandard.ZstdDecompressor().stream_reader(f) as z:
            assert json.loads(z.read())['text'] == 'nguyên văn'


def test_shard_split_preserves_all_large_query_pairs():
    rows = [[i, i] for i in range(640)]
    shards = [rows[i::3] for i in range(3)]
    assert max(map(len, shards)) <= 500
    assert sorted(h[0] for sh in shards for h in sh) == list(range(640))


def test_required_real_sanity_stage():
    code = Path(vi37.__file__).read_text(encoding='utf-8')
    assert "v29.require_sanity(cmp)" in code
    assert "'vi37_sanity', 56" in code
    assert 'enabled=False' in code


def test_capped_uses_requested_count_not_already_capped_count():
    cands = {'1': [[1, 0, 0, 2., 'same'], [1, 0, 0, 1., 'same'],
                   [2, 0, 0, 0., 'outside']], '2': [[1, 0, 0, 1., 'ok']]}
    lists = {'1': ['1'], '2': ['1']}
    # Query 1 has one eligible candidate, query 2 exactly meets the requested count.
    assert vi37.count_capped(cands, {}, lists, {'1': 2, '2': 1}, 0) == 1


def test_fused_doclist_keeps_reference_source_docs(monkeypatch):
    monkeypatch.setattr(vi37.dw, 'read', lambda p: [
        {'id': 1, 'relevant_chunks': [{'doc_id': 99, 'chunk_text': 'reference'}]}])
    data = {'pool': {'1': [[1, 0, 2.], [2, 1, -4.]]},
            'q8b': {}, 'dense': {}, 'hyde': {}}
    _, lists = vi37.fused_doclist(data, {'1': {'1': 3.}})
    assert lists['1'] == [1, 2, 99]


@pytest.mark.parametrize('rows', [[[1, 0, None]], [[1, 0, float('nan')]],
                                  [[1, 0, float('inf')]], [[2, 0, 1.]],
                                  [[1, 0, 1.], [1, 0, 1.]]])
def test_checked_q8b_rejects_bad_pairs(rows):
    with pytest.raises(AssertionError):
        vi37.checked_q8b({'1': rows}, {'1': [[1, 0]]})


def test_checked_q8b_valid():
    vi37.checked_q8b({'1': [[1, 0, -2.]]}, {'1': [[1, 0]]})


def test_complete_guard_uses_latest_selection(monkeypatch):
    reports = {'q8b_merge_report.json': {'merged_round': 2},
               'q8b_select_round3.json': {'pairs': 0}}
    monkeypatch.setattr(vi37.dw, 'read', lambda p: reports[p.name])
    vi37.require_q8b_complete()
    reports['q8b_select_round3.json']['pairs'] = 1
    with pytest.raises(AssertionError):
        vi37.require_q8b_complete()


def test_qmerge_keeps_first_and_supplementary_rounds(monkeypatch):
    data = {'q8b': {'1': [[1, 0, 1.]]}, 'pool': {'1': [[i, i - 1, 0.] for i in range(1, 5)]}}
    source = {'modal_out/q8b_shard0_out.json': {'1': [[2, 1, 2.]]},
              'modal_out/q8b_shard1_out.json': {'1': [[3, 2, 3.]]},
              'q8b_input/cands_q8b_shard0.json': {'1': [[2, 1]]},
              'q8b_input/cands_q8b_shard1.json': {'1': [[3, 2]]},
              'modal_out/q8b_r2_out.json': {'1': [[4, 3, 4.]]},
              'q8b_input/cands_q8b_r2.json': {'1': [[4, 3]]},
              'q8b_targets_round2.json': {'1': [4]},
              'q8b_input/bench_q8b_old.json': [('1', 1, 1.)] * 200,
              'modal_out/q8b_bench_out.json': {'1': [[1, 0, 1.]]}}
    written = {}
    monkeypatch.setattr(vi37, 'inputs', lambda: data)
    monkeypatch.setattr(vi37.dw, 'read', lambda p: source[p.relative_to(vi37.OUT).as_posix()])
    monkeypatch.setattr(vi37.dw, 'write', lambda p, obj: written.__setitem__(p.name, obj))
    monkeypatch.setattr(vi37.dw, 'overlap_report', lambda pairs: {'pearson': 1.})
    check = vi37.q8b31.check_merged
    monkeypatch.setattr(vi37.q8b31, 'check_merged', lambda out, new, pool: check(out, new, pool, 1))
    vi37.stage_qmerge(2)
    assert [h[0] for h in written['q8b_v37.json']['1']] == [4, 3, 2, 1]
    assert written['q8b_merge_report.json']['round_pairs'] == {'1': 2, '2': 1}
