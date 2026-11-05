import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import vi49 as v


def chunk(d, text):
    return {'doc_id': d, 'chunk_text': text}


def candidate(d, text, q=1):
    c = chunk(d, text)
    return {**c, 'cid': v.cid(q, c)}


def base(q=1):
    return {'id': q, 'relevant_docs': [10, 20],
            'relevant_chunks': [chunk(10, 'old')], 'extra': 'unchanged'}


def test_nonpositive_scored_only_doc_gate_and_baseline_identity():
    r = base()
    a, excluded = v.select_row(r, [(candidate(10, 'old'), -99),
        (candidate(10, 'positive'), .01), (candidate(10, 'zero'), 0),
        (candidate(20, 'negative'), -1), (candidate(99, 'outside'), -1)])
    assert [c['chunk_text'] for c in a] == ['zero', 'negative']
    assert excluded == 1 and r == base()


def test_global_order_not_per_query_quota_and_ties():
    a, _ = v.select_row(base(), [(candidate(10, 'a'), -1), (candidate(20, 'b'), 0)])
    b, _ = v.select_row(base(2), [(candidate(10, 'c', 2), -.5), (candidate(20, 'd', 2), -1)])
    ranked = v.global_order({'1': a, '2': b})
    assert [c['logit8c'] for c in ranked] == [0, -.5, -1, -1]
    assert ranked[2]['cid'] < ranked[3]['cid']
    sub = v.append_prefix([base(), base(2)], ranked, 2)
    assert sub[0]['relevant_chunks'][-1] == chunk(20, 'b')
    assert sub[1]['relevant_chunks'][-1] == chunk(10, 'c')
    assert v.check_relations(sub, [base(), base(2)], ranked, 2)['vi48_literal_prefix'] == 2


@pytest.mark.parametrize('score', [math.nan, math.inf, -math.inf])
def test_nonfinite_rejected(score):
    with pytest.raises(ValueError, match='nonfinite'):
        v.select_row(base(), [(candidate(10, 'new'), score)])


def test_corrupt_duplicate_cid_or_bad_baseline_rejected():
    c = candidate(10, 'new')
    for pairs in [[(dict(c, cid='bad'), 0)], [(c, 0), (c, 0)]]:
        with pytest.raises(ValueError, match='cid'):
            v.select_row(base(), pairs)
    r = base()
    r['relevant_chunks'].append(chunk(10, 'old'))
    with pytest.raises(ValueError, match='duplicate baseline'):
        v.select_row(r, [])
    r['relevant_chunks'] = [chunk(99, 'outside')]
    with pytest.raises(ValueError, match='outside docs'):
        v.select_row(r, [])


def test_string_doc_keys_and_doc_sensitive_dedup():
    r = base()
    r['relevant_docs'] = ['10', '20']
    a, excluded = v.select_row(r, [(candidate('10', 'old'), 0), (candidate(20, 'old'), -1)])
    assert len(a) == 1 and excluded == 0
    assert v.append_prefix([r], a, 1)[0]['relevant_docs'] == ['10', '20']


def test_equal_logit_global_cut_keeps_ascending_cid_across_queries():
    additions = {}
    for q in [1, 2, 3]:
        additions[str(q)], _ = v.select_row(base(q), [(candidate(20, f'new{q}', q), 0)])
    ranked = v.global_order(additions)
    assert [c['cid'] for c in ranked] == sorted(c['cid'] for c in ranked)
    sub = v.append_prefix([base(1), base(2), base(3)], ranked, 1)
    changed = [str(r['id']) for r in sub if len(r['relevant_chunks']) == 2]
    assert changed == [ranked[0]['qid']]


def test_relations_reject_any_prefix_doc_suffix_or_extra_change():
    a, _ = v.select_row(base(), [(candidate(20, 'new'), 0)])
    for mutate in [lambda r: r['relevant_docs'].reverse(),
                   lambda r: r['relevant_chunks'].reverse(),
                   lambda r: r.update(extra='different')]:
        sub = v.append_prefix([base()], a, 1)
        sub = copy.deepcopy(sub)
        mutate(sub[0])
        with pytest.raises(AssertionError):
            v.check_relations(sub, [base()], a, 1)


def test_budget_certifies_adjacent_real_prefix_and_restores_zip():
    a, _ = v.select_row(base(), [(candidate(10, str(i)), -i) for i in range(10)])
    calls = []
    def measure(sub, n):
        calls.append(n)
        assert sub[0]['relevant_chunks'][:1] == base()['relevant_chunks']
        return {'zip_bytes': 100+n*100}
    sub, n, attempts = v.fit_budget([base()], a, measure, 100, limit=650)
    assert n == 5 and calls[-1] == 5
    assert any(x['count'] == 6 and x['zip_bytes'] == 700 for x in attempts)
    assert len(sub[0]['relevant_chunks']) == 6


def test_budget_exact_limit_and_all_candidates():
    a, _ = v.select_row(base(), [(candidate(10, 'new'), 0)])
    sub, n, attempts = v.fit_budget([base()], a, lambda s, n: {'zip_bytes': 200}, 100, 200)
    assert n == 1 and len(attempts) == 1
    assert len(sub[0]['relevant_chunks']) == 2


def test_budget_no_candidate_fits_restores_baseline_without_removing_it():
    a, _ = v.select_row(base(), [(candidate(10, 'new'), 0)])
    calls = []
    def measure(sub, n):
        calls.append(n)
        return {'zip_bytes': 100 if n == 0 else 201}
    sub, n, attempts = v.fit_budget([base()], a, measure, 100, 200)
    assert n == 0 and sub == [base()] and calls == [1, 0]
    assert attempts[-1]['restore']


def test_budget_limited_to_adjacent_certificate_not_global_monotonicity():
    a, _ = v.select_row(base(), [(candidate(10, str(i)), -i) for i in range(4)])
    sizes = {0: 100, 1: 150, 2: 201, 3: 190, 4: 250}
    sub, n, attempts = v.fit_budget([base()], a, lambda s, n: {'zip_bytes': sizes[n]},
                                   100, 200)
    # Sequential rule stops before first overflowing chunk, even if a later
    # larger prefix could compress smaller; no skipping or selecting later rows.
    assert n == 1
    assert any(t['count'] == 2 and t['zip_bytes'] == 201 for t in attempts)
    assert len(sub[0]['relevant_chunks']) == 2


def test_budget_empty_baseline_overflow_and_iteration_guard():
    assert v.fit_budget([base()], [], lambda s, n: {'zip_bytes': 100}, 100, 100)[1] == 0
    with pytest.raises(ValueError, match='baseline alone'):
        v.fit_budget([base()], [], None, 101, 100)
    a, _ = v.select_row(base(), [(candidate(10, str(i)), -i) for i in range(10)])
    with pytest.raises(RuntimeError, match='not certified'):
        v.fit_budget([base()], a, lambda s, n: {'zip_bytes': 100+n*100}, 100, 650, 1)


def test_real_1200_validate_clean_zip_and_verbatim(tmp_path):
    import subprocess
    import zipfile
    import zstandard as zstd
    from reorder_zip import write_reordered_zip
    from vi23 import check_zip
    from vi29 import assert_clean
    refs, additions = [], {}
    for q in range(1, 1201):
        refs.append(base(q))
        additions[str(q)], _ = v.select_row(refs[-1], [(candidate(20, 'nguyên văn', q), -q)])
    ranked = v.global_order(additions)
    sub = v.append_prefix(refs, ranked, 600)
    zp, jp = tmp_path/'submission.zip', tmp_path/'submission.json'
    write_reordered_zip(sub, zp)
    with zipfile.ZipFile(zp) as z:
        assert len(z.infolist()) == 1 and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        jp.write_bytes(z.read('submission.json'))
    subprocess.run([sys.executable, str(Path(v.__file__).with_name('evaluate.py')),
                    'validate', str(jp)], check=True)
    assert check_zip(str(jp), str(zp), str(tmp_path/'check.json'), 'test_vi49', max_zip=v.LIMIT) == 0
    assert assert_clean(sub, {99}, {10, 20})['cap_doc_text_trung'] == 0
    assert v.check_relations(v.load_zip(zp), refs, ranked, 600)['docs_equal_vi48'] == 1200
    text = tmp_path/'texts.jsonl.zst'
    with zstd.open(text, 'wt', encoding='utf-8') as f:
        for d, t in [(10, 'old'), (20, 'prefix nguyên văn suffix')]:
            f.write(json.dumps({'id': d, 'text': t})+'\n')
    assert v.verify_verbatim(sub, [text])['verbatim_all_final_chunks'] == 1800
