import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import vi48 as v


def chunk(d, text):
    return {'doc_id': d, 'chunk_text': text}


def candidate(d, text, q=1):
    c = chunk(d, text)
    c['cid'] = v.cid(q, c)
    return c


def base():
    return {'id': 1, 'relevant_docs': [10, 20],
            'relevant_chunks': [chunk(10, 'old')], 'extra_field': 'preserved'}


def test_strict_positive_scored_only_and_doc_gate():
    r = base()
    extras, excluded = v.select_row(r, [(candidate(10, 'old'), 100),
        (candidate(10, 'zero'), 0), (candidate(10, 'negative'), -1),
        (candidate(10, 'new'), .01), (candidate(99, 'outside'), 2)])
    assert [c['chunk_text'] for c in extras] == ['new']
    assert len(excluded) == 1 and excluded[0]['doc_id'] == 99
    assert r == base()


def test_sort_descending_logit_and_deterministic_ties():
    pairs = [(candidate(20, 'b'), 1), (candidate(10, 'c'), 5), (candidate(10, 'a'), 1)]
    a, _ = v.select_row(base(), pairs)
    b, _ = v.select_row(base(), reversed(pairs))
    assert a == b
    assert a[0]['chunk_text'] == 'c'
    assert a[1]['cid'] < a[2]['cid']


def test_doc_sensitive_dedup_and_literal_prefix():
    r = base()
    extras, _ = v.select_row(r, [(candidate(20, 'old'), 1), (candidate(10, 'old'), 2)])
    additions = {'1': extras}
    sub = v.append([r], additions)
    assert sub[0]['relevant_chunks'] == [chunk(10, 'old'), chunk(20, 'old')]
    assert v.check_relations(sub, [r], additions)['vi45_literal_prefix'] == 1


@pytest.mark.parametrize('score', [math.nan, math.inf, -math.inf])
def test_nonfinite_rejected(score):
    with pytest.raises(ValueError, match='nonfinite'):
        v.select_row(base(), [(candidate(10, 'a'), score)])


def test_corrupt_or_duplicate_cid_rejected():
    c = candidate(10, 'a')
    with pytest.raises(ValueError, match='cid'):
        v.select_row(base(), [(dict(c, cid='bad'), 1)])
    with pytest.raises(ValueError, match='cid'):
        v.select_row(base(), [(c, 1), (c, 1)])


def test_duplicate_base_and_outside_doc_rejected():
    r = base()
    r['relevant_chunks'].append(chunk(10, 'old'))
    with pytest.raises(ValueError, match='duplicate baseline'):
        v.select_row(r, [])
    r = base()
    r['relevant_chunks'].append(chunk(99, 'x'))
    with pytest.raises(ValueError, match='outside docs'):
        v.select_row(r, [])


def test_budget_no_drop_when_fits():
    a, _ = v.select_row(base(), [(candidate(10, 'a'), 1)])
    sub, removed, attempts = v.fit_budget([base()], {'1': a}, lambda s: {'zip_bytes': 99}, 99)
    assert not removed and len(attempts) == 1
    assert len(sub[0]['relevant_chunks']) == 2


def test_budget_global_lowest_one_at_a_time_no_monotonicity_assumption():
    r2 = {'id': 2, 'relevant_docs': [10], 'relevant_chunks': []}
    a, _ = v.select_row(base(), [(candidate(10, 'a'), 3), (candidate(10, 'b'), 1)])
    b, _ = v.select_row(r2, [(candidate(10, 'c', q=2), 2)])
    calls = []
    def measure(sub):
        calls.append(copy.deepcopy(sub))
        return {'zip_bytes': [110, 111, 99][len(calls)-1]}
    sub, removed, attempts = v.fit_budget([base(), r2], {'1': a, '2': b}, measure, 99)
    assert removed == {v.cid(1, chunk(10, 'b')), v.cid(2, chunk(10, 'c'))}
    assert [r['removed'] for r in attempts] == [0, 1, 2]
    assert sub[0]['relevant_chunks'] == [chunk(10, 'old'), chunk(10, 'a')]
    assert sub[1]['relevant_chunks'] == []


def test_budget_baseline_overflow_stops():
    with pytest.raises(ValueError, match='baseline alone'):
        v.fit_budget([base()], {'1': []}, lambda s: {'zip_bytes': 101}, 100)


def test_budget_equal_logits_keeps_ascending_cid_tie_order():
    a, _ = v.select_row(base(), [(candidate(10, 'a'), 1), (candidate(20, 'b'), 1)])
    calls = []
    def measure(sub):
        calls.append(sub)
        return {'zip_bytes': 101 if len(calls) == 1 else 100}
    sub, removed, _ = v.fit_budget([base()], {'1': a}, measure, 100)
    assert removed == {max(c['cid'] for c in a)}
    assert sub[0]['relevant_chunks'][-1]['chunk_text'] == a[0]['chunk_text']


def test_string_doc_normalization_does_not_change_base_fields():
    r = base()
    r['relevant_docs'] = ['10', '20']
    a, rejected = v.select_row(r, [(candidate('10', 'old'), 5), (candidate(20, 'new'), 1)])
    assert not rejected and len(a) == 1
    sub = v.append([r], {'1': a})
    assert sub[0]['relevant_docs'] == ['10', '20']
    assert v.check_relations(sub, [r], {'1': a})['docs_equal_vi47'] == 1


def test_relations_reject_prefix_doc_or_suffix_changes():
    a, _ = v.select_row(base(), [(candidate(10, 'a'), 1)])
    expected = v.append([base()], {'1': a})
    for mutate in [lambda r: r['relevant_docs'].reverse(),
                   lambda r: r['relevant_chunks'].reverse(),
                   lambda r: r['relevant_chunks'].append(chunk(10, 'x')),
                   lambda r: r.update(extra_field='changed')]:
        sub = copy.deepcopy(expected)
        mutate(sub[0])
        with pytest.raises(AssertionError):
            v.check_relations(sub, [base()], {'1': a})


def test_distribution_linear_quantile_and_empty():
    d = v.distribution([0, 1, 2, 3])
    assert d['p50'] == 1.5 and d['p90'] == pytest.approx(2.7)
    assert v.distribution([])['count'] == 0


def test_verbatim_full_export_and_missing_or_nonverbatim(tmp_path):
    import zstandard as zstd
    p = tmp_path/'text.jsonl.zst'
    def write_text(text):
        with zstd.open(p, 'wt', encoding='utf-8') as f:
            f.write(json.dumps({'id': 10, 'text': text})+'\n')
    write_text('prefix old suffix')
    assert v.verify_verbatim([base()], [p])['verbatim_bad'] == 0
    write_text('not the source')
    with pytest.raises(AssertionError, match='not verbatim'):
        v.verify_verbatim([base()], [p])
    with pytest.raises(AssertionError, match='missing export'):
        v.verify_verbatim([base()], [])


def test_reorder_real_zip_preserves_prefix_and_docs(tmp_path):
    from reorder_zip import write_reordered_zip
    a, _ = v.select_row(base(), [(candidate(20, 'new'), 1)])
    sub = v.append([base()], {'1': a})
    p = tmp_path/'test.zip'
    info = write_reordered_zip(sub, p)
    assert info['zip_bytes'] > 0
    assert v.check_relations(v.load_zip(p), [base()], {'1': a})['append_order_exact'] == 1


def test_1200_rows_real_validate_check_zip_and_clean(tmp_path):
    import subprocess
    import zipfile
    from reorder_zip import write_reordered_zip
    from vi23 import check_zip
    from vi29 import assert_clean
    refs, additions = [], {}
    for q in range(1, 1201):
        r = {'id': q, 'relevant_docs': [10, 20], 'relevant_chunks': [chunk(10, 'nguyên văn')]}
        refs.append(r)
        additions[str(q)], _ = v.select_row(r, [(candidate(20, 'bổ sung', q), .5)])
    sub = v.append(refs, additions)
    p = tmp_path/'submission.zip'
    jp = tmp_path/'submission.json'
    write_reordered_zip(sub, p)
    with zipfile.ZipFile(p) as z:
        jp.write_bytes(z.read('submission.json'))
    subprocess.run([sys.executable, str(Path(v.__file__).with_name('evaluate.py')),
                    'validate', str(jp)], check=True)
    assert check_zip(str(jp), str(p), str(tmp_path/'check.json'), 'test_vi48') == 0
    assert assert_clean(sub, {99}, {10, 20}) == {
        'cau': 1200, 'bad_doc': 0, 'bad_chunk': 0, 'chunk_ngoai_doclist': 0,
        'doc_ngoai_kho': 0, 'chunk_ngoai_kho': 0, 'cap_doc_text_trung': 0}
    assert v.check_relations(v.load_zip(p), refs, additions)['vi45_literal_prefix'] == 1200


def test_budget_never_removes_baseline_even_if_it_has_low_or_unknown_logit():
    r = base()
    a, _ = v.select_row(r, [(candidate(10, 'new'), .001)])
    attempts = []
    def measure(sub):
        attempts.append(copy.deepcopy(sub))
        return {'zip_bytes': 101 if len(attempts) == 1 else 100}
    sub, removed, _ = v.fit_budget([r], {'1': a}, measure, 100)
    assert removed == {a[0]['cid']}
    assert sub == [r]


def test_empty_additions_is_exact_identity_and_string_query_key():
    r = base()
    a, excluded = v.select_row(r, [])
    assert a == excluded == []
    assert v.append([r], {'1': a}) == [r]
    assert v.check_relations([r], [r], {'1': []})['other_fields_equal_vi47'] == 1
