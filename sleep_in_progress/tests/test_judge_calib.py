import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from judge_calib import iter_json_array, metrics, pair_stats, loo_predictions, correlation


def test_stream_utf8_and_bad_input():
    values = [{'id': 1, 'text': 'bệnh ' * 50}, {'id': 2, 'text': 'x'}]
    assert list(iter_json_array(io.BytesIO(json.dumps(values, ensure_ascii=False).encode()), 7)) == values
    with pytest.raises(ValueError):
        list(iter_json_array(io.BytesIO(b'[{"id":1}'), 3))


def test_missing_bounds_not_fabricated():
    rows = [{'id': 1, 'relevant_docs': [1, 2]}, {'id': 2, 'relevant_docs': [3]}]
    result = metrics(rows, {'1': {'1': 1}, '2': {'3': -1}}, 0)
    assert result['missing_rate'] == pytest.approx(1/3)
    assert result['p_lower'] == .25 and result['p_upper'] == .5
    assert result['p_conditional'] == .5 and result['correct_per_query_adjusted'] == 1


def test_all_missing_query_abstains():
    result = metrics([{'id': 1, 'relevant_docs': [10]}], {}, 0)
    assert result['missing_rate'] == 1
    assert result['p_conditional'] is None and result['r_proxy'] is None
    assert result['p_lower'] == 0 and result['p_upper'] == 1


def test_near_pairs_exclude_duplicates_and_count_ties():
    rows = [{'fingerprint': f, 'Final': v, 'p': p, 'Docs_P': y} for f, v, p, y in
            [('a', .30, 1, 1), ('b', .301, 2, 2), ('c', .302, 2, 3), ('a', .303, 1, 4)]]
    p = pair_stats(rows, 'p', 'Docs_P')
    assert (p['correct'], p['wrong'], p['proxy_ties']) == (2, 2, 1)


def test_loo_does_not_train_on_matching_doclist():
    rows = [{'fingerprint': str(i), 'p': i, 'r': i*i, 'Final': 2*i + 1} for i in range(5)]
    rows.append({**rows[0], 'Final': 999})
    loo_predictions(rows)
    assert rows[0]['loo_final'] == pytest.approx(1)


def test_correlation_is_undefined_for_constant_or_small_sample():
    assert correlation([1, 1, 1], [1, 2, 3])['pearson'] is None
    assert correlation([1, 2], [2, 1])['spearman'] is None
    assert correlation([1, 2, 3], [3, 2, 1])['spearman'] == pytest.approx(-1)
