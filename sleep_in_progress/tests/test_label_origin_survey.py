import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('label_origin_survey', Path(__file__).parents[1] / 'scripts/label_origin_survey.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_score_order_and_censoring():
    assert m.ranked([[3, 0, 1], [2, 0, 3], [3, 0, 2]]) == {2: 1, 3: 2}
    s = m.summarize([1, 30, None], 300)
    assert s['observed_median'] == 15.5
    assert s['topk_fraction']['30'] == 2 / 3
    assert s['clipped_mean'] == 332 / 3


def test_precision_and_cluster_bootstrap():
    assert abs(m.marginal_precision(10, .4, 30, .3) - .25) < 1e-10
    assert m.marginal_precision(30, .3, 30, .4) is None
    assert m.paired_bootstrap([(1, 2), (2, 4)], 100) == [.5, .5]
