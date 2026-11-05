import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import vi39


def test_boundary_measures_successor():
    calls = []
    def measure(d):
        calls.append(d)
        return {'zip_bytes': vi39.LIMIT + d - 91, 'hist_kcap': [[0, 200, 1]]}
    delta, reason, got = vi39.find_boundary(measure)
    assert (delta, reason) == (91, 'boundary')
    assert 91 in calls and 92 in calls and len(calls) <= 12


def test_smallest_saturation():
    def measure(d):
        return {'zip_bytes': 1, 'hist_kcap': [[3, 63, 1], [5, 80, 1]]}
    delta, reason, _ = vi39.find_boundary(measure)
    assert (delta, reason) == (75, 'saturated')


def test_by_id_not_position():
    rows = [{'id': 1, 'relevant_docs': [7], 'relevant_chunks': []},
            {'id': 2, 'relevant_docs': [9], 'relevant_chunks': []}]
    assert vi39.compare_by_id(rows[::-1], rows) == {'count': 2, 'docs': 2, 'chunks': 2}
    with pytest.raises(AssertionError):
        vi39.compare_by_id(rows + rows[:1], rows)


def test_lock_does_not_remove_other_owner(tmp_path):
    lock = tmp_path / 'lock'
    lock.write_text('other')
    with pytest.raises(TimeoutError):
        with vi39.wait_lock(lock, 'mine', seconds=0):
            pass
    assert lock.read_text() == 'other'
