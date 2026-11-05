import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fuse3
import vi29
import vi37
import vi37z


def test_zd_extra_rank_only_and_fhyde_unchanged():
    pool = {'1': [[1, 0, 3.], [2, 1, 2.], [3, 2, 1.], [4, 3, -5.]]}
    q8 = {'1': [[2, 1, 1.], [1, 0, 0.]]}
    dense = {'1': [[1, 0, 1.]]}
    hyde = {'1': [[3, 2, 1.]]}
    scores = {'1': {'2': 5., '3': 5., '4': None, '999': 8.}}
    args = (pool, q8, dense, hyde, {}, {'1': 0}, {'1': []})
    kwargs = dict(lo=1, hi=4, delta=0, dedup_mode='doc_text')
    with vi37.chain_docwin(scores):
        base = vi29.chain_build_v29(*args, **kwargs)
    with vi37z.chain_docwin_zd(scores, enabled=False):
        disabled = vi29.chain_build_v29(*args, **kwargs)
    with vi37z.chain_docwin_zd(scores):
        enabled = vi29.chain_build_v29(*args, **kwargs)
    assert disabled == base
    assert enabled[1:] == base[1:]
    expected = fuse3.fuse3(pool, q8, dense, (1., 1., 2., 0.), 60,
                           extras=[(vi37.docwin_hits(pool, scores), 1.)])
    assert enabled[0] == expected
    assert enabled[0] != base[0]
    assert [x[0] for x in vi37.docwin_hits(pool, scores)['1']] == [2, 3]


def test_empty_docwin_adds_no_fake_score():
    pool = {'1': [[1, 0, 1.], [2, 1, 0.]], '2': []}
    args = (pool, {}, {}, (1., 1., 2., 0.), 60)
    expected = fuse3.fuse3(*args)
    with vi37z.chain_docwin_zd({'1': {'1': None}}):
        assert fuse3.fuse3(*args) == expected


def test_context_restored_on_error():
    original = fuse3.fuse3
    with pytest.raises(RuntimeError):
        with vi37z.chain_docwin_zd({}):
            raise RuntimeError('test')
    assert fuse3.fuse3 is original


def test_sanity_flag_is_required(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['vi37z.py', 'sanity'])
    with pytest.raises(SystemExit) as exc:
        vi37z.main()
    assert exc.value.code == 2
