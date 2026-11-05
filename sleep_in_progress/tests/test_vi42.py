import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import vi42 as v


def text():
    return ' '.join(f'từ{i}' for i in range(2200))


def test_geometry_verbatim_token_boundary_and_character_midpoint():
    t = text()
    chunks, bs = v.bridge_geometry(t)
    assert len(chunks) == 3 and len(bs) == 2
    spans = [m.span() for m in v.WORD.finditer(t)]
    for i, b in enumerate(bs):
        assert b['start'] in {s for s, e in spans}
        assert b['end'] in {s for s, e in spans}
        assert b['text'] == t[b['start']:b['end']].strip()
        assert b['boundary'] == chunks[i+1].start
        assert b['band'] == (spans[(i+1)*900-300][0], spans[min((i+1)*900+300, 2200)-1][1])


@pytest.mark.parametrize('wins,expected', [([], None), ([(0, 2, 9)], None),
    ([(2, 5, -0.01)], None), ([(2, 5, 0)], 0), ([(2, 5, 1), (4, 8, 2)], 2),
    ([(5, 8, 10)], None)])
def test_gate_positive_intersection_inclusive_zero(wins, expected):
    assert v.gate_score(wins, (2, 5)) == expected


def test_selection_neighbor_top50_order_dedup_and_prefix():
    t = text()
    chunks, bs = v.bridge_geometry(t)
    it = {'id': 1, 'relevant_docs': [7, 8]+list(range(100, 149)), 'relevant_chunks': [
        {'doc_id': 7, 'chunk_text': chunks[0].text},
        {'doc_id': 8, 'chunk_text': chunks[1].text},
        {'doc_id': 148, 'chunk_text': chunks[0].text}]}
    got = v.select_bridges(it, {7: t, 8: t}, {'7': [[0, len(t), 1]], '8': [[0, len(t), 2]]})
    assert [(b['doc_id'], b['boundary_index']) for b in got] == [(8, 0), (8, 1), (7, 0)]
    out = v.append_bridges([it], {'1': got})[0]
    assert out['relevant_docs'] == it['relevant_docs']
    assert out['relevant_chunks'][:3] == it['relevant_chunks']
    assert v.append_bridges([it], {'1': got}, enabled=False) == [it]
    assert len({(c['doc_id'], c['chunk_text']) for c in out['relevant_chunks']}) == 6


def test_no_duplicate_bridge_text_even_repeated_geometry():
    t = text()
    chunks, bs = v.bridge_geometry(t)
    it = {'id': 1, 'relevant_docs': [7], 'relevant_chunks': [{'doc_id': 7, 'chunk_text': chunks[0].text}]}
    geometry = {'7': (chunks, [bs[0], bs[0]])}
    assert len(v.select_bridges(it, {7: t}, {'7': [[0, len(t), 0]]}, geometry)) == 1


def test_budget_all_and_score_group_boundary():
    assert v.budget_threshold([0, 1, 1, 2], lambda t: 9, 10)[0] == 0
    calls = []
    def measure(t):
        calls.append(t)
        return 1 if t is None else 12-2*t
    assert v.budget_threshold([0, 1, 1, 2, 3], measure, 10)[0] == 1
    assert 0 in calls and 1 in calls
    assert v.budget_threshold([0, 1], lambda t: 1 if t is None else 20, 10)[0] is None
    with pytest.raises(AssertionError):
        v.budget_threshold([0], lambda t: 20, 10)


def test_midpoint_tie_lower_and_grid_tail():
    assert v.midpoint([(0, 1), (4, 5)], 0, 4) == 0
    assert v.window_grid('a b') == [(0, 3)]
    assert len(v.window_grid(text())) == 9


def test_midpoint_uses_characters_not_half_word_count():
    # Unequal word widths make these two definitions observably different.
    tokens = [(0, 1), (2, 3), (4, 5), (6, 107)]
    assert v.midpoint(tokens, 0, 107) == 6


def test_short_tail_midpoint_and_original_whitespace():
    t = '\n  ' + '\t '.join(f'w{i}' for i in range(905)) + '\n'
    chunks, bs = v.bridge_geometry(t)
    # chunking merges a <20-unit tail if total fits 1.5 U900, so no phantom bridge.
    assert len(chunks) == 1 and bs == []
    t = '\n  ' + '\t '.join(f'w{i}' for i in range(950)) + '\n'
    chunks, bs = v.bridge_geometry(t)
    assert len(chunks) == 2 and len(bs) == 1
    assert '\t ' in bs[0]['text'] and bs[0]['text'] in t


def test_1200_disabled_json_same_by_id_and_budget_removes_low_scores():
    ref = [{'id': i, 'relevant_docs': [7], 'relevant_chunks': [
        {'doc_id': 7, 'chunk_text': 'original'}]} for i in range(1, 1201)]
    bridges = {str(i): [{'doc_id': 7, 'chunk_text': 'bridge', 'score': i % 3}]
               for i in range(1, 1201)}
    disabled = v.append_bridges(ref, bridges, enabled=False)
    assert v.compare_by_id(disabled[::-1], ref) == {'count': 1200, 'docs': 1200, 'chunks': 1200}
    assert v.json.dumps(disabled) == v.json.dumps(ref)
    kept = v.append_bridges(ref, bridges, threshold=2)
    assert sum(len(x['relevant_chunks'])-1 for x in kept) == 400
    assert all(x['relevant_chunks'][0] == old['relevant_chunks'][0] for x, old in zip(kept, ref))


def test_missing_selected_grid_or_cache_fails_loudly():
    t = text()
    chunks, _ = v.bridge_geometry(t)
    it = {'id': 1, 'relevant_docs': [7], 'relevant_chunks': [{'doc_id': 7, 'chunk_text': 'bad'}]}
    with pytest.raises(AssertionError, match='grid mismatch'):
        v.select_bridges(it, {7: t}, {})
    it['relevant_chunks'][0]['chunk_text'] = chunks[0].text
    with pytest.raises(AssertionError, match='missing cached'):
        v.select_bridges(it, {7: t}, {})


def test_real_reordered_zip_budget_and_offline(tmp_path):
    it = {'id': 1, 'relevant_docs': [7], 'relevant_chunks': [{'doc_id': 7, 'chunk_text': 'gốc'}]}
    bs = {'1': [{'doc_id': 7, 'chunk_text': 'cầu nối nguyên văn', 'score': 0, 'rank': 3}]}
    path = tmp_path/'test.zip'
    def measure(thr):
        return v.write_reordered_zip(v.append_bridges([it], bs, thr, enabled=thr is not None), path)['zip_bytes']
    threshold, _ = v.budget_threshold([0], measure, limit=10000)
    assert threshold == 0
    with v.zipfile.ZipFile(path) as z:
        assert z.namelist() == ['submission.json']
        assert z.infolist()[0].compress_type == v.zipfile.ZIP_LZMA
        assert z.testzip() is None
        assert v.json.loads(z.read('submission.json')) == v.append_bridges([it], bs)
    stats = v.offline([it], bs)
    assert stats['bridges_per_query'] == stats['docs_with_bridges_per_query'] == 1
    assert stats['chunks_per_query'] == 2 and stats['length_syllables']['p50'] == 4
    assert stats['doc_rank_buckets']['1-10'] == 1


def test_window_cache_fresh_wins_else_exact_provenance(monkeypatch):
    objects = {
        'docwin_max_provenance.json': {'1': {'7': 'old', '8': 'old', '9': 'later'}},
        'docwin_new.json.zst': {'1': {'7': [[0, 3, 7]]}},
        'old.zst': {'1': {'7': [[0, 3, -7]], '8': [[0, 3, 8]], '9': [[0, 3, -9]]}},
        'later.zst': {'1': {'8': [[0, 3, -8]], '9': [[0, 3, 9]]}}}
    monkeypatch.setattr(v.dw, 'read', lambda path: objects[path.name])
    monkeypatch.setattr(v.dw, 'layer_specs', lambda: [('old', 'old.zst', True), ('later', 'later.zst', False)])
    got, used = v.load_windows({'1': {'7', '8', '9'}})
    assert got == {'1': {'7': [[0, 3, 7]], '8': [[0, 3, 8]], '9': [[0, 3, 9]]}}
    assert used == {'old': 1, 'later': 1}
    with pytest.raises(AssertionError):
        v.load_windows({'1': {'missing'}})
