import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import vi38 as v
import llm_judge_run as judge


def fixture():
    item = {'id': 1, 'relevant_docs': list(range(1, 11)),
            'relevant_chunks': [{'doc_id': 1, 'chunk_text': 'đa lõi ☃ nguyên văn', 'chunk_order': 0}]}
    rows, judgments = [], {}
    for role, ids in [('A', range(1, 11)), ('B', range(11, 21))]:
        for d in ids:
            row = dict(pair_id=f'1:{d}', qid='1', doc_id=d, role=role, rank=d,
                       text_sha256='text', prompt_sha256='prompt')
            rows.append(row)
            judgments[row['pair_id']] = dict(status='ok', label=0 if role == 'A' else 2,
                  score=-1 if role == 'A' else 1, confirm_full=True, confirm_text_sha256='text')
    return item, rows, judgments


def test_protect_chunks_cap5_position_and_count():
    item, rows, judgments = fixture()
    original = copy.deepcopy(item)
    docs, changes = v.swap_docs(item, rows, judgments)
    assert docs == [1, 11, 12, 13, 14, 15, 7, 8, 9, 10]
    assert len(changes) == 5 and len(set(docs)) == len(item['relevant_docs'])
    assert item == original
    assert {c['incoming'] for c in changes} <= {r['doc_id'] for r in rows if r['role'] == 'B'}
    assert all(docs[c['position']] == c['incoming'] for c in changes)


@pytest.mark.parametrize('bad', [{}, {'status': 'abstain'}, {'status': 'ok', 'score': float('nan'), 'label': 0},
    {'status': 'ok', 'score': float('inf'), 'label': 0},
    {'status': 'ok', 'score': 0, 'label': 0},
    {'status': 'ok', 'score': 0, 'label': 0, 'confirm_full': True, 'confirm_text_sha256': 'wrong'}])
def test_missing_nonfinite_unconfirmed_abstain(bad):
    item, rows, judgments = fixture()
    for r in rows:
        if r['role'] == 'A':
            judgments[r['pair_id']] = bad
    assert v.swap_docs(item, rows, judgments)[0] == item['relevant_docs']


def test_no_b_never_drop_a():
    item, rows, judgments = fixture()
    assert v.swap_docs(item, [r for r in rows if r['role'] == 'A'], judgments)[0] == item['relevant_docs']


def test_cap0_all_label1():
    item, rows, judgments = fixture()
    assert v.swap_docs(item, rows, judgments, 0)[0] == item['relevant_docs']
    for j in judgments.values():
        j['label'] = 1
    assert v.swap_docs(item, rows, judgments)[0] == item['relevant_docs']


def test_tie_rank_then_id():
    item, rows, judgments = fixture()
    for r in rows:
        r['rank'] = 100
    docs, changes = v.swap_docs(item, list(reversed(rows)), judgments, 2)
    assert [(c['out'], c['incoming']) for c in changes] == [(2, 11), (3, 12)]


def test_byte_preservation():
    item, _, _ = fixture()
    raw = ('[\n '+json.dumps(item, ensure_ascii=False, indent=2)+'\n]').encode('utf-8')
    assert v.replace_docs_bytes(raw, [item], {'1': item['relevant_docs']}) == raw
    changed = v.replace_docs_bytes(raw, [item], {'1': [1, 11, *range(3, 11)]})
    a = raw[raw.index(b'"relevant_chunks"'):]
    b = changed[changed.index(b'"relevant_chunks"'):]
    assert a == b and json.loads(changed)[0]['relevant_chunks'] == item['relevant_chunks']


def test_select_no_empty_a_queries_and_clean_pool():
    fh = {'1': [[d, d, 1/d] for d in range(1, 171)], '2': [[d, d, 1/d] for d in range(1, 171)]}
    pool = {q: [[d, d, 0 if d <= 120 else -3] for d in range(1, 171)] for q in fh}
    sub = [{'id': 1, 'relevant_docs': list(range(1, 121)), 'relevant_chunks': [{'doc_id': 100, 'chunk_text': 'x'}]},
           {'id': 2, 'relevant_docs': list(range(1, 121)),
            'relevant_chunks': [{'doc_id': d, 'chunk_text': 'x'} for d in range(100, 121)]}]
    rows, controls, nq = v.select_pairs(fh, pool, sub)
    assert nq == {'1': 120, '2': 120}
    assert sum(r['role'] == 'A' for r in rows) == 20
    assert sum(r['role'] == 'B' for r in rows) == 30
    assert {r['qid'] for r in rows} == {'1'}
    assert all(r['doc_id'] != 100 for r in rows)
    assert len(controls['top30']) == 60


@pytest.mark.parametrize('raw,label,score', [({15: -3, 16: -2, 17: -1}, 2, 2),
     ({15: -1, 16: -2, 17: -3}, 0, -2), ({15: -1, 16: -1, 17: -1}, 0, 0),
     ({'15': -3, '16': -.1, '17': -2}, 1, 1)])
def test_parse_raw(raw, label, score):
    got = judge.parse_logprobs(raw)
    assert got['label'] == label and got['score'] == score


@pytest.mark.parametrize('raw', [{}, {15: -1, 16: -2}, {15: None, 16: -1, 17: -2},
                                      {15: float('-inf'), 16: -1, 17: -2}])
def test_parse_abstain(raw):
    assert judge.parse_logprobs(raw)['status'] == 'abstain'


def test_parse_vllm_objects():
    raw = {15: SimpleNamespace(logprob=-2), 16: SimpleNamespace(logprob=-1), 17: SimpleNamespace(logprob=-3)}
    assert judge.parse_logprobs(raw)['label'] == 1


def test_equivalence_gate_thresholds():
    rows = [dict(request_id=str(i), prompt_sha256=str(i), revision=judge.REVISION,
                 label=2, score=1.) for i in range(146)]
    assert judge.compare_equivalence(rows, rows)['passed']
    changed = copy.deepcopy(rows)
    changed[0]['label'] = 1
    changed[0]['score'] += .04
    assert judge.compare_equivalence(rows, changed)['passed']
    changed[1]['label'] = 0
    assert not judge.compare_equivalence(rows, changed)['passed']
    changed = copy.deepcopy(rows)
    changed[0]['score'] += .051
    assert not judge.compare_equivalence(rows, changed)['passed']


def test_gpu_gather_mock_preserves_raw_values_and_three_ids(monkeypatch):
    import torch
    class Sampler:
        pass
    monkeypatch.setitem(sys.modules, 'vllm.v1.sample.sampler', SimpleNamespace(Sampler=Sampler))
    monkeypatch.setitem(sys.modules, 'vllm.v1.outputs', SimpleNamespace(LogprobsTensors=lambda *args: args))
    monkeypatch.setitem(sys.modules, 'vllm.v1.sample.ops.logprobs', SimpleNamespace(
        batched_count_greater_than=lambda lp, value: (lp > value).sum(-1)+1))
    judge.TargetedLogprobWorker().install_label_logprobs()
    raw = torch.log_softmax(torch.arange(20, dtype=torch.float32).unsqueeze(0), -1)
    indices, values, ranks = Sampler.gather_logprobs(raw, 3, torch.tensor([17]))
    assert indices.tolist() == [[17, 15, 16, 17]]
    assert torch.equal(values, raw[:, [17, 15, 16, 17]])
    assert float(values[0, 3]-values[0, 1]) == pytest.approx(2., abs=1e-6)
    assert ranks.tolist() == [3]


def test_merge_missing_duplicate_nonfinite():
    ex = dict(pair_id='1:2', prompt_sha256='p', text_sha256='t')
    row = dict(ex, revision=judge.REVISION, logp0=-3, logp1=-2, logp2=-1)
    assert v.checked_merge([ex], [row])['1:2']['label'] == 2
    for bad in ([], [row, row], [dict(row, logp0=None)], [dict(row, prompt_sha256='wrong')]):
        with pytest.raises(AssertionError):
            v.checked_merge([ex], bad)


def test_token_window_union():
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({'x': 0, '[UNK]': 1}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    text = ' '.join(['x']*1000)
    head, win, offsets, full = judge.snippets(tok, text, [200, 1800])
    assert len(tok.encode(head).ids) == 256 and len(tok.encode(win).ids) <= 512
    assert offsets[1][0] >= offsets[0][1] and full == 1000


def test_latest_owner_not_cross_layer_max(monkeypatch):
    values = {'docwin_max_provenance.json': {'1': {'1': 'vi36', '2': 'top100'}},
              'docwin_new.json.zst': {'1': {'2': [[0, 3, 1]]}},
              'old.json': {'1': {'1': [[0, 9, 99]], '2': [[0, 9, 99]]}},
              'new.json': {'1': {'1': [[3, 5, 1], [5, 8, 2]]}}}
    monkeypatch.setattr(v.dw, 'read', lambda p: values[Path(p).name])
    monkeypatch.setattr(v.dw, 'layer_specs', lambda: [('top100', 'old.json', True), ('vi36', 'new.json', False)])
    out = v.recover_windows([dict(qid='1', doc_id=1), dict(qid='1', doc_id=2)])
    assert out['1', '1'] == ('vi36', [5, 8, 2])
    assert out['1', '2'] == ('vi37_new', [0, 3, 1])


def test_parts_token_sharding(tmp_path):
    files = v.write_parts(tmp_path, 'test', [{'n_tokens': 60, 'i': i} for i in range(5)], 130)
    assert [len(v.dw.read(tmp_path/f)) for f in files] == [2, 2, 1]


def test_modal_source_compiles_no_launch():
    p = Path(__file__).resolve().parents[1]/'deploy/modal_llmj.py'
    compile(p.read_text(encoding='utf-8'), str(p), 'exec')


def test_rule_b_fulltext_score_and_label1_eligibility():
    item, rows, js = fixture()
    for r in rows:
        j = js[r['pair_id']]
        if r['role'] == 'A':
            j['label'] = 1
            j['confirmation'] = dict(label=1, score=-r['doc_id'])
    docs, changes = v.swap_docs(item, rows, js, 2, rule='b')
    assert [(c['out'], c['incoming']) for c in changes] == [(10, 11), (9, 12)]
    assert docs[8:10] == [12, 11]
    assert v.swap_docs(item, rows, js, 2, rule='a')[1] == []


def test_rule_b_confirm_label2_abstain_nonfinite_hash():
    item, rows, js = fixture()
    for r in rows:
        if r['role'] == 'A':
            js[r['pair_id']]['confirmation'] = dict(label=0, score=-1.)
    for pid, update in [('1:2', {'label': 2, 'score': -100}),
                        ('1:3', {'label': 1, 'score': float('nan')}),
                        ('1:4', {}), ('1:5', {'label': None, 'score': -100})]:
        js[pid]['confirmation'] = update
    js['1:6']['confirm_text_sha256'] = 'wrong'
    assert [c['out'] for c in v.swap_docs(item, rows, js, rule='b')[1]] == [7, 8, 9, 10]


def test_rule_b_confirm_every_eligible_no_b_no_cap():
    _, rows, js = fixture()
    rows = [r for r in rows if r['role'] == 'A']
    for j in js.values():
        j['label'] = 1
    js['1:1']['score'] = float('inf')
    assert len(v.confirmation_candidates(rows, js, 'b')) == 9
    assert not v.confirmation_candidates(rows, js, 'a')


def test_rule_b_fulltext_truncates_original_order_prompt_cap():
    class CharTok:
        def encode(self, text, **kwargs):
            return SimpleNamespace(ids=list(text))
    text = 'áβ☃原文'*2000
    tok = CharTok()
    prompt, n, cut = v.fulltext_prompt(tok, 'q', text)
    assert cut and n <= 4095 and prompt.startswith(judge.chat('q', '', '')[:20])
    prefix = prompt.split('Đầu bài: ', 1)[1].split('\nĐoạn bằng chứng', 1)[0]
    assert text.startswith(prefix) and prefix
    assert v.fulltext_prompt(tok, 'q', 'nguyên văn') == (
        judge.chat('q', 'nguyên văn', ''), len(judge.chat('q', 'nguyên văn', '')), False)


def test_gate2_sample_blind_new_balanced_stratified():
    import vi38_gate2 as gate
    ex = {str(q): [dict(out=q*10+i, incoming=10000+q*10+i) for i in range(5)] for q in range(100)}
    texts = {c[k]: 'toàn văn' for cs in ex.values() for c in cs for k in ('out', 'incoming')}
    class Tok:
        def encode(self, s, **kwargs):
            return SimpleNamespace(ids=list(s), offsets=[(i, i+1) for i in range(len(s))])
    blind, key = gate.create_blind(ex, {str(q): 'query' for q in range(100)}, texts, Tok(), {'1:10'})
    from collections import Counter
    assert len(blind) == 300 and Counter(k['role'] for k in key.values()) == {'out': 150, 'in': 150}
    assert all(set(r) == {'code', 'qid', 'query', 'doc_id', 'text', 'truncated'} for r in blind)
    assert max(Counter((r['role'], r['qid']) for r in key.values()).values()) <= 2
    assert all(r['pair_id'] != '1:10' for r in key.values())


def test_rule_b_cmerge_uses_fulltext_labels_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(v, 'OUT', tmp_path)
    ex = dict(pair_id='1:2', prompt_sha256='p', text_sha256='t')
    row = dict(ex, revision=judge.REVISION, logp0=-3, logp1=-1, logp2=-2)
    v.dw.write(tmp_path/'export/confirm.part000.json', [ex])
    fp = v.fingerprint(tmp_path/'export/confirm.part000.json')
    v.dw.write(tmp_path/'export/manifest.json', dict(modes={'confirm': ['confirm.part000.json']},
                                                       files={'confirm.part000.json': fp}))
    v.dw.write(tmp_path/'confirm_out/confirm.part000.json.out.json',
               dict(input_sha256=fp['sha256'], rows=[row]))
    v.dw.write(tmp_path/'merged.json', {'1:2': dict(qid='1', status='ok', label=0, score=-100)})
    v.stage_cmerge(SimpleNamespace(rule='b'))
    got = v.dw.read(tmp_path/'merged.json')['1:2']
    assert got['confirm_full'] and got['score'] == -100
    assert got['confirmation']['label'] == 1 and got['confirmation']['score'] == 1
    assert v.dw.read(tmp_path/'cmerge_report.json')['eligible_after'] == 1


def test_rule_b_confirm_stage_all_a_including_long_truncated(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({'x': 0, '[UNK]': 1}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path/'tokenizer.json'))
    monkeypatch.setattr(v, 'OUT', tmp_path)
    monkeypatch.setattr(v, 'OLD', tmp_path/'old')
    texts = {d: ' '.join(['x']*(10000 if d == 1 else 10)) for d in range(1, 9)}
    rows = [dict(pair_id=f'1:{d}', qid='1', doc_id=d, role='A', rank=d+99,
                 status='ready', text_sha256=judge.sha(texts[d]), prompt_sha256='excerpt', revision=judge.REVISION)
            for d in texts]
    v.dw.write(tmp_path/'export/pairs.json', rows)
    v.dw.write(tmp_path/'merged.json', {r['pair_id']: dict(status='ok', label=1, score=0.) for r in rows})
    v.dw.write(tmp_path/'export/manifest.json', dict(modes={'confirm': []}, files={}))
    (tmp_path/'old/modal_input').mkdir(parents=True)
    (tmp_path/'old/modal_input/queries.jsonl').write_text(json.dumps(dict(id=1, query='x')), encoding='utf-8')
    import rerank_run
    monkeypatch.setattr(rerank_run, 'load_docs', lambda *args, **kwargs: texts)
    v.stage_confirm(SimpleNamespace(rule='b'))
    rep = v.dw.read(tmp_path/'confirm_report.json')
    assert rep['eligible_before'] == rep['eligible'] == 8 and rep['truncated'] == 1
    assert rep['abstain'] == []
    manifest = v.dw.read(tmp_path/'export/manifest.json')
    output = [r for n in manifest['modes']['confirm'] for r in v.dw.read(tmp_path/'export'/n)]
    assert len(output) == 8 and all(r['role'] == 'A' and r['n_tokens'] < 4096 for r in output)
    assert output[0]['truncated'] and output[0]['text_sha256'] == judge.sha(texts[1])


@pytest.mark.parametrize('bad', [{}, {'status': 'abstain', 'label': 0, 'score': 0},
                                {'status': 'ok', 'label': 0, 'score': float('nan')}])
def test_rule_b_missing_first_pass_never_reintroduced_by_confirmation(bad):
    item, rows, js = fixture()
    for r in rows:
        if r['role'] == 'A':
            js[r['pair_id']] = dict(bad, confirm_full=True, confirm_text_sha256='text',
                                     confirmation=dict(label=0, score=-100))
    assert not v.swap_docs(item, rows, js, rule='b')[1]


def test_mock_vllm_run_resume_and_budget(tmp_path, monkeypatch):
    src, dst = tmp_path/'in', tmp_path/'out'
    src.mkdir()
    prompt = judge.chat('a', 'b', 'c')
    row = dict(pair_id='1:2', prompt=prompt, prompt_sha256=judge.sha(prompt),
                text_sha256='t', n_tokens=1, revision=judge.REVISION)
    v.dw.write(src/'p.json', [row])
    v.dw.write(src/'manifest.json', {'modes': {'bench': ['p.json']},
         'files': {'p.json': v.fingerprint(src/'p.json')}})
    class Tok:
        def encode(self, text, add_special_tokens=False):
            if text in ('0', '1', '2'):
                return [15+int(text)]
            if text and text[-1:] in ('0', '1', '2'):
                return [8]+[15+int(text[-1])]
            return [8]
        def apply_chat_template(self, *args, **kwargs):
            return prompt
        def __call__(self, texts, **kwargs):
            return {'input_ids': [[8] for _ in texts]}
    class FakeLLM:
        def __init__(self, **kw):
            assert kw['dtype'] == 'bfloat16' and kw['logprobs_mode'] == 'raw_logprobs'
            assert kw['max_logprobs'] == 3 and kw['enable_prefix_caching']
            assert kw['max_num_seqs'] == 256
        def collective_rpc(self, method):
            assert method == 'install_label_logprobs'
            return [judge.LOGPROB_METHOD]
        def get_tokenizer(self):
            return Tok()
        def generate(self, prompts, params, **kw):
            assert params['logprobs'] == 3 and params['allowed_token_ids'] == [15, 16, 17]
            assert params['temperature'] == 0 and params['max_tokens'] == 1
            assert all(p == {'prompt_token_ids': [8]} for p in prompts)
            return [SimpleNamespace(prompt_token_ids=[8], outputs=[SimpleNamespace(
                    logprobs=[{15: -.1, 16: -1., 17: -3.}])]) for p in prompts]
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(__version__=judge.VLLM,
                        LLM=FakeLLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(judge.subprocess, 'check_output', lambda *a, **kw: '40000\n')
    rep = judge.run(src, dst, 'bench', 1.)
    assert rep['new_pairs'] == 1 and rep['peak_vram_gib'] > 39
    assert judge.run(src, dst, 'bench', 1.)['new_pairs'] == 0
    (dst/'p.json.out.json').unlink()
    with pytest.raises(RuntimeError, match='budget'):
        judge.run(src, dst, 'bench', .01)


def test_bench_four_regions_length_query_strata():
    rows, controls, lengths, queries = [], {'top30': [], 'tail': []}, {}, {}
    for role in ('A', 'B', 'top30', 'tail'):
        for i in range(512):
            q, d = str(i % 128), len(lengths)+1
            queries[q] = 'q'*(500 if int(q) % 2 else 20)
            lengths[str(d)] = {'chars': i*10+1}
            row = dict(qid=q, doc_id=d, pair_id=f'{q}:{d}', role=role)
            (rows if role in ('A', 'B') else controls[role]).append(row)
    sample = v.stratified_bench(rows, controls, lengths, queries)
    assert len(sample) == 1024 and len({r['qid'] for r in sample}) >= 60
    assert all(sum(r['role'] == role for r in sample) == 256 for role in ('A', 'B', 'top30', 'tail'))
    assert sum(len(queries[r['qid']]) >= 240 for r in sample) == 512


def test_confirm_gates_only_removal_and_does_not_backfill():
    item, rows, js = fixture()
    for d in range(2, 11):
        js[f'1:{d}']['confirm_full'] = d <= 3
    docs, changes = v.swap_docs(item, rows, js)
    assert len(changes) == 2 and docs == [1, 11, 12, 4, 5, 6, 7, 8, 9, 10]


def test_empty_snippets():
    from tokenizers import Tokenizer, models
    tok = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    assert judge.snippets(tok, '', None) == ('', '', [[0, 0], [0, 0]], 0)


def test_empty_candidate_cannot_be_promoted_by_spurious_result():
    item, rows, js = fixture()
    for r in rows:
        if r['role'] == 'B':
            r['status'] = 'abstain'
    assert v.swap_docs(item, rows, js)[0] == item['relevant_docs']


def test_window_offsets_out_of_range_fail():
    from tokenizers import Tokenizer, models
    tok = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    with pytest.raises(AssertionError):
        judge.snippets(tok, 'text', [0, 100])


def test_unicode_boundary_reencode_cap_guard():
    class BoundaryTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Simulate a split byte token sharing the final complete character's offset.
            n = len(text)+(1 if len(text) in (256, 512) else 0)
            return SimpleNamespace(ids=list(range(n)), offsets=[(i, min(i+1, len(text))) for i in range(n)])
    tok = BoundaryTokenizer()
    h, w, offsets, _ = judge.snippets(tok, 'x'*2000, [600, 1900])
    assert len(tok.encode(h).ids) <= 256 and len(tok.encode(w).ids) <= 512


def test_bench_blind_mapping_only_code_label_score(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1]/'deploy/vi38_bench_report.py'
    spec = importlib.util.spec_from_file_location('bench_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path/'bench_out'
    out.mkdir()
    (tmp_path/'private').mkdir()
    key = {f'secret_code_{i}': {'pair_id': str(i), 'role': 'hidden'} for i in range(1024)}
    (tmp_path/'private/blind_key.json').write_text(json.dumps(key), encoding='utf-8')
    rows = [dict(pair_id=str(i), label=i % 3, score=i/10) for i in range(1024)]
    (out/'bench.part000.json.out.json').write_text(json.dumps({'rows': rows}), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['report', 'map-blind', '--dir', str(out)])
    module.main()
    labels = json.loads((tmp_path/'judge/qwen_labels_by_code.json').read_text(encoding='utf-8'))
    assert len(labels) == 1024 and labels['secret_code_1'] == {'label': 1, 'score': .1}
    assert all(set(row) == {'label', 'score'} for row in labels.values())
    assert 'secret_code_' not in capsys.readouterr().out


def test_second_bench_mapping_includes_three_logprobs(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1]/'deploy/vi38_bench_report.py'
    spec = importlib.util.spec_from_file_location('bench_report2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path/'bench_out2'
    module.write(tmp_path/'private/blind_key.json',
                 {f'private_{i}': {'pair_id': str(i)} for i in range(1024)})
    rows = [dict(pair_id=str(i), **judge.parse_logprobs({15: -3, 16: -2, 17: -1})) for i in range(1024)]
    module.write(out/'bench.part000.json.out.json', {'rows': rows})
    monkeypatch.setattr(sys, 'argv', ['report', 'map-blind', '--dir', str(out), '--second'])
    module.main()
    labels = module.read(tmp_path/'judge/qwen_labels_by_code.json')
    assert len(labels) == 1024
    assert all(set(row) == {'label', 'score', 'logp0', 'logp1', 'logp2'} for row in labels.values())
    assert 'private_' not in capsys.readouterr().out


def test_bench_report_coverage_stability_forecast(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1]/'deploy/vi38_bench_report.py'
    spec = importlib.util.spec_from_file_location('bench_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path/'data/sub/vi38'
    dest = base/'bench_out'
    def put(p, ob):
        module.write(p, ob)
    originals = []
    for i in range(1024):
        role = ('top30', 'A', 'B', 'tail')[i//256]
        raw = {0: -2., 1: -3., 2: -1.} if role in ('top30', 'B') else {0: -1., 1: -3., 2: -2.}
        originals.append(dict(pair_id=f'1:{i}', qid='1', doc_id=i, role=role, request_id=f'base:{i}',
            prompt_sha256=f'p{i}', text_sha256=f't{i}', revision=judge.REVISION, n_tokens=100,
            **judge.parse_logprobs(raw, (0, 1, 2))))
    repeats = [dict(r, request_id=f'repeat:{i}', repeat_of=r['pair_id']) for i, r in enumerate(originals[:128])]
    rows = originals+repeats
    put(base/'export/bench.json', rows)
    put(base/'export/manifest.json', dict(modes={'bench': ['bench.part000.json']},
        files={'bench.part000.json': {'sha256': 'digest'}}, bulk_prompt_tokens=78662609))
    put(dest/'bench.part000.json.out.json', {'input_sha256': 'digest', 'rows': rows})
    source = {'1': [[i, None, float(i)] for i in range(1024)]}
    put(tmp_path/'data/sub/vi37/q8b_v37.json', source)
    put(tmp_path/'data/sub/vi37/cache_fhyde_v37.json', source)
    put(tmp_path/'data/sub/diag37/docwin_max_v37.json', {'1': {str(i): float(i) for i in range(1024)}})
    put(dest/'report_bench.json', dict(prefill_token_s=10000, load_wall=30, token_validation={},
        prefill_note='test', inference_wall=11.52, paid_function_wall=50, peak_vram_gib=70,
        peak_vram_method='test', usd_est=.1))
    put(dest/'launch_audit.json', dict(all_wall_h100_upper_usd=.2, client_wall_s=100))
    put(dest/'ledger_test.json', dict(paid_function_wall=55, usd_est=.11))
    monkeypatch.setattr(sys, 'argv', ['report', 'report', '--root', str(tmp_path)])
    module.main()
    capsys.readouterr()
    report = module.read(dest/'bench_report.json')
    assert report['finite'] == 1152 and report['input_tokens'] == 115200
    assert report['label_agreement'] == 1 and report['max_abs_delta_score'] == 0
    assert report['simulated_swaps'] == 5
    assert report['bulk_forecasts']['2_H100']['wall_s'] < report['bulk_forecasts']['1_H100']['wall_s']
    assert report['bulk_forecasts']['2_H100']['usd'] > report['bulk_forecasts']['1_H100']['usd']
    assert report['paid_function_wall_s'] == 55 and report['usd_function_est'] == .11
    assert report['repeated_bench_swap_decisions_unchanged']
    bad = copy.deepcopy(rows)
    bad[0]['logp1'] = None
    put(dest/'bench.part000.json.out.json', {'input_sha256': 'digest', 'rows': bad})
    with pytest.raises(AssertionError):
        module.main()
    put(dest/'bench.part000.json.out.json', {'input_sha256': 'wrong', 'rows': rows})
    with pytest.raises(AssertionError):
        module.main()
    put(dest/'bench.part000.json.out.json', {'input_sha256': 'digest', 'rows': rows[:-1]})
    with pytest.raises(AssertionError):
        module.main()


def test_bench_failure_does_not_invent_metrics(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1]/'deploy/vi38_bench_report.py'
    spec = importlib.util.spec_from_file_location('bench_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dest = tmp_path/'bench_out'
    module.write(dest/'launch_audit.json', dict(all_wall_h100_upper_usd=.8, client_wall_s=660,
                                               rate_upper_usd_s=.0012564444, app_ids=[]))
    monkeypatch.setattr(sys, 'argv', ['report', 'failure', '--dir', str(dest)])
    module.main()
    capsys.readouterr()
    report = module.read(dest/'bench_report.json')
    assert report['received'] == 0 and report['coverage'] == 0
    assert report['throughput'] is None and report['bulk_forecasts'] is None
    assert not report['gates_passed'] and report['usd_bench_all_client_wall_upper'] == .8


def test_bench_client_watchdog_reserves_cleanup_under_dollar():
    path = Path(__file__).resolve().parents[1]/'deploy/vi38_bench_launch.py'
    spec = importlib.util.spec_from_file_location('bench_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # One allowed app: timeout + process teardown + stop request, with $0.09 extra.
    assert (660+20+40)*module.RATE < .91
    source = path.read_text(encoding='utf-8')
    assert "'--yes'" in source and 'launch audit already exists' in source
