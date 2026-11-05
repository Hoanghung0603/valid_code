"""Bench-only CPU audit. No bulk/confirm/build; blind key used only by map-blind."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, ob):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ob, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fingerprint(p):
    return dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'report', 'verify', 'map-blind', 'failure'])
    ap.add_argument('--root', type=Path, default=Path('D:/r2ai'))
    ap.add_argument('--dir', type=Path)
    ap.add_argument('--allow-partial', action='store_true', help='Map only received labels after a stopped bench')
    ap.add_argument('--second', action='store_true', help='Separate approved bench2 output/report')
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    base = a.root/'data/sub/vi38'
    dest = a.dir or base/('bench_out2' if a.second else 'bench_out')
    report_name = 'bench_report2.json' if a.second else 'bench_report.json'
    if a.mode == 'prepare':
        dest.mkdir(parents=True, exist_ok=True)
        lock = a.root/'locks/heavy_job.lock'
        assert not lock.exists(), 'Unexpected heavy lock; not removed'
        print('READY', str(dest), 'lock absent')
        print(json.dumps({name: fingerprint(base/'export'/name)
                          for name in ('manifest.json', 'bench.json')}, indent=2))
        return
    if a.mode == 'verify':
        report = {p.name: fingerprint(p) for p in sorted(dest.iterdir())
                  if p.is_file() and p.name != 'transfer_verification.json'}
        write(dest/'transfer_verification.json', report)
        print(json.dumps(report, indent=2))
        return
    if a.mode == 'failure':
        launch = read(dest/'launch_audit.json')
        parts = [read(p) for p in sorted(dest.glob('bench.part*.json.out.json'))]
        rows = [r for part in parts for r in part['rows']]
        ledgers = [read(p) for p in dest.glob('ledger_*.json')]
        from datetime import datetime
        apps = json.loads((dest/'apps_after.json').read_text(encoding='utf-8-sig')) if (dest/'apps_after.json').exists() else []
        app_wall = sum((datetime.fromisoformat(r['stopped_at'])-datetime.fromisoformat(r['created_at'])).total_seconds()
                       for r in apps if r['app_id'] in launch['app_ids'] and r.get('stopped_at'))
        app_upper = app_wall*launch['rate_upper_usd_s']
        billing = json.loads((dest/'billing_after.json').read_text(encoding='utf-8-sig')) if (dest/'billing_after.json').exists() else []
        billing = [r for r in billing if r['object_id'] in launch['app_ids']]
        egress = sum(float(r['cost']) for r in billing if r['resource'] == 'Network Egress')
        finite = sum(all(isinstance(r.get(f'logp{i}'), (int, float)) and math.isfinite(r[f'logp{i}']) for i in range(3))
                     and math.isfinite(r['score']) for r in rows)
        text = (dest/'modal_bench.log').read_text(encoding='utf-8') if (dest/'modal_bench.log').exists() else ''
        error_lines = [line for line in text.splitlines() if any(word in line for word in
                       ('Error:', 'Error(', 'STOP ', 'AssertionError', 'Timed out', 'TimeoutExpired'))]
        report = dict(status='STOP incomplete bench', expected=1152, received=len(rows),
            unique=len({r['pair_id'] for r in rows if 'repeat_of' not in r}),
            coverage=len(rows)/1152, finite_verified=False, partial_finite=finite, gates_passed=False,
            usd_bench_all_client_wall_upper=max(launch['all_wall_h100_upper_usd'], app_upper)+egress,
            billing_reported_usd=sum(float(r['cost']) for r in billing),
            billing_resources={r['resource']: float(r['cost']) for r in billing},
            billing_note='CLI report snapshot after stop; settlement may lag. Wall upper adds reported network egress.',
            usd_app_lifetime_upper=app_upper, app_lifetime_s=app_wall,
            client_wall_s=launch['client_wall_s'], launch=launch, ledgers=ledgers,
            paid_function_wall_s=sum(r['paid_function_wall'] for r in ledgers),
            usd_function_est=sum(r['usd_est'] for r in ledgers),
            partial_input_tokens=sum(r['n_tokens'] for r in rows),
            partial_roles=dict(Counter(r['role'] for r in rows)),
            partial_labels=dict(Counter(str(r['label']) for r in rows)),
            partial_mean_score=sum(r['score'] for r in rows)/len(rows) if rows else None,
            partial_completed_part_wall_s=max((p['elapsed_since_start'] for p in parts), default=None),
            partial_startup_inclusive_token_s=sum(r['n_tokens'] for r in rows)/max(p['elapsed_since_start'] for p in parts) if parts else None,
            partial_token_boundary_template_rawlogprob_pass=bool(rows) and finite == len(rows),
            partial_validation_note='Successful rows imply hard assertions single-token/template/boundary and raw labels passed; no final token_validation report',
            throughput=None, peak_vram_gib=None, bulk_forecasts=None,
            correlations=None, regions=None, simulated_swaps=None,
            error_lines=error_lines[-12:],
            note='No invented measurements; incomplete benchmark cannot clear gate or authorize bulk')
        if (dest/'equivalence.json').exists():
            report['equivalence'] = read(dest/'equivalence.json')
        write(dest/report_name, report)
        print(json.dumps(report, indent=2))
        return
    if a.mode == 'map-blind':
        # Only operation authorized to access the local private key; never print contents.
        local = dest.parent
        key = read(local/'private/blind_key.json')
        outputs = [r for p in sorted(dest.glob('bench.part*.json.out.json')) for r in read(p)['rows']
                   if 'repeat_of' not in r]
        bypair = {r['pair_id']: r for r in outputs}
        assert 0 < len(bypair) <= 1024
        assert len(bypair) == 1024 or a.allow_partial, 'Incomplete bench: explicit --allow-partial required'
        fields = ('label', 'score', 'logp0', 'logp1', 'logp2') if a.second else ('label', 'score')
        labels = {code: {k: bypair[row['pair_id']][k] for k in fields}
                  for code, row in key.items() if row['pair_id'] in bypair}
        assert len(labels) == len(bypair)
        assert all(v['label'] in (0, 1, 2) and math.isfinite(v['score']) for v in labels.values())
        path = local/'judge/qwen_labels_by_code.json'
        write(path, labels)
        print('MAPPED', len(labels), json.dumps(fingerprint(path)))
        return
    sys.path.insert(0, str(a.root/'scripts'))
    import dwmax_diag as dw
    from llm_judge_run import REVISION, parse_logprobs
    manifest = read(base/'export/manifest.json')
    expected = read(base/'export/bench.json')
    outputs = []
    for name in manifest['modes']['bench']:
        part = read(dest/(name+'.out.json'))
        assert part['input_sha256'] == manifest['files'][name]['sha256']
        outputs.extend(part['rows'])
    byid = {r['request_id']: r for r in outputs}
    assert len(byid) == len(outputs) == len(expected) == 1152
    for row in expected:
        ob = byid[row['request_id']]
        assert all(ob[k] == row[k] for k in ('pair_id', 'prompt_sha256', 'text_sha256', 'role'))
        assert ob['revision'] == REVISION
        parsed = parse_logprobs({i: ob.get(f'logp{i}') for i in range(3)}, (0, 1, 2))
        assert parsed['status'] == 'ok'
        assert parsed['label'] == ob['label'] and parsed['score'] == ob['score']
        assert ob['n_tokens'] == row['n_tokens']
    originals = {r['pair_id']: r for r in outputs if 'repeat_of' not in r}
    repeats = [(originals[r['repeat_of']], r) for r in outputs if 'repeat_of' in r]
    assert len(originals) == 1024 and len(repeats) == 128
    agree = sum(x['label'] == y['label'] for x, y in repeats)/128
    regions = {}
    scores = defaultdict(dict)
    byq = defaultdict(list)
    for row in originals.values():
        scores[row['qid']][str(row['doc_id'])] = row['score']
        byq[row['qid']].append(row)
    for role in ('top30', 'A', 'B', 'tail'):
        rows = [r for r in originals.values() if r['role'] == role]
        counts = Counter(r['label'] for r in rows)
        regions[role] = dict(n=len(rows), counts={str(i): counts[i] for i in range(3)},
                             fractions={str(i): counts[i]/len(rows) for i in range(3)},
                             mean_score=sum(r['score'] for r in rows)/len(rows))
    old = a.root/'data/sub/vi37'
    sources = {'q8b': read(old/'q8b_v37.json'), 'fhyde': read(old/'cache_fhyde_v37.json')}
    docwin = read(a.root/'data/sub/diag37/docwin_max_v37.json')
    sources['docwin'] = {q: [[int(d), None, s] for d, s in ds.items()] for q, ds in docwin.items()}
    lists = {q: list(ds) for q, ds in scores.items()}
    correlations = {name: dw.correlation_report(lists, scores, source) for name, source in sources.items()}
    for name, source in sources.items():
        common_sizes = []
        for q, docs in lists.items():
            available = {str(h[0]) for h in source.get(q, []) if h[2] is not None and math.isfinite(h[2])}
            common_sizes.append(sum(d in available for d in docs))
        correlations[name]['queries_common_ge2'] = sum(n >= 2 for n in common_sizes)
        correlations[name]['queries_common_ge3'] = sum(n >= 3 for n in common_sizes)
        correlations[name]['common_pairs_per_query'] = dict(Counter(common_sizes))
        correlations[name]['caution'] = 'Sparse stratified bench; 2-document query Spearman is +/-1; macro is not a gold-quality measure'
    swaps = {q: min(5, sum(r['role'] == 'A' and r['label'] == 0 for r in rows),
                         sum(r['role'] == 'B' and r['label'] == 2 for r in rows))
             for q, rows in byq.items()}
    def decision(rows):
        aa = sorted([r for r in rows if r['role'] == 'A' and r['label'] == 0],
                    key=lambda r: (r['score'], r.get('rank', 0), r['doc_id']))
        bb = sorted([r for r in rows if r['role'] == 'B' and r['label'] == 2],
                    key=lambda r: (-r['score'], r.get('rank', 0), r['doc_id']))
        return [(x['doc_id'], y['doc_id']) for x, y in list(zip(aa, bb))[:5]]
    replacement = {y['pair_id']: y for _, y in repeats}
    swap_decisions_stable = all(decision(rows) == decision([replacement.get(r['pair_id'], r) for r in rows])
                                for rows in byq.values())
    gpu = read(dest/'report_bench.json')
    launch = read(dest/'launch_audit.json')
    ledgers = [read(p) for p in dest.glob('ledger_*.json')]
    assert len(ledgers) == 1, 'Single approved bench must have exactly one function ledger'
    rate = (3.95+4*.0473+48*.008)/3600
    tokens = sum(r['n_tokens'] for r in outputs)
    bulk_s = manifest['bulk_prompt_tokens']/gpu['prefill_token_s']
    forecasts = {}
    for n in (1, 2):
        wall = bulk_s/n + gpu['load_wall']
        forecasts[str(n)+'_H100'] = dict(wall_s=wall, usd=wall*n*rate,
            usd_with_30pct=wall*n*rate*1.3,
            note='Ideal independent sharding; same measured per-GPU throughput, no 2GPU measurement; excludes full-text confirmation')
    decisions_stable = all(x['label'] == y['label'] for x, y in repeats if x['role'] in ('A', 'B'))
    report = dict(expected=1152, received=len(outputs), unique=1024, finite=1152, coverage=1.,
        repeats=128, label_agreement=agree,
        repeated_ab_eligibility_unchanged=decisions_stable,
        repeated_bench_swap_decisions_unchanged=swap_decisions_stable,
        max_abs_delta_score=max(abs(x['score']-y['score']) for x, y in repeats),
        token_validation=gpu['token_validation'], raw_three_labels_finite=True,
        input_tokens=tokens, tokens_match_cpu=True, prefill_token_s=gpu['prefill_token_s'],
        prefill_note=gpu['prefill_note'], inference_wall_s=gpu['inference_wall'],
        model_load_s=gpu['load_wall'], paid_function_wall_s=sum(r['paid_function_wall'] for r in ledgers),
        runner_wall_s=gpu['paid_function_wall'],
        peak_vram_gib=gpu['peak_vram_gib'], peak_vram_method=gpu['peak_vram_method'],
        usd_function_est=sum(r['usd_est'] for r in ledgers), usd_bench_all_client_wall_upper=launch['all_wall_h100_upper_usd'],
        client_wall_s=launch['client_wall_s'], budget_usd=1, billing_verified=False,
        bulk_pairs=99675, bulk_tokens=manifest['bulk_prompt_tokens'], bulk_forecasts=forecasts,
        regions=regions, correlations=correlations,
        stop_rho_q8b=correlations['q8b']['spearman_macro'] is not None and correlations['q8b']['spearman_macro'] > .9,
        expected_region_order=regions['top30']['mean_score'] > max(regions['A']['mean_score'], regions['B']['mean_score'])
            and min(regions['A']['mean_score'], regions['B']['mean_score']) > regions['tail']['mean_score'],
        simulated_swaps=sum(swaps.values()), simulated_swap_queries=sum(v > 0 for v in swaps.values()),
        swap_note='Bench pairs only, cap5; hypothetical excerpt-only upper count, no full-text confirmation; actual authorized swaps=0',
        gates=dict(finite=True, stability=agree >= .99 and decisions_stable and swap_decisions_stable,
                   budget_upper=launch['all_wall_h100_upper_usd'] < 1),
        unverified=['actual billing/cold-start paid interval', 'independent blind agreement',
                    'full-text confirmation', 'pure kernel prefill throughput', '2-H100 scaling', 'F2 gain'])
    if a.second:
        equivalence = read(dest/'equivalence.json')
        previous = read(base/'bench_out/bench_report.json')
        from datetime import datetime
        apps = read(dest/'apps_after.json') if (dest/'apps_after.json').exists() else []
        appwall = sum((datetime.fromisoformat(r['stopped_at'])-datetime.fromisoformat(r['created_at'])).total_seconds()
                      for r in apps if r['app_id'] in launch['app_ids'] and r.get('stopped_at'))
        billing = read(dest/'billing_after.json') if (dest/'billing_after.json').exists() else []
        cumulative_snapshot = sum(float(r['cost']) for r in billing
                                  if r['object_id'] in launch['app_ids']+previous['launch']['app_ids'])
        billing = [r for r in billing if r['object_id'] in launch['app_ids']]
        egress = sum(float(r['cost']) for r in billing if r['resource'] == 'Network Egress')
        upper = max(appwall*rate, launch['all_wall_h100_upper_usd'])+egress
        report.update(equivalence=equivalence, logprob_method=gpu['logprob_method'],
                      usd_app_lifetime_upper=appwall*rate, app_lifetime_s=appwall,
                      billing_reported_usd=sum(float(r['cost']) for r in billing),
                      billing_note='Snapshot; settlement may lag. Conservative wall upper includes image/cold start plus reported egress.',
                      usd_bench_all_client_wall_upper=upper,
                      usd_vi38_cumulative_upper=upper+previous['usd_bench_all_client_wall_upper'],
                      usd_vi38_cumulative_billing_snapshot=cumulative_snapshot)
        report['bulk_forecast_caution'] = ('Measured effective prompt token/s with prefix caching and 128 repeated requests; '
            'not pure GPU prefill and bulk cache hit rate may differ. 1/2-H100 forecasts include per-GPU measured load, '
            'exclude confirm, assume ideal sharding; +30% is a planning reserve, not a guarantee.')
        report['gates'].update(equivalence=equivalence['passed'], budget_upper=upper < 1)
    write(dest/report_name, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
