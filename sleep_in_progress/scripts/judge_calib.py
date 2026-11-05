"""Aggregate historical submissions against fixed, existing judge scores; no gold labels.

Run: python scripts/judge_calib.py --temp <HIST_TMP>
The output is diagnostic, not a submission selector or a source of per-document labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
G_EST = 72.0  # rough 168 * 0.2211 / 0.5136; ratio of macros, NOT actual mean gold count
THRESHOLDS = {'q8b': 0.0, 'bge': -2.0, 'docwin': -2.0}  # fixed, not probabilities


def iter_json_array(stream, chunk_size=262144):
    """Decode one submission query at a time, without holding all chunk text in RAM."""
    import codecs
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder('utf-8')()
    buf = ''
    eof = started = ended = False
    while not ended:
        if not eof:
            data = stream.read(chunk_size)
            eof = not data
            buf += utf8.decode(data, final=eof)
        while True:
            buf = buf.lstrip()
            if not started:
                if not buf:
                    break
                if buf[0] != '[':
                    raise ValueError('submission must be an array')
                started, buf = True, buf[1:]
                continue
            if not buf:
                break
            if buf[0] == ']':
                ended, buf = True, buf[1:]
                break
            if buf[0] == ',':
                buf = buf[1:].lstrip()
            try:
                item, pos = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                if eof:
                    raise
                break
            yield item
            buf = buf[pos:]
        if eof and not ended:
            raise ValueError('truncated JSON array')


def score_map(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return {str(q): {str(h[0]): float(h[2]) for h in hits
                     if h[2] is not None and math.isfinite(float(h[2]))}
            for q, hits in data.items()}


def docwin_map(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return {str(q): {str(d): float(s) for d, s in hits.items()
                     if s is not None and math.isfinite(float(s))} for q, hits in data.items()}


def metrics(rows, scores, threshold):
    precisions, lower, upper, counts = [], [], [], []
    missing = total = 0
    for row in rows:
        docs = [str(d) for d in row['relevant_docs']]
        if len(set(docs)) != len(docs):
            raise ValueError('duplicate doc ID')
        hits = scores.get(str(row['id']), {})
        observed = [hits[d] for d in docs if d in hits]
        n, k = len(docs), len(observed)
        good = sum(v > threshold for v in observed)
        total += n
        missing += n - k
        lower.append(good / n if n else 0.0)
        upper.append((good + n - k) / n if n else 0.0)
        precisions.append(good / k if k else (0.0 if not n else float('nan')))
        counts.append(good / k * n if k else (0.0 if not n else float('nan')))
    # Missing-at-random adjustment is only a sensitivity proxy; all-missing query => abstain.
    valid = all(math.isfinite(v) for v in precisions)
    return {'missing': missing, 'total': total, 'missing_rate': missing / total if total else 0,
            'p_conditional': float(np.mean(precisions)) if valid else None,
            'p_lower': float(np.mean(lower)), 'p_upper': float(np.mean(upper)),
            'correct_per_query_adjusted': float(np.mean(counts)) if valid else None,
            'r_proxy': float(np.mean(counts)) / G_EST if valid else None}


def correlation(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {'n': len(x), 'pearson': None, 'spearman': None}
    return {'n': len(x), 'pearson': float(pearsonr(x, y).statistic),
            'spearman': float(spearmanr(x, y).statistic)}


def pair_stats(rows, key, target, max_final_diff=.01):
    correct = wrong = ties = 0
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if abs(a['Final']-b['Final']) >= max_final_diff or a['fingerprint'] == b['fingerprint']:
                continue
            x, y = a[key] - b[key], a[target] - b[target]
            if abs(y) < 1e-12:
                continue  # public ties cannot have a direction
            if abs(x) < 1e-12:
                ties += 1
            elif x*y > 0:
                correct += 1
            else:
                wrong += 1
    return {'correct': correct, 'wrong': wrong, 'proxy_ties': ties,
            'accuracy_decided': correct/(correct+wrong) if correct+wrong else None,
            'accuracy_all': correct/(correct+wrong+ties) if correct+wrong+ties else None}


def loo_predictions(rows):
    """Two-feature linear fit, held out by *doclist* to avoid duplicate leakage."""
    for row in rows:
        train = [r for r in rows if r['fingerprint'] != row['fingerprint']]
        if len(train) < 4:
            row['loo_final'] = None
            continue
        x = np.array([[1, r['p'], r['r']] for r in train])
        y = np.array([r['Final'] for r in train])
        row['loo_final'] = float(np.array([1, row['p'], row['r']]) @ np.linalg.lstsq(x, y, rcond=None)[0])


def run(temp):
    scores = {'q8b': score_map(temp / 'diag33/q8b_v37_calib.json'),
              'bge': score_map(temp / 'diag33/pool_v36_calib.json'),
              'docwin': docwin_map(temp / 'diag33/docwin_v37_calib.json')}
    records = {}
    for date in ('20261026', '20261030'):
        with (ROOT / f'data/submissions_backup_{date}.json').open(encoding='utf-8') as f:
            backup = json.load(f)
        if isinstance(backup, list):
            names = {'FINAL_SCORE': 'Final', 'DOCS_F2MACRO': 'Docs_F2',
                     'CHUNKS_F2MACRO': 'Chunks_F2', 'DOCS_PRECISION': 'Docs_P',
                     'DOCS_RECALL': 'Docs_R'}
            backup = {'records': [{'id': r['id'], 'filename': r['file'],
                                   'created_when': r['t'],
                                   'metrics': {v: r.get('scores', {}).get(k) for k, v in names.items()}}
                                  for r in backup]}
        for r in backup['records']:
            records[r['id']] = r
    paths = {}
    for p in temp.rglob('submission*.zip'):
        paths.setdefault(p.name, []).append(p)
    result, skipped = [], []
    for rec in sorted(records.values(), key=lambda r: r['id']):
        m = rec.get('metrics') or {}
        if not all(m.get(k) is not None for k in ('Docs_P', 'Docs_R', 'Final')):
            continue
        size = (rec.get('dataset') or {}).get('submission_file_size_bytes')
        candidates = [p for p in paths.get(rec['filename'], []) if not size or p.stat().st_size == size]
        if not size and len(candidates) != 1:
            continue  # old backup lacks byte size: require unambiguous path
        if not candidates:
            continue
        path = sorted(candidates)[0]
        try:
            with zipfile.ZipFile(path) as z:
                if z.namelist() != ['submission.json']:
                    raise ValueError('unexpected ZIP entry')
                with z.open('submission.json') as src:
                    # Store docs only; chunk text is released as each row is read.
                    rows = [{'id': x['id'], 'relevant_docs': x['relevant_docs']}
                            for x in iter_json_array(src)]
            if len(rows) != 1200 or len({str(x['id']) for x in rows}) != 1200:
                raise ValueError('expected 1200 unique queries')
            digest = hashlib.sha256(json.dumps(sorted((str(x['id']), sorted(map(str, x['relevant_docs'])))
                              for x in rows), separators=(',', ':')).encode()).hexdigest()
            out = {'id': rec['id'], 'filename': rec['filename'], 'path': str(path),
                   'date': rec['created_when'], 'fingerprint': digest,
                   **{k: float(m[k]) for k in ('Docs_P', 'Docs_R', 'Final', 'Docs_F2', 'Chunks_F2')}}
            out['judge'] = {k: metrics(rows, scores[k], threshold) for k, threshold in THRESHOLDS.items()}
            result.append(out)
            print(f"{out['id']} {path.name} q8missing={out['judge']['q8b']['missing_rate']:.4f}", flush=True)
        except (ValueError, KeyError, zipfile.BadZipFile) as exc:
            skipped.append({'id': rec['id'], 'path': str(path), 'reason': str(exc)})
    report = {'thresholds': THRESHOLDS, 'G_est_heuristic': G_EST, 'submissions': result, 'skipped': skipped}
    modern_all = [r for r in result if r['date'] >= '2026-10-24T00:00:00']
    same_docs = [(a, b) for i, a in enumerate(modern_all) for b in modern_all[i+1:]
                 if a['fingerprint'] == b['fingerprint'] and a['id'] != b['id']]
    report['same_docs_different_final'] = [
        {'ids': [a['id'], b['id']], 'delta_final': round(b['Final'] - a['Final'], 4)}
        for a, b in same_docs if abs(a['Final']-b['Final']) > 1e-12]
    for judge in scores:
        # 95% coverage chosen a priori; the transition day itself is excluded for Final.
        valid = [r for r in result if r['judge'][judge]['missing_rate'] <= .05
                 and r['judge'][judge]['p_conditional'] is not None]
        flat = [{**r, 'p': r['judge'][judge]['p_conditional'], 'r': r['judge'][judge]['r_proxy']}
                for r in valid]
        corr = {f'{proxy}_vs_{target}': correlation([r[proxy] for r in flat], [r[target] for r in flat])
                for proxy in ('p', 'r') for target in ('Docs_P', 'Docs_R')}
        unique = list({r['fingerprint']: r for r in flat}.values())
        unique_corr = {f'{proxy}_vs_{target}': correlation([r[proxy] for r in unique], [r[target] for r in unique])
                       for proxy in ('p', 'r') for target in ('Docs_P', 'Docs_R')}
        close = {f'{proxy}_vs_{target}': pair_stats(flat, proxy, target)
                 for proxy in ('p', 'r') for target in ('Docs_P', 'Docs_R')}
        modern = [r for r in flat if r['date'] >= '2026-10-24T00:00:00']
        close_modern = {f'{proxy}_vs_{target}': pair_stats(modern, proxy, target)
                        for proxy in ('p', 'r') for target in ('Docs_P', 'Docs_R')}
        loo_predictions(modern)
        modern = [r for r in modern if r['loo_final'] is not None]
        errors = [abs((a['loo_final']-b['loo_final'])-(a['Final']-b['Final']))
                  for i, a in enumerate(modern) for b in modern[i+1:]
                  if a['fingerprint'] != b['fingerprint'] and abs(a['Final']-b['Final']) < .01]
        report[judge] = {'eligible': len(flat), 'coverage_rule': 'missing <= 5%; no all-missing query',
                         'correlation': corr, 'unique_doclist_correlation': unique_corr, 'close_pairs': close,
                         'close_pairs_modern': close_modern,
                         'final_loo': {'n': len(modern), 'pair': pair_stats(modern, 'loo_final', 'Final'),
                                       'pair_delta_mae': float(np.mean(errors)) if errors else None,
                                       'pair_delta_p90': float(np.percentile(errors,90)) if errors else None}}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--temp', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'data/judge_calib.json')
    args = ap.parse_args()
    report = run(args.temp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('q8b', 'bge', 'docwin', 'skipped')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
