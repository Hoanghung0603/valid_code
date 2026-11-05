"""CPU-only q3e8B ablation against identical vi52/vi53 rules, no ZIP build."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import dwmax_diag as dw
import vi52
import vi54

TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))


def main():
    ctx = dw.read(TEMP/'vi50/fusion_context.json.zst')
    data, bge = ctx['data'], ctx['bge']
    d8w = dw.read(TEMP/'vi50/d8w.json')
    q3e = dw.read(TEMP/'vi51/cosines_vi51b.json')
    original = vi52.fused(data, bge, d8w, q3e)
    ablation = vi54.fused(data, bge, d8w, q3e, {})
    exact = sum(original[q] == ablation[q] for q in original)
    assert exact == 1200
    report = dict(q3e_only_exact_rankings=exact, total_queries=1200,
                  reference='self-built vi41c+d8w+q3e8B using vi52.fused; no ZIP compression',
                  zip_builds=0, modal_gpu_calls=0)
    (TEMP/'vi54/sanity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
