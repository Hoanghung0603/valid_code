"""One predeclared forecast; never launches GPU or changes candidate policy."""
import json
from pathlib import Path

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')


def forecast(bench, prepare, spent_upper, credit=5., budget=6.):
    # Mean from the 512 unique actual vi42 chunks, not cache-hit repeats.
    # Record both the measured mean and the conservative max-len bound.
    # input_tokens includes repeat50, so using /512 intentionally overestimates.
    token_est=prepare['scored_candidates']*min(1024.,bench['input_tokens']/bench['unique_pairs'])
    token_upper=prepare['scored_candidates']*1024
    seconds=(token_est/bench['token_s']+bench['load_wall'])*1.3
    rate=(3.95+4*.0473+48*.008)/3600
    cost=seconds*rate
    return dict(pairs=prepare['scored_candidates'],input_token_est=token_est,input_token_cap=token_upper,
        measured_token_s=bench['token_s'],bulk_seconds_plus30pct=seconds,bulk_usd_plus30pct=cost,
        bench_spent_upper=spent_upper,forecast_total_upper=cost+spent_upper,credit_assumed=credit,
        budget_cap=budget,allowed=cost+spent_upper<=min(credit,budget),
        note='512 random vi42 chunks; mean conservatively includes 50 repeat tokens; +30% load/inference; no tuning')


if __name__=='__main__':
    bench=json.loads((TEMP/'bench_report.json').read_text())
    prepare=json.loads((TEMP/'prepare_report.json').read_text())
    state=json.loads((TEMP/'state.json').read_text())
    wall=json.loads((TEMP/'bench_function_wall.json').read_text())
    prior=json.loads((TEMP/'bench_attempt1_wall.json').read_text())
    # Include both terminal attempts; 20 seconds scheduling/shutdown each + CPU/image reserve.
    upper=(wall['seconds']+prior['seconds']+40)*4.5232/3600+.03
    result=forecast(bench,prepare,upper)
    (TEMP/'forecast.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
