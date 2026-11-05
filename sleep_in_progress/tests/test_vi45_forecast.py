import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy'))
from vi45_forecast import forecast


def test_forecast_blocks_before_credit_or_six():
    bench=dict(input_tokens=512000,unique_pairs=512,token_s=30000,load_wall=60)
    result=forecast(bench,dict(scored_candidates=296973),.3)
    assert not result['allowed']
    assert result['forecast_total_upper']>6


def test_forecast_keeps_fixed_margin_and_population():
    bench=dict(input_tokens=512000,unique_pairs=512,token_s=100000,load_wall=50)
    result=forecast(bench,dict(scored_candidates=1000),.2)
    assert result['allowed']
    assert result['bulk_seconds_plus30pct']==78
    assert result['pairs']==1000


def test_explicit_approved_budget_twelve():
    bench=dict(input_tokens=512000,unique_pairs=512,token_s=40000,load_wall=60)
    result=forecast(bench,dict(scored_candidates=200000),.32,credit=30,budget=12)
    assert result['allowed'] and result['budget_cap']==12
