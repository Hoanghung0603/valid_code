import pytest

from medret.eval.tokenize import hf_tokenize


@pytest.mark.network
def test_bge_m3_tokenizer_has_no_special_tokens():
    tok = hf_tokenize("BAAI/bge-m3")
    ids = tok(["sỏi thận", "腎結石", "kidney stone"])
    assert all(ids) and 0 not in ids[0] and 2 not in ids[0]  # <s>=0, </s>=2 in XLM-R vocab
