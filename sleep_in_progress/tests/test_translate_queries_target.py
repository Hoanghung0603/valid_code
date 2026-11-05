import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from translate_queries import is_english, translate, _TARGETS  # noqa: E402


def test_targets_have_en():
    assert set(_TARGETS) == {"zh", "en"}
    assert _TARGETS["en"]["field"] == "en"
    assert _TARGETS["zh"]["field"] == "zh"


def test_is_english_accepts_medical_query():
    assert is_english("What does a Beta-HCG result below 0.10 U/L mean?")
    assert is_english("Black spots on teeth and bleeding on waking up")


def test_is_english_rejects_vietnamese_copy():
    assert not is_english("Răng có lỗ chấm đen, chảy máu khi thức dậy là sao?")
    assert not is_english("")
    assert not is_english("ab")


def test_translate_uses_check_for_target():
    # giả lập server: luôn trả lại câu gốc tiếng Việt -> không qua được check -> ''
    def fake_urlopen(req, timeout=None):
        import io
        import json as _json

        class Resp:
            def read(self):
                return _json.dumps({"choices": [{"message": {"content": "Răng bị đau thì làm sao?"},
                                                 "finish_reason": "stop"}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return Resp()

    import translate_queries as tq
    orig = tq.urllib.request.urlopen
    tq.urllib.request.urlopen = fake_urlopen
    try:
        assert translate("Răng bị đau thì làm sao?", "http://x", "m", 100, target="en", tries=1) == ""
    finally:
        tq.urllib.request.urlopen = orig
