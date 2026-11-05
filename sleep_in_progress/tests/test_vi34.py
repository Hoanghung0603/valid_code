import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vi31  # noqa: E402
import vi33  # noqa: E402
import vi34  # noqa: E402


def test_import_does_not_touch_vi31_vi33():
    assert vi31.OUT.name == "vi31" and vi33.OUT.name == "vi33"
    assert vi33.Q8.name == "q8b_v32.json"


def test_bind_rebinds_then_unbind():
    old = vi34.bind()
    try:
        assert vi31.OUT.name == "vi34" and vi31.G.name == "vi34"
        assert vi31.Q8_V31.as_posix().endswith("diag33/q8b_v33.json")
        assert vi31.CHUNK30.name == "chunk_dw_u900_v33.json.zst" and vi31.CHUNK30.parent.name == "vi33"
        assert vi31.CHUNK31.name == "chunk_dw_u900_v34.json.zst"
        assert vi31.NAME == "vi34" and vi31.DELTA0 == 56
        assert vi33.Q8 == vi34.Q8 and vi33.OUT32.name == "vi33" and vi33.W8F == 2.0
        assert vi33.CHUNK_V32.name == "chunk_dw_u900_v33.json.zst"
    finally:
        vi34.unbind(old)
    assert vi31.OUT.name == "vi31" and vi33.OUT.name == "vi33" and vi33.Q8.name == "q8b_v32.json"


def test_sanity_inputs_are_vi33_inputs():
    assert vi34.Q8_OLD.as_posix().endswith("diag32/q8b_v32.json")
    assert vi34.CHUNK_V33.parent.name == "vi33"
    assert vi34.W8F == vi33.W8F == 2.0 and vi34.DELTA == 56
