import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vi31  # noqa: E402
import vi32  # noqa: E402


def test_import_does_not_touch_vi31():
    assert vi31.OUT.name == "vi31" and vi31.NAME == "vi31"


def test_bind_rebinds_then_restore():
    old = vi32.bind()
    try:
        assert vi31.OUT == vi32.OUT and vi31.OUT.name == "vi32"
        assert vi31.G.name == "vi32"
        assert vi31.Q8_V31 == vi32.Q8_NEW and vi32.Q8_NEW.name == "q8b_v32.json"
        assert vi31.CHUNK30 == vi32.CHUNK_V31 and vi32.CHUNK_V31.parent.name == "vi31"
        assert vi31.CHUNK31.name == "chunk_dw_u900_v32.json.zst"
        assert vi31.NAME == "vi32" and vi31.DELTA0 == 57
    finally:
        for k, v in old.items():
            setattr(vi31, k, v)
    assert vi31.OUT.name == "vi31"


def test_export_all_fresh_keeps_new_ids_and_drops_empty():
    fresh = {"1": [[10, 5], [11, 6]], "2": []}
    out = vi32.export_all_fresh(fresh)
    assert out == {"1": [[10, 5, 0.0], [11, 6, 0.0]]}


def test_export_name_matches_vi31_glob():
    assert Path("docs_vi31_old_v32.part000.jsonl.zst").match("docs_vi31_old*.jsonl.zst")
    assert Path("docs_vi31_old_v32.jsonl.zst").match("docs_vi31_old*.jsonl.zst")
