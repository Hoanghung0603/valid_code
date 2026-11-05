"""Offline tests for the pure logic of the Kaggle embedding runner (no torch / GPU needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "embed_runner" / "embed_runner.py"
spec = importlib.util.spec_from_file_location("embed_runner", RUNNER)
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)


def cfg(**kw):
    return {**er.DEFAULTS, **kw}


def test_make_batches_covers_all_and_respects_budget():
    lens = [5, 300, 12, 12, 250, 40, 900, 7]
    batches = er.make_batches(lens, budget=512)
    assert sorted(i for b in batches for i in b) == list(range(len(lens)))
    for b in batches:
        width = max(lens[i] for i in b)
        assert len(b) == 1 or len(b) * width <= 512
    assert [900] == [lens[i] for i in batches[-1]]  # an oversize item sits alone in the last (longest) batch


def test_start_limit_matches_documented_budget():
    c = cfg()
    assert er.start_limit_s(c) == 43200 - 180 - 600 - 1200  # 11 h 27 min before the shard margin
    assert er.start_limit_s(cfg(target_work_s=36000)) == 36000


def test_stop_reason_deadline_margin_and_priorities():
    c = cfg(est_shard_s=600.0)  # margin = 1.25 * 600 = 750 s
    ok = dict(out_gb=1.0, free_gb=10.0, since_spawn_s=0, shards_since_spawn=0, recycle_flag=False)
    limit = er.start_limit_s(c)
    assert er.stop_reason(c, limit - 751, None, **ok) is None
    assert er.stop_reason(c, limit - 749, None, **ok) == "deadline"
    assert er.stop_reason(c, limit - 749, 100.0, **ok) is None  # measured shard time replaces the estimate
    assert er.stop_reason(c, 10, None, **{**ok, "recycle_flag": True}) == "recycle"
    assert er.stop_reason(c, 10, None, **{**ok, "since_spawn_s": 2400}) == "recycle"
    assert er.stop_reason(c, limit, None, **{**ok, "recycle_flag": True}) == "deadline"  # deadline outranks recycle
    assert er.stop_reason(c, 10, None, **{**ok, "out_gb": 16.0}) == "cap"
    assert er.stop_reason(c, 10, None, **{**ok, "free_gb": 1.0}) == "disk"


def test_claims_are_exclusive_resumable_and_released(tmp_path: Path):
    (tmp_path / "claims").mkdir()
    names = ["s0", "s1", "s2"]
    (tmp_path / "s0.done.json").write_text("{}")
    assert er.claim_next(names, tmp_path, 0) == "s1"
    assert er.claim_next(names, tmp_path, 1) == "s2"  # s1 is taken, s0 is done
    assert er.claim_next(names, tmp_path, 1) is None
    assert er.release_claims(tmp_path, 0) == ["s1"]  # worker 0 died holding s1
    assert er.claim_next(names, tmp_path, 1) == "s1"
    (tmp_path / "s2.done.json").write_text("{}")
    assert er.release_claims(tmp_path, 1) == ["s1"]  # finished shard s2 keeps its claim


@pytest.mark.parametrize("cut", [0, 1])
def test_config_rendering_roundtrip(cut):
    import json

    rendered = json.dumps(json.dumps({"run": "x", "budget": 4096}))
    er.CONFIG_JSON = json.loads(rendered)
    try:
        assert er.load_config()["budget"] == 4096
    finally:
        er.CONFIG_JSON = "__CONFIG__"


def test_load_shard_uses_configured_id_column(tmp_path: Path):
    import pandas as pd

    p = tmp_path / "shard-00000.parquet"
    pd.DataFrame({"input_id": [7, 8], "embed_sha1": ["a", "b"], "text": ["x", "y"], "n_tokens": [1, 1]}).to_parquet(p)
    ids, texts = er.load_shard(str(p), cfg(id_col="input_id"))
    assert ids.dtype.name == "int64" and ids.tolist() == [7, 8] and texts == ["x", "y"]
    assert er.load_shard(str(p), cfg(id_col="input_id", prefix="p: "))[1] == ["p: x", "p: y"]


def test_launcher_checks_ids_against_input_shards(tmp_path: Path):
    import json

    import numpy as np
    import pandas as pd

    spec_l = importlib.util.spec_from_file_location("launch_embed", RUNNER.parents[1] / "launch_embed.py")
    le = importlib.util.module_from_spec(spec_l)
    spec_l.loader.exec_module(le)
    inputs, out = tmp_path / "inputs", tmp_path / "out"
    inputs.mkdir()
    out.mkdir()
    pd.DataFrame({"input_id": [0, 1, 2], "text": ["a", "b", "c"]}).to_parquet(inputs / "shard-00000.parquet")
    (out / "shard-00000.done.json").write_text(json.dumps({"shard": "shard-00000"}))
    np.save(out / "shard-00000.ids.npy", np.array([0, 1, 2], dtype=np.int64))
    assert le.verify_against_inputs(out, inputs, "input_id")
    np.save(out / "shard-00000.ids.npy", np.array([0, 2, 1], dtype=np.int64))  # rows out of order
    assert not le.verify_against_inputs(out, inputs, "input_id")
