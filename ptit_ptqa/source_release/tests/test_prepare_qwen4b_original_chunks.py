from scripts.prepare_qwen4b_original_chunks import digest


def test_digest_is_utf8_deterministic() -> None:
    assert digest("bầu") == digest("bầu")
    assert digest("bầu") != digest("phụ nữ mang thai")
