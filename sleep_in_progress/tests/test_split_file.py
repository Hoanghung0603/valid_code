import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from split_file import split_file


def _join(parts):
    return b"".join(p.read_bytes() for p in parts)


def test_split_and_rejoin(tmp_path):
    data = bytes(range(256)) * 1000  # 256.000 byte
    f = tmp_path / "a.zip"
    f.write_bytes(data)
    parts, digest = split_file(f, 100_000)
    assert [p.name for p in parts] == ["a.zip.part000", "a.zip.part001", "a.zip.part002"]
    assert all(p.stat().st_size <= 100_000 for p in parts)
    assert _join(parts) == data
    assert digest == hashlib.sha256(data).hexdigest()


def test_exact_multiple_has_no_empty_part(tmp_path):
    f = tmp_path / "b.bin"
    f.write_bytes(b"x" * 200)
    parts, _ = split_file(f, 100)
    assert len(parts) == 2
    assert _join(parts) == b"x" * 200


def test_small_file_single_part(tmp_path):
    f = tmp_path / "c.bin"
    f.write_bytes(b"abc")
    parts, digest = split_file(f, 1 << 20)
    assert len(parts) == 1 and parts[0].read_bytes() == b"abc"
    assert digest == hashlib.sha256(b"abc").hexdigest()


def test_bad_part_size(tmp_path):
    f = tmp_path / "d.bin"
    f.write_bytes(b"a")
    with pytest.raises(ValueError):
        split_file(f, 0)
