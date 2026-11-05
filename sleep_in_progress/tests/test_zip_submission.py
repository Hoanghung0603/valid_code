import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from zip_submission import ENTRY, METHODS, pack  # noqa: E402


def _sub(n_doc=2, n_chunk=2):
    return [{"id": i, "relevant_docs": list(range(n_doc)),
             "relevant_chunks": [{"doc_id": j, "chunk_text": f"chunk {j} cua {i}"} for j in range(n_chunk)]}
            for i in range(1, 4)]


def test_pack_default_is_deflate9_and_single_entry(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_sub()), encoding="utf-8")
    z = tmp_path / "a.zip"
    st = pack(p, z)
    assert st["method"] == "deflate" and st["cau"] == 3
    assert st["doc_cau"] == 2 and st["chunk_cau"] == 2
    with zipfile.ZipFile(z) as zf:
        assert zf.namelist() == [ENTRY]
        info = zf.getinfo(ENTRY)
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert zf.read(ENTRY) == p.read_bytes()
    ref = tmp_path / "ref.zip"
    with zipfile.ZipFile(ref, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(p, arcname=ENTRY)
    assert z.stat().st_size == ref.stat().st_size


def test_pack_lzma_roundtrip_and_method_map(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps(_sub(), ensure_ascii=False), encoding="utf-8")
    z = tmp_path / "b.zip"
    st = pack(p, z, method="lzma")
    assert st["method"] == "lzma"
    with zipfile.ZipFile(z) as zf:
        assert zf.namelist() == [ENTRY]
        assert zf.getinfo(ENTRY).compress_type == zipfile.ZIP_LZMA
        obj = json.loads(zf.read(ENTRY))
    assert obj == json.loads(p.read_text(encoding="utf-8"))
    assert set(METHODS) >= {"deflate", "lzma", "bzip2", "stored"}


def test_pack_unknown_method_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("[]", encoding="utf-8")
    try:
        pack(p, tmp_path / "c.zip", method="zzz")
        assert False
    except ValueError:
        pass


def test_cli_lzma_and_deflate(tmp_path):
    script = Path(__file__).resolve().parent.parent / "scripts" / "zip_submission.py"
    p = tmp_path / "d.json"
    p.write_text(json.dumps(_sub()), encoding="utf-8")
    for method in ("deflate", "lzma"):
        z = tmp_path / f"d_{method}.zip"
        r = subprocess.run([sys.executable, str(script), "--json", str(p), "--zip", str(z),
                            "--method", method], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        with zipfile.ZipFile(z) as zf:
            assert zf.read(ENTRY) == p.read_bytes()
        assert method in r.stdout
