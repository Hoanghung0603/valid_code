"""Cho phép scripts/ import module trong src/ theo kiểu absolute (giống pytest.ini pythonpath = src)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
