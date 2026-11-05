from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleaseConfig:
    path: Path
    root: Path
    payload: dict[str, Any]

    @property
    def release_id(self) -> str:
        return str(self.payload["release_id"])

    def artifact(self, name: str) -> Path:
        return (self.root / self.payload["paths"][name]).resolve()

    def expected_hash(self, name: str) -> str:
        return str(self.payload["sha256"][name])

    def expected_count(self, name: str) -> int:
        return int(self.payload["counts"][name])


def load_release_config(path: str | Path) -> ReleaseConfig:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "r2ai_release_v1":
        raise ValueError("Unsupported or missing release schema_version")
    required = {"release_id", "paths", "sha256", "counts", "models"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Release config is missing: {sorted(missing)}")
    root = config_path.parent.parent
    return ReleaseConfig(path=config_path, root=root, payload=payload)
