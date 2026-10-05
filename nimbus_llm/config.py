"""Shared config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    full = Path(path)
    if not full.is_absolute():
        full = ROOT / full
    with full.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {full}")
    return data


def repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
