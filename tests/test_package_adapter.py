from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.package_adapter import package_adapter, validate_adapter


def write_adapter(path: Path, rank: int = 16) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"r": rank}), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


def test_validate_adapter_accepts_rank_under_limit(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    write_adapter(adapter_dir, rank=32)

    result = validate_adapter(adapter_dir, max_rank=32)

    assert result["rank"] == 32


def test_validate_adapter_rejects_rank_over_limit(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    write_adapter(adapter_dir, rank=64)

    with pytest.raises(ValueError):
        validate_adapter(adapter_dir, max_rank=32)


def test_package_adapter_writes_required_files_at_zip_root(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    output = tmp_path / "submission.zip"
    write_adapter(adapter_dir)

    package_adapter(adapter_dir, output)

    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "adapter_config.json",
            "adapter_model.safetensors",
        }
