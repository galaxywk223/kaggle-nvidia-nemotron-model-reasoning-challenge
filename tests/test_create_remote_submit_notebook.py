from __future__ import annotations

import json
from pathlib import Path

from src.create_remote_submit_notebook import create_remote_submit_notebook


def test_create_remote_submit_notebook_writes_metadata_and_fallback_script(tmp_path: Path) -> None:
    notebook_dir = create_remote_submit_notebook(
        slug="submit_example_candidate",
        kernel_source="author/source-kernel",
        message="public example candidate output",
        kaggle_id="galaxy2025/nemotron-example-submit",
        title="Nemotron Example Submit",
        output_dir=tmp_path,
    )

    metadata = json.loads((notebook_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    script = (notebook_dir / "submit_example_candidate.py").read_text(encoding="utf-8")

    assert metadata["id"] == "galaxy2025/nemotron-example-submit"
    assert metadata["kernel_sources"] == ["author/source-kernel"]
    assert metadata["competition_sources"] == ["nvidia-nemotron-model-reasoning-challenge"]
    assert "public example candidate output" in script
    assert "find_valid_source_zip" in script
    assert 'rglob("*.zip")' in script
    assert "zip_priority" in script
    assert "find_adapter_dir" in script
    assert "package_adapter_dir" in script
    assert "MIN_ADAPTER_BYTES" in script


def test_create_remote_submit_notebook_supports_model_sources(tmp_path: Path) -> None:
    notebook_dir = create_remote_submit_notebook(
        slug="submit_model_candidate",
        model_source="author/model/Transformers/adapter/1",
        message="public model candidate output",
        output_dir=tmp_path,
    )

    metadata = json.loads((notebook_dir / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["kernel_sources"] == []
    assert metadata["model_sources"] == ["author/model/Transformers/adapter/1"]
