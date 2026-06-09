from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from src.config import ADAPTER_DIR, REQUIRED_ADAPTER_FILES, SUBMISSION_ZIP


def _read_rank(config: dict[str, Any]) -> int | None:
    raw_rank = config.get("r") or config.get("rank") or config.get("lora_rank")
    if raw_rank is None:
        return None
    try:
        return int(raw_rank)
    except (TypeError, ValueError):
        return None


def validate_adapter(adapter_dir: Path = ADAPTER_DIR, max_rank: int = 32) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required adapter files: {', '.join(missing)}")

    config_path = adapter_dir / "adapter_config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rank = _read_rank(config)
    if rank is not None and rank > max_rank:
        raise ValueError(f"LoRA rank exceeds maximum {max_rank}: {rank}")

    return {"rank": rank, "config": config}


def package_adapter(
    adapter_dir: Path = ADAPTER_DIR,
    output: Path = SUBMISSION_ZIP,
    max_rank: int = 32,
) -> Path:
    validate_adapter(adapter_dir, max_rank=max_rank)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_ADAPTER_FILES:
            archive.write(adapter_dir / name, arcname=name)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    if "adapter_config.json" not in names:
        raise RuntimeError("submission.zip does not contain adapter_config.json at the root")
    if "adapter_model.safetensors" not in names:
        raise RuntimeError("submission.zip does not contain adapter_model.safetensors at the root")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and package a LoRA adapter.")
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--output", type=Path, default=SUBMISSION_ZIP)
    parser.add_argument("--max-rank", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = package_adapter(args.adapter_dir, args.output, max_rank=args.max_rank)
    print(f"[submission] zip={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
