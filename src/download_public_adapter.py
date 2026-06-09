from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path

from src.config import ADAPTER_DIR, PUBLIC_ADAPTER_VERSION, kaggle_executable


def _extract_downloaded_archives(adapter_dir: Path) -> None:
    for archive_path in adapter_dir.glob("*.tar"):
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(adapter_dir, filter="data")
    for archive_path in adapter_dir.glob("*.tar.gz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(adapter_dir, filter="data")


def download_public_adapter(
    adapter_dir: Path = ADAPTER_DIR,
    model_version: str = PUBLIC_ADAPTER_VERSION,
    force: bool = False,
) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    command = [
        kaggle_executable(),
        "models",
        "variations",
        "versions",
        "download",
        model_version,
        "-p",
        str(adapter_dir),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)
    _extract_downloaded_archives(adapter_dir)
    return adapter_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the public baseline adapter.")
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--model-version", default=PUBLIC_ADAPTER_VERSION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = download_public_adapter(args.adapter_dir, args.model_version, args.force)
    print(f"[adapter] downloaded={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
