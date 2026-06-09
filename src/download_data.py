from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

from src.config import COMPETITION_SLUG, DATA_DIR, kaggle_executable


def download_data(data_dir: Path = DATA_DIR, competition: str = COMPETITION_SLUG) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / f"{competition}.zip"
    command = [
        kaggle_executable(),
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(data_dir),
        "--force",
    ]
    subprocess.run(command, check=True)
    if not archive_path.exists():
        raise FileNotFoundError(f"Expected Kaggle archive was not created: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(data_dir)

    for name in ("train.csv", "test.csv"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Missing expected competition file: {name}")
    return data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract competition data.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = download_data(args.data_dir, args.competition)
    print(f"[data] downloaded={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
