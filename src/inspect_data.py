from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR


def inspect_data(data_dir: Path = DATA_DIR) -> dict[str, tuple[int, int]]:
    summary: dict[str, tuple[int, int]] = {}
    for name in ("train.csv", "test.csv"):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV file: {path}")
        frame = pd.read_csv(path)
        summary[name] = frame.shape
        print(f"[{name}] shape={frame.shape}")
        print(frame.dtypes.to_string())
        print(frame.head(3).to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect competition CSV files.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inspect_data(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
