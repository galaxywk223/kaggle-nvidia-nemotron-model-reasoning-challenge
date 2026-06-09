from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from src.config import COMPETITION_SLUG, PROJECT_DIR
from src.notebook_templates import REMOTE_COPY_SUBMIT_TEMPLATE


NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"


def title_from_slug(slug: str) -> str:
    words = slug.replace("_", "-").split("-")
    return " ".join(word.upper() if word in {"lb", "svd"} else word.capitalize() for word in words)


def create_remote_submit_notebook(
    *,
    slug: str,
    kernel_source: str | None = None,
    kernel_sources: Sequence[str] | None = None,
    model_source: str | None = None,
    model_sources: Sequence[str] | None = None,
    dataset_source: str | None = None,
    dataset_sources: Sequence[str] | None = None,
    message: str,
    kaggle_id: str | None = None,
    title: str | None = None,
    output_dir: Path = NOTEBOOKS_DIR,
) -> Path:
    if not slug.startswith("submit_"):
        raise ValueError("slug must start with 'submit_'")

    all_kernel_sources = [item for item in ([kernel_source] if kernel_source else []) + list(kernel_sources or []) if item]
    all_model_sources = [item for item in ([model_source] if model_source else []) + list(model_sources or []) if item]
    all_dataset_sources = [item for item in ([dataset_source] if dataset_source else []) + list(dataset_sources or []) if item]
    if not (all_kernel_sources or all_model_sources or all_dataset_sources):
        raise ValueError("at least one source must be provided")

    notebook_dir = output_dir / slug
    notebook_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{slug}.py"
    metadata = {
        "id": kaggle_id or f"galaxy2025/{slug.replace('_', '-')}",
        "title": title or title_from_slug(slug),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": all_dataset_sources,
        "kernel_sources": all_kernel_sources,
        "competition_sources": [COMPETITION_SLUG],
        "model_sources": all_model_sources,
    }

    (notebook_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (notebook_dir / code_file).write_text(
        REMOTE_COPY_SUBMIT_TEMPLATE.format(message=message),
        encoding="utf-8",
    )
    return notebook_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a private Kaggle submit kernel from a source kernel output.")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--kernel-source", action="append", default=[])
    parser.add_argument("--model-source", action="append", default=[])
    parser.add_argument("--dataset-source", action="append", default=[])
    parser.add_argument("--message", required=True)
    parser.add_argument("--kaggle-id", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--output-dir", type=Path, default=NOTEBOOKS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = create_remote_submit_notebook(
        slug=args.slug,
        kernel_sources=args.kernel_source,
        model_sources=args.model_source,
        dataset_sources=args.dataset_source,
        message=args.message,
        kaggle_id=args.kaggle_id,
        title=args.title,
        output_dir=args.output_dir,
    )
    print(f"[notebook] path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
