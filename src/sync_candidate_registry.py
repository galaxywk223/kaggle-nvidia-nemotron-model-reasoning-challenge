from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.check_submissions import SubmissionRow, load_submissions
from src.config import COMPETITION_SLUG, PROJECT_DIR
from src.kaggle_submit import coerce_submission_datetime


CANDIDATE_REGISTRY = PROJECT_DIR / "candidate_registry.json"


def load_registry(path: Path = CANDIDATE_REGISTRY) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_submission(left: SubmissionRow, right: SubmissionRow) -> SubmissionRow:
    left_date = coerce_submission_datetime(left.date)
    right_date = coerce_submission_datetime(right.date)
    if left_date is None:
        return right
    if right_date is None:
        return left
    return right if right_date > left_date else left


def submission_indexes(submissions: list[SubmissionRow]) -> tuple[dict[int, SubmissionRow], dict[str, SubmissionRow]]:
    by_ref = {row.ref: row for row in submissions}
    by_description: dict[str, SubmissionRow] = {}
    for row in submissions:
        if not row.description:
            continue
        existing = by_description.get(row.description)
        by_description[row.description] = row if existing is None else latest_submission(existing, row)
    return by_ref, by_description


def sync_registry_scores(
    registry: list[dict[str, Any]],
    submissions: list[SubmissionRow],
) -> list[dict[str, Any]]:
    by_ref, by_description = submission_indexes(submissions)
    synced = []
    for candidate in registry:
        updated = dict(candidate)
        ref = updated.get("submission_ref")
        row = by_ref.get(ref)
        if row is None:
            message = updated.get("submission_message")
            if message:
                row = by_description.get(message)
        if row is not None:
            updated["submission_ref"] = row.ref
            updated["submission_status"] = row.status
            updated["public_score"] = row.public_score
            updated["private_score"] = row.private_score
            updated["submission_date"] = row.date
        synced.append(updated)
    return synced


def save_registry(registry: list[dict[str, Any]], path: Path = CANDIDATE_REGISTRY) -> None:
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync candidate registry scores from Kaggle submissions.")
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    parser.add_argument("--registry", type=Path, default=CANDIDATE_REGISTRY)
    parser.add_argument("--page-size", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    submissions = load_submissions(args.competition, page_size=args.page_size)
    synced = sync_registry_scores(registry, submissions)
    save_registry(synced, args.registry)
    print(f"[sync] candidates={len(synced)} submissions={len(submissions)} registry={args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
