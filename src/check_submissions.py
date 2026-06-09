from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from src.config import COMPETITION_SLUG


@dataclass(frozen=True)
class SubmissionRow:
    ref: int
    date: str
    status: str
    public_score: float | None
    private_score: float | None
    description: str


def score_value(raw_score: Any) -> float | None:
    if raw_score in (None, ""):
        return None
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return None


def status_value(raw_status: Any) -> str:
    name = getattr(raw_status, "name", None)
    if name:
        return str(name)
    return str(raw_status).rsplit(".", 1)[-1]


def load_submissions(competition: str = COMPETITION_SLUG, page_size: int = 20) -> list[SubmissionRow]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    rows = []
    for submission in api.competition_submissions(competition=competition, page_size=page_size):
        public_score = getattr(submission, "public_score", getattr(submission, "publicScore", None))
        private_score = getattr(submission, "private_score", getattr(submission, "privateScore", None))
        rows.append(
            SubmissionRow(
                ref=int(submission.ref),
                date=str(submission.date),
                status=status_value(submission.status),
                public_score=score_value(public_score),
                private_score=score_value(private_score),
                description=str(submission.description),
            )
        )
    return rows


def best_complete(rows: list[SubmissionRow]) -> SubmissionRow | None:
    best = None
    for row in rows:
        if row.status == "COMPLETE" and row.public_score is not None:
            if best is None or row.public_score > best.public_score:
                best = row
    return best


def print_rows(rows: list[SubmissionRow], limit: int) -> None:
    for row in rows[:limit]:
        public_score = "" if row.public_score is None else f"{row.public_score:.4f}"
        private_score = "" if row.private_score is None else f"{row.private_score:.4f}"
        print(f"{row.ref}\t{row.status}\t{public_score}\t{private_score}\t{row.date}\t{row.description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print recent Kaggle submissions.")
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_submissions(args.competition, page_size=max(args.page_size, args.limit))
    if not rows:
        print("No submissions found")
        return 0
    print_rows(rows, args.limit)
    best = best_complete(rows)
    if best is not None:
        print(f"best_complete\t{best.ref}\t{best.public_score:.4f}\t{best.description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
