from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from src.check_submissions import SubmissionRow, load_submissions
from src.config import COMPETITION_SLUG
from src.kaggle_submit import coerce_submission_datetime


@dataclass(frozen=True)
class SubmissionQuota:
    competition: str
    utc_date: str
    max_daily_submissions: int | None
    submissions_today: int

    @property
    def remaining(self) -> int | None:
        if self.max_daily_submissions is None:
            return None
        return max(self.max_daily_submissions - self.submissions_today, 0)

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0 if self.remaining is not None else False


def max_daily_submissions(api: KaggleApi, competition: str) -> int | None:
    response = api.competitions_list(search=competition)
    for item in getattr(response, "competitions", []) or []:
        ref = getattr(item, "ref", "") or ""
        url = getattr(item, "url", "") or ""
        if ref.endswith(f"/{competition}") or url.endswith(f"/{competition}"):
            value = getattr(item, "max_daily_submissions", getattr(item, "maxDailySubmissions", None))
            return int(value) if value is not None else None
    return None


def submission_utc_date(row: SubmissionRow) -> str | None:
    parsed = coerce_submission_datetime(row.date)
    if parsed is None:
        return None
    return parsed.replace(tzinfo=timezone.utc).date().isoformat()


def quota_status(
    *,
    competition: str = COMPETITION_SLUG,
    rows: list[SubmissionRow] | None = None,
    now: datetime | None = None,
    api: KaggleApi | None = None,
    page_size: int = 20,
) -> SubmissionQuota:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    utc_date = current.astimezone(timezone.utc).date().isoformat()

    kaggle_api = api or KaggleApi()
    if api is None:
        kaggle_api.authenticate()
    max_daily = max_daily_submissions(kaggle_api, competition)
    submissions = rows if rows is not None else load_submissions(competition, page_size=page_size)
    submissions_today = sum(1 for row in submissions if submission_utc_date(row) == utc_date)
    return SubmissionQuota(
        competition=competition,
        utc_date=utc_date,
        max_daily_submissions=max_daily,
        submissions_today=submissions_today,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print current UTC daily submission quota status.")
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    parser.add_argument("--page-size", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = quota_status(competition=args.competition, page_size=args.page_size)
    print(f"competition\t{status.competition}")
    print(f"utc_date\t{status.utc_date}")
    print(f"max_daily_submissions\t{status.max_daily_submissions}")
    print(f"submissions_today\t{status.submissions_today}")
    print(f"remaining\t{status.remaining}")
    print(f"exhausted\t{status.exhausted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
