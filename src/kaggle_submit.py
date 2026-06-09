from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import COMPETITION_SLUG, SUBMISSION_LOG, SUBMISSION_ZIP


def coerce_submission_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    return None


def empty_string_to_none(value: Any) -> Any:
    if value == "":
        return None
    return value


def submission_status_name(submission: Any) -> str | None:
    status = getattr(submission, "status", None)
    if status is None:
        return None
    return getattr(status, "name", str(status))


def fetch_submission_pages(
    api: KaggleApi,
    competition: str,
    max_pages: int,
    page_size: int,
) -> list[Any]:
    submissions: list[Any] = []
    for page_number in range(1, max_pages + 1):
        page = api.competition_submissions(
            competition=competition,
            page_number=page_number,
            page_size=page_size,
        )
        if not page:
            break
        submissions.extend(page)
        if len(page) < page_size:
            break
    return submissions


def find_matching_submission(
    submissions: list[Any],
    file_name: str,
    message: str,
    started_after: datetime,
) -> Any | None:
    matches = []
    for submission in submissions:
        submission_date = coerce_submission_datetime(getattr(submission, "date", None))
        if submission_date is None or submission_date < started_after.replace(tzinfo=None):
            continue
        if getattr(submission, "file_name", "") != file_name:
            continue
        description = getattr(submission, "description", "") or ""
        if message and description != message:
            continue
        matches.append(submission)

    if not matches:
        return None
    matches.sort(key=lambda item: coerce_submission_datetime(getattr(item, "date", None)) or datetime.min)
    return matches[-1]


def build_submission_record(
    competition: str,
    file_path: Path,
    message: str,
    response_message: str,
    submission: Any | None,
    submitted_at: datetime,
) -> dict[str, Any]:
    record = {
        "competition": competition,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "message": message,
        "response_message": response_message,
        "submitted_at_utc": submitted_at.astimezone(timezone.utc).isoformat(),
        "submission_ref": None,
        "submission_date": None,
        "status": None,
        "public_score": None,
        "private_score": None,
        "team_name": None,
        "submitted_by": None,
        "error_description": None,
    }
    if submission is None:
        return record

    submission_date = coerce_submission_datetime(getattr(submission, "date", None))
    record.update(
        {
            "submission_ref": getattr(submission, "ref", None),
            "submission_date": submission_date.isoformat() if submission_date else None,
            "status": submission_status_name(submission),
            "public_score": empty_string_to_none(
                getattr(submission, "public_score", getattr(submission, "publicScore", None))
            ),
            "private_score": empty_string_to_none(
                getattr(submission, "private_score", getattr(submission, "privateScore", None))
            ),
            "team_name": empty_string_to_none(getattr(submission, "team_name", None)),
            "submitted_by": empty_string_to_none(getattr(submission, "submitted_by", None)),
            "error_description": empty_string_to_none(getattr(submission, "error_description", None)),
        }
    )
    return record


def append_submission_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def submit_and_wait(
    file_path: Path,
    message: str,
    competition: str = COMPETITION_SLUG,
    log_path: Path = SUBMISSION_LOG,
    wait_timeout_seconds: int = 900,
    poll_interval_seconds: int = 20,
    page_size: int = 20,
    max_pages: int = 2,
) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Submission file does not exist: {file_path}")

    api = KaggleApi()
    api.authenticate()
    started_at = datetime.now(timezone.utc)
    started_after = started_at - timedelta(minutes=1)

    response = api.competition_submit(
        file_name=str(file_path),
        message=message,
        competition=competition,
        quiet=False,
    )
    response_message = getattr(response, "message", str(response))

    submission = None
    deadline = time.monotonic() + wait_timeout_seconds
    while time.monotonic() <= deadline:
        submission = find_matching_submission(
            submissions=fetch_submission_pages(api, competition, max_pages, page_size),
            file_name=file_path.name,
            message=message,
            started_after=started_after,
        )
        if submission is not None:
            status = submission_status_name(submission)
            public_score = empty_string_to_none(
                getattr(submission, "public_score", getattr(submission, "publicScore", None))
            )
            private_score = empty_string_to_none(
                getattr(submission, "private_score", getattr(submission, "privateScore", None))
            )
            if status == "COMPLETE" or public_score is not None or private_score is not None:
                break
            if status in {"ERROR", "FAILED"}:
                break
        time.sleep(poll_interval_seconds)

    record = build_submission_record(
        competition=competition,
        file_path=file_path,
        message=message,
        response_message=response_message,
        submission=submission,
        submitted_at=started_at,
    )
    append_submission_log(log_path, record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the packaged LoRA adapter to Kaggle.")
    parser.add_argument("--file", type=Path, default=SUBMISSION_ZIP)
    parser.add_argument("--message", required=True)
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    parser.add_argument("--log-path", type=Path, default=SUBMISSION_LOG)
    parser.add_argument("--wait-timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-interval-seconds", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = submit_and_wait(
        file_path=args.file.expanduser().resolve(),
        message=args.message,
        competition=args.competition,
        log_path=args.log_path,
        wait_timeout_seconds=args.wait_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    print(f"[submit] file={record['file_name']}")
    print(f"[status] ref={record['submission_ref']} status={record['status']}")
    print(f"[score] public={record['public_score']} private={record['private_score']}")
    print(f"[saved] log={args.log_path}")
    if record["status"] in {"ERROR", "FAILED"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
