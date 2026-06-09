from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import COMPETITION_SLUG, SUBMISSION_LOG, SUBMISSION_ZIP
from src.kaggle_submit import append_submission_log, build_submission_record, empty_string_to_none


def status_name(submission: Any) -> str:
    status = getattr(submission, "status", None)
    return getattr(status, "name", str(status))


def public_score(submission: Any) -> Any:
    return empty_string_to_none(
        getattr(submission, "public_score", getattr(submission, "publicScore", None))
    )


def load_matching_submission(
    api: KaggleApi,
    competition: str,
    ref: int | None,
    message: str | None,
    page_size: int,
) -> Any | None:
    rows = api.competition_submissions(competition=competition, page_size=page_size)
    for submission in rows:
        if ref is not None and int(getattr(submission, "ref", -1)) == ref:
            return submission
        if message and getattr(submission, "description", "") == message:
            return submission
    return None


def watch_submission(
    *,
    competition: str,
    file_path: Path,
    ref: int | None,
    message: str | None,
    log_path: Path,
    timeout_seconds: int,
    poll_interval_seconds: int,
    page_size: int,
) -> dict[str, Any] | None:
    api = KaggleApi()
    api.authenticate()
    deadline = time.monotonic() + timeout_seconds
    latest = None

    while time.monotonic() <= deadline:
        latest = load_matching_submission(api, competition, ref, message, page_size)
        if latest is not None:
            status = status_name(latest)
            score = public_score(latest)
            print(f"[watch] ref={getattr(latest, 'ref', None)} status={status} public_score={score}")
            if status in {"COMPLETE", "ERROR", "FAILED"} or score is not None:
                record = build_submission_record(
                    competition=competition,
                    file_path=file_path,
                    message=message or getattr(latest, "description", ""),
                    response_message="watcher final state",
                    submission=latest,
                    submitted_at=datetime.now(timezone.utc),
                )
                append_submission_log(log_path, record)
                return record
        else:
            print("[watch] submission_not_found")
        time.sleep(poll_interval_seconds)

    if latest is not None:
        record = build_submission_record(
            competition=competition,
            file_path=file_path,
            message=message or getattr(latest, "description", ""),
            response_message="watcher timeout",
            submission=latest,
            submitted_at=datetime.now(timezone.utc),
        )
        append_submission_log(log_path, record)
        return record
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a Kaggle submission until it finishes.")
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    parser.add_argument("--file", type=Path, default=SUBMISSION_ZIP)
    parser.add_argument("--ref", type=int, default=None)
    parser.add_argument("--message", default=None)
    parser.add_argument("--log-path", type=Path, default=SUBMISSION_LOG)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--poll-interval-seconds", type=int, default=120)
    parser.add_argument("--page-size", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = watch_submission(
        competition=args.competition,
        file_path=args.file.expanduser().resolve(),
        ref=args.ref,
        message=args.message,
        log_path=args.log_path,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        page_size=args.page_size,
    )
    if record is None:
        print("[watch] no final record")
        return 1
    print(f"[watch] final_status={record['status']} public_score={record['public_score']}")
    return 0 if record["status"] not in {"ERROR", "FAILED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
