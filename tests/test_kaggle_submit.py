from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.kaggle_submit import append_submission_log, build_submission_record, find_matching_submission


class FakeSubmission:
    def __init__(
        self,
        *,
        file_name: str,
        description: str,
        date: datetime,
        ref: int,
        status: str,
        public_score: str = "",
        private_score: str = "",
    ) -> None:
        self.file_name = file_name
        self.description = description
        self.date = date
        self.ref = ref
        self.status = status
        self.public_score = public_score
        self.private_score = private_score
        self.team_name = "galaxy2025"
        self.submitted_by = "galaxy2025"
        self.error_description = ""


def test_find_matching_submission_uses_file_message_and_time() -> None:
    submissions = [
        FakeSubmission(
            file_name="submission.zip",
            description="old",
            date=datetime(2026, 6, 8, 1, 0, 0),
            ref=1,
            status="COMPLETE",
        ),
        FakeSubmission(
            file_name="submission.zip",
            description="new baseline",
            date=datetime(2026, 6, 8, 2, 0, 0),
            ref=2,
            status="PENDING",
        ),
    ]

    matched = find_matching_submission(
        submissions=submissions,
        file_name="submission.zip",
        message="new baseline",
        started_after=datetime(2026, 6, 8, 1, 30, 0),
    )

    assert matched is not None
    assert matched.ref == 2


def test_build_submission_record_and_append_log(tmp_path: Path) -> None:
    submission = FakeSubmission(
        file_name="submission.zip",
        description="baseline",
        date=datetime(2026, 6, 8, 2, 0, 0),
        ref=7,
        status="COMPLETE",
        public_score="0.5",
    )
    record = build_submission_record(
        competition="nvidia-nemotron-model-reasoning-challenge",
        file_path=tmp_path / "submission.zip",
        message="baseline",
        response_message="accepted",
        submission=submission,
        submitted_at=datetime(2026, 6, 8, 2, 0, 0, tzinfo=timezone.utc),
    )
    log_path = tmp_path / "history.jsonl"
    append_submission_log(log_path, record)

    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["submission_ref"] == 7
    assert payload["status"] == "COMPLETE"
    assert payload["public_score"] == "0.5"
