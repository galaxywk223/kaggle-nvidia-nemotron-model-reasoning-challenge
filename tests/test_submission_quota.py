from __future__ import annotations

from datetime import datetime, timezone

from src.check_submissions import SubmissionRow
from src.submission_quota import SubmissionQuota, quota_status


class FakeApi:
    def competitions_list(self, search: str):  # noqa: ANN001
        class Competition:
            ref = "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge"
            url = ref
            maxDailySubmissions = 5

        class Response:
            competitions = [Competition()]

        return Response()


def test_quota_status_counts_current_utc_day() -> None:
    rows = [
        SubmissionRow(
            ref=1,
            date="2026-06-08 01:00:00",
            status="COMPLETE",
            public_score=0.1,
            private_score=0.1,
            description="same day",
        ),
        SubmissionRow(
            ref=2,
            date="2026-06-07 23:00:00",
            status="COMPLETE",
            public_score=0.1,
            private_score=0.1,
            description="previous day",
        ),
    ]

    status = quota_status(
        rows=rows,
        now=datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc),
        api=FakeApi(),
    )

    assert status == SubmissionQuota(
        competition="nvidia-nemotron-model-reasoning-challenge",
        utc_date="2026-06-08",
        max_daily_submissions=5,
        submissions_today=1,
    )
    assert status.remaining == 4
    assert not status.exhausted


def test_quota_exhausted_when_today_matches_daily_limit() -> None:
    rows = [
        SubmissionRow(
            ref=idx,
            date=f"2026-06-08 0{idx}:00:00",
            status="PENDING",
            public_score=None,
            private_score=None,
            description="candidate",
        )
        for idx in range(5)
    ]

    status = quota_status(
        rows=rows,
        now=datetime(2026, 6, 8, 17, 0, 0, tzinfo=timezone.utc),
        api=FakeApi(),
    )

    assert status.remaining == 0
    assert status.exhausted
