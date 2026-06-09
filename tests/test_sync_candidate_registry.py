from __future__ import annotations

from src.check_submissions import SubmissionRow
from src.sync_candidate_registry import sync_registry_scores


def test_sync_registry_scores_updates_status_score_and_date() -> None:
    registry = [
        {
            "id": "complete_candidate",
            "submission_ref": 1,
            "public_score": None,
        },
        {
            "id": "pending_candidate",
            "submission_ref": 2,
            "public_score": 0.9,
        },
    ]
    submissions = [
        SubmissionRow(
            ref=1,
            date="2026-06-08 16:00:00",
            status="COMPLETE",
            public_score=0.86,
            private_score=0.85,
            description="complete",
        ),
        SubmissionRow(
            ref=2,
            date="2026-06-08 17:00:00",
            status="PENDING",
            public_score=None,
            private_score=None,
            description="pending",
        ),
    ]

    synced = sync_registry_scores(registry, submissions)

    assert synced[0]["submission_status"] == "COMPLETE"
    assert synced[0]["public_score"] == 0.86
    assert synced[0]["private_score"] == 0.85
    assert synced[0]["submission_date"] == "2026-06-08 16:00:00"
    assert synced[1]["submission_status"] == "PENDING"
    assert synced[1]["public_score"] is None
    assert synced[1]["private_score"] is None


def test_sync_registry_scores_matches_submission_message_when_ref_missing() -> None:
    registry = [
        {
            "id": "message_candidate",
            "submission_message": "public candidate output",
            "submission_status": "PLANNED",
            "public_score": None,
        },
    ]
    submissions = [
        SubmissionRow(
            ref=10,
            date="2026-06-11 01:00:00",
            status="PENDING",
            public_score=None,
            private_score=None,
            description="public candidate output",
        ),
        SubmissionRow(
            ref=11,
            date="2026-06-11 02:00:00",
            status="COMPLETE",
            public_score=0.87,
            private_score=0.86,
            description="public candidate output",
        ),
    ]

    synced = sync_registry_scores(registry, submissions)

    assert synced[0]["submission_ref"] == 11
    assert synced[0]["submission_status"] == "COMPLETE"
    assert synced[0]["public_score"] == 0.87
    assert synced[0]["private_score"] == 0.86
    assert synced[0]["submission_date"] == "2026-06-11 02:00:00"
