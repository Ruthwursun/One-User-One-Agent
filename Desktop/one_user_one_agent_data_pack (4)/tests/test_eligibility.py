"""Focused unit tests for deterministic goal eligibility and runtime rule updates."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from stage_agent.config import UTC
from stage_agent.db import init_db
from stage_agent.eligibility import (
    GoalRegistry,
    evaluate_all_goals,
    evaluate_goal_eligibility,
)


@pytest.fixture
def test_db():
    """Provides a fresh in-memory database with tables initialized."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


def test_disqualified_goals_g10_and_g11(test_db: sqlite3.Connection):
    """Verify that G10 (null event) and G11 (tv_app_open) are rejected deterministically."""
    # Insert dummy profile
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-TEST', 'active', '2026-09-01T00:00:00.000Z', 1, 1, '2026-09-01T00:00:00.000Z');
        """
    )
    test_db.commit()

    res_g10 = evaluate_goal_eligibility("U-TEST", "G10", test_db)
    assert res_g10.eligible is False
    assert res_g10.reason_code == "DISQUALIFIED_NULL_SUCCESS_EVENT"

    res_g11 = evaluate_goal_eligibility("U-TEST", "G11", test_db)
    assert res_g11.eligible is False
    assert res_g11.reason_code == "DISQUALIFIED_UNSUPPORTED_TV_EVENT"


def test_g01_first_play_24h(test_db: sqlite3.Connection):
    """Verify G01: Signed up in last 7 days and has never started a play."""
    # User 1: signed up 3 days ago, no plays -> eligible
    test_db.execute(
        """
        INSERT INTO profiles (user_id, signup_at, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-NEW', '2026-09-28T00:00:00.000Z', 'never', '2026-09-28T00:00:00.000Z', 1, 0, '2026-09-28T00:00:00.000Z');
        """
    )
    # User 2: signed up 10 days ago (> 7 days) -> ineligible
    test_db.execute(
        """
        INSERT INTO profiles (user_id, signup_at, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-OLD', '2026-09-20T00:00:00.000Z', 'never', '2026-09-20T00:00:00.000Z', 1, 0, '2026-09-20T00:00:00.000Z');
        """
    )
    # User 3: signed up 3 days ago but has a play -> ineligible
    test_db.execute(
        """
        INSERT INTO profiles (user_id, signup_at, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-PLAYED', '2026-09-28T00:00:00.000Z', 'never', '2026-09-28T00:00:00.000Z', 1, 0, '2026-09-28T00:00:00.000Z');
        """
    )
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, properties_json, ingested_at)
        VALUES ('e1', 'U-PLAYED', 'play_started', '2026-09-29T10:00:00.000Z', '2026-09-29T10:00:00.000Z', '2026-09-29T10:00:00.000Z', '{}', '2026-09-29T10:00:00.000Z');
        """
    )
    test_db.commit()

    at_time = "2026-10-01T00:00:00.000Z"
    assert evaluate_goal_eligibility("U-NEW", "G01", test_db, at_time).eligible is True
    assert evaluate_goal_eligibility("U-OLD", "G01", test_db, at_time).eligible is False
    assert evaluate_goal_eligibility("U-PLAYED", "G01", test_db, at_time).eligible is False


def test_g02_first_full_episode_3d(test_db: sqlite3.Connection):
    """Verify G02: Has never completed an episode."""
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-INCOMPLETE', 'active', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z'),
               ('U-COMPLETED', 'active', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z');
        """
    )
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, properties_json, ingested_at)
        VALUES ('e2', 'U-COMPLETED', 'episode_completed', '2026-09-25T10:00:00.000Z', '2026-09-25T10:00:00.000Z', '2026-09-25T10:00:00.000Z', '{}', '2026-09-25T10:00:00.000Z');
        """
    )
    test_db.commit()

    at_time = "2026-10-01T00:00:00.000Z"
    assert evaluate_goal_eligibility("U-INCOMPLETE", "G02", test_db, at_time).eligible is True
    assert evaluate_goal_eligibility("U-COMPLETED", "G02", test_db, at_time).eligible is False


def test_g03_and_runtime_rule_correction(test_db: sqlite3.Connection):
    """Verify G03 eligibility (signed up in last 2 days) and dynamic runtime rule correction."""
    test_db.execute(
        """
        INSERT INTO profiles (user_id, signup_at, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-RECENT', '2026-09-30T00:00:00.000Z', 'never', '2026-09-30T00:00:00.000Z', 1, 0, '2026-09-30T00:00:00.000Z');
        """
    )
    test_db.commit()

    registry = GoalRegistry()
    initial_g03 = registry.get_goal("G03")
    assert initial_g03 is not None
    assert initial_g03.success["event"] == "app_open"

    # Evaluate eligibility before correction
    at_time_1 = "2026-10-01T06:00:00.000Z"
    res1 = evaluate_goal_eligibility("U-RECENT", "G03", test_db, at_time_1, registry=registry)
    assert res1.eligible is True

    # Simulate control event at 08:30:00Z: goal_rule_corrected
    control_event = {
        "at": "2026-10-01T08:30:00.000Z",
        "type": "goal_rule_corrected",
        "goal_id": "G03",
        "old_success": {
            "event": "app_open",
            "condition": "on a later IST calendar day within the window",
        },
        "new_success": {
            "event": "play_started",
            "condition": "on a later IST calendar day within the window",
        },
        "applies_to": "every G03 goal that is still open, including goals opened before this correction",
        "note": "Correction from the analytics owner: an app open alone was never meant to count as a return.",
    }

    updated = registry.apply_control_event(control_event)
    assert updated is True

    # Verify updated success condition
    g03_updated = registry.get_goal("G03")
    assert g03_updated.success["event"] == "play_started"
    assert len(registry.correction_history) == 1

    # Eligibility remains True, but success condition has updated
    res2 = evaluate_goal_eligibility("U-RECENT", "G03", test_db, "2026-10-01T09:00:00.000Z", registry=registry)
    assert res2.eligible is True
    assert res2.details["success_rule"]["event"] == "play_started"


def test_g04_g05_g06_subscription_status(test_db: sqlite3.Connection):
    """Verify subscription status goals: G04 (trial_active), G05 (never), G06 (lapsed)."""
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-TRIAL', 'trial_active', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z'),
               ('U-NEVER', 'never', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z'),
               ('U-LAPSED', 'lapsed', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z');
        """
    )
    test_db.commit()

    at_time = "2026-10-01T00:00:00.000Z"
    # G04: trial_to_paid
    assert evaluate_goal_eligibility("U-TRIAL", "G04", test_db, at_time).eligible is True
    assert evaluate_goal_eligibility("U-NEVER", "G04", test_db, at_time).eligible is False

    # G05: first_paid
    assert evaluate_goal_eligibility("U-NEVER", "G05", test_db, at_time).eligible is True
    assert evaluate_goal_eligibility("U-TRIAL", "G05", test_db, at_time).eligible is False

    # G06: reactivate_lapsed
    assert evaluate_goal_eligibility("U-LAPSED", "G06", test_db, at_time).eligible is True
    assert evaluate_goal_eligibility("U-NEVER", "G06", test_db, at_time).eligible is False


def test_g07_resume_series_3d(test_db: sqlite3.Connection):
    """Verify G07: Started a show/micro_drama, has unwatched episodes left, not played for >= 3 days."""
    # Insert catalogue and episode records
    test_db.execute(
        """
        INSERT INTO catalogue (title_id, content_slug, title, dialect, language, content_type, genres, available_from, is_premium, ingested_at)
        VALUES ('T-SHOW', 'slug-thriller', 'Thriller Show', 'd1', 'hin', 'show', 'Thriller', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z');
        """
    )
    test_db.execute(
        """
        INSERT INTO episodes (episode_id, content_slug, season, episode, minutes, ingested_at)
        VALUES ('EP-1', 'slug-thriller', 1, 1, 20.0, '2026-01-01T00:00:00.000Z'),
               ('EP-2', 'slug-thriller', 1, 2, 22.0, '2026-01-01T00:00:00.000Z');
        """
    )

    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-SERIES', 'active', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z');
        """
    )

    # User played Ep 1 four days ago (2026-09-27)
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, title_id, episode_id, properties_json, ingested_at)
        VALUES ('e-p1', 'U-SERIES', 'play_started', '2026-09-27T00:00:00.000Z', '2026-09-27T00:00:00.000Z', '2026-09-27T00:00:00.000Z', 'T-SHOW', 'EP-1', '{"title_id": "T-SHOW"}', '2026-09-27T00:00:00.000Z'),
               ('e-c1', 'U-SERIES', 'episode_completed', '2026-09-27T00:30:00.000Z', '2026-09-27T00:30:00.000Z', '2026-09-27T00:30:00.000Z', 'T-SHOW', 'EP-1', '{"title_id": "T-SHOW", "episode_id": "EP-1"}', '2026-09-27T00:30:00.000Z');
        """
    )
    test_db.commit()

    at_time = "2026-10-01T00:00:00.000Z"  # 4 days later -> eligible!
    res = evaluate_goal_eligibility("U-SERIES", "G07", test_db, at_time)
    assert res.eligible is True
    assert res.reason_code == "ELIGIBLE"

    # But if user played 1 day ago (2026-09-30) -> ineligible (< 3 days)
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, title_id, episode_id, properties_json, ingested_at)
        VALUES ('e-p2', 'U-SERIES', 'play_started', '2026-09-30T10:00:00.000Z', '2026-09-30T10:00:00.000Z', '2026-09-30T10:00:00.000Z', 'T-SHOW', 'EP-2', '{"title_id": "T-SHOW"}', '2026-09-30T10:00:00.000Z');
        """
    )
    test_db.commit()
    res_recent = evaluate_goal_eligibility("U-SERIES", "G07", test_db, at_time)
    assert res_recent.eligible is False


def test_g09_weekly_habit_14d(test_db: sqlite3.Connection):
    """Verify G09: Active subscriber with plays on at most 1 day in the last 14 days."""
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-HABIT-1', 'active', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z'),
               ('U-HABIT-MULTI', 'active', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z');
        """
    )
    # U-HABIT-1 played on only 1 distinct day in last 14 days
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, properties_json, ingested_at)
        VALUES ('eh1', 'U-HABIT-1', 'play_started', '2026-09-20T10:00:00.000Z', '2026-09-20T10:00:00.000Z', '2026-09-20T10:00:00.000Z', '{}', '2026-09-20T10:00:00.000Z'),
               ('eh2', 'U-HABIT-1', 'play_started', '2026-09-20T14:00:00.000Z', '2026-09-20T14:00:00.000Z', '2026-09-20T14:00:00.000Z', '{}', '2026-09-20T14:00:00.000Z');
        """
    )
    # U-HABIT-MULTI played on 3 distinct days
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, properties_json, ingested_at)
        VALUES ('em1', 'U-HABIT-MULTI', 'play_started', '2026-09-20T10:00:00.000Z', '2026-09-20T10:00:00.000Z', '2026-09-20T10:00:00.000Z', '{}', '2026-09-20T10:00:00.000Z'),
               ('em2', 'U-HABIT-MULTI', 'play_started', '2026-09-22T10:00:00.000Z', '2026-09-22T10:00:00.000Z', '2026-09-22T10:00:00.000Z', '{}', '2026-09-22T10:00:00.000Z'),
               ('em3', 'U-HABIT-MULTI', 'play_started', '2026-09-25T10:00:00.000Z', '2026-09-25T10:00:00.000Z', '2026-09-25T10:00:00.000Z', '{}', '2026-09-25T10:00:00.000Z');
        """
    )
    test_db.commit()

    at_time = "2026-10-01T00:00:00.000Z"
    assert evaluate_goal_eligibility("U-HABIT-1", "G09", test_db, at_time).eligible is True
    assert evaluate_goal_eligibility("U-HABIT-MULTI", "G09", test_db, at_time).eligible is False


def test_g12_return_after_absence_7d(test_db: sqlite3.Connection):
    """Verify G12: No app_open in the 14 days before the trigger."""
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-ABSENT', 'never', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z'),
               ('U-VISITOR', 'never', '2026-09-01T00:00:00.000Z', 1, 0, '2026-09-01T00:00:00.000Z');
        """
    )
    # U-VISITOR opened app 5 days ago (2026-09-26)
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, properties_json, ingested_at)
        VALUES ('ev-open', 'U-VISITOR', 'app_open', '2026-09-26T10:00:00.000Z', '2026-09-26T10:00:00.000Z', '2026-09-26T10:00:00.000Z', '{}', '2026-09-26T10:00:00.000Z');
        """
    )
    test_db.commit()

    at_time = "2026-10-01T00:00:00.000Z"
    assert evaluate_goal_eligibility("U-ABSENT", "G12", test_db, at_time).eligible is True
    assert evaluate_goal_eligibility("U-VISITOR", "G12", test_db, at_time).eligible is False
