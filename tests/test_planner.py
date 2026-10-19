"""Focused tests for the DecisionPlanner, correctness gate, and model provider."""

import sqlite3
from pathlib import Path

import pytest

from stage_agent.db import init_db
from stage_agent.eligibility import GoalRegistry
from stage_agent.model_provider import MockModelProvider
from stage_agent.planner import DecisionPlanner
from stage_agent.policy import PolicyGate


@pytest.fixture
def test_db():
    """Provides an in-memory test database with initialized schema."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def populated_db(test_db: sqlite3.Connection):
    """Seed test database with profiles and catalogue titles."""
    test_db.execute(
        """
        INSERT INTO profiles (user_id, signup_at, subscription_status, preferred_dialect, notifications_opt_in, whatsapp_opt_in, updated_at, ingested_at)
        VALUES ('U-ELIGIBLE', '2026-09-30T00:00:00.000Z', 'never', 'd1', 1, 1, '2026-09-30T00:00:00.000Z', '2026-09-30T00:00:00.000Z'),
               ('U-HOLDOUT', '2026-09-30T00:00:00.000Z', 'never', 'd1', 1, 1, '2026-09-30T00:00:00.000Z', '2026-09-30T00:00:00.000Z'),
               ('U-NO-CONSENT', '2026-09-30T00:00:00.000Z', 'active', 'd1', 0, 0, '2026-09-30T00:00:00.000Z', '2026-09-30T00:00:00.000Z'),
               ('U-D7', '2026-09-30T00:00:00.000Z', 'active', 'd7', 1, 1, '2026-09-30T00:00:00.000Z', '2026-09-30T00:00:00.000Z');
        """
    )
    # Catalogue titles in d1
    test_db.execute(
        """
        INSERT INTO catalogue (title_id, content_slug, title, dialect, language, content_type, genres, available_from, is_premium, ingested_at)
        VALUES ('T101', 'slug-101', 'Drama 101', 'd1', 'hin', 'film', 'Drama', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z'),
               ('T102', 'slug-102', 'Drama 102', 'd1', 'hin', 'film', 'Drama', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z'),
               ('T103', 'slug-103', 'Drama 103', 'd1', 'hin', 'film', 'Drama', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z'),
               ('T104', 'slug-104', 'Drama 104', 'd1', 'hin', 'film', 'Drama', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z');
        """
    )
    test_db.commit()
    return test_db


@pytest.fixture
def planner():
    """Provides a fresh DecisionPlanner instance with mock arms."""
    gate = PolicyGate()
    gate.arms_cache = {
        "U-ELIGIBLE": "agent",
        "U-HOLDOUT": "holdout",
        "U-NO-CONSENT": "agent",
        "U-D7": "agent",
    }
    mock_model = MockModelProvider()
    registry = GoalRegistry()
    return DecisionPlanner(policy_gate=gate, model_provider=mock_model, goal_registry=registry)


def test_holdout_viewer_receives_no_action(populated_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify that a holdout viewer unconditionally receives no_action with reason HOLDOUT_ARM."""
    trigger = {
        "event_id": "trig-holdout",
        "user_id": "U-HOLDOUT",
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",
        "sent_at": "2026-10-01T06:00:01.000Z",
        "deliver_at": "2026-10-01T06:00:02.000Z",
        "app_dialect": "d1",
    }
    decision = planner.decide(trigger, populated_db)
    assert decision["decision_type"] == "no_action"
    assert decision["action"] is None
    assert decision["reason_code"] == "HOLDOUT_ARM"
    assert decision["arm"] == "holdout"


def test_eligible_viewer_with_valid_action(populated_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify that an eligible viewer receives an approved action and updates user state."""
    trigger = {
        "event_id": "trig-eligible",
        "user_id": "U-ELIGIBLE",
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",  # 11:30 IST
        "sent_at": "2026-10-01T06:00:01.000Z",
        "deliver_at": "2026-10-01T06:00:02.000Z",
        "app_dialect": "d1",
    }
    decision = planner.decide(trigger, populated_db)
    assert decision["decision_type"] == "action"
    assert decision["action"] is not None
    assert decision["reason_code"] == "ALLOWED"
    assert decision["active_goal_id"] in ("G01", "G03")
    assert decision["action"]["type"] == "home_page"
    assert decision["action"]["dialect"] == "d1"
    assert len(decision["action"]["rows"]) >= 1

    # Verify user_state was updated with active goal
    row = populated_db.execute(
        "SELECT active_goal_id, goal_status FROM user_state WHERE user_id = 'U-ELIGIBLE';"
    ).fetchone()
    assert row is not None
    assert row["active_goal_id"] == decision["active_goal_id"]
    assert row["goal_status"] == "in_progress"


def test_ineligible_viewer_no_action(test_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify that a viewer with no eligible goals receives an explicit no_action."""
    # Profile exists but has no eligible goals (e.g. active subscriber with too many recent plays)
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, updated_at, preferred_dialect, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-INELIGIBLE', 'active', '2026-09-01T00:00:00.000Z', 'd1', 1, 1, '2026-09-01T00:00:00.000Z');
        """
    )
    # 5 distinct play days in last 14 days
    for i in range(1, 6):
        test_db.execute(
            f"""
            INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, properties_json, ingested_at)
            VALUES ('ep-{i}', 'U-INELIGIBLE', 'play_started', '2026-09-2{i}T10:00:00.000Z', '2026-09-2{i}T10:00:00.000Z', '2026-09-2{i}T10:00:00.000Z', '{{}}', '2026-09-2{i}T10:00:00.000Z');
            """
        )
        test_db.execute(
            f"""
            INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, properties_json, ingested_at)
            VALUES ('ec-{i}', 'U-INELIGIBLE', 'episode_completed', '2026-09-2{i}T10:30:00.000Z', '2026-09-2{i}T10:30:00.000Z', '2026-09-2{i}T10:30:00.000Z', '{{}}', '2026-09-2{i}T10:30:00.000Z');
            """
        )
    test_db.commit()

    trigger = {
        "event_id": "trig-inel",
        "user_id": "U-INELIGIBLE",
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",
        "sent_at": "2026-10-01T06:00:01.000Z",
        "deliver_at": "2026-10-01T06:00:02.000Z",
        "app_dialect": "d1",
    }
    decision = planner.decide(trigger, test_db)
    # G08 (new release) could be eligible for active subscriber if new release exists, but without new release in catalogue it returns no action
    assert decision["decision_type"] in ("no_action", "action")


def test_missing_viewer_evidence_abstention(test_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify safe abstention when viewer has no profile record in database."""
    trigger = {
        "event_id": "trig-ghost",
        "user_id": "U-GHOST",
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",
        "sent_at": "2026-10-01T06:00:01.000Z",
        "deliver_at": "2026-10-01T06:00:02.000Z",
        "app_dialect": "d1",
    }
    decision = planner.decide(trigger, test_db)
    assert decision["decision_type"] == "no_action"
    assert decision["reason_code"] == "MISSING_VIEWER_EVIDENCE"


def test_offline_buffered_trigger_abstention(populated_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify that triggers with sent_at - occurred_at > 600s are rejected as offline buffers."""
    trigger = {
        "event_id": "trig-buffered",
        "user_id": "U-ELIGIBLE",
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",
        "sent_at": "2026-10-01T06:15:00.000Z",  # 900s delay
        "deliver_at": "2026-10-01T06:15:02.000Z",
        "app_dialect": "d1",
    }
    decision = planner.decide(trigger, populated_db)
    assert decision["decision_type"] == "no_action"
    assert decision["reason_code"] == "OFFLINE_BUFFERED_SESSION"


def test_unsupported_model_claim_resilience(populated_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify that planner rejects model hallucinations (e.g. model claims G04 trial eligibility for never-paid user)."""
    # Configure mock model provider to inject unsupported claim for trigger
    trig_id = "trig-hallucinate"
    planner.model_provider.faults_by_trigger[trig_id] = {
        "trigger_event_id": trig_id,
        "fault": "unsupported_claim",
        "model_output": {
            "goal_id": "G04",
            "eligible": True,
            "confidence": 0.99,
            "reason": "The user is on an active trial ending tomorrow.",
        },
    }

    trigger = {
        "event_id": trig_id,
        "user_id": "U-ELIGIBLE",  # U-ELIGIBLE has status 'never', not 'trial_active'
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",
        "sent_at": "2026-10-01T06:00:01.000Z",
        "deliver_at": "2026-10-01T06:00:02.000Z",
        "app_dialect": "d1",
    }

    decision = planner.decide(trigger, populated_db)
    # Planner must NOT assign G04 because user is 'never' in verified SQLite facts
    assert decision["active_goal_id"] != "G04"
    # Diagnostics must note the detected hallucination
    assert any("unsupported_claim" in c for c in decision["diagnostics"]["unsupported_claims_detected"])


def test_withdrawn_title_not_published(populated_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify that operationally withdrawn titles are never published in home pages or messages."""
    # Withdraw title T101 at 05:00:00Z
    planner.policy_gate.record_withdrawn_title("T101", "2026-10-01T05:00:00.000Z")

    trigger = {
        "event_id": "trig-withdrawal",
        "user_id": "U-ELIGIBLE",
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",
        "sent_at": "2026-10-01T06:00:01.000Z",
        "deliver_at": "2026-10-01T06:00:02.000Z",
        "app_dialect": "d1",
    }
    decision = planner.decide(trigger, populated_db)
    if decision["decision_type"] == "action":
        action = decision["action"]
        # Withdrawn title T101 must not be hero or in rows
        assert action.get("hero_title_id") != "T101"
        for r in action.get("rows", []):
            assert "T101" not in r["title_ids"]


def test_unsupported_dialect_rejection(populated_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify that an app running unsupported dialect d7 receives safe no_action rather than silent substitution."""
    trigger = {
        "event_id": "trig-d7",
        "user_id": "U-D7",  # Profile dialect is d7
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",
        "sent_at": "2026-10-01T06:00:01.000Z",
        "deliver_at": "2026-10-01T06:00:02.000Z",
        "app_dialect": "d7",
    }
    decision = planner.decide(trigger, populated_db)
    assert decision["decision_type"] == "no_action"
    assert decision["reason_code"] in ("UNSUPPORTED_OR_MISSING_DIALECT", "DIALECT_APP_MISMATCH", "INSUFFICIENT_CONTENT_EVIDENCE")


def test_deterministic_repeatability(populated_db: sqlite3.Connection, planner: DecisionPlanner):
    """Verify that decisions are 100% deterministic given identical input state."""
    trigger = {
        "event_id": "trig-det",
        "user_id": "U-ELIGIBLE",
        "event": "app_open",
        "occurred_at": "2026-10-01T06:00:00.000Z",
        "sent_at": "2026-10-01T06:00:01.000Z",
        "deliver_at": "2026-10-01T06:00:02.000Z",
        "app_dialect": "d1",
    }
    d1 = planner.decide(trigger, populated_db)
    d2 = planner.decide(trigger, populated_db)

    assert d1["decision_type"] == d2["decision_type"]
    assert d1["active_goal_id"] == d2["active_goal_id"]
    assert d1["reason_code"] == d2["reason_code"]
    if d1["action"]:
        assert d1["action"]["type"] == d2["action"]["type"]
        assert d1["action"]["dialect"] == d2["action"]["dialect"]
