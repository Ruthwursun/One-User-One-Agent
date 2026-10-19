"""Comprehensive automated test suite for sequential replay runner and simulator integration."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stage_agent.config import parse_iso_datetime
from stage_agent.db import get_db_connection, init_db
from stage_agent.eligibility import get_default_registry
from stage_agent.model_provider import (
    MockModelProvider,
    MockModelTimeoutError,
    ModelUnavailableError,
)
from stage_agent.planner import DecisionPlanner
from stage_agent.policy import PolicyGate
from stage_agent.replay import ReplayRunner, StreamItem


@pytest.fixture
def replay_fixture_db(tmp_path: Path):
    """Create a temporary initialized SQLite database with seed catalogue and profiles."""
    db_file = tmp_path / "test_replay.db"
    conn = init_db(db_file)

    with conn:
        # Seed Catalogue: 6 titles in d1, available throughout 2026
        for i in range(1, 10):
            tid = f"T{i:04d}"
            conn.execute(
                """
                INSERT INTO catalogue (
                    title_id, content_slug, title, dialect, language, content_type,
                    genres, release_date, available_from, available_until, is_premium, description, ingested_at
                ) VALUES (?, ?, ?, 'd1', 'hin', 'film', 'Drama', '2026-01-01',
                          '2026-01-01T00:00:00.000Z', NULL, 0, 'Test title', '2026-09-30T00:00:00.000Z');
                """,
                (tid, f"slug-{tid}", f"Movie {tid}"),
            )

        # Seed Profiles:
        # U-001: agent arm, active consent, dialect d1, subscription never
        conn.execute(
            """
            INSERT INTO profiles (
                user_id, signup_at, subscription_status, preferred_dialect,
                notifications_opt_in, whatsapp_opt_in, updated_at, ingested_at
            ) VALUES ('U-001', '2026-09-28T00:00:00.000Z', 'never', 'd1', 1, 1, '2026-09-30T00:00:00.000Z', '2026-09-30T00:00:00.000Z');
            """
        )
        # U-002: holdout arm
        conn.execute(
            """
            INSERT INTO profiles (
                user_id, signup_at, subscription_status, preferred_dialect,
                notifications_opt_in, whatsapp_opt_in, updated_at, ingested_at
            ) VALUES ('U-002', '2026-09-28T00:00:00.000Z', 'never', 'd1', 1, 1, '2026-09-30T00:00:00.000Z', '2026-09-30T00:00:00.000Z');
            """
        )
        # U-003: agent arm, trial_active
        conn.execute(
            """
            INSERT INTO profiles (
                user_id, signup_at, subscription_status, preferred_dialect,
                notifications_opt_in, whatsapp_opt_in, updated_at, ingested_at
            ) VALUES ('U-003', '2026-09-28T00:00:00.000Z', 'trial_active', 'd1', 1, 1, '2026-09-30T00:00:00.000Z', '2026-09-30T00:00:00.000Z');
            """
        )

    return conn, db_file


def test_duplicate_and_late_triggers_idempotency(replay_fixture_db, tmp_path: Path):
    """Verify duplicate triggers are suppressed idempotently and late/offline buffers are abstained."""
    conn, db_file = replay_fixture_db
    arms_map = {"U-001": "agent"}

    gate = PolicyGate(arms_path=tmp_path / "arms.csv")
    planner = DecisionPlanner(policy_gate=gate)
    seen_ids: set[str] = set()
    sent_cache: set[tuple[str, str]] = set()

    # Trigger 1: live session at 10:00 IST (04:30 UTC)
    t1 = {
        "event_id": "trig-dup-01",
        "user_id": "U-001",
        "event": "app_open",
        "occurred_at": "2026-10-01T04:30:00.000Z",
        "sent_at": "2026-10-01T04:30:00.000Z",
        "deliver_at": "2026-10-01T04:30:01.000Z",
        "session_id": "s-01",
        "app_dialect": "d1",
    }

    # First delivery
    dec1 = planner.decide(t1, conn, arms_map, sent_cache)
    seen_ids.add("trig-dup-01")
    assert dec1["decision_type"] == "action"
    assert dec1["action"] is not None

    # Duplicate delivery of same event_id
    runner = ReplayRunner(db_path=db_file, out_dir=tmp_path / "out")
    runner.conn = conn
    runner.arms_map = arms_map
    runner.planner = planner
    runner.seen_trigger_event_ids = seen_ids

    item_dup = StreamItem(
        kind="trigger",
        timestamp_str="2026-10-01T04:35:00.000Z",
        timestamp_dt=datetime(2026, 10, 1, 4, 35, tzinfo=timezone.utc),
        data=t1,
        priority=2,
        index=1,
    )
    dec_dup, act_dup = runner.handle_trigger(item_dup)

    assert dec_dup["decision_type"] == "no_action"
    assert dec_dup["reason_code"] == "DUPLICATE_TRIGGER"
    assert act_dup is None  # Zero duplicate action published!

    # Late / offline-buffered trigger (>600s delay)
    t_late = {
        "event_id": "trig-late-01",
        "user_id": "U-001",
        "event": "app_open",
        "occurred_at": "2026-09-30T10:00:00.000Z",
        "sent_at": "2026-10-01T04:30:00.000Z",  # Delayed > 18 hours
        "deliver_at": "2026-10-01T04:30:01.000Z",
        "session_id": "s-late",
        "app_dialect": "d1",
    }
    dec_late = planner.decide(t_late, conn, arms_map, sent_cache)
    assert dec_late["decision_type"] == "no_action"
    assert dec_late["reason_code"] == "OFFLINE_BUFFERED_SESSION"
    assert dec_late["action"] is None


def test_goal_correction_applied_at_correct_time(replay_fixture_db, tmp_path: Path):
    """Verify G03 goal correction is applied at exactly 08:30:00Z."""
    conn, db_file = replay_fixture_db
    runner = ReplayRunner(db_path=db_file, out_dir=tmp_path / "out")
    runner.conn = conn

    # Initial G03 success is app_open
    g03_init = runner.goal_registry.get_goal("G03")
    assert g03_init.success.get("event") == "app_open"

    # Control event at 08:30:00Z
    ctrl_item = StreamItem(
        kind="control_event",
        timestamp_str="2026-10-01T08:30:00.000Z",
        timestamp_dt=datetime(2026, 10, 1, 8, 30, tzinfo=timezone.utc),
        data={
            "at": "2026-10-01T08:30:00.000Z",
            "type": "goal_rule_corrected",
            "goal_id": "G03",
            "old_success": {"event": "app_open"},
            "new_success": {"event": "play_started", "condition": "on a later IST calendar day within the window"},
        },
        priority=0,
        index=0,
    )
    runner.handle_control_event(ctrl_item)

    # Updated G03 success is play_started
    g03_updated = runner.goal_registry.get_goal("G03")
    assert g03_updated.success.get("event") == "play_started"
    assert "play_started" in g03_updated.success.get("condition", "") or "later IST calendar day" in g03_updated.success.get("condition", "")


def test_injected_faults_and_outages(replay_fixture_db, tmp_path: Path):
    """Verify injected timeout, malformed output, unsupported claims, and HTTP 503 outages are handled safely."""
    conn, db_file = replay_fixture_db
    arms_map = {"U-001": "agent"}

    gate = PolicyGate(arms_path=tmp_path / "arms.csv")
    provider = MockModelProvider()
    planner = DecisionPlanner(policy_gate=gate, model_provider=provider)

    # 1. Unsupported claim fault (Model asserts G04 eligibility for user on 'never' plan)
    provider.faults_by_trigger["trig-claim-01"] = {
        "trigger_event_id": "trig-claim-01",
        "fault": "unsupported_claim",
        "model_output": {"goal_id": "G04", "eligible": True, "confidence": 0.99, "reason": "Hallucinated trial"},
    }
    t_claim = {
        "event_id": "trig-claim-01",
        "user_id": "U-001",  # Subscription status is 'never', ineligible for G04!
        "event": "app_open",
        "occurred_at": "2026-10-01T04:30:00.000Z",
        "sent_at": "2026-10-01T04:30:00.000Z",
        "deliver_at": "2026-10-01T04:30:01.000Z",
        "app_dialect": "d1",
    }
    dec_claim = planner.decide(t_claim, conn, arms_map)
    # Planner MUST reject G04 and fall back to verified eligible goal (e.g. G01 or G05)
    assert dec_claim["active_goal_id"] != "G04"
    diag = dec_claim.get("diagnostics", {})
    claims_detected = diag.get("unsupported_claims_detected", [])
    assert any("rejected_unsupported_claim_G04" in c for c in claims_detected)

    # 2. Timeout fault
    provider.faults_by_trigger["trig-timeout-01"] = {
        "trigger_event_id": "trig-timeout-01",
        "fault": "timeout",
        "seconds": 45,
    }
    t_timeout = {
        "event_id": "trig-timeout-01",
        "user_id": "U-001",
        "event": "app_open",
        "occurred_at": "2026-10-01T04:30:00.000Z",
        "sent_at": "2026-10-01T04:30:00.000Z",
        "deliver_at": "2026-10-01T04:30:01.000Z",
        "app_dialect": "d1",
    }
    dec_timeout = planner.decide(t_timeout, conn, arms_map)
    # Does not crash! Falls back safely.
    assert dec_timeout["decision_type"] == "action"
    assert "model_timeout_fallback" in dec_timeout["diagnostics"]["unsupported_claims_detected"]

    # 3. Model outage (HTTP 503) window
    provider.add_outage_window("2026-10-01T10:35:00.000Z", "2026-10-01T10:42:00.000Z")
    t_outage = {
        "event_id": "trig-outage-01",
        "user_id": "U-001",
        "event": "app_open",
        "occurred_at": "2026-10-01T10:36:00.000Z",
        "sent_at": "2026-10-01T10:36:00.000Z",
        "deliver_at": "2026-10-01T10:36:01.000Z",
        "app_dialect": "d1",
    }
    dec_outage = planner.decide(t_outage, conn, arms_map)
    assert dec_outage["decision_type"] == "action"
    assert "model_outage_503_fallback" in dec_outage["diagnostics"]["unsupported_claims_detected"]


def test_title_withdrawal_after_earlier_decision(replay_fixture_db, tmp_path: Path):
    """Verify title withdrawal removes title from catalogue availability and policy gates."""
    conn, db_file = replay_fixture_db
    runner = ReplayRunner(db_path=db_file, out_dir=tmp_path / "out")
    runner.conn = conn

    # Withdraw T0001 at 07:00:00Z
    withdraw_ctrl = StreamItem(
        kind="control_event",
        timestamp_str="2026-10-01T07:00:00.000Z",
        timestamp_dt=datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc),
        data={
            "at": "2026-10-01T07:00:00.000Z",
            "type": "title_withdrawn",
            "title_id": "T0001",
            "reason": "licence ended early",
        },
        priority=0,
        index=0,
    )
    runner.handle_control_event(withdraw_ctrl)

    # Database updated
    row = conn.execute("SELECT available_until FROM catalogue WHERE title_id = 'T0001';").fetchone()
    assert row["available_until"] == "2026-10-01T07:00:00.000Z"

    # Policy gate verifies rejection
    dec = runner.policy_gate.check_dialect_and_availability(
        user_id="U-001",
        target_dialect="d1",
        title_id="T0001",
        at_time="2026-10-01T08:00:00.000Z",
        conn=conn,
    )
    assert not dec.allowed
    assert dec.reason_code == "TITLE_WITHDRAWN"


def test_purchase_events_and_subscription_transitions(replay_fixture_db, tmp_path: Path):
    """Verify organic purchase events transition subscription state appropriately."""
    conn, db_file = replay_fixture_db
    runner = ReplayRunner(db_path=db_file, out_dir=tmp_path / "out")
    runner.conn = conn

    # 1. Paid purchase transitions 'never' to 'active'
    live_paid = StreamItem(
        kind="live_event",
        timestamp_str="2026-10-01T03:00:00.000Z",
        timestamp_dt=datetime(2026, 10, 1, 3, 0, tzinfo=timezone.utc),
        data={
            "event_id": "live-pur-01",
            "user_id": "U-001",
            "event": "purchase_completed",
            "received_at": "2026-10-01T03:00:00.000Z",
            "occurred_at": "2026-10-01T03:00:00.000Z",
            "sent_at": "2026-10-01T03:00:00.000Z",
            "properties": {"plan_id": "P_MONTH", "price_inr": 149, "is_trial": False},
        },
        priority=1,
        index=0,
    )
    runner.handle_live_event(live_paid)

    p_row = conn.execute("SELECT subscription_status FROM profiles WHERE user_id = 'U-001';").fetchone()
    assert p_row["subscription_status"] == "active"

    # 2. Trial purchase transitions to 'trial_active', not 'active'
    live_trial = StreamItem(
        kind="live_event",
        timestamp_str="2026-10-01T03:05:00.000Z",
        timestamp_dt=datetime(2026, 10, 1, 3, 5, tzinfo=timezone.utc),
        data={
            "event_id": "live-pur-02",
            "user_id": "U-002",
            "event": "purchase_completed",
            "received_at": "2026-10-01T03:05:00.000Z",
            "occurred_at": "2026-10-01T03:05:00.000Z",
            "sent_at": "2026-10-01T03:05:00.000Z",
            "properties": {"plan_id": "P_TRIAL", "price_inr": 1, "is_trial": True},
        },
        priority=1,
        index=1,
    )
    runner.handle_live_event(live_trial)

    p_row2 = conn.execute("SELECT subscription_status FROM profiles WHERE user_id = 'U-002';").fetchone()
    assert p_row2["subscription_status"] == "trial_active"


def test_holdout_users_zero_actions_guarantee(replay_fixture_db, tmp_path: Path):
    """Verify holdout arm viewers never receive any action."""
    conn, db_file = replay_fixture_db
    arms_map = {"U-002": "holdout"}

    gate = PolicyGate(arms_path=tmp_path / "arms.csv")
    planner = DecisionPlanner(policy_gate=gate)

    t_holdout = {
        "event_id": "trig-holdout-01",
        "user_id": "U-002",
        "event": "app_open",
        "occurred_at": "2026-10-01T04:30:00.000Z",
        "sent_at": "2026-10-01T04:30:00.000Z",
        "deliver_at": "2026-10-01T04:30:01.000Z",
        "app_dialect": "d1",
    }
    dec = planner.decide(t_holdout, conn, arms_map)
    assert dec["decision_type"] == "no_action"
    assert dec["reason_code"] == "HOLDOUT_ARM"
    assert dec["action"] is None


def test_output_schema_validity(replay_fixture_db, tmp_path: Path):
    """Verify output artifact schemas adhere to expected JSONL keys and structures."""
    conn, db_file = replay_fixture_db
    out_dir = tmp_path / "sample_run"

    runner = ReplayRunner(db_path=db_file, out_dir=out_dir)
    runner.conn = conn
    runner.arms_map = {"U-001": "agent", "U-002": "holdout"}

    # Run small synthetic stream
    stream = [
        StreamItem(
            kind="trigger",
            timestamp_str="2026-10-01T04:30:00.000Z",
            timestamp_dt=datetime(2026, 10, 1, 4, 30, tzinfo=timezone.utc),
            data={
                "event_id": "trig-schema-01",
                "user_id": "U-001",
                "event": "app_open",
                "occurred_at": "2026-10-01T04:30:00.000Z",
                "sent_at": "2026-10-01T04:30:00.000Z",
                "deliver_at": "2026-10-01T04:30:00.000Z",
                "app_dialect": "d1",
            },
            priority=2,
            index=0,
        ),
        StreamItem(
            kind="trigger",
            timestamp_str="2026-10-01T04:31:00.000Z",
            timestamp_dt=datetime(2026, 10, 1, 4, 31, tzinfo=timezone.utc),
            data={
                "event_id": "trig-schema-02",
                "user_id": "U-002",
                "event": "app_open",
                "occurred_at": "2026-10-01T04:31:00.000Z",
                "sent_at": "2026-10-01T04:31:00.000Z",
                "deliver_at": "2026-10-01T04:31:00.000Z",
                "app_dialect": "d1",
            },
            priority=2,
            index=1,
        ),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = []
    actions = []
    for item in stream:
        dec, act = runner.handle_trigger(item)
        decisions.append(dec)
        if act:
            actions.append(act)

    with open(out_dir / "decisions.jsonl", "w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    with open(out_dir / "actions.jsonl", "w", encoding="utf-8") as f:
        for a in actions:
            f.write(json.dumps(a) + "\n")
    runner._export_viewer_state(out_dir / "viewer_state.jsonl")

    # Validate decisions.jsonl
    with open(out_dir / "decisions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            assert "decision_id" in d
            assert "user_id" in d
            assert "trigger_event_id" in d
            assert "decision_type" in d
            assert "reason_code" in d

    # Validate actions.jsonl
    with open(out_dir / "actions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            assert "action_id" in a
            assert "decision_id" in a
            assert "user_id" in a
            assert "type" in a
            assert "published_at" in a

    # Validate viewer_state.jsonl
    with open(out_dir / "viewer_state.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            v = json.loads(line)
            assert "user_id" in v
            assert "agent_owner" in v
            assert "arm" in v
            assert "updated_at" in v


def test_rerunning_replay_idempotency(replay_fixture_db, tmp_path: Path):
    """Verify running the replay stream again does not produce duplicate actions or corrupt state."""
    conn, db_file = replay_fixture_db
    out_dir = tmp_path / "rerun_test"

    runner = ReplayRunner(db_path=db_file, out_dir=out_dir)
    runner.conn = conn
    runner.arms_map = {"U-001": "agent"}

    stream = [
        StreamItem(
            kind="trigger",
            timestamp_str="2026-10-01T04:30:00.000Z",
            timestamp_dt=datetime(2026, 10, 1, 4, 30, tzinfo=timezone.utc),
            data={
                "event_id": "trig-rerun-01",
                "user_id": "U-001",
                "event": "app_open",
                "occurred_at": "2026-10-01T04:30:00.000Z",
                "sent_at": "2026-10-01T04:30:00.000Z",
                "deliver_at": "2026-10-01T04:30:00.000Z",
                "app_dialect": "d1",
            },
            priority=2,
            index=0,
        ),
    ]

    # Run 1
    dec1, act1 = runner.handle_trigger(stream[0])
    assert dec1["decision_type"] == "action"
    assert act1 is not None

    # Run 2 with same runner (simulating rerun on same session)
    dec2, act2 = runner.handle_trigger(stream[0])
    assert dec2["decision_type"] == "no_action"
    assert dec2["reason_code"] == "DUPLICATE_TRIGGER"
    assert act2 is None


def test_stale_goal_judgment_cleared_on_new_replay_commitment(replay_fixture_db, tmp_path: Path):
    """Verify replaying a new goal commitment clears prior post-simulation judgment & evidence."""
    conn, db_file = replay_fixture_db
    out_dir = tmp_path / "replay_stale_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Seed user_state with an existing completed goal judgment and learning evidence for U-001
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_state (
                user_id, agent_owner, arm, subscription_status, active_goal_id,
                goal_committed_at, goal_deadline_at, goal_status, goal_judgment,
                goal_judgment_at, goal_learned, goal_evidence_events,
                active_plan_json, last_event_at, last_app_dialect, updated_at
            ) VALUES (
                'U-001', 'agent-U-001', 'agent', 'never', 'G01',
                '2026-09-30T00:00:00.000Z', '2026-10-01T00:00:00.000Z', 'succeeded', 'succeeded',
                '2026-10-01T01:00:00.000Z', 'Prior G01 play learned summary', '["EV-OLD-PLAY"]',
                '{"type": "home_page"}', '2026-09-30T00:00:00.000Z', 'd1', '2026-10-01T01:00:00.000Z'
            );
            """
        )

    # 2. Replay a new goal commitment for that viewer
    runner = ReplayRunner(db_path=db_file, out_dir=out_dir)
    runner.conn = conn
    runner.arms_map = {"U-001": "agent"}

    trigger_item = StreamItem(
        kind="trigger",
        timestamp_str="2026-10-01T04:30:00.000Z",
        timestamp_dt=datetime(2026, 10, 1, 4, 30, tzinfo=timezone.utc),
        data={
            "event_id": "trig-stale-01",
            "user_id": "U-001",
            "event": "app_open",
            "occurred_at": "2026-10-01T04:30:00.000Z",
            "sent_at": "2026-10-01T04:30:00.000Z",
            "deliver_at": "2026-10-01T04:30:00.000Z",
            "session_id": "s-stale-01",
            "app_dialect": "d1",
        },
        priority=2,
        index=0,
    )
    decision, action = runner.handle_trigger(trigger_item)
    assert decision["decision_type"] == "action"
    assert action is not None
    assert decision["active_goal_id"] == "G01"

    # 3. Verify the new state is in_progress and all old judgment/evidence fields are cleared
    row = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-001';").fetchone()
    assert row["active_goal_id"] == "G01"
    assert row["goal_status"] == "in_progress"
    assert row["goal_judgment"] is None
    assert row["goal_judgment_at"] is None
    assert row["goal_learned"] is None
    assert row["goal_evidence_events"] is None
    assert row["goal_committed_at"] == action["published_at"]
    assert row["goal_deadline_at"] is not None


    # 4. Verify subsequent evaluator can populate the new judgment normally
    from stage_agent.evaluator import GoalEvaluator
    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        ev = {
            "event_id": "EV-NEW-PLAY-999",
            "user_id": "U-001",
            "event": "play_started",
            "received_at": "2026-10-01T05:00:00.000Z",
            "properties": {"title_id": "T0001"},
        }
        f.write(json.dumps(ev) + "\n")

    evaluator = GoalEvaluator(
        pack_dir=Path("one_user_one_agent_pack"),
        db_path=db_file,
        out_dir=out_dir,
    )
    evaluator.evaluate_outcomes(outcomes_file)

    row_after = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-001';").fetchone()
    assert row_after["goal_status"] == "succeeded"
    assert row_after["goal_judgment"] == "succeeded"
    assert row_after["goal_judgment_at"] is not None
    assert "EV-NEW-PLAY-999" in json.loads(row_after["goal_evidence_events"])
    assert "EV-OLD-PLAY" not in json.loads(row_after["goal_evidence_events"])


