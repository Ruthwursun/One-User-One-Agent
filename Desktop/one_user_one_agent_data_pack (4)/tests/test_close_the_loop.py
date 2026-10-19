"""Targeted tests for close-the-loop goal evaluation.

Verifies:
1. Success: Attributable event satisfies goal condition before deadline.
2. Failure after deadline: Deadline expired within observation window with no success event.
3. Unexpired goal: Observation window ends before deadline with no success event (must be inconclusive, not failed).
4. Unavailable evidence: Goal requires undefined or unavailable signal (G10/G11 -> inconclusive).
5. Unsimulated viewer: Viewer without simulated outcomes handled honestly (inconclusive, not failed/zero).
6. Repeated evaluation: Idempotent execution without duplicating or corrupting state.
7. Attribution temporal filtering: Pre-commitment events are excluded from attribution.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from stage_agent.db import init_db
from stage_agent.evaluator import GoalEvaluator


@pytest.fixture
def test_env(tmp_path: Path):
    """Sets up a complete isolated test environment with db, pack config, and out dir."""
    db_path = tmp_path / "test_stage.db"
    conn = init_db(db_path)

    # Insert dummy catalogue
    conn.execute(
        """
        INSERT INTO catalogue (title_id, content_slug, title, dialect, language, content_type, genres, release_date, available_from, is_premium, ingested_at)
        VALUES ('T1', 'series-alpha', 'Alpha Show', 'd1', 'hin', 'show', 'Drama', '2026-09-15T00:00:00.000Z', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z'),
               ('T2', 'movie-beta', 'Beta Movie', 'd1', 'hin', 'film', 'Action', '2026-09-28T00:00:00.000Z', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z');
        """
    )
    conn.commit()
    conn.close()

    # Create dummy goals.json in pack
    pack_dir = tmp_path / "pack"
    config_dir = pack_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    goals_data = {
        "goals": [
            {"goal_id": "G01", "name": "first_play_24h", "description": "First play within 24h", "target_metric": "play_started", "window_hours": 24},
            {"goal_id": "G02", "name": "first_full_episode_3d", "description": "First episode within 3d", "target_metric": "episode_completed", "window_hours": 72},
            {"goal_id": "G03", "name": "second_day_return_2d", "description": "Second day return", "target_metric": "play_started", "window_hours": 48},
            {"goal_id": "G04", "name": "trial_to_paid_7d", "description": "Trial to paid within 7d", "target_metric": "purchase_completed", "window_hours": 168},
            {"goal_id": "G06", "name": "reactivate_lapsed_14d", "description": "Reactivate lapsed within 14d", "target_metric": "purchase_completed", "window_hours": 336},
            {"goal_id": "G10", "name": "early_engagement_7d", "description": "Meaningful engagement", "target_metric": None, "window_hours": 168},
            {"goal_id": "G11", "name": "tv_screen_transition_14d", "description": "TV app transition", "target_metric": "tv_app_open", "window_hours": 336},
        ]
    }
    with open(config_dir / "goals.json", "w", encoding="utf-8") as f:
        json.dump(goals_data, f)

    out_dir = tmp_path / "sample_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "db_path": db_path,
        "pack_dir": pack_dir,
        "out_dir": out_dir,
    }


def _seed_viewer_state(
    db_path: Path,
    user_id: str,
    goal_id: str | None,
    committed_at: str | None,
    deadline_at: str | None = None,
    arm: str = "treatment",
    plan: dict | None = None,
) -> None:
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO user_state (
                user_id, agent_owner, arm, subscription_status, active_goal_id,
                goal_committed_at, goal_deadline_at, goal_status, active_plan_json,
                last_event_at, last_app_dialect, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                user_id,
                f"agent-{user_id}",
                arm,
                "trial",
                goal_id,
                committed_at,
                deadline_at,
                "committed" if goal_id else "none",
                json.dumps(plan) if plan else None,
                committed_at,
                "d1",
                committed_at,
            ),
        )
    conn.close()


def test_goal_evaluation_success(test_env):
    """Test 1: Attributable event satisfies goal condition before deadline -> succeeded."""
    db_path = test_env["db_path"]
    pack_dir = test_env["pack_dir"]
    out_dir = test_env["out_dir"]

    # Viewer committed G01 (24h window)
    _seed_viewer_state(
        db_path,
        user_id="U-SUCCESS",
        goal_id="G01",
        committed_at="2026-10-01T00:00:00.000Z",
        deadline_at="2026-10-02T00:00:00.000Z",
    )

    # Simulated outcomes include play_started before deadline
    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        ev = {
            "event_id": "EV-PLAY-1",
            "user_id": "U-SUCCESS",
            "event": "play_started",
            "received_at": "2026-10-01T04:30:00.000Z",
            "properties": {"title_id": "T1"},
        }
        f.write(json.dumps(ev) + "\n")

    evaluator = GoalEvaluator(pack_dir=pack_dir, db_path=db_path, out_dir=out_dir)
    summary = evaluator.evaluate_outcomes(outcomes_file)

    assert summary["summary_counts"]["succeeded"] == 1

    # Verify persisted DB state
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-SUCCESS';").fetchone()
    conn.close()

    assert row["goal_status"] == "succeeded"
    assert row["goal_judgment"] == "succeeded"
    assert "EV-PLAY-1" in json.loads(row["goal_evidence_events"])
    assert "T1" in row["goal_learned"]
    assert "satisfying G01" in row["goal_learned"]


def test_goal_evaluation_failure_after_deadline(test_env):
    """Test 2: Deadline passed within observation window without success -> failed."""
    db_path = test_env["db_path"]
    pack_dir = test_env["pack_dir"]
    out_dir = test_env["out_dir"]

    # Viewer committed G01 (deadline 2026-10-02T00:00:00.000Z)
    _seed_viewer_state(
        db_path,
        user_id="U-FAIL",
        goal_id="G01",
        committed_at="2026-10-01T00:00:00.000Z",
        deadline_at="2026-10-02T00:00:00.000Z",
    )

    # Observation horizon extends to 2026-10-05 (past deadline), but user only had unrelated event
    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        # Event for U-FAIL (unrelated event type)
        ev1 = {
            "event_id": "EV-SEARCH",
            "user_id": "U-FAIL",
            "event": "search_performed",
            "received_at": "2026-10-01T05:00:00.000Z",
            "properties": {},
        }
        # Another user event establishing observation horizon up to 2026-10-05
        ev2 = {
            "event_id": "EV-HORIZON",
            "user_id": "U-OTHER",
            "event": "play_started",
            "received_at": "2026-10-05T00:00:00.000Z",
            "properties": {"title_id": "T1"},
        }
        f.write(json.dumps(ev1) + "\n")
        f.write(json.dumps(ev2) + "\n")

    evaluator = GoalEvaluator(pack_dir=pack_dir, db_path=db_path, out_dir=out_dir)
    summary = evaluator.evaluate_outcomes(outcomes_file)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-FAIL';").fetchone()
    conn.close()

    assert row["goal_status"] == "failed"
    assert row["goal_judgment"] == "failed"
    assert "failed: deadline" in row["goal_learned"]
    assert json.loads(row["goal_evidence_events"]) == []


def test_goal_evaluation_unexpired_deadline(test_env):
    """Test 3: Observation window ends before deadline -> inconclusive (do not mark failed)."""
    db_path = test_env["db_path"]
    pack_dir = test_env["pack_dir"]
    out_dir = test_env["out_dir"]

    # Viewer committed G06 (14-day window: deadline 2026-10-15T00:00:00.000Z)
    _seed_viewer_state(
        db_path,
        user_id="U-OPEN",
        goal_id="G06",
        committed_at="2026-10-01T00:00:00.000Z",
        deadline_at="2026-10-15T00:00:00.000Z",
    )

    # Observation window only runs through Day 3 (2026-10-04T00:00:00.000Z)
    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        ev = {
            "event_id": "EV-APP",
            "user_id": "U-OPEN",
            "event": "app_open",
            "received_at": "2026-10-02T12:00:00.000Z",
            "properties": {},
        }
        f.write(json.dumps(ev) + "\n")

    evaluator = GoalEvaluator(pack_dir=pack_dir, db_path=db_path, out_dir=out_dir)
    summary = evaluator.evaluate_outcomes(outcomes_file)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-OPEN';").fetchone()
    conn.close()

    assert row["goal_status"] == "inconclusive"
    assert row["goal_judgment"] == "inconclusive"
    assert "remains open" in row["goal_learned"]
    assert "cannot be judged as failed" in row["goal_learned"]


def test_goal_evaluation_unavailable_evidence(test_env):
    """Test 4: Goal requires undefined or unavailable signal (G10/G11) -> inconclusive."""
    db_path = test_env["db_path"]
    pack_dir = test_env["pack_dir"]
    out_dir = test_env["out_dir"]

    _seed_viewer_state(
        db_path,
        user_id="U-G10",
        goal_id="G10",
        committed_at="2026-10-01T00:00:00.000Z",
        deadline_at="2026-10-08T00:00:00.000Z",
    )
    _seed_viewer_state(
        db_path,
        user_id="U-G11",
        goal_id="G11",
        committed_at="2026-10-01T00:00:00.000Z",
        deadline_at="2026-10-15T00:00:00.000Z",
    )

    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        # Both users simulated
        ev1 = {"event_id": "E1", "user_id": "U-G10", "event": "app_open", "received_at": "2026-10-02T00:00:00.000Z"}
        ev2 = {"event_id": "E2", "user_id": "U-G11", "event": "app_open", "received_at": "2026-10-02T00:00:00.000Z"}
        f.write(json.dumps(ev1) + "\n")
        f.write(json.dumps(ev2) + "\n")

    evaluator = GoalEvaluator(pack_dir=pack_dir, db_path=db_path, out_dir=out_dir)
    summary = evaluator.evaluate_outcomes(outcomes_file)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    r10 = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-G10';").fetchone()
    r11 = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-G11';").fetchone()
    conn.close()

    assert r10["goal_status"] == "inconclusive"
    assert "unavailable or underspecified" in r10["goal_learned"]

    assert r11["goal_status"] == "inconclusive"
    assert "unavailable or underspecified" in r11["goal_learned"]


def test_goal_evaluation_unsimulated_viewer(test_env):
    """Test 5: Viewer not in post-replay simulation -> inconclusive (do not treat as failed/zero)."""
    db_path = test_env["db_path"]
    pack_dir = test_env["pack_dir"]
    out_dir = test_env["out_dir"]

    # Viewer committed G01
    _seed_viewer_state(
        db_path,
        user_id="U-UNSIMULATED",
        goal_id="G01",
        committed_at="2026-10-01T00:00:00.000Z",
        deadline_at="2026-10-02T00:00:00.000Z",
    )

    # Outcomes file exists but has events ONLY for another user
    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        ev = {
            "event_id": "E-DIFF",
            "user_id": "U-OTHER",
            "event": "play_started",
            "received_at": "2026-10-03T00:00:00.000Z",
            "properties": {"title_id": "T1"},
        }
        f.write(json.dumps(ev) + "\n")

    evaluator = GoalEvaluator(pack_dir=pack_dir, db_path=db_path, out_dir=out_dir)
    summary = evaluator.evaluate_outcomes(outcomes_file)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-UNSIMULATED';").fetchone()
    conn.close()

    assert row["goal_status"] == "inconclusive"
    assert row["goal_judgment"] == "inconclusive"
    assert "unobserved" in row["goal_learned"]
    assert "cannot establish goal result from missing evidence" in row["goal_learned"]


def test_goal_evaluation_idempotency(test_env):
    """Test 6: Repeated evaluation does not duplicate or corrupt state."""
    db_path = test_env["db_path"]
    pack_dir = test_env["pack_dir"]
    out_dir = test_env["out_dir"]

    _seed_viewer_state(
        db_path,
        user_id="U-REPEAT",
        goal_id="G01",
        committed_at="2026-10-01T00:00:00.000Z",
        deadline_at="2026-10-02T00:00:00.000Z",
    )

    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        ev = {
            "event_id": "EV-P1",
            "user_id": "U-REPEAT",
            "event": "play_started",
            "received_at": "2026-10-01T06:00:00.000Z",
            "properties": {"title_id": "T1"},
        }
        f.write(json.dumps(ev) + "\n")

    evaluator = GoalEvaluator(pack_dir=pack_dir, db_path=db_path, out_dir=out_dir)

    # First evaluation
    res1 = evaluator.evaluate_outcomes(outcomes_file)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row1 = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-REPEAT';").fetchone()
    conn.close()

    assert row1["goal_status"] == "succeeded"
    assert json.loads(row1["goal_evidence_events"]) == ["EV-P1"]

    # Second evaluation (rerun)
    res2 = evaluator.evaluate_outcomes(outcomes_file)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    total_rows = conn.execute("SELECT count(*) FROM user_state WHERE user_id = 'U-REPEAT';").fetchone()[0]
    row2 = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-REPEAT';").fetchone()
    conn.close()

    assert total_rows == 1
    assert row2["goal_status"] == "succeeded"
    assert row2["goal_judgment"] == "succeeded"
    assert json.loads(row2["goal_evidence_events"]) == ["EV-P1"]
    assert res1["summary_counts"] == res2["summary_counts"]


def test_goal_attribution_temporal_filtering(test_env):
    """Test 7: Events occurring prior to goal commitment timestamp are excluded from attribution."""
    db_path = test_env["db_path"]
    pack_dir = test_env["pack_dir"]
    out_dir = test_env["out_dir"]

    # Goal committed at 12:00
    _seed_viewer_state(
        db_path,
        user_id="U-TEMPORAL",
        goal_id="G01",
        committed_at="2026-10-01T12:00:00.000Z",
        deadline_at="2026-10-02T12:00:00.000Z",
    )

    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        # Pre-commitment event (e.g. 09:00)
        ev_pre = {
            "event_id": "EV-PRE",
            "user_id": "U-TEMPORAL",
            "event": "play_started",
            "received_at": "2026-10-01T09:00:00.000Z",
            "properties": {"title_id": "T1"},
        }
        # Later horizon marker past deadline
        ev_horizon = {
            "event_id": "EV-HORIZON",
            "user_id": "U-TEMPORAL",
            "event": "app_open",
            "received_at": "2026-10-03T00:00:00.000Z",
            "properties": {},
        }
        f.write(json.dumps(ev_pre) + "\n")
        f.write(json.dumps(ev_horizon) + "\n")

    evaluator = GoalEvaluator(pack_dir=pack_dir, db_path=db_path, out_dir=out_dir)
    evaluator.evaluate_outcomes(outcomes_file)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-TEMPORAL';").fetchone()
    conn.close()

    # Pre-commitment play should NOT count toward G01; deadline passed -> failed
    assert row["goal_status"] == "failed"
    assert "EV-PRE" not in json.loads(row["goal_evidence_events"])


def test_stale_goal_judgment_cleared_on_new_commitment(test_env):
    """Test 8 (Regression): Committing a new goal clears any prior post-simulation judgment & evidence."""
    db_path = test_env["db_path"]
    pack_dir = test_env["pack_dir"]
    out_dir = test_env["out_dir"]

    # 1. Seed user_state with an existing completed goal judgment and learning evidence
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO user_state (
                user_id, agent_owner, arm, subscription_status, active_goal_id,
                goal_committed_at, goal_deadline_at, goal_status, goal_judgment,
                goal_judgment_at, goal_learned, goal_evidence_events,
                active_plan_json, last_event_at, last_app_dialect, updated_at
            ) VALUES (
                'U-STALE-REG', 'agent-U-STALE-REG', 'treatment', 'trial', 'G01',
                '2026-10-01T00:00:00.000Z', '2026-10-02T00:00:00.000Z', 'succeeded', 'succeeded',
                '2026-10-02T01:00:00.000Z', 'Old G01 play learning summary', '["EV-OLD-P1"]',
                '{"type": "home_page"}', '2026-10-01T00:00:00.000Z', 'd1', '2026-10-02T01:00:00.000Z'
            );
            """
        )
    conn.close()

    # 2. Replay a new goal commitment for that viewer (G02: first_full_episode_3d)
    from stage_agent.policy import PolicyGate
    from stage_agent.model_provider import MockModelProvider
    from stage_agent.eligibility import GoalRegistry
    from stage_agent.planner import DecisionPlanner

    gate = PolicyGate()
    model_provider = MockModelProvider()
    registry = GoalRegistry(pack_dir / "config" / "goals.json")
    planner = DecisionPlanner(policy_gate=gate, model_provider=model_provider, goal_registry=registry)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Execute goal commitment update for G02
    planner._update_user_state(
        user_id="U-STALE-REG",
        goal_id="G02",
        action_dict={"type": "home_page", "hero_title_id": "T1"},
        at_iso="2026-10-03T10:00:00.000Z",
        dialect="d1",
        conn=conn,
        subscription_status="trial",
        arm="treatment",
    )

    # 3. Verify the new state is in_progress and all old judgment/evidence fields are cleared to NULL
    row = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-STALE-REG';").fetchone()
    assert row["active_goal_id"] == "G02"
    assert row["goal_status"] == "in_progress"
    assert row["goal_judgment"] is None
    assert row["goal_judgment_at"] is None
    assert row["goal_learned"] is None
    assert row["goal_evidence_events"] is None
    assert row["goal_committed_at"] == "2026-10-03T10:00:00.000Z"
    # G02 deadline is +72h = 2026-10-06T10:00:00.000Z
    assert row["goal_deadline_at"] == "2026-10-06T10:00:00.000Z"
    conn.close()

    # 4. Verify subsequent evaluator populates the new judgment normally
    outcomes_file = out_dir / "outcomes.jsonl"
    with open(outcomes_file, "w", encoding="utf-8") as f:
        ev = {
            "event_id": "EV-NEW-EP-COMPLETED",
            "user_id": "U-STALE-REG",
            "event": "episode_completed",
            "received_at": "2026-10-03T15:00:00.000Z",
            "properties": {"episode_id": "EP101"},
        }
        f.write(json.dumps(ev) + "\n")

    evaluator = GoalEvaluator(pack_dir=pack_dir, db_path=db_path, out_dir=out_dir)
    evaluator.evaluate_outcomes(outcomes_file)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row_after = conn.execute("SELECT * FROM user_state WHERE user_id = 'U-STALE-REG';").fetchone()
    conn.close()

    assert row_after["goal_status"] == "succeeded"
    assert row_after["goal_judgment"] == "succeeded"
    assert row_after["goal_judgment_at"] is not None
    assert "EV-NEW-EP-COMPLETED" in json.loads(row_after["goal_evidence_events"])
    assert "EV-OLD-P1" not in json.loads(row_after["goal_evidence_events"])
    assert "satisfying G02" in row_after["goal_learned"]
    assert "EP101" in row_after["goal_learned"]

