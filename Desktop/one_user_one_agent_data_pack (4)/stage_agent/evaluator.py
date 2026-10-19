"""Post-simulation close-the-loop evaluator for STAGE One User, One Agent.

Evaluates each viewer's active goal against simulated post-replay outcome events
using only attributable signals, actual success conditions, and deadlines.
Persists judgments (succeeded, failed, inconclusive) and evidence-grounded
learning summaries to SQLite user_state and exports updated viewer state.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from stage_agent.config import (
    DATA_PACK_DIR,
    DB_PATH,
    GOAL_WINDOWS_HOURS,
    UTC,
    get_db_path,
    ist_date_str,
    now_utc_iso,
    parse_iso_datetime,
)


class GoalEvaluator:
    """Post-simulation close-the-loop evaluator.

    Attributes:
        pack_dir: Directory containing configuration and reference data.
        db_path: Path to the SQLite storage database.
        out_dir: Directory where sample run artifacts reside.
    """

    def __init__(
        self,
        pack_dir: Path | str = DATA_PACK_DIR,
        db_path: Path | str | None = None,
        out_dir: Path | str = "sample_run",
    ) -> None:
        self.pack_dir = Path(pack_dir)
        self.db_path = get_db_path(db_path)
        self.out_dir = Path(out_dir)
        self.goals_config = self._load_goals_config()

    def _load_goals_config(self) -> dict[str, dict[str, Any]]:
        """Load candidate goals configuration from pack."""
        goals_path = self.pack_dir / "config" / "goals.json"
        if not goals_path.is_file():
            return {}
        try:
            with open(goals_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {g["goal_id"]: g for g in data.get("goals", [])}
        except Exception:
            return {}

    def evaluate_outcomes(
        self,
        outcomes_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Evaluate active goals against attributable outcomes and update user_state.

        Args:
            outcomes_path: Path to outcomes JSONL (defaults to out_dir / outcomes.jsonl).

        Returns:
            Dictionary containing summary counts and per-goal judgment statistics.
        """
        if outcomes_path is None:
            outcomes_file = self.out_dir / "outcomes.jsonl"
            if not outcomes_file.is_file():
                outcomes_file = self.out_dir / "sim" / "outcome_events.jsonl"
        else:
            outcomes_file = Path(outcomes_path)

        # Load outcome events grouped by user_id and find simulation horizon end
        user_outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        simulated_users: set[str] = set()
        horizon_end_dt: datetime | None = None

        if outcomes_file.is_file():
            with open(outcomes_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        u = ev.get("user_id")
                        if u:
                            user_outcomes[u].append(ev)
                            simulated_users.add(u)
                        rcv = parse_iso_datetime(ev.get("received_at"))
                        if rcv and (horizon_end_dt is None or rcv > horizon_end_dt):
                            horizon_end_dt = rcv
                    except json.JSONDecodeError:
                        continue

        # Sort each user's events chronologically by received_at
        for u in user_outcomes:
            user_outcomes[u].sort(key=lambda x: x.get("received_at") or "")

        horizon_end_iso = (
            horizon_end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            if horizon_end_dt
            else now_utc_iso()
        )
        judgment_time_iso = now_utc_iso()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Query all viewer states
        with conn:
            users = conn.execute(
                """
                SELECT user_id, arm, subscription_status, active_goal_id,
                       goal_committed_at, goal_deadline_at, active_plan_json
                FROM user_state
                ORDER BY user_id;
                """
            ).fetchall()

            judgments: list[dict[str, Any]] = []

            for row in users:
                uid = row["user_id"]
                goal_id = row["active_goal_id"]
                committed_iso = row["goal_committed_at"]
                deadline_iso = row["goal_deadline_at"]
                plan_json = row["active_plan_json"]

                judgment_record = self._judge_single_user(
                    user_id=uid,
                    goal_id=goal_id,
                    committed_iso=committed_iso,
                    deadline_iso=deadline_iso,
                    plan_json=plan_json,
                    user_events=user_outcomes.get(uid, []),
                    is_simulated=(uid in simulated_users),
                    horizon_end_dt=horizon_end_dt,
                    horizon_end_iso=horizon_end_iso,
                    conn=conn,
                )
                judgment_record["judgment_at"] = judgment_time_iso
                judgments.append(judgment_record)

                # Persist to SQLite user_state idempotently
                conn.execute(
                    """
                    UPDATE user_state
                    SET goal_status = ?,
                        goal_judgment = ?,
                        goal_judgment_at = ?,
                        goal_learned = ?,
                        goal_evidence_events = ?,
                        goal_deadline_at = COALESCE(goal_deadline_at, ?),
                        updated_at = ?
                    WHERE user_id = ?;
                    """,
                    (
                        judgment_record["status"],
                        judgment_record["status"],
                        judgment_time_iso,
                        judgment_record["learned"],
                        json.dumps(judgment_record["evidence_events"], ensure_ascii=False),
                        judgment_record["deadline_iso"],
                        judgment_time_iso,
                        uid,
                    ),
                )

        conn.close()

        # Compute summary metrics
        summary_counts: dict[str, int] = defaultdict(int)
        goal_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for j in judgments:
            st = j["status"]
            gid = j["goal_id"] or "none"
            summary_counts[st] += 1
            goal_breakdown[gid][st] += 1

        return {
            "evaluation_time_iso": judgment_time_iso,
            "horizon_end_iso": horizon_end_iso,
            "total_viewers_evaluated": len(judgments),
            "simulated_viewers_count": len(simulated_users),
            "summary_counts": dict(summary_counts),
            "goal_breakdown": {k: dict(v) for k, v in goal_breakdown.items()},
        }

    def _judge_single_user(
        self,
        user_id: str,
        goal_id: str | None,
        committed_iso: str | None,
        deadline_iso: str | None,
        plan_json: str | None,
        user_events: list[dict[str, Any]],
        is_simulated: bool,
        horizon_end_dt: datetime | None,
        horizon_end_iso: str,
        conn: sqlite3.Connection,
    ) -> dict[str, Any]:
        """Evaluate a single viewer's goal judgment and attribution."""
        # Case 1: No active goal was committed
        if not goal_id or goal_id == "none":
            return {
                "user_id": user_id,
                "goal_id": None,
                "status": "none",
                "deadline_iso": None,
                "learned": "No active goal committed for viewer.",
                "evidence_events": [],
            }

        goal_def = self.goals_config.get(goal_id, {})
        goal_name = goal_def.get("name", goal_id)

        # Compute deadline if missing
        committed_dt = parse_iso_datetime(committed_iso) if committed_iso else None
        deadline_dt = parse_iso_datetime(deadline_iso) if deadline_iso else None

        if committed_dt and not deadline_dt:
            w_hours = GOAL_WINDOWS_HOURS.get(goal_id, 72)
            deadline_dt = committed_dt + timedelta(hours=w_hours)
            deadline_iso = deadline_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Case 2: Viewer was not simulated in post-replay outcomes
        if not is_simulated:
            return {
                "user_id": user_id,
                "goal_id": goal_id,
                "status": "inconclusive",
                "deadline_iso": deadline_iso,
                "learned": (
                    f"Outcome events unobserved for goal {goal_id} ({goal_name}): "
                    "viewer was not included in post-replay simulation (restricted to Day-0 triggered users); "
                    "cannot establish goal result from missing evidence."
                ),
                "evidence_events": [],
            }

        # Case 3: Goal success signal is undefined or unavailable in event stream (G10, G11)
        if goal_id in ("G10", "G11"):
            signal_name = "engages meaningfully" if goal_id == "G10" else "tv_app_open"
            return {
                "user_id": user_id,
                "goal_id": goal_id,
                "status": "inconclusive",
                "deadline_iso": deadline_iso,
                "learned": (
                    f"Success signal for {goal_id} ({goal_name}: {signal_name}) is unavailable or "
                    "underspecified in event stream; cannot establish goal result from attributable events."
                ),
                "evidence_events": [],
            }

        # Filter attributable outcome events occurring at or after goal commitment
        attributable_events: list[dict[str, Any]] = []
        for ev in user_events:
            ev_ts = parse_iso_datetime(ev.get("received_at"))
            if not ev_ts:
                continue
            if committed_dt and ev_ts < committed_dt:
                continue
            attributable_events.append(ev)

        # Check goal-specific success condition
        success_event, success_learned = self._check_success_condition(
            user_id=user_id,
            goal_id=goal_id,
            goal_def=goal_def,
            committed_dt=committed_dt,
            deadline_dt=deadline_dt,
            deadline_iso=deadline_iso or "",
            plan_json=plan_json,
            events=attributable_events,
            conn=conn,
        )

        if success_event is not None:
            ev_id = success_event.get("event_id", "unknown")
            return {
                "user_id": user_id,
                "goal_id": goal_id,
                "status": "succeeded",
                "deadline_iso": deadline_iso,
                "learned": success_learned,
                "evidence_events": [ev_id] if ev_id != "unknown" else [],
            }

        # Case 4: No success event occurred. Check if deadline has expired.
        if deadline_dt and horizon_end_dt and horizon_end_dt < deadline_dt:
            # Deadline has NOT passed yet within the observation window
            return {
                "user_id": user_id,
                "goal_id": goal_id,
                "status": "inconclusive",
                "deadline_iso": deadline_iso,
                "learned": (
                    f"Goal {goal_id} ({goal_name}) remains open: deadline {deadline_iso} "
                    f"has not passed within observation window ending {horizon_end_iso}; result cannot be judged as failed."
                ),
                "evidence_events": [],
            }

        # Case 5: Deadline passed and no success event was observed
        return {
            "user_id": user_id,
            "goal_id": goal_id,
            "status": "failed",
            "deadline_iso": deadline_iso,
            "learned": (
                f"Goal {goal_id} ({goal_name}) failed: deadline {deadline_iso} passed "
                "with no attributable success events observed."
            ),
            "evidence_events": [],
        }

    def _check_success_condition(
        self,
        user_id: str,
        goal_id: str,
        goal_def: dict[str, Any],
        committed_dt: datetime | None,
        deadline_dt: datetime | None,
        deadline_iso: str,
        plan_json: str | None,
        events: list[dict[str, Any]],
        conn: sqlite3.Connection,
    ) -> tuple[dict[str, Any] | None, str]:
        """Check if any attributable event satisfies the goal's success condition before deadline."""
        goal_name = goal_def.get("name", goal_id)

        # G01: first_play_24h (event == 'play_started')
        if goal_id == "G01":
            for ev in events:
                if ev.get("event") == "play_started":
                    ev_ts = parse_iso_datetime(ev.get("received_at"))
                    if deadline_dt and ev_ts and ev_ts > deadline_dt:
                        continue
                    tid = ev.get("properties", {}).get("title_id", "unknown")
                    return (
                        ev,
                        f"Viewer started play for title {tid} at {ev.get('received_at')} (event: {ev.get('event_id')}), "
                        f"satisfying G01 ({goal_name}) within deadline {deadline_iso}.",
                    )
            return None, ""

        # G02: first_full_episode_3d (event == 'episode_completed')
        if goal_id == "G02":
            for ev in events:
                if ev.get("event") == "episode_completed":
                    ev_ts = parse_iso_datetime(ev.get("received_at"))
                    if deadline_dt and ev_ts and ev_ts > deadline_dt:
                        continue
                    epid = ev.get("properties", {}).get("episode_id", "unknown")
                    return (
                        ev,
                        f"Viewer completed episode {epid} at {ev.get('received_at')} (event: {ev.get('event_id')}), "
                        f"satisfying G02 ({goal_name}) within deadline {deadline_iso}.",
                    )
            return None, ""

        # G03: second_day_return_2d (Corrected rule: event == 'play_started' on later IST day)
        if goal_id == "G03":
            start_ist = ist_date_str(committed_dt) if committed_dt else None
            for ev in events:
                if ev.get("event") == "play_started":
                    ev_ts = parse_iso_datetime(ev.get("received_at"))
                    if deadline_dt and ev_ts and ev_ts > deadline_dt:
                        continue
                    ev_ist = ist_date_str(ev_ts) if ev_ts else None
                    if start_ist and ev_ist and ev_ist > start_ist:
                        return (
                            ev,
                            f"Viewer returned and played on later IST day {ev_ist} at {ev.get('received_at')} "
                            f"(event: {ev.get('event_id')}), satisfying G03 ({goal_name}) within deadline {deadline_iso}.",
                        )
            return None, ""

        # G04 & G05: trial_to_paid_7d & first_paid_7d (event == 'purchase_completed', is_trial = false)
        if goal_id in ("G04", "G05"):
            for ev in events:
                if ev.get("event") == "purchase_completed":
                    props = ev.get("properties", {})
                    if props.get("is_trial") is False or not props.get("is_trial"):
                        ev_ts = parse_iso_datetime(ev.get("received_at"))
                        if deadline_dt and ev_ts and ev_ts > deadline_dt:
                            continue
                        plan = props.get("plan_id", "P_MONTH")
                        price = props.get("price_inr", 149)
                        return (
                            ev,
                            f"Viewer completed paid purchase for plan {plan} (₹{price}) at {ev.get('received_at')} "
                            f"(event: {ev.get('event_id')}), satisfying {goal_id} ({goal_name}) within deadline {deadline_iso}.",
                        )
            return None, ""

        # G06: reactivate_lapsed_14d (event == 'purchase_completed', any paid plan)
        if goal_id == "G06":
            for ev in events:
                if ev.get("event") == "purchase_completed":
                    ev_ts = parse_iso_datetime(ev.get("received_at"))
                    if deadline_dt and ev_ts and ev_ts > deadline_dt:
                        continue
                    plan = ev.get("properties", {}).get("plan_id", "unknown")
                    return (
                        ev,
                        f"Lapsed viewer repurchased plan {plan} at {ev.get('received_at')} (event: {ev.get('event_id')}), "
                        f"satisfying G06 ({goal_name}) within deadline {deadline_iso}.",
                    )
            return None, ""

        # G07: resume_series_3d (event == 'play_started' on same series slug as unfinished show)
        if goal_id == "G07":
            candidate_slugs = self._get_unfinished_series_slugs(user_id, committed_dt, plan_json, conn)
            for ev in events:
                if ev.get("event") == "play_started":
                    ev_ts = parse_iso_datetime(ev.get("received_at"))
                    if deadline_dt and ev_ts and ev_ts > deadline_dt:
                        continue
                    tid = ev.get("properties", {}).get("title_id")
                    if tid:
                        slug_row = conn.execute(
                            "SELECT content_slug FROM catalogue WHERE title_id = ?;", (tid,)
                        ).fetchone()
                        if slug_row:
                            ev_slug = slug_row["content_slug"]
                            if ev_slug in candidate_slugs or not candidate_slugs:
                                return (
                                    ev,
                                    f"Viewer resumed series {ev_slug} with play on title {tid} at {ev.get('received_at')} "
                                    f"(event: {ev.get('event_id')}), satisfying G07 ({goal_name}) within deadline {deadline_iso}.",
                                )
            return None, ""

        # G08: try_new_release_7d (event == 'play_started' on title released in 30 days before goal commitment)
        if goal_id == "G08":
            for ev in events:
                if ev.get("event") == "play_started":
                    ev_ts = parse_iso_datetime(ev.get("received_at"))
                    if deadline_dt and ev_ts and ev_ts > deadline_dt:
                        continue
                    tid = ev.get("properties", {}).get("title_id")
                    if tid and committed_dt:
                        rel_row = conn.execute(
                            "SELECT release_date FROM catalogue WHERE title_id = ?;", (tid,)
                        ).fetchone()
                        if rel_row and rel_row["release_date"]:
                            try:
                                rel_dt = datetime.fromisoformat(rel_row["release_date"]).replace(tzinfo=UTC)
                                diff_days = (committed_dt - rel_dt).days
                                if 0 <= diff_days <= 30:
                                    return (
                                        ev,
                                        f"Viewer played new release title {tid} (released {rel_row['release_date']}) "
                                        f"at {ev.get('received_at')} (event: {ev.get('event_id')}), satisfying G08 ({goal_name}) within deadline {deadline_iso}.",
                                    )
                            except Exception:
                                pass
            return None, ""

        # G09: weekly_habit_14d (play_started on >= 3 distinct IST calendar days)
        if goal_id == "G09":
            distinct_days: set[str] = set()
            habit_events: list[dict[str, Any]] = []
            for ev in events:
                if ev.get("event") == "play_started":
                    ev_ts = parse_iso_datetime(ev.get("received_at"))
                    if deadline_dt and ev_ts and ev_ts > deadline_dt:
                        continue
                    day_str = ist_date_str(ev_ts) if ev_ts else None
                    if day_str and day_str not in distinct_days:
                        distinct_days.add(day_str)
                        habit_events.append(ev)
                    if len(distinct_days) >= 3:
                        return (
                            habit_events[-1],
                            f"Viewer watched on {len(distinct_days)} distinct IST days ({sorted(list(distinct_days))[:3]}), "
                            f"satisfying G09 ({goal_name}) within deadline {deadline_iso}.",
                        )
            return None, ""

        # G12: return_after_absence_7d (play_started on any title within deadline)
        if goal_id == "G12":
            for ev in events:
                if ev.get("event") == "play_started":
                    ev_ts = parse_iso_datetime(ev.get("received_at"))
                    if deadline_dt and ev_ts and ev_ts > deadline_dt:
                        continue
                    tid = ev.get("properties", {}).get("title_id", "unknown")
                    return (
                        ev,
                        f"Viewer returned after absence and started play on title {tid} at {ev.get('received_at')} "
                        f"(event: {ev.get('event_id')}), satisfying G12 ({goal_name}) within deadline {deadline_iso}.",
                    )
            return None, ""

        return None, ""

    def _get_unfinished_series_slugs(
        self,
        user_id: str,
        committed_dt: datetime | None,
        plan_json: str | None,
        conn: sqlite3.Connection,
    ) -> set[str]:
        """Identify candidate unfinished series content slugs for user."""
        slugs: set[str] = set()

        # 1. From action plan hero title if present
        if plan_json:
            try:
                plan = json.loads(plan_json)
                hero_tid = plan.get("hero_title_id") or plan.get("title_id")
                if hero_tid:
                    row = conn.execute(
                        "SELECT content_slug FROM catalogue WHERE title_id = ?;", (hero_tid,)
                    ).fetchone()
                    if row:
                        slugs.add(row["content_slug"])
            except Exception:
                pass

        # 2. From historical series started with unwatched episodes
        committed_iso = (
            committed_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            if committed_dt
            else now_utc_iso()
        )
        series_rows = conn.execute(
            """
            SELECT DISTINCT c.content_slug
            FROM events e
            JOIN catalogue c ON e.title_id = c.title_id
            WHERE e.user_id = ? AND e.event = 'play_started' AND e.received_at <= ?
              AND c.content_type IN ('show', 'micro_drama');
            """,
            (user_id, committed_iso),
        ).fetchall()

        for r in series_rows:
            slugs.add(r["content_slug"])

        return slugs
