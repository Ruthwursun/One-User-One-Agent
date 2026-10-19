"""Deterministic goal eligibility verification and dynamic goal configuration engine."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from stage_agent.config import (
    DATA_PACK_DIR,
    UTC,
    ist_date_str,
    parse_iso_datetime,
    to_ist,
)
from stage_agent.models import EligibilityResult, GoalDefinition


class GoalRegistry:
    """Manages candidate goals and supports runtime definition updates from control events."""

    def __init__(self, goals_path: str | Path | None = None) -> None:
        self.goals: dict[str, GoalDefinition] = {}
        self.correction_history: list[dict[str, Any]] = []

        target_path = Path(goals_path) if goals_path else (DATA_PACK_DIR / "config" / "goals.json")
        if target_path.is_file():
            self.load_from_json(target_path)

    def load_from_json(self, path: Path) -> None:
        """Load goal configurations from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for g in data.get("goals", []):
                goal = GoalDefinition(
                    goal_id=g["goal_id"],
                    name=g["name"],
                    description=g["description"],
                    eligibility=g.get("eligibility", {}),
                    success=g.get("success", {}),
                )
                self.goals[goal.goal_id] = goal

    def get_goal(self, goal_id: str) -> GoalDefinition | None:
        """Retrieve goal definition by goal_id."""
        return self.goals.get(goal_id)

    def apply_control_event(self, control_event: dict[str, Any]) -> bool:
        """Apply a runtime control event (e.g. goal_rule_corrected) dynamically.

        Schema:
            {
                "at": "...Z",
                "type": "goal_rule_corrected",
                "goal_id": "G03",
                "old_success": {...},
                "new_success": {...},
                "applies_to": "every G03 goal that is still open...",
                "note": "..."
            }
        """
        if control_event.get("type") == "goal_rule_corrected":
            goal_id = control_event.get("goal_id")
            new_success = control_event.get("new_success")
            if goal_id and goal_id in self.goals and new_success:
                old_success = self.goals[goal_id].success
                self.goals[goal_id].success = new_success
                self.correction_history.append(
                    {
                        "at": control_event.get("at"),
                        "goal_id": goal_id,
                        "old_success": old_success,
                        "new_success": new_success,
                        "note": control_event.get("note"),
                    }
                )
                return True
        return False


# Singleton registry
_DEFAULT_REGISTRY: GoalRegistry | None = None


def get_default_registry() -> GoalRegistry:
    """Get or instantiate default global GoalRegistry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = GoalRegistry()
    return _DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# Deterministic Goal Eligibility Checks
# ---------------------------------------------------------------------------


def evaluate_goal_eligibility(
    user_id: str,
    goal_id: str,
    conn: sqlite3.Connection,
    at_time: datetime | str | None = None,
    registry: GoalRegistry | None = None,
) -> EligibilityResult:
    """Deterministically evaluate whether a user meets the eligibility criteria for a given goal.

    Grounded strictly in verified SQLite data records and source policy rules.
    """
    reg = registry or get_default_registry()
    goal = reg.get_goal(goal_id)
    if not goal:
        return EligibilityResult(
            goal_id=goal_id,
            eligible=False,
            reason_code="UNKNOWN_GOAL",
            message=f"Goal {goal_id} does not exist in configuration.",
        )

    eval_dt = parse_iso_datetime(at_time) if isinstance(at_time, str) else at_time
    if eval_dt is None:
        eval_dt = datetime.now(UTC)
    eval_iso = eval_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # 1. Permanently Disqualified Goals
    if goal_id == "G10":
        return EligibilityResult(
            goal_id=goal_id,
            eligible=False,
            reason_code="DISQUALIFIED_NULL_SUCCESS_EVENT",
            message="Goal G10 has null success event and is disqualified from automated execution.",
            details={"condition": goal.success.get("condition")},
        )
    if goal_id == "G11":
        return EligibilityResult(
            goal_id=goal_id,
            eligible=False,
            reason_code="DISQUALIFIED_UNSUPPORTED_TV_EVENT",
            message="Goal G11 requires tv_app_open which is unsupported in event streams and simulator.",
            details={"required_event": "tv_app_open"},
        )

    # 2. Fetch User Profile
    row = conn.execute(
        "SELECT user_id, signup_at, subscription_status, preferred_dialect FROM profiles WHERE user_id = ?;",
        (user_id,),
    ).fetchone()

    if not row:
        return EligibilityResult(
            goal_id=goal_id,
            eligible=False,
            reason_code="MISSING_PROFILE",
            message=f"No profile record found in database for user {user_id}.",
        )

    sub_status = row["subscription_status"]
    signup_at_str = row["signup_at"]
    signup_dt = parse_iso_datetime(signup_at_str)

    # Check for recent organic purchases that might override snapshot status
    recent_purchase = conn.execute(
        """
        SELECT event, plan_id, properties_json, received_at FROM events
        WHERE user_id = ? AND event IN ('purchase_completed', 'subscription_renewed', 'subscription_lapsed')
          AND received_at <= ?
        ORDER BY received_at DESC LIMIT 1;
        """,
        (user_id, eval_iso),
    ).fetchone()

    if recent_purchase:
        ev_type = recent_purchase["event"]
        if ev_type == "purchase_completed":
            props = {}
            if recent_purchase["properties_json"]:
                try:
                    props = json.loads(recent_purchase["properties_json"])
                except Exception:
                    pass
            is_trial = bool(props.get("is_trial", False))
            plan = props.get("plan_id") or recent_purchase["plan_id"]
            if is_trial or plan == "P_TRIAL":
                sub_status = "trial_active"
            else:
                sub_status = "active"
        elif ev_type == "subscription_renewed":
            sub_status = "active"
        elif ev_type == "subscription_lapsed":
            sub_status = "lapsed"

    # 3. Goal-Specific Rule Evaluation

    # G01: first_play_24h (Signed up in the last 7 days and has never started a play)
    if goal_id == "G01":
        if not signup_dt:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="MISSING_SIGNUP_DATE",
                message="Cannot verify 7-day signup window without signup_at.",
            )
        days_since_signup = (eval_dt - signup_dt).total_seconds() / 86400.0
        if days_since_signup > 7.0 or days_since_signup < 0.0:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="SIGNUP_OUTSIDE_7_DAYS",
                message=f"User signed up {days_since_signup:.1f} days ago (> 7.0 limit).",
                details={"days_since_signup": days_since_signup},
            )
        play_count = conn.execute(
            "SELECT count(*) FROM events WHERE user_id = ? AND event = 'play_started' AND received_at <= ?;",
            (user_id, eval_iso),
        ).fetchone()[0]
        if play_count > 0:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="PRIOR_PLAY_EXISTS",
                message="User has already started at least one play.",
                details={"prior_plays": play_count},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message="User signed up within 7 days and has 0 prior plays.",
            details={"days_since_signup": days_since_signup},
        )

    # G02: first_full_episode_3d (Has never completed an episode)
    if goal_id == "G02":
        comp_count = conn.execute(
            "SELECT count(*) FROM events WHERE user_id = ? AND event = 'episode_completed' AND received_at <= ?;",
            (user_id, eval_iso),
        ).fetchone()[0]
        if comp_count > 0:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="PRIOR_COMPLETION_EXISTS",
                message="User has already completed at least one episode.",
                details={"completions": comp_count},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message="User has never completed an episode.",
        )

    # G03: second_day_return_2d (Signed up in the last 2 days)
    if goal_id == "G03":
        if not signup_dt:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="MISSING_SIGNUP_DATE",
                message="Cannot verify 2-day signup window without signup_at.",
            )
        days_since_signup = (eval_dt - signup_dt).total_seconds() / 86400.0
        if days_since_signup > 2.0 or days_since_signup < 0.0:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="SIGNUP_OUTSIDE_2_DAYS",
                message=f"User signed up {days_since_signup:.1f} days ago (> 2.0 limit).",
                details={"days_since_signup": days_since_signup},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message="User signed up in the last 2 days.",
            details={"days_since_signup": days_since_signup, "success_rule": goal.success},
        )

    # G04: trial_to_paid_7d (Subscription status is trial_active)
    if goal_id == "G04":
        if sub_status != "trial_active":
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="NOT_TRIAL_ACTIVE",
                message=f"User subscription status is '{sub_status}', requires 'trial_active'.",
                details={"subscription_status": sub_status},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message="User is on an active trial.",
        )

    # G05: first_paid_7d (Has never had a paid plan)
    if goal_id == "G05":
        if sub_status != "never":
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="NOT_NEVER_PAID",
                message=f"User subscription status is '{sub_status}', requires 'never'.",
                details={"subscription_status": sub_status},
            )
        paid_events = conn.execute(
            """
            SELECT count(*) FROM events
            WHERE user_id = ? AND event IN ('purchase_completed', 'subscription_renewed')
              AND received_at <= ?;
            """,
            (user_id, eval_iso),
        ).fetchone()[0]
        if paid_events > 0:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="PRIOR_PURCHASE_FOUND",
                message="User has prior purchase event records in event stream.",
                details={"paid_events": paid_events},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message="User has never had a paid subscription.",
        )

    # G06: reactivate_lapsed_14d (Subscription recently lapsed)
    if goal_id == "G06":
        if sub_status != "lapsed":
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="NOT_LAPSED",
                message=f"User subscription status is '{sub_status}', requires 'lapsed'.",
                details={"subscription_status": sub_status},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message="User subscription is lapsed.",
        )

    # G07: resume_series_3d (Started a show or micro-drama, has unwatched episodes left, not played for >= 3 days)
    if goal_id == "G07":
        series_played = conn.execute(
            """
            SELECT c.content_slug, c.content_type, max(e.received_at) as last_play
            FROM events e
            JOIN catalogue c ON e.title_id = c.title_id
            WHERE e.user_id = ? AND e.event = 'play_started' AND e.received_at <= ?
              AND c.content_type IN ('show', 'micro_drama')
            GROUP BY c.content_slug;
            """,
            (user_id, eval_iso),
        ).fetchall()

        if not series_played:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="NO_SERIES_STARTED",
                message="User has never started any show or micro-drama.",
            )

        eligible_series = []
        for s in series_played:
            slug = s["content_slug"]
            last_play_dt = parse_iso_datetime(s["last_play"])
            if not last_play_dt:
                continue
            hours_since_play = (eval_dt - last_play_dt).total_seconds() / 3600.0
            if hours_since_play < 72.0:
                continue  # Played within last 3 days

            # Check total episodes in catalogue
            total_eps = conn.execute(
                "SELECT count(*) FROM episodes WHERE content_slug = ?;", (slug,)
            ).fetchone()[0]

            # Check user watched episodes
            watched_eps = conn.execute(
                """
                SELECT count(DISTINCT episode_id) FROM events
                WHERE user_id = ? AND event = 'episode_completed' AND received_at <= ?
                  AND episode_id IN (SELECT episode_id FROM episodes WHERE content_slug = ?);
                """,
                (user_id, eval_iso, slug),
            ).fetchone()[0]

            if total_eps > watched_eps:
                eligible_series.append(
                    {
                        "content_slug": slug,
                        "hours_since_play": hours_since_play,
                        "unwatched_episodes": total_eps - watched_eps,
                    }
                )

        if not eligible_series:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="NO_UNFINISHED_SERIES_OVER_3D",
                message="No series found with unwatched episodes left that has been idle for >= 3 days.",
            )

        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message=f"Found {len(eligible_series)} eligible unfinished series.",
            details={"series": eligible_series},
        )

    # G08: try_new_release_7d (Active subscriber)
    if goal_id == "G08":
        if sub_status != "active":
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="NOT_ACTIVE_SUBSCRIBER",
                message=f"User subscription status is '{sub_status}', requires 'active'.",
                details={"subscription_status": sub_status},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message="User is an active subscriber.",
        )

    # G09: weekly_habit_14d (Active subscriber with plays on at most 1 day in the last 14 days)
    if goal_id == "G09":
        if sub_status != "active":
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="NOT_ACTIVE_SUBSCRIBER",
                message=f"User subscription status is '{sub_status}', requires 'active'.",
                details={"subscription_status": sub_status},
            )
        window_start = (eval_dt - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        play_events = conn.execute(
            """
            SELECT received_at FROM events
            WHERE user_id = ? AND event = 'play_started'
              AND received_at BETWEEN ? AND ?;
            """,
            (user_id, window_start, eval_iso),
        ).fetchall()

        distinct_ist_days = {
            ist_date_str(parse_iso_datetime(p["received_at"]))
            for p in play_events
            if parse_iso_datetime(p["received_at"])
        }

        if len(distinct_ist_days) > 1:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="TOO_MANY_PLAY_DAYS_IN_14D",
                message=f"User watched on {len(distinct_ist_days)} distinct IST days in last 14 days (max 1 allowed).",
                details={"play_days": sorted(list(distinct_ist_days))},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message=f"User played on {len(distinct_ist_days)} distinct days in last 14 days (<= 1).",
            details={"play_days": sorted(list(distinct_ist_days))},
        )

    # G12: return_after_absence_7d (No app_open in the 14 days before the trigger)
    if goal_id == "G12":
        window_start = (eval_dt - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        open_count = conn.execute(
            """
            SELECT count(*) FROM events
            WHERE user_id = ? AND event = 'app_open'
              AND received_at BETWEEN ? AND ?;
            """,
            (user_id, window_start, eval_iso),
        ).fetchone()[0]

        if open_count > 0:
            return EligibilityResult(
                goal_id=goal_id,
                eligible=False,
                reason_code="RECENT_APP_OPEN_FOUND",
                message=f"User opened app {open_count} times in the last 14 days.",
                details={"recent_app_opens": open_count},
            )
        return EligibilityResult(
            goal_id=goal_id,
            eligible=True,
            reason_code="ELIGIBLE",
            message="No app_open events in the 14 days prior to evaluation.",
        )

    return EligibilityResult(
        goal_id=goal_id,
        eligible=False,
        reason_code="UNHANDLED_GOAL",
        message=f"Goal {goal_id} does not have an automated rule implementation.",
    )


def evaluate_all_goals(
    user_id: str,
    conn: sqlite3.Connection,
    at_time: datetime | str | None = None,
    registry: GoalRegistry | None = None,
) -> dict[str, EligibilityResult]:
    """Evaluate user eligibility across all registered candidate goals."""
    reg = registry or get_default_registry()
    results: dict[str, EligibilityResult] = {}
    for goal_id in sorted(reg.goals.keys()):
        results[goal_id] = evaluate_goal_eligibility(
            user_id=user_id,
            goal_id=goal_id,
            conn=conn,
            at_time=at_time,
            registry=reg,
        )
    return results
