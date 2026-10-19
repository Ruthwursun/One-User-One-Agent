"""Deterministic decision planner and intervention synthesizer."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from stage_agent.config import (
    GOAL_WINDOWS_HOURS,
    OFFLINE_BUFFER_THRESHOLD_SECONDS,
    UTC,
    parse_iso_datetime,
)
from stage_agent.eligibility import (
    GoalRegistry,
    evaluate_all_goals,
    get_default_registry,
)
from stage_agent.model_provider import (
    MockModelProvider,
    MockModelTimeoutError,
    ModelUnavailableError,
    ModelUsage,
)
from stage_agent.models import PolicyDecision, ProposedAction
from stage_agent.policy import PolicyGate


class DecisionPlanner:
    """Orchestrates goal selection, policy validation, and intervention decisions for each viewer trigger."""

    def __init__(
        self,
        policy_gate: PolicyGate | None = None,
        model_provider: MockModelProvider | None = None,
        goal_registry: GoalRegistry | None = None,
    ) -> None:
        self.policy_gate = policy_gate or PolicyGate()
        self.model_provider = model_provider or MockModelProvider()
        self.goal_registry = goal_registry or get_default_registry()
        self.decision_counter = 0
        self.action_counter = 0

    def decide(
        self,
        trigger: dict[str, Any],
        conn: sqlite3.Connection,
        arms_map: dict[str, str] | None = None,
        sent_today_cache: set[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Produce a structured decision for a viewer trigger event.

        Enforces:
        - Holdout isolation.
        - Idempotency and offline-buffer rejection.
        - Deterministic goal eligibility.
        - Untrusted model output validation.
        - Single active goal and single active plan per viewer.
        - Safety policy compliance before publishing any action.
        """
        start_time = time.perf_counter()
        self.decision_counter += 1
        decision_id = f"D-{self.decision_counter:06d}"

        user_id = trigger.get("user_id")
        trigger_event_id = trigger.get("event_id") or f"trig-{self.decision_counter}"
        deliver_at_str = trigger.get("deliver_at") or trigger.get("received_at") or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        deliver_at_dt = parse_iso_datetime(deliver_at_str) or datetime.now(UTC)

        usage = ModelUsage()

        # Gate 0: Holdout Arm Check
        usage.record_tool_call()
        is_holdout = self.policy_gate.is_holdout_viewer(user_id, conn, arms_map)
        if is_holdout:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "decision_id": decision_id,
                "user_id": user_id,
                "trigger_event_id": trigger_event_id,
                "timestamp": deliver_at_str,
                "arm": "holdout",
                "active_goal_id": None,
                "decision_type": "no_action",
                "action": None,
                "reason_code": "HOLDOUT_ARM",
                "rationale": f"Viewer {user_id} is in the randomized holdout arm; no interventions may be delivered.",
                "evidence": {"arm": "holdout"},
                "policy_decision": {"allowed": False, "reason_code": "HOLDOUT_ARM"},
                "diagnostics": {
                    "model_calls": usage.model_calls,
                    "tool_calls": usage.tool_calls,
                    "processing_time_ms": round(elapsed_ms, 2),
                },
            }

        # Gate 1: Offline Buffer Check
        occurred_at_str = trigger.get("occurred_at")
        sent_at_str = trigger.get("sent_at")
        if occurred_at_str and sent_at_str:
            dt_occ = parse_iso_datetime(occurred_at_str)
            dt_sent = parse_iso_datetime(sent_at_str)
            if dt_occ and dt_sent and (dt_sent - dt_occ).total_seconds() > OFFLINE_BUFFER_THRESHOLD_SECONDS:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "decision_id": decision_id,
                    "user_id": user_id,
                    "trigger_event_id": trigger_event_id,
                    "timestamp": deliver_at_str,
                    "arm": "agent",
                    "active_goal_id": None,
                    "decision_type": "no_action",
                    "action": None,
                    "reason_code": "OFFLINE_BUFFERED_SESSION",
                    "rationale": "Trigger is a historical offline buffer (> 600s delay); dropped per simulator rules.",
                    "evidence": {"delay_seconds": (dt_sent - dt_occ).total_seconds()},
                    "policy_decision": {"allowed": False, "reason_code": "OFFLINE_BUFFERED_SESSION"},
                    "diagnostics": {
                        "model_calls": usage.model_calls,
                        "tool_calls": usage.tool_calls,
                        "processing_time_ms": round(elapsed_ms, 2),
                    },
                }

        # Gate 2: Fetch Profile Evidence from SQLite
        usage.record_tool_call()
        prof_row = conn.execute(
            """
            SELECT user_id, signup_at, subscription_status, preferred_dialect,
                   notifications_opt_in, whatsapp_opt_in
            FROM profiles WHERE user_id = ?;
            """,
            (user_id,),
        ).fetchone()

        if not prof_row:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "decision_id": decision_id,
                "user_id": user_id,
                "trigger_event_id": trigger_event_id,
                "timestamp": deliver_at_str,
                "arm": "agent",
                "active_goal_id": None,
                "decision_type": "no_action",
                "action": None,
                "reason_code": "MISSING_VIEWER_EVIDENCE",
                "rationale": f"No profile record found in database for user {user_id}.",
                "evidence": {},
                "policy_decision": {"allowed": False, "reason_code": "MISSING_VIEWER_EVIDENCE"},
                "diagnostics": {
                    "model_calls": usage.model_calls,
                    "tool_calls": usage.tool_calls,
                    "processing_time_ms": round(elapsed_ms, 2),
                },
            }

        profile_data = dict(prof_row)
        app_dialect = trigger.get("app_dialect") or profile_data.get("preferred_dialect")

        # Gate 3: Evaluate Goal Eligibility
        usage.record_tool_call()
        all_evals = evaluate_all_goals(
            user_id=user_id,
            conn=conn,
            at_time=deliver_at_dt,
            registry=self.goal_registry,
        )
        eligible_goals = [gid for gid, r in all_evals.items() if r.eligible]

        if not eligible_goals:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "decision_id": decision_id,
                "user_id": user_id,
                "trigger_event_id": trigger_event_id,
                "timestamp": deliver_at_str,
                "arm": "agent",
                "active_goal_id": None,
                "decision_type": "no_action",
                "action": None,
                "reason_code": "NO_ELIGIBLE_GOALS",
                "rationale": "Viewer is not currently eligible for any candidate goals.",
                "evidence": {
                    "subscription_status": profile_data.get("subscription_status"),
                    "evaluated_goals_count": len(all_evals),
                },
                "policy_decision": {"allowed": False, "reason_code": "NO_ELIGIBLE_GOALS"},
                "diagnostics": {
                    "model_calls": usage.model_calls,
                    "tool_calls": usage.tool_calls,
                    "processing_time_ms": round(elapsed_ms, 2),
                },
            }

        # Gate 4: Strategy Provider Consult (with Fault Resilience)
        unsupported_claims_detected = []
        try:
            suggestion = self.model_provider.suggest_strategy(
                trigger_event_id=trigger_event_id,
                user_id=user_id,
                eligible_goals=eligible_goals,
                profile_data=profile_data,
                usage=usage,
                at_time=deliver_at_dt,
            )
        except MockModelTimeoutError:
            # Model timeout fault: fall back safely to highest priority verified eligible goal
            suggestion = None
            unsupported_claims_detected.append("model_timeout_fallback")
        except ModelUnavailableError:
            # Model provider outage (e.g. HTTP 503): fall back safely to deterministic rules
            suggestion = None
            unsupported_claims_detected.append("model_outage_503_fallback")

        selected_goal_id: str | None = None
        action_type = "home_page"

        if suggestion and not suggestion.is_fault:
            if suggestion.suggested_goal_id in eligible_goals:
                selected_goal_id = suggestion.suggested_goal_id
                action_type = suggestion.suggested_action_type or "home_page"
            else:
                unsupported_claims_detected.append("hallucinated_ineligible_goal")

        elif suggestion and suggestion.is_fault:
            if suggestion.fault_type == "unsupported_claim":
                # Detect hallucinated claim (e.g. Model claiming G04 eligibility for non-trial user)
                claimed_goal = suggestion.suggested_goal_id
                if claimed_goal not in eligible_goals:
                    unsupported_claims_detected.append(
                        f"rejected_unsupported_claim_{claimed_goal}"
                    )

        # Fallback to deterministic priority among verified eligible goals
        if not selected_goal_id:
            selected_goal_id = eligible_goals[0]
            if selected_goal_id in ("G04", "G05", "G06", "G12"):
                action_type = "push"
            else:
                action_type = "home_page"

        # Gate 5: Synthesize Candidate Action
        published_at_dt = deliver_at_dt  # Add measured delay in final published string
        proposed_act = self._synthesize_action(
            user_id=user_id,
            action_type=action_type,
            app_dialect=app_dialect,
            goal_id=selected_goal_id,
            conn=conn,
            at_time=deliver_at_dt,
            usage=usage,
        )

        if not proposed_act:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "decision_id": decision_id,
                "user_id": user_id,
                "trigger_event_id": trigger_event_id,
                "timestamp": deliver_at_str,
                "arm": "agent",
                "active_goal_id": selected_goal_id,
                "decision_type": "no_action",
                "action": None,
                "reason_code": "INSUFFICIENT_CONTENT_EVIDENCE",
                "rationale": f"Could not construct a compliant {action_type} for dialect {app_dialect}.",
                "evidence": {"eligible_goals": eligible_goals, "selected_goal": selected_goal_id},
                "policy_decision": {"allowed": False, "reason_code": "INSUFFICIENT_CONTENT_EVIDENCE"},
                "diagnostics": {
                    "model_calls": usage.model_calls,
                    "tool_calls": usage.tool_calls,
                    "processing_time_ms": round(elapsed_ms, 2),
                },
            }

        # Gate 6: Safety Policy Gate
        usage.record_tool_call()
        policy_decision = self.policy_gate.validate_action(
            action=proposed_act,
            conn=conn,
            arms_map=arms_map,
            sent_today_cache=sent_today_cache,
            trigger_app_dialect=trigger.get("app_dialect"),
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        published_at_dt = deliver_at_dt + timedelta(milliseconds=elapsed_ms)
        published_at_str = published_at_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        if not policy_decision.allowed:
            return {
                "decision_id": decision_id,
                "user_id": user_id,
                "trigger_event_id": trigger_event_id,
                "timestamp": deliver_at_str,
                "arm": "agent",
                "active_goal_id": selected_goal_id,
                "decision_type": "no_action",
                "action": None,
                "reason_code": policy_decision.reason_code,
                "rationale": f"Policy blocked proposed action: {policy_decision.message}",
                "evidence": {
                    "eligible_goals": eligible_goals,
                    "selected_goal": selected_goal_id,
                    "proposed_action_type": action_type,
                },
                "policy_decision": policy_decision.to_dict(),
                "diagnostics": {
                    "model_calls": usage.model_calls,
                    "tool_calls": usage.tool_calls,
                    "processing_time_ms": round(elapsed_ms, 2),
                    "unsupported_claims_detected": unsupported_claims_detected,
                },
            }

        # Action Approved -> Package output payload
        self.action_counter += 1
        action_id = f"A-{self.action_counter:06d}"

        action_dict: dict[str, Any] = {
            "action_id": action_id,
            "decision_id": decision_id,
            "user_id": user_id,
            "trigger_event_id": trigger_event_id,
            "type": action_type,
            "published_at": published_at_str,
        }

        if action_type == "home_page":
            action_dict["dialect"] = proposed_act.dialect
            if proposed_act.hero_title_id:
                action_dict["hero_title_id"] = proposed_act.hero_title_id
            action_dict["rows"] = proposed_act.rows
        elif action_type == "push":
            action_dict["send_at"] = published_at_str
            action_dict["title_id"] = proposed_act.title_id
            action_dict["text"] = proposed_act.text
            if proposed_act.deeplink:
                action_dict["deeplink"] = proposed_act.deeplink
        elif action_type == "whatsapp":
            action_dict["send_at"] = published_at_str
            action_dict["template_id"] = proposed_act.template_id
            action_dict["variables"] = proposed_act.variables
            action_dict["title_id"] = proposed_act.title_id

        # Record in sent cache if message
        if action_type in ("push", "whatsapp") and sent_today_cache is not None:
            sent_today_cache.add((user_id, deliver_at_str[:10]))

        # Update durable viewer state in SQLite
        usage.record_tool_call()
        self._update_user_state(
            user_id=user_id,
            goal_id=selected_goal_id,
            action_dict=action_dict,
            at_iso=published_at_str,
            dialect=app_dialect,
            conn=conn,
            subscription_status=profile_data.get("subscription_status"),
            arm="agent",
        )

        return {
            "decision_id": decision_id,
            "user_id": user_id,
            "trigger_event_id": trigger_event_id,
            "timestamp": deliver_at_str,
            "arm": "agent",
            "active_goal_id": selected_goal_id,
            "decision_type": "action",
            "action": action_dict,
            "reason_code": "ALLOWED",
            "rationale": f"Published {action_type} intervention advancing goal {selected_goal_id}.",
            "evidence": {
                "eligible_goals": eligible_goals,
                "selected_goal": selected_goal_id,
                "dialect": proposed_act.dialect,
            },
            "policy_decision": policy_decision.to_dict(),
            "diagnostics": {
                "model_calls": usage.model_calls,
                "tool_calls": usage.tool_calls,
                "processing_time_ms": round(elapsed_ms, 2),
                "unsupported_claims_detected": unsupported_claims_detected,
            },
        }

    def _synthesize_action(
        self,
        user_id: str,
        action_type: str,
        app_dialect: str | None,
        goal_id: str,
        conn: sqlite3.Connection,
        at_time: datetime,
        usage: ModelUsage,
    ) -> ProposedAction | None:
        """Construct compliant payload for the candidate action type."""
        eval_iso = at_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        target_dialect = app_dialect or "d1"

        if action_type == "home_page":
            usage.record_tool_call()
            # Fetch available titles in dialect, excluding withdrawn titles
            titles = conn.execute(
                """
                SELECT title_id, content_slug, title, genres, release_date
                FROM catalogue
                WHERE dialect = ? AND available_from <= ?
                  AND (available_until IS NULL OR available_until > ?)
                ORDER BY release_date DESC;
                """,
                (target_dialect, eval_iso, eval_iso),
            ).fetchall()

            # Filter out operationally withdrawn titles
            valid_titles = [
                t for t in titles
                if not (self.policy_gate.withdrawn_titles.get(t["title_id"])
                        and at_time >= self.policy_gate.withdrawn_titles[t["title_id"]])
            ]

            if len(valid_titles) < 3:
                return None  # Cannot make a valid row

            hero_id = valid_titles[0]["title_id"]
            seen_slugs = {valid_titles[0]["content_slug"]}

            # Construct up to 4 rows ensuring slug uniqueness across the page
            rows = []
            row_types = ["popular_in_dialect", "new_releases", "genre", "because_you_watched"]

            curr_idx = 1
            for r_type in row_types:
                row_tids = []
                while curr_idx < len(valid_titles) and len(row_tids) < 4:
                    t = valid_titles[curr_idx]
                    curr_idx += 1
                    if t["content_slug"] not in seen_slugs:
                        seen_slugs.add(t["content_slug"])
                        row_tids.append(t["title_id"])

                if len(row_tids) >= 3:
                    rows.append({"row_type": r_type, "title_ids": row_tids})

            if not rows and not hero_id:
                return None

            return ProposedAction(
                user_id=user_id,
                type="home_page",
                published_at=eval_iso,
                dialect=target_dialect,
                hero_title_id=hero_id,
                rows=rows,
            )

        elif action_type == "push":
            usage.record_tool_call()
            title_row = conn.execute(
                """
                SELECT title_id, title FROM catalogue
                WHERE dialect = ? AND available_from <= ?
                  AND (available_until IS NULL OR available_until > ?)
                ORDER BY release_date DESC LIMIT 1;
                """,
                (target_dialect, eval_iso, eval_iso),
            ).fetchone()

            if not title_row:
                return None

            tid = title_row["title_id"]
            tname = title_row["title"]

            return ProposedAction(
                user_id=user_id,
                type="push",
                send_at=eval_iso,
                title_id=tid,
                text={
                    "title": "Aapke Liye Naya Content",
                    "body": f"{tname} abhi dekhein apne app par.",
                },
                deeplink=f"ott://title/{tid}",
                dialect=target_dialect,
            )

        elif action_type == "whatsapp":
            usage.record_tool_call()
            title_row = conn.execute(
                """
                SELECT title_id, title FROM catalogue
                WHERE dialect = ? AND available_from <= ?
                  AND (available_until IS NULL OR available_until > ?)
                ORDER BY release_date DESC LIMIT 1;
                """,
                (target_dialect, eval_iso, eval_iso),
            ).fetchone()

            if not title_row:
                return None

            tid = title_row["title_id"]
            tname = title_row["title"]

            return ProposedAction(
                user_id=user_id,
                type="whatsapp",
                send_at=eval_iso,
                template_id="wa_recommend_title_v1",
                variables={
                    "title_name": tname,
                    "dialect_name": target_dialect,
                },
                title_id=tid,
                dialect=target_dialect,
            )

        return None

    def _update_user_state(
        self,
        user_id: str,
        goal_id: str | None,
        action_dict: dict[str, Any] | None,
        at_iso: str,
        dialect: str | None,
        conn: sqlite3.Connection,
        subscription_status: str | None = None,
        arm: str = "agent",
    ) -> None:
        """Update durable viewer state machine in user_state table."""
        agent_owner = f"agent-{user_id}"
        plan_json = json.dumps(action_dict, ensure_ascii=False) if action_dict else None
        goal_deadline_iso = None
        if goal_id:
            w_hours = GOAL_WINDOWS_HOURS.get(goal_id, 72)
            c_dt = parse_iso_datetime(at_iso)
            if c_dt:
                goal_deadline_iso = (c_dt + timedelta(hours=w_hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        with conn:
            conn.execute(
                """
                INSERT INTO user_state (
                    user_id, agent_owner, arm, subscription_status, active_goal_id,
                    goal_committed_at, goal_deadline_at, goal_status, goal_judgment,
                    goal_judgment_at, goal_learned, goal_evidence_events,
                    active_plan_json, last_event_at, last_app_dialect, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'in_progress', NULL, NULL, NULL, NULL, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    agent_owner = excluded.agent_owner,
                    arm = excluded.arm,
                    subscription_status = COALESCE(excluded.subscription_status, user_state.subscription_status),
                    active_goal_id = excluded.active_goal_id,
                    goal_committed_at = excluded.goal_committed_at,
                    goal_deadline_at = excluded.goal_deadline_at,
                    goal_status = 'in_progress',
                    goal_judgment = NULL,
                    goal_judgment_at = NULL,
                    goal_learned = NULL,
                    goal_evidence_events = NULL,
                    active_plan_json = excluded.active_plan_json,
                    last_event_at = excluded.last_event_at,
                    last_app_dialect = excluded.last_app_dialect,
                    updated_at = excluded.updated_at;
                """,
                (
                    user_id,
                    agent_owner,
                    arm,
                    subscription_status,
                    goal_id,
                    at_iso,
                    goal_deadline_iso,
                    plan_json,
                    at_iso,
                    dialect,
                    at_iso,
                ),
            )
