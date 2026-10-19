"""Deterministic local mock strategy and model provider with injected fault reproduction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stage_agent.config import DATA_PACK_DIR, parse_iso_datetime
from datetime import datetime


class ModelCallLimitExceeded(Exception):
    """Raised when model call count exceeds configured maximum."""


class ToolCallLimitExceeded(Exception):
    """Raised when tool call count exceeds configured maximum."""


class MockModelTimeoutError(Exception):
    """Raised when an injected timeout fault occurs."""


class ModelUnavailableError(Exception):
    """Raised when model provider dependency is unavailable (e.g. HTTP 503)."""


@dataclass
class ModelUsage:
    """Usage counters for model and tool calls during a decision."""

    model_calls: int = 0
    tool_calls: int = 0
    max_model_calls: int = 8
    max_tool_calls: int = 20

    def record_model_call(self) -> None:
        self.model_calls += 1
        if self.model_calls > self.max_model_calls:
            raise ModelCallLimitExceeded(
                f"Model call limit exceeded ({self.model_calls} > {self.max_model_calls})"
            )

    def record_tool_call(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            raise ToolCallLimitExceeded(
                f"Tool call limit exceeded ({self.tool_calls} > {self.max_tool_calls})"
            )


@dataclass
class ModelSuggestion:
    """Structured candidate suggestion emitted by the model provider."""

    suggested_goal_id: str | None
    suggested_action_type: str | None  # 'home_page', 'push', 'whatsapp', 'no_action'
    confidence: float
    reasoning: str
    claims: dict[str, Any] = field(default_factory=dict)
    is_fault: bool = False
    fault_type: str | None = None
    raw_output: Any = None


class MockModelProvider:
    """Deterministic local mock model provider reproducing injected faults and generating strategy."""

    def __init__(self, faults_path: str | Path | None = None) -> None:
        self.faults_by_trigger: dict[str, dict[str, Any]] = {}
        self.outage_windows: list[tuple[datetime, datetime]] = []
        target_path = Path(faults_path) if faults_path else (DATA_PACK_DIR / "replay" / "model_faults.jsonl")
        if target_path.is_file():
            self._load_faults(target_path)

    def add_outage_window(self, start: datetime | str, until: datetime | str) -> None:
        """Register an operational outage window for the model provider."""
        start_dt = parse_iso_datetime(start) if isinstance(start, str) else start
        until_dt = parse_iso_datetime(until) if isinstance(until, str) else until
        if start_dt and until_dt:
            self.outage_windows.append((start_dt, until_dt))

    def is_in_outage(self, at_time: datetime | str | None) -> bool:
        """Check if given timestamp falls in an active outage window."""
        if not at_time:
            return False
        dt = parse_iso_datetime(at_time) if isinstance(at_time, str) else at_time
        if not dt:
            return False
        for start_dt, until_dt in self.outage_windows:
            if start_dt <= dt <= until_dt:
                return True
        return False

    def _load_faults(self, path: Path) -> None:
        """Load scripted model faults from model_faults.jsonl."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    item = json.loads(line_str)
                    trig_id = item.get("trigger_event_id")
                    if trig_id:
                        self.faults_by_trigger[trig_id] = item

    def suggest_strategy(
        self,
        trigger_event_id: str,
        user_id: str,
        eligible_goals: list[str],
        profile_data: dict[str, Any],
        usage: ModelUsage | None = None,
        at_time: datetime | str | None = None,
    ) -> ModelSuggestion:
        """Generate strategy suggestion or reproduce scripted model fault for the given trigger."""
        if usage:
            usage.record_model_call()

        # Check operational outage (e.g. HTTP 503)
        if self.is_in_outage(at_time):
            raise ModelUnavailableError("Model provider unavailable (HTTP 503)")

        # Check for scripted model faults on this trigger
        fault = self.faults_by_trigger.get(trigger_event_id)
        if fault:
            f_type = fault.get("fault")
            if f_type == "timeout":
                raise MockModelTimeoutError(f"Model call timed out after {fault.get('seconds', 45)}s")

            if f_type == "malformed_output":
                return ModelSuggestion(
                    suggested_goal_id=None,
                    suggested_action_type=None,
                    confidence=0.0,
                    reasoning="Malformed model output",
                    is_fault=True,
                    fault_type="malformed_output",
                    raw_output=fault.get("model_output"),
                )

            if f_type == "unsupported_claim":
                out = fault.get("model_output", {})
                return ModelSuggestion(
                    suggested_goal_id=out.get("goal_id"),
                    suggested_action_type="push",
                    confidence=float(out.get("confidence", 0.95)),
                    reasoning=out.get("reason", "Model assertion"),
                    claims={"asserted_eligible": out.get("eligible", True)},
                    is_fault=True,
                    fault_type="unsupported_claim",
                    raw_output=out,
                )

        # Deterministic heuristic priority ranking
        priority_order = [
            ("G04", "push"),         # trial_to_paid -> push notification
            ("G01", "home_page"),    # first_play -> personalised home page
            ("G03", "home_page"),    # second_day_return -> home page
            ("G07", "home_page"),    # resume_series -> home page with next_episode
            ("G02", "home_page"),    # first_full_episode -> home page
            ("G06", "push"),         # reactivate_lapsed -> push discount/plan
            ("G08", "home_page"),    # try_new_release -> home page
            ("G09", "home_page"),    # weekly_habit -> home page
            ("G05", "push"),         # first_paid -> push
            ("G12", "push"),         # return_after_absence -> push
        ]

        for gid, act_type in priority_order:
            if gid in eligible_goals:
                return ModelSuggestion(
                    suggested_goal_id=gid,
                    suggested_action_type=act_type,
                    confidence=0.90,
                    reasoning=f"Heuristic priority selected eligible goal {gid}.",
                    is_fault=False,
                )

        # Fallback if no goals are eligible
        return ModelSuggestion(
            suggested_goal_id=None,
            suggested_action_type="no_action",
            confidence=1.0,
            reasoning="No eligible goals found for user.",
            is_fault=False,
        )
