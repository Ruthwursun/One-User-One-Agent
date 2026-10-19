"""STAGE One User, One Agent Platform."""

from stage_agent.config import (
    DATA_PACK_DIR,
    DB_PATH,
    IST,
    UTC,
    get_db_path,
    is_quiet_hours_ist,
    ist_date_str,
    now_utc_iso,
    parse_iso_datetime,
    to_ist,
)
from stage_agent.db import (
    IngestionResult,
    get_db_connection,
    init_db,
    ingest_catalogue,
    ingest_episodes,
    ingest_events,
    ingest_profiles,
)
from stage_agent.eligibility import (
    GoalRegistry,
    evaluate_all_goals,
    evaluate_goal_eligibility,
    get_default_registry,
)
from stage_agent.model_provider import (
    MockModelProvider,
    MockModelTimeoutError,
    ModelCallLimitExceeded,
    ModelUsage,
    ToolCallLimitExceeded,
)
from stage_agent.models import (
    EligibilityResult,
    GoalDefinition,
    PolicyDecision,
    ProposedAction,
)
from stage_agent.planner import DecisionPlanner
from stage_agent.policy import PolicyGate

__version__ = "0.1.0"

__all__ = [
    "DATA_PACK_DIR",
    "DB_PATH",
    "DecisionPlanner",
    "EligibilityResult",
    "GoalDefinition",
    "GoalRegistry",
    "IST",
    "IngestionResult",
    "MockModelProvider",
    "MockModelTimeoutError",
    "ModelCallLimitExceeded",
    "ModelUsage",
    "PolicyDecision",
    "PolicyGate",
    "ProposedAction",
    "ToolCallLimitExceeded",
    "UTC",
    "__version__",
    "evaluate_all_goals",
    "evaluate_goal_eligibility",
    "get_db_connection",
    "get_db_path",
    "get_default_registry",
    "ingest_catalogue",
    "ingest_episodes",
    "ingest_events",
    "ingest_profiles",
    "init_db",
    "is_quiet_hours_ist",
    "ist_date_str",
    "now_utc_iso",
    "parse_iso_datetime",
    "to_ist",
]
