"""Configuration and environment paths for STAGE One User, One Agent."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Base directories
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PACK_DIR = WORKSPACE_ROOT / "one_user_one_agent_pack"
DEFAULT_DB_DIR = WORKSPACE_ROOT / "data_store"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "stage_agent.db"

# Environment overrides
DATA_PACK_DIR = Path(os.environ.get("STAGE_DATA_PACK_DIR", str(DEFAULT_DATA_PACK_DIR)))
DB_PATH = Path(os.environ.get("STAGE_DB_PATH", str(DEFAULT_DB_PATH)))

# Timezones
UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))

# Operational thresholds
OFFLINE_BUFFER_THRESHOLD_SECONDS = 600.0  # Events buffered > 10 min are flagged as offline
DEFAULT_BATCH_SIZE = 5000


def get_db_path(override_path: str | Path | None = None) -> Path:
    """Resolve database path with fallback to configured default."""
    if override_path is not None:
        return Path(override_path)
    return DB_PATH


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse ISO-8601 string to a timezone-aware UTC datetime."""
    if not value or not isinstance(value, str):
        return None
    val = value.strip()
    if not val:
        return None
    try:
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except (ValueError, TypeError):
        return None


def now_utc_iso() -> str:
    """Return current timestamp in ISO-8601 UTC string format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def to_ist(dt: datetime) -> datetime:
    """Convert any datetime to timezone-aware IST."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(IST)


def ist_date_str(dt: datetime) -> str:
    """Return 'YYYY-MM-DD' calendar date in IST."""
    return to_ist(dt).strftime("%Y-%m-%d")


def is_quiet_hours_ist(dt: datetime) -> bool:
    """Return True if timestamp is during quiet hours (22:00 to 08:00 IST).

    In IST:
    22:00:00 to 23:59:59 -> True (Quiet)
    00:00:00 to 07:59:59 -> True (Quiet)
    08:00:00 to 21:59:59 -> False (Allowed daytime)
    """
    h = to_ist(dt).hour
    return h >= 22 or h < 8


# Goal commitment windows in hours
GOAL_WINDOWS_HOURS: dict[str, int] = {
    "G01": 24,   # first_play_24h
    "G02": 72,   # first_full_episode_3d
    "G03": 48,   # second_day_return_2d
    "G04": 168,  # trial_to_paid_7d
    "G05": 168,  # first_paid_7d
    "G06": 336,  # reactivate_lapsed_14d
    "G07": 72,   # resume_series_3d
    "G08": 168,  # try_new_release_7d
    "G09": 336,  # weekly_habit_14d
    "G10": 168,  # meaningful_engagement_7d (underspecified)
    "G11": 336,  # tv_activation_14d (signal unavailable)
    "G12": 168,  # return_after_absence_7d
}

