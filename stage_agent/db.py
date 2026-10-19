"""SQLite database initialization, schema definitions, and robust ingestion pipeline."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stage_agent.config import (
    DEFAULT_BATCH_SIZE,
    OFFLINE_BUFFER_THRESHOLD_SECONDS,
    get_db_path,
    now_utc_iso,
    parse_iso_datetime,
)


@dataclass
class IngestionResult:
    """Detailed summary of an ingestion run."""

    source_file: str
    records_read: int = 0
    records_inserted: int = 0
    duplicates_skipped: int = 0
    records_updated: int = 0
    malformed_count: int = 0
    malformed_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Database Connection and Initialization
# ---------------------------------------------------------------------------


def get_db_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Create a configured SQLite connection with foreign keys and WAL mode enabled."""
    target_path = get_db_path(db_path)
    is_memory = str(target_path) == ":memory:"

    if not is_memory:
        target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_path), timeout=60.0)
    conn.row_factory = sqlite3.Row

    if not is_memory:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")

    return conn


def init_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Initialize SQLite tables and indexes for the STAGE agent platform."""
    conn = get_db_connection(db_path)

    ddl_statements = [
        # 1. Events Table (Append-only / Immutable Raw Event Stream)
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            event TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            sent_at TEXT NOT NULL,
            received_at TEXT NOT NULL,
            session_id TEXT,
            app_dialect TEXT,
            app_language TEXT,
            platform TEXT,
            app_version TEXT,
            properties_json TEXT NOT NULL,
            title_id TEXT,
            episode_id TEXT,
            plan_id TEXT,
            price_inr REAL,
            channel TEXT,
            is_offline_buffered INTEGER NOT NULL DEFAULT 0,
            ingested_at TEXT NOT NULL
        );
        """,
        # 2. Raw Profile Snapshot History (Preserves Full Provenance & Conflicts)
        """
        CREATE TABLE IF NOT EXISTS profiles_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            signup_at TEXT,
            subscription_status TEXT NOT NULL,
            plan_started_at TEXT,
            plan_ends_at TEXT,
            preferred_dialect TEXT,
            preferred_language TEXT,
            notifications_opt_in INTEGER NOT NULL,
            whatsapp_opt_in INTEGER NOT NULL,
            platform TEXT,
            updated_at TEXT NOT NULL,
            snapshot_at TEXT,
            ingested_at TEXT NOT NULL
        );
        """,
        # 3. Deduplicated Authoritative Profiles (Latest updated_at per user)
        """
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            signup_at TEXT,
            subscription_status TEXT NOT NULL,
            plan_started_at TEXT,
            plan_ends_at TEXT,
            preferred_dialect TEXT,
            preferred_language TEXT,
            notifications_opt_in INTEGER NOT NULL,
            whatsapp_opt_in INTEGER NOT NULL,
            platform TEXT,
            updated_at TEXT NOT NULL,
            snapshot_at TEXT,
            ingested_at TEXT NOT NULL
        );
        """,
        # 4. Catalogue Titles
        """
        CREATE TABLE IF NOT EXISTS catalogue (
            title_id TEXT PRIMARY KEY,
            content_slug TEXT NOT NULL,
            title TEXT NOT NULL,
            dialect TEXT NOT NULL,
            language TEXT NOT NULL,
            content_type TEXT NOT NULL,
            genres TEXT NOT NULL,
            release_date TEXT,
            available_from TEXT NOT NULL,
            available_until TEXT,
            is_premium INTEGER NOT NULL,
            description TEXT,
            ingested_at TEXT NOT NULL
        );
        """,
        # 5. Episodes
        """
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            content_slug TEXT NOT NULL,
            season INTEGER NOT NULL,
            episode INTEGER NOT NULL,
            minutes REAL NOT NULL,
            ingested_at TEXT NOT NULL
        );
        """,
        # 6. User State (Single Durable Owner Record per Viewer)
        """
        CREATE TABLE IF NOT EXISTS user_state (
            user_id TEXT PRIMARY KEY,
            agent_owner TEXT,
            arm TEXT,
            subscription_status TEXT,
            active_goal_id TEXT,
            goal_committed_at TEXT,
            goal_deadline_at TEXT,
            goal_status TEXT DEFAULT 'none',
            goal_judgment TEXT,
            goal_judgment_at TEXT,
            goal_learned TEXT,
            goal_evidence_events TEXT,
            active_plan_json TEXT,
            last_event_at TEXT,
            last_app_dialect TEXT,
            notifications_opt_in INTEGER DEFAULT 0,
            whatsapp_opt_in INTEGER DEFAULT 0,
            push_opt_out_at TEXT,
            whatsapp_opt_out_at TEXT,
            last_message_sent_at TEXT,
            messages_sent_today_ist INTEGER DEFAULT 0,
            last_message_ist_date TEXT,
            updated_at TEXT NOT NULL
        );
        """,
        # 7. Ingestion Audit Log
        """
        CREATE TABLE IF NOT EXISTS ingestion_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            records_read INTEGER NOT NULL,
            records_inserted INTEGER NOT NULL,
            duplicates_skipped INTEGER NOT NULL,
            records_updated INTEGER NOT NULL,
            malformed_count INTEGER NOT NULL,
            details_json TEXT,
            completed_at TEXT NOT NULL
        );
        """,
        # Performance Indexes
        "CREATE INDEX IF NOT EXISTS idx_events_user_received ON events(user_id, received_at);",
        "CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event);",
        "CREATE INDEX IF NOT EXISTS idx_events_user_event ON events(user_id, event, received_at);",
        "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);",
        "CREATE INDEX IF NOT EXISTS idx_events_title ON events(title_id);",
        "CREATE INDEX IF NOT EXISTS idx_profiles_raw_user ON profiles_raw(user_id, updated_at);",
        "CREATE INDEX IF NOT EXISTS idx_profiles_status ON profiles(subscription_status);",
        "CREATE INDEX IF NOT EXISTS idx_profiles_dialect ON profiles(preferred_dialect);",
        "CREATE INDEX IF NOT EXISTS idx_cat_dialect_avail ON catalogue(dialect, available_from, available_until);",
        "CREATE INDEX IF NOT EXISTS idx_cat_slug ON catalogue(content_slug);",
        "CREATE INDEX IF NOT EXISTS idx_cat_type ON catalogue(content_type);",
        "CREATE INDEX IF NOT EXISTS idx_episodes_slug_order ON episodes(content_slug, season, episode);",
        "CREATE INDEX IF NOT EXISTS idx_user_state_arm ON user_state(arm);",
        "CREATE INDEX IF NOT EXISTS idx_user_state_goal ON user_state(active_goal_id, goal_status);",
    ]

    with conn:
        for stmt in ddl_statements:
            conn.execute(stmt)

        # Ensure backwards-compatible columns on existing user_state tables
        try:
            conn.execute("ALTER TABLE user_state ADD COLUMN agent_owner TEXT;")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE user_state ADD COLUMN active_plan_json TEXT;")
        except sqlite3.OperationalError:
            pass
        for col_def in [
            "goal_judgment TEXT",
            "goal_judgment_at TEXT",
            "goal_learned TEXT",
            "goal_evidence_events TEXT",
        ]:
            try:
                conn.execute(f"ALTER TABLE user_state ADD COLUMN {col_def};")
            except sqlite3.OperationalError:
                pass

    return conn


# ---------------------------------------------------------------------------
# Ingestion Functions
# ---------------------------------------------------------------------------


def log_ingestion_audit(result: IngestionResult, conn: sqlite3.Connection) -> None:
    """Persist ingestion audit record into ingestion_audit table."""
    details = {
        "malformed_errors_sample": result.malformed_errors[:10],
    }
    with conn:
        conn.execute(
            """
            INSERT INTO ingestion_audit (
                source_file, records_read, records_inserted, duplicates_skipped,
                records_updated, malformed_count, details_json, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.source_file,
                result.records_read,
                result.records_inserted,
                result.duplicates_skipped,
                result.records_updated,
                result.malformed_count,
                json.dumps(details, ensure_ascii=False),
                now_utc_iso(),
            ),
        )


def ingest_events(
    file_path: str | Path,
    conn: sqlite3.Connection,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestionResult:
    """Ingest events from JSONL file into events table.

    Deduplicates by event_id using ON CONFLICT DO NOTHING.
    Identifies and flags offline-buffered events where (sent_at - occurred_at) > 600s.
    """
    path = Path(file_path)
    result = IngestionResult(source_file=str(path))
    if not path.is_file():
        raise FileNotFoundError(f"Events file not found: {path}")

    insert_sql = """
    INSERT INTO events (
        event_id, user_id, event, occurred_at, sent_at, received_at,
        session_id, app_dialect, app_language, platform, app_version,
        properties_json, title_id, episode_id, plan_id, price_inr, channel,
        is_offline_buffered, ingested_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(event_id) DO NOTHING
    """

    batch: list[tuple[Any, ...]] = []
    line_number = 0
    now_iso = now_utc_iso()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_number += 1
            line_str = line.strip()
            if not line_str:
                continue

            result.records_read += 1
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as exc:
                result.malformed_count += 1
                result.malformed_errors.append(f"Line {line_number}: Invalid JSON - {exc}")
                continue

            event_id = data.get("event_id")
            user_id = data.get("user_id")
            event_name = data.get("event")
            received_at = data.get("received_at")

            if not event_id or not user_id or not event_name or not received_at:
                result.malformed_count += 1
                result.malformed_errors.append(
                    f"Line {line_number}: Missing required field (event_id, user_id, event, received_at)"
                )
                continue

            occurred_at = data.get("occurred_at") or received_at
            sent_at = data.get("sent_at") or occurred_at

            # Calculate offline buffering
            is_offline = 0
            dt_occ = parse_iso_datetime(occurred_at)
            dt_sent = parse_iso_datetime(sent_at)
            if dt_occ and dt_sent:
                delay = (dt_sent - dt_occ).total_seconds()
                if delay > OFFLINE_BUFFER_THRESHOLD_SECONDS:
                    is_offline = 1

            props = data.get("properties") or {}
            props_json = json.dumps(props, ensure_ascii=False)

            title_id = props.get("title_id")
            episode_id = props.get("episode_id")
            plan_id = props.get("plan_id")
            price_inr = props.get("price_inr")
            try:
                price_inr = float(price_inr) if price_inr is not None else None
            except (ValueError, TypeError):
                price_inr = None
            channel = props.get("channel")

            batch.append(
                (
                    event_id,
                    user_id,
                    event_name,
                    occurred_at,
                    sent_at,
                    received_at,
                    data.get("session_id"),
                    data.get("app_dialect"),
                    data.get("app_language"),
                    data.get("platform"),
                    data.get("app_version"),
                    props_json,
                    title_id,
                    episode_id,
                    plan_id,
                    price_inr,
                    channel,
                    is_offline,
                    now_iso,
                )
            )

            if len(batch) >= batch_size:
                with conn:
                    cur = conn.executemany(insert_sql, batch)
                    inserted = cur.rowcount
                    result.records_inserted += inserted
                    result.duplicates_skipped += len(batch) - inserted
                batch.clear()

        if batch:
            with conn:
                cur = conn.executemany(insert_sql, batch)
                inserted = cur.rowcount
                result.records_inserted += inserted
                result.duplicates_skipped += len(batch) - inserted
            batch.clear()

    log_ingestion_audit(result, conn)
    return result


def ingest_profiles(
    file_path: str | Path,
    conn: sqlite3.Connection,
) -> IngestionResult:
    """Ingest user profiles snapshot CSV.

    Preserves every row in profiles_raw for provenance/conflict investigation.
    Maintains authoritative state in profiles table where latest updated_at wins.
    """
    path = Path(file_path)
    result = IngestionResult(source_file=str(path))
    if not path.is_file():
        raise FileNotFoundError(f"Profiles snapshot file not found: {path}")

    raw_insert_sql = """
    INSERT INTO profiles_raw (
        user_id, signup_at, subscription_status, plan_started_at, plan_ends_at,
        preferred_dialect, preferred_language, notifications_opt_in, whatsapp_opt_in,
        platform, updated_at, snapshot_at, ingested_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    dedup_upsert_sql = """
    INSERT INTO profiles (
        user_id, signup_at, subscription_status, plan_started_at, plan_ends_at,
        preferred_dialect, preferred_language, notifications_opt_in, whatsapp_opt_in,
        platform, updated_at, snapshot_at, ingested_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(user_id) DO UPDATE SET
        signup_at = excluded.signup_at,
        subscription_status = excluded.subscription_status,
        plan_started_at = excluded.plan_started_at,
        plan_ends_at = excluded.plan_ends_at,
        preferred_dialect = excluded.preferred_dialect,
        preferred_language = excluded.preferred_language,
        notifications_opt_in = excluded.notifications_opt_in,
        whatsapp_opt_in = excluded.whatsapp_opt_in,
        platform = excluded.platform,
        updated_at = excluded.updated_at,
        snapshot_at = excluded.snapshot_at,
        ingested_at = excluded.ingested_at
    WHERE excluded.updated_at > profiles.updated_at;
    """

    now_iso = now_utc_iso()
    raw_batch: list[tuple[Any, ...]] = []
    line_number = 1  # 1-based header

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_number += 1
            result.records_read += 1

            user_id = (row.get("user_id") or "").strip()
            status = (row.get("subscription_status") or "").strip()
            updated_at = (row.get("updated_at") or "").strip()

            if not user_id or not status or not updated_at:
                result.malformed_count += 1
                result.malformed_errors.append(
                    f"Line {line_number}: Missing required profile field (user_id, subscription_status, updated_at)"
                )
                continue

            signup_at = (row.get("signup_at") or "").strip() or None
            plan_started = (row.get("plan_started_at") or "").strip() or None
            plan_ends = (row.get("plan_ends_at") or "").strip() or None
            dialect = (row.get("preferred_dialect") or "").strip() or None
            lang = (row.get("preferred_language") or "").strip() or None
            platform = (row.get("platform") or "").strip() or None
            snapshot_at = (row.get("snapshot_at") or "").strip() or None

            notif_str = (row.get("notifications_opt_in") or "").strip().lower()
            notif_opt_in = 1 if notif_str in ("true", "1", "yes") else 0

            wa_str = (row.get("whatsapp_opt_in") or "").strip().lower()
            wa_opt_in = 1 if wa_str in ("true", "1", "yes") else 0

            record = (
                user_id,
                signup_at,
                status,
                plan_started,
                plan_ends,
                dialect,
                lang,
                notif_opt_in,
                wa_opt_in,
                platform,
                updated_at,
                snapshot_at,
                now_iso,
            )
            raw_batch.append(record)

    if raw_batch:
        with conn:
            # 1. Store full raw records for provenance
            conn.executemany(raw_insert_sql, raw_batch)

            # 2. Measure counts for authoritative profiles table
            pre_count = conn.execute("SELECT count(*) FROM profiles;").fetchone()[0]
            total_affected = 0
            for record in raw_batch:
                cur = conn.execute(dedup_upsert_sql, record)
                total_affected += cur.rowcount
            post_count = conn.execute("SELECT count(*) FROM profiles;").fetchone()[0]

            newly_inserted = post_count - pre_count
            updated_records = total_affected - newly_inserted
            duplicates_or_older = result.records_read - result.malformed_count - total_affected

            result.records_inserted = newly_inserted
            result.records_updated = updated_records
            result.duplicates_skipped = duplicates_or_older

    log_ingestion_audit(result, conn)
    return result


def ingest_catalogue(
    file_path: str | Path,
    conn: sqlite3.Connection,
) -> IngestionResult:
    """Ingest catalogue titles CSV into catalogue table."""
    path = Path(file_path)
    result = IngestionResult(source_file=str(path))
    if not path.is_file():
        raise FileNotFoundError(f"Catalogue file not found: {path}")

    upsert_sql = """
    INSERT INTO catalogue (
        title_id, content_slug, title, dialect, language, content_type,
        genres, release_date, available_from, available_until, is_premium,
        description, ingested_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(title_id) DO UPDATE SET
        content_slug = excluded.content_slug,
        title = excluded.title,
        dialect = excluded.dialect,
        language = excluded.language,
        content_type = excluded.content_type,
        genres = excluded.genres,
        release_date = excluded.release_date,
        available_from = excluded.available_from,
        available_until = excluded.available_until,
        is_premium = excluded.is_premium,
        description = excluded.description,
        ingested_at = excluded.ingested_at
    """

    now_iso = now_utc_iso()
    batch: list[tuple[Any, ...]] = []
    line_number = 1

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_number += 1
            result.records_read += 1

            title_id = (row.get("title_id") or "").strip()
            slug = (row.get("content_slug") or "").strip()
            title = (row.get("title") or "").strip()
            dialect = (row.get("dialect") or "").strip()
            available_from = (row.get("available_from") or "").strip()

            if not title_id or not slug or not title or not dialect or not available_from:
                result.malformed_count += 1
                result.malformed_errors.append(
                    f"Line {line_number}: Missing required catalogue field (title_id, slug, title, dialect, available_from)"
                )
                continue

            lang = (row.get("language") or "").strip() or "hin"
            content_type = (row.get("content_type") or "").strip() or "show"
            genres = (row.get("genres") or "").strip()
            release_date = (row.get("release_date") or "").strip() or None
            available_until = (row.get("available_until") or "").strip() or None

            prem_str = (row.get("is_premium") or "").strip().lower()
            is_premium = 1 if prem_str in ("true", "1", "yes") else 0
            desc = (row.get("description") or "").strip() or None

            batch.append(
                (
                    title_id,
                    slug,
                    title,
                    dialect,
                    lang,
                    content_type,
                    genres,
                    release_date,
                    available_from,
                    available_until,
                    is_premium,
                    desc,
                    now_iso,
                )
            )

    if batch:
        with conn:
            pre_count = conn.execute("SELECT count(*) FROM catalogue;").fetchone()[0]
            conn.executemany(upsert_sql, batch)
            post_count = conn.execute("SELECT count(*) FROM catalogue;").fetchone()[0]

            newly_inserted = post_count - pre_count
            result.records_inserted = newly_inserted
            result.records_updated = len(batch) - newly_inserted

    log_ingestion_audit(result, conn)
    return result


def ingest_episodes(
    file_path: str | Path,
    conn: sqlite3.Connection,
) -> IngestionResult:
    """Ingest episodes CSV into episodes table."""
    path = Path(file_path)
    result = IngestionResult(source_file=str(path))
    if not path.is_file():
        raise FileNotFoundError(f"Episodes file not found: {path}")

    upsert_sql = """
    INSERT INTO episodes (
        episode_id, content_slug, season, episode, minutes, ingested_at
    ) VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(episode_id) DO UPDATE SET
        content_slug = excluded.content_slug,
        season = excluded.season,
        episode = excluded.episode,
        minutes = excluded.minutes,
        ingested_at = excluded.ingested_at
    """

    now_iso = now_utc_iso()
    batch: list[tuple[Any, ...]] = []
    line_number = 1

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_number += 1
            result.records_read += 1

            episode_id = (row.get("episode_id") or "").strip()
            slug = (row.get("content_slug") or "").strip()
            season_str = (row.get("season") or "").strip()
            ep_str = (row.get("episode") or "").strip()
            mins_str = (row.get("minutes") or "").strip()

            if not episode_id or not slug:
                result.malformed_count += 1
                result.malformed_errors.append(f"Line {line_number}: Missing episode_id or content_slug")
                continue

            try:
                season = int(season_str)
                episode = int(ep_str)
                minutes = float(mins_str)
            except (ValueError, TypeError) as exc:
                result.malformed_count += 1
                result.malformed_errors.append(f"Line {line_number}: Invalid numeric field ({exc})")
                continue

            batch.append((episode_id, slug, season, episode, minutes, now_iso))

    if batch:
        with conn:
            pre_count = conn.execute("SELECT count(*) FROM episodes;").fetchone()[0]
            conn.executemany(upsert_sql, batch)
            post_count = conn.execute("SELECT count(*) FROM episodes;").fetchone()[0]

            newly_inserted = post_count - pre_count
            result.records_inserted = newly_inserted
            result.records_updated = len(batch) - newly_inserted

    log_ingestion_audit(result, conn)
    return result
