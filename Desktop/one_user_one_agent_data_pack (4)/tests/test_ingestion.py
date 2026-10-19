"""Unit tests for SQLite database initialization and ingestion pipelines."""

import json
import sqlite3
from pathlib import Path

import pytest

from stage_agent.db import (
    init_db,
    ingest_catalogue,
    ingest_episodes,
    ingest_events,
    ingest_profiles,
)


@pytest.fixture
def memory_db():
    """Provides a fresh in-memory database with tables initialized."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def file_db(tmp_path: Path):
    """Provides a fresh on-disk database in a temporary directory."""
    db_file = tmp_path / "test_stage_agent.db"
    conn = init_db(db_file)
    yield conn, db_file
    conn.close()


def test_database_initialization(memory_db: sqlite3.Connection):
    """Verify that all core tables and indexes are created successfully."""
    cursor = memory_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    )
    tables = {row[0] for row in cursor.fetchall()}

    expected_tables = {
        "events",
        "profiles_raw",
        "profiles",
        "catalogue",
        "episodes",
        "user_state",
        "ingestion_audit",
    }
    assert expected_tables.issubset(tables)

    # Check key indexes exist
    cursor = memory_db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';"
    )
    indexes = {row[0] for row in cursor.fetchall()}
    assert "idx_events_user_received" in indexes
    assert "idx_profiles_status" in indexes
    assert "idx_cat_dialect_avail" in indexes


def test_ingest_fixtures_success(memory_db: sqlite3.Connection, tmp_path: Path):
    """Verify standard ingestion of events, profiles, catalogue, and episodes."""
    # 1. Events fixture
    events_file = tmp_path / "sample_events.jsonl"
    events_data = [
        {
            "event_id": "ev-001",
            "user_id": "U-100",
            "event": "app_open",
            "occurred_at": "2026-09-01T10:00:00.000Z",
            "sent_at": "2026-09-01T10:00:02.000Z",
            "received_at": "2026-09-01T10:00:03.000Z",
            "session_id": "sess-01",
            "app_dialect": "d1",
            "app_language": "hin",
            "platform": "android",
            "app_version": "6.1.4",
            "properties": {"deeplink": "ott://home"},
        },
        {
            "event_id": "ev-002",
            "user_id": "U-100",
            "event": "play_started",
            "occurred_at": "2026-09-01T10:05:00.000Z",
            "sent_at": "2026-09-01T10:05:01.000Z",
            "received_at": "2026-09-01T10:05:02.000Z",
            "session_id": "sess-01",
            "app_dialect": "d1",
            "app_language": "hin",
            "platform": "android",
            "app_version": "6.1.4",
            "properties": {"title_id": "T0001", "episode_id": "EP-01"},
        },
    ]
    with open(events_file, "w", encoding="utf-8") as f:
        for item in events_data:
            f.write(json.dumps(item) + "\n")

    res_events = ingest_events(events_file, memory_db)
    assert res_events.records_read == 2
    assert res_events.records_inserted == 2
    assert res_events.duplicates_skipped == 0
    assert res_events.malformed_count == 0

    # 2. Profiles fixture
    profiles_file = tmp_path / "sample_profiles.csv"
    profiles_file.write_text(
        "user_id,signup_at,subscription_status,plan_started_at,plan_ends_at,"
        "preferred_dialect,preferred_language,notifications_opt_in,whatsapp_opt_in,"
        "platform,updated_at,snapshot_at\n"
        "U-100,2026-08-01T00:00:00.000Z,active,2026-08-01T00:00:00.000Z,2026-10-01T00:00:00.000Z,"
        "d1,hin,true,false,android,2026-09-01T00:00:00.000Z,2026-09-30T05:30:00.000Z\n",
        encoding="utf-8",
    )
    res_profiles = ingest_profiles(profiles_file, memory_db)
    assert res_profiles.records_read == 1
    assert res_profiles.records_inserted == 1
    assert res_profiles.duplicates_skipped == 0

    # 3. Catalogue fixture
    cat_file = tmp_path / "sample_catalogue.csv"
    cat_file.write_text(
        "title_id,content_slug,title,dialect,language,content_type,genres,"
        "release_date,available_from,available_until,is_premium,description\n"
        "T0001,show-alpha,Show Alpha,d1,hin,show,Drama|Crime,2026-01-01,2026-01-01T00:00:00.000Z,"
        ",true,A great drama show\n",
        encoding="utf-8",
    )
    res_cat = ingest_catalogue(cat_file, memory_db)
    assert res_cat.records_read == 1
    assert res_cat.records_inserted == 1

    # 4. Episodes fixture
    ep_file = tmp_path / "sample_episodes.csv"
    ep_file.write_text(
        "episode_id,content_slug,season,episode,minutes\n"
        "EP-01,show-alpha,1,1,24.5\n",
        encoding="utf-8",
    )
    res_ep = ingest_episodes(ep_file, memory_db)
    assert res_ep.records_read == 1
    assert res_ep.records_inserted == 1

    # Verify audit log recorded all 4 runs
    audit_rows = memory_db.execute("SELECT count(*) FROM ingestion_audit;").fetchone()[0]
    assert audit_rows == 4


def test_duplicate_events_handling(memory_db: sqlite3.Connection, tmp_path: Path):
    """Verify that duplicate events with same event_id are skipped idempotently and audited."""
    events_file = tmp_path / "dup_events.jsonl"
    events_data = [
        {
            "event_id": "ev-dup",
            "user_id": "U-100",
            "event": "app_open",
            "occurred_at": "2026-09-01T10:00:00.000Z",
            "sent_at": "2026-09-01T10:00:02.000Z",
            "received_at": "2026-09-01T10:00:03.000Z",
            "properties": {},
        },
        {
            "event_id": "ev-dup",  # Duplicate event_id
            "user_id": "U-100",
            "event": "app_open",
            "occurred_at": "2026-09-01T10:00:00.000Z",
            "sent_at": "2026-09-01T10:00:02.000Z",
            "received_at": "2026-09-01T10:00:03.000Z",
            "properties": {},
        },
    ]
    with open(events_file, "w", encoding="utf-8") as f:
        for item in events_data:
            f.write(json.dumps(item) + "\n")

    res = ingest_events(events_file, memory_db)
    assert res.records_read == 2
    assert res.records_inserted == 1
    assert res.duplicates_skipped == 1

    # Verify database has exactly 1 row
    count = memory_db.execute("SELECT count(*) FROM events;").fetchone()[0]
    assert count == 1


def test_offline_buffered_event_flagging(memory_db: sqlite3.Connection, tmp_path: Path):
    """Verify that events with (sent_at - occurred_at) > 600s are flagged as is_offline_buffered = 1."""
    events_file = tmp_path / "offline_events.jsonl"
    events_data = [
        {
            "event_id": "ev-fresh",
            "user_id": "U-101",
            "event": "app_open",
            "occurred_at": "2026-09-01T10:00:00.000Z",
            "sent_at": "2026-09-01T10:00:05.000Z",  # 5s delay -> live
            "received_at": "2026-09-01T10:00:06.000Z",
            "properties": {},
        },
        {
            "event_id": "ev-buffered",
            "user_id": "U-101",
            "event": "app_open",
            "occurred_at": "2026-09-01T10:00:00.000Z",
            "sent_at": "2026-09-01T10:15:00.000Z",  # 900s delay -> offline buffered
            "received_at": "2026-09-01T10:15:02.000Z",
            "properties": {},
        },
    ]
    with open(events_file, "w", encoding="utf-8") as f:
        for item in events_data:
            f.write(json.dumps(item) + "\n")

    ingest_events(events_file, memory_db)
    rows = memory_db.execute(
        "SELECT event_id, is_offline_buffered FROM events ORDER BY event_id;"
    ).fetchall()
    assert rows[0]["is_offline_buffered"] == 1  # ev-buffered
    assert rows[1]["is_offline_buffered"] == 0  # ev-fresh


def test_conflicting_profiles_latest_selection(memory_db: sqlite3.Connection, tmp_path: Path):
    """Verify that multi-row profiles select latest updated_at in profiles table while preserving both in profiles_raw."""
    profiles_file = tmp_path / "conflicting_profiles.csv"
    # Column order: user_id,signup_at,subscription_status,plan_started_at,plan_ends_at,preferred_dialect,preferred_language,notifications_opt_in,whatsapp_opt_in,platform,updated_at,snapshot_at
    profiles_file.write_text(
        "user_id,signup_at,subscription_status,plan_started_at,plan_ends_at,"
        "preferred_dialect,preferred_language,notifications_opt_in,whatsapp_opt_in,"
        "platform,updated_at,snapshot_at\n"
        "U-CONFLICT,2026-08-01T00:00:00.000Z,never,,,,hin,true,false,android,2026-09-10T00:00:00.000Z,2026-09-30T05:30:00.000Z\n"
        "U-CONFLICT,2026-08-01T00:00:00.000Z,active,2026-09-20T00:00:00.000Z,,d2,hin,true,true,android,2026-09-25T12:00:00.000Z,2026-09-30T05:30:00.000Z\n"
        "U-CONFLICT,2026-08-01T00:00:00.000Z,lapsed,,,,hin,false,false,android,2026-09-15T00:00:00.000Z,2026-09-30T05:30:00.000Z\n",
        encoding="utf-8",
    )

    res = ingest_profiles(profiles_file, memory_db)
    assert res.records_read == 3
    assert res.records_inserted == 1
    assert res.records_updated == 1  # The 2026-09-25 row updated the 2026-09-10 row
    assert res.duplicates_skipped == 1  # The 2026-09-15 row was older than 2026-09-25 and was skipped

    # 1. Authoritative profiles table must have selected the latest record (2026-09-25T12:00:00.000Z -> active, d2)
    profile = memory_db.execute("SELECT * FROM profiles WHERE user_id = 'U-CONFLICT';").fetchone()
    assert profile is not None
    assert profile["subscription_status"] == "active"
    assert profile["preferred_dialect"] == "d2"
    assert profile["whatsapp_opt_in"] == 1
    assert profile["updated_at"] == "2026-09-25T12:00:00.000Z"

    # 2. Raw table must preserve all 3 records for audit and provenance
    raw_rows = memory_db.execute(
        "SELECT count(*) FROM profiles_raw WHERE user_id = 'U-CONFLICT';"
    ).fetchone()[0]
    assert raw_rows == 3


def test_missing_optional_fields(memory_db: sqlite3.Connection, tmp_path: Path):
    """Verify that missing optional CSV and JSON fields convert cleanly to SQL NULL / None without error."""
    import csv

    # Profile with missing optional fields
    profiles_file = tmp_path / "sparse_profiles.csv"
    fieldnames = [
        "user_id", "signup_at", "subscription_status", "plan_started_at", "plan_ends_at",
        "preferred_dialect", "preferred_language", "notifications_opt_in", "whatsapp_opt_in",
        "platform", "updated_at", "snapshot_at"
    ]
    with open(profiles_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "user_id": "U-SPARSE",
            "signup_at": "",
            "subscription_status": "never",
            "plan_started_at": "",
            "plan_ends_at": "",
            "preferred_dialect": "",
            "preferred_language": "",
            "notifications_opt_in": "false",
            "whatsapp_opt_in": "false",
            "platform": "",
            "updated_at": "2026-09-01T00:00:00.000Z",
            "snapshot_at": "",
        })

    res = ingest_profiles(profiles_file, memory_db)
    assert res.records_inserted == 1
    p = memory_db.execute("SELECT * FROM profiles WHERE user_id = 'U-SPARSE';").fetchone()
    assert p is not None
    assert p["signup_at"] is None
    assert p["plan_started_at"] is None
    assert p["preferred_dialect"] is None
    assert p["subscription_status"] == "never"

    # Catalogue with missing optional available_until and description
    cat_file = tmp_path / "sparse_cat.csv"
    cat_file.write_text(
        "title_id,content_slug,title,dialect,language,content_type,genres,"
        "release_date,available_from,available_until,is_premium,description\n"
        "T-OPEN,slug-open,Open Title,d2,hin,film,Drama,,2026-01-01T00:00:00.000Z,,false,\n",
        encoding="utf-8",
    )
    ingest_catalogue(cat_file, memory_db)
    c = memory_db.execute("SELECT * FROM catalogue WHERE title_id = 'T-OPEN';").fetchone()
    assert c["available_until"] is None
    assert c["description"] is None
    assert c["release_date"] is None
    assert c["is_premium"] == 0


def test_malformed_input_reporting(memory_db: sqlite3.Connection, tmp_path: Path):
    """Verify that corrupt JSON or CSV lines are recorded as malformed rather than crashing."""
    # 1. Corrupt JSON in events
    events_file = tmp_path / "bad_events.jsonl"
    with open(events_file, "w", encoding="utf-8") as f:
        f.write('{"event_id": "ev-good", "user_id": "U-1", "event": "app_open", "received_at": "2026-09-01T00:00:00.000Z"}\n')
        f.write('{"event_id": "ev-corrupt", "user_id": "U-1", truncated_json...\n')
        f.write('{"event_id": "", "user_id": "U-1", "event": "missing_event_id", "received_at": "2026-09-01T00:00:00.000Z"}\n')

    res_ev = ingest_events(events_file, memory_db)
    assert res_ev.records_read == 3
    assert res_ev.records_inserted == 1
    assert res_ev.malformed_count == 2
    assert len(res_ev.malformed_errors) == 2

    # 2. Corrupt episodes CSV
    ep_file = tmp_path / "bad_episodes.csv"
    ep_file.write_text(
        "episode_id,content_slug,season,episode,minutes\n"
        "EP-GOOD,slug-1,1,1,20.0\n"
        "EP-BAD,slug-1,not_a_number,1,20.0\n",
        encoding="utf-8",
    )
    res_ep = ingest_episodes(ep_file, memory_db)
    assert res_ep.records_read == 2
    assert res_ep.records_inserted == 1
    assert res_ep.malformed_count == 1


def test_ingestion_idempotency(file_db: tuple[sqlite3.Connection, Path], tmp_path: Path):
    """Verify that running ingestion multiple times is idempotent with zero extra rows created."""
    conn, db_path = file_db

    # Create test data files
    events_file = tmp_path / "idem_events.jsonl"
    events_file.write_text(
        json.dumps(
            {
                "event_id": "ev-idem-1",
                "user_id": "U-IDEM",
                "event": "app_open",
                "occurred_at": "2026-09-01T00:00:00.000Z",
                "sent_at": "2026-09-01T00:00:00.000Z",
                "received_at": "2026-09-01T00:00:00.000Z",
                "properties": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    profiles_file = tmp_path / "idem_profiles.csv"
    profiles_file.write_text(
        "user_id,signup_at,subscription_status,plan_started_at,plan_ends_at,"
        "preferred_dialect,preferred_language,notifications_opt_in,whatsapp_opt_in,"
        "platform,updated_at,snapshot_at\n"
        "U-IDEM,2026-08-01T00:00:00.000Z,active,,,,hin,true,true,android,2026-09-01T00:00:00.000Z,2026-09-30T05:30:00.000Z\n",
        encoding="utf-8",
    )

    # First run
    r1_ev = ingest_events(events_file, conn)
    r1_prof = ingest_profiles(profiles_file, conn)
    assert r1_ev.records_inserted == 1
    assert r1_prof.records_inserted == 1

    ev_count_1 = conn.execute("SELECT count(*) FROM events;").fetchone()[0]
    prof_count_1 = conn.execute("SELECT count(*) FROM profiles;").fetchone()[0]
    assert ev_count_1 == 1
    assert prof_count_1 == 1

    # Second run (exact duplicate)
    r2_ev = ingest_events(events_file, conn)
    r2_prof = ingest_profiles(profiles_file, conn)
    assert r2_ev.records_inserted == 0
    assert r2_ev.duplicates_skipped == 1
    assert r2_prof.records_inserted == 0

    ev_count_2 = conn.execute("SELECT count(*) FROM events;").fetchone()[0]
    prof_count_2 = conn.execute("SELECT count(*) FROM profiles;").fetchone()[0]
    assert ev_count_2 == ev_count_1
    assert prof_count_2 == prof_count_1
