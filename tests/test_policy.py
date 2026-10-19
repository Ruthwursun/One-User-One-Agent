"""Focused unit tests for safety policy verification, quiet hours, consent, and pricing."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from stage_agent.config import IST, UTC, is_quiet_hours_ist
from stage_agent.db import init_db
from stage_agent.models import ProposedAction
from stage_agent.policy import PolicyGate


@pytest.fixture
def test_db():
    """Provides a fresh in-memory database with tables initialized."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def policy_gate():
    """Provides PolicyGate with mock arms."""
    gate = PolicyGate()
    gate.arms_cache = {
        "U-AGENT": "agent",
        "U-HOLDOUT": "holdout",
    }
    return gate


def test_holdout_users_receive_no_actions(test_db: sqlite3.Connection, policy_gate: PolicyGate):
    """Verify that holdout viewers are unconditionally blocked from receiving any actions."""
    # Attempting to send a push to holdout user
    action = ProposedAction(
        user_id="U-HOLDOUT",
        type="push",
        send_at="2026-10-01T10:00:00.000Z",  # 15:30 IST (daytime)
        text={"title": "New Title", "body": "Check this out"},
        dialect="d1",
    )
    decision = policy_gate.validate_action(action, test_db)
    assert decision.allowed is False
    assert decision.reason_code == "HOLDOUT_ARM"
    assert "holdout" in decision.message.lower()


def test_consent_and_opt_in_combinations(test_db: sqlite3.Connection, policy_gate: PolicyGate):
    """Verify push and WhatsApp consent rules, including subsequent opt-out events overriding snapshot."""
    # User A: Push True, WhatsApp False
    # User B: Push False, WhatsApp True
    # User C: Push True, but has later notification_opt_out event
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, notifications_opt_in, whatsapp_opt_in, updated_at, preferred_dialect, ingested_at)
        VALUES ('U-A', 'active', 1, 0, '2026-09-01T00:00:00.000Z', 'd1', '2026-09-01T00:00:00.000Z'),
               ('U-B', 'active', 0, 1, '2026-09-01T00:00:00.000Z', 'd1', '2026-09-01T00:00:00.000Z'),
               ('U-C', 'active', 1, 1, '2026-09-01T00:00:00.000Z', 'd1', '2026-09-01T00:00:00.000Z');
        """
    )
    # User C later opted out of push
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, channel, properties_json, ingested_at)
        VALUES ('opt-1', 'U-C', 'notification_opt_out', '2026-09-20T00:00:00.000Z', '2026-09-20T00:00:00.000Z', '2026-09-20T00:00:00.000Z', 'push', '{"channel": "push"}', '2026-09-20T00:00:00.000Z');
        """
    )
    test_db.commit()

    send_time = "2026-10-01T06:00:00.000Z"  # 11:30 IST (daytime)

    # 1. User A: Push OK, WhatsApp Blocked
    dec_a_push = policy_gate.check_consent("U-A", "push", test_db, send_time)
    assert dec_a_push.allowed is True
    dec_a_wa = policy_gate.check_consent("U-A", "whatsapp", test_db, send_time)
    assert dec_a_wa.allowed is False
    assert dec_a_wa.reason_code == "NO_WHATSAPP_CONSENT"

    # 2. User B: Push Blocked, WhatsApp OK
    dec_b_push = policy_gate.check_consent("U-B", "push", test_db, send_time)
    assert dec_b_push.allowed is False
    assert dec_b_push.reason_code == "NO_PUSH_CONSENT"
    dec_b_wa = policy_gate.check_consent("U-B", "whatsapp", test_db, send_time)
    assert dec_b_wa.allowed is True

    # 3. User C: Push Overridden by Opt-out event, WhatsApp OK
    dec_c_push = policy_gate.check_consent("U-C", "push", test_db, send_time)
    assert dec_c_push.allowed is False
    assert dec_c_push.reason_code == "PUSH_OPTED_OUT"
    dec_c_wa = policy_gate.check_consent("U-C", "whatsapp", test_db, send_time)
    assert dec_c_wa.allowed is True


def test_quiet_hours_boundaries_ist(policy_gate: PolicyGate):
    """Verify quiet hours enforcement (22:00 to 08:00 IST), testing exact boundary minutes."""
    # IST is UTC+05:30.
    # 08:00 IST = 02:30 UTC
    # 07:59 IST = 02:29 UTC
    # 21:59 IST = 16:29 UTC
    # 22:00 IST = 16:30 UTC

    # 1. Morning boundary: 07:59:59 IST -> Quiet (Blocked)
    dt_morning_quiet = datetime(2026, 10, 1, 7, 59, 59, tzinfo=IST)
    dec1 = policy_gate.check_quiet_hours(dt_morning_quiet)
    assert dec1.allowed is False
    assert dec1.reason_code == "QUIET_HOURS_IST"

    # 2. Morning boundary: 08:00:00 IST -> Allowed daytime
    dt_morning_open = datetime(2026, 10, 1, 8, 0, 0, tzinfo=IST)
    dec2 = policy_gate.check_quiet_hours(dt_morning_open)
    assert dec2.allowed is True
    assert dec2.reason_code == "ALLOWED"

    # 3. Evening boundary: 21:59:59 IST -> Allowed daytime
    dt_evening_open = datetime(2026, 10, 1, 21, 59, 59, tzinfo=IST)
    dec3 = policy_gate.check_quiet_hours(dt_evening_open)
    assert dec3.allowed is True

    # 4. Evening boundary: 22:00:00 IST -> Quiet (Blocked)
    dt_evening_quiet = datetime(2026, 10, 1, 22, 0, 0, tzinfo=IST)
    dec4 = policy_gate.check_quiet_hours(dt_evening_quiet)
    assert dec4.allowed is False
    assert dec4.reason_code == "QUIET_HOURS_IST"


def test_frequency_cap_across_midnight(test_db: sqlite3.Connection, policy_gate: PolicyGate):
    """Verify combined push/WhatsApp daily frequency cap across midnight in IST."""
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, notifications_opt_in, whatsapp_opt_in, updated_at, ingested_at)
        VALUES ('U-CAP', 'active', 1, 1, '2026-09-01T00:00:00.000Z', '2026-09-01T00:00:00.000Z');
        """
    )
    # Delivered message on Day 1 in IST: 2026-10-01 10:00 IST = 2026-10-01 04:30 UTC
    test_db.execute(
        """
        INSERT INTO events (event_id, user_id, event, occurred_at, sent_at, received_at, channel, properties_json, ingested_at)
        VALUES ('m1', 'U-CAP', 'message_delivered', '2026-10-01T04:30:00.000Z', '2026-10-01T04:30:00.000Z', '2026-10-01T04:30:00.000Z', 'push', '{"channel": "push"}', '2026-10-01T04:30:00.000Z');
        """
    )
    test_db.commit()

    # Attempt second message on the SAME IST calendar day (2026-10-01 18:00 IST = 12:30 UTC)
    send_same_day = "2026-10-01T12:30:00.000Z"
    dec_same_day = policy_gate.check_frequency_cap("U-CAP", send_same_day, test_db)
    assert dec_same_day.allowed is False
    assert dec_same_day.reason_code == "FREQUENCY_CAP_EXCEEDED"

    # Attempt message on the NEXT IST calendar day (2026-10-02 09:00 IST = 03:30 UTC) -> Allowed!
    send_next_day = "2026-10-02T03:30:00.000Z"
    dec_next_day = policy_gate.check_frequency_cap("U-CAP", send_next_day, test_db)
    assert dec_next_day.allowed is True


def test_approved_templates_and_forbidden_phrases(policy_gate: PolicyGate):
    """Verify rejection of unapproved WhatsApp templates and forbidden urgency words in push copy."""
    # 1. Approved WhatsApp template with all variables -> OK
    dec_wa_ok = policy_gate.check_template_and_copy(
        channel="whatsapp",
        template_id="wa_recommend_title_v1",
        variables={"title_name": "Show Alpha", "dialect_name": "Bhojpuri"},
    )
    assert dec_wa_ok.allowed is True

    # 2. Unapproved WhatsApp template -> Rejected
    dec_wa_bad = policy_gate.check_template_and_copy(
        channel="whatsapp",
        template_id="wa_custom_offer_v99",
        variables={"title_name": "Show Alpha"},
    )
    assert dec_wa_bad.allowed is False
    assert dec_wa_bad.reason_code == "UNAPPROVED_TEMPLATE"

    # 3. Missing WhatsApp template variables -> Rejected
    dec_wa_missing = policy_gate.check_template_and_copy(
        channel="whatsapp",
        template_id="wa_recommend_title_v1",
        variables={"title_name": "Show Alpha"},  # missing dialect_name
    )
    assert dec_wa_missing.allowed is False
    assert dec_wa_missing.reason_code == "MISSING_TEMPLATE_VARIABLES"

    # 4. Push with compliant text -> OK
    dec_push_ok = policy_gate.check_template_and_copy(
        channel="push",
        text={"title": "Naya Episode", "body": "Aapke liye naya episode uplabdh hai."},
    )
    assert dec_push_ok.allowed is True

    # 5. Push with forbidden phrase ("last chance") -> Rejected
    dec_push_forbidden = policy_gate.check_template_and_copy(
        channel="push",
        text={"title": "Special Deal", "body": "This is your last chance to watch!"},
    )
    assert dec_push_forbidden.allowed is False
    assert dec_push_forbidden.reason_code == "FORBIDDEN_PHRASE"


def test_pricing_honesty(policy_gate: PolicyGate):
    """Verify rejection of unapproved prices and non-existent subscription plans."""
    # 1. Approved plans and exact prices
    assert policy_gate.check_pricing("P_TRIAL", 1.0).allowed is True
    assert policy_gate.check_pricing("P_MONTH", 149.0).allowed is True
    assert policy_gate.check_pricing("P_QUARTER", 399.0).allowed is True

    # 2. Invented price (e.g. ₹99) -> Rejected
    dec_bad_price = policy_gate.check_pricing("P_MONTH", 99.0)
    assert dec_bad_price.allowed is False
    assert dec_bad_price.reason_code == "PRICE_MISMATCH"

    # 3. Non-existent plan ID -> Rejected
    dec_bad_plan = policy_gate.check_pricing("P_LIFETIME", 999.0)
    assert dec_bad_plan.allowed is False
    assert dec_bad_plan.reason_code == "INVALID_PLAN_ID"


def test_dialect_and_withdrawn_title_handling(test_db: sqlite3.Connection, policy_gate: PolicyGate):
    """Verify rejection of unsupported dialects (d7) and withdrawn titles (T0001)."""
    # Insert available title T1 in d1
    test_db.execute(
        """
        INSERT INTO catalogue (title_id, content_slug, title, dialect, language, content_type, genres, available_from, is_premium, ingested_at)
        VALUES ('T1', 'slug-1', 'Title 1', 'd1', 'hin', 'film', 'Drama', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z');
        """
    )
    test_db.execute(
        """
        INSERT INTO profiles (user_id, subscription_status, preferred_dialect, updated_at, notifications_opt_in, whatsapp_opt_in, ingested_at)
        VALUES ('U-D7', 'active', 'd7', '2026-09-01T00:00:00.000Z', 1, 1, '2026-09-01T00:00:00.000Z'),
               ('U-D1', 'active', 'd1', '2026-09-01T00:00:00.000Z', 1, 1, '2026-09-01T00:00:00.000Z');
        """
    )
    test_db.commit()

    at_time = "2026-10-01T10:00:00.000Z"

    # User with unsupported dialect d7 and no valid fallback -> Rejected
    dec_d7 = policy_gate.check_dialect_and_availability(
        user_id="U-D7",
        target_dialect="d7",
        title_id="T1",
        at_time=at_time,
        conn=test_db,
    )
    assert dec_d7.allowed is False
    assert dec_d7.reason_code == "UNSUPPORTED_OR_MISSING_DIALECT"

    # User with valid dialect d1 and available title T1 -> Allowed
    dec_d1 = policy_gate.check_dialect_and_availability(
        user_id="U-D1",
        target_dialect="d1",
        title_id="T1",
        at_time=at_time,
        conn=test_db,
    )
    assert dec_d1.allowed is True

    # Operational withdrawal of title T1 at 07:00:00Z
    policy_gate.record_withdrawn_title("T1", "2026-10-01T07:00:00.000Z")

    # Evaluating after 07:00:00Z -> Rejected as withdrawn!
    dec_withdrawn = policy_gate.check_dialect_and_availability(
        user_id="U-D1",
        target_dialect="d1",
        title_id="T1",
        at_time="2026-10-01T08:00:00.000Z",
        conn=test_db,
    )
    assert dec_withdrawn.allowed is False
    assert dec_withdrawn.reason_code == "TITLE_WITHDRAWN"


def test_offline_buffer_counting_consistency():
    """Verify that (sent_at - occurred_at) > 600s is mathematically identical to simulator logic."""
    # Simulator calculates:
    # true_t = r - (sent_at - occurred_at)
    # (r - true_t).total_seconds() > 600
    # Note: r - (r - (sent_at - occurred_at)) == sent_at - occurred_at.
    t_occ = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    t_sent_normal = datetime(2026, 10, 1, 10, 0, 5, tzinfo=UTC)   # 5s delay
    t_sent_delayed = datetime(2026, 10, 1, 10, 15, 0, tzinfo=UTC)  # 900s delay
    t_recv = datetime(2026, 10, 1, 10, 15, 2, tzinfo=UTC)

    # 1. Normal event
    delay_normal = (t_sent_normal - t_occ).total_seconds()
    true_t_normal = t_recv - (t_sent_normal - t_occ)
    sim_delay_normal = (t_recv - true_t_normal).total_seconds()
    assert delay_normal == sim_delay_normal == 5.0
    assert (delay_normal > 600.0) is False

    # 2. Offline-buffered event
    delay_delayed = (t_sent_delayed - t_occ).total_seconds()
    true_t_delayed = t_recv - (t_sent_delayed - t_occ)
    sim_delay_delayed = (t_recv - true_t_delayed).total_seconds()
    assert delay_delayed == sim_delay_delayed == 900.0
    assert (delay_delayed > 600.0) is True
