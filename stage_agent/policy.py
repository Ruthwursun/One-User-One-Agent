"""Deterministic safety verification and policy enforcement gate."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from stage_agent.config import (
    DATA_PACK_DIR,
    UTC,
    is_quiet_hours_ist,
    ist_date_str,
    parse_iso_datetime,
    to_ist,
)
from stage_agent.models import PolicyDecision, ProposedAction

# Allowed constants from source configurations
SUPPORTED_DIALECTS = {"d1", "d2", "d3", "d4", "d5", "d6"}

APPROVED_PRICING: dict[str, dict[str, Any]] = {
    "P_TRIAL": {"price_inr": 1.0, "days": 3, "renews_to": "P_MONTH"},
    "P_MONTH": {"price_inr": 149.0, "days": 30},
    "P_QUARTER": {"price_inr": 399.0, "days": 90},
}

ALLOWED_PRICES_INR = {p["price_inr"] for p in APPROVED_PRICING.values()}

APPROVED_WHATSAPP_TEMPLATES: dict[str, dict[str, Any]] = {
    "wa_new_episode_v2": {
        "language": "hin",
        "variables": ["title_name", "episode_number"],
    },
    "wa_recommend_title_v1": {
        "language": "hin",
        "variables": ["title_name", "dialect_name"],
    },
    "wa_trial_ending_v1": {
        "language": "hin",
        "variables": ["plan_price"],
    },
}

PUSH_FORBIDDEN_PHRASES = [
    "last chance",
    "hurry",
    "only today",
    "free forever",
    "guaranteed",
]


class PolicyGate:
    """Independent policy validation engine enforcing consent, quiet hours, caps, pricing, and holdouts."""

    def __init__(
        self,
        arms_path: str | Path | None = None,
        withdrawn_titles: dict[str, datetime] | None = None,
    ) -> None:
        self.arms_cache: dict[str, str] = {}
        self.withdrawn_titles: dict[str, datetime] = withdrawn_titles or {}

        target_arms = Path(arms_path) if arms_path else (DATA_PACK_DIR / "replay" / "arms.csv")
        if target_arms.is_file():
            self._load_arms(target_arms)

    def _load_arms(self, path: Path) -> None:
        """Load randomized arm assignment cache (user_id -> arm)."""
        import csv

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("user_id")
                arm = row.get("arm")
                if uid and arm:
                    self.arms_cache[uid] = arm.strip().lower()

    def record_withdrawn_title(self, title_id: str, at_time: datetime | str) -> None:
        """Record operational title withdrawal."""
        dt = parse_iso_datetime(at_time) if isinstance(at_time, str) else at_time
        if dt:
            self.withdrawn_titles[title_id] = dt

    def is_holdout_viewer(
        self,
        user_id: str,
        conn: sqlite3.Connection | None = None,
        arms_map: dict[str, str] | None = None,
    ) -> bool:
        """Check if user belongs to the randomized holdout arm."""
        if arms_map and user_id in arms_map:
            return arms_map[user_id] == "holdout"

        if user_id in self.arms_cache:
            return self.arms_cache[user_id] == "holdout"

        if conn is not None:
            # Check user_state table if populated
            row = conn.execute(
                "SELECT arm FROM user_state WHERE user_id = ?;", (user_id,)
            ).fetchone()
            if row and row["arm"]:
                return str(row["arm"]).lower() == "holdout"

        return False

    def check_holdout(
        self,
        user_id: str,
        conn: sqlite3.Connection | None = None,
        arms_map: dict[str, str] | None = None,
    ) -> PolicyDecision:
        """Holdout isolation rule: Holdout viewers must receive no agent action of any kind."""
        if self.is_holdout_viewer(user_id, conn, arms_map):
            return PolicyDecision(
                allowed=False,
                reason_code="HOLDOUT_ARM",
                message=f"Viewer {user_id} is in the holdout arm and must not receive any interventions.",
                details={"user_id": user_id, "arm": "holdout"},
            )
        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            message="Viewer is in the active agent arm.",
        )

    def check_consent(
        self,
        user_id: str,
        channel: str,
        conn: sqlite3.Connection,
        at_time: datetime | str | None = None,
    ) -> PolicyDecision:
        """Consent rule: Send push only when notifications_opt_in is true, WhatsApp only when whatsapp_opt_in is true.

        A later notification_opt_out event overrides the snapshot.
        In-app messages do not require consent.
        """
        ch = channel.lower()
        if ch == "in_app_message":
            return PolicyDecision(
                allowed=True,
                reason_code="ALLOWED",
                channel=ch,
                message="In-app message does not require prior consent.",
            )

        row = conn.execute(
            "SELECT notifications_opt_in, whatsapp_opt_in FROM profiles WHERE user_id = ?;",
            (user_id,),
        ).fetchone()

        if not row:
            return PolicyDecision(
                allowed=False,
                reason_code="MISSING_PROFILE",
                channel=ch,
                message=f"No profile found for user {user_id} to verify consent.",
            )

        eval_dt = parse_iso_datetime(at_time) if isinstance(at_time, str) else at_time
        eval_iso = (
            eval_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            if eval_dt
            else datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

        if ch == "push":
            if row["notifications_opt_in"] != 1:
                return PolicyDecision(
                    allowed=False,
                    reason_code="NO_PUSH_CONSENT",
                    channel=ch,
                    message="Viewer has not opted in to push notifications.",
                )
            opt_out = conn.execute(
                """
                SELECT count(*) FROM events
                WHERE user_id = ? AND event = 'notification_opt_out' AND channel = 'push'
                  AND received_at <= ?;
                """,
                (user_id, eval_iso),
            ).fetchone()[0]
            if opt_out > 0:
                return PolicyDecision(
                    allowed=False,
                    reason_code="PUSH_OPTED_OUT",
                    channel=ch,
                    message="Viewer subsequently opted out of push notifications.",
                    details={"opt_out_events": opt_out},
                )

        elif ch == "whatsapp":
            if row["whatsapp_opt_in"] != 1:
                return PolicyDecision(
                    allowed=False,
                    reason_code="NO_WHATSAPP_CONSENT",
                    channel=ch,
                    message="Viewer has not opted in to WhatsApp messages.",
                )
            opt_out = conn.execute(
                """
                SELECT count(*) FROM events
                WHERE user_id = ? AND event = 'notification_opt_out' AND channel = 'whatsapp'
                  AND received_at <= ?;
                """,
                (user_id, eval_iso),
            ).fetchone()[0]
            if opt_out > 0:
                return PolicyDecision(
                    allowed=False,
                    reason_code="WHATSAPP_OPTED_OUT",
                    channel=ch,
                    message="Viewer subsequently opted out of WhatsApp messages.",
                    details={"opt_out_events": opt_out},
                )
        else:
            return PolicyDecision(
                allowed=False,
                reason_code="INVALID_CHANNEL",
                channel=ch,
                message=f"Unknown intervention channel: {ch}",
            )

        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            channel=ch,
            message=f"Consent verified for {ch}.",
        )

    def check_quiet_hours(self, send_at: datetime | str) -> PolicyDecision:
        """Quiet hours rule: No push or WhatsApp between 22:00 and 08:00 IST."""
        dt = parse_iso_datetime(send_at) if isinstance(send_at, str) else send_at
        if not dt:
            return PolicyDecision(
                allowed=False,
                reason_code="INVALID_TIMESTAMP",
                message="Cannot verify quiet hours with an invalid timestamp.",
            )

        if is_quiet_hours_ist(dt):
            ist_dt = to_ist(dt)
            return PolicyDecision(
                allowed=False,
                reason_code="QUIET_HOURS_IST",
                message=f"Proposed send time {ist_dt.strftime('%H:%M:%S')} IST falls in quiet hours (22:00 to 08:00 IST).",
                details={
                    "utc_time": dt.isoformat(),
                    "ist_time": ist_dt.isoformat(),
                    "ist_hour": ist_dt.hour,
                },
            )

        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            message="Send time is within allowed daytime hours in IST.",
            details={"ist_hour": to_ist(dt).hour},
        )

    def check_frequency_cap(
        self,
        user_id: str,
        send_at: datetime | str,
        conn: sqlite3.Connection,
        sent_today_cache: set[tuple[str, str]] | None = None,
    ) -> PolicyDecision:
        """Frequency cap rule: At most 1 push or WhatsApp per user per IST calendar day, both combined."""
        dt = parse_iso_datetime(send_at) if isinstance(send_at, str) else send_at
        if not dt:
            return PolicyDecision(
                allowed=False,
                reason_code="INVALID_TIMESTAMP",
                message="Cannot verify frequency cap with an invalid timestamp.",
            )

        ist_day = ist_date_str(dt)

        # Check in-memory batch cache
        if sent_today_cache and (user_id, ist_day) in sent_today_cache:
            return PolicyDecision(
                allowed=False,
                reason_code="FREQUENCY_CAP_EXCEEDED",
                message=f"Viewer {user_id} already sent a message on {ist_day} IST.",
                details={"user_id": user_id, "ist_day": ist_day},
            )

        # Query database for prior delivered messages in events
        delivered_events = conn.execute(
            """
            SELECT received_at, channel FROM events
            WHERE user_id = ? AND event = 'message_delivered' AND channel IN ('push', 'whatsapp');
            """,
            (user_id,),
        ).fetchall()

        for ev in delivered_events:
            ev_dt = parse_iso_datetime(ev["received_at"])
            if ev_dt and ist_date_str(ev_dt) == ist_day:
                return PolicyDecision(
                    allowed=False,
                    reason_code="FREQUENCY_CAP_EXCEEDED",
                    message=f"Viewer {user_id} already received a {ev['channel']} message on {ist_day} IST.",
                    details={"user_id": user_id, "ist_day": ist_day, "channel": ev["channel"]},
                )

        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            message=f"Viewer has 0 prior messages on {ist_day} IST.",
            details={"ist_day": ist_day},
        )

    def check_template_and_copy(
        self,
        channel: str,
        template_id: str | None = None,
        variables: dict[str, Any] | None = None,
        text: dict[str, str] | None = None,
    ) -> PolicyDecision:
        """Validate copy rules: approved WhatsApp templates and non-forbidden push copy."""
        ch = channel.lower()
        if ch == "whatsapp":
            if not template_id or template_id not in APPROVED_WHATSAPP_TEMPLATES:
                return PolicyDecision(
                    allowed=False,
                    reason_code="UNAPPROVED_TEMPLATE",
                    channel=ch,
                    message=f"Template '{template_id}' is not an approved WhatsApp template.",
                    details={
                        "provided_template": template_id,
                        "approved_templates": list(APPROVED_WHATSAPP_TEMPLATES.keys()),
                    },
                )
            req_vars = APPROVED_WHATSAPP_TEMPLATES[template_id]["variables"]
            vars_dict = variables or {}
            missing_vars = [v for v in req_vars if not vars_dict.get(v)]
            if missing_vars:
                return PolicyDecision(
                    allowed=False,
                    reason_code="MISSING_TEMPLATE_VARIABLES",
                    channel=ch,
                    message=f"Template '{template_id}' missing required variables: {missing_vars}",
                    details={"missing_variables": missing_vars},
                )

        elif ch == "push":
            if not text:
                return PolicyDecision(
                    allowed=False,
                    reason_code="MISSING_PUSH_TEXT",
                    channel=ch,
                    message="Push message requires 'text' with 'title' and 'body'.",
                )
            title = text.get("title") or ""
            body = text.get("body") or ""
            if len(title) > 40:
                return PolicyDecision(
                    allowed=False,
                    reason_code="TITLE_EXCEEDS_MAX_CHARS",
                    channel=ch,
                    message=f"Push title length {len(title)} exceeds maximum of 40 characters.",
                )
            if len(body) > 120:
                return PolicyDecision(
                    allowed=False,
                    reason_code="BODY_EXCEEDS_MAX_CHARS",
                    channel=ch,
                    message=f"Push body length {len(body)} exceeds maximum of 120 characters.",
                )
            combined_copy = f"{title} {body}".lower()
            for phrase in PUSH_FORBIDDEN_PHRASES:
                if phrase in combined_copy:
                    return PolicyDecision(
                        allowed=False,
                        reason_code="FORBIDDEN_PHRASE",
                        channel=ch,
                        message=f"Push copy contains forbidden urgency/guarantee phrase: '{phrase}'.",
                        details={"forbidden_phrase": phrase},
                    )

        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            channel=ch,
            message="Copy and template approved.",
        )

    def check_pricing(
        self,
        plan_id: str | None = None,
        price_inr: float | int | None = None,
    ) -> PolicyDecision:
        """Pricing honesty rule: Only prices and plans that exist in config/pricing.json."""
        if plan_id is not None:
            if plan_id not in APPROVED_PRICING:
                return PolicyDecision(
                    allowed=False,
                    reason_code="INVALID_PLAN_ID",
                    message=f"Plan '{plan_id}' does not exist in pricing policy.",
                    details={"approved_plans": list(APPROVED_PRICING.keys())},
                )
            expected_price = APPROVED_PRICING[plan_id]["price_inr"]
            if price_inr is not None and float(price_inr) != expected_price:
                return PolicyDecision(
                    allowed=False,
                    reason_code="PRICE_MISMATCH",
                    message=f"Price ₹{price_inr} does not match official price ₹{expected_price} for plan {plan_id}.",
                    details={"expected_price": expected_price, "received_price": price_inr},
                )

        if price_inr is not None:
            if float(price_inr) not in ALLOWED_PRICES_INR:
                return PolicyDecision(
                    allowed=False,
                    reason_code="UNAPPROVED_PRICE",
                    message=f"Price ₹{price_inr} is not an approved price tier.",
                    details={"approved_prices": sorted(list(ALLOWED_PRICES_INR))},
                )

        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            message="Pricing verified against official policy.",
        )

    def check_dialect_and_availability(
        self,
        user_id: str,
        target_dialect: str | None,
        title_id: str | None,
        at_time: datetime | str,
        conn: sqlite3.Connection,
    ) -> PolicyDecision:
        """Dialect and availability rule: Never invent titles, dialects, or use unavailable content."""
        eval_dt = parse_iso_datetime(at_time) if isinstance(at_time, str) else at_time
        if not eval_dt:
            return PolicyDecision(
                allowed=False,
                reason_code="INVALID_TIMESTAMP",
                message="Cannot evaluate availability with an invalid timestamp.",
            )
        eval_iso = eval_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # 1. Validate Dialect
        if not target_dialect or target_dialect not in SUPPORTED_DIALECTS:
            # Check if profile has a valid preferred dialect
            row = conn.execute(
                "SELECT preferred_dialect FROM profiles WHERE user_id = ?;", (user_id,)
            ).fetchone()
            pref_dialect = row["preferred_dialect"] if row else None
            if not pref_dialect or pref_dialect not in SUPPORTED_DIALECTS:
                return PolicyDecision(
                    allowed=False,
                    reason_code="UNSUPPORTED_OR_MISSING_DIALECT",
                    message=f"Dialect '{target_dialect}' is unsupported and user has no valid preferred dialect.",
                    details={"target_dialect": target_dialect, "preferred_dialect": pref_dialect},
                )
            target_dialect = pref_dialect

        # 2. Validate Title Availability (if a title is being recommended/shown)
        if title_id:
            # Check operational withdrawal
            withdrawn_at = self.withdrawn_titles.get(title_id)
            if withdrawn_at and eval_dt >= withdrawn_at:
                return PolicyDecision(
                    allowed=False,
                    reason_code="TITLE_WITHDRAWN",
                    message=f"Title {title_id} was operationally withdrawn at {withdrawn_at.isoformat()}.",
                    details={"title_id": title_id, "withdrawn_at": withdrawn_at.isoformat()},
                )

            t_row = conn.execute(
                "SELECT title_id, dialect, available_from, available_until FROM catalogue WHERE title_id = ?;",
                (title_id,),
            ).fetchone()

            if not t_row:
                return PolicyDecision(
                    allowed=False,
                    reason_code="TITLE_NOT_IN_CATALOGUE",
                    message=f"Title {title_id} does not exist in the catalogue.",
                    details={"title_id": title_id},
                )

            if t_row["dialect"] != target_dialect:
                return PolicyDecision(
                    allowed=False,
                    reason_code="TITLE_DIALECT_MISMATCH",
                    message=f"Title {title_id} dialect '{t_row['dialect']}' does not match page/message dialect '{target_dialect}'.",
                    details={"title_dialect": t_row["dialect"], "target_dialect": target_dialect},
                )

            from_dt = parse_iso_datetime(t_row["available_from"])
            until_dt = parse_iso_datetime(t_row["available_until"])

            if from_dt and eval_dt < from_dt:
                return PolicyDecision(
                    allowed=False,
                    reason_code="TITLE_NOT_YET_AVAILABLE",
                    message=f"Title {title_id} available_from is in the future ({t_row['available_from']}).",
                )
            if until_dt and eval_dt >= until_dt:
                return PolicyDecision(
                    allowed=False,
                    reason_code="TITLE_EXPIRED",
                    message=f"Title {title_id} expired on {t_row['available_until']}.",
                )

        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            message="Dialect and content availability verified.",
            details={"verified_dialect": target_dialect},
        )

    def validate_home_page_structure(
        self,
        action: ProposedAction,
        conn: sqlite3.Connection,
        trigger_app_dialect: str | None = None,
    ) -> PolicyDecision:
        """Validate personalized home page structure matching backend and simulator rules.

        Rules:
        - Max 6 rows.
        - Min 3 and max 10 titles per valid row.
        - Content slug can appear only once per page across all rows.
        - Page dialect must match trigger session app dialect (no silent substitution).
        - If all rows drop below minimum and hero is missing/invalid, page is rejected.
        """
        page_dialect = action.dialect
        if not page_dialect or page_dialect not in SUPPORTED_DIALECTS:
            return PolicyDecision(
                allowed=False,
                reason_code="UNSUPPORTED_OR_MISSING_DIALECT",
                channel="home_page",
                message=f"Page dialect '{page_dialect}' is unsupported (only d1-d6 allowed).",
            )

        if trigger_app_dialect and trigger_app_dialect != page_dialect:
            return PolicyDecision(
                allowed=False,
                reason_code="DIALECT_APP_MISMATCH",
                channel="home_page",
                message=f"Page dialect '{page_dialect}' does not match active app dialect '{trigger_app_dialect}'.",
                details={"page_dialect": page_dialect, "app_dialect": trigger_app_dialect},
            )

        at_time = action.published_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        eval_dt = parse_iso_datetime(at_time)
        if not eval_dt:
            return PolicyDecision(
                allowed=False,
                reason_code="INVALID_TIMESTAMP",
                channel="home_page",
                message="Invalid published_at timestamp for home page.",
            )

        # Validate Hero Title
        hero_id = action.hero_title_id
        hero_ok = False
        if hero_id:
            withdrawn_at = self.withdrawn_titles.get(hero_id)
            if not (withdrawn_at and eval_dt >= withdrawn_at):
                t_row = conn.execute(
                    "SELECT dialect, available_from, available_until FROM catalogue WHERE title_id = ?;",
                    (hero_id,),
                ).fetchone()
                if t_row and t_row["dialect"] == page_dialect:
                    f_dt = parse_iso_datetime(t_row["available_from"])
                    u_dt = parse_iso_datetime(t_row["available_until"])
                    if (not f_dt or eval_dt >= f_dt) and (not u_dt or eval_dt < u_dt):
                        hero_ok = True

        # Validate Rows
        rows = action.rows or []
        if len(rows) > 6:
            rows = rows[:6]  # Simulator truncates over 6 rows

        valid_rows = []
        seen_slugs: set[str] = set()

        for r in rows:
            row_type = r.get("row_type")
            title_ids = r.get("title_ids") or []
            valid_tids = []

            for tid in title_ids[:10]:
                withdrawn_at = self.withdrawn_titles.get(tid)
                if withdrawn_at and eval_dt >= withdrawn_at:
                    continue

                t_row = conn.execute(
                    "SELECT content_slug, dialect, available_from, available_until FROM catalogue WHERE title_id = ?;",
                    (tid,),
                ).fetchone()

                if not t_row or t_row["dialect"] != page_dialect:
                    continue

                slug = t_row["content_slug"]
                if slug in seen_slugs:
                    continue  # Slug already used on this page

                f_dt = parse_iso_datetime(t_row["available_from"])
                u_dt = parse_iso_datetime(t_row["available_until"])
                if (f_dt and eval_dt < f_dt) or (u_dt and eval_dt >= u_dt):
                    continue

                seen_slugs.add(slug)
                valid_tids.append(tid)

            if len(valid_tids) >= 3:
                valid_rows.append({"row_type": row_type, "title_ids": valid_tids})

        if not valid_rows and not hero_ok:
            return PolicyDecision(
                allowed=False,
                reason_code="HOME_PAGE_REJECTED_EMPTY",
                channel="home_page",
                message="Home page has no valid rows with >= 3 titles and no valid hero.",
                details={"rows_received": len(rows), "seen_slugs": len(seen_slugs)},
            )

        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            channel="home_page",
            message="Home page structure validated successfully.",
            details={"valid_rows": len(valid_rows), "hero_ok": hero_ok, "total_titles": len(seen_slugs)},
        )

    def validate_action(
        self,
        action: ProposedAction | dict[str, Any],
        conn: sqlite3.Connection,
        arms_map: dict[str, str] | None = None,
        sent_today_cache: set[tuple[str, str]] | None = None,
        trigger_app_dialect: str | None = None,
    ) -> PolicyDecision:
        """Full end-to-end policy gate validation for any proposed intervention."""
        act = action if isinstance(action, ProposedAction) else ProposedAction(**action)
        uid = act.user_id
        action_type = act.type.lower()
        at_time = act.send_at or act.published_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Gate 1: Check Holdout Isolation
        holdout_dec = self.check_holdout(uid, conn, arms_map)
        if not holdout_dec.allowed:
            return holdout_dec

        # Gate 2: Check Pricing Honesty (if action mentions pricing/plans)
        if act.plan_id or act.price_inr:
            pricing_dec = self.check_pricing(act.plan_id, act.price_inr)
            if not pricing_dec.allowed:
                return pricing_dec

        # Gate 3: Channel-Specific Checks
        if action_type == "home_page":
            hp_dec = self.validate_home_page_structure(
                action=act,
                conn=conn,
                trigger_app_dialect=trigger_app_dialect,
            )
            if not hp_dec.allowed:
                return hp_dec

        elif action_type in ("push", "whatsapp"):
            # Consent
            consent_dec = self.check_consent(uid, action_type, conn, at_time)
            if not consent_dec.allowed:
                return consent_dec

            # Quiet Hours
            quiet_dec = self.check_quiet_hours(at_time)
            if not quiet_dec.allowed:
                return quiet_dec

            # Frequency Cap
            freq_dec = self.check_frequency_cap(uid, at_time, conn, sent_today_cache)
            if not freq_dec.allowed:
                return freq_dec

            # Template and Copy Rules
            copy_dec = self.check_template_and_copy(
                channel=action_type,
                template_id=act.template_id,
                variables=act.variables,
                text=act.text,
            )
            if not copy_dec.allowed:
                return copy_dec

            # Dialect & Content Availability for message target title
            dialect_dec = self.check_dialect_and_availability(
                user_id=uid,
                target_dialect=act.dialect,
                title_id=act.title_id or act.hero_title_id,
                at_time=at_time,
                conn=conn,
            )
            if not dialect_dec.allowed:
                return dialect_dec

        return PolicyDecision(
            allowed=True,
            reason_code="ALLOWED",
            channel=action_type,
            message="Proposed action passed all policy and safety rules.",
        )
