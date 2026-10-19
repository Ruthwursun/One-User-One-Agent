# System Architecture: STAGE "One User, One Agent"

This document details the architectural design, safety gates, state machines, and causal evaluation methodology of the STAGE OTT lifecycle agent platform.

---

## 1. Architectural Principles

1. **Deterministic Safety Overrides Probabilistic Models:**
   No LLM or heuristic policy may publish an action directly to a user. Every action must pass an independent deterministic policy engine that enforces hard constraints (holdouts, consent, frequency caps, quiet hours, pricing honesty).
2. **Strict Causal Isolation:**
   Holdout-assigned viewers must never receive an intervention under any circumstance (0 actions published). All causal comparisons strictly respect original randomized arm assignments.
3. **Immutable Source Data Pack:**
   The reference dataset in `one_user_one_agent_pack/` is treated as read-only. All runtime databases, state logs, and generated artifacts are written strictly to `data_store/` and `sample_run/`.
4. **Idempotency & Replay Determinism:**
   Replays can be executed repeatedly from scratch or resumed without event duplication, corrupting durable viewer state, or producing divergent decisions.

---

## 2. Core Subsystems

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        SEQUENTIAL REPLAY HARNESS                       │
│  (Interleaves: Triggers, Live Events, Control Events, Model Faults)     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────┴───────────────────────────────────┐
│                          STORAGE LAYER (SQLite)                        │
│  - Events Log (deduped, offline-buffered flag)                         │
│  - Profiles (raw provenance + latest snapshot deduplication)           │
│  - Catalogue & Episode Index (availability windows, withdrawals)       │
│  - Durable Viewer State (single active goal + single plan)             │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────┴───────────────────────────────────┐
│                     DETERMINISTIC ELIGIBILITY ENGINE                   │
│  - Verifies criteria for goals G01–G12 from config/goals.json          │
│  - Dynamically applies runtime goal corrections (G03 at 14:00 IST)     │
│  - Evaluates verified subscription transitions from purchase events    │
│  - Disqualifies unconfigured goals (G10, G11)                          │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────┴───────────────────────────────────┐
│                        DECISION PLANNER & MODEL                        │
│  - Checks offline buffering on trigger (abstains if delayed >600s)     │
│  - Selects eligible goal & matches titles by viewer dialect (d1–d6)   │
│  - Invokes Mock Model Provider (reproduces injected faults/timeouts)   │
│  - Falls back deterministically if model fails or hallucinates claims  │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────┴───────────────────────────────────┐
│                      INDEPENDENT POLICY GATE ENGINE                    │
│  - Rule 1: Randomized Holdout Isolation (0 actions for holdouts)       │
│  - Rule 2: Channel Opt-In & Consent Verification                       │
│  - Rule 3: Quiet Hours Enforcement (22:00 to 09:00 IST)                │
│  - Rule 4: Frequency Caps (max 1 action per 24 hours across midnight)  │
│  - Rule 5: Copy & WhatsApp Template Schema Validation                  │
│  - Rule 6: Pricing Honesty (approved tiers only, no false discounts)   │
│  - Rule 7: Catalogue Availability & Title Withdrawal Check             │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼                                       ▼
     [Action Approved & Logged]              [Action Rejected / No-Op]
      sample_run/actions.jsonl               sample_run/decisions.jsonl
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        OFFICIAL OUTCOME SIMULATOR                      │
│                  (simulate_outcomes.py, 14-day horizon)                │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        CAUSAL IMPACT EVALUATION                        │
│  - Intention-to-Treat (ITT): Full Population vs. Active Triggered      │
│  - Newcombe Hybrid Score CIs & Exact Fisher Tests for Conversions     │
│  - Descriptive Revenue Reconciliation                                  │
│  - Exploratory CACE Instrumental Variable Evaluation                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Specifications

### 3.1 Storage Layer (`stage_agent/db.py`)
- **Engine:** SQLite with Write-Ahead Logging (`PRAGMA journal_mode=WAL`), foreign keys enabled, and indexed queries.
- **Event Deduplication:** Enforces unique `event_id` constraints on ingestion; identical re-delivered events are recorded in audit logs and ignored.
- **Offline Buffering:** An event is flagged as `is_offline_buffered = 1` whenever `(sent_at - occurred_at) > 600` seconds. Triggers arriving with this flag are rejected by the planner to prevent acting on stale context.
- **Profile Resolution:** Maintains raw profile exports in `profiles_raw` for provenance while serving queries from `profiles`, resolving duplicate user exports by latest `updated_at`.

### 3.2 Eligibility & State Engine (`stage_agent/eligibility.py`)
- Evaluates goals against verified viewer history:
  - `G01`: New user first play within 24h.
  - `G02`: First full episode completion within 3 days.
  - `G03`: Trial viewer paywall re-engagement (dynamically updated at 14:00 IST on Replay Day).
  - `G04`, `G05`, `G06`: Lapsed, active, and expiring subscription transitions.
  - `G07`: Series resume after 3-day drop-off.
  - `G09`: Weekly habit formation (14-day recurrence).
  - `G12`: Re-engagement after 7-day absence.
  - `G10`, `G11`: Disqualified (undefined in `config/goals.json`).

### 3.3 Independent Policy Engine (`stage_agent/policy.py`)
Before any action is published to `actions.jsonl`, it must pass all deterministic policy checks:
1. **Holdout Check:** If `user_id` belongs to the `holdout` arm in `replay/arms.csv`, the action is unconditionally blocked (`REJECT_HOLDOUT`).
2. **Consent & Channel Opt-In:** Verifies `notifications_opt_in = 1` for push/in-app and `whatsapp_opt_in = 1` for WhatsApp. Honors `notification_opt_out` events.
3. **Quiet Hours:** Blocks outbound messaging (`push`, `whatsapp`) between 22:00:00 and 08:59:59 IST. Permitted only between 09:00:00 and 21:59:59 IST. In-app messages and home page changes are exempt from quiet hours as they are in-session.
4. **Frequency Capping:** Enforces at most 1 message per user in any rolling 24-hour window, evaluated across midnight IST.
5. **Copy & WhatsApp Templates:** WhatsApp messages must strictly match approved template IDs (`wa_recommend_title_v1`, `wa_trial_expiring_v1`, `wa_resume_series_v1`) and include all required parameter variables. Push messages must not contain forbidden phrases (`click here`, `buy now`, `limited offer`, `guaranteed`).
6. **Pricing Honesty:** Subscriptions must match approved tiers in `config/pricing.json` (e.g., `P_MONTH` at ₹149, `P_QUARTER` at ₹399, `P_YEAR` at ₹999). Misleading discounts or unapproved prices are blocked.
7. **Catalogue Availability:** Titles must be available (`available_from <= at < available_until`). Real-time control events (`title_withdrawn`) immediately disqualify withdrawn titles.

### 3.4 Decision Planner & Fault Handling (`stage_agent/planner.py`, `stage_agent/model_provider.py`)
- Evaluates candidate goals and titles matching the viewer's current app dialect (`d1`–`d6`).
- Passes context to the model provider.
- Injects simulated model faults specified in `replay/model_faults.jsonl` (timeouts, ungrounded claims, malformed JSON).
- On model timeout or malformed output, the planner smoothly falls back to a deterministic template action, guaranteeing an action decision within the latency budget (<1000ms P50, <60s max).

### 3.5 Replay Harness (`stage_agent/replay.py`)
- Consolidates 2,769 triggers, 597 live events, and 14 control events into a single chronological stream sorted by `deliver_at`.
- Updates viewer state idempotently.
- Produces:
  - `sample_run/decisions.jsonl`: 2,769 structured decisions (recording decision ID, trigger event ID, policy pass/fail, and latency).
  - `sample_run/actions.jsonl`: 989 published actions.
  - `sample_run/viewer_state.jsonl`: Authoritative durable viewer state snapshot with goal judgments and learned summaries.
  - `sample_run/viewer_state/<user_id>.json`: Individual per-viewer state records.
  - `sample_run/latency_report.json`: Precision latency percentiles.

### 3.6 Close-the-Loop Goal Evaluator (`stage_agent/evaluator.py`)
- Post-simulation attribution & judgment engine executing after simulator run and before final state export.
- **Goal Deadlines & Windows (`stage_agent/config.py`):** Configures explicit deadlines for all candidate goals (24h for G01; 72h for G02, G07; 48h for G03; 168h for G04, G05, G08, G10, G12; 336h for G06, G09, G11).
- **Attributable Success Conditions:**
  - `G01`: First play started within 24h of commitment.
  - `G02`: Episode completed within 72h.
  - `G03`: Play started on a later IST calendar day within 48h (runtime-corrected rule).
  - `G04` & `G05`: Paid purchase completed (`is_trial = false`) within 7 days.
  - `G06`: Lapsed viewer reactivation purchase within 14 days.
  - `G07`: Resume play started on matching series slug within 72h.
  - `G08`: Play started on title released in 30 days prior to commitment within 7 days.
  - `G09`: Play started across at least 3 distinct IST calendar days within 14 days.
  - `G12`: Return play started after absence within 7 days.
- **Classification & Honesty:**
  - `succeeded`: Attributable event satisfies success condition before deadline.
  - `failed`: Deadline expired within observation window with no attributable success event.
  - `inconclusive`: Unsimulated viewer (outcomes unobserved; never assumed zero/failure), unexpired deadline (observation window ends before deadline), or unavailable/underspecified signal (`G10` undefined signal, `G11` `tv_app_open` unavailable).
- **Evidence-Grounded Learning:** Emits concise summaries with exact event IDs, title/episode identifiers, and timestamps.
- **Durable Persistence & Dual Export:** Persists `goal_judgment`, `goal_judgment_at`, `goal_learned`, and `goal_evidence_events` into SQLite `user_state`, exporting both `sample_run/viewer_state.jsonl` and individual files in `sample_run/viewer_state/<user_id>.json`.

---

## 4. Causal Impact & Statistical Evaluation (`stage_agent/impact.py`)

### 4.1 Estimand Definitions
1. **Primary ITT Population Estimand ($N = 5,000$, Sensitivity Benchmark):**
   - Evaluates all randomized viewers (2,513 Treatment vs 2,487 Holdout).
   - Treats unsimulated viewers (3,254 non-triggered viewers) as zero post-replay consumption ($Y_i = 0$) under an explicit sensitivity assumption.
2. **Active Triggered Sub-Cohort Estimand ($N = 1,746$, Fully Simulated):**
   - Evaluates viewers active on Day 0 who generated triggers (863 Treatment vs 883 Holdout).
   - Day-0 trigger occurrence is verified to be independent of treatment assignment ($p = 0.3883$).
   - Represents the Average Treatment Effect on Day-0 Active Viewers ($\text{ATE} \mid \text{Active Day 0}$).

### 4.2 Statistical Methods
- **Confidence Intervals for Proportions:** Newcombe hybrid score method using Wilson score bounds with zero-bound clamping.
- **Sparse Conversion Testing:** Two-sided Fisher's exact test using hypergeometric combinations (`math.comb`) to avoid normal z-approximation distortions.
- **Continuous Metrics (Mean Plays):** Welch's two-sample t-test with Satterthwaite degrees of freedom.
- **Revenue Accounting:** Descriptive aggregation of observed purchases (all ₹149 `P_MONTH`), computing total lift (+55.56%), population per-viewer lift (+53.95%), and triggered cohort per-viewer lift (+59.16%).
- **Complier Average Causal Effect (CACE):** Wald instrumental variable estimator with explicit identification assumption critiques (monotonicity and exclusion restriction).
