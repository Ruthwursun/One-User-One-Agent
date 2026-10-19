# STAGE "One User, One Agent" — Replay and Verification Report

**Replay Date:** 2026-10-01 IST (2026-09-30 18:30 UTC to 2026-10-01 18:30 UTC)  
**Simulation Seed:** `oua-practice`  
**Pack Directory:** `one_user_one_agent_pack`  
**Execution Timestamp:** 2026-10-19T05:53:57.526Z  

---

## 1. Executive Summary

The STAGE "One User, One Agent" sequential replay harness processed the entire replay stream of **2,769** triggers, **607** organic live events, and **4** operational control events in strict delivery order.

- **Simulator Policy Violations:** **0** (Zero violations achieved)
- **Holdout Actions Published:** **0** (100% holdout arm isolation preserved)
- **Decision Budget Compliance:** **100%** (0 decisions exceeded the 60.0s budget; max observed: 13.2 ms)
- **Model/Tool Call Limits:** **100%** compliant (Max model calls: 1/8; Max tool calls: 6/20)

---

## 2. Replay & Decision Stream Breakdown

| Metric | Count | Description |
|---|---|---|
| **Total Triggers Read** | 2,769 | Raw triggers delivered in `triggers.jsonl` |
| **Duplicate Triggers Suppressed** | 90 | Redelivered triggers idempotently skipped |
| **Holdout Triggers (Abstained)** | 1,325 | Randomized holdout viewers strictly isolated |
| **Agent Arm Decisions** | 1,354 | Live sessions evaluated for interventions |
| **Actions Approved & Published** | 989 | Passed planner, safety gates, and consent |
| **Explicit Abstentions (`no_action`)** | 1,780 | Documented non-interventions |

### Published Actions by Intervention Channel
| Channel | Count | Simulator Accepted | Simulator Delivered / Shown |
|---|---|---|---|
| `home_page` | 954 | 954 | 748 |
| `push` | 35 | 35 | 31 |
| `whatsapp` | 0 | 0 | 0 |
| **Total Actions** | **989** | — | — |

### Abstentions by Reason Code
| Reason Code | Count | Rationale |
|---|---|---|
| `HOLDOUT_ARM` | 1,325 | Policy/eligibility boundary enforcement |
| `OFFLINE_BUFFERED_SESSION` | 191 | Policy/eligibility boundary enforcement |
| `MISSING_VIEWER_EVIDENCE` | 100 | Policy/eligibility boundary enforcement |
| `DUPLICATE_TRIGGER` | 90 | Policy/eligibility boundary enforcement |
| `INSUFFICIENT_CONTENT_EVIDENCE` | 47 | Policy/eligibility boundary enforcement |
| `FREQUENCY_CAP_EXCEEDED` | 12 | Policy/eligibility boundary enforcement |
| `NO_PUSH_CONSENT` | 8 | Policy/eligibility boundary enforcement |
| `QUIET_HOURS_IST` | 7 | Policy/eligibility boundary enforcement |

---

## 3. Simulator Validation & Outcome Play

The original, unmodified STAGE outcome simulator (`simulate_outcomes.py`) was executed against the published actions over the 14-day observation horizon.

- **Total Policy Problems Detected:** `0`
- **Triggered Users Simulated:** `1,746`
- **Outcome Events Written:** `31,613`

### Simulator Event Counts Summary
| Event / Outcome Metric | Count |
|---|---|
| `home_page_accepted` | 954 |
| `home_page_shown_personalised` | 748 |
| `home_page_published_but_not_shown` | 28 |
| `home_page_rejected_empty` | 0 |
| `push_delivered` | 31 |
| `push_clicked` | 0 |
| `whatsapp_delivered` | 0 |
| `whatsapp_clicked` | 0 |
| `organic_purchases_scripted` | 36 |

### Close-the-Loop Goal Evaluation Summary

Post-simulation evaluation judged each viewer's active goal against simulated outcome events using only attributable signals, actual success conditions, and deadlines.

| Judgment Status | Count | Description |
|---|---|---|
| `succeeded` | 72 | Goal success condition met before deadline with attributable event |
| `failed` | 591 | Deadline expired with no attributable success event |
| `inconclusive` | 0 | Unsimulated viewer, unexpired deadline, or unavailable/underspecified signal |
| `none` | 1,083 | No active goal committed |
| **Total Viewers Evaluated** | **1,746** | Authoritative durable state updated in SQLite `user_state` |

---

## 4. Latency and Resource Limits

| Metric | Target Limit | Observed Performance | Status |
|---|---|---|---|
| **P50 Decision Latency** | < 1,000 ms | 0.01 ms | PASSED |
| **P90 Decision Latency** | < 5,000 ms | 1.86 ms | PASSED |
| **P95 Decision Latency** | < 10,000 ms | 2.08 ms | PASSED |
| **P99 Decision Latency** | < 30,000 ms | 2.67 ms | PASSED |
| **Max Decision Latency** | < 60,000 ms | 13.2 ms | PASSED |
| **Budget Violations (>60s)** | 0 | 0 | PASSED |
| **Max Model Calls per Decision** | 8 | 1 | PASSED |
| **Max Tool Calls per Decision** | 20 | 6 | PASSED |

---

## 5. Artifact Checklist

All required evaluation artifacts have been produced in `sample_run/`:
- `sample_run/decisions.jsonl`: Exactly 2,769 structured decision records (one per delivered trigger).
- `sample_run/actions.jsonl`: Exactly 989 published interventions (0 holdout actions).
- `sample_run/viewer_state.jsonl`: Authoritative durable viewer state records with close-the-loop goal judgments and learned summaries.
- `sample_run/viewer_state/<user_id>.json`: Individual per-viewer durable state records.
- `sample_run/outcomes.jsonl`: Generated post-replay simulated events (31,613 events).
- `sample_run/latency_report.json`: High-resolution latency and call diagnostics.
- `sample_run/final_report.md`: This comprehensive execution summary.

---

## 6. Causal Impact Analysis (Intention-to-Treat)

The causal impact of the agent was rigorously evaluated using an **Intention-to-Treat (ITT)** design. To eliminate selection bias, the primary benchmark evaluates all **5,000 randomized viewers** (2,513 Treatment vs 2,487 Holdout), with the 1,746 active triggered viewers evaluated as a secondary sub-cohort.

### Primary ITT Analysis: Full Randomized Population (N = 5,000, Sensitivity Benchmark)

> **Methodological Note on Incomplete Population ITT:** The 5,000 randomized viewers represent the intended target population. However, the simulation harness (`simulate_outcomes.py`) was restricted to the 1,746 viewers active on Day 0; post-replay outcomes were not simulated for the 3,254 un-triggered viewers. The figures below impute zero consumption for unsimulated viewers under sensitivity analysis rather than verified empirical observation. Complete-population ITT cannot be definitively measured without assuming zero post-replay consumption for unsimulated viewers.

| Outcome Metric | Treatment (2,513) | Holdout (2,487) | Absolute Diff (pts) | 95% CI (Newcombe) | Relative Lift | p-value | Stat. Sig? |
|---|---|---|---|---|---|---|---|
| **Session Play Started Rate** | **15.80%** (397) | **12.38%** (308) | **+3.41 pts** | [+1.49, +5.34] | **+27.6%** | `0.0005` | **YES (p < 0.001)** |
| **Days 1–3 Return Open Rate** | **21.85%** (549) | **21.71%** (540) | **+0.13 pts** | [-2.15, +2.42] | **+0.6%** | `0.9090` | No |
| **14-Day Paid Conversion** | **0.56%** (14) | **0.36%** (9) | **+0.20 pts** | [-0.20, +0.61] | **+54.0%** | `0.4039` (Fisher) | No |
| **Total Revenue (INR)** | **₹2,086.00** | **₹1,341.00** | **₹+745.00** | — | **+55.6%** | — (Descriptive) |

*Population Revenue Breakdown: 14 Treatment purchasers (₹2,086.00 / 2,513 = ₹0.830 / viewer) vs 9 Holdout purchasers (₹1,341.00 / 2,487 = ₹0.539 / viewer). Incremental per-viewer lift: +₹0.291 / viewer (+53.9%). Revenue differences are descriptive aggregates; inferential p-values apply to purchase conversion.*

### Secondary Active Cohort Analysis (N = 1,746 Triggered Viewers, Fully Simulated)

> **Methodological Note on Active-Viewer Estimand:** Represents the Average Treatment Effect on Day-0 Active Viewers (ATE | Active Day 0). Day-0 triggers are fixed historical app events delivered prior to agent action, and empirical trigger rates between arms (34.34% vs 35.50%) fail to reject the null hypothesis of independence (z = -0.863, p = 0.3883). While failure to reject H0 does not formally prove independence, it is fully consistent with pre-treatment independence and supports using this cohort as an unconfounded conditional estimand without post-treatment conditioning bias. For this cohort, 14-day outcomes were fully simulated and observed.

| Outcome Metric | Treatment (863) | Holdout (883) | Absolute Diff (pts) | 95% CI (Newcombe) | Relative Lift | p-value | Stat. Sig? |
|---|---|---|---|---|---|---|---|
| **Session Play Started Rate** | **46.00%** (397) | **34.88%** (308) | **+11.12 pts** | [+6.52, +15.66] | **+31.9%** | `0.0000` | **YES (p < 0.001)** |
| **Days 1–3 Return Open Rate** | **63.62%** (549) | **61.16%** (540) | **+2.46 pts** | [-2.08, +6.99] | **+4.0%** | `0.2887` | No |
| **14-Day Paid Conversion** | **1.62%** (14) | **1.02%** (9) | **+0.60 pts** | [-0.51, +1.79] | **+59.2%** | `0.2994` (Fisher) | No |
| **Total Revenue (INR)** | **₹2,086.00** | **₹1,341.00** | **₹+745.00** | — | **+55.6%** | — (Descriptive) |

*Triggered Cohort Revenue Breakdown: 14 Treatment purchasers (₹2,086.00 / 863 = ₹2.417 / viewer) vs 9 Holdout purchasers (₹1,341.00 / 883 = ₹1.519 / viewer). Incremental per-viewer lift: +₹0.898 / viewer (+59.2%). Revenue differences are descriptive aggregates; inferential p-values apply to purchase conversion.*

Detailed statistical tables, selection bias demonstrations, and Newcombe confidence intervals are recorded in `sample_run/impact_report.json` and `sample_run/impact_report.md`.
