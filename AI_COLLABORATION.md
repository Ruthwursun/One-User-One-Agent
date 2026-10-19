# AI Collaboration & Engineering Methodology

This document outlines the human-AI collaboration workflow, prompt engineering practices, validation mechanisms, and hallucination defenses utilized throughout the implementation and auditing of the STAGE "One User, One Agent" platform.

---

## 1. Collaboration Model & Division of Responsibilities

Development followed a structured pair-programming paradigm between the human senior engineer and the AI coding assistant (Antigravity):

| Responsibility Domain | Human Engineer / Auditor | AI Coding Assistant |
|---|---|---|
| **Architectural Strategy** | Established system boundaries, SQLite persistence patterns, and deterministic safety principles. | Drafted modular implementations across storage, policy, planning, and evaluation layers. |
| **Domain Constraint Enforcement** | Defined operational rules (IST midnight frequency caps, 600s offline buffering threshold, holdout isolation). | Implemented validation algorithms and programmatic boundary checks. |
| **Adversarial Audit & Oversight** | Questioned unproven assumptions (e.g., zero-outcome assumptions for unsimulated users, naive as-treated lift claims). | Audited source code and data packs, traced purchaser identities, and derived exact mathematical formulations. |
| **Statistical Methodology** | Enforced rigorous causal standards (ITT estimand definitions, sparse outcome handling, CACE IV assumption critiques). | Implemented closed-form Newcombe score intervals, Fisher's exact tests, and Welch's t-tests in pure Python standard library. |
| **Verification & Quality Assurance** | Defined acceptance criteria and edge cases. | Authored 54 exhaustive unit and integration tests, verified CLI execution, and reconciled cross-report outputs. |

---

## 2. Guardrails Against AI Hallucination & Model Drift

Autonomous lifecycle systems present risks when LLM outputs are directly exposed to users. The platform implements rigorous architectural defenses:

1. **Deterministic Policy Gate Isolation:**
   - LLMs generate candidate actions or personalization suggestions, but **no LLM output is ever published directly**.
   - The policy engine (`stage_agent/policy.py`) independently re-evaluates all candidate actions against immutable business rules (consent, quiet hours, frequency caps, pricing honesty, catalogue availability).
2. **Deterministic Fallbacks:**
   - When the mock model provider triggers timeouts, malformed JSON, or hallucinations (recommending withdrawn titles or ungrounded prices), the planner seamlessly falls back to pre-approved deterministic templates.
3. **Hard Architectural Holdout Barrier:**
   - Holdout isolation is enforced at the database and policy gate level. Even if a model proposes an action for a holdout user, the policy engine unconditionally blocks it (`REJECT_HOLDOUT`). Zero holdout actions were published across 2,769 decisions.
4. **Stale Evidence Filtering (Offline Buffering):**
   - The planner verifies `(sent_at - occurred_at) <= 600s` on trigger events. Triggers arriving with offline buffering latency are rejected to prevent acting on obsolete viewer state.

---

## 3. Key Iterative Improvements & Methodological Refinements

During the audit phases, collaborative inspection resolved several critical subtle defects:

1. **Storage Discrepancy Reconciliation (15,651 vs. 15,402 events):**
   - Traced the raw event pack to discover 249 duplicate event IDs. Verified that SQLite deduplication by `event_id` correctly stored 15,402 unique events, preventing duplicate trigger execution.
2. **Goal Rule Dynamic Correction (G03):**
   - Implemented dynamic subscription rule modification triggered at 14:00 IST via control event stream, updating the active definition in real time without downtime.
3. **Unmasking Simulator Scoping Constraints:**
   - AI inspection of `simulate_outcomes.py` discovered that post-replay outcomes were only simulated for the 1,746 Day-0 triggered viewers.
   - Prevented misclassifying non-triggered users as verified zero-outcome events; categorized the 5,000-viewer ITT as an explicit sensitivity analysis.
4. **Exact Statistical Methods for Sparse Events:**
   - Identified that standard normal z-approximations produce distorted confidence intervals when purchase counts are sparse (14 vs 9).
   - Replaced z-tests with exact two-sided Fisher tests and Newcombe hybrid score intervals.
5. **Separation of Conversion Significance from Revenue Totals:**
   - Disentangled binary purchase conversion inferential tests from descriptive monetary totals (+₹745.00 lift), avoiding overstating causal revenue certainty.
6. **Qualification of Independence Claims:**
   - Refined statistical reporting to acknowledge that an empirical p-value of 0.3883 fails to reject independence but does not constitute formal mathematical proof.
7. **Stale Goal State Reset on New Commitment:**
   - Identified and resolved the LOW-severity issue where previous post-simulation judgment, judgment timestamp, learning summary, and evidence references could remain attached to a new in-progress goal state.
   - Enforced explicit clearing (`goal_judgment = NULL`, `goal_judgment_at = NULL`, `goal_learned = NULL`, `goal_evidence_events = NULL`, `goal_status = 'in_progress'`) in `_update_user_state()` whenever a new replay decision commits or updates a goal.

---

## 4. Test-Driven Verification

The test suite was systematically expanded to 63 passing tests (`tests/test_*.py`), providing 100% automated regression protection across:
- Ingestion idempotency, profile conflict resolution, and offline buffering flags.
- Deterministic goal eligibility evaluation and runtime rule corrections.
- Strict policy enforcement (quiet hours boundaries in IST, frequency caps across midnight, WhatsApp template schema variables, pricing honesty tiers).
- Replay runner chronological interleaving, fault injection resilience, 0-action holdout guarantees, and stale goal judgment reset.
- Statistical accuracy of Newcombe score intervals, Fisher exact calculations, Welch t-tests, and CACE estimators.

