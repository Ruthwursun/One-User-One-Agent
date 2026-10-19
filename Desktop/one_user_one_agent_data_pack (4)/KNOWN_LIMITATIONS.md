# Known Limitations & Methodological Constraints

This document details the known system constraints, simulation scope boundaries, statistical power limitations, and identifying assumptions for the STAGE "One User, One Agent" lifecycle platform.

---

## 1. Simulation Scope & Incomplete Population ITT

### 1.1 Unsimulated Viewer Outcomes
In the provided outcome simulator (`one_user_one_agent_pack/simulator/simulate_outcomes.py`, line 122):
```python
users = set(triggers)  # Exactly 1,746 viewers
```
The simulator **only models post-replay behavioral outcomes for viewers who generated a trigger on Day 0**.

For the remaining 3,254 randomized viewers (1,650 in Treatment, 1,604 in Holdout):
- No post-replay events (opens, plays, completions, purchases) were simulated or logged.
- Although 60 non-triggered viewers had scripted events in `replay/live_events.jsonl`, they were excluded from post-replay simulation.
- Consequently, **their 14-day outcomes are unobserved, not empirically verified zeros**.

### 1.2 Sensitivity Imputation Benchmark
In our evaluation reports ([impact_report.json](sample_run/impact_report.json), [impact_report.md](sample_run/impact_report.md), [final_report.md](sample_run/final_report.md)), the 5,000-viewer ITT analysis assumes zero consumption for unsimulated viewers ($Y_i = 0$). This is strictly an **imputation assumption for sensitivity analysis**, not an observed fact. Complete-population ITT cannot be definitively measured without assuming zero post-replay consumption for unsimulated viewers.

---

## 2. Statistical Power & Sparse Outcome Limitations

### 2.1 Sparse Purchase Conversion Counts
Across the entire 14-day horizon:
- **Treatment Arm:** 14 unique purchasers out of 863 triggered (1.62%) and 2,513 randomized (0.56%).
- **Holdout Arm:** 9 unique purchasers out of 883 triggered (1.02%) and 2,487 randomized (0.36%).

Because purchase counts are sparse:
- Two-sided Fisher's exact test yields $p = 0.2994$ (triggered cohort) and $p = 0.4039$ (population).
- The 95% Newcombe hybrid score confidence interval for the conversion difference spans zero in both cohorts ($[-0.51, +1.79]$ pts triggered, $[-0.20, +0.61]$ pts population).
- **The observed lift in purchase conversion is not statistically significant.** The 14-day trial is heavily underpowered to prove or disprove monetization impact.

### 2.2 Descriptive Nature of Revenue Lift
- The observed +₹745.00 total revenue lift (+₹0.898 / active viewer, +₹0.291 / population viewer) is reported as a **descriptive aggregate**.
- Inferential p-values (Fisher's exact test) apply strictly to binary conversion proportions, not monetary revenue totals.
- No parametric or bootstrap revenue variance model is established for skewed financial totals; claiming statistically proven revenue gains is scientifically unsupported.

---

## 3. Exploratory CACE & Instrumental Variable Assumptions

Under the Wald IV estimator:
$$\text{CACE} = \frac{\text{ITT}}{\text{Compliance Rate}} = \frac{+11.12 \text{ pts}}{0.7683} = +14.48 \text{ pts}$$

While mathematically derived, CACE is classified as **exploratory** due to identifying assumptions:
1. **Monotonicity ($D_i(1) \ge D_i(0)$):**
   Requires that no viewer who would receive an action under holdout would be denied an action under treatment (no "defiers"). Observing 0 holdout actions confirms enforcement of the holdout gate, but does not empirically prove individual counterfactual monotonicity across potential outcomes.
2. **Exclusion Restriction:**
   Assumes that assignment to treatment affects outcomes solely through the receipt of an action. If passive database updates, eligibility checks, or state transitions slightly alter downstream simulator state for unserved treatment viewers, the exclusion restriction would be violated.
3. **Algorithmic Compliance Selection:**
   Compliance (663 / 863 = 76.8% in triggered cohort; 663 / 2,513 = 26.4% in population) is not driven by voluntary viewer adherence, but by the agent's deterministic policy rules (consent opt-in, quiet hours, frequency caps). Compliers are systematically more active and compliant than unserved viewers.

---

## 4. Operational & Replay Constraints

### 4.1 Single Replay Window (24 Hours)
The evaluation evaluates a single 24-hour replay day (2026-10-01 IST). Longer-term behavioral patterns, day-of-week seasonality, monthly subscription renewal cycles, and long-term retention decay cannot be fully captured within a single-day replay and 14-day horizon.

### 4.2 Offline Profile Export Freshness
The warehouse profile export (`profiles_snapshot.csv`) was captured at `2026-09-30 05:30 IST` (23 hours prior to the replay start). Any viewer preference updates or subscription status changes occurring between the snapshot and trigger delivery depend entirely on real-time event logs.

### 4.3 Mock Model Provider Simplifications
The model provider simulates faults (timeouts, malformed JSON, unsupported claims) using a fixed sequence defined in `replay/model_faults.jsonl`. In production, real LLM calls introduce variable network latencies, non-deterministic phrasing, and token budget variances that require real-time circuit breakers.

---

## 5. Close-the-Loop Goal Evaluation & Signal Limitations

### 5.1 Underspecified and Unavailable Target Metrics
- **Goal G10 (`early_engagement_7d`):** Target metric is defined as `"engages meaningfully"` with `event: null` in the goals schema. Because meaningful engagement lacks an attributable event schema in the event stream, the evaluator cannot verify success and honestly marks all G10 evaluations as `inconclusive`.
- **Goal G11 (`tv_screen_transition_14d`):** Target metric requires a `tv_app_open` event. However, the simulation harness models mobile and web interactions without emitting TV application events. Because the signal cannot be observed, the evaluator classifies G11 evaluations as `inconclusive`.

### 5.2 Observation Horizon Truncation & Unexpired Deadlines
- Certain lifecycle goals carry multi-day or 14-day observation windows (e.g., G06 at 336h, G09 at 336h).
- If an outcome observation window ends prior to a goal's commitment deadline and no success event has occurred yet, the evaluator marks the judgment as `inconclusive` rather than `failed`, ensuring incomplete windows are not penalized prematurely.

### 5.3 Missing Post-Replay Outcomes for Non-Triggered Viewers
- For the 3,254 viewers who did not trigger on Day 0, post-replay outcomes are unsimulated and missing.
- In accordance with close-the-loop evaluation requirements, missing simulated evidence is handled honestly: judgments for unsimulated viewers cannot establish goal results and are classified as `inconclusive` rather than assumed zero or failed.
