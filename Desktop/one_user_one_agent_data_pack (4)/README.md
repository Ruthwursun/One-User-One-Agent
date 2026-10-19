# STAGE OTT: One User, One Agent

Evidence-grounded, goal-directed lifecycle agent platform for STAGE OTT viewers.

---

## 1. Executive Overview

This repository implements an autonomous lifecycle agent architecture for OTT viewers under real-world constraints:
- **Deterministic State & Eligibility:** Rigid verification of viewer states, goal transitions (G01–G12), and active goal corrections.
- **Independent Safety & Policy Engine:** Enforces consent opt-ins, quiet hours (22:00–09:00 IST), frequency caps across midnight IST, strict pricing honesty, WhatsApp template schema compliance, and zero holdout actions.
- **Resilient Decision Planner:** Combines evidence-grounded LLM decisions with deterministic fallbacks, dialect routing, and strict tool/latency budgets.
- **Sequential Replay Runner:** Processes chronological triggers (2,769), live events, control events (title withdrawals, goal rule corrections, dependency faults), and mock model faults idempotently.
- **Causal Impact & Revenue Evaluation:** Computes Intention-to-Treat (ITT) metrics across the full randomized population (N = 5,000) and triggered active cohort (N = 1,746), using Newcombe hybrid score confidence intervals, Fisher's exact tests for sparse purchase conversions, and exploratory Complier Average Causal Effect (CACE) IV analysis.

---

## 2. Directory Layout

```text
├── config/                     # Configuration definitions
├── data_store/                 # SQLite durable state and ingestion databases
├── one_user_one_agent_pack/    # Immutable reference data pack (events, catalogue, replay streams)
├── sample_run/                 # Primary run artifacts and evaluation reports
│   ├── actions.jsonl           # 989 published actions
│   ├── decisions.jsonl         # 2,769 structured decision logs
│   ├── viewer_state.jsonl      # Durable viewer state records with goal judgments & learning
│   ├── viewer_state/           # Per-viewer individual state JSON files (<user_id>.json)
│   ├── outcomes.jsonl          # Post-replay simulated outcomes (31,613 events)
│   ├── latency_report.json     # P50/P90/P95/P99 latency & call metrics
│   ├── impact_report.json      # Structured statistical causal evaluation
│   ├── impact_report.md        # Comprehensive markdown statistical report
│   ├── final_report.md         # End-to-end milestone & impact report
│   └── sim/                    # Simulator logs and outcome events
├── scripts/                    # CLI runner scripts
│   ├── ingest_all.py           # Ingestion pipeline
│   └── run_replay.py           # Full sequential replay runner
├── stage_agent/                # Core application package
│   ├── config.py               # Constants, paths, and policy parameters
│   ├── db.py                   # SQLite storage layer & ingestion logic
│   ├── eligibility.py          # Deterministic goal eligibility evaluation
│   ├── policy.py               # Policy, consent, frequency cap, and holdout gates
│   ├── model_provider.py       # Mock model provider with fault reproduction
│   ├── planner.py              # Decision planner & fallback logic
│   ├── replay.py               # Replay harness & chronological event processor
│   ├── evaluator.py            # Post-simulation close-the-loop goal evaluator
│   └── impact.py               # Statistical causal impact analyzer
└── tests/                      # Pytest test suite (63 unit and integration tests)
    ├── test_ingestion.py       # Storage, dedup, and offline buffering tests
    ├── test_eligibility.py     # Goal rules, subscription transitions, G03 correction
    ├── test_policy.py          # Holdout isolation, caps, quiet hours, pricing
    ├── test_planner.py         # Grounding, dialect matching, model fault resilience
    ├── test_replay.py          # Sequential replay integration & idempotency
    ├── test_close_the_loop.py  # Attributable goal evaluation, deadlines, and learning
    └── test_impact.py          # ITT, Newcombe intervals, Fisher tests, CACE IV
```

---

## 3. Setup & Environment

### Requirements
- Python 3.9+ (tested on Python 3.14)
- Standard library dependencies + `pytest` and `pydantic`

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. Execution Commands

### 1. Run Complete Test Suite
```bash
python -m pytest -v
```
*Expected output: Exactly 63 passed tests.*

### 2. Ingest Reference Data Pack
```bash
python scripts/ingest_all.py
```

### 3. Run Full Replay Pipeline & Simulator
```bash
python scripts/run_replay.py --pack one_user_one_agent_pack --out sample_run --seed oua-practice
```
This executes:
1. Sequential chronological replay of 3,380 events (2,769 triggers, 597 live events, 14 control events).
2. Production of `sample_run/decisions.jsonl` (2,769 decisions) and `sample_run/actions.jsonl` (989 actions; 0 holdout actions).
3. Invocation of the official outcome simulator (`one_user_one_agent_pack/simulator/simulate_outcomes.py`).
4. Close-the-loop goal evaluation (`stage_agent/evaluator.py`), assessing active goals against attributable outcomes, deadlines, and success conditions, persisting judgments to SQLite `user_state`, and exporting `sample_run/viewer_state.jsonl` and `sample_run/viewer_state/<user_id>.json`.
5. Automatic calculation and generation of causal impact reports (`sample_run/impact_report.json`, `sample_run/impact_report.md`, and `sample_run/final_report.md`).

### 4. Run Standalone Goal Evaluation & Causal Impact Analysis
```bash
# Run post-simulation goal evaluator
python -c "from stage_agent.evaluator import GoalEvaluator; GoalEvaluator().evaluate_outcomes()"

# Run causal impact analysis
python -m stage_agent.impact
```

---

## 5. Documentation Deliverables
- [`ARCHITECTURE.md`](ARCHITECTURE.md): System architecture, safety gates, and state design.
- [`AI_COLLABORATION.md`](AI_COLLABORATION.md): Human-AI pair programming workflows, validation methodologies, and guardrails.
- [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md): Simulator scoping assumptions, statistical power constraints, and identification assumptions.
- [`sample_run/final_report.md`](sample_run/final_report.md): Final end-to-end audit and execution report.
- [`sample_run/impact_report.md`](sample_run/impact_report.md): Detailed causal inference report.
