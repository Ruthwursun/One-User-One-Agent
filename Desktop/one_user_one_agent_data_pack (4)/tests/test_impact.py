"""Automated test suite for causal impact analysis, ITT estimators, and denominators."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from stage_agent.config import DATA_PACK_DIR
from stage_agent.impact import (
    ImpactAnalyzer,
    continuous_stats,
    fisher_exact_2x2,
    newcombe_score_diff_ci,
    proportion_stats,
    wilson_score_interval,
)


def test_arm_assignment_loading():
    """Verify randomized arm assignments load accurately from arms.csv."""
    analyzer = ImpactAnalyzer()
    assert len(analyzer.arms_population) == 5000
    arms_set = set(analyzer.arms_population.values())
    assert arms_set == {"agent", "holdout"}

    agent_count = sum(1 for a in analyzer.arms_population.values() if a == "agent")
    holdout_count = sum(1 for a in analyzer.arms_population.values() if a == "holdout")
    assert agent_count == 2513
    assert holdout_count == 2487


def test_denominators_and_study_cohort():
    """Verify denominators for triggered cohort (1,746) and population (5,000) are exact."""
    analyzer = ImpactAnalyzer()
    assert len(analyzer.triggered_user_ids) == 1746

    treat_cohort = [u for u in analyzer.triggered_user_ids if analyzer.arms_population[u] == "agent"]
    hold_cohort = [u for u in analyzer.triggered_user_ids if analyzer.arms_population[u] == "holdout"]

    assert len(treat_cohort) == 863
    assert len(hold_cohort) == 883
    assert len(treat_cohort) + len(hold_cohort) == 1746

    # Verify no overlap
    assert set(treat_cohort).isdisjoint(set(hold_cohort))


def test_proportion_stats_mathematical_properties():
    """Test proportion difference, standard error, z-score, and p-value estimators."""
    # Case 1: Identical proportions
    res_ident = proportion_stats(100, 1000, 100, 1000)
    assert res_ident["diff_pts"] == 0.0
    assert res_ident["z_score"] == 0.0
    assert res_ident["p_value"] == 1.0
    assert not res_ident["statistically_significant"]
    assert res_ident["ci95"][0] < 0.0 < res_ident["ci95"][1]

    # Case 2: Large significant difference (20% vs 10%)
    res_sig = proportion_stats(200, 1000, 100, 1000)
    assert res_sig["diff_pts"] == 10.0
    assert res_sig["z_score"] > 5.0
    assert res_sig["p_value"] < 0.0001
    assert res_sig["statistically_significant"]
    assert res_sig["ci95"][0] > 0.0  # CI strictly above zero

    # Case 3: Empty inputs
    res_empty = proportion_stats(0, 0, 0, 0)
    assert res_empty["diff_pts"] == 0.0
    assert res_empty["p_value"] == 1.0


def test_sparse_outcomes_fisher_exact_and_newcombe_ci():
    """Test Fisher's exact test and Newcombe hybrid score CI on sparse purchase counts."""
    # Example matching replay run purchases: 14 / 863 vs 9 / 883
    k_t, n_t = 14, 863
    k_c, n_c = 9, 883

    # Exact Fisher p-value
    p_fisher = fisher_exact_2x2(k_t, n_t, k_c, n_c)
    assert 0.25 < p_fisher < 0.35  # ~0.2994

    # Newcombe hybrid score interval
    ci = newcombe_score_diff_ci(k_t, n_t, k_c, n_c)
    assert ci[0] < 0.0 < ci[1]  # Confirms interval crosses zero
    assert round(ci[0], 2) == -0.51
    assert round(ci[1], 2) == 1.79

    # proportion_stats switches to Fisher for sparse counts
    stats = proportion_stats(k_t, n_t, k_c, n_c, use_exact_for_sparse=True)
    assert stats["p_value"] == round(p_fisher, 4)
    assert not stats["statistically_significant"]
    assert "ci95_wald" in stats
    assert stats["ci95"] == ci


def test_continuous_stats_welch_t_test():
    """Test continuous means, standard deviations, and t-statistic calculations."""
    vals_t = [10.0, 12.0, 11.0, 13.0, 12.0]
    vals_c = [5.0, 6.0, 5.5, 6.5, 5.0]

    res = continuous_stats(vals_t, vals_c)
    assert res["treatment_mean"] > res["holdout_mean"]
    assert res["diff"] > 5.0
    assert res["t_score"] > 5.0
    assert res["p_value"] < 0.001
    assert res["statistically_significant"]


def test_outcome_definitions_and_window_filtering(tmp_path: Path):
    """Verify that events are strictly partitioned by observation window."""
    fake_pack = tmp_path / "pack"
    fake_run = tmp_path / "run"
    (fake_pack / "replay").mkdir(parents=True)
    fake_run.mkdir(parents=True)

    # 1. arms.csv
    with open(fake_pack / "replay" / "arms.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "arm"])
        writer.writerow(["U-T1", "agent"])
        writer.writerow(["U-C1", "holdout"])

    # 2. triggers.jsonl
    with open(fake_pack / "replay" / "triggers.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"event_id": "trig-1", "user_id": "U-T1", "event": "app_open", "deliver_at": "2026-10-01T04:00:00.000Z"}) + "\n")
        f.write(json.dumps({"event_id": "trig-2", "user_id": "U-C1", "event": "app_open", "deliver_at": "2026-10-01T04:00:00.000Z"}) + "\n")

    # 3. actions.jsonl
    with open(fake_run / "actions.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"action_id": "A-1", "user_id": "U-T1", "type": "home_page", "published_at": "2026-10-01T04:00:01.000Z"}) + "\n")

    # 4. outcomes.jsonl with timed events:
    # - U-T1: replay day play, Day 2 open, Day 5 purchase
    # - U-C1: Day 10 open only
    outcomes = [
        # Replay day session (<= 2026-10-01T18:30:00Z)
        {"event_id": "o-1", "user_id": "U-T1", "event": "play_started", "received_at": "2026-10-01T05:00:00.000Z"},
        # Day 2 (Early return <= 2026-10-04T18:30:00Z)
        {"event_id": "o-2", "user_id": "U-T1", "event": "app_open", "received_at": "2026-10-03T10:00:00.000Z"},
        # Day 5 (Post-replay horizon)
        {"event_id": "o-3", "user_id": "U-T1", "event": "purchase_completed", "received_at": "2026-10-06T12:00:00.000Z", "properties": {"price_inr": 149}},
        # Control user: Day 10 open
        {"event_id": "o-4", "user_id": "U-C1", "event": "app_open", "received_at": "2026-10-11T12:00:00.000Z"},
    ]
    with open(fake_run / "outcomes.jsonl", "w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")

    analyzer = ImpactAnalyzer(pack_dir=fake_pack, sample_run_dir=fake_run)
    res = analyzer.evaluate()

    # Verify session play
    imm = res["immediate_replay_day"]["results"]
    assert imm["treatment_success"] == 1
    assert imm["holdout_success"] == 0
    assert imm["diff_pts"] == 100.0

    # Verify early return
    early = res["early_return_days_1_3"]["results"]
    assert early["treatment_success"] == 1
    assert early["holdout_success"] == 0

    # Verify 14-day retention (both were active in post-replay window)
    ret = res["full_14_day_horizon"]["active_retention_rate"]["results"]
    assert ret["treatment_success"] == 1
    assert ret["holdout_success"] == 1
    assert ret["diff_pts"] == 0.0

    # Verify purchases
    pur = res["full_14_day_horizon"]["paid_conversion_rate"]["results"]
    assert pur["treatment_success"] == 1
    assert pur["holdout_success"] == 0

    # Verify compliance
    meta = res["metadata"]
    assert meta["compliance_rate"] == 1.0


def test_missing_outcomes_preserves_denominator(tmp_path: Path):
    """Verify viewers with 0 outcome events are counted as 0 without shrinking the denominator."""
    fake_pack = tmp_path / "pack"
    fake_run = tmp_path / "run"
    (fake_pack / "replay").mkdir(parents=True)
    fake_run.mkdir(parents=True)

    with open(fake_pack / "replay" / "arms.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "arm"])
        writer.writerow(["U-T1", "agent"])
        writer.writerow(["U-T2", "agent"])  # No events
        writer.writerow(["U-C1", "holdout"])
        writer.writerow(["U-C2", "holdout"])  # No events

    with open(fake_pack / "replay" / "triggers.jsonl", "w", encoding="utf-8") as f:
        for uid in ["U-T1", "U-T2", "U-C1", "U-C2"]:
            f.write(json.dumps({"event_id": f"trig-{uid}", "user_id": uid, "event": "app_open", "deliver_at": "2026-10-01T04:00:00.000Z"}) + "\n")

    with open(fake_run / "actions.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"action_id": "A-1", "user_id": "U-T1", "type": "home_page", "published_at": "2026-10-01T04:00:01.000Z"}) + "\n")

    # Only U-T1 and U-C1 have post-replay events
    with open(fake_run / "outcomes.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"event_id": "o-1", "user_id": "U-T1", "event": "app_open", "received_at": "2026-10-05T00:00:00.000Z"}) + "\n")
        f.write(json.dumps({"event_id": "o-2", "user_id": "U-C1", "event": "app_open", "received_at": "2026-10-05T00:00:00.000Z"}) + "\n")

    analyzer = ImpactAnalyzer(pack_dir=fake_pack, sample_run_dir=fake_run)
    res = analyzer.evaluate()

    ret = res["full_14_day_horizon"]["active_retention_rate"]["results"]
    # Denominators MUST be 2 and 2, not 1 and 1
    assert ret["treatment_n"] == 2
    assert ret["holdout_n"] == 2
    assert ret["treatment_success"] == 1
    assert ret["holdout_success"] == 1
    assert ret["treatment_rate"] == 0.50
    assert ret["holdout_rate"] == 0.50


def test_primary_population_itt_includes_all_5000_viewers():
    """Verify primary ITT analysis evaluates all 5,000 randomized viewers."""
    analyzer = ImpactAnalyzer()
    res = analyzer.evaluate()

    pop = res["primary_itt_population"]
    assert pop["sample_size"]["total"] == 5000
    assert pop["sample_size"]["treatment"] == 2513
    assert pop["sample_size"]["holdout"] == 2487

    # Check session plays at population level
    imm = pop["immediate_replay_day"]["results"]
    assert imm["treatment_n"] == 2513
    assert imm["holdout_n"] == 2487
    assert imm["treatment_success"] == 397
    assert imm["holdout_success"] == 308
    assert imm["diff_pts"] > 3.0
    assert imm["statistically_significant"]

    # Check purchases at population level
    pur = pop["full_14_day_horizon"]["paid_conversion_rate"]["results"]
    assert pur["treatment_n"] == 2513
    assert pur["holdout_n"] == 2487
    assert pur["treatment_success"] == 14
    assert pur["holdout_success"] == 9
    assert not pur["statistically_significant"]  # Underpowered

    # Check revenue
    rev = pop["full_14_day_horizon"]["revenue_inr"]
    assert rev["treatment_revenue"] == 2086.0
    assert rev["holdout_revenue"] == 1341.0
    assert rev["diff_inr"] == 745.0
    assert rev["treatment_revenue_per_viewer"] > rev["holdout_revenue_per_viewer"]


def test_secondary_triggered_cohort_analysis():
    """Verify secondary triggered cohort analysis evaluates exactly 1,746 viewers."""
    analyzer = ImpactAnalyzer()
    res = analyzer.evaluate()

    trig = res["secondary_triggered_cohort"]
    assert trig["sample_size"]["total"] == 1746
    assert trig["sample_size"]["treatment"] == 863
    assert trig["sample_size"]["holdout"] == 883

    imm = trig["immediate_replay_day"]["results"]
    assert imm["treatment_n"] == 863
    assert imm["holdout_n"] == 883
    assert imm["treatment_success"] == 397
    assert imm["holdout_success"] == 308
    assert imm["diff_pts"] == 11.121
    assert imm["statistically_significant"]


def test_exploratory_cace_and_assumptions():
    """Verify CACE estimation and documentation of IV assumptions."""
    analyzer = ImpactAnalyzer()
    res = analyzer.evaluate()

    cace = res["exploratory_cace"]
    assert "triggered_compliance_rate" in cace
    assert cace["triggered_compliance_rate"] == 0.7683
    assert "session_play_started_cace_pts" in cace
    assert cace["session_play_started_cace_pts"] == 14.476
    assert "instrumental_variable_assumptions" in cace
    assumptions = cace["instrumental_variable_assumptions"]
    assert "exclusion_restriction" in assumptions
    assert "monotonicity" in assumptions
    assert "selection_into_compliance_limitation" in assumptions
    # Verify monotonicity is documented as an unproven counterfactual assumption
    assert "unproven" in assumptions["monotonicity"].lower()


def test_selection_bias_naive_comparison():
    """Verify served vs unserved comparison reveals observational selection bias."""
    analyzer = ImpactAnalyzer()
    res = analyzer.evaluate()

    bias = res["methodology_and_selection_bias"]
    naive_t = bias["triggered_served_vs_unserved"]
    assert naive_t["treatment_n"] == 663  # Served
    assert naive_t["holdout_n"] == 1083   # Unserved
    assert naive_t["diff_pts"] > 9.0      # +9.70 pts apparent lift

    naive_p = bias["population_served_vs_unserved"]
    assert naive_p["treatment_n"] == 663
    assert naive_p["holdout_n"] == 4337
    assert naive_p["diff_pts"] > 50.0     # +57.99 pts apparent lift!


def test_reproducibility_across_evaluations(tmp_path: Path):
    """Verify impact analysis is completely deterministic across repeated evaluations."""
    analyzer = ImpactAnalyzer()
    res1 = analyzer.evaluate()
    res2 = analyzer.evaluate()

    assert res1["immediate_replay_day"]["results"]["diff_pts"] == res2["immediate_replay_day"]["results"]["diff_pts"]
    assert res1["full_14_day_horizon"]["paid_conversion_rate"]["results"]["diff_pts"] == res2["full_14_day_horizon"]["paid_conversion_rate"]["results"]["diff_pts"]
    assert res1["metadata"]["compliance_rate"] == res2["metadata"]["compliance_rate"]
    assert res1["primary_itt_population"]["sample_size"]["total"] == res2["primary_itt_population"]["sample_size"]["total"]


def test_reproducibility_of_replay_decision_semantics():
    """Verify replay decision semantics are identical across runs while handling wall-clock latency."""
    # Read decisions from sample_run/decisions.jsonl
    decisions = []
    with open("sample_run/decisions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                decisions.append(json.loads(line))

    assert len(decisions) == 2769
    # Confirm deterministic properties: user_id, decision_type, and reason_code
    action_decisions = [d for d in decisions if d["decision_type"] == "action"]
    no_action_decisions = [d for d in decisions if d["decision_type"] == "no_action"]

    assert len(action_decisions) == 989
    assert len(no_action_decisions) == 1780

    # Holdout isolation is 100% deterministic
    holdout_decisions = [d for d in decisions if d.get("reason_code") == "HOLDOUT_ARM"]
    assert len(holdout_decisions) == 1325

    # Nondeterministic wall-clock latency is strictly bounded under budget
    max_latency = max(d["diagnostics"]["processing_time_ms"] for d in decisions)
    assert max_latency < 60000.0  # < 60s budget

