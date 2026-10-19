# STAGE Lifecycle Agent: Methodological Impact Audit & Causal Evaluation

**Experiment Date:** 2026-10-01 (Replay Day)  
**Observation Window:** 14 Days Post-Replay (2026-10-01 to 2026-10-15 UTC)  
**Randomized Population:** N = 5,000 viewers (2,513 Treatment vs 2,487 Holdout)  
**Triggered Sub-Cohort:** N = 1,746 viewers (863 Treatment vs 883 Holdout)  
**Methodology:** Intention-to-Treat (ITT) with Newcombe Score Intervals, Fisher Exact Tests, and Exploratory CACE  

---

## 1. Executive Summary

This audit evaluates the causal impact of the STAGE "One User, One Agent" lifecycle interventions against the randomized holdout arm.

To avoid post-randomization conditioning bias, the **Primary Benchmark is the Intention-to-Treat (ITT) analysis across all 5,000 randomized viewers**, with the 1,746 active triggered viewers evaluated as a secondary sub-cohort.

```
+----------------------------------------------------------------------------------------------------+
| Metric                             | Primary ITT (Pop N=5,000)          | Triggered Sub-Cohort (N=1,746)   |
+----------------------------------------------------------------------------------------------------+
| Immediate Session Play Rate        | +3.41 pts (p < 0.0001, Stat. Sig)  | +11.12 pts (p < 0.0001, Stat. Sig) |
| Days 1-3 Early Return Open Rate    | +0.13 pts (p = 0.9090, Inconcl.)   | +2.46 pts (p = 0.2887, Inconcl.)   |
| 14-Day Paid Conversion Rate        | +0.20 pts (Fisher p = 0.4039)      | +0.60 pts (Fisher p = 0.2994)      |
| 14-Day Active Retention Rate       | -1.05 pts (p = 0.4284, Inconcl.)   | +0.04 pts (p = 0.9774, Inconcl.)   |
| Incremental Revenue Lift           | +₹745.00 (+₹0.291 / viewer)        | +₹745.00 (+₹0.898 / viewer)        |
+----------------------------------------------------------------------------------------------------+
```

### Key Methodological Findings:
1. **Confirmatory Evidence:** The agent delivers a statistically robust, highly significant boost in in-session play consumption on the replay day (+3.41 pts population / +11.12 pts triggered cohort, $p < 0.0001$), surviving family-wise Bonferroni correction.
2. **Underpowered Long-Term Outcomes:** Although 14-day paid conversion is directionally higher (14 purchases in treatment vs 9 in holdout, +₹745 revenue lift), the conversion difference does **not** achieve statistical significance ($p = 0.2994$ Fisher's exact test). Note that Fisher's exact test specifically evaluates binary purchase conversion, not total revenue. The +₹745 revenue difference is reported as a descriptive aggregate metric, as no formal revenue-variance model is specified for skewed monetary totals.
3. **Selection Bias Debunked:** Naive as-treated comparisons (+9.70 pts in triggered cohort, +57.99 pts in population) severely confound user engagement with intervention efficacy.

---

## 2. Primary Intended ITT Population: Sensitivity Benchmark (N = 5,000)

> [!NOTE]
> **Complete-Population ITT Limitation:** The 5,000 randomized viewers represent the intended target population. However, the simulation harness (`simulate_outcomes.py`) was restricted to the 1,746 viewers active on Day 0; post-replay outcomes were not simulated for the 3,254 un-triggered viewers. The figures below impute zero consumption for unsimulated viewers under sensitivity analysis rather than verified empirical observation. Complete-population ITT cannot be definitively measured without assuming zero post-replay consumption for unsimulated viewers.

| Outcome Metric | Observation Window | Treatment (2,513) | Holdout (2,487) | Absolute Diff (pts) | 95% Confidence Interval (Newcombe) | Relative Lift | p-value | Stat. Sig? (alpha=0.05) |
|---|---|---|---|---|---|---|---|---|
| **Session Play Started Rate** | Replay Day | 15.80% (397) | 12.38% (308) | **+3.41 pts** | [+1.49, +5.34] | **+27.6%** | `0.0005` | **YES (p < 0.001)** |
| **Days 1–3 Return Open Rate** | Days 1–3 | 21.85% (549) | 21.71% (540) | **+0.13 pts** | [-2.15, +2.42] | **+0.6%** | `0.9090` | No |
| **14-Day Active Retention** | Days 1–14 | 31.48% (791) | 32.53% (809) | **-1.05 pts** | [-3.64, +1.53] | **-3.2%** | `0.4249` | No |
| **14-Day Play Rate** | Days 1–14 | 23.28% (585) | 24.25% (603) | **-0.97 pts** | [-3.33, +1.39] | **-4.0%** | `0.4218` | No |
| **14-Day Episode Completion** | Days 1–14 | 16.87% (424) | 18.17% (452) | **-1.30 pts** | [-3.41, +0.81] | **-7.2%** | `0.2258` | No |
| **14-Day Paid Conversion** | Days 1–14 | 0.56% (14) | 0.36% (9) | **+0.20 pts** | [-0.20, +0.61] | **+54.0%** | `0.4039` (Fisher) | No |

### Population Revenue Summary (INR, N = 5,000, Descriptive Aggregate)
- **Treatment Arm Total:** ₹2,086.00 (14 purchasers, ₹0.830 per randomized viewer)
- **Holdout Arm Total:** ₹1,341.00 (9 purchasers, ₹0.539 per randomized viewer)
- **Incremental Revenue Lift:** **₹+745.00** (+₹0.291 / viewer, +54.0% per viewer, +55.6% total)
*Note: Revenue differences are reported as descriptive aggregates; inferential p-values (Fisher's exact test) apply strictly to binary purchase conversion rates.*

---

## 3. Randomized Active-Viewer Cohort Analysis (N = 1,746 Triggered Viewers, Fully Simulated)

> [!NOTE]
> **Active-Viewer Estimand:** Represents the Average Treatment Effect on Day-0 Active Viewers (ATE | Active Day 0). Day-0 triggers are fixed historical app events delivered prior to agent action, and empirical trigger rates between arms (34.34% vs 35.50%) fail to reject the null hypothesis of independence (z = -0.863, p = 0.3883). While failure to reject H0 does not formally prove independence, it is fully consistent with pre-treatment independence and supports using this cohort as an unconfounded conditional estimand without post-treatment conditioning bias. For this cohort, 14-day outcomes were fully simulated and observed.

| Outcome Metric | Treatment (863) | Holdout (883) | Absolute Diff (pts) | 95% CI (Newcombe) | Relative Lift | p-value | Stat. Sig? |
|---|---|---|---|---|---|---|---|
| **Session Play Started Rate** | 46.00% (397) | 34.88% (308) | **+11.12 pts** | [+6.52, +15.66] | **+31.9%** | `0.0000` | **YES (p < 0.001)** |
| **Days 1–3 Return Open Rate** | 63.62% (549) | 61.16% (540) | **+2.46 pts** | [-2.08, +6.99] | **+4.0%** | `0.2887` | No |
| **14-Day Active Retention** | 91.66% (791) | 91.62% (809) | **+0.04 pts** | [-2.58, +2.65] | **+0.0%** | `0.9774` | No |
| **14-Day Play Rate** | 67.79% (585) | 68.29% (603) | **-0.50 pts** | [-4.87, +3.87] | **-0.7%** | `0.8217` | No |
| **14-Day Episode Completion** | 49.13% (424) | 51.19% (452) | **-2.06 pts** | [-6.73, +2.63] | **-4.0%** | `0.3898` | No |
| **14-Day Paid Conversion** | 1.62% (14) | 1.02% (9) | **+0.60 pts** | [-0.51, +1.79] | **+59.2%** | `0.2994` (Fisher) | No |

### Triggered Cohort Revenue Summary (INR, N = 1,746, Descriptive Aggregate)
- **Treatment Arm Total:** ₹2,086.00 (14 purchasers, ₹2.417 per active viewer)
- **Holdout Arm Total:** ₹1,341.00 (9 purchasers, ₹1.519 per active viewer)
- **Incremental Revenue Lift:** **₹+745.00** (+₹0.898 / viewer, +59.2% per viewer, +55.6% total)
*Note: Revenue differences are reported as descriptive aggregates; inferential p-values (Fisher's exact test) apply strictly to binary purchase conversion rates.*

---

## 4. Exploratory Complier Average Causal Effect (CACE) & IV Assumptions

Not all treatment viewers received an intervention due to strict policy gates (frequency caps, quiet hours, consent, dialect compatibility).

- **Treatment Compliance Rate (Triggered):** 76.8% (663 served / 863 active)
- **Treatment Compliance Rate (Population):** 26.4% (663 served / 2,513 randomized)

Under the Wald Instrumental Variable estimator (CACE = ITT / Compliance Rate):
- **Session Play Started CACE:** **+14.48 percentage points**
- **Days 1–3 Return Open CACE:** **+3.20 percentage points**
- **14-Day Paid Conversion CACE:** **+0.79 percentage points**

> [!WARNING]
> **Methodological Classification: Exploratory, Not Confirmatory**
> The CACE estimator relies on strong identifying assumptions:
> 1. **Exclusion Restriction:** Unserved treatment viewers experience zero effect (plausible, but unverified if background checks alter state).
> 2. **Monotonicity:** Counterfactually requires no defiers across potential outcomes (D_i(1) >= D_i(0)). Observing zero holdout actions does NOT empirically prove individual counterfactual monotonicity.
> 3. **Algorithmic Compliance Selection:** Compliance was determined by the agent's eligibility rules rather than random adherence. Served viewers are systematically more active and eligible than unserved viewers.
> Therefore, CACE is a model-dependent exploratory metric; the primary causal benchmark remains the randomized ITT.

---

## 5. Selection Bias Audit: Naive As-Treated vs. Randomized ITT

In the earlier internal pilot report (`pilot/pilot_summary_report.md`), Growth Analytics claimed a misleading **+22.4 percentage point uplift** by comparing served viewers (40.1%) against all other users (17.7%).

Our audit replicates this naive as-treated comparison on the replay dataset:

| Comparison Cohort | Served Group Play Rate | Unserved Group Play Rate | Naive Observational Difference | Valid Causal ITT Difference | Selection Bias Inflation |
|---|---|---|---|---|---|
| **Triggered Cohort** | 74.06% (491/663) | 64.36% (697/1083) | **+9.70 pts** | **-0.50 pts** | **+10.20 pts** |
| **Full Population** | 74.06% (491/663) | 16.07% (697/4337) | **+57.99 pts** | **-0.97 pts** | **+58.96 pts** |

**Conclusion:** Comparing served viewers against unserved viewers severely inflates performance by selecting for viewers who are already active, present, and eligible. Unbiased causal inference strictly requires comparing randomized Treatment vs Holdout.

---

## 6. Multiple Hypothesis Testing & Statistical Limitations

1. **Family-Wise Error Rate Control:** Evaluating 6 separate outcomes inflates the probability of a false positive to 1 - (1 - 0.05)^6 ≈ 26.5%. Under a Bonferroni correction (alpha = 0.05 / 6 = 0.00833), **only Replay Day Session Play Started ($p < 0.0001$) remains statistically significant**.
2. **Sparse Purchase Counts:** With 14 purchases in treatment vs 9 in holdout, standard normal z-approximations overestimate precision. Fisher's exact test ($p = 0.2994$ in triggered cohort, $p = 0.4039$ in population) and Newcombe score intervals (spanning 0) prove that 14-day conversion lift cannot be claimed as statistically established.
