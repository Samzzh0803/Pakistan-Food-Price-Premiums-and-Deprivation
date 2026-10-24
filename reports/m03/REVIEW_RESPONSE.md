# Response to the Codex Milestone 03 review

Responds to `CODEX_M03_REVIEW.md` (commit `328e136`). Every result in the notebook, numbers JSON, tables, figures and
manuscript was regenerated after these fixes.

## Blockers

**1. The outer test boundary is now protected during model selection: fixed.**
- **Problem:** CV fold 5 validated on week 2026-03-26, whose labels come from 2026-04-02, the first test week.
- **New CV design:** `development_weeks` (`pfp/model.py`) keeps only training weeks whose *t+1* is before the test
  period. That leaves 10 weeks, ending 2025-12-18. `forward_chain_folds` now starts at week 5, which keeps 5 folds.
- **New assert:** `make_fold_data(..., test_start=...)` checks that every CV label, in training and in validation,
  ends before the first test week.
- **Consequences:**
  - Logistic regression (C = 0.01) and the tree (Gini, depth 2, leaf 60) keep their settings and test AUCs.
  - KNN's chosen setting changed from uniform to distance weighting. Its test AUC is now 0.561, with a CI that
    includes 0.5.
  - The CV-chosen cost thresholds changed. At r = 35 the category rule, not logistic regression, is now the
    cheapest policy.
  - The manuscript (§6.2) explains the original error and the fix.

**2. The formal evidence claims are qualified: fixed.**
- **Shuffled-label fits:** these are now described everywhere as a sanity check. The text says it is not a formal
  test of H3 and could not have caught the boundary problem.
- **Monte Carlo p:** reported as (b + 1)/(B + 1) = 0.005, never 0.
- **Row bootstrap:** labelled as treating rows as independent, and therefore too narrow.
- **Per-week check:** added (`test_auc_by_week.csv`, Appendix Table A3). Logistic regression is above 0.5 in all 4
  test weeks.
- **H3 wording:** H0 is rejected for logistic regression *in these test weeks*, explicitly not as a general result.
- **"Scored once":** replaced by "one final evaluation per fixed model". The text also notes that the diagnostics
  reuse the same test predictions.

## Should fix

**3. Eid dates: fixed.** Pakistan's observed dates are 21 March and 27 May 2026, cited to APP and Radio Pakistan
(sources supplied by the review). The text keeps them as an untested hypothesis and notes that some abnormal moves
are falls.

**4. Per-week alert budget: added.** `alert_budget` now returns pooled and per-week tables
(`alert_budget_weekly.csv`, Appendix Table A4).

**5. Confusion matrices for every scoring model: added.** Fig. 10 now has 9 panels: the majority class, plus the
category rule, logistic regression, the tree and KNN, each at 0.5 and at the r = 20 threshold.

**6. Tighter explanatory claims: done.**
- **Random vs forward CV:** the gap is now called a comparison, not a controlled measurement.
- **r = 20:** called an illustrative value judgment.
- **Cost results:** described for these test weeks only, not as a measured break-even.
- **Sample counts:** the 7,095-row data card is distinguished from the 6,384-row fitted sample.
- **PR-AUC:** labelled as average precision.
- **Class weights:** distinguished from stratification.
- **KNN scaling:** the min-max range is now described as 0-1 *on training rows*.

## Nice to have

**7. Tested environment: recorded in the README.** That section names the Python and package versions used, notes
that byte stability holds only within the same environment, and names the Edge dependency for the PDF.

**8. Page count: 15 pages.** The appendix tree diagram moved to the notebook and figure sizes were reduced. No
evaluation evidence was cut.

**9. Exact cost crossover: stated.** The text now gives r = (1 − p)/p = 32.8, next to the inverse base rate of 33.8.
