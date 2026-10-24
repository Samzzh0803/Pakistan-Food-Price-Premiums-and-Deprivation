# Milestone 03 viva preparation

Likely questions for the 5-minute viva, with short answers in plain words. "NB" points to the section heading in
`notebooks/M03_baseline_models.ipynb`; functions are in `pfp/model.py`. Numbers are from `reports/m03/m03_numbers.json`.

---

**1. What exactly are you predicting?**
One row is one city × food item × week *t*. `y = 1` if the price moves "abnormally" between week *t* and the next
weekly release *t+1*. Abnormal means the absolute log change is bigger than that item's own 90th percentile of
non-zero changes in the training weeks. The positive class is the abnormal move, because that is what a price
monitor wants to catch.
- **Description:** with whole-period thresholds there are 7,095 labelled rows, 205 abnormal (2.9%). This is for
  description only.
- **Fitted sample:** the models use 4,560 training rows and 1,824 test rows (6,384 in total). Thresholds there are
  fitted on training rows only, which leaves 5 items unlabelled instead of 4, and the purged week is removed.
*NB 0.3; `make_labels`, `build_features`.*

**2. Why the 90th percentile of non-zero changes and not of all changes?**
75% of weekly food price changes are exactly 0. The plain 90th percentile is 0 for 13 of the 32 food items, so
"abnormal" would just mean "changed at all". Using non-zero changes gives each item a meaningful cut-off. The
plain rule is reported as a sensitivity check: test AUC 0.585.
*NB C.1 "Sensitivity to the label rule".*

**3. Why only 15 labelled weeks out of 27?**
A label needs the next week's price, and only 15 weeks are followed by a PBS release 6-8 days later. The rest are
followed by gaps in PBS's published archive. That is a limit of the data source, not of the method.
*NB 0.2; `build_features`.*

**4. Why time-ordered cross-validation instead of random K-fold?**
The rows are a panel: the same city-item series recurs every week, and items in the same week share shocks. Random
folds train on later weeks to predict earlier ones, which can never happen in real use, and they let the model learn
week-specific shocks seen on both sides. We use forward chaining: train on weeks 1..k, validate on week k+1, for
5 folds. We compared the two: random stratified 5-fold gives AUC 0.710, forward chaining 0.641. That gap points
the way we expected. It is a comparison, not a clean measurement of leakage, because the two schemes also differ
in training size and label fitting.
*NB B.2 and B.6; `forward_chain_folds`, `random_kfold_auc`.*

**5. The brief asks for stratified sampling. Did you ignore it?**
No. Stratification exists to keep the rare class present in every part. We keep time order and achieve that goal
differently. Every validation week has positives (16-28), and the test set has 54. Logistic regression and the
tree also use `class_weight="balanced"`. Class weights change what the model optimises; they are not the same as
stratifying, which is why we check the positive counts. A random stratified split would break time order, which
matters more here.
*NB B.1; `make_fold_data` asserts each fold has positives.*

**6. What does the purge rule do?**
A training row at week *t* gets its label from week *t+1*. If *t+1* is a validation or test week, that label is
built from evaluation-period prices, so we drop the row. In the final split, the last training week (26 Mar 2026)
is followed by the first test week (2 Apr 2026), so its 541 rows are purged and the final model learns from 10
weeks. Four of the five CV folds also lose their last training week.
**Outer boundary (fixed after review).** For the same reason, that week cannot be a CV *validation* week either:
its labels come from a test-week price. Our first version used it as fold 5, so test-period prices shaped tuning
and threshold choice. Codex's review caught this. CV now uses only the 10 development weeks whose *t+1* is before
the test period, and `make_fold_data` asserts it.
*NB B.1; `split_data`.*

**7. Why are the label thresholds refitted in every fold?**
The threshold decides what counts as abnormal. If we fitted it on all weeks, the validation or test weeks would
help define their own labels, which is a form of leakage. `split_data` fits thresholds on the purged training rows
only, then applies them unchanged to the evaluation rows. A side effect is visible: the test set has 54 positives,
whereas whole-period thresholds would give a different count.
*`make_labels(rows, fit_rows)`, called inside `split_data`.*

**8. Why ROC-AUC for tuning?**
It measures how well the model *ranks* abnormal rows above normal ones, at every threshold at once, so it does not
depend on the arbitrary 0.5 cut-off. It is also not fooled by class imbalance: the majority class gets 0.5 even
though it is 97% accurate. We choose the threshold separately, from costs.
*NB B.4; `tune`.*

**9. Why also PR-AUC?**
With 3% positives, a model can have a decent ROC-AUC while its precision stays tiny, because the false-positive
*rate* is divided by a huge number of negatives. PR-AUC focuses on the positives. A random model's PR-AUC equals
the base rate (0.030 on test), so that is the comparison. Ours is about 0.05: better than random, but low.
*NB C.1.*

**10. What does `class_weight="balanced"` do?**
It weights each positive row by n/(2 × positives) and each negative by n/(2 × negatives), so both classes carry
equal total weight in the loss. In our training set that is about 16.9 per positive and 0.52 per negative.
Without it, the model can get 97% accuracy by never predicting "abnormal". A side effect: predicted probabilities
are shifted up, so 0.5 is not a natural threshold.
*NB Part A; `gini_example` prints the weights.*

**11. How does the decision tree choose a split? Work through one.**
For each feature and cut-off, the tree computes Gini = 1 − p² − (1 − p)² in each child and takes the size-weighted
average; the split with the largest drop from the parent wins.
- **Unweighted root:** 135 positives in 4,560 rows (Gini 0.0575). Splitting on `cat_branded_packaged` sends 130 of
  3,540 rows left and 5 of 1,020 right. Children Gini 0.0707 and 0.0098, weighted 0.0571, gain 0.0004.
- **With balanced weights** (what sklearn actually used): the root is 50/50 (Gini 0.5), the children have Gini
  0.494 and 0.239, the left child holds 86.7% of the weight, and the gain is 0.040.

*NB D.2; `gini_example`.*

**12. What does an odds ratio mean in your table?**
Inputs are z-scored, so a coefficient is the change in log-odds for a one-standard-deviation increase, and
e^coef is the odds ratio. Example: `price_changed` has OR 1.47, so one SD more is associated with about 1.47 times
the odds of an abnormal move next week. `cat_branded_packaged` has OR 0.60, i.e. 40% lower odds. These are
associations, not causes.
*NB D.1; `importance`.*

**13. Why does KNN need scaling, and why did it do badly?**
KNN uses distances, so a feature with a large range (like the within-city range in %) would dominate a 0/1 dummy.
Min-max scaling puts every feature on 0-1 *for the training rows*; it is fitted on training rows only, inside
the `Pipeline`, so test values can fall slightly outside 0-1.
It still struggles because:
- with 3% positives, most small neighbourhoods contain no abnormal row, so the vote is 0;
- dummies count as much as real price changes, and irrelevant features add noise;
- many rows are identical, which produces ties.

AUC kept rising up to K = 201, the edge of the grid, where KNN is just a smoothed local base rate. Test AUC 0.561,
and its CI includes 0.5.
*NB B.4; `make_models`.*

**14. How did you choose the decision threshold, and why not 0.5?**
0.5 is arbitrary, especially with balanced weights. We set cost(FP) = 1 and cost(FN) = r. For each r, we picked
the threshold that minimises FP + r × FN on the pooled out-of-fold CV predictions, never on the test set. We then
applied it to the test set once.
*NB C.2; `best_threshold`, `cost_analysis`.*

**15. What is r and who should decide it?**
r is how many false alarms one missed abnormal move is worth. It is a value judgment, so we show r = 1, 5, 10, 20
and 35. Flagging a row is worth it when p > 1/(1 + r). For an average row (p = 2.96%), "flag everything" beats
"flag nothing" exactly when r > (1 − p)/p ≈ 32.8. Our results, in these test weeks:
- for r ≤ 10, flagging nothing is cheapest;
- at r = 20, logistic regression and flagging nothing are essentially tied (591.6 vs 592.1 per 1,000);
- at r = 35, the category rule is cheapest.

r = 20 is used in the confusion matrices only as an illustrative reference.

A ministry or price committee should set r, not the analysts.
*NB C.2; Table 4 of the manuscript (Table 14 of the technical report).*

**16. Did any ML model beat the simple category rule?**
Not convincingly. In CV, logistic regression is ahead on average (0.641 vs 0.617), but only in 3 of 5 folds. On
test, the rule is slightly ahead (0.708 vs 0.692), and the paired bootstrap difference CI (−0.074 to 0.041)
includes 0. On cost, the rule is also the cheapest policy at r = 35. The practical lesson: most of the signal is
"what kind of food is it". The ablation shows the same.
*NB C.1.*

**17. Is H3 rejected?**
Yes, for logistic regression in these test weeks, with a caveat.
- **Bootstrap:** the row-level 95% CI for test ROC-AUC is above 0.5 for logistic regression (0.636-0.746) and the
  tree (0.576-0.650), but not for KNN (0.487-0.636).
- **Why that is not enough:** the bootstrap treats rows as independent, and rows in the same week share shocks, so
  these intervals are too narrow.
- **Per-week check:** scored week by week, logistic regression is above 0.5 in all 4 test weeks (0.54-0.75).

Four weeks is evidence of a modest signal, not an established general result. The Week 14 holdout is the next
test. The shuffled-label check is a sanity check on the pipeline, not a formal test of H3.
*NB C.1 and B.7; `bootstrap_auc`.*

**18. What does the shuffled-label test prove?**
If we destroy the link between features and labels and still get AUC above 0.5, something outside the features
must be leaking the answer. Over 200 shuffles, the mean test AUC is 0.503. A single shuffle varies a lot (the
seed-42 one gave 0.583), because the test set has only 54 positives, which is why we repeat it.
What it does *not* prove: that nothing leaks. It passed even while the CV boundary problem (Q6) was present,
because that problem affected model selection, not the test labels. It is also not a formal p-value for H3. If
quoted as a Monte Carlo p, use (0 + 1)/(200 + 1) ≈ 0.005, never 0.
*NB B.7; `shuffled_label_test`.*

**19. How did you extract feature importance?**
- **Logistic regression:** standardised coefficients from the fitted pipeline (`coef_`) and their exponentials (odds
  ratios).
- **Tree:** `feature_importances_`, the total weighted Gini decrease each feature produced, normalised to sum to 1.
- **Cross-check:** we compare the two rankings. Correlated features such as `abs_dlog` and `price_changed`
  (r ≈ 0.5) split credit between them.

*NB D.1-D.3; `importance`, `tree_rules`.*

**20. How will the Week 14 holdout be used?**
The post-cutoff weeks in `data/raw/pbs_holdout/` have never been opened. In Week 14 we will:
1. score the frozen model on them, with no refitting, as a true out-of-time test;
2. retrain on an earlier part of the new data and test on a later, unseen part.

Every M03 script asserts that no week after 24 Sep 2026 is used.
*`load_master`, `build_features` asserts.*

---

**Quick facts:**
- train 4,560 rows / 135 positives; test 1,824 rows / 54 positives
- base rate 2.96% in both
- CV uses 10 development weeks, 5 folds
- chosen settings: logistic C = 0.01; tree Gini, depth 2, leaf 60; KNN K = 201, distance-weighted, Manhattan, all
  features
- "PR-AUC" is sklearn's average precision, a step-wise estimate of the area under the PR curve
- Eid-ul-Fitr was observed in Pakistan on 21 March 2026 and Eid-ul-Adha on 27 May 2026; any holiday link to the test
  weeks is an untested hypothesis
- every missed abnormal move at the r = 20 threshold had a flat or unobserved price at week *t*
