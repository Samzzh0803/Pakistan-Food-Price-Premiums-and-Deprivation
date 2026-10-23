"""Generate notebooks/M03_baseline_models.ipynb. The notebook calls the same pfp.model functions as the script.

Execute it in place with:
    cd notebooks && jupyter nbconvert --to notebook --execute --inplace M03_baseline_models.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Milestone 03: Baseline models for abnormal next-week food price movements (H3)
**Who Pays More, and Who Can Least Afford It?** CS/SDP 312/314 Data Science for Social Good, Habib University, Fall 2026.
Group 30: Yousuf Uyghur, Sameer Hassan.

**Question (H3, SDG 2.c early warning).** Using only what is known at week *t*, can we tell which city-item prices
will move abnormally between week *t* and week *t+1*?
H0: a classifier achieves ROC-AUC = 0.5 on a time-ordered test set. H1: AUC > 0.5.

This notebook runs offline from `data/processed/m02/master_city_item_week.parquet`. All functions live in
`pfp/model.py`; `scripts/m03_models.py` runs the same steps. Every number quoted in the manuscript is written to
`reports/m03/m03_numbers.json`.

Layout, following the Milestone 03 brief:
* **Part 0.** Data, label and features.
* **Part A.** Baseline models and why we chose them.
* **Part B.** Experimental setup: time-ordered split, forward-chaining cross-validation, tuning, leakage checks.
* **Part C.** Metrics, asymmetric error costs, confusion matrices and error analysis.
* **Part D.** Feature importance, policy takeaways and the Milestone 04 roadmap.

Everything here uses methods taught in the course: logistic regression, decision trees, KNN, the confusion-matrix
metrics, ROC/PR curves, the bootstrap and permutation logic. Nothing beyond the course is used.
""")
code("""
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.precision", 4)
from pfp import model
from pfp.config import TRAIN_CUTOFF
print("Training cutoff:", TRAIN_CUTOFF, "| random seed:", model.SEED)
""")

md("""
## Part 0. Data, label and features

### 0.1 Load the master dataset and keep the holdout closed
The master dataset from Milestone 02 holds only weeks up to the training cutoff (24 Sep 2026). The newer weeks in
`data/raw/pbs_holdout/` are reserved for the Week 14 new-data milestone. `load_master` asserts that no week after the
cutoff is present, and nothing in this notebook opens the holdout folder.
""")
code("""
master = model.load_master()
assert master.week_end.max() <= pd.Timestamp(TRAIN_CUTOFF)
print(master.shape, "| weeks:", master.week_end.nunique(), "| last week:", master.week_end.max().date())
""")

md("""
### 0.2 One row = one city × food item × week *t*
`build_features` keeps food rows with a price and adds the change from week *t* to the next week (`dlog_next`).
A row can only carry a label if the next PBS release is exactly one week later (6–8 days). PBS's published archive
has gaps, so only **15 of the 27 weeks** have a consecutive next week. That is a limit of the archive, not of the method.
""")
code("""
rows = model.build_features(master)
weeks = model.labelled_weeks(rows)
print(f"candidate rows: {len(rows):,} over {len(weeks)} labelled weeks")
print("labelled weeks:", [str(w)[:10] for w in weeks])
assert rows.next_week.max() <= pd.Timestamp(TRAIN_CUTOFF)
""")

md("""
### 0.3 The label: "abnormal" next-week move
`y = 1` if |log change from *t* to *t+1*| is larger than that item's own **90th percentile of non-zero changes**.
We use non-zero changes because most weekly food price changes are exactly 0: the plain 90th percentile is 0 for 13
of the 32 food items, so "abnormal" would just mean "any change". Items with fewer than 5 non-zero changes in the
training rows get no threshold and are dropped (we report how many).

**The thresholds are fitted on training rows only** (`make_labels(rows, fit_rows)`), and re-fitted inside every
cross-validation fold. If we fitted them on all weeks, the test weeks would help decide what counts as "abnormal".

The data card below uses whole-period thresholds **for description only**; the models never see them.
""")
code("""
allw, thr_all = model.make_labels(rows, rows)
print(f"labelled rows {len(allw):,}; positives {allw.y.sum()}; base rate {allw.y.mean():.4f}; "
      f"rows per week {len(allw) / len(weeks):.0f}; items with a label {allw.item_id.nunique()} of {rows.item_id.nunique()}")
print(f"share of rows with no observed change at t (has_dlog = 0): {1 - allw.has_dlog.mean():.3f}")
print(f"share of rows with no observed change at t-1 (has_lag1 = 0): {1 - allw.has_lag1.mean():.3f}")
""")

md("""
### 0.4 Features: every one is known at week *t*
No feature uses week *t+1*. A missing change (because the previous week was not published) is set to 0 **and**
flagged with its own `has_` column, so the model can tell "no change" from "not observed" (the M02 rule: flag, don't
impute). We do not add one dummy per item: with about 200 positives that would overfit and swamp KNN distances.
""")
code("""
feat = pd.DataFrame({"feature": model.FEATURES, "why it might matter": [model.FEATURE_WHY[f] for f in model.FEATURES]})
pd.set_option("display.max_colwidth", 120); feat
""")
code("""
# Check: no feature column is built from next-week information.
assert not any("next" in f for f in model.FEATURES)
rows[model.FEATURES].describe().T[["mean", "std", "min", "max"]]
""")

md("""
## Part A. Baseline models and why we chose them

| Model | Idea | Why it fits this problem |
|---|---|---|
| **Majority class** (naive benchmark 1) | always predict "normal" | shows why accuracy misleads with rare events (Unit 10, "Limitation of Accuracy") |
| **Category base-rate rule** (naive benchmark 2) | score = abnormal rate of the item's category in training weeks | a one-line domain heuristic ("perishables move more"); any ML model must beat it to be worth using |
| **Logistic regression** | linear log-odds, z-scored inputs, `class_weight="balanced"`, tune C | gives probabilities for ROC and cost thresholds, stable with ~200 positives, coefficients become odds ratios an analyst can read |
| **Decision tree (CART)** | if-then splits by Gini or entropy, `class_weight="balanced"`, tune depth and leaf size | rules a district price committee could follow without software; captures interactions; no scaling needed |
| **KNN** | vote of the K most similar rows, min-max scaled, tune K, weighting, distance | a different idea: "similar city-item situations behave alike"; no training step |

`class_weight="balanced"` gives every positive row weight n/(2·positives) and every negative n/(2·negatives), so
both classes count equally in the loss. Without it, a model can reach 97% accuracy by never predicting "abnormal".

The scalers sit **inside** each `Pipeline`, so they are refitted on the training rows of every fold and never see
validation or test rows (Unit 11, "Handling new records with a normalized model").
""")
code("""
specs = model.make_models()
for name, s in specs.items():
    print(f"{name:9s} grid size {len(s['grid']):3d} | {s['est']}")
""")

md("""
## Part B. Experimental setup and cross-validation

### B.1 Time-ordered train/test split (and why not a random stratified split)
We train on the first 11 labelled weeks (17 Jul 2025 to 26 Mar 2026) and test on the last 4 (2 Apr, 4 Jun, 13 Aug
and 20 Aug 2026): 73/27 by weeks. The rows form a panel: the same city-item series appears every week. A random split
would train on the future to predict the past, which PBS could never do in real use.

"Stratified" in the brief is about keeping the rare positive class represented in every part. We handle that by
checking positives in every fold and in the test set, and by using `class_weight="balanced"`.

**Purge rule.** A row at week *t* gets its label from week *t+1*; if *t+1* is an evaluation week, that training row's
label is built from evaluation-period prices, so we drop it. Here the last training week (26 Mar 2026) has
*t+1* = 2 Apr 2026, the first test week, so its 541 candidate rows are purged and the final model learns from 10 weeks.
""")
code("""
train_weeks, test_weeks = model.time_split(rows)
final = model.split_data(rows, train_weeks, test_weeks)
train, test = final["train"], final["eval"]
print("train weeks:", [str(w)[:10] for w in train_weeks])
print("test weeks: ", [str(w)[:10] for w in test_weeks])
print(f"purged rows: {final['n_purged']} (weeks {[str(w)[:10] for w in final['purged_weeks']]})")
print(f"items without a label (fewer than 5 non-zero training changes): {final['items_unlabelled']}")
print(f"train: {len(train):,} rows, {train.y.sum()} positives, base rate {train.y.mean():.4f}")
print(f"test:  {len(test):,} rows, {test.y.sum()} positives, base rate {test.y.mean():.4f}")
# Leakage asserts
assert train.week_end.max() < min(test_weeks) and not train.next_week.isin(test_weeks).any()
assert not set(train.week_end) & set(test_weeks)
""")
md("""
The two base rates are almost equal. The pre-test, which fitted thresholds on all weeks and did not purge, had 3.2% in
train and 2.2% in test. Fitting thresholds on training weeks only lowers some items' cut-offs, so more test rows count
as abnormal. The label is now defined only by the past, as it would be in real use.
""")

md("""
### B.2 Forward-chaining ("expanding-window") cross-validation
Inside the 11 training weeks: fold 1 trains on labelled weeks 1–6 and validates on week 7, fold 2 trains on weeks 1–7
and validates on week 8, and so on: **5 folds**. Each fold re-fits label thresholds, scaler and model on its own
training weeks and applies the purge rule. Each validation week has only 16–26 positives, so fold-to-fold variance
will be large. We report the mean ± SD.
""")
code("""
folds = model.forward_chain_folds(train_weeks)
fold_data = model.make_fold_data(rows, folds)
for d in fold_data:  # leakage asserts per fold
    assert d["train"].week_end.max() < min(d["eval_weeks"]) and not d["train"].next_week.isin(d["eval_weeks"]).any()
display(Image(model.fig_design(rows, train_weeks, test_weeks, folds, final["purged_weeks"])))
""")

md("""
### B.3 Run everything
`run_all` performs the whole pipeline with exactly the folds and split shown above: tuning, CV, the single
test-set scoring, leakage checks, costs, errors and importance. It also writes all tables, figures and the numbers
JSON. It takes about two minutes. The cells after it display and explain each piece.
""")
code("""
o = model.run_all(write=True)
N = model.N
o["fold_tab"]
""")

md("""
### B.4 Hyperparameter tuning (selected by mean validation ROC-AUC)
Each grid runs inside the 5 forward-chaining folds. We use **ROC-AUC** to tune because it does not depend on a
threshold and is not fooled by the 97% majority class. If several settings are within one fold-SD of the best mean,
we take the simplest: smallest C (strongest regularisation), smallest depth with larger leaves, or largest K.
""")
code("""display(Image(model.FIGURES / "fig16_tuning.png")); print(json.dumps(N["tune"], indent=1))""")
code("""o["tuning"]["logistic"].round(3)""")
code("""o["tuning"]["tree"].sort_values("mean_auc", ascending=False).head(10).round(3)""")
code("""o["tuning"]["knn"].sort_values("mean_auc", ascending=False).head(10).round(3)""")
md("""
**Reading the tuning results.**
* *Logistic regression.* C barely matters. The rule picks C = 0.01, the strongest regularisation in the grid.
* *Decision tree.* Depth 4 has the best mean, but depth 2 is within one SD, so the rule picks the simpler tree.
  A depth-2 tree is also easy to read as rules.
* *KNN.* AUC rises steadily with K and is best at K = 201, the edge of the grid. With a 3% positive rate, small
  neighbourhoods mostly contain no positives at all, so the vote is 0 for nearly everyone. Only very large
  neighbourhoods give a usable score, and then KNN is just a smoothed local base rate.
* *KNN features.* Using the continuous features only is clearly worse. The category dummies carry much of the signal.
  The Unit 11 problems all show up here: binary dummies count as much as real distances, irrelevant features add
  noise, and many rows are identical zeros, which creates ties.
""")

md("""
### B.5 Variance across folds
Per-fold ROC-AUC for every model (chosen settings). Note how much a single fold can swing.
""")
code("""
pf = o["per_fold"].pivot(index="model", columns="fold", values="roc_auc")
pf["mean"], pf["sd"] = pf.mean(axis=1), pf.std(axis=1, ddof=1)
pf.round(3)
""")

md("""
### B.6 Leakage demonstration: random K-fold is over-optimistic here
We repeat CV for the chosen logistic model with ordinary `StratifiedKFold(5, shuffle=True, random_state=42)` on the
training rows. Random folds put later weeks of the same city-item series into training and earlier weeks into
validation, so the model "knows" about periods it is validated on. The gap below is the optimism that a random
split would have reported.
""")
code("""
print(f"random stratified 5-fold AUC: {N['leak']['random_kfold_auc_mean']:.3f} ± {N['leak']['random_kfold_auc_sd']:.3f}")
print(f"forward-chaining AUC:         {N['leak']['forward_auc_mean']:.3f} ± {N['cv']['logistic']['roc_auc_sd']:.3f}")
print(f"gap: {N['leak']['gap']:.3f}")
""")

md("""
### B.7 Sanity check: shuffled training labels
We refit the final logistic model on randomly permuted training labels and score the real test labels. If nothing
leaks, AUC should sit near 0.5. One shuffle is noisy with only 54 test positives, so we repeat it 200 times.
The first shuffle uses seed 42. The share of shuffled AUCs that reach the real AUC is also a permutation p-value
for H3.
""")
code("""
pa = o["shuffled_aucs"]
print(f"seed-42 shuffle AUC {pa[0]:.3f}; mean over {len(pa)} shuffles {pa.mean():.3f} (SD {pa.std(ddof=1):.3f}); "
      f"5th-95th percentile {np.percentile(pa, 5):.3f}-{np.percentile(pa, 95):.3f}")
print(f"real logistic test AUC {N['test']['logistic']['roc_auc']:.3f}; share of shuffles >= real: {N['leak']['shuffled_share_ge_real']:.3f}")
assert abs(pa.mean() - 0.5) < 0.03
""")

md("""
## Part C. Metrics, asymmetric costs and confusion matrices

### C.1 Main metrics: CV (mean ± SD over 5 folds) and test (scored once)
Threshold 0.5 for the threshold-based metrics. **PR-AUC** matters here because, with 3% positives, ROC-AUC can look
decent while precision stays tiny. A random model's PR-AUC equals the base rate (0.03), so compare against that, not
against 0.5. The **bootstrap 95% CI** for test ROC-AUC uses 2,000 resamples of test rows (seed 42). If its lower
bound is above 0.5, we reject H0 for that model.
""")
code("""
mm = o["main"].set_index("model")
cols = []
for c in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
    mm[f"CV {c}"] = mm[f"cv_{c}_mean"].map("{:.3f}".format) + " ± " + mm[f"cv_{c}_sd"].map("{:.3f}".format)
    cols.append(f"CV {c}")
mm["test ROC-AUC [95% CI]"] = mm.test_roc_auc.map("{:.3f}".format) + " [" + mm.test_auc_ci_lo.map("{:.3f}".format) + ", " + mm.test_auc_ci_hi.map("{:.3f}".format) + "]"
display(mm[cols])
display(mm[["test_accuracy", "test_precision", "test_recall", "test_specificity", "test_f1", "test ROC-AUC [95% CI]", "test_pr_auc", "test_flagged_at_0.5"]].round(3))
""")
code("""display(Image(model.FIGURES / "fig17_roc_pr.png"))""")
code("""
t = N["test"]
print(f"logistic minus category test AUC: {t['diff_logistic_minus_category']:.3f}, paired bootstrap 95% CI "
      f"[{t['diff_logistic_minus_category_ci_lo']:.3f}, {t['diff_logistic_minus_category_ci_hi']:.3f}]")
print("folds where logistic beats the category rule:", N["cv"]["folds_logistic_beats_category"], "of", N["cv"]["folds"])
print("H3 rejected for", N["h3"]["n_models_reject"], "of 3 ML models (CI lower bound > 0.5)")
""")
md("""
**Reading the table.**
* *Accuracy.* The majority class is about 97% accurate and catches nothing. The category rule and KNN never produce
  a score of 0.5 or more, so at that threshold they also flag nothing.
* *H3.* All three ML models have a test ROC-AUC confidence interval above 0.5, so H0 is rejected.
* *The simple rule.* The **category base-rate rule is as good as logistic regression on the test set**: their
  difference has a confidence interval spanning 0. On average across CV folds, logistic regression was ahead.
* *Practical lesson.* Most of the usable signal is "what kind of food is it". Recent price movement adds a little
  on top of that.
* *Caveat.* The bootstrap resamples rows, but rows from the same week share shocks. The CIs are therefore probably
  too narrow.

The ablation below asks directly whether recent price movements add anything beyond the item category.
""")
code("""o["ablation"].round(3)""")
md("""
**Sensitivity to the label rule.** With the M02 "contract" rule (plain 90th percentile, where any change counts for
13 items), the base rate is higher and the task changes. We report it only as a check.
""")
code("""N["contract"]""")

md("""
### C.2 Asymmetric error costs (Provost & Fawcett, Ch. 7)
**Who uses the alert?** A PBS analyst, or a district Price Control Committee or price magistrate deciding where to
send inspectors or which items to watch next week.

* **False negative (missed abnormal move):** a jump in tomatoes, onions, flour or ghee that nobody was warned
  about. No early stock release, inspection or public notice happens. Households absorb the shock, and poorer
  households spend a larger share of income on food, so the harm falls on those least able to absorb it.
* **False positive (false alarm):** an inspection visit or watch-list entry that was not needed. It costs staff
  time and may cause friction with traders. Many false alarms lead to "alert fatigue": people stop trusting the
  system.

**Cost matrix.** We put the cost in one cell only, to avoid double counting.

| | predicted abnormal | predicted normal |
|---|---|---|
| **actual abnormal** | 0 (TP) | *r* (FN) |
| **actual normal** | 1 (FP) | 0 (TN) |

*r* is a value judgment made by people, not something the data decides, so we run a sensitivity analysis over
*r* ∈ {1, 5, 10, 20, 35}. **Intuition.** Flagging a row with probability *p* of being abnormal is worth it when
*p*·*r* > (1 − *p*), i.e. when *p* > 1/(1 + *r*). With a base rate near 3%, an average row passes that test only when
*r* is about 32 or more, roughly the inverse base rate. So around *r* ≈ 35 "flag everything" starts to look rational.

**Procedure.** For each *r*, we pick each model's threshold to minimise cost on the **pooled CV out-of-fold
predictions**, then apply that threshold to the test set once.
""")
code("""
c = o["costs"]
display(c.pivot(index="policy", columns="r", values="test_cost_per_1000").round(1))
display(c.pivot(index="policy", columns="r", values="test_recall").round(3))
display(Image(model.FIGURES / "fig19_costs.png"))
""")
md("""
**Reading the cost results.**
* *r ≤ 10.* The cheapest policy is to **flag nothing**: no model is confident enough. At *r* = 10 the logistic
  model's CV threshold flags a few rows, and on test that costs slightly more than flagging nothing.
* *r = 20.* Logistic regression and "flag nothing" cost about the same.
* *r = 35.* Logistic regression is the cheapest policy.

So the model only pays off if a missed shock is judged to cost roughly 20 or more false alarms. Whoever deploys it
has to make that call; we cannot.
""")
code("""
print("Metrics at the cost-based threshold, r =", model.R_REF, "(threshold chosen on CV, applied to test)")
o["at_ref"].round(3)
""")

md("""
### C.3 Alert-budget view
An analyst can check only the top *N* of the roughly 456 city-items each week. How many real abnormal moves would
each model put in front of them? The category rule gives tied scores, so we report the expected count under random
tie-breaking.
""")
code("""o["budget"].round(3)""")

md("""
### C.4 Equity weighting (preview of the M04 fairness audit)
We repeat the cost analysis with the FN cost multiplied by the city's deprivation tercile weight (low 1, mid 1.5,
high 2): a missed shock in a more deprived city counts more.
""")
code("""ce = o["costs_eq"]; display(ce.pivot(index="policy", columns="r", values="test_cost_per_1000").round(1)); N["cost_equity"]""")

md("""
### C.5 Confusion matrices (Unit 10 layout: actual in rows, predicted in columns, positive class first)
""")
code("""display(Image(model.FIGURES / "fig18_confusion.png")); o["cm_tab"].round(3)""")

md("""
### C.6 Where the errors fall (logistic regression, cost threshold *r* = 20), descriptive only
FN rate = missed share of real abnormal moves; FP rate = false-alarm share of normal rows. The formal fairness test
is Milestone 04. Group sizes are small (a few positives per province), so read these as patterns, not findings.
""")
code("""o["errors"].round(3)""")

md("""
### C.7 Edge cases: the most confident mistakes
Ten most confident false negatives (lowest score among missed abnormal moves) and ten most confident false positives.
""")
code("""display(o["fn_cases"].round(3)); display(o["fp_cases"].round(3))""")
code("""{k: v for k, v in N["errors"].items() if k.startswith(("fn_", "tp_", "n_", "positives"))}""")
md("""
**Patterns.**
* *Missed moves (FN).* Every missed abnormal move at this threshold had a flat or unobserved price at week *t*
  (`price_changed = 0`). Nothing in the recent history pointed to a move. Moves that come "out of the blue" are
  invisible to a model built on recent changes.
* *False alarms (FP).* The most confident false alarms are perishables in Sukkur, Bannu and Larkana, mostly
  records with one effective quote (`single_quote = 1`) whose price changed this week and then stayed put. A single
  quote that jumps may be noise in one shop rather than a market move. That links to the H4 data-quality concern.
* *Event timing (unverified).* The two earlier test weeks (2 Apr and 4 Jun 2026) have about twice the base rate of
  the two August weeks. Each falls within about two weeks of an Eid holiday: Eid al-Fitr is expected around
  20 March 2026 and Eid al-Adha around 27 May 2026. Post-festival price adjustment is **one possible explanation,
  which we have not tested**.
""")

md("""
## Part D. Feature importance and initial insights

### D.1 Logistic regression: standardised coefficients and odds ratios per 1 SD
Inputs are z-scored inside the pipeline, so each coefficient is the change in log-odds for a one-standard-deviation
increase. The odds ratio = e^coef. These are associations, not causes.
""")
code("""imp = o["importance"]; display(imp.round(3)); display(Image(model.FIGURES / "fig20_importance.png"))""")
code("""
for r in imp.head(5).itertuples():
    direction = "higher" if r.odds_ratio_per_sd > 1 else "lower"
    print(f"{r.feature}: one SD higher is associated with {r.odds_ratio_per_sd:.2f} times the odds of an abnormal move next week ({direction})")
print("correlation between abs_dlog and price_changed in training rows:", round(N["imp"]["corr_abs_dlog_price_changed"], 3))
""")
md("""
**Reading the coefficients.**
* *Item type.* Branded and packaged goods are much less likely to move abnormally; perishables more likely.
* *Recent movement.* A price that changed this week (`price_changed`) is more likely to move abnormally next week.
  `abs_dlog` gets only a small coefficient because it is correlated with `price_changed` (r about 0.5), and the two share
  the credit (the Unit 09 point about correlated features splitting importance).
* *Market size.* Larger markets (`log_pop_urban`, Islamabad) are associated with fewer abnormal moves.
* *Observation flags.* The `has_` flags are negative. They mostly mark *which weeks* had a published previous
  week, so they partly stand in for time period. They are not a market effect.
""")

md("""
### D.2 Decision tree: importance, rules and a worked Gini example
""")
code("""print(o["rules_text"]); display(Image(model.FIGURES / "fig21_tree.png")); o["rules"]""")
md("""
The plotted `value` is the **class-weighted** count, because the tree was fitted with balanced class weights.
The rule table gives the actual (unweighted) abnormal rate of each leaf on the training rows. The top rule reads:
**IF the item is not branded/packaged AND its price moved by more than about 1% this week, THEN the chance of an
abnormal move next week is about 7%**, more than twice the base rate.

**Worked Gini example (root split).** Gini = 1 − p² − (1 − p)². The split's gain is the parent's Gini minus the
size-weighted Gini of the two children. sklearn computes this with the class weights, which make the root
exactly 50/50.
""")
code("""
g = o["gini_example"]
print(f"root split: {g['feature']} <= {g['threshold']}")
print(f"unweighted counts: parent {g['pos']}/{g['n']} positive -> left {g['pos_left']}/{g['n_left']}, right {g['pos_right']}/{g['n_right']}")
print(f"unweighted Gini: parent {g['gini_parent']:.4f}, left {g['gini_left']:.4f}, right {g['gini_right']:.4f}, weighted children {g['gini_weighted_children']:.4f}, gain {g['gini_gain']:.4f}")
print(f"class weights: positive {g['w_pos']:.2f}, negative {g['w_neg']:.3f}")
print(f"weighted Gini (what sklearn used): parent {g['wgini_parent']:.4f}, left {g['wgini_left']:.4f}, right {g['wgini_right']:.4f}, "
      f"left share of weight {g['wshare_left']:.3f}, children {g['wgini_weighted_children']:.4f}, gain {g['wgini_gain']:.4f}")
""")

md("""
### D.3 Cross-model check
Features that rank high in both models are the most trustworthy drivers: the branded/packaged flag (both models) and
recent price movement (`price_changed` in the logistic model, `abs_dlog` in the tree, which are two views of the same
signal). A depth-2 tree can use only three features, so most features have no tree importance.
""")
code("""imp[["feature", "rank_logistic", "rank_tree", "odds_ratio_per_sd", "tree_importance"]].round(3)""")

md("""
### D.4 Initial policy takeaways (SDG 2.c), hedged
1. **Where to look.** Concentrate weekly monitoring on perishables, especially those whose price already moved this
   week. Branded and packaged goods almost never move abnormally.
2. **Triage aid, not a trigger.** With test AUC around 0.7 and precision around 5%, most alerts will be false.
   The model can rank where an inspector looks first; it should not trigger action automatically.
3. **A simple rule already captures most of the signal.** On the test weeks, "watch perishables" does as well as
   the logistic model. Any ML system must beat that rule to justify its cost.
4. **Publish every week.** 40% of labelled rows have no previous consecutive week, because PBS's published weekly
   archive has gaps, and the moves we miss are the ones with no visible recent history. Consistent publication of
   the weekly annexure is a cheap way to improve early warning.
5. **The threshold is a policy choice.** Whether alerts are worth issuing depends on how many false alarms one
   missed shock is worth (*r*). Below about 20, issuing no alerts is cheaper. That judgment belongs to decision-makers.

### D.5 Limitations and the Milestone 04 roadmap
**Limitations**
* Only about 200 positives over 15 labelled weeks; 4 test weeks with 54 positives, so wide confidence intervals.
* The bootstrap ignores clustering by week, so the CIs are probably too narrow.
* Gaps in the archive; class imbalance; modest signal; KNN unsuitable.
* The 90th-percentile threshold is a convention.
* City-level external variables are constant within a city. Islamabad and Rawalpindi function as one market.
* Possible place-based bias in who gets missed.

**Milestone 04**
* Tree ensembles (random forest, boosting) compared against today's baselines.
* Fairness audit (H4): recall, FN rate and calibration across province, population tercile and deprivation
  tercile, with a hypothesis test.
* Thresholds that equalise miss rates, and an ethical evaluation of who bears the false negatives.
* The Week 14 design: first score the frozen model on the post-cutoff holdout weeks, then retrain on the earlier part
  of the new data and test on a later, unseen part.
""")
code("""
# Final checks before the numbers are used in the manuscript.
assert N["data"]["max_week_used"] <= str(TRAIN_CUTOFF)
assert abs(N["leak"]["shuffled_mean"] - 0.5) < 0.03
print("all checks passed; numbers file:", model.NUMBERS_JSON.relative_to(ROOT))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "M03_baseline_models.ipynb"
nbf.write(nb, out)
print("wrote", out)
