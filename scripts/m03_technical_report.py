"""Assemble the extended Milestone 03 technical report (HTML -> PDF via headless Edge).

This is the long version kept for reference and viva preparation; the submitted manuscript is built by
scripts/m03_report.py.

The brief asks for the M03 report to be embedded into the M02 manuscript draft. Sections 1-4 are therefore taken from
reports/m02/M02_report.html (their numbers come from m02_numbers.json), with Section 4's plan updated to say what was
done. Sections 5-10 are new. Every M03 number in the text is read from reports/m03/m03_numbers.json and every table
is rendered from CSVs written by scripts/m03_models.py. Nothing is hand-typed.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "m03"
TAB = REP / "tables"
M02_HTML = ROOT / "reports" / "m02" / "M02_report.html"
N = json.loads((REP / "m03_numbers.json").read_text(encoding="utf-8"))
N2 = json.loads((ROOT / "reports" / "m02" / "m02_numbers.json").read_text(encoding="utf-8"))


def g(path, src=None):
    d = N if src is None else src
    for k in path.split("."):
        d = d[k]
    return d


def f(path, d=3):
    return f"{g(path):,.{d}f}"


def pct(path, d=1):
    return f"{100 * g(path):.{d}f}%"


def n_(path):
    return f"{g(path):,}"


def t(name, **kw):
    return pd.read_csv(TAB / f"{name}.csv", **kw)


def html_table(df: pd.DataFrame, caption: str, cls: str = "", digits: int = 3) -> str:
    body = df.to_html(index=False, border=0, float_format=lambda x: f"{x:,.{digits}f}", escape=True, na_rep="–")
    return f'<div class="tbl {cls}"><p class="cap">{caption}</p>{body}</div>'


def fig(name: str, caption: str, width: str = "100%") -> str:
    return (f'<figure><img src="figures/{name}.png" style="width:{width}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


LAB = {"majority": "Majority class", "category": "Category base-rate rule", "logistic": "Logistic regression",
       "tree": "Decision tree", "knn": "KNN"}
ci = lambda m: f"{f(f'test.{m}.roc_auc')} [{f(f'test.{m}.auc_ci_lo')}, {f(f'test.{m}.auc_ci_hi')}]"
cv = lambda m, c="roc_auc": f"{f(f'cv.{m}.{c}_mean')} ± {f(f'cv.{m}.{c}_sd')}"

# ---------------------------------------------------------------- M02 sections 1-4, embedded
m02 = M02_HTML.read_text(encoding="utf-8")
css = re.search(r"<style>(.*?)</style>", m02, re.S).group(1)
body = m02[m02.index("<h2>1. Project alignment"):m02.index("<h2>5. Limitations</h2>")]
body = body.replace('src="figures/', 'src="../m02/figures/')
body = body.replace("are in Appendix Table A1.", "are in Appendix Table A1 of the Milestone 02 report.")
old_intro = body[body.index("<p>Each hypothesis is mapped"):body.index("<ul>", body.index("<p>Each hypothesis is mapped"))]
body = body.replace(old_intro, f"""<p>Each hypothesis is mapped to master-dataset columns in Table 9. Targets marked † are derived from existing columns,
with the rule for each fixed in Milestone 02. Milestone 03 builds the final H3 label (Section 5) and tests H3 (Sections 6-8); the
base rate and row count quoted for H3 below are the whole-period description from Milestone 02, which the models never use.</p>
""")
body = body.replace("abnormal_next_week † (derived provisionally; final in M03)", "abnormal_next_week † (final rule in §5)")
body = body.replace("dlog_price, dlog_price_lag1, price_changed, rel_price, within_city_range_pct, single_quote, cat_* dummies",
                    "|dlog_price|, |dlog_price_lag1| with observed flags, price_changed, rel_price, within_city_range_pct, single_quote, cat_* dummies")
body = body.replace("<td>prov_* dummies, log_pop_urban, iso_week</td>", "<td>prov_* dummies, log_pop_urban, deprivation_composite</td>")
plan_start = body.index("<p><b>Analytical plan.</b>")
plan_end = body.index("</p>", body.index("The notebook also contains exploratory")) + 4
body = body[:plan_start] + f"""<p><b>Analytical plan: what Milestone 03 did.</b> Milestone 03 is the classification milestone, so it tests H3.
<ul>
<li><b>Done:</b> logistic regression, a decision tree and KNN against two naive benchmarks. Validation uses a time-ordered split and
forward-chaining cross-validation, because random folds leak later weeks of the same city-item series (§6.4 measures how much).
Models are evaluated with ROC-AUC, PR-AUC and the full Unit 10 metrics, not accuracy alone: a model that always predicts "normal" is
{100 * (1 - g('split.test_rate')):.0f}% accurate on the test weeks. Errors are broken down by city group as a descriptive preview of H4 (§8).</li>
<li><b>Moved to Milestone 04:</b> tree ensembles (beyond the methods covered so far), the formal H4 fairness test, and the fixed-effects
estimate of H1. H2 and H5 were tested descriptively in Milestone 02.</li>
<li><b>Still closed:</b> the post-cutoff holdout. It is opened once, in the Week 14 new-data milestone.</li>
</ul></p>""" + body[plan_end:]

# ---------------------------------------------------------------- tables from the pipeline
feats = t("features").rename(columns={"why_it_might_matter": "why it might matter (all known at week t)"})
card = t("data_card")
card_tab = card.rename(columns={"part": "part", "share_has_dlog_0": "no change observed at t", "share_has_lag1_0": "no change observed at t−1",
                                "base_rate": "base rate"})
folds = t("folds")
main = t("metrics_main")
cvp = t("cv_per_fold")
costs = t("costs")
budget = t("alert_budget")
errs = t("error_breakdown")
fn_c, fp_c = t("edge_false_negatives"), t("edge_false_positives")
imp = t("importance")
rules = t("tree_rules")
ablation = t("ablation")
at_ref = t("metrics_at_cost_threshold")
cm = t("confusion_matrices")

models_tab = pd.DataFrame([
    ("Majority class (benchmark 1)", "Always predicts 'normal'", "Shows that accuracy misleads on rare events (Unit 10)", "none", "–"),
    ("Category base-rate rule (benchmark 2)", "Score = item category's abnormal rate in training weeks",
     "One-line domain heuristic: 'perishables move more'. A model must beat it to be worth deploying", "none", "–"),
    ("Logistic regression", "Linear log-odds; z-scored inputs; balanced class weights",
     "Probabilities for ROC and cost thresholds; stable with few positives; odds ratios an analyst can read", "C ∈ {0.01, 0.1, 1, 10}",
     g("tune.logistic.chosen_params")),
    ("Decision tree (CART)", "Recursive if-then splits; balanced class weights",
     "Rules a price committee can follow without software; captures interactions; no scaling", "criterion ∈ {gini, entropy}; depth 2-6; min leaf ∈ {10, 30, 60}",
     g("tune.tree.chosen_params")),
    ("KNN", "Vote of the K nearest rows; min-max scaling fitted on training rows",
     "A different idea: similar city-item situations behave alike; no training step", "K ∈ {5, 15, 25, 51, 101, 201}; uniform/distance; Euclidean/Manhattan; all vs continuous features",
     g("tune.knn.chosen_params")),
], columns=["Model", "How it works", "Why it fits this problem", "Grid (tuned in forward-chaining CV)", "Chosen"])

test_tab = pd.DataFrame([{
    "model": LAB[k], "accuracy": g(f"test.{k}.accuracy"), "precision": g(f"test.{k}.precision"), "recall": g(f"test.{k}.recall"),
    "specificity": g(f"test.{k}.specificity"), "F1": g(f"test.{k}.f1"), "ROC-AUC [95% CI]": ci(k), "PR-AUC": g(f"test.{k}.pr_auc"),
    "flagged at 0.5": g(f"test.{k}.flagged_at_05")} for k in LAB])
cv_tab = pd.DataFrame([{"model": LAB[k], **{c: cv(k, m) for c, m in [("accuracy", "accuracy"), ("precision", "precision"),
                        ("recall", "recall"), ("specificity", "specificity"), ("F1", "f1"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]}}
                       for k in LAB])

pol_lab = {**LAB, "flag_nothing": "Flag nothing (= majority class)", "flag_everything": "Flag everything"}
cost_tab = costs.pivot(index="policy", columns="r", values="test_cost_per_1000")
rec_tab = costs.pivot(index="policy", columns="r", values="test_recall")
order = ["category", "logistic", "tree", "knn", "flag_nothing", "flag_everything"]
cost_show = pd.DataFrame({"policy": [pol_lab[p] for p in order]})
for r in sorted(costs.r.unique()):
    cost_show[f"r = {r}"] = [f"{cost_tab.loc[p, r]:.0f} ({100 * rec_tab.loc[p, r]:.0f}%)" for p in order]

bud = budget.assign(model=budget.model.map(LAB)).pivot(index="model", columns="N_per_week", values="recall_at_N")
bud_p = budget.assign(model=budget.model.map(LAB)).pivot(index="model", columns="N_per_week", values="precision_at_N")
bud_show = pd.DataFrame({"model": bud.index})
for n in bud.columns:
    bud_show[f"top {n}/week: recall"] = bud[n].values
    bud_show[f"top {n}/week: precision"] = bud_p[n].values

err_show = errs[errs.grouping.isin(["item category", "city-size tercile", "deprivation tercile", "test week"])].copy()
err_show = err_show[["grouping", "group", "rows", "positives", "base_rate", "flagged", "fn_rate", "fp_rate"]].rename(
    columns={"base_rate": "base rate", "fn_rate": "FN rate", "fp_rate": "FP rate"})
prov_show = errs[errs.grouping == "province"][["group", "positives", "fn_rate", "fp_rate"]]
edge_cols = ["city", "item_short", "week", "score", "dlog_next", "threshold", "abs_dlog", "price_changed", "single_quote"]
edge = pd.concat([fn_c.head(5)[edge_cols].assign(error="FN"), fp_c.head(5)[edge_cols].assign(error="FP")])
edge = edge[["error"] + edge_cols].rename(columns={"item_short": "item", "dlog_next": "next-week change", "abs_dlog": "|change| at t",
                                                   "price_changed": "changed at t", "single_quote": "MIN = MAX"})
imp_show = imp[["feature", "coef_per_sd", "odds_ratio_per_sd", "rank_logistic", "tree_importance", "rank_tree"]].rename(
    columns={"coef_per_sd": "coef per SD", "odds_ratio_per_sd": "odds ratio per SD", "rank_logistic": "rank (logistic)",
             "tree_importance": "tree importance", "rank_tree": "rank (tree)"})
rules_show = rules[["rule", "train_rows", "train_positives", "abnormal_rate"]].rename(
    columns={"train_rows": "training rows", "train_positives": "abnormal", "abnormal_rate": "abnormal rate"})
cvp_show = cvp.pivot(index="model", columns="fold", values="roc_auc").reindex([LAB[k] for k in LAB])
cvp_show.columns = [f"fold {c}" for c in cvp_show.columns]
cvp_show = cvp_show.reset_index()
folds_show = folds[["fold", "train_weeks", "val_week", "purged_rows", "train_rows", "train_pos", "val_rows", "val_pos", "items_unlabelled"]].rename(
    columns={"train_weeks": "training weeks (t)", "val_week": "validation week", "purged_rows": "purged", "train_rows": "train rows",
             "train_pos": "train pos.", "val_rows": "val. rows", "val_pos": "val. pos.", "items_unlabelled": "items w/o label"})

top5 = imp.sort_values("rank_logistic").head(5)
pretty = {"cat_branded_packaged": "being a branded/packaged item", "price_changed": "a price change this week",
          "cat_perishable": "being a perishable", "has_dlog": "having an observed change this week (previous week published)",
          "prov_ICT": "being in Islamabad", "has_lag1": "having an observed change last week", "log_pop_urban": "a larger urban population",
          "deprivation_composite": "higher district deprivation"}
or_sentences = "".join(
    f"<li><code>{r.feature}</code> (OR {r.odds_ratio_per_sd:.2f}): one SD more of {pretty.get(r.feature, r.feature)} is associated with "
    f"{'about ' + format(r.odds_ratio_per_sd, '.2f') + ' times the odds' if r.odds_ratio_per_sd > 1 else format(100 * (1 - r.odds_ratio_per_sd), '.0f') + '% lower odds'}"
    f" of an abnormal move next week.</li>" for r in top5.itertuples())
gx = N["gini_example"]
prec_n = [v["precision"] for m_ in ["category", "logistic", "tree", "knn"] for v in N["budget"][m_].values()]

# Every qualitative claim in the text is checked against the numbers here, so a rerun cannot silently contradict it.
assert all(g(f"cost.r{r}.flag_nothing.cost") <= min(g(f"cost.r{r}.{m_}.cost") for m_ in ["category", "logistic", "tree", "knn", "flag_everything"])
           for r in [1, 5, 10]), "text: flag nothing is cheapest for r <= 10"
assert g("cost.r35.category.cost") < min(g(f"cost.r35.{m_}.cost") for m_ in ["logistic", "tree", "knn", "flag_nothing", "flag_everything"])
assert g("cost.r35.logistic.cost") < g("cost.r35.flag_everything.cost") < g("cost.r35.flag_nothing.cost")
assert all(g(f"cost.r10.{m_}.cost") > g("cost.r10.flag_nothing.cost") for m_ in ["logistic", "tree", "knn"])
assert abs(g("cost.r20.logistic.cost") - g("cost.r20.flag_nothing.cost")) < 10
assert g("cost_equity.r35.cheapest") == "logistic" and g("cost_equity.r20.cheapest") in ("category", "flag_nothing")
assert max(["category", "logistic", "tree", "knn"], key=lambda m_: g(f"cv.{m_}.roc_auc_mean")) == "knn"
assert g("tune.knn.best_params").startswith("n_neighbors=201") and g("tune.logistic.chosen_params") == "C=0.01"
assert g("test.category.roc_auc") > g("test.logistic.roc_auc") and g("test.diff_logistic_minus_category_ci_lo") < 0 < g("test.diff_logistic_minus_category_ci_hi")
assert g("test.diff_tree_minus_category_ci_hi") < 0 and g("cv.logistic.roc_auc_mean") > g("cv.category.roc_auc_mean")
assert g("test.logistic.reject_h0") and g("test.tree.reject_h0") and not g("test.knn.reject_h0")
assert g("test_week.n_above_05.logistic") == 4 and g("test_week.n_above_05.category") == 3
assert g("test.category.flagged_at_05") == 0 and g("test.knn.flagged_at_05") < 10
assert g("leak.shuffled_n_ge_real") == 0 and g("cv.dev_weeks") == 10
assert g("errors.fn_share_flat_or_unobserved") == 1.0
assert N["errors"]["by"]["province"]["ICT"]["fn_rate"] == 1.0 and N["errors"]["by"]["province"]["Sindh"]["fn_rate"] == 0.0
assert g("test_ref.tree.recall") == 1.0 and g("ablation.cat_only_test_auc") >= g("test.logistic.roc_auc")
assert g("imp.cat_branded_packaged.rank_logistic") == 1 and g("imp.price_changed.rank_logistic") == 2
eb = N["errors"]["by"]

# ---------------------------------------------------------------- document
H = []
H.append(f"""<!doctype html><html><head><meta charset="utf-8"><title>Milestone 03 Technical Report</title><style>{css}
.hl {{ background: #fff6dc; border-left: 3px solid #E69F00; padding: 4px 8px; }}
</style></head><body>
<h1>Who Pays More, and Who Can Least Afford It? Food Price Premiums, Deprivation and Abnormal Price Movements Across Selected Pakistani Urban Markets</h1>
<div class="meta"><b>Milestone 03 extended technical report (the submitted manuscript is the condensed M03_report.pdf)</b><br/>
<b>Group 30</b> · Yousuf Uyghur (mu07486), Computer Science · Sameer Hassan (sh09036), Computer Science<br/>
CS/SDP 312/314 L1 Data Science for Social Good, Habib University, Fall 2026 · Dr. Muhammad Usman Arif<br/>
Notebook, code and data: github.com/Samzzh0803/Pakistan-Food-Price-Premiums-and-Deprivation (branch <code>milestone-03</code>;
notebook <code>notebooks/M03_baseline_models.ipynb</code>)</div>

<div class="abstract"><b>Summary.</b> Pakistan's weekly Sensitive Price Indicator (SPI) is reported as one national figure, but PBS publishes
prices for 51 items in 17 cities. Milestone 02 built a {N2['panel']['rows']:,}-row city-item-week dataset and found that city food price premiums
are negatively rank-correlated with district deprivation. Milestone 03 asks whether the same data can give <b>early warning</b> (SDG 2.c):
using only what is known in week <i>t</i>, can we flag the city-item prices that will move abnormally by week <i>t</i>+1? Abnormal means a
change above the item's own 90th percentile of non-zero changes, fitted on training weeks only. On {n_('data.rows')} labelled rows
({n_('data.positives')} abnormal, {pct('data.base_rate')}) we compare two naive benchmarks with logistic regression, a decision tree and KNN.
We use a time-ordered split and forward-chaining cross-validation that never uses a label from the test period. On the four held-out weeks,
logistic regression reaches ROC-AUC {ci('logistic')} (row-level bootstrap), and it is above 0.5 in each test week. We reject H3's null for these
weeks, while noting that four weeks cannot establish a general result. A one-line rule that scores each item by its category's past abnormal rate
does as well ({ci('category')}), and it is also the cheapest alerting policy in our cost analysis. Precision stays near
{pct('test.logistic.precision', 0)}. The models are at most a <b>triage aid</b>, not an automatic trigger. In these test weeks, alerts pay off only
if a missed shock is judged to cost about 20 or more false alarms.</div>
""")
H.append(body)

H.append(f"""
<h2>5. Baseline models and justification</h2>
<h3>5.1 Prediction task, label and features</h3>
<p><b>Unit and target.</b> One row is one city × food item × week <i>t</i>. The target <code>y</code> = 1 if the item's price in that city
moves abnormally between week <i>t</i> and the next consecutive release <i>t</i>+1. "Abnormal" means |Δlog price| above the item's own
90th percentile of <b>non-zero</b> weekly changes, because the plain 90th percentile is 0 for {N2['h3']['items_p90_zero']} of the 32 food items.
Items with fewer than five non-zero changes in the training rows get no threshold and are dropped
({g('split.items_unlabelled')} items in the final split). The positive class is the abnormal move, the event a price monitor wants to catch.
<b>Thresholds are fitted on training rows only</b>, separately in every fold and in the final split. If they were fitted on all weeks, the
test weeks would help define what counts as abnormal.</p>
<p><b>Why only {g('data.labelled_weeks')} of {g('data.weeks_in_panel')} weeks carry a label.</b> A label needs the next week's price, and
only {g('data.labelled_weeks')} weeks are followed by a release 6-8 days later. This is a gap in PBS's published weekly archive, not in our
method. Described with whole-period thresholds, the labelled data has {n_('data.rows')} rows ({g('data.rows_per_week'):.0f} per week), of
which {n_('data.positives')} are abnormal ({pct('data.base_rate')}). The models never see these whole-period thresholds.</p>
<p><b>Features.</b> Every feature is known at week <i>t</i> (Table 10). A change that cannot be computed because the previous week was not published
({pct('data.share_no_dlog', 0)} of rows for this week's change, {pct('data.share_no_lag1', 0)} for last week's) is set to 0 <i>and</i> flagged
with a <code>has_</code> indicator. The model can then tell "no change" from "not observed" (the Milestone 02 rule: flag, don't impute).
We deliberately do not add one dummy per item: with about 200 positives this would overfit, and it would swamp KNN's distances.</p>
""")
H.append(html_table(feats, "Table 10. Model features and why each might matter. Reference levels: storable staple (category) and Punjab (province).", cls="small"))
H.append(f"""
<h3>5.2 Benchmarks and models</h3>
<p>Table 11 lists the five approaches. The two benchmarks set the floor. The <b>majority class</b> shows why accuracy is misleading here. The
<b>category base-rate rule</b> encodes the domain view that perishables move more, which Milestone 02 supported (H2). It is a tough baseline: it
needs no fitting beyond four rates, and any model that cannot beat it is not worth deploying. The three course models represent three different
ideas: a linear probability model, axis-aligned rules and similarity. Logistic regression and the tree use <code>class_weight="balanced"</code>, which weights each
positive row by n/(2 × positives) and each negative by n/(2 × negatives), so the rare class counts as much as the common one. KNN has no class weights;
its min-max scaler, like the logistic model's z-score scaler, sits inside an sklearn <code>Pipeline</code>, so it is refitted on the training rows of each fold.</p>
""")
H.append(html_table(models_tab, "Table 11. Naive benchmarks and baseline models, with the reason each was chosen, its tuning grid and the setting selected in cross-validation.", cls="small"))

H.append(f"""
<h2>6. Experimental setup</h2>
<h3>6.1 Time-ordered split, purge and stratification</h3>
<p>We train on the first {g('split.train_weeks')} labelled weeks ({g('split.train_first')} to {g('split.train_last')}) and test on the last
{g('split.test_weeks')} ({g('split.test_list')}): {pct('split.week_share_train', 0)} / {100 - 100 * g('split.week_share_train'):.0f}% by weeks.
<b>Purge rule:</b> a row at week <i>t</i> is labelled from week <i>t</i>+1, so a training row whose <i>t</i>+1 is an evaluation week
is dropped. Here the last training week ({g('split.purged_week')}) is followed by the first test week, so its {g('split.purged_rows')}
candidate rows are purged, and the final models learn from {g('split.train_weeks_after_purge')} weeks. After purging and labelling, the training set has
{n_('split.train_rows')} rows ({g('split.train_pos')} abnormal, {pct('split.train_rate', 2)}) and the test set has {n_('split.test_rows')} rows
({g('split.test_pos')} abnormal, {pct('split.test_rate', 2)}): {pct('split.row_share_train', 0)} / {100 - 100 * g('split.row_share_train'):.0f}% by rows.
The two base rates are nearly equal, so the test weeks are not unusually calm or volatile. The fitted sample
({g('split.train_rows') + g('split.test_rows'):,} rows) is smaller than the {n_('data.rows')}-row description in §5.1 for two reasons: the purged week is removed,
and thresholds fitted on training rows alone leave {g('split.items_unlabelled')} items without a label, against
{g('data.items_food') - g('data.items_labelled')} under whole-period thresholds.</p>
<p><b>Why not a random stratified split?</b> The brief's example is a random stratified split. Our rows are a panel: the same 17 × {g('data.items_labelled')}
city-item series recur every week, and neighbouring weeks share shocks. A random split would train on later weeks to predict earlier ones, which PBS could never do
in practice. It also lets the model learn week-specific shocks that appear on both sides of the split. We keep time order. The purpose of stratification,
keeping the rare class represented in every part, is met differently: every validation fold and the test set are checked to contain positives
({g('cv.val_pos_min')}-{g('cv.val_pos_max')} per validation week; {g('split.test_pos')} in test), and logistic regression and the tree use balanced class weights. Class weights change what the model optimises; they are not a substitute for
stratification, which is why the positive counts are checked. §6.4 compares random and time-ordered folds.</p>

<h3>6.2 Forward-chaining cross-validation</h3>
<p><b>Outer boundary.</b> Cross-validation may only use labels that end before the test period. The last training week ({g('split.purged_week')}) is
labelled from the first test week, so it is excluded from CV as well as from final training. This leaves {g('cv.dev_weeks')} development weeks
(to {g('cv.dev_last')}). An earlier version of our code used that week as a CV validation week, which let test-period prices shape its labels;
an independent review caught this, and every result here comes from the corrected design. The code now asserts that every CV label, in training
and validation, ends before the first test week.
<b>Folds.</b> Within the development weeks we use an expanding window (Fig. 6). Fold 1 trains on labelled weeks 1-5 and validates on week 6. Fold 2 trains on weeks 1-6
and validates on week 7, and so on, giving {g('cv.folds')} folds. Each fold rebuilds its own label thresholds, scaler and model from its training weeks and
applies the purge rule ({g('cv.purged_folds')} folds lose their last training week to it). Each validation week holds only
{g('cv.val_pos_min')}-{g('cv.val_pos_max')} abnormal moves, so fold-to-fold variance is large. We report the mean ± SD across folds and every
fold's AUC (Appendix Table A1).</p>
{fig('fig15_split_design', 'Fig. 6. Forward-chaining folds and the final split, by labelled week t. Blue: training; orange: validation or test; grey ×: training week purged because its t+1 is an evaluation week.', '70%')}

<h3>6.3 Hyperparameter tuning</h3>
<p>Each grid in Table 11 runs inside the five folds; no test-period information enters this step. Settings are compared by mean validation <b>ROC-AUC</b>, which does not depend on a threshold
and is not fooled by the majority class. When several settings are within one fold-SD of the best, we take the simplest: smallest C, smallest
depth with the largest leaves, or largest K. With so few positives per fold, the difference between near-equal settings is mostly noise (Fig. 7).
<b>Logistic regression:</b> C hardly matters. C = 0.01, the strongest regularisation, has the best mean ({cv('logistic')}).
<b>Decision tree:</b> depth 4 had the best mean ({f('tune.tree.best_mean_auc')} ± {f('tune.tree.best_sd_auc')}), but depth 2 is within one SD,
so we keep the simpler tree ({cv('tree')}); a depth-2 tree is also four readable rules.
<b>KNN:</b> AUC rises steadily with K and peaks at the grid's edge ({g('tune.knn.chosen_params')}; {cv('knn')}). Using the four continuous
features alone is clearly worse (best {f('tune.knn.best_auc_continuous')}), so the category dummies carry most of what KNN finds.</p>
{fig('fig16_tuning', 'Fig. 7. Tuning results: mean validation ROC-AUC over five forward-chaining folds for each setting (left panel shows ± 1 SD). The dotted line is chance (0.5).')}

<h3>6.4 Leakage checks</h3>
<p><b>Random folds are over-optimistic.</b> For the chosen logistic model, ordinary shuffled <code>StratifiedKFold</code> (5 folds) on the training rows gives
AUC {f('leak.random_kfold_auc_mean')} ± {f('leak.random_kfold_auc_sd')}, against {f('leak.forward_auc_mean')} for forward chaining: a gap
of {f('leak.gap')}. This is a comparison, not a controlled measurement of leakage: the two schemes also differ in training size, and the random folds
reuse labels fitted on all training rows. But it points the way we expected: ignoring time order makes the model look better than it is.
<b>Shuffled-label sanity check.</b> Refitting the final logistic model on permuted training labels and scoring the real test labels gives AUC
{f('leak.shuffled_label_test_auc')} for the seed-42 shuffle. Over {g('leak.shuffled_reps')} shuffles the mean is {f('leak.shuffled_mean')}
(5th-95th percentile {f('leak.shuffled_p05')}-{f('leak.shuffled_p95')}), and {g('leak.shuffled_n_ge_real')} of {g('leak.shuffled_reps')} reached the real test AUC.
This is consistent with the pipeline behaving correctly, but it is a sanity check, not proof that nothing leaks (it could not have detected the
CV boundary problem above). It is also not a formal test of H3, because rows in the same week are not exchangeable and tuning is not repeated
under the null.
<b>Asserts in the code:</b> no week after the cutoff is read; no evaluation week is in any training set; every CV label ends before the test period;
thresholds come from training rows; the purge is applied; the mean shuffled AUC is within 0.03 of 0.5.</p>

<h2>7. Results</h2>
<p>Table 12 reports the test set: one final evaluation of each model, with settings fixed in cross-validation. The diagnostics in §8 reuse
these same test predictions; nothing on the test set was used to choose a model, setting or threshold. Table 13 reports the same metrics as mean ± SD
across the five folds. Threshold-based metrics use 0.5. <b>PR-AUC</b> (computed as average precision, sklearn's step-wise estimate of the area under the PR curve) matters because, with about 3% positives, ROC-AUC can look respectable
while precision stays tiny. A random model's PR-AUC equals the base rate ({f('test.majority.pr_auc')} on test), so that is the bar to compare against.</p>
""")
H.append(html_table(test_tab, "Table 12. Test-set metrics (4 weeks, " + n_('split.test_rows') + " rows, " + str(g('split.test_pos')) +
                    " abnormal). Threshold 0.5 for accuracy to F1. ROC-AUC with a percentile bootstrap 95% CI (2,000 resamples of test rows, seed 42)."))
H.append(html_table(cv_tab, "Table 13. Forward-chaining cross-validation metrics: mean ± SD over 5 validation weeks (threshold 0.5).", cls="small"))
H.append(f"""
{fig('fig17_roc_pr', 'Fig. 8. ROC curves (left; AUC in brackets) and precision-recall curves (right; PR-AUC in brackets; y-axis cut at 0.2) on the test set. The dotted lines are chance: the diagonal, and the base rate.', '84%')}
<div class="hl"><p><b>H3 decision.</b> For logistic regression and the decision tree, the lower bound of the row-level bootstrap 95% CI for test
ROC-AUC is above 0.5 (logistic {ci('logistic')}; tree {ci('tree')}); for KNN it is not ({ci('knn')}). The bootstrap treats rows as independent,
but rows in the same week share shocks, so these intervals are too narrow. As a check that respects weeks, we score each test week separately (Appendix
Table A3): logistic regression is above 0.5 in all {g('test_week.n_above_05.logistic')} weeks (range {f('test_week.min.logistic')}-{f('test_week.max.logistic')}),
the category rule in {g('test_week.n_above_05.category')}. We therefore <b>reject H0 for logistic regression in these test weeks</b>: information
available at week <i>t</i> ranked next-week abnormal moves better than chance. With only four weeks this is evidence of a modest signal, not an
established general result; the Week 14 holdout is the next test. Test PR-AUC is {f('test.logistic.pr_auc')}, under twice the base rate.</p></div>
<p><b>Does any ML model beat the simple rule?</b> In cross-validation, logistic regression is ahead of the category rule on average
({cv('logistic')} vs {cv('category')}), but only in {g('cv.folds_logistic_beats_category')} of {g('cv.folds')} folds. On the test weeks, the category rule is
slightly ahead ({f('test.category.roc_auc')} vs {f('test.logistic.roc_auc')}). The paired bootstrap difference (logistic minus rule) is
{f('test.diff_logistic_minus_category')}, 95% CI [{f('test.diff_logistic_minus_category_ci_lo')}, {f('test.diff_logistic_minus_category_ci_hi')}],
so we cannot tell them apart. The depth-2 tree scores below the rule on test (row-bootstrap difference CI
[{f('test.diff_tree_minus_category_ci_lo')}, {f('test.diff_tree_minus_category_ci_hi')}], with the same independence caveat). KNN had the highest CV mean
but also the largest SD, and it fell to {f('test.knn.roc_auc')} on test. <b>The honest reading is that most of the usable signal is "what kind of food is it".</b> The ablation agrees.
Logistic regression with only the three category dummies scores {f('ablation.cat_only_cv_auc')} in CV and {f('ablation.cat_only_test_auc')} on test.
The full feature set adds about {g('leak.forward_auc_mean') - g('ablation.cat_only_cv_auc'):.2f} AUC in CV and nothing on test.</p>
<p><b>Accuracy misleads.</b> The majority class is {pct('test.majority.accuracy')} accurate and catches nothing. The category rule never reaches
a score of 0.5 and KNN does so for only {g('test.knn.flagged_at_05')} rows, so at that threshold they flag (almost) nothing, and their accuracy-to-F1 rows
look like the majority class.
Balanced class weights push logistic regression the other way: at 0.5 it flags {g('test.logistic.flagged_at_05')} rows, with recall {pct('test.logistic.recall', 0)}
but precision {pct('test.logistic.precision', 1)}. 0.5 is an arbitrary cut-off. §8 chooses thresholds from costs.</p>
<p><b>Why KNN struggles</b> (Unit 11 limitations). With about 3% positives, most neighbourhoods of 5-25 rows contain no abnormal move at all, so the vote is 0
almost everywhere. Binary dummies count as much in the distance as real price changes, and irrelevant features add noise. Many rows are identical
(no change, not observed), which produces ties. Only very large neighbourhoods give a usable score, and then KNN is a smoothed local base rate,
which explains its peak at the edge of the K grid.</p>
<p><b>Label sensitivity.</b> Under the Milestone 02 "contract" rule (plain 90th percentile; any change counts for 13 items) the test base rate is
{pct('contract.test_rate')}. Logistic regression reaches {f('contract.logistic_test_auc')} and the category rule {f('contract.category_test_auc')}, so the
task is harder. The refined label is the one we defend.</p>

<h2>8. Asymmetric error costs and confusion-matrix diagnostics</h2>
<h3>8.1 Who bears each error</h3>
<p>The alert is meant for a PBS analyst, or a district Price Control Committee or price magistrate deciding where to send inspectors or which
items to watch next week.
<ul>
<li>A <b>false negative</b> (missed abnormal move) is, at its most harmful, a jump in tomatoes, onions, flour or ghee that nobody was warned about
(abnormal moves also include sharp falls, which matter to producers and traders but less to consumers). No early stock release,
inspection or public notice happens, and households absorb the shock. Households in more deprived districts spend a larger share of income on food,
so the harm falls hardest on those least able to absorb it (H5).</li>
<li>A <b>false positive</b> (false alarm) is an inspection visit or watch-list entry that was not needed. It costs staff time and may cause friction
with traders. Repeated false alarms cause <i>alert fatigue</i>: if most alerts are wrong, people stop trusting the system.</li>
</ul></p>
<p><b>Cost matrix</b> (Provost and Fawcett, Ch. 7). cost(TP) = cost(TN) = 0, cost(FP) = 1 unit and cost(FN) = <i>r</i> units. We put the cost in one cell only,
to avoid the double counting Provost and Fawcett warn about. We do not also credit a "benefit" for true positives.
Flagging a row with probability <i>p</i> of being abnormal is worth it when <i>p</i>·<i>r</i> &gt; 1 − <i>p</i>, i.e. <i>p</i> &gt; 1/(1 + <i>r</i>).
For an average row, <i>p</i> is the training base rate ({pct('split.train_rate', 2)}), so "flag everything" beats "flag nothing" exactly when
<i>r</i> &gt; (1 − <i>p</i>)/<i>p</i> = {f('cost.flag_all_breakeven_r', 1)}, close to the inverse base rate ({f('cost.inverse_base_rate', 1)}). <b><i>r</i> is a value judgment made by
people, not something the data decides</b>, so we vary it over {{1, 5, 10, 20, 35}}. For each <i>r</i> and model, the threshold that minimises total cost is
chosen on the pooled out-of-fold CV predictions (all from development weeks) and then applied to the test predictions.</p>
""")
H.append(html_table(cost_show, "Table 14. Test cost per 1,000 city-item-weeks (in false-alarm units) and, in brackets, the share of abnormal moves caught, for each cost ratio r. Thresholds were chosen on CV predictions only."))
H.append(f"""
{fig('fig19_costs', 'Fig. 9. Test cost per 1,000 city-item-weeks against the cost ratio r. For r ≤ 10 the cheapest policy is to issue no alerts; at r = 35 the category rule is cheapest.', '64%')}
<p><b>Reading the sensitivity</b> (test weeks only). For <i>r</i> ≤ 10 the cheapest policy is to flag nothing. At <i>r</i> = 10 the three ML models'
CV thresholds raise some alerts, and on test each costs more than staying silent. At <i>r</i> = 20, logistic regression
({f('cost.r20.logistic.cost', 1)}) and flagging nothing ({f('cost.r20.flag_nothing.cost', 1)}) are essentially tied. At <i>r</i> = 35 the
<b>category rule</b> is cheapest ({f('cost.r35.category.cost', 0)}), ahead of logistic regression ({f('cost.r35.logistic.cost', 0)}), flagging
everything ({f('cost.r35.flag_everything.cost', 0)}) and flagging nothing ({f('cost.r35.flag_nothing.cost', 0)}). So, in these test weeks, <b>no ML model beats
the simple rule on cost, and alerting only pays off at all when a missed shock is judged to be worth roughly 20 or more unnecessary inspections.</b>
These are the points we chose to try, not a measured break-even, and the recommended threshold depends on <i>r</i>. We use <i>r</i> = {g('cost.r_ref')} below
as an illustrative reference (a value judgment, not chosen from the data). At that threshold logistic regression flags
{pct('test_ref.logistic.flagged_share', 0)} of test rows, catches {pct('test_ref.logistic.recall', 0)} of abnormal moves, and has precision
{pct('test_ref.logistic.precision', 1)} (Appendix Table A2 has all models).</p>
<p><b>Alert budget.</b> A decision-maker may think in terms of capacity rather than <i>r</i>: "we can check <i>N</i> city-items a week". Table 15 ranks each test
week's {g('split.test_rows_per_week'):.0f} rows by score and takes the top <i>N</i>. With 50 checks a week, logistic regression puts
{f('budget.logistic.n50.caught', 0)} of the {g('budget.test_positives')} abnormal moves in front of an inspector ({pct('budget.logistic.n50.recall', 0)});
the category rule, with random tie-breaking, puts {f('budget.category.n50.caught', 1)} in front of them.
Precision@<i>N</i> stays between {100 * min(prec_n):.0f}% and {100 * max(prec_n):.0f}%, against {pct('budget.random_precision', 1)} for random checks.</p>
""")
H.append(html_table(bud_show, "Table 15. Alert budget, pooled over the four test weeks: share of abnormal moves caught (recall) and share of checks that find one (precision) when only the top N rows per week are checked. Ties (category rule) are counted at their expected value. Per-week results are in Appendix Table A4.", cls="small"))
H.append(f"""
<p><b>Equity weighting</b> (preview of the Milestone 04 fairness audit). If a missed shock counts 1, 1.5 or 2 times as much in low-, mid- and high-deprivation
cities, the cost-minimising choices shift towards flagging. At <i>r</i> = 10 the logistic threshold falls to {f('cost_equity.r10.logistic_threshold', 2)}
and recall rises to {pct('cost_equity.r10.logistic_recall', 0)}, although on test flagging nothing is still cheapest. At <i>r</i> = 35 logistic regression
becomes the cheapest policy. Weighting by need changes who gets protected, and that is a decision for policy-makers, not for us.</p>

<h3>8.2 Confusion matrices</h3>
<p>Fig. 10 uses the Unit 10 layout: actual class in rows, predicted in columns, positive (abnormal) class first, with counts and row percentages. So the
top-left cell over its row is recall, and the bottom-right over its row is specificity. The majority class and the category rule (at <i>r</i> = 20) have an empty first column. Logistic regression
at 0.5 catches {g('test.logistic.tp_at_05')} of {g('split.test_pos')} abnormal moves at the price of
{n_('test.logistic.fp_at_05')} false alarms. The <i>r</i> = 20 threshold trades a little more recall for more false alarms. The
depth-2 tree at its <i>r</i> = 20 threshold flags every non-branded item, which is the category rule in disguise.</p>
{fig('fig18_confusion', 'Fig. 10. Test-set confusion matrices: counts and row percentages. Rows are the actual class, columns the predicted class, abnormal first. "cost threshold" is the threshold minimising cost at r = 20 on CV predictions; its value is in brackets.', '58%')}

<h3>8.3 Where the errors fall, and edge cases</h3>
<p>Table 16 breaks down the logistic model's errors at the <i>r</i> = 20 threshold. This is descriptive only; the formal test is H4 in Milestone 04, and many cells
hold under 20 positives. Two patterns stand out.
<ul>
<li><b>Flags concentrate in perishables.</b> The FP rate is {pct('errors.by.item_category.perishable.fp_rate', 0)} for perishables against
{pct('errors.by.item_category.branded_packaged.fp_rate', 0)} for branded goods.</li>
<li><b>Flags concentrate in high-deprivation cities.</b> The FP rate is {pct('errors.by.deprivation_tercile.high.fp_rate', 0)} there against
{pct('errors.by.deprivation_tercile.low.fp_rate', 0)} in low-deprivation cities. The miss rate runs the other way:
{pct('errors.by.deprivation_tercile.high.fn_rate', 0)} in high- and {pct('errors.by.deprivation_tercile.low.fn_rate', 0)} in low-deprivation cities.</li>
</ul>
By province, Islamabad's {eb['province']['ICT']['positives']} abnormal moves were all missed, while Sindh's
{eb['province']['Sindh']['positives']} were all caught. Both groups are tiny.</p>
""")
H.append(html_table(err_show, "Table 16. Logistic regression errors on the test set at the r = 20 threshold, by group. FN rate = share of abnormal moves missed; FP rate = share of normal rows flagged. Terciles are over the 17 cities.", cls="small"))
H.append(f"""
<p><b>Edge cases</b> (Table 17; the full ten of each are in the notebook).
<ul>
<li><b>Every missed abnormal move had a flat or unobserved price at week <i>t</i></b>
({pct('errors.fn_share_flat_or_unobserved', 0)} of the {g('errors.n_fn')} false negatives have <code>price_changed</code> = 0). Moves that come with no
visible recent history are invisible to any model built on recent changes.</li>
<li><b>The most confident false alarms</b> are perishables in Sukkur, Bannu and Larkana. Most are records where PBS's minimum and maximum quotes are
equal (one effective quote), which changed this week and then did not move again. A single quote that jumps may reflect one shop rather than the
market. That links model errors to the measurement-quality concern in H4.</li>
<li><b>Timing.</b> The two earlier test weeks have about twice the base rate of the August weeks
({pct('errors.by.test_week.2026-04-02.base_rate', 1)} and {pct('errors.by.test_week.2026-06-04.base_rate', 1)} against
{pct('errors.by.test_week.2026-08-13.base_rate', 1)} and {pct('errors.by.test_week.2026-08-20.base_rate', 1)}). The 2 April week comes about 12 days after
Eid-ul-Fitr, observed in Pakistan on 21 March 2026, and the 4 June week about a week after Eid-ul-Adha, observed on 27 May 2026 (APP, 2026; Radio
Pakistan, 2026). Post-festival price adjustment is <i>one possible explanation that we have not tested</i>; the dates alone are not evidence of a
holiday effect, and some of these abnormal moves are falls rather than rises.</li>
</ul></p>
""")
H.append(html_table(edge, "Table 17. The five most confident false negatives (lowest score among missed abnormal moves) and false positives (highest score among normal rows), logistic regression, r = 20 threshold. Changes are in log points.", cls="small"))

H.append(f"""
<h2>9. Drivers and initial policy insights</h2>
<p><b>Logistic regression.</b> Inputs are z-scored inside the pipeline, so each coefficient is the change in log-odds per one standard deviation, and
e<sup>coef</sup> is the odds ratio (OR) per SD (Table 18, Fig. 11). For a 0/1 feature, one SD is a fixed fraction of the switch from 0 to 1, so these
ORs rank features rather than describe a full category change. They are associations, not causes. The top five:
<ul>{or_sentences}</ul>
<code>abs_dlog</code> gets a small coefficient because it is correlated with <code>price_changed</code> (r = {f('imp.corr_abs_dlog_price_changed', 2)}
in the training rows), and the two share the credit (Unit 09). The <code>has_</code> flags are negative mostly because they mark <i>which weeks</i> had a
published previous week, so they partly stand in for time period, not for a market effect.</p>
<p><b>Decision tree.</b> The depth-2 tree splits first on branded/packaged, then on this week's change for other items, and on relative price for branded
ones; the diagram is in the notebook (Part D.2) and <code>reports/m03/figures/fig21_tree.png</code>. Its impurity importance is {f('imp.cat_branded_packaged.tree', 2)} for <code>cat_branded_packaged</code>,
{f('imp.abs_dlog.tree', 2)} for <code>abs_dlog</code> and {f('imp.rel_price.tree', 2)} for <code>rel_price</code>. Its leaves read as rules (Table 19).
The top rule is: <b>IF the item is not branded/packaged AND its price moved by more than about 1% this week, THEN the training abnormal rate is
{pct('rules.top1.rate', 1)}</b>, against {pct('split.train_rate', 1)} overall. Branded items whose price is at or above the national reference
had no abnormal move in training.
<b>Worked Gini (root).</b> Unweighted, the root holds {gx['pos']} positives in {gx['n']:,} rows (Gini {gx['gini_parent']:.4f}); the split sends
{gx['pos_left']}/{gx['n_left']:,} left and {gx['pos_right']}/{gx['n_right']:,} right, gain {gx['gini_gain']:.4f}. With balanced weights (positive
{gx['w_pos']:.1f}, negative {gx['w_neg']:.3f}), the root is exactly 50/50 (Gini {gx['wgini_parent']:.2f}), the children have Gini {gx['wgini_left']:.3f}
and {gx['wgini_right']:.3f}, and the weighted gain is {gx['wgini_gain']:.3f}, the largest available.</p>
<p><b>Cross-model check.</b> Both models put the branded/packaged flag first, and both give recent price movement a top role (<code>price_changed</code> in
the logistic model, <code>abs_dlog</code> in the tree: two views of the same signal). These are the trustworthy drivers. Perishability, smaller markets and
higher district deprivation are positive in the logistic model only. A depth-2 tree can use just three features, so its silence on them is not evidence
against them.</p>
""")
H.append('<div class="two">' + html_table(imp_show.head(10), "Table 18. Top 10 features by |logistic coefficient| and their tree importance (blank = unused by the tree).", cls="small")
         + html_table(rules_show, "Table 19. Decision-tree leaves as IF-THEN rules, with their actual abnormal rate on the training rows.", cls="small") + "</div>")
H.append(f"""
{fig('fig20_importance', 'Fig. 11. Feature importance. Left: logistic regression coefficients per 1 SD (red raises the odds of an abnormal move next week, blue lowers them). Right: decision-tree impurity importance.', '76%')}
<p><b>Initial policy takeaways (SDG 2.c).</b> These are hedged: they rest on four test weeks and modest signal.
<ol>
<li><b>Where to look.</b> Concentrate weekly monitoring on perishables, especially those whose price already moved this week. Branded and packaged
goods almost never move abnormally.</li>
<li><b>At most a triage aid, not a trigger.</b> With test AUC around 0.7 and precision around 5%, most alerts will be false. A model score can rank where an
inspector looks first; it should not trigger stock releases or penalties by itself.</li>
<li><b>A simple rule already captures most of the signal.</b> "Watch perishables" matched the logistic model on the test weeks. Any data system PBS or a
province builds should be benchmarked against that rule before it is trusted.</li>
<li><b>Publish every week.</b> {pct('data.share_no_dlog', 0)} of labelled rows have no previous consecutive week, because the published weekly archive has
gaps. The moves we miss are exactly those with no visible recent history. Consistent publication of the weekly annexure is a cheap way to improve early warning.</li>
<li><b>The threshold is a policy choice.</b> Whether alerts are worth issuing depends on how many false alarms one missed shock is worth. Below about 20,
silence is cheaper. That judgment, and whether to weight misses in deprived districts more, belongs to decision-makers.</li>
</ol></p>

<h2>10. Limitations and Milestone 04 roadmap</h2>
<p><b>Limitations.</b>
<ul>
<li><b>Sample size.</b> Only {n_('data.positives')} positives over {g('data.labelled_weeks')} labelled weeks. The test set has {g('split.test_pos')}
positives in four weeks, so every test number has a wide interval.</li>
<li><b>Intervals and the archive.</b> The bootstrap treats rows as independent, but rows in the same week share shocks, so the true uncertainty is
larger; the per-week check uses only four weeks. Several diagnostics (costs, alert budget, error breakdown) reuse the same test predictions. PBS archive gaps cost {pct('data.share_no_dlog', 0)} of rows their recent history and limit how many weeks can be labelled.</li>
<li><b>Signal.</b> Class imbalance and modest signal. KNN is unsuitable for this data. Two tuned settings sit at the edge of their grids
(C = 0.01, K = 201), which suggests the data favour very smooth models.</li>
<li><b>Label.</b> The 90th-percentile threshold is a convention. A stricter or looser cut-off would change the base rate and the results (§7).</li>
<li><b>Context variables.</b> City-level variables (population, deprivation from 2019-20) are constant within a city. Islamabad and Rawalpindi
function as one market but are scored separately.</li>
<li><b>Scope.</b> 17 urban SPI cities only, so possible place-based bias in who gets missed (§8.3). The Milestone 02 limitations on coverage,
deprivation data and measurement still apply.</li>
</ul></p>
<p><b>Milestone 04 plan.</b>
<ul>
<li><b>Advanced models:</b> random forest and gradient boosting, compared against today's two benchmarks and three baselines under the same
forward-chaining CV, purge and cost analysis.</li>
<li><b>Fairness audit (H4):</b> recall, FN rate and calibration compared across province, population tercile and deprivation tercile, with a
hypothesis test, and including whether the single-quote data-quality flag drives errors.</li>
<li><b>Thresholds and ethics:</b> group-aware thresholds that equalise miss rates, and their cost. An ethical evaluation of who bears the false
negatives and the false alarms.</li>
<li><b>Week 14 new data:</b> the design already committed in Milestones 01-02. First, score the frozen model on the post-cutoff holdout weeks without
refitting. Then retrain on the earlier part of the new data and test on a later, unseen part.</li>
</ul></p>

<h2>References</h2>
<p class="note">Pakistan Bureau of Statistics. Weekly Sensitive Price Indicator, Annexure workbooks, weeks ended {N2['panel']['first_week']} to
{N2['panel']['last_week']}. https://www.pbs.gov.pk/<br/>
Pakistan Bureau of Statistics. Pakistan Social and Living Standards Measurement Survey 2019-20, District Level: Tables 2.14(a), 2.15, 7.1, 9.1.<br/>
Pakistan Bureau of Statistics. 7th Population and Housing Census 2023 (Digital Census): Table 1, district results.<br/>
Pakistan Bureau of Statistics. CPI (Urban) Group-wise Cumulative Indices, base 2015-16; Monthly Review on Price Indices, August and September 2026.<br/>
Provost, F. and Fawcett, T. (2013). <i>Data Science for Business</i>. O'Reilly. Ch. 7 (expected value, cost-benefit matrix) and Ch. 8 (ROC, profit curves).<br/>
Associated Press of Pakistan (2026, 19 March). Eid-ul-Fitr on Saturday as Shawwal moon not sighted. https://www.app.com.pk/national/eid-ul-fitr-on-saturday-as-shawwal-moon-not-sighted/<br/>
Radio Pakistan (2026, 17 May). Moon sighted, Eid-ul-Adha to be celebrated on 27 May. https://radio.gov.pk/17-05-2026/moon-sighted-eid-ul-adha-to-be-celebrated-on-27-may<br/>
Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.<br/>
Wickham, H. (2014). Tidy Data. <i>Journal of Statistical Software</i>, 59(10).<br/>
Anscombe, F. J. (1973). Graphs in Statistical Analysis. <i>The American Statistician</i>, 27(1), 17-21.</p>

<h2>Appendix</h2>
""")
H.append(html_table(folds_show, "Table A1 (part 1). Forward-chaining folds: weeks, purged rows, and rows and positives after labelling. 'items w/o label' = items with fewer than five non-zero changes in that fold's training rows.", cls="small"))
H.append(html_table(cvp_show, "Table A1 (part 2). Validation ROC-AUC per fold for the chosen settings.", cls="small"))
H.append(html_table(at_ref.rename(columns={"threshold": "threshold (from CV)"}), "Table A2. Test metrics at the cost-based threshold for r = 20 (chosen on pooled CV predictions, applied once to test).", cls="small"))
wk = t("test_auc_by_week").rename(columns={"category": "category rule", "logistic": "logistic", "tree": "tree", "knn": "KNN"})
H.append(html_table(wk, "Table A3. Test ROC-AUC computed separately for each test week (the dependence check used in §7).", cls="small"))
bw = t("alert_budget_weekly")
bw = bw[bw.model.isin(["category", "logistic"])].assign(model=lambda d: d.model.map(LAB))
bw_show = bw.pivot_table(index=["model", "N_per_week"], columns="week", values="caught").reset_index()
bw_show.columns = [str(c) for c in bw_show.columns]
pos_w = t("alert_budget_weekly").drop_duplicates("week").set_index("week").positives
H.append(html_table(bw_show.rename(columns={"N_per_week": "checks per week"}),
                    "Table A4. Alert budget per test week: abnormal moves caught (expected count under random tie-breaking) by the category rule and logistic regression. "
                    "Abnormal moves per week: " + ", ".join(f"{w} {int(v)}" for w, v in pos_w.items()) + ".", cls="small", digits=1))
H.append("</body></html>")

doc = "\n".join(H)
out_html = REP / "M03_technical_report.html"
out_html.write_text(doc, encoding="utf-8")
print("wrote", out_html)

edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
out_pdf = REP / "M03_technical_report.pdf"
if edge.exists():
    subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out_pdf}",
                    out_html.resolve().as_uri()], check=True, timeout=180)
    print("wrote", out_pdf)
else:
    print("Edge not found; open the HTML and print to PDF")
