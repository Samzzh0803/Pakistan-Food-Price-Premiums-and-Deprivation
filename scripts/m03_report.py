"""Assemble the submitted Milestone 03 manuscript (condensed, about nine pages; HTML -> PDF via headless Edge).

The extended version with every table and appendix is built by scripts/m03_technical_report.py. This script reads
its numbers from reports/m03/m03_numbers.json (M03) and reports/m02/m02_numbers.json (the Milestone 02 sections), and
its tables from CSVs the pipeline wrote. It redraws four figures with plain-language labels from those same tables,
so no analysis is recomputed here. Qualitative claims in the text are checked against the numbers before writing.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pfp.config import PROVINCE_COLORS  # noqa: E402

REP = ROOT / "reports" / "m03"
TAB = REP / "tables"
FIG = REP / "figures"
M02TAB = ROOT / "reports" / "m02" / "tables"
N = json.loads((REP / "m03_numbers.json").read_text(encoding="utf-8"))
N2 = json.loads((ROOT / "reports" / "m02" / "m02_numbers.json").read_text(encoding="utf-8"))


def g(path, src=None):
    d = N if src is None else src
    for k in path.split("."):
        d = d[k]
    return d


g2 = lambda path: g(path, N2)
a2 = lambda path: f"{g(path):.2f}"                      # AUCs and other proportions: two decimals
pc = lambda path, d=0: f"{100 * g(path):.{d}f}%"       # percentages: whole numbers unless d is given
pc2 = lambda path, d=0: f"{100 * g2(path):.{d}f}%"
ci = lambda m: f"{g(f'test.{m}.roc_auc'):.2f} [{g(f'test.{m}.auc_ci_lo'):.2f}, {g(f'test.{m}.auc_ci_hi'):.2f}]"
t = lambda name: pd.read_csv(TAB / f"{name}.csv")
LAB = {"majority": "Majority class", "category": "Category base-rate rule", "logistic": "Logistic regression",
       "tree": "Decision tree", "knn": "K-nearest neighbours"}
PLAIN = {"abs_dlog": "size of this week's change", "has_dlog": "this week's change observed", "price_changed": "price changed this week",
         "abs_dlog_lag1": "size of last week's change", "has_lag1": "last week's change observed",
         "rel_price": "price relative to national reference", "within_city_range_pct": "price spread across the city's outlets",
         "single_quote": "single effective quote", "cat_perishable": "perishable item", "cat_branded_packaged": "branded or packaged item",
         "cat_prepared_food": "prepared food", "prov_Sindh": "Sindh", "prov_KP": "Khyber Pakhtunkhwa", "prov_Balochistan": "Balochistan",
         "prov_ICT": "Islamabad", "log_pop_urban": "urban population (log)", "deprivation_composite": "district deprivation"}

# ---------------------------------------------------------------- claim checks (the text below relies on these)
ML = ["logistic", "tree", "knn"]
assert all(g(f"cost.r{r}.flag_nothing.cost") <= min(g(f"cost.r{r}.{m}.cost") for m in ["category", *ML, "flag_everything"]) for r in [1, 5, 10])
assert all(g(f"cost.r10.{m}.cost") > g("cost.r10.flag_nothing.cost") for m in ML)
assert abs(g("cost.r20.logistic.cost") - g("cost.r20.flag_nothing.cost")) < 10
assert g("cost.r35.category.cost") < min(g(f"cost.r35.{m}.cost") for m in [*ML, "flag_nothing", "flag_everything"])
assert max(["category", *ML], key=lambda m: g(f"cv.{m}.roc_auc_mean")) == "knn"
assert g("test.category.roc_auc") > g("test.logistic.roc_auc") and g("test.diff_logistic_minus_category_ci_lo") < 0 < g("test.diff_logistic_minus_category_ci_hi")
assert g("cv.logistic.roc_auc_mean") > g("cv.category.roc_auc_mean") and g("test.diff_tree_minus_category_ci_hi") < 0
assert g("test.logistic.reject_h0") and g("test.tree.reject_h0") and not g("test.knn.reject_h0")
assert g("test_week.n_above_05.logistic") == 4 and g("test.category.flagged_at_05") == 0 and g("test.knn.flagged_at_05") < 10
assert g("errors.fn_share_flat_or_unobserved") == 1.0 and g("leak.shuffled_n_ge_real") == 0
assert g("tune.logistic.chosen_params") == "C=0.01" and g("tune.knn.best_params").startswith("n_neighbors=201")
assert g("imp.cat_branded_packaged.rank_logistic") == 1 and g("imp.price_changed.rank_logistic") == 2 and g("imp.cat_perishable.rank_logistic") == 3
assert g("ablation.cat_only_test_auc") > g("test.logistic.roc_auc") and 0.045 < g("test.logistic.precision") < 0.055
assert g("test_ref.logistic.flagged_share") > 0.45 and g("errors.by.test_week.2026-04-02.base_rate") >= 1.9 * g("errors.by.test_week.2026-08-13.base_rate")
assert g2("h5.deprivation_composite.spearman_rho") < 0 and g2("h2.p_one_tailed") < 0.05 < g2("h2.mannwhitney_p_one_tailed")


# ---------------------------------------------------------------- figures redrawn with plain labels
def style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "legend.frameon": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


style()
# EDA: city premiums with intervals (left) and premium vs deprivation composite (right).
prem = pd.read_csv(M02TAB / "premium.csv")
city = pd.read_csv(ROOT / "data" / "processed" / "m02" / "city_table.csv")
prem = prem.merge(city[["city", "deprivation_composite"]], on="city").sort_values("pct_mean")
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
lo, hi = 100 * (np.exp(prem.ci_lo) - 1), 100 * (np.exp(prem.ci_hi) - 1)
for i, r in enumerate(prem.itertuples()):
    c = PROVINCE_COLORS[r.province]
    a.plot([lo.iloc[i], hi.iloc[i]], [i, i], c=c, lw=1.2)
    a.scatter(r.pct_mean, i, c=c if r.bonferroni_sig else "white", edgecolors=c, s=22, zorder=3)
a.axvline(0, c="k", lw=0.7)
a.set_yticks(range(len(prem)), prem.city, fontsize=8)
a.set_xlabel("Mean premium over national reference (%)")
a.set_title("City food price premium, 95% interval")
for prov, c in PROVINCE_COLORS.items():
    a.scatter([], [], c=c, s=18, label=prov)
a.legend(fontsize=6.5, loc="lower right")
for r in prem.itertuples():
    b.scatter(r.deprivation_composite, r.pct_mean, c=PROVINCE_COLORS[r.province], s=24)
    if r.city in {"Islamabad", "Rawalpindi", "Karachi", "Quetta", "Khuzdar", "Sukkur", "Bannu", "Larkana", "Peshawar", "Multan"}:
        b.annotate(r.city, (r.deprivation_composite, r.pct_mean), fontsize=7, textcoords="offset points",
                   xytext=(3, -9) if r.city == "Bannu" else (3, 2))
b.axhline(0, c="k", lw=0.7)
b.set_xlabel("District deprivation composite (0-1, higher = more deprived)")
b.set_ylabel("Mean premium (%)")
b.set_title(f"Premium vs deprivation (Spearman ρ = {g2('h5.deprivation_composite.spearman_rho'):.2f}, n = 17)")
fig.tight_layout()
save(fig, "m03r_eda")

# Tuning: one panel per model.
tl, tt, tk = t("tuning_logistic"), t("tuning_tree"), t("tuning_knn")
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
ax = axes[0]
ax.errorbar(range(len(tl)), tl.mean_auc, yerr=tl.sd_auc, marker="o", capsize=3, c="#0072B2")
ax.set_xticks(range(len(tl)), [f"{c:g}" for c in tl["clf__C"]])
ax.set_xlabel("C (smaller = stronger regularisation)"); ax.set_ylabel("Mean CV ROC-AUC"); ax.set_title("Logistic regression (± 1 SD)")
ax = axes[1]
for leaf, d in tt[tt.criterion == "gini"].groupby("min_samples_leaf"):
    ax.plot(d.max_depth, d.mean_auc, marker="o", ms=3.5, lw=1.2, label=f"Gini, leaf ≥ {leaf}")
best_ent = tt[tt.criterion == "entropy"].groupby("max_depth").mean_auc.max()
ax.plot(best_ent.index, best_ent.values, c="#777777", ls="--", lw=1.2, label="entropy (best leaf)")
ax.set_xlabel("Maximum depth"); ax.set_title("Decision tree"); ax.legend(fontsize=7, loc="lower left")
ax = axes[2]
for fs, lab_, c in [("all", "all features", "#CC79A7"), ("continuous", "price features only", "#999999")]:
    d = tk[tk.features == fs].groupby("clf__n_neighbors").mean_auc.max()
    ax.plot(d.index, d.values, marker="o", ms=3.5, lw=1.2, c=c, label=lab_)
ax.set_xscale("log"); ax.set_xlabel("K (neighbours, log scale)"); ax.set_title("KNN (best weighting and distance)")
ax.legend(fontsize=7, loc="lower right")
for ax in axes:
    ax.axhline(0.5, c="k", lw=0.6, ls=":")
fig.tight_layout()
save(fig, "m03r_tuning")

# Confusion matrices: four panels in the standard layout (actual rows, predicted columns, abnormal first).
cm = t("confusion_matrices").set_index("matrix")
pick = [("Majority class, (flags nothing)", "Majority class"),
        ("Category base-rate rule, cost threshold, r = 20 (0.05)", "Category rule, cost threshold (r = 20)"),
        ("Logistic regression, threshold 0.5 (0.50)", "Logistic regression, threshold 0.5"),
        (f"Logistic regression, cost threshold, r = 20 ({g('test_ref.logistic.threshold'):.2f})", "Logistic regression, cost threshold (r = 20)")]
fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.3))
for ax, (key, title) in zip(axes, pick):
    r = cm.loc[key]
    mat = np.array([[r.TP, r.FN], [r.FP, r.TN]]).astype(int)
    rp = mat / mat.sum(axis=1, keepdims=True)
    ax.imshow(rp, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}\n{100 * rp[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if rp[i, j] > 0.6 else "black")
    ax.set_xticks([0, 1], ["abnormal", "normal"], fontsize=7.5); ax.set_yticks([0, 1], ["abnormal", "normal"], fontsize=7.5)
    ax.set_xlabel("predicted", fontsize=7.5); ax.set_ylabel("actual", fontsize=7.5)
    ax.set_title(title.replace(", ", ",\n", 1), fontsize=8.5); ax.grid(False)
fig.tight_layout()
save(fig, "m03r_confusion")

# Feature importance with plain names.
imp = t("importance")
imp["name"] = imp.feature.map(PLAIN)
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
d = imp.sort_values("coef_per_sd")
a.barh(d.name, d.coef_per_sd, color=np.where(d.coef_per_sd > 0, "#D55E00", "#0072B2"))
a.axvline(0, c="k", lw=0.7); a.tick_params(axis="y", labelsize=8)
a.set_xlabel("Change in log-odds per 1 SD"); a.set_title("Logistic regression")
d = imp[imp.tree_importance > 0].sort_values("tree_importance")
b.barh(d.name, d.tree_importance, color="#009E73"); b.tick_params(axis="y", labelsize=8)
b.set_xlabel("Share of impurity reduction"); b.set_title("Decision tree")
fig.tight_layout()
save(fig, "m03r_importance")


# ---------------------------------------------------------------- tables
def html_table(df: pd.DataFrame, caption: str, cls: str = "") -> str:
    body = df.to_html(index=False, border=0, escape=True, na_rep="–")
    return f'<div class="tbl {cls}"><p class="cap">{caption}</p>{body}</div>'


def fig(name: str, caption: str, width: str = "100%") -> str:
    return f'<figure><img src="figures/{name}.png" style="width:{width}"/><figcaption>{caption}</figcaption></figure>'


sources = pd.DataFrame([
    ("PBS weekly SPI annexure", "City-item prices (minimum, average, maximum)", f"{g2('panel.weeks')} weeks, {g2('panel.first_week')} to {g2('panel.last_week')}"),
    ("PSLM district survey", "Deprivation: food insecurity, illiteracy, schooling, water", "2019-20"),
    ("Census, Table 1", "Total and urban population by district", "2023"),
    ("CPI (urban food)", "Deflator only", "Monthly"),
], columns=["Source (all PBS)", "What it provides", "Coverage"])


def plain_params(m):
    p = dict(kv.split("=") for kv in g(f"tune.{m}.chosen_params").split(", "))
    if m == "logistic":
        return f"C = {p['C']}"
    if m == "tree":
        return f"{p['criterion'].capitalize()}, depth {p['max_depth']}, leaf ≥ {p['min_samples_leaf']}"
    return f"K = {p['n_neighbors']}, {p['weights']}-weighted, {p['metric'].capitalize()}, {p['features']} features"


models = pd.DataFrame([
    ("Majority class", "Shows how misleading accuracy is when events are rare", "–", "–"),
    ("Category base-rate rule", "Encodes the view that perishables move more; a model must beat it to be worth using", "–", "–"),
    ("Logistic regression", "Gives probabilities for cost-based thresholds and odds ratios an analyst can read; stable with few positives",
     "C ∈ {0.01, 0.1, 1, 10}", plain_params("logistic")),
    ("Decision tree", "Produces if-then rules a price committee can apply without software, and captures interactions",
     "Gini or entropy; depth 2-6; minimum leaf 10, 30 or 60", plain_params("tree")),
    ("K-nearest neighbours", "Tests a different idea: similar city-item situations behave alike",
     "K from 5 to 201; uniform or distance weights; Euclidean or Manhattan; all or continuous features", plain_params("knn")),
], columns=["Model", "Why it suits this problem", "Grid searched", "Chosen in CV"])

res = []
for m in LAB:
    res.append({"Model": LAB[m], "Accuracy": a2(f"test.{m}.accuracy"), "Precision": a2(f"test.{m}.precision"),
                "Recall": a2(f"test.{m}.recall"), "Specificity": a2(f"test.{m}.specificity"), "F1": a2(f"test.{m}.f1"),
                "Test ROC-AUC [95% CI]": ci(m) if m != "majority" else "0.50", "Test PR-AUC": a2(f"test.{m}.pr_auc"),
                "CV ROC-AUC (mean ± SD)": f"{g(f'cv.{m}.roc_auc_mean'):.2f} ± {g(f'cv.{m}.roc_auc_sd'):.2f}"})
res = pd.DataFrame(res)

pol = [("category", "Category base-rate rule"), ("logistic", "Logistic regression"), ("flag_nothing", "Flag nothing"),
       ("flag_everything", "Flag everything")]
costs = pd.DataFrame([{"Policy": lab, **{f"r = {r}": f"{g(f'cost.r{r}.{k}.cost'):.0f}" for r in [5, 10, 20, 35]}} for k, lab in pol])

eb = N["errors"]["by"]
err_rows = [("Perishable", "item_category", "perishable"), ("Storable staple", "item_category", "storable_staple"),
            ("Branded or packaged", "item_category", "branded_packaged"), ("Low deprivation", "deprivation_tercile", "low"),
            ("Mid deprivation", "deprivation_tercile", "mid"), ("High deprivation", "deprivation_tercile", "high")]
errs = pd.DataFrame([{"Group": lab, "Abnormal moves": eb[k][v]["positives"],
                      "Missed (FN rate)": "–" if eb[k][v]["fn_rate"] is None else f"{100 * eb[k][v]['fn_rate']:.0f}%",
                      "False-alarm rate": f"{100 * eb[k][v]['fp_rate']:.0f}%"} for lab, k, v in err_rows])


def plain_rule(rule: str) -> str:
    out = []
    for c in rule.replace("IF ", "").split(" AND "):
        f_, op, v = c.split(" ")
        v = float(v)
        if f_ == "cat_branded_packaged":
            out.append("not branded or packaged" if op == "<=" else "branded or packaged")
        elif f_ == "abs_dlog":
            out.append(f"this week's change {'at most' if op == '<=' else 'above'} {100 * v:.0f}%")
        elif f_ == "rel_price":
            out.append("priced below the national reference" if op == "<=" else "priced at or above the national reference")
        else:
            out.append(f"{PLAIN[f_]} {op} {v:g}")
    return "; ".join(out)


rules = t("tree_rules")
rules_tab = pd.DataFrame({"Leaf (if ...)": rules.rule.map(plain_rule), "Training rows": rules.train_rows.map("{:,}".format),
                          "Abnormal rate": (100 * rules.abnormal_rate).map("{:.1f}%".format)})

# ---------------------------------------------------------------- document
css = """
@page { size: A4; margin: 20mm 20mm 20mm 20mm; }
body { font-family: 'Cambria', 'Georgia', serif; font-size: 10.3pt; line-height: 1.34; color: #111; background: #fff; }
h1 { font-size: 15pt; margin: 0 0 4px 0; line-height: 1.2; }
h2 { font-size: 12pt; margin: 12px 0 4px 0; border-bottom: 1px solid #999; padding-bottom: 2px; break-after: avoid; }
p { margin: 3px 0 6px 0; text-align: justify; }
.meta { font-size: 9.6pt; color: #333; margin-bottom: 7px; }
.abstract { border: 1px solid #bbb; padding: 6px 10px; background: #f7f7f7; font-size: 10pt; }
figure { margin: 6px 0 8px 0; text-align: center; break-inside: avoid; }
figcaption { font-size: 9pt; color: #222; text-align: left; margin-top: 2px; }
.tbl { margin: 5px 0 8px 0; break-inside: avoid; }
.cap { font-size: 9pt; font-weight: bold; margin: 0 0 2px 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; }
th { border-bottom: 1.2px solid #333; border-top: 1.2px solid #333; text-align: left; padding: 2px 4px; background: #f2f2f2; }
td { border-bottom: 0.5px solid #ddd; padding: 1.5px 4px; vertical-align: top; }
ol, ul { margin: 3px 0 6px 18px; padding: 0; } li { margin-bottom: 3px; }
.refs { font-size: 9pt; }
.two { display: flex; gap: 12px; align-items: flex-start; } .two > * { flex: 1; }
"""
ow = N["test_week"]
H = [f"""<!doctype html><html><head><meta charset="utf-8"><title>Milestone 03 Report</title><style>{css}</style></head><body>
<h1>Who Pays More, and Who Can Least Afford It? Food Price Premiums, Deprivation and Abnormal Price Movements Across Selected Pakistani Urban Markets</h1>
<div class="meta"><b>Milestone 03: Baseline models, evaluation and initial findings</b><br/>
<b>Group 30</b> · Yousuf Uyghur (mu07486), Computer Science · Sameer Hassan (sh09036), Computer Science<br/>
CS/SDP 312/314 L1 Data Science for Social Good, Habib University, Fall 2026 · Dr. Muhammad Usman Arif<br/>
Code, data and the executed notebook: github.com/Samzzh0803/Pakistan-Food-Price-Premiums-and-Deprivation (branch milestone-03)</div>

<div class="abstract"><b>Abstract.</b> We ask whether Pakistan's weekly city-level food price data can warn of abnormal price moves a week
ahead. Using {g2('panel.weeks')} weekly PBS releases for 17 cities, we label a city-item-week as abnormal when next week's price change exceeds
the item's usual range ({pc('data.base_rate', 1)} of {g('data.rows'):,} rows). We compare two simple benchmarks with logistic regression, a
decision tree and k-nearest neighbours, using a time-ordered split and forward-chaining cross-validation. On four held-out weeks,
logistic regression reaches a ROC-AUC of {a2('test.logistic.roc_auc')}, better than chance in every week. A one-line rule that ranks items
by their category's past rate of abnormal moves does as well ({a2('test.category.roc_auc')}) and is the cheapest alerting policy once a
missed shock is valued highly. Weekly prices carry a modest early-warning signal, and most of it is the type of food.</div>

<h2>1. Introduction and data</h2>
<p>Pakistan's Sensitive Price Indicator is reported each week as one national figure, but the Pakistan Bureau of Statistics (PBS)
also publishes the prices behind it for {g2('panel.items')} items in {g2('panel.cities')} cities. That city detail could show whether some
places pay persistently more for food, whether those places are the ones least able to absorb it, and whether sudden local price
moves can be seen coming. The first two questions concern access to food (SDG 2.1) and inequality (SDG 10); the third concerns
functioning food markets and timely market information (SDG 2.c). Milestone 02 explored all three. This milestone builds and
evaluates the first classifiers for the third.</p>
""", html_table(sources, "Table 1. Data sources merged into the master dataset."),
f"""
<p>Each weekly annexure is an Excel sheet with three city tables stacked on top of each other. Our Milestone 01 parser read only the
first table and so reported seven cities; the corrected parser reads all three, giving {g2('panel.rows'):,} rows (17 cities, 51 items,
{g2('panel.weeks')} weeks, of which {g2('panel.food_items')} items are food). We validated every week against PBS's own national columns.
The cross-city minimum and maximum match PBS's national figures in all {g2('validation.item_weeks_checked'):,} item-weeks, and the geometric
mean of city prices reproduces PBS's national average in all of them, so we use the geometric mean as the national reference price.</p>
<p>Item categories, census population, district deprivation and the urban food price index were joined on item, city and month,
each by a many-to-one join whose row count we checked. Missing values were flagged rather than imputed. {g2('missing.structural_rows')} rows
where a city never quotes an item are kept as missing. Weekly changes exist only between releases a week apart, and gaps in the
published archive leave {g2('missing.lag_missing_pct'):.0f}% of rows without a previous week. Releases after 24 September 2026 were
downloaded but not opened; they are reserved for the new-data milestone.</p>

<h2>2. What the exploratory analysis showed</h2>
""", fig("m03r_eda", "Fig. 1. City food price premiums with 95% intervals (left; filled markers stay significant after a Bonferroni correction) "
         "and premium against district deprivation (right); the cities that pay most tend to be the least deprived."),
f"""
<p>For each city we averaged its log price relative to the national reference over the 32 food items. Islamabad pays
{g2('premium.city.Islamabad.pct_mean'):.0f}% above the reference and Rawalpindi {g2('premium.city.Rawalpindi.pct_mean'):.0f}%, while Sukkur and
Bannu pay about {abs(g2('premium.city.Bannu.pct_mean')):.0f}% less. {['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'][g2('premium.n_sig_bonferroni')]} cities remain significant after a
Bonferroni correction for 17 tests. The pattern persists. In {pc2('h1b.overall_stay_rate')} of {g2('h1b.n_pairs'):,} cases where a city
was above the cross-city median for an item, it was still above it the following week.</p>
<p>Perishables vary more across cities than storable staples (mean coefficient of variation {g2('h2.mean_cv_perishable'):.2f} against
{g2('h2.mean_cv_storable'):.2f}; one-tailed Welch p = {g2('h2.p_one_tailed'):.3f}). A rank-based test gives p = {g2('h2.mannwhitney_p_one_tailed'):.3f},
and each group has only about ten items, so we treat this as suggestive.</p>
<p>The cities paying the highest premiums are mostly the least deprived (Spearman ρ = {g2('h5.deprivation_composite.spearman_rho'):.2f},
p = {g2('h5.deprivation_composite.spearman_p'):.2f}, n = 17). Khuzdar and Quetta are the exception, combining above-reference prices with the
highest food insecurity in the panel. Data quality also varies by place. The share of records in which every quote PBS collected was
identical ranges from {pc2('h4.single_quote_min')} in {g2('h4.single_quote_min_city')} to {pc2('h4.single_quote_max')} in
{g2('h4.single_quote_max_city')}.</p>

<h2>3. Hypotheses</h2>
<ul>
<li>H1, persistent premiums (SDG 2.1): some cities pay more than the national reference for the same items, and stay above the median week to week.</li>
<li>H2, perishability and dispersion (SDG 2.c): perishables vary more across cities than storable staples.</li>
<li>H3, abnormal next-week movement (SDG 2.c): a classifier using only information available in week <i>t</i> ranks next week's abnormal
moves better than chance (H0: ROC-AUC = 0.5 on a time-ordered test set).</li>
<li>H4, representation and fairness (SDG 10): model errors and data quality differ across provinces, city sizes and deprivation levels.</li>
<li>H5, burden (SDG 2.1, SDG 10): city premiums are correlated with deprivation.</li>
</ul>
<p>This report tests H3. H1, H2 and H5 were examined in Milestone 02, and the formal H4 audit is planned for Milestone 04.</p>

<h2>4. Prediction task, features and models</h2>
<p>Each row is one city, one food item and one week <i>t</i>. The target is 1 when the item's price in that city changes between week
<i>t</i> and the following week by more than the 90th percentile of that item's non-zero weekly changes. We use non-zero changes because
{pc2('summary.dlog_share_zero_food')} of weekly food price changes are exactly zero; for {g2('h3.items_p90_zero')} of the 32 food items an
ordinary 90th percentile would be zero and every change would count as abnormal. The percentiles are fitted on the training weeks of
each split and fold, so the evaluation weeks never help define their own labels. Items with fewer than five non-zero changes in
training get no label. Because a label needs a release exactly one week later, only {g('data.labelled_weeks')} of the
{g('data.weeks_in_panel')} weeks can be labelled.</p>
<p>All features are known in week <i>t</i>. Recent movement is described by whether the price changed this week and by the size of this
week's and last week's changes. When the previous week was not published, the change is set to zero and a separate indicator records
that it was unobserved. Position relative to other cities is the log price against the national reference. Data quality enters
through the price spread across a city's outlets and whether only one effective quote was recorded. Item type is a set of category
indicators (perishable, branded or packaged, prepared food; storable staples are the reference), and place is described by province
indicators, log urban population and the district deprivation composite. We left out one indicator per item because, with about
200 positive rows, it would overfit and dominate the distances used by k-nearest neighbours.</p>
""", html_table(models, "Table 2. Benchmarks and models, why each suits the problem, and the settings searched and chosen in cross-validation."),
f"""
<p>Logistic regression and the tree use balanced class weights, so that the rare abnormal rows carry as much total weight as the
normal ones. Logistic regression inputs are standardised and k-nearest-neighbour inputs are scaled to 0-1; both scalers are fitted on
the training rows of each fold. All models were fitted with scikit-learn (Pedregosa et al., 2011).</p>

<h2>5. Experimental setup</h2>
<p>We train on the first {g('split.train_weeks')} labelled weeks ({g('split.train_first')} to {g('split.train_last')}) and test on the last
{g('split.test_weeks')} ({g('split.test_list')}). After labelling, training has {g('split.train_rows'):,} rows with {g('split.train_pos')} abnormal moves and
the test set has {g('split.test_rows'):,} rows with {g('split.test_pos')}. That is a {pc('split.row_share_train')}/{100 - 100 * g('split.row_share_train'):.0f}% split by rows,
and the base rate is {pc('split.test_rate', 1)} in both parts. A row in week <i>t</i> takes its label from week <i>t</i>+1, so any training row
whose <i>t</i>+1 falls in an evaluation week is removed. Here that purges the last training week, and the final models learn from
{g('split.train_weeks_after_purge')} weeks.</p>
<p>We did not use a random stratified split. The same 17 × {g('data.items_labelled')} city-item series appear every week and items in the
same week share shocks, so a random split would let the model learn from later weeks to predict earlier ones, which no price monitor
could do. Stratification exists to keep the rare class present in every part. We checked it directly. Every validation week holds
between {g('cv.val_pos_min')} and {g('cv.val_pos_max')} abnormal moves and the test set {g('split.test_pos')}. Balanced class weights then keep
the rare class from being ignored during training.</p>
<p>Cross-validation uses only labels that end before the test period, which leaves {g('cv.dev_weeks')} development weeks. Within them,
fold 1 trains on the first five labelled weeks and validates on the sixth, and each later fold adds one week, giving
{g('cv.folds')} folds. Label percentiles, scalers and models are refitted inside every fold. Each validation week has only {g('cv.val_pos_min')} to {g('cv.val_pos_max')}
abnormal moves, so we report the mean and standard deviation across folds.</p>
""", fig("fig15_split_design", "Fig. 2. Forward-chaining folds and the final split by labelled week; each fold trains only on weeks before "
         "the one it validates, and grey crosses mark weeks removed because their label comes from an evaluation week.", "70%"),
f"""

""", fig("m03r_tuning", "Fig. 3. Mean validation ROC-AUC for every setting searched; the regularisation strength barely matters, "
         "shallow trees do about as well as deep ones, and KNN only improves as K grows."),
f"""
<p>Settings were compared by mean validation ROC-AUC, which ranks predictions without needing a threshold. When several settings fell
within one fold-SD of the best, we chose the simplest. The strongest regularisation (C = 0.01) did best for logistic regression. A depth-2
tree was within one SD of the best depth-4 tree, so we kept the shallower one. KNN peaked at the largest K in the grid.</p>
<p>Two checks support the design. Ordinary shuffled five-fold cross-validation on the training rows gives a mean AUC of
{a2('leak.random_kfold_auc_mean')} for logistic regression, against {a2('leak.forward_auc_mean')} with forward chaining, which suggests
random folds would have overstated performance. When we refit the final logistic model on randomly permuted training labels
{g('leak.shuffled_reps')} times, the test AUC averages {a2('leak.shuffled_mean')}, and none of the permutations reaches the real AUC.</p>

<h2>6. Results</h2>
""", html_table(res, "Table 3. Test-set metrics for every model (threshold-based metrics at 0.5; ROC-AUC with a bootstrap 95% interval) "
                "and the cross-validation ROC-AUC used for model selection."),
fig("fig17_roc_pr", "Fig. 4. ROC curves (left) and precision-recall curves (right) on the test weeks; logistic regression and the "
    f"category rule sit close together, and all precision-recall curves stay near the {pc('test.majority.pr_auc')} base rate.", "96%"),
f"""
<p><b>We reject H0 for logistic regression on these four test weeks.</b> Its test ROC-AUC is {ci('logistic')}, and
it stays above 0.5 when each week is scored separately, ranging from {ow['min']['logistic']:.2f} to {ow['max']['logistic']:.2f}. The
decision tree's interval also excludes 0.5; that of k-nearest neighbours does not. The bootstrap interval treats rows as independent,
so the per-week check is the more trustworthy evidence, and four weeks are too few to call the result general (Section 9).</p>
<p>No model clearly beats the category rule. In cross-validation logistic regression averaged {a2('cv.logistic.roc_auc_mean')} against
{a2('cv.category.roc_auc_mean')} for the rule, but won only {g('cv.folds_logistic_beats_category')} of the {g('cv.folds')} folds. On the test
weeks the rule was slightly ahead, and the bootstrap interval for the difference between them includes zero. A logistic model given
only the three category indicators reaches a test AUC of {a2('ablation.cat_only_test_auc')}, slightly above the full model, so recent price
movement adds little to knowing what kind of food it is. Even logistic regression finds an abnormal move in only about
{pc('test.logistic.precision')} of the rows it flags.</p>
<p>Accuracy is misleading here. The majority class is {pc('test.majority.accuracy')} accurate while catching nothing, and the category
rule and k-nearest neighbours almost never produce a score above 0.5, so at that threshold they look the same. Balanced class weights
push logistic regression the other way. At 0.5 it flags {g('test.logistic.flagged_at_05')} rows to catch {g('test.logistic.tp_at_05')} of
{g('split.test_pos')} abnormal moves. K-nearest neighbours had the best cross-validation mean but the widest spread, and it fell to
{a2('test.knn.roc_auc')} on test. With about one abnormal row in {g('cost.inverse_base_rate'):.0f}, most small neighbourhoods contain none. The category indicators
also count as much in the distance as real price changes, and many rows are identical. Only very large neighbourhoods give a usable
score, and those amount to a smoothed local base rate.</p>

<h2>7. Error costs and diagnostics</h2>
<p>We have in mind a PBS analyst or a district price control committee deciding where to send inspectors next week. A false negative
is an abnormal move nobody was warned about. When the move is a jump in tomatoes, onions, flour or ghee, no inspection or public
notice happens and households absorb it, and households in more deprived districts spend a larger share of their income on food.</p>
<p>A false positive is an unnecessary inspection or watch-list entry. Each one costs staff time and goodwill with traders, and a
system whose alerts are mostly wrong soon stops being trusted.</p>
<p>Following Provost and Fawcett (2013), we give correct decisions zero cost, a false alarm a cost of 1 and a missed move a cost of
<i>r</i>, placing each cost in one cell only to avoid counting it twice. Flagging a row whose chance of being abnormal is <i>p</i> pays when
<i>p</i>·<i>r</i> &gt; 1 − <i>p</i>, so for an average row flagging everything beats flagging nothing only once <i>r</i> exceeds about
{g('cost.flag_all_breakeven_r'):.0f}. The value of <i>r</i> is a judgment for decision-makers, so we vary it. For each <i>r</i> the threshold
is chosen on the cross-validation predictions and then applied to the test weeks.</p>
""", html_table(costs, "Table 4. Test cost per 1,000 city-item-weeks, in units of one false alarm, for each cost ratio r (thresholds chosen "
                "in cross-validation)."),
f"""
<p>For <i>r</i> up to 10, issuing no alerts is cheapest. At <i>r</i> = 20 logistic regression and flagging nothing cost the same to the
nearest unit, and at <i>r</i> = 35 the category rule is cheapest. In these test weeks, alerting pays only when a missed move is judged to
be worth about 20 or more needless inspections, and even then the simple rule does at least as well. If inspectors can check only the
50 highest-scoring city-items each week, logistic regression puts {pc('budget.logistic.n50.recall')} of the abnormal moves in front of them,
with {pc('budget.logistic.n50.precision', 1)} of checks finding one, against {pc('budget.random_precision', 1)} for random checks.</p>
""", fig("m03r_confusion", "Fig. 5. Test-set confusion matrices in the standard layout (actual class in rows, abnormal first); the cost "
         "threshold at r = 20 catches most abnormal moves at the price of flagging about half of all rows.", "100%"),
html_table(errs, "Table 5. Logistic regression errors at the r = 20 threshold by item category and city deprivation tercile."),
f"""
<p>Table 5 shows where the errors fall. Branded and packaged goods had no abnormal moves in the test weeks and are never flagged,
while about {100 * eb['item_category']['perishable']['fp_rate']:.0f}% of normal perishable rows are flagged. More deprived cities receive more
false alarms and fewer misses than less deprived ones. The error analysis also shows what the model cannot see. Every abnormal move it
missed at this threshold came after a week with no observed price change. Quetta's broken basmati rice in the last test week is one
such case, with a flat price followed by a fall beyond the item's usual range. The most confident false alarms are perishables in Sukkur, Bannu
and Larkana that had just changed price on a single effective quote and then stayed put, such as onions in Larkana. A single shop's
quote can move without the market moving, which ties model errors to the data-quality gap noted in Section 2. The two earlier test
weeks had twice the abnormal rate of the August weeks; both fell within two weeks after an Eid holiday (21 March and 27 May 2026;
APP, 2026), which is a possible but untested explanation.</p>

<h2>8. Drivers and policy implications</h2>
""", fig("m03r_importance", "Fig. 6. Standardised logistic regression coefficients (left) and decision-tree importance (right); "
         "item type and recent price movement lead in both models.", "92%"),
f"""
<p>Both models put item type first. In the logistic model, a one-SD step towards a branded or packaged item multiplies the odds of an
abnormal move next week by {g('imp.cat_branded_packaged.odds_ratio'):.2f}, while a price change this week multiplies them by
{g('imp.price_changed.odds_ratio'):.2f} and being perishable by {g('imp.cat_perishable.odds_ratio'):.2f}. The size of this week's change gets a small
coefficient because it is correlated with whether the price changed at all, and the two share the credit. The tree uses the same
two ideas, splitting first on branded or packaged items and then on this week's change. Its four leaves read as rules (Table 6).</p>
""", html_table(rules_tab, "Table 6. The decision tree's leaves as rules, with each leaf's actual abnormal rate in the training weeks."),
f"""
<p>The root split shows how the tree chose. Without weights, separating branded items reduces Gini impurity only from
{g('gini_example.gini_parent'):.3f} to {g('gini_example.gini_weighted_children'):.3f}, because abnormal rows are so rare. With balanced weights
the root has a Gini of {g('gini_example.wgini_parent'):.1f}, and the same split lowers it to {g('gini_example.wgini_weighted_children'):.2f}, the
largest reduction available.</p>
<ol>
<li>Weekly monitoring should concentrate on perishables, especially those whose price moved this week. Branded and packaged goods
almost never move abnormally.</li>
<li>A model score is useful for deciding where an inspector looks first. With precision near {pc('test.logistic.precision')}, it should
never trigger stock releases or penalties on its own.</li>
<li>Any system PBS or a province builds should first be compared with the simple rule of watching perishables, which matched the
models here.</li>
<li>Publishing the weekly annexure every week would itself improve early warning, since {pc('data.share_no_dlog')} of labelled rows have no
observed previous week and the moves we miss are those without visible recent history.</li>
</ol>

<h2>9. Limitations and Milestone 04 plan</h2>
<p>The test set covers four weeks and {g('split.test_pos')} abnormal moves, and rows within a week share shocks, so every test figure has
wide uncertainty and the bootstrap intervals understate it. All results describe associations, not causes. Deprivation data come from
2019-20 district totals that include rural areas, population and deprivation are constant within each city, and Islamabad and
Rawalpindi are scored separately although they form one market. The 90th-percentile label is a convention, the archive gaps limit how
many weeks can be labelled, and the 17 urban SPI cities do not represent rural markets. The signal is modest throughout.</p>
<p>Milestone 04 will compare random forests and gradient boosting with today's benchmarks under the same cross-validation and cost
analysis. It will test H4 by comparing recall, miss rates and calibration across provinces, city sizes and deprivation terciles, and
will examine whether weighting misses in deprived districts more heavily, or using group-specific thresholds, changes who is
protected and at what cost. Finally, the frozen model will be scored on the unopened post-cutoff weeks, then retrained on the earlier
part of the new data and tested on the later part.</p>

<h2>References</h2>
<p class="refs">Associated Press of Pakistan (2026, 19 March). Eid-ul-Fitr on Saturday as Shawwal moon not sighted. app.com.pk.<br/>
Pakistan Bureau of Statistics. Weekly Sensitive Price Indicator, annexure workbooks, weeks ended {g2('panel.first_week')} to {g2('panel.last_week')}. pbs.gov.pk.<br/>
Pakistan Bureau of Statistics. Pakistan Social and Living Standards Measurement Survey 2019-20, District Level: Tables 2.14(a), 2.15, 7.1, 9.1.<br/>
Pakistan Bureau of Statistics. 7th Population and Housing Census 2023: Table 1, district results.<br/>
Pakistan Bureau of Statistics. CPI (Urban) group-wise indices, base 2015-16; Monthly Review on Price Indices, August and September 2026.<br/>
Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.<br/>
Provost, F. and Fawcett, T. (2013). <i>Data Science for Business</i>. O'Reilly, Chapters 7 and 8.</p>
</body></html>"""]

doc = "\n".join(H)
# Prose must not contain code names (identifiers with underscores) or file names.
prose = re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", "", doc, flags=re.S))
leftovers = sorted(set(re.findall(r"\b\w+_\w+\b", prose)))
assert not leftovers, f"code names left in the manuscript: {leftovers}"
out_html = REP / "M03_report.html"
out_html.write_text(doc, encoding="utf-8")
print("wrote", out_html)

edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
out_pdf = REP / "M03_report.pdf"
if edge.exists():
    subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out_pdf}",
                    out_html.resolve().as_uri()], check=True, timeout=180)
    print("wrote", out_pdf)
else:
    print("Edge not found; open the HTML and print to PDF")
