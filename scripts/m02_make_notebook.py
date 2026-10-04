"""Generate notebooks/M02_data_prep_eda.ipynb. The notebook calls the same pfp functions as the scripts."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Milestone 02: Data preparation, EDA and hypotheses
**Who Pays More, and Who Can Least Afford It? Food Price Premiums, Deprivation and Abnormal Price Movements Across Selected Pakistani Urban Markets**

CS/SDP 312/314 Data Science for Social Good, Habib University, Fall 2026.

This notebook runs top to bottom from the raw files in the repository, with no network access. It calls the same
functions as `scripts/m02_eda.py` (package `pfp/`). Every statistic quoted in the manuscript is written by this code
to `reports/m02/m02_numbers.json`.

Sections follow the course sequence. Part A uses techniques taught in Units 02 to 07. Part B uses later-syllabus
methods in **exploratory** form only; they are labelled as such and nothing in Part B is confirmatory.
""")
code("""
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.precision", 4)
from pfp import eda
from pfp.config import TRAIN_CUTOFF, BACKFILL_MANIFEST
from pfp.build import build_all
from pfp.external import parse_census_table1
eda.N.clear()
print("Training cutoff:", TRAIN_CUTOFF)
""")

md("""
## 1. Ingestion: parse every Appendix-A block and build the master dataset
`build_all()` parses all training-week workbooks (three stacked blocks, 17 cities, 51 items), runs the acceptance
assertions on each week, deduplicates byte-identical copies, merges the external sources and writes the master
dataset. Holdout weeks (after the cutoff) are skipped by file name before they are opened.
""")
code("""
o = build_all(write=True)
m, city = o["master"], o["city"]
manifest = pd.read_csv(BACKFILL_MANIFEST)
weeks = eda.inventory(o); eda.backfill_summary(manifest); eda.external_summary(o)
print(f"master: {m.shape}, weeks {m.week_end.min().date()} to {m.week_end.max().date()} ({m.week_end.nunique()})")
print(json.dumps({k: eda.N[k] for k in ["panel", "validation", "files", "backfill"]}, indent=1))
""")
md("""
### Acceptance checks against PBS's own national columns
For every item-week, the minimum and maximum across the 17 parsed cities must equal PBS's National Average MIN and MAX,
and the geometric mean of the quoting cities' AVG must be within 0.1% of PBS's National Average AVG. The arithmetic
mean and the median do not reproduce PBS's figure, which identifies the geometric mean as PBS's reference price.
""")
md("""### Merged external sources
City table: Census 2023 population and four PSLM 2019-20 deprivation indicators (district totals), plus the
composite. Urban food CPI by month is used as a deflator only.""")
code("""display(o["crosswalk"]); display(city.round(3)); o["cpi"]""")
code("""o["checks"].describe().loc[["mean", "min", "max"]]""")
code("""o["inventory"][["source_file", "week_end", "status", "duplicate_of"]]""")
code("""display(Image(eda.fig_coverage(o, manifest)))""")

md("## Part A. Techniques taught in Units 02 to 07")
md("""
### Unit 02: Attribute types (codebook)
`city_code` and `item_id` are nominal even though they look like numbers. Price is ratio. `week_end` is interval.
`rel_price` is interval, so ratios of it mean nothing. Terciles are ordinal.
""")
code("""codebook = eda.codebook(m); codebook""")

md("""
### Unit 03: Tidy data and melting
In the raw sheet, city names are column headers spanning MIN/AVG/MAX, and three tables are stacked vertically. That
breaks all three tidy rules: a variable (city) is stored in headers, one observation (city-item-week) is spread over
three cells in a row, and one sheet holds three tables. The parser melts it to one row per city, item and week.
""")
code("""
tidy = eda.tidy_demo(o, ROOT / o["inventory"].query("status == 'kept'").source_file.iloc[-1])
print("raw sheet (rows x cols):", tidy["raw_shape"], "-> tidy rows per week:", tidy["tidy_shape_per_week"][0])
m[m.week_end == m.week_end.max()][["week_end", "city", "city_code", "item_id", "item_short", "unit",
                                    "price_min", "price_avg", "price_max"]].head(6)
""")
md("""### Unit 03: Cleaning
Labels, implausible values and formats. We record what changed and delete nothing.""")
code("""eda.cleaning_log(o)""")
md("""
### Unit 03: Missing data
The options taught are: remove, impute from internal data, impute from external data, or fill with a constant. Below,
each of our three kinds of missingness is matched to a choice. The trade-off is statistical convenience against
selection bias. Imputing structural non-quotes would invent markets that PBS does not observe. Interpolating across
week gaps would invent price changes.
""")
code("""kinds, miss_cols = eda.missing_table(m); display(kinds); miss_cols""")
md("""### Unit 03: Integration (stacking, augmenting, merging)
Each join's key and cardinality; the row count is asserted unchanged after every merge (see `pfp/build.py`).""")
code("""eda.integration_table(o)""")

md("""
### Unit 04: Aggregation and change of scale
Moving from city to province to the national figure removes variability, and that removed variability is exactly
what this project studies. Moving from week to month barely changes cross-city dispersion because relative prices
are persistent.
""")
code("""agg = eda.aggregation(m); display(agg); display(Image(eda.fig_aggregation(agg)))""")
md("### Unit 04: Splitting `week_end` into year, month and ISO week")
code("""eda.splitting(m).head(8)""")
md("""
### Unit 04: Sampling and the bootstrap
The 17 SPI cities are a **purposive** sample chosen by PBS, not a random one, so nothing here generalises to towns or
rural markets. Inside the panel, we resample items with replacement to put an interval on each city's median premium.
""")
code("""prem = eda.city_premiums(m); boot = eda.bootstrap_median(m); boot""")
md("""
### Unit 04: Feature subset selection
**Redundant:** log MIN, AVG and MAX are almost perfectly correlated, so we keep AVG plus the within-city range.
**Irrelevant:** provenance columns. **Ranking H3 candidates:** we use the slide's mean-and-variance test,
|mean A - mean B| / SE(A - B), between rows labelled abnormal and normal next week. Thresholds are fitted on training
weeks only.
""")
code("""
labelled, thresholds = eda.abnormal_labels(m)
print(json.dumps(eda.N["h3"], indent=1))
eda.feature_selection(m, labelled)
""")
md("""
### Unit 04: Discretisation
Karachi's population skews the distribution, so equal-width bins put 15 of the 17 cities in the lowest bin. We
therefore use equal-frequency terciles for the contingency tables and the H4 strata.
""")
code("""disc = eda.discretisation(city); cityx = eda.add_terciles(city, prem); disc""")
md("""
### Unit 04: Normalisation
We use min-max scaling for the deprivation indicators (in `city_table`, columns `*_mm`) and z-scores for prices within
each item and week (`price_z_item_week`). Decimal scaling would divide wheat flour by 10^4 but the salt packet by
10^2, so it does not make items comparable, and we do not use it. Scaling matters for the distance-based methods in
Part B (clustering, nearest neighbours, PCA), which would otherwise give the most weight to the largest-valued
features. All constants are fitted on the training panel only.
""")
code("""display(eda.normalisation_demo(m, city)); city[["city"] + [c for c in city.columns if c.endswith("_mm")] + ["deprivation_composite"]].round(3)""")
md("""### Unit 04: Encoding
We use 0/1 dummy columns for province and item category. Cities use sum-to-zero (deviation) coding, so each city's
coefficient is its premium relative to the average city.""")
code("""display(eda.encoding_demo(m)); eda.sum_to_zero_matrix(eda.city_order(m)).iloc[[0, 1, 15, 16], :4]""")

md("""
### Unit 05: Central tendency, including the geometric mean
Mean, median and mode for four items in the latest week. The mode is only meaningful where prices cluster on round
figures. PBS's national average is the **geometric** mean of the city averages (validated above), and we use it as
the reference price.
""")
code("""eda.central_tendency(m)""")
md("""### Unit 05: Variation (range, IQR, variance, SD, coefficient of variation)
CV is the slide's measure for comparing spread across variables on different scales (PKR per kg, per dozen, per plate),
so it carries the cross-item dispersion analysis.""")
code("""item = eda.variation_by_item(m); item.sort_values("mean_cv", ascending=False).head(12)""")
code("""eda.summary_table(m, city)""")
md("""### Unit 05: Resistant and non-resistant statistics
We compare the mean premium with the median premium for each city. Where they diverge, a few items drive the mean,
and the median is the more resistant summary.""")
code("""prem[["city", "pct_mean", "pct_median", "n_items"]].assign(gap=lambda d: d.pct_mean - d.pct_median).sort_values("gap")""")
md("""
### Unit 05: Outlier identification (quartile and mean/SD methods, mild and regular tiers)
On weekly log changes the IQR is 0, because most prices do not change. The quartile fences then collapse to zero and
flag every change; this is the masking and degenerate-fence problem. The SD method is inflated by the large moves it
is meant to find. We check the extremes against PBS's own national MIN and MAX columns and keep them flagged, because
large moves are what H3 studies.
""")
code("""display(eda.outliers(m)); eda.extreme_check(m, o["national"])""")
code("""display(Image(eda.fig_relprice_hist(m)))""")
md("### Unit 05: Histograms, bar charts and box plots")
code("""display(Image(eda.fig_dispersion(item))); display(Image(eda.fig_box_province_category(m)))""")
md("""
### Unit 05: Scatter plots and the Anscombe lesson
We plot before computing any coefficient. With n = 17, one city can make or break a correlation, so we also report
leave-one-out ranges.
""")
code("""h5 = eda.premium_vs_deprivation(cityx); display(h5); display(Image(eda.fig_premium_deprivation(cityx, h5)))""")
md("### Unit 05: Covariance and correlation (Pearson and Spearman, city level, n = 17)")
code("""cov, pear, spear = eda.correlations(cityx, m); display(cov.round(4)); display(pear.round(2)); display(spear.round(2)); display(Image(eda.fig_corr(spear)))""")
md("### Unit 05: Categorical by categorical (province by premium tercile)")
code("""ct = eda.contingency(cityx); display(ct); display(Image(eda.fig_province_tercile(ct)))""")

md("""
### Unit 06: Normal distribution, z-values and the CLT
Pooled relative prices are roughly symmetric but heavy-tailed. Each city's mean premium averages over about 32 items,
which justifies t-based intervals by the CLT. That justification is weaker where a city's item-level means are skewed.
""")
code("""eda.normality(m, prem); {k: eda.N["normal"][k] for k in eda.N["normal"]}""")
md("""
### Unit 07: Confidence intervals with sigma unknown
Each observation is one food item's mean relative price for that city, so n is the number of food items (31 or 32),
not the number of rows.
""")
code("""prem[["city", "province", "n_items", "pct_mean", "ci_lo", "ci_hi", "t", "p", "sig_05", "bonferroni_sig"]]""")
code("""display(Image(eda.fig_city_premiums(prem)))""")
md("""
### Unit 07: Hypothesis tests, Type I and II errors
* **City premiums (two-tailed, 17 tests):** we report the count significant at 0.05 and the count that survives
  Bonferroni at 0.05/17. A Type I error would label a city as paying more when it does not; a Type II error would miss
  a real premium.
* **H2 (one-tailed):** mean cross-city CV, perishables against storables, using the SE(A - B) form. The effective
  sample size is the number of items.
* **Persistence:** each city's rate of staying above the median, tested against 0.5 with n = 17 cities, both for all
  pairs and for pairs where the price changed.
""")
code("""print(json.dumps(eda.N["premium"] | {}, indent=1, default=str)[:600]); eda.h2_test(item)""")
code("""pc, pres = eda.persistence(m); display(pc); pres""")

md("## Part B. Later-syllabus methods, exploratory only")
md("""### Measurement quality and representation (fairness, weeks 8 and 15): exploratory
Measurement quality is the share of food records where MIN = MAX, meaning the city average may rest on a single
quote. Representation compares the 17 SPI cities with each province's urban population (Census 2023).""")
code("""q = eda.measurement_quality(m, cityx); display(Image(eda.fig_single_quote(q))); eda.representation(city, parse_census_table1())""")
md("### Evaluation metrics (week 6): base rate and unchanged-price share, exploratory")
code("""display(eda.unchanged_by_category(m)); {k: eda.N["h3"][k] for k in ["base_rate_contract", "base_rate_refined"]}""")
md("### Distance and nearest neighbours (week 7): exploratory")
code("""mat = eda.city_item_matrix(m); eda.nearest_neighbours(mat)""")
md("### Hierarchical clustering, Ward linkage (weeks 13 and 14): exploratory")
code("""L, clusters = eda.hier_cluster(mat); display(pd.crosstab(clusters.cluster, clusters.province)); display(Image(eda.fig_dendrogram(L, mat)))""")
md("### Descriptive regression (week 10): exploratory, coefficients only")
code("""coef, simple = eda.regression(m, prem, cityx); display(coef.round(4)); print(eda.N["h1a"]); simple""")
md("### PCA: exploratory")
code("""p, scores = eda.pca(mat, prem); display(Image(eda.fig_pca(scores, p))); p.explained_variance_ratio_[:5]""")
md("### k-means market typology of items (week 13): exploratory")
code("""km, km_ct = eda.item_kmeans(item); km_ct""")
md("### H5 burden re-ranking (descriptive) and the time-ordered split design (week 8)")
code("""rr = eda.burden_reranking(cityx); display(rr); display(Image(eda.fig_reranking(rr))); display(Image(eda.fig_split_design(o)))""")

md("## Save the numbers file used by the manuscript")
code("""
eda.put("figures.count", len(list((ROOT / "reports/m02/figures").glob("fig*.png"))))
eda.save_numbers()
print("written", sum(1 for _ in json.dumps(eda.N)), "chars to reports/m02/m02_numbers.json")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "M02_data_prep_eda.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
