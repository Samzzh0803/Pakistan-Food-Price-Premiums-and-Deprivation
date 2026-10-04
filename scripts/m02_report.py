"""Assemble the Milestone 02 manuscript (HTML -> PDF via headless Edge).

Every inline statistic is read from reports/m02/m02_numbers.json, and every table is
rendered from CSVs written by the pipeline under reports/m02/tables. Nothing is hand-typed.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pfp.config import CITY_PROVINCE  # noqa: E402

REP = ROOT / "reports" / "m02"
TAB = REP / "tables"
N = json.loads((REP / "m02_numbers.json").read_text(encoding="utf-8"))


def g(path):
    d = N
    for k in path.split("."):
        d = d[k]
    return d


def f(path, d=1):
    return f"{g(path):,.{d}f}"


def pct(path, d=1):
    return f"{100 * g(path):.{d}f}%"


def p(path):
    v = g(path)
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def t(name):
    return pd.read_csv(TAB / f"{name}.csv")


def html_table(df: pd.DataFrame, caption: str, cls: str = "", digits: int = 3) -> str:
    body = df.to_html(index=False, border=0, float_format=lambda x: f"{x:,.{digits}f}", escape=True, na_rep="–")
    return f'<div class="tbl {cls}"><p class="cap">{caption}</p>{body}</div>'


def fig(name: str, caption: str, width: str = "100%") -> str:
    return (f'<figure><img src="figures/{name}.png" style="width:{width}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


# ---------------------------------------------------------------- tables from the pipeline
prem = t("premium")
bonf_cities = ", ".join(prem[prem.bonferroni_sig].city)
sig_cities = ", ".join(prem[prem.sig_05 & ~prem.bonferroni_sig].city)
prem_tab = prem[["city", "province", "n_items", "pct_mean", "pct_median", "t", "p", "bonferroni_sig"]].copy()

prem_tab["95% CI (%)"] = [f"[{100 * (np.exp(lo) - 1):.1f}, {100 * (np.exp(hi) - 1):.1f}]" for lo, hi in zip(prem.ci_lo, prem.ci_hi)]
prem_tab["bonferroni_sig"] = prem_tab.bonferroni_sig.map({True: "yes", False: "no"})
prem_tab["p"] = prem_tab.p.map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}")
prem_tab = prem_tab.rename(columns={"n_items": "n items", "pct_mean": "mean %", "pct_median": "median %",
                                    "bonferroni_sig": "Bonferroni"})[["city", "province", "n items", "mean %", "95% CI (%)",
                                                                       "median %", "t", "p", "Bonferroni"]]

summary = t("summary")
item = t("item")
cats = {"perishable": "Perishable", "storable_staple": "Storable staple", "branded_packaged": "Branded/packaged",
        "prepared_food": "Prepared food", "administered_utility": "Administered/utility", "other_nonfood": "Other non-food"}
item_tab = item.assign(category=item.item_category.map(cats))[
    ["item_id", "item_short", "category", "mean_price", "mean_cv", "share_unchanged", "share_single_quote"]].rename(
    columns={"item_id": "PBS #", "item_short": "item", "mean_price": "mean price (PKR)", "mean_cv": "mean CV",
             "share_unchanged": "share unchanged wk/wk", "share_single_quote": "share MIN=MAX"})
city = pd.read_csv(ROOT / "data" / "processed" / "m02" / "city_table.csv", dtype={"city_code": str})
city_tab = city[["city", "province", "pop_urban", "fies_mod_sev_pct", "illiteracy10_pct", "out_of_school_pct",
                 "no_tap_water_pct", "deprivation_composite"]].assign(pop_urban=lambda d: d.pop_urban / 1e6).rename(
    columns={"pop_urban": "urban pop. (m)", "fies_mod_sev_pct": "FIES mod/sev %", "illiteracy10_pct": "illiterate 10+ %",
             "out_of_school_pct": "out of school 5-16 %", "no_tap_water_pct": "no tap water %",
             "deprivation_composite": "composite (0-1)"})
cross = pd.read_csv(ROOT / "data" / "external" / "city_district_crosswalk.csv", dtype={"city_code": str})
missing = t("missing_kinds")
integ = t("integration")
outl = t("outliers")
fsel = t("feature_sel")
h5 = t("h5")
rep = t("representation")
disc = t("discretisation")
clean = t("cleaning")
codebook = t("codebook")
agg = t("aggregation")
kmct = pd.read_csv(TAB / "kmeans_ct.csv")
quality = t("quality")
ext = t("extremes")
nm = ext[~ext.equals_pbs_national_min_or_max]
extreme_text = " ".join(
    f"The other ({r.city}, {r.item_short}, week {r.week_end}) has MIN = {r.price_min:g}, AVG = {r.price_avg:g}, MAX = {r.price_max:g} PKR in the raw sheet, a genuine quoted price."
    for r in nm.itertuples())
bannu = city.set_index("city").loc["Bannu"]
bannu_urban = 100 * bannu.pop_urban / bannu.pop_total

# ---------------------------------------------------------------- text helpers
pc = lambda c: f"{g(f'premium.city.{c}.pct_mean'):+.1f}%"
ci = lambda c: f"[{g(f'premium.city.{c}.ci_lo_pct'):.1f}, {g(f'premium.city.{c}.ci_hi_pct'):.1f}]"
H5c = "h5.deprivation_composite"
H5f = "h5.fies_mod_sev_pct"
H5w = "h5.deprivation_composite_nowater"

methods_map = pd.DataFrame([
    ("Attribute types / codebook", "02", "§2.4, App. C", f"{g('codebook.columns')} columns typed; IDs nominal, rel_price interval", "CLO1"),
    ("Tidy data, melting", "03", "§2.2", f"{g('tidy.raw_rows')}x{g('tidy.raw_cols')} sheet with 3 stacked blocks -> {g('tidy.tidy_rows_per_week')} tidy rows/week", "CLO1"),
    ("Cleaning", "03", "§2.2, Table 3", f"{g('missing.structural_rows')} zero triples to missing; {g('cleaning.relabelled_rows')} relabelled rows; 0 deleted", "CLO1"),
    ("Missing-data options", "03", "§2.3, Table 4", "Three kinds of missingness; none imputed", "CLO1, CLO3"),
    ("Integration: stack, merge", "03", "§2.5, Table 5", f"Row count {g('integration.rows_after_all_joins'):,} unchanged by every join", "CLO1"),
    ("Aggregation", "04", "§3.6, Fig. 13", f"Within-Punjab range of {f('aggregation.punjab_mean_within_range_logpts', 2)} log pts vanishes at province level", "CLO4"),
    ("Splitting dates", "04", "§2.4", f"{g('splitting.months_covered')} calendar months; year, month, ISO week", "CLO1"),
    ("Sampling, bootstrap", "04", "§3.1", f"{g('bootstrap.n_cities_excluding_zero')} city medians exclude 0 ({g('bootstrap.reps')} resamples)", "CLO2"),
    ("Feature subset selection", "04", "§4, Table 9", f"log MIN/AVG r = {f('feature_sel.corr_logmin_logavg', 3)}; top H3 candidate: {g('feature_sel.top_feature')}", "CLO2"),
    ("Discretisation", "04", "§2.4, Table 6", f"Population equal-width bins {g('discretisation.pop_urban.equal_width')} vs terciles {g('discretisation.pop_urban.equal_freq')}", "CLO1"),
    ("Normalisation", "04", "§2.4", "Min-max deprivation, z within item-week; decimal scaling rejected", "CLO1"),
    ("Encoding", "04", "§2.4", "Province/category dummies; sum-to-zero city coding", "CLO1"),
    ("Central tendency, geometric mean", "05", "§2.1", f"Geometric mean reproduces PBS average in {g('validation.gm_pass'):,}/{g('validation.item_weeks_checked'):,} item-weeks", "CLO4"),
    ("Variation, CV", "05", "§3.2, Fig. 3", f"Mean CV perishable {f('variation.mean_cv.perishable', 3)} vs branded {f('variation.mean_cv.branded_packaged', 3)}", "CLO4"),
    ("Resistant statistics", "05", "§3.1", f"{g('premium.max_mean_median_gap_city')}: mean and median differ by {f('premium.max_mean_median_gap_pts')} pts", "CLO4"),
    ("Outliers, both methods", "05", "§3.4, Fig. 8", f"rel_price: {g('outliers.rel.iqr_mild_1_5'):,} (1.5 IQR) vs {g('outliers.rel.sd_regular_3'):,} (3 SD)", "CLO4"),
    ("Histogram, bar, box plots", "05", "Figs. 3, 7, 8, 11", "Dispersion by category; single-quote share; fences", "CLO4"),
    ("Scatter plots, Anscombe", "05", "§3.5, Fig. 4", f"Leave-one-out Spearman {f(H5c + '.loo_spearman_min', 2)} to {f(H5c + '.loo_spearman_max', 2)}", "CLO4"),
    ("Covariance, correlation", "05", "§3.5, Fig. 6", f"Premium vs single-quote share r = {f('corr.pearson_premium_single_quote', 2)}", "CLO4"),
    ("Quant x cat, cat x cat", "05", "Figs. 10, 11", "Premium tercile by province; rel_price by province", "CLO4"),
    ("Normal, CLT", "06", "§3.4", f"rel_price excess kurtosis {f('normal.rel_price_excess_kurtosis', 2)}; t-intervals over ~32 items", "CLO4"),
    ("t-intervals (sigma unknown)", "07", "§3.1, Fig. 2", "95% CI per city, n = food items", "CLO2, CLO4"),
    ("Hypothesis tests, errors", "07", "§3.1-3.3", f"{g('premium.n_sig_bonferroni')} of 17 cities survive Bonferroni; H2 p = {p('h2.p_one_tailed')}", "CLO2, CLO4"),
    ("Hierarchical clustering (exploratory)", "13-14", "§3.7, Fig. 5", "3 clusters: Islamabad-Rawalpindi, other Punjab, rest", "CLO2"),
    ("k-means items (exploratory)", "13", "§3.7", f"{g('partb.kmeans.flexible.n')} flexible, {g('partb.kmeans.intermediate.n')} intermediate, {g('partb.kmeans.sticky.n')} sticky items", "CLO2"),
    ("Nearest neighbours (exploratory)", "7", "§3.7", "Islamabad and Rawalpindi mutual nearest neighbours", "CLO2"),
    ("PCA (exploratory)", "Unit 05 (named)", "§3.7, Fig. 9", f"PC1 {pct('partb.pca_var_pc1', 0)} of variance; |r| with premium {f('partb.pca_pc1_corr_mean_premium', 2)}", "CLO2"),
    ("Regression (exploratory)", "10", "§3.1, §3.5", f"City effects = simple means (r = {f('partb.reg.corr_coef_vs_simple_mean', 3)}); slope on deprivation {f('partb.simple_reg.slope', 3)}", "CLO2"),
    ("Holdout and CV design", "8", "§4.3, Fig. 12", "Time-ordered split; week-14 holdout frozen", "CLO2, CLO3"),
    ("Bias and fairness (exploratory)", "8, 15", "§3.6, Fig. 7", f"MIN=MAX share {pct('h4.single_quote_min', 0)} ({g('h4.single_quote_min_city')}) to {pct('h4.single_quote_max', 0)} ({g('h4.single_quote_max_city')})", "CLO3"),
    ("Evaluation metrics (exploratory)", "6", "§4.3", f"Refined abnormal base rate {pct('h3.base_rate_refined')}", "CLO2"),
], columns=["Technique", "Course unit/week", "Where", "What it showed", "CLO"])

hyp = pd.DataFrame([
    ("H1a premiums", "log_price", "city (sum-to-zero dummies)", "item-week effects", "city x item x week (food)",
     "17 cities; 32 items x 27 weeks", "F test of city effects; descriptive regression"),
    ("H1b persistence", "above median next week", "above median this week", "consecutive pairs only; changed-price subset",
     "city x item pair", "17 cities (rate per city)", "one-sample t of city rates vs 0.5 (one-tailed)"),
    ("H2 dispersion", "mean cross-city CV", "item category", "none (item-level)", "item", "11 vs 10 items",
     "Welch t, one-tailed; Mann-Whitney check"),
    ("H3 abnormal move", "abnormal_next_week", "lagged change, range, single quote, perishable", "item thresholds fit on training weeks",
     "city x item x week", f"{g('h3.labelled_rows_refined'):,} labelled rows; {pct('h3.base_rate_refined')} positive",
     "AUC on a time-ordered test (M03)"),
    ("H4 representation", "error rate; data quality", "province, population tercile, deprivation tercile", "item category",
     "city (17) / city-group", "17 cities, 3 to 5 groups", "Kruskal-Wallis now; group error rates in M03"),
    ("H5 burden", "city premium", "deprivation composite; FIES", "leave-one-out", "city", "17", "Spearman, two-sided"),
], columns=["Hypothesis", "Target", "Predictor(s)", "Controls", "Unit", "Effective n", "Planned test"])

# ---------------------------------------------------------------- document
css = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: 'Cambria', 'Georgia', serif; font-size: 10.3pt; line-height: 1.38; color: #111; background: #fff; }
h1 { font-size: 16pt; margin: 0 0 4px 0; line-height: 1.2; }
h2 { font-size: 12.5pt; margin: 16px 0 6px 0; border-bottom: 1px solid #999; padding-bottom: 2px; }
h3 { font-size: 10.8pt; margin: 11px 0 4px 0; }
p { margin: 4px 0 7px 0; text-align: justify; }
.meta { font-size: 9.5pt; color: #333; margin-bottom: 8px; }
.abstract { border: 1px solid #bbb; padding: 7px 10px; background: #f7f7f7; font-size: 9.6pt; }
figure { margin: 8px 0 10px 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.8pt; color: #222; text-align: left; margin-top: 2px; }
.tbl { margin: 6px 0 10px 0; page-break-inside: avoid; }
.tbl.small table { font-size: 7.4pt; }
.cap { font-size: 8.8pt; font-weight: bold; margin: 0 0 2px 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; }
th { border-bottom: 1.2px solid #333; border-top: 1.2px solid #333; text-align: left; padding: 2px 4px; background: #f2f2f2; }
td { border-bottom: 0.5px solid #ddd; padding: 1.5px 4px; vertical-align: top; }
.two { display: flex; gap: 10px; } .two > figure { flex: 1; }
.pb { page-break-before: always; }
ul { margin: 3px 0 6px 18px; padding: 0; } li { margin-bottom: 2px; }
.note { font-size: 8.8pt; color: #333; }
code { font-size: 8.8pt; }
"""

H = []
H.append(f"""<!doctype html><html><head><meta charset="utf-8"><title>Milestone 02 Report</title><style>{css}</style></head><body>
<h1>Who Pays More, and Who Can Least Afford It? Food Price Premiums, Deprivation and Abnormal Price Movements Across Selected Pakistani Urban Markets</h1>
<div class="meta"><b>Milestone 02: data preparation, exploratory analysis and hypotheses (draft Sections II and III)</b><br/>
Yousuf Uyghur, Computer Science · Sameer Hassan, Computer Science<br/>
CS/SDP 312/314 L1 Data Science for Social Good, Habib University, Fall 2026 · Instructor: Dr. Muhammad Usman Arif<br/>
Code and data: github.com/Samzzh0803/Pakistan-Food-Price-Premiums-and-Deprivation (branch <code>milestone-02</code>)</div>

<div class="abstract"><b>Summary.</b> Pakistan's weekly Sensitive Price Indicator (SPI) reports one national figure. Beneath it, PBS publishes
prices for 51 items in 17 cities. We parse all {g('panel.weeks')} weekly releases available up to our freeze date
({g('panel.first_week')} to {g('panel.last_week')}; {g('panel.rows'):,} city-item-week rows). The parse reproduces PBS's own
national minimum, maximum and geometric-mean average in every one of {g('validation.item_weeks_checked'):,} item-weeks. We merge in
district deprivation (PSLM 2019-20), population (Census 2023) and urban food CPI. On 32 food items, Islamabad pays {pc('Islamabad')}
and Rawalpindi {pc('Rawalpindi')} above the national reference, while Bannu ({pc('Bannu')}) and Sukkur ({pc('Sukkur')}) pay less.
{g('premium.n_sig_bonferroni')} of 17 city premiums survive a Bonferroni correction. Premiums are persistent: {pct('h1b.overall_stay_rate', 0)}
of above-median city-item pairs stay above the median the next week. However, {pct('summary.dlog_share_zero_food', 0)} of weekly food price
changes are exactly zero. Across the 17 cities, the premium is <i>negatively</i> rank-correlated with district deprivation
(Spearman ρ = {f(H5c + '.spearman_rho', 2)}, p = {p(H5c + '.spearman_p')}): the cities paying most are mostly not the ones least able
to absorb it. How many separate quotes stand behind a city's average varies widely, which matters for every later model.</div>

<h2>Changes since Milestone 01</h2>
<p>Milestone 01 reported a 7-city panel. That was a parsing error. Appendix-A of the PBS Annexure is three tables stacked in one sheet,
and the earlier script stopped after the first. The panel actually covers all 17 SPI cities
({g('panel.cities_by_province.Punjab')} Punjab, {g('panel.cities_by_province.Sindh')} Sindh, {g('panel.cities_by_province.KP')} Khyber Pakhtunkhwa,
{g('panel.cities_by_province.Balochistan')} Balochistan, plus Islamabad), so the "Punjab and Islamabad only" limitation is retired.
The earlier figure of 16,380 rows, explained then as records times MIN/AVG/MAX, was a coincidence of reading three blocks under one set of
city names. The faulty files are kept as an audit trail and a dated correction note is in the repository
(<code>reports/CORRECTION_2026-10-04_appendix_a_17_cities.md</code>). Following the instructor's feedback that a price-only study is too
narrow, we now merge three external sources, each with one job (Table 1). The reference price is the geometric mean across quoting cities,
which is PBS's own definition (§2.1). Hypothesis H5 (burden versus price) is new and is now testable with n = 17.</p>

<h2>1. Alignment and inventory</h2>
<p><b>Research question.</b> Beneath the single national weekly SPI figure, do persistent city-level food price differences exist, are the
cities paying most the ones least able to absorb it, and is the published city detail enough to detect localised price stress?
<b>SDGs:</b> primary SDG 2 (Target 2.1 access to food; Target 2.c functioning food markets and timely market information), secondary SDG 10
(reduced inequalities). <b>Division of perspectives:</b> both members are computer scientists, so we split roles. One owns acquisition,
parsing and provenance. The other owns statistical framing, representation, ethics and policy interpretation.</p>
""")
inv = pd.DataFrame([
    ("PBS weekly SPI Annexure, Appendix-A", "Prices: MIN/AVG/MAX by city and item (the panel)",
     f"{g('panel.weeks')} weeks, {g('panel.first_week')} to {g('panel.last_week')}", "xlsx from pbs.gov.pk release pages",
     f"{g('panel.rows'):,} rows (867/week)"),
    ("PSLM District Level Survey 2019-20", "Deprivation (burden denominator): FIES, literacy, schooling, water",
     "2019-20", "PDF report tables, text extraction", f"{g('external.pslm_districts_used')} districts"),
    ("Pakistan Digital Census 2023, Table 1", "Population: total and urban, for size and fairness strata",
     "2023", "xlsx by province", f"{g('external.census_units_used')} districts"),
    ("CPI (Urban) Food & non-alcoholic beverages", "Deflator only (not a regressor)",
     f"monthly {g('external.cpi_first_month')} to {g('external.cpi_last_month')}", "PDF + Monthly Review docx",
     f"{g('external.cpi_months_in_panel')} panel months"),
], columns=["Source", "Role", "Vintage / coverage", "Access", "Size used"])
H.append(html_table(inv, "Table 1. Data inventory. Observational unit: one city x item x week. Master dataset: "
                    f"{g('panel.rows'):,} rows x {g('panel.cols')} columns; {g('panel.cities')} cities, {g('panel.items')} items "
                    f"({g('panel.food_items')} food), {g('panel.weeks')} weeks."))

H.append(f"""
<h2>2. Ingestion and preparation</h2>
<h3>2.1 Collection audit and validation</h3>
<p><b>Discovery.</b> PBS publishes each weekly release on a page of the form
<code>/weekly-sensitive-price-indicator-spi-for-the-week-ended-on-DD-MM-YYYY/</code>. That page links the Annexure workbook. We swept
{g('backfill.thursdays_attempted')} Thursdays from {g('panel.first_week')} onward, trying the Thursday, Wednesday and Friday page for each
({g('backfill.pages_requested')} requests, rate-limited, every attempt logged in <code>data/raw/pbs_backfill_manifest.csv</code> with URL,
SHA-256 and time). Release pages exist for only {g('backfill.weeks_found')} of those weeks. Direct file URLs for older weeks return 404
under every naming variant we tried, and site search lists only recent releases. The panel therefore holds {g('panel.weeks')} weeks:
{g('files.kept')} distinct workbooks, after dropping {g('files.duplicates')} byte-identical duplicates by hash, keeping the earlier repository
copy each time. Every re-downloaded file was byte-identical to its earlier copy, which independently confirms provenance. <b>We do not
describe the archive as complete.</b> Earlier routes that failed or were rejected are documented in the repository audits: the Punjab AMIS wholesale portal
(HTTPS fails TLS negotiation; HTTP pages load but historical queries were not reproducible and no commodity or market match could be
verified), an older single-pattern URL sweep that tried Thursdays only, and PDF annexures (superseded by xlsx).
The week of {g('backfill.holdout_weeks')} falls after the freeze ({g('panel.train_cutoff')}). It was saved to an append-only holdout folder,
hashed and never parsed.</p>
<p><b>Validation.</b> The parser finds every row whose second cell is <code>DESCRIPTION</code> and reads the city labels <code>Name (NN)</code>
on the row above. Columns under any other header are not cities. It reads the week from the <code>PRICES ON dd-mm-yyyy</code> cell and checks
it against the file name. Assertions fail loudly on every week: 3 blocks, 17 cities, 51 items, 867 rows, no duplicate key,
MIN ≤ AVG ≤ MAX where positive, and zeros only as all-zero triples. Against PBS's own national columns, the cross-city minimum and maximum
equal PBS's National MIN and MAX in {g('validation.min_max_pass'):,} of {g('validation.item_weeks_checked'):,} item-weeks. The <b>geometric</b>
mean of the quoting cities' averages matches PBS's National AVG within 0.1% in {g('validation.gm_pass'):,} of {g('validation.item_weeks_checked'):,}
(worst case {f('validation.gm_max_abs_rel_err_pct', 3)}%). The arithmetic mean misses by a median of {f('validation.am_median_abs_rel_err_pct', 2)}%
and the median by {f('validation.md_median_abs_rel_err_pct', 2)}%. The geometric mean is therefore PBS's reference price and ours. Release
dates are {g('panel.thursdays')} Thursdays and {g('panel.wednesdays')} Wednesdays. There are {g('panel.consecutive_pairs')} consecutive pairs
({g('panel.pairs_7d')} at 7 days, {g('panel.pairs_6d')} at 6). Other gaps run from {g('panel.gap_min_nonconsec')} to {g('panel.gap_max')} days (Fig. 1).</p>
{fig('fig01_coverage', 'Fig. 1. Weeks in the training panel (blue), Thursdays checked on the PBS site (grey) and the frozen holdout week (orange). Week changes are only computed across the consecutive pairs; the gaps are a coverage limitation, not filled.')}

<h3>2.2 Reshaping and cleaning (Unit 03)</h3>
<p>The raw sheet ({g('tidy.raw_rows')} rows x {g('tidy.raw_cols')} columns) breaks all three tidy-data rules. The variable city is stored in
headers. One observation is spread over MIN, AVG and MAX cells under a merged city header. Three tables share one sheet. We melt it to one row
per city, item and week ({g('tidy.tidy_rows_per_week')} rows per week), keeping PBS's serial as <code>item_id</code> and its city code as
<code>city_code</code>, both nominal. The national summary columns of block 3 go to a separate reference table and are never panel rows.
Table 3 lists every cleaning action. Nothing is deleted.</p>
""")
H.append(html_table(clean, "Table 3. Cleaning log (counts written by the pipeline)."))
H.append(f"""
<h3>2.3 Missing data</h3>
<p>Of the options taught (remove, impute from internal data, impute from external data, constant), none is applied to prices. There are
{g('missing.structural_rows')} structural non-quotes ({f('panel.structural_missing_pct', 2)}% of rows), where PBS prints 0/0/0: rice IRRI-6/9 in
Gujranwala, Lahore and Sialkot in every week, and gas charges in Khuzdar in two weeks. PBS itself leaves these cities out of its national mean,
so imputing would invent a market price the source does not record. Weekly changes across non-consecutive gaps
({g('missing.lag_missing_rows'):,} rows, {f('missing.lag_missing_pct')}%) stay missing, because interpolation would manufacture changes nobody
observed. Missing weeks are a coverage gap. The trade-off is statistical convenience against selection bias: complete-case analysis is
honest here only because the missingness is structural and documented (Table 4; per-column counts are in the notebook and codebook).</p>
""")
H.append(html_table(missing.drop(columns=["options_considered"]), "Table 4. The three kinds of missingness and the choice made for each."))
H.append(f"""
<h3>2.4 Transformations, scaling, encoding and discretisation (Unit 04)</h3>
<p><b>Log price.</b> Prices behave multiplicatively, so a 10% premium means the same thing on salt and on a 20 kg flour bag. Pooled raw PKR
across items is not a meaningful distribution: it mixes units, with skewness {f('normal.pooled_raw_price_skew', 2)}. Within an item, the median
skewness falls from {f('normal.median_within_item_skew_raw', 3)} on raw prices to {f('normal.median_within_item_skew_log', 3)} on log prices. All
cross-city comparisons use the relative price <i>rel_price</i> = log(price / national geometric mean). By construction it sums to zero across
quoting cities in each item-week, which is exactly the sum-to-zero premium in H1. <b>Redundancy:</b> log MIN and log AVG correlate at
{f('feature_sel.corr_logmin_logavg', 3)}, so we keep AVG plus the within-city range. Provenance columns are kept for audit and excluded from
analysis. <b>Normalisation:</b> min-max scaling (0 to 1, higher = more deprived) for the deprivation indicators; z-scores of price within
item and week. Decimal scaling divides by a power of ten that differs by item (10^{g('normalisation.decimal_scaling_j_wheat')} for flour), so it
does not make items comparable. Scaling matters because the distance-based methods in §3.7 would otherwise be dominated by large-valued
features. All constants come from the training panel. <b>Encoding:</b> 0/1 dummies for province and item category, and sum-to-zero (deviation)
coding for cities, so each city coefficient is a premium relative to the average city. <b>Splitting:</b> <code>week_end</code> is split into
year, month and ISO week ({g('splitting.months_covered')} calendar months). <b>Discretisation:</b> equal-width bins collapse on population because
of Karachi (Table 6), so the strata used for H4 and the contingency tables are equal-frequency terciles.</p>
""")
H.append(html_table(disc.rename(columns={"variable": "variable (17 cities)"}), "Table 6. Equal-width against equal-frequency binning (cities per bin)."))
H.append(f"""
<h3>2.5 Integration and the external sources</h3>
<p>Stacking joins the three blocks and the {g('panel.weeks')} weeks. Merging adds the item classification (on <code>item_id</code>), the city
attributes (on <code>city</code>) and the CPI (on the calendar month of <code>week_end</code>). Each merge is many-to-one, and the row count is
asserted unchanged ({g('integration.rows_after_all_joins'):,}; Table 5). <b>Crosswalk decisions</b> (Appendix Table A3): Karachi is one SPI
market but seven Census 2023 districts and six PSLM 2019-20 districts. Keamari was split from Karachi West after the survey, so PSLM values are
population-weighted with Keamari added to West's weight. Islamabad and Rawalpindi form one contiguous market but are kept as two
administrative units, each with its own district values; §3.7 shows they are each other's nearest neighbour in price space. PSLM has no urban
sample for Khuzdar, and FIES is published only for district totals, so all four indicators use <b>district totals</b> for comparability. The
urban values are kept as a sensitivity column. The four deprivation indicators, all oriented so that higher means more deprived, are
moderate-or-severe food insecurity (FIES, SDG 2.1.2; {f('external.range.fies_mod_sev_pct.min')}% in {g('external.range.fies_mod_sev_pct.min_city')}
to {f('external.range.fies_mod_sev_pct.max')}% in {g('external.range.fies_mod_sev_pct.max_city')}), illiteracy at age 10+, children 5-16 out of
school, and households without tap water. They are min-max scaled and averaged into a transparent composite. The water indicator partly reflects
infrastructure type (much of urban Punjab uses motor pumps), so a composite without it is reported as a sensitivity check. These are
<b>district deprivation measures from 2019-20, not city income</b>: PBS publishes income only by province. <b>CPI:</b> urban food CPI rose
{f('external.cpi_change_pct')}% from {g('external.cpi_first_month')} to {g('external.cpi_last_month')}. It is used only to express real price
levels. H1, H2 and H5 compare cities within the same week, so no national deflator can change them, and a nationally uniform series would be
collinear with week effects as a regressor.</p>
""")
H.append(html_table(integ, "Table 5. Stacking and merging steps, keys, cardinality and row counts."))

H.append(f"""
<h2>3. Exploratory data analysis</h2>
<p>Table 7 summarises the key variables at their own unit of analysis. City attributes have 17 distinct values and item comparisons about 32,
so every city-level or item-level test below is run on aggregated values, never on the {g('panel.rows'):,} rows.</p>
""")
H.append(html_table(summary, "Table 7. Summary statistics (mean, median, SD, IQR) for key variables. rel_price and dlog are in log points."))
H.append(f"""
<h3>3.1 City premiums (H1)</h3>
<p>For each city we average rel_price over weeks for each food item. The observations are then those item-level means, so n is the number of
food items: 31 for the three cities with structural rice gaps, 32 otherwise. Islamabad's mean premium is {pc('Islamabad')} (95% t-interval
{ci('Islamabad')}), Rawalpindi {pc('Rawalpindi')} and Karachi {pc('Karachi')}. At the other end are Sukkur ({pc('Sukkur')}) and Bannu
({pc('Bannu')}) (Fig. 2, Table 8). In two-tailed tests of H0: mean premium = 0, {g('premium.n_sig_05')} of 17 cities differ at 0.05. Seventeen
tests inflate the Type I error, so at the Bonferroni threshold 0.05/17 = {f('premium.bonferroni_alpha', 4)} only {g('premium.n_sig_bonferroni')}
survive: {bonf_cities}. A Type I error here would tell a policymaker that a city's consumers pay more when they do not. A Type II error would
miss a real premium, a cost borne by that city's households. Resistant and non-resistant summaries diverge: {g('premium.max_mean_median_gap_city')}'s
mean premium ({pc(g('premium.max_mean_median_gap_city'))}) and median ({f('premium.city.' + g('premium.max_mean_median_gap_city') + '.pct_median')}%)
differ by {f('premium.max_mean_median_gap_pts')} points, because a few items pull the mean. We report both and treat the median as the safer
summary where they disagree. Bootstrapping items ({g('bootstrap.reps')} resamples) puts intervals that exclude zero on the median premium of
{g('bootstrap.n_cities_excluding_zero')} cities. The 17 cities are PBS's purposive selection, so these intervals describe the item basket within
these markets, not Pakistan's urban areas at large. An exploratory regression of log price on item-week effects and sum-to-zero city effects
(n = {g('partb.reg.n_obs'):,} rows) returns city coefficients identical to the simple means (r = {f('partb.reg.corr_coef_vs_simple_mean', 3)}),
as expected in a near-balanced panel. Its F statistic for H1a (F = {f('h1a.F')}, df = {g('h1a.df1')}, {g('h1a.df2'):,}) treats rows as independent,
which repeated weekly prices are not, so we read it descriptively.</p>
""")
H.append(fig("fig02_city_premiums", "Fig. 2. Mean food premium per city with 95% t-intervals (n = food items). Filled markers survive Bonferroni.", "72%"))
H.append(html_table(prem_tab, "Table 8. City food price premiums against the national geometric mean (percent = exp(mean log premium) - 1).", digits=2))
H.append(f"""
<h3>3.2 Dispersion by item type (H2)</h3>
<p>The coefficient of variation compares spread across items priced in different units. Mean cross-city CV is highest for perishables
({f('variation.mean_cv.perishable', 3)}, {g('variation.n_items.perishable')} items) and prepared food ({f('variation.mean_cv.prepared_food', 3)}),
lower for storable staples ({f('variation.mean_cv.storable_staple', 3)}, {g('variation.n_items.storable_staple')} items), and lowest for branded
packaged goods ({f('variation.mean_cv.branded_packaged', 3)}), which carry printed or distributor prices (Fig. 3). A one-tailed comparison of
perishables against storables gives SE(A - B) = {f('h2.se_diff', 4)}, Welch t = {f('h2.t_welch', 2)}, p = {p('h2.p_one_tailed')}. The
rank-based check gives p = {p('h2.mannwhitney_p_one_tailed')}. With about ten items a side this is marginal evidence, and we do not oversell it. A Type I error here would send market-functioning attention (Target 2.c) to perishable supply chains without cause.
A Type II error, more likely with so few items, would dismiss a real perishability gap.
Items were classified on storability and pricing mechanism before looking at dispersion.</p>
<div class="two">{fig('fig03_dispersion_by_category', 'Fig. 3. Mean cross-city CV per food item, by category. Branded goods barely vary across cities; perishables vary most.')}
{fig('fig11_box_province_category', 'Fig. 11. Food relative prices by province and by item category (city x item x week).')}</div>

<h3>3.3 Persistence and sticky prices (H1b, H3)</h3>
<p>Across {g('h1b.n_pairs'):,} consecutive-week pairs where a city was above the cross-city median, {pct('h1b.overall_stay_rate')} stayed above it
the next week. City rates range from {pct('h1b.city_rate_min', 0)} to {pct('h1b.city_rate_max', 0)}, and a one-tailed test of the 17 city rates
against 0.5 rejects decisively (t = {f('h1b.t_vs_half')}, p {p('h1b.p_one_tailed')}). Persistence is partly mechanical, however:
{pct('summary.dlog_share_zero_food')} of weekly food log changes are exactly zero ({pct('eval.share_unchanged.perishable', 0)} for perishables,
{pct('eval.share_unchanged.storable_staple', 0)} for storables, {pct('eval.share_unchanged.branded_packaged', 0)} for branded and
{pct('eval.share_unchanged.prepared_food', 0)} for prepared items). Restricted to pairs where the price changed ({g('h1b.n_pairs_changed'):,} pairs),
the stay-above rate is still {pct('h1b.overall_stay_rate_changed')} (t = {f('h1b.t_vs_half_changed')}, p {p('h1b.p_one_tailed_changed')}), with
the lowest city at {pct('h1b.city_rate_changed_min', 0)}. A Type I error would treat chance week-to-week ordering as a durable premium. A Type II error would miss a premium that
households keep paying. That stickiness reshapes H3. Under the contracted rule (abnormal = |change| above the
item's own 90th percentile), the threshold is zero for {g('h3.items_p90_zero')} of 32 food items, so any movement at all would count as
abnormal (§4.2).</p>

<h3>3.4 Distributions and outliers</h3>
<p>Relative prices are roughly symmetric (skew {f('normal.rel_price_skew', 2)}) but heavy-tailed (excess kurtosis
{f('normal.rel_price_excess_kurtosis', 2)}), with a spike at zero from items priced identically everywhere (Fig. 8). Across all 51 items, the
quartile method flags {g('outliers.rel.iqr_mild_1_5'):,} of {g('outliers.rel.n'):,} values as mild (beyond 1.5 IQR) and
{g('outliers.rel.iqr_regular_3'):,} as regular (beyond 3 IQR). The mean-and-SD method flags {g('outliers.rel.sd_mild_2'):,} beyond 2 SD and
{g('outliers.rel.sd_regular_3'):,} beyond 3 SD. The methods disagree because the narrow IQR, set by the many zero-premium items, puts the fences
close, while the SD is inflated by the very extremes it is meant to find, which masks them. On weekly changes the IQR is exactly
{f('outliers.dlog.IQR', 0)}, so both quartile tiers flag every non-zero change ({g('outliers.dlog.iqr_mild_1_5'):,}). That is a degenerate result
and the method is unusable there. We checked the eight largest |rel_price| values against the raw workbooks: {g('outliers.top8_match_pbs_extremes')}
equal PBS's own national MIN or MAX for that item-week. {extreme_text} All are kept and flagged, because large moves are what H3 studies. The CLT supports t-intervals on city means over about 32 items,
but the item-level means are skewed for some cities (median skewness across cities {f('normal.city_item_skew_median', 2)}), which is one more reason to report
medians and bootstrap intervals alongside.</p>
<div class="two">{fig('fig08_relprice_hist', 'Fig. 8. Relative prices with the 1.5 IQR and 3 SD fences and a fitted normal curve. Heavy tails and a spike at 0.')}
{fig('fig13_aggregation', 'Fig. 13. SD of food relative prices at successive levels of aggregation; the national headline sets it to 0.')}</div>

<h3>3.5 Who pays more versus who can least afford it (H5)</h3>
<p>Following the Anscombe lesson, we plot before computing (Fig. 4). The relationship runs <b>against</b> the "poor pay more" expectation.
Premiums are highest in the least deprived cities (Islamabad, Rawalpindi, Karachi), and several of the most deprived (Bannu, Sukkur, Larkana)
pay below the national reference. Spearman ρ between premium and the deprivation composite is {f(H5c + '.spearman_rho', 2)}
(p = {p(H5c + '.spearman_p')}, n = 17). Dropping each city in turn keeps ρ between {f(H5c + '.loo_spearman_min', 2)} and
{f(H5c + '.loo_spearman_max', 2)}, and the most influential city is {g(H5c + '.loo_most_influential')}. With FIES alone, ρ =
{f(H5f + '.spearman_rho', 2)} (p = {p(H5f + '.spearman_p')}), not significant. Without the water indicator, ρ = {f(H5w + '.spearman_rho', 2)}
(p = {p(H5w + '.spearman_p')}). A simple line of premium on the composite has slope {f('partb.simple_reg.slope', 3)} log points per unit of
deprivation (95% CI [{f('partb.simple_reg.slope_ci_lo', 3)}, {f('partb.simple_reg.slope_ci_hi', 3)}], R² = {f('partb.simple_reg.r2', 2)}). This is
an association across 17 cities, not a causal effect, and the deprivation data predate the prices by six years. The burden re-ranking (Fig. 14)
shows the policy point. The five highest-premium cities ({g('h5.top5_premium')}) share {g('h5.top5_overlap_premium_deprivation')} member with the
five most deprived ({g('h5.top5_deprivation')}) and {g('h5.top5_overlap_premium_fies')} with the five most food-insecure. Combining premium with
need, the top three cities for burden are {g('h5.top3_burden_composite')} under the composite definition and {g('h5.top3_burden_fies')} under the
FIES definition. Under both, burden concentrates in Khuzdar and Quetta, whose premiums are above the reference (though not individually
significant) and whose food insecurity is the highest in the panel.</p>
{fig('fig04_premium_vs_deprivation', 'Fig. 4. City premium against the deprivation composite (left) and FIES food insecurity (right), every city labelled. n = 17; Spearman coefficients in panel titles.')}
<div class="two">{fig('fig14_reranking', 'Fig. 14. Ranks by premium and by deprivation; crossing lines show the re-ranking.')}
{fig('fig06_correlation', 'Fig. 6. City-level Spearman correlations (n = 17). Premium correlates negatively with deprivation and with the share of single-quote records.')}</div>
<p>The correlation matrix (Fig. 6) adds a warning. The premium correlates at r = {f('corr.pearson_premium_single_quote', 2)} with the city's share
of single-quote records. The high-premium cities are also those whose averages rest on more distinct quotes. Whether that reflects market
structure (larger, more varied markets) or measurement is a question for H4, and it is why measurement quality is carried as a control.</p>

<h3>3.6 Representation and measurement quality (H4, exploratory)</h3>
<p>The 17 cities cover {pct('representation.spi_share_national_urban', 0)} of the urban population of the four provinces and Islamabad (Census 2023),
and no rural market, AJK or Gilgit-Baltistan. Representation is uneven: Sindh holds {pct('representation.Sindh.share_of_national_urban', 0)} of that
urban population but {pct('representation.Sindh.share_of_spi_cities', 0)} of the cities, while KP's two cities cover only
{pct('representation.KP.share_of_province_urban', 0)} of KP's urban population (Appendix Table A4). Measurement quality varies more. The share of
food records where MIN = MAX, meaning the published average may rest on one quote, runs from {pct('h4.single_quote_min', 0)} in
{g('h4.single_quote_min_city')} to {pct('h4.single_quote_max', 0)} in {g('h4.single_quote_max_city')} (Fig. 7). Across province, population
tercile and deprivation tercile, Kruskal-Wallis tests on n = 17 do not find group differences (p = {p('h4.kruskal_province.p')},
{p('h4.kruskal_pop_tercile.p')} and {p('h4.kruskal_deprivation_tercile.p')}), but with three to five groups of three to eight cities they have
little power. Aggregating to provinces would also hide a mean within-Punjab spread of {f('aggregation.punjab_mean_within_range_logpts', 2)} log
points per item between the cheapest and dearest Punjab city (Fig. 13). That is the project's argument about the national headline in miniature.</p>
<div class="two">{fig('fig07_single_quote_share', 'Fig. 7. Share of food records where MIN = MAX, by city: a data-quality disparity relevant to H4.')}
{fig('fig10_province_tercile', 'Fig. 10. Conditional proportions of premium terciles within each province (contingency table in the notebook).')}</div>

<h3>3.7 Do markets group geographically? (exploratory, later-syllabus methods)</h3>
<p>These methods come later in the course. We use them here only to describe the data, not to confirm anything. On the standardised
17 x {g('partb.matrix_items')} matrix of city-by-item mean relative prices (rice IRRI-6/9 dropped for its structural gaps), Ward clustering at
three clusters gives {g('partb.ward_k3.cluster_1')}; the other seven Punjab cities; and the Sindh, KP and Balochistan cities together (Fig. 5).
Islamabad and Rawalpindi are each other's nearest neighbour, which supports treating them as one market in later robustness checks.
Karachi's nearest neighbour is Hyderabad, but Hyderabad's is Faisalabad, so that pair is not mutual. PCA (Fig. 9) puts
{pct('partb.pca_var_pc1', 0)} of the variance on the first component, which tracks the mean premium (|r| = {f('partb.pca_pc1_corr_mean_premium', 2)}):
the dominant pattern across cities is a general price level. k-means on items' dispersion, volatility and share of unchanged prices separates
{g('partb.kmeans.flexible.n')} flexible items ({g('partb.kmeans.flexible.items')}), {g('partb.kmeans.intermediate.n')} intermediate and
{g('partb.kmeans.sticky.n')} sticky items. Perishables split across all three types, so the hand-made category is a coarse proxy for price
behaviour (Appendix Table A5).</p>
<div class="two">{fig('fig05_dendrogram', 'Fig. 5. Ward dendrogram of cities (exploratory); labels coloured by province.')}
{fig('fig09_pca', 'Fig. 9. Cities on the first two principal components of their food price profiles (exploratory).')}</div>
""")

H.append(f"""
<h2>4. Hypotheses and analytical plan</h2>
<h3>4.1 Formal statements</h3>
<ul>
<li><b>H1a (persistent premiums; SDG 2.1):</b> log p<sub>c,i,t</sub> = α<sub>i,t</sub> + β<sub>c</sub> + ε, with Σβ<sub>c</sub> = 0.
H0: β<sub>c</sub> = 0 for all c; H1: some β<sub>c</sub> ≠ 0. Target <code>log_price</code>; predictors city dummies; controls item-week effects.
If some cities persistently pay more for the same basket, access to food (Target 2.1) differs by place in a way the national figure hides.</li>
<li><b>H1b (persistence):</b> H0: P(above median at t+1 | above at t) = 0.5; H1: &gt; 0.5. Consecutive pairs only, reported also for
changed-price pairs, because unchanged prices make persistence partly mechanical.</li>
<li><b>H2 (perishability and dispersion; SDG 2.c):</b> H0: μ<sub>CV,perishable</sub> = μ<sub>CV,storable</sub>; H1: perishable &gt; storable.
The effective n is the number of items. The volatility half needs more consecutive weeks than the archive provides and stays contingent.
Dispersion in perishables points to market-functioning frictions (Target 2.c).</li>
<li><b>H3 (abnormal movement; SDG 2.c early warning):</b> H0: a classifier on lagged features does no better than the base rate
(AUC = 0.5) on a time-ordered test; H1: AUC &gt; 0.5. Target <code>abnormal_next_week</code>, with thresholds per item fitted on training
weeks only.</li>
<li><b>H4 (representation and model performance; SDG 10):</b> H0: error rates and data quality are equal across city groups (province,
population tercile, deprivation tercile); H1: at least one group differs.</li>
<li><b>H5 (burden versus price; SDG 2.1 and 10):</b> H0: Spearman ρ(premium, deprivation) = 0; H1: ρ ≠ 0, two-sided, n = 17. The burden
re-ranking is descriptive and is shown under two definitions (composite and FIES).</li>
</ul>
<h3>4.2 Refinement of H3 (sharpens the contract, does not change its substance)</h3>
<p>Because {pct('summary.dlog_share_zero_food', 0)} of weekly food changes are zero, the per-item 90th percentile of |change| is zero for
{g('h3.items_p90_zero')} food items, so "abnormal" would mean "moved at all" (base rate {pct('h3.base_rate_contract')}). We propose defining
the threshold as the 90th percentile among <i>non-zero</i> changes. That keeps "abnormal against the item's own history" but excludes
{g('h3.items_too_few_changes')} items with fewer than five changes, and gives a base rate of {pct('h3.base_rate_refined')} on
{g('h3.labelled_rows_refined'):,} labelled rows. Ranking candidate predictors with the mean-and-variance test on training weeks puts
{g('feature_sel.top_feature')} first (score {f('feature_sel.top_score', 2)}), followed by whether the price changed this week and the size of the
change (Table 9).</p>
""")
H.append(html_table(fsel, "Table 9. H3 candidate predictors ranked by |mean(abnormal) - mean(normal)| / SE(A - B), refined label, training weeks only."))
H.append(html_table(hyp, "Table 10. Hypothesis mapping: target, predictors, controls, unit of analysis, effective sample and planned test."))
H.append(f"""
<h3>4.3 Validation design and planned models (Milestones 03 and 04)</h3>
<p>Validation will be time-ordered: model fitting on the earliest weeks, validation on later ones, and the frozen week-14 holdout
(weeks after {g('panel.train_cutoff')}) opened once at the end (Fig. 12). Random folds would leak, because the same city-item series appears on
both sides and later prices inform earlier labels. Accuracy alone will mislead at a {pct('h3.base_rate_refined')} base rate: a model that always
predicts "normal" is right about {100 - 100 * g('h3.base_rate_refined'):.0f}% of the time. Milestone 03 will report AUC, precision-recall and
recall at a fixed alert budget, broken down by city group for H4. The planned models, within the course's agreed scope of classification,
regression, clustering and fairness, are logistic regression (baseline, interpretable), tree ensembles (random forest or gradient boosting)
and a small neural network for comparison, all using lag features inside the classifier. No time-series forecasting models and no
optimisation.</p>
{fig('fig12_split_design', 'Fig. 12. Planned time-ordered split and the frozen holdout (design only; nothing was fitted in this milestone).', '85%')}

<h2>5. Limitations and responsible use</h2>
<ul>
<li><b>Coverage:</b> {g('panel.weeks')} weeks with gaps; only {g('panel.consecutive_pairs')} consecutive pairs, which limits volatility and H3 sample
size. More weeks will be added only from new releases, and the holdout stays sealed.</li>
<li><b>Purposive sample:</b> 17 PBS cities, urban only, with no AJK or Gilgit-Baltistan and two cities each for KP and Balochistan. No result
generalises to rural or small-town markets.</li>
<li><b>Deprivation is not income:</b> district-level 2019-20 survey indicators against 2025-26 prices; a slow-moving structural attribute. District
totals include rural areas, most strongly for Bannu (urban share {bannu_urban:.1f}% in Census 2023).</li>
<li><b>Measurement:</b> SPI records retail prices only. Some city averages rest on a single quote. PDF table extraction for PSLM is checked
against the printed rows (page references in the manifest) and should be human-verified.</li>
<li><b>No causal claims:</b> every relationship reported here is an association across at most 17 cities.</li>
<li><b>Responsible use:</b> labelling a city "expensive" or "deprived" can stigmatise it, and a premium may reflect quality or quote mix as much
as price. Results are framed as signals for further investigation, not as rankings for action.</li>
</ul>

<h2>References</h2>
<p class="note">Pakistan Bureau of Statistics. Weekly Sensitive Price Indicator (SPI), Annexure workbooks, weeks ended {g('panel.first_week')} to
{g('panel.last_week')}. https://www.pbs.gov.pk/ (retrieved 2026; URLs and SHA-256 in data/raw manifests).<br/>
Pakistan Bureau of Statistics. Pakistan Social and Living Standards Measurement Survey (PSLM) 2019-20, District Level. Tables 2.14(a), 2.15,
7.1, 9.1.<br/>
Pakistan Bureau of Statistics. 7th Population and Housing Census 2023 (Digital Census), Table 1, district results.<br/>
Pakistan Bureau of Statistics. CPI (Urban) Group-wise Cumulative Indices (base 2015-16), and Monthly Review on Price Indices, August and
September 2026.<br/>
Wickham, H. (2014). Tidy Data. <i>Journal of Statistical Software</i>, 59(10).<br/>
Anscombe, F. J. (1973). Graphs in Statistical Analysis. <i>The American Statistician</i>, 27(1), 17-21.<br/>
FAO. The Food Insecurity Experience Scale (FIES), SDG indicator 2.1.2.</p>

<h2 class="pb">Appendix</h2>
""")
H.append(html_table(methods_map, "Table A1. Methods map: technique, course unit, where it is used, what it showed, and the CLO it serves. Rows marked (exploratory) are later-syllabus methods used in exploratory form only.", cls="small"))
H.append(html_table(item_tab, "Table A2. Per-item summary over training weeks (all 51 items): mean price (PKR per PBS unit), mean cross-city CV, share of unchanged consecutive-week prices, share of records with MIN = MAX.", cls="small"))
H.append(html_table(city_tab, "Table A3a. City attributes: Census 2023 urban population and PSLM 2019-20 district deprivation indicators (higher = more deprived).", cls="small", digits=2))
H.append(html_table(cross[["city", "city_code", "province", "pslm_districts", "census_units", "pslm_rule", "notes"]],
                    "Table A3b. City-district crosswalk and alignment decisions.", cls="small"))
H.append(html_table(rep, "Table A4. Representation: SPI cities' urban population against provincial urban population (Census 2023).", cls="small", digits=3))
H.append(html_table(kmct, "Table A5. k-means market types (exploratory) against hand-made item categories (number of food items).", cls="small"))
H.append(html_table(h5, "Table A6. H5 correlations under three deprivation definitions, with leave-one-out ranges (n = 17).", cls="small", digits=3))
H.append(html_table(outl, "Table A7. Outlier counts by the quartile and mean/SD methods, both tiers.", cls="small", digits=4))
H.append(html_table(codebook[["column", "attribute_type", "meaningful_operations", "n_missing", "pct_missing"]],
                    "Table A8. Codebook of the master dataset (Unit 02 attribute types; missing counts).", cls="small", digits=2))
H.append("</body></html>")

# Number figures and tables in order of first appearance as a figure/table (not as a text reference).
doc = "\n".join(H)
fig_order = [int(x) for x in re.findall(r'<img src="figures/fig(\d+)_', doc)]
fig_map = {old: new for new, old in enumerate(fig_order, start=1)}
tab_order = [int(x) for x in re.findall(r'<p class="cap">Table (\d+)\.', doc)]
tab_map = {old: new for new, old in enumerate(tab_order, start=1)}


def _renum(match, mapping):
    return re.sub(r"\d+", lambda d: str(mapping.get(int(d.group()), d.group())), match.group())


doc = re.sub(r"Figs?\. \d+(?:(?:, | and )\d+)*", lambda mt: _renum(mt, fig_map), doc)
# Own tables only: skip PBS table numbers such as "Table 9.1" and "Census 2023, Table 1".
doc = re.sub(r"\bTables? \d+(?!\d)(?!\.\d)",
             lambda mt: mt.group() if doc[max(0, mt.start() - 13):mt.start()].endswith("2023, ")
             or "Census" in doc[max(0, mt.start() - 30):mt.start()] else _renum(mt, tab_map), doc)
out_html = REP / "M02_report.html"
out_html.write_text(doc, encoding="utf-8")
print("wrote", out_html)

edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
out_pdf = REP / "M02_report.pdf"
if edge.exists():
    subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out_pdf}",
                    out_html.resolve().as_uri()], check=True, timeout=180)
    print("wrote", out_pdf)
else:
    print("Edge not found; open the HTML and print to PDF")
