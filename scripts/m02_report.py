"""Assemble the Milestone 02 manuscript (HTML -> PDF via headless Edge).

Every inline statistic is read from reports/m02/m02_numbers.json and every table is
rendered from CSVs written by the pipeline under reports/m02/tables. Nothing is hand-typed.
The notebook carries the full analysis; this PDF keeps what the assignment asks for.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REP = ROOT / "reports" / "m02"
TAB = REP / "tables"
PROC = ROOT / "data" / "processed" / "m02"
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


def t(name, **kw):
    return pd.read_csv(TAB / f"{name}.csv", **kw)


def html_table(df: pd.DataFrame, caption: str, cls: str = "", digits: int = 3) -> str:
    body = df.to_html(index=False, border=0, float_format=lambda x: f"{x:,.{digits}f}", escape=True, na_rep="–")
    return f'<div class="tbl {cls}"><p class="cap">{caption}</p>{body}</div>'


def fig(name: str, caption: str, width: str = "100%") -> str:
    return (f'<figure><img src="figures/{name}.png" style="width:{width}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


code = lambda s: f"<code>{s}</code>"
pc = lambda c: f"{g(f'premium.city.{c}.pct_mean'):+.1f}%"
ci = lambda c: f"[{g(f'premium.city.{c}.ci_lo_pct'):.1f}, {g(f'premium.city.{c}.ci_hi_pct'):.1f}]"
H5c, H5f, H5w = "h5.deprivation_composite", "h5.fies_mod_sev_pct", "h5.deprivation_composite_nowater"

# ---------------------------------------------------------------- tables from the pipeline
prem = t("premium")
bonf_cities = ", ".join(prem[prem.bonferroni_sig].city)
codebook = t("codebook")
missing = t("missing_kinds")
missing["rows"] = missing["rows"].map(lambda v: "–" if pd.isna(v) else f"{int(v):,}")
integ = t("integration")
summary = t("summary")
outl = t("outliers")
enc = t("encoding")
item = t("item")
ext = t("extremes")
nm = ext[~ext.equals_pbs_national_min_or_max]
city = pd.read_csv(PROC / "city_table.csv", dtype={"city_code": str})
bannu = city.set_index("city").loc["Bannu"]
ntw = city.set_index("city")["no_tap_water_pct"]
bannu_urban = 100 * bannu.pop_urban / bannu.pop_total
model_ready = pd.read_parquet(PROC / "model_ready_food.parquet")
units = pd.read_parquet(PROC / "master_city_item_week.parquet", columns=["item_id", "unit"]).drop_duplicates("item_id").set_index("item_id")["unit"]
dummy_cols = [c for c in model_ready.columns if c.startswith(("prov_", "cat_"))]

spear = t("spearman", index_col=0)
keep = ["premium", "deprivation_composite", "fies_mod_sev_pct", "log_pop_urban", "single_quote_share"]
short = {"premium": "premium", "deprivation_composite": "deprivation composite", "fies_mod_sev_pct": "FIES food insecurity",
         "log_pop_urban": "log urban population", "single_quote_share": "share MIN = MAX"}
corr_tab = spear.loc[keep, keep].rename(index=short, columns=short).reset_index().rename(columns={"index": "Spearman ρ (n = 17)"})
cont = t("contingency")
cont = cont.rename(columns={"province": "province", "low": "low premium", "mid": "mid premium", "high": "high premium"})

cats = {"perishable": "Perishable", "storable_staple": "Storable staple", "branded_packaged": "Branded/packaged",
        "prepared_food": "Prepared food", "administered_utility": "Administered/utility", "other_nonfood": "Other non-food"}
item_tab = item.assign(category=item.item_category.map(cats), unit=item.item_id.map(units))[
    ["item_id", "item_short", "unit", "category", "mean_price", "mean_cv", "share_unchanged", "share_single_quote"]].rename(
    columns={"item_id": "PBS #", "item_short": "item", "unit": "PBS unit", "mean_price": "mean price (PKR per PBS unit)", "mean_cv": "mean CV",
             "share_unchanged": "share unchanged wk/wk", "share_single_quote": "share MIN = MAX"})
city_tab = city[["city", "province", "pop_urban", "fies_mod_sev_pct", "illiteracy10_pct", "out_of_school_pct",
                 "no_tap_water_pct", "deprivation_composite"]].assign(pop_urban=lambda d: d.pop_urban / 1e6).rename(
    columns={"pop_urban": "district urban pop. (m)", "fies_mod_sev_pct": "FIES mod/sev %", "illiteracy10_pct": "illiterate 10+ %",
             "out_of_school_pct": "out of school 5-16 %", "no_tap_water_pct": "no tap water %",
             "deprivation_composite": "composite (0-1)"})
miss_tab = codebook[["column", "attribute_type", "n_missing", "pct_missing"]].rename(
    columns={"attribute_type": "attribute type", "n_missing": "missing (n)", "pct_missing": "missing (%)"})
miss_tab["missing (n)"] = miss_tab["missing (n)"].map(lambda v: f"{v:,}")

summ = summary.rename(columns={"variable": "variable (unit of analysis; unit)"})
outl_tab = outl[["variable", "n", "Q1", "Q3", "IQR", "mean", "sd", "iqr_mild_1.5", "iqr_regular_3", "sd_mild_2", "sd_regular_3"]].rename(
    columns={"iqr_mild_1.5": "> 1.5 IQR", "iqr_regular_3": "> 3 IQR", "sd_mild_2": "|z| > 2", "sd_regular_3": "|z| > 3"})

hyp = pd.DataFrame([
    ("H1a premiums", "log_price", "city (sum-to-zero dummies of city_code)", "item × week fixed effects (item_id, week_end)",
     "master, food rows", "17 cities; 32 items × 27 weeks"),
    ("H1b persistence", "above_median", "above_median_prev", "price_changed, item_category; rows with is_consecutive",
     "master, food rows", "17 cities (one rate each)"),
    ("H2 dispersion", "cv, averaged per item", "item_category (perishable vs storable_staple)", "n_quoting",
     "item_week_dispersion view", "11 vs 10 items"),
    ("H3 abnormal move", "abnormal_next_week † (to be derived in M03)",
     "dlog_price, dlog_price_lag1, price_changed, rel_price, within_city_range_pct, single_quote, cat_* dummies",
     "prov_* dummies, log_pop_urban, iso_week", "model_ready_food", f"{g('h3.labelled_rows_refined'):,} labelled rows"),
    ("H4 representation", "model error † (M03); single_quote (now)", "province; terciles of pop_urban and deprivation_composite †",
     "item_category", "master / city table", "17 cities in 3-5 groups"),
    ("H5 burden", "city premium = city mean of rel_price †", "deprivation_composite, fies_mod_sev_pct",
     "log_pop_urban, city share of single_quote", "city table (17 rows)", "17"),
], columns=["Hypothesis", "Target (Y)", "Predictors (X)", "Controls / confounders", "Data", "Effective n"])

# ---------------------------------------------------------------- document
css = """
@page { size: A4; margin: 17mm 17mm 17mm 17mm; }
body { font-family: 'Cambria', 'Georgia', serif; font-size: 10.3pt; line-height: 1.36; color: #111; background: #fff; }
h1 { font-size: 15.5pt; margin: 0 0 4px 0; line-height: 1.2; }
h2 { font-size: 12.3pt; margin: 14px 0 5px 0; border-bottom: 1px solid #999; padding-bottom: 2px; }
h3 { font-size: 10.7pt; margin: 10px 0 3px 0; }
p { margin: 3px 0 6px 0; text-align: justify; }
.meta { font-size: 9.5pt; color: #333; margin-bottom: 7px; }
.abstract { border: 1px solid #bbb; padding: 6px 10px; background: #f7f7f7; font-size: 9.5pt; }
figure { margin: 6px 0 9px 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.8pt; color: #222; text-align: left; margin-top: 2px; }
.tbl { margin: 5px 0 9px 0; page-break-inside: avoid; }
.tbl.small table { font-size: 7.3pt; }
.cap { font-size: 8.8pt; font-weight: bold; margin: 0 0 2px 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; }
th { border-bottom: 1.2px solid #333; border-top: 1.2px solid #333; text-align: left; padding: 2px 4px; background: #f2f2f2; }
td { border-bottom: 0.5px solid #ddd; padding: 1.5px 4px; vertical-align: top; }
.two { display: flex; gap: 10px; align-items: flex-start; } .two > * { flex: 1; }
.pb { page-break-before: always; }
ul { margin: 3px 0 6px 18px; padding: 0; } li { margin-bottom: 2px; }
.note { font-size: 8.8pt; color: #333; }
code { font-size: 8.6pt; }
.cols2 { column-count: 2; column-gap: 14px; } .cols2 .tbl { break-inside: avoid; }
"""

H = []
H.append(f"""<!doctype html><html><head><meta charset="utf-8"><title>Milestone 02 Report</title><style>{css}</style></head><body>
<h1>Who Pays More, and Who Can Least Afford It? Food Price Premiums, Deprivation and Abnormal Price Movements Across Selected Pakistani Urban Markets</h1>
<div class="meta"><b>Milestone 02: Data preparation, EDA and hypotheses (draft Sections II and III of the manuscript)</b><br/>
<b>Group 30</b> · Yousuf Uyghur (mu07486), Computer Science · Sameer Hassan (sh09036), Computer Science<br/>
CS/SDP 312/314 L1 Data Science for Social Good, Habib University, Fall 2026 · Dr. Muhammad Usman Arif<br/>
Notebook, code and data: github.com/Samzzh0803/Pakistan-Food-Price-Premiums-and-Deprivation (branch <code>milestone-02</code>)</div>

<div class="abstract"><b>Summary.</b> Pakistan's weekly Sensitive Price Indicator (SPI) is reported as one national figure, but PBS also
publishes prices for 51 items in 17 cities. We built a master dataset of {g('panel.rows'):,} city-item-week rows from {g('panel.weeks')} weekly
releases ({g('panel.first_week')} to {g('panel.last_week')}) and checked it against PBS's own national columns in all
{g('validation.item_weeks_checked'):,} item-weeks. We merged district deprivation (PSLM 2019-20), population (Census 2023) and urban food CPI. Across
32 food items, Islamabad pays {pc('Islamabad')} and Rawalpindi {pc('Rawalpindi')} relative to the national reference, while Sukkur ({pc('Sukkur')}) and
Bannu ({pc('Bannu')}) pay less. Perishables vary most across cities and branded goods least. City premiums are <i>negatively</i>
rank-correlated with a four-indicator district deprivation composite (Spearman ρ = {f(H5c + '.spearman_rho', 2)}, p = {p(H5c + '.spearman_p')}, n = 17),
so the cities paying most are mostly not the ones least able to absorb it. On food insecurity alone the correlation is weaker and not significant
(ρ = {f(H5f + '.spearman_rho', 2)}, p = {p(H5f + '.spearman_p')}), so this pattern is indicative, not established. We state five hypotheses for Milestones 03-04.</div>

<h2>1. Project alignment and master dataset inventory</h2>
<p><b>Team.</b> Both members are computer scientists. One owns acquisition, parsing and provenance; the other owns statistical framing,
representation, ethics and policy interpretation. <b>Research question.</b> Beneath the single national weekly SPI figure, do persistent
city-level food price differences exist, are the cities paying most the ones least able to absorb it, and is the published city detail
enough to detect localised price stress? <b>SDGs:</b> primary SDG 2 (Target 2.1, access to food; Target 2.c, functioning food markets and
market information); secondary SDG 10 (reduced inequalities).</p>
<p><b>Changes since Milestone 01.</b> The 7-city panel reported in Milestone 01 was a parsing error: Appendix-A stacks three tables in one sheet,
and the old script read only the first. The panel covers all 17 SPI cities ({g('panel.cities_by_province.Punjab')} Punjab,
{g('panel.cities_by_province.Sindh')} Sindh, {g('panel.cities_by_province.KP')} KP, {g('panel.cities_by_province.Balochistan')} Balochistan, plus
Islamabad). A correction note is in the repository. Following the instructor's feedback, three external sources are now merged.</p>
""")
inv = pd.DataFrame([
    ("PBS weekly SPI Annexure, Appendix-A", "City-item prices (MIN/AVG/MAX): the panel",
     f"{g('panel.weeks')} weeks, {g('panel.first_week')} to {g('panel.last_week')}", "xlsx from pbs.gov.pk release pages"),
    ("PSLM District Level Survey 2019-20", "District deprivation: FIES food insecurity, literacy, schooling, water",
     "2019-20", "PDF report tables (text extraction)"),
    ("Digital Census 2023, Table 1", "Total and urban population", "2023", "xlsx, one file per province"),
    ("CPI (Urban), food and non-alcoholic beverages", "Deflator only, not a regressor",
     f"monthly, {g('external.cpi_first_month')} to {g('external.cpi_last_month')}", "PDF and Monthly Review docx"),
], columns=["Source", "Role", "Coverage", "Access"])
H.append(html_table(inv, f"Table 1. Master dataset: {g('panel.rows'):,} rows × {g('panel.cols')} columns. "
                    f"Observational unit: 1 row = 1 item's retail price in 1 city in 1 week ({g('panel.cities')} cities × "
                    f"{g('panel.items')} items × {g('panel.weeks')} weeks; {g('panel.food_items')} of the 51 items are food)."))

H.append(f"""
<h2>2. Data ingestion, cleaning and structural readiness</h2>
<h3>2.1 Collection audit</h3>
<p>Each PBS weekly release page (<code>/weekly-sensitive-price-indicator-spi-for-the-week-ended-on-DD-MM-YYYY/</code>) links an Annexure workbook.
A rate-limited script tried the Thursday, Wednesday and Friday page for each of {g('backfill.thursdays_attempted')} weeks from
{g('panel.first_week')} and logged every request, URL and SHA-256 hash. It found release pages for {g('backfill.weeks_found')} weeks, most of them weeks already retrieved for Milestone 01 (the
re-downloads were byte-identical). After dropping {g('files.duplicates')} duplicate files, the panel holds {g('panel.weeks')} weeks. Other weeks may exist under URLs we did not try, so we do not treat the archive as complete. The week after our freeze date
({g('backfill.holdout_weeks')}) was stored, unopened, as a holdout. External files were downloaded from pbs.gov.pk and are kept unchanged with
their URLs and hashes.</p>
<p><b>Validation.</b> The parser asserts, for every week: 3 blocks, 17 cities, 51 items, 867 rows; no duplicate key; MIN ≤ AVG ≤ MAX; and zeros only as
all-zero triples. Against PBS's own national columns, the cross-city minimum and maximum match PBS's national MIN and MAX in
{g('validation.min_max_pass'):,} of {g('validation.item_weeks_checked'):,} item-weeks. The <b>geometric mean</b> of the city averages matches PBS's national
average within 0.1% in {g('validation.gm_pass'):,} of {g('validation.item_weeks_checked'):,}. The arithmetic mean misses by a median of
{f('validation.am_median_abs_rel_err_pct', 2)}%. We therefore use the geometric mean as the reference price.</p>

<h3>2.2 Reshaping and alignment</h3>
<p>The raw sheet ({g('tidy.raw_rows')} rows × {g('tidy.raw_cols')} columns) is not tidy. City names are column headers, each spanning MIN/AVG/MAX
cells, and three tables are stacked in one sheet. We <b>melt</b> each block into one row per city × item, <b>concatenate</b> the blocks and weeks,
and move PBS's national columns into a separate validation table. We then <b>join</b> the item classification (on <code>item_id</code>), the
city attributes (on <code>city</code>) and the CPI (on the month of <code>week_end</code>). Every join is many-to-one, and the row count is
asserted unchanged (Table 2).</p>
<p><b>Key alignment decisions.</b>
<ul>
<li><b>Karachi:</b> one SPI market, but six PSLM districts and seven census districts. PSLM values are population-weighted.</li>
<li><b>Islamabad and Rawalpindi:</b> one contiguous market, kept as two administrative units.</li>
<li><b>District totals:</b> PSLM has no urban sample for Khuzdar and publishes FIES only for district totals, so all indicators use district
totals. These are <b>deprivation measures, not income</b>.</li>
</ul></p>
""")
H.append(html_table(integ, "Table 2. Reshaping and merge steps with keys, cardinality and the asserted row count."))
H.append(f"""
<h3>2.3 Cleaning and missing data</h3>
<p>Nothing is deleted. Zero MIN/AVG/MAX triples ({g('missing.structural_rows')} rows, {f('panel.structural_missing_pct', 2)}%) are set to missing and
flagged: rice IRRI-6/9 is never quoted in Gujranwala, Lahore or Sialkot, and Khuzdar has no gas quote in two weeks. PBS itself excludes these cities
from its national mean. Missing counts and percentages for <b>every column</b> are in Appendix Table A1. Most of the missingness is structural:
weekly change columns are undefined across non-consecutive weeks ({f('missing.lag_missing_pct')}% of rows), because only
{g('panel.consecutive_pairs')} pairs of weeks are 6-8 days apart. Table 3 gives the choice made for each kind of missingness.
<b>Selection bias:</b> we exclude structural non-quotes from price comparisons, so results describe quoting markets and may still
reflect selection bias, because a missing quote does not prove an item is unavailable in that city. The gaps in weekly coverage also mean the
volatility analysis sees only the weeks we could retrieve.</p>
""")
H.append(html_table(missing.drop(columns=["options_considered"]), "Table 3. Kinds of missingness, the choice made for each, and why."))
H.append(f"""
<h3>2.4 Transformations, scaling and encoding</h3>
<p><b>Log transformation.</b> Prices are compared in logs because premiums are multiplicative: a 10% premium means the same thing for salt and for a
20 kg flour bag. Within an item, the median skewness of prices falls from {f('normal.median_within_item_skew_raw', 2)} (raw) to
{f('normal.median_within_item_skew_log', 2)} (log). Prices pooled across items in PKR are not a meaningful distribution (skewness
{f('normal.pooled_raw_price_skew', 2)}), because they mix units. The main variable is
<code>rel_price</code> = log(city price / national geometric mean), which sums to zero across cities in each item-week. District urban population
is strongly right-skewed (skewness {f('summary.pop_urban_skew', 2)}, driven by Karachi). Its log (<code>log_pop_urban</code>) has skewness
{f('summary.log_pop_urban_skew', 2)}. <b>Scaling.</b> The four deprivation indicators are min-max scaled to 0-1 (higher = more deprived) and averaged
into <code>deprivation_composite</code>. Prices are z-scored within each item and week (<code>price_z_item_week</code>). All constants come from
the training weeks only. <b>Encoding.</b> Nominal strings become 0/1 dummy flags with fixed reference levels: Punjab for province, storable
staple for item category (Table 4). The model-ready file <code>data/processed/m02/model_ready_food.parquet</code> ({len(model_ready):,} food rows ×
{model_ready.shape[1]} columns) carries {len(dummy_cols)} dummy columns. Cities will enter H1 with sum-to-zero coding, so each coefficient is a
premium relative to the average city.</p>
""")
H.append(html_table(enc.drop(columns=["item_short"]), "Table 4. Dummy encoding example: province and item_category strings (left) become 0/1 flags (right). Reference levels: Punjab and storable_staple (all zeros)."))

H.append(f"""
<h2>3. Exploratory data analysis</h2>
<h3>3.1 Univariate summary</h3>
<p>Table 5 reports each variable at its own unit of analysis. City attributes take 17 values, so city-level relationships are always analysed with
n = 17, never with the {g('panel.rows'):,} rows. <b>Skewed distributions</b> show up where the resistant and non-resistant measures disagree. The
within-city price range has mean {f('summary.range_pct_mean')}% but median {f('summary.range_pct_median')}% (skewness
{f('summary.range_pct_skew', 2)}). District urban population has mean {f('summary.pop_urban_mean_m', 2)} million but median {f('summary.pop_urban_median_m', 2)} million.
For city premiums, {g('premium.max_mean_median_gap_city')}'s mean ({pc(g('premium.max_mean_median_gap_city'))}) and median
({f('premium.city.' + g('premium.max_mean_median_gap_city') + '.pct_median')}%) differ by {f('premium.max_mean_median_gap_pts')} points, because a few
items drive the mean. We report medians alongside means wherever the two disagree. Weekly price changes are dominated by zeros:
{pct('summary.dlog_share_zero_food', 0)} of consecutive-week food changes are exactly 0, so their median and IQR are both 0.</p>
""")
H.append(html_table(summ, "Table 5. Summary statistics for key numerical variables: central tendency (mean, median), spread (SD, IQR) and skewness."))
H.append(f"""
<h3>3.2 City food price premiums</h3>
<p>For each city we average <code>rel_price</code> over weeks for each food item. The item-level means are the observations, so n is 31-32 items,
not thousands of rows. Islamabad's mean premium is {pc('Islamabad')} (95% t-interval {ci('Islamabad')}), followed by Rawalpindi {pc('Rawalpindi')}
and Karachi {pc('Karachi')}. Sukkur ({pc('Sukkur')}) and Bannu ({pc('Bannu')}) are lowest (Fig. 2). {g('premium.n_sig_05')} of 17 cities differ from
zero at α = 0.05. After a Bonferroni correction for 17 tests, {g('premium.n_sig_bonferroni')} remain: {bonf_cities}. Premiums persist: of
{g('h1b.n_pairs'):,} consecutive-week pairs in which a city was above the cross-city median, {pct('h1b.overall_stay_rate')} stayed above it the next
week. Restricting to pairs where the price actually changed, the figure is {pct('h1b.overall_stay_rate_changed')}.</p>
{fig('fig02_city_premiums', 'Fig. 2. Mean food price premium of each city relative to the national geometric mean, with 95% t-intervals over food items (n = 31-32). Colour shows province; filled markers remain significant after Bonferroni correction (α = 0.05/17). Islamabad and Rawalpindi pay most; Sukkur and Bannu least.', '74%')}

<h3>3.3 Dispersion by food category</h3>
<p>The coefficient of variation (CV) puts items priced in different units on one scale. Mean cross-city CV is highest for perishables
({f('variation.mean_cv.perishable', 3)}, {g('variation.n_items.perishable')} items) and prepared food ({f('variation.mean_cv.prepared_food', 3)}),
lower for storable staples ({f('variation.mean_cv.storable_staple', 3)}, {g('variation.n_items.storable_staple')} items), and lowest for branded
packaged goods ({f('variation.mean_cv.branded_packaged', 3)}), which carry printed or distributor prices (Fig. 3). A one-tailed Welch comparison of
perishables against storables gives p = {p('h2.p_one_tailed')}, and a rank-based check gives p = {p('h2.mannwhitney_p_one_tailed')}. With about ten
items per group this is suggestive, not conclusive. Categories were assigned on storability and pricing mechanism before any dispersion was
computed.</p>
{fig('fig03_dispersion_by_category', 'Fig. 3. Mean cross-city coefficient of variation (unitless) of each food item, grouped by category; one dot per item. Perishables are the most dispersed across cities; branded packaged goods barely differ.', '68%')}

<h3>3.4 Outlier diagnostic</h3>
<p>We apply both rules taught in the course to <code>rel_price</code> and to weekly log changes, each at two tiers (Table 6, Fig. 8).
<b>Masking and variance inflation:</b> the two rules disagree sharply. On <code>rel_price</code>, the IQR is narrow because many branded items are
priced identically everywhere, so the quartile fences sit close and flag {g('outliers.rel.iqr_mild_1_5'):,} values. The SD, by contrast, is inflated
by the extreme values themselves, which pulls the 3-SD fence outward and masks them: only {g('outliers.rel.sd_regular_3'):,} are flagged. On weekly
changes the IQR is exactly 0, so the quartile rule flags every non-zero change and is unusable. We checked the eight most extreme relative prices
against the raw workbooks. {g('outliers.top8_match_pbs_extremes')} equal PBS's own national minimum or maximum for that item-week, and the other is a
genuine printed quote ({", ".join(f"{r.city} {r.item_short.lower()}, PKR {r.price_avg:g} per {units[r.item_id].lower()}" for r in nm.itertuples())}). They are therefore true
prices: we keep and flag them rather than trim them, because large moves are what H3 studies.</p>
""")
H.append(html_table(outl_tab, "Table 6. Outliers by the quartile rule (beyond Q1 − k·IQR or Q3 + k·IQR) and the z-score rule (|z| > 2, 3). Units: log points.", digits=3))
H.append(fig("fig08_relprice_hist", "Fig. 8. Distribution of relative prices (log points; all 51 items × 17 cities × 27 weeks) with the 1.5 IQR fences (dashed) and the ±3 SD fences (dotted), plus a normal curve with the same mean and SD. The spike at 0 and the heavy tails explain why the two rules disagree.", "74%"))

H.append(f"""
<h3>3.5 Relationships: premium, deprivation and data quality</h3>
<p>Following Anscombe's advice, we plot before computing any coefficient (Fig. 4). The relationship runs <b>against</b> the "poor pay more"
expectation. The three least deprived cities (Karachi, Islamabad, Rawalpindi) have the highest premiums, while several of the most deprived (Bannu,
Sukkur, Larkana) pay below the national reference. The Spearman correlation between premium and the deprivation composite is ρ =
{f(H5c + '.spearman_rho', 2)} (p = {p(H5c + '.spearman_p')}, n = 17). Dropping each city in turn keeps ρ between {f(H5c + '.loo_spearman_min', 2)} and
{f(H5c + '.loo_spearman_max', 2)}. Against FIES food insecurity alone, ρ = {f(H5f + '.spearman_rho', 2)} (p = {p(H5f + '.spearman_p')}), which is not
significant. The tap-water indicator is a weak deprivation measure: it rates Gujranwala and Sialkot ({ntw['Sialkot']:.0f}% and {ntw['Gujranwala']:.0f}% of households without tap water) as worse off
than Quetta or Peshawar, and Karachi and Hyderabad as best off. This mostly reflects reliance on motor pumps and filtration plants in Punjab cities. Without it, the composite gives ρ = {f(H5w + '.spearman_rho', 2)} (p = {p(H5w + '.spearman_p')}).
Khuzdar and Quetta are the exception: their premiums are above zero, though not individually significant, and they have the
highest food insecurity in the panel, so burden concentrates there. These are associations across 17 cities, with deprivation data from 2019-20,
and support no causal claim.</p>
{fig('fig04_premium_vs_deprivation', 'Fig. 4. City mean food price premium (%) against the PSLM 2019-20 deprivation composite (left; 0-1, higher = more deprived) and moderate-or-severe food insecurity (right; % of population). Every city is labelled; colour shows province; n = 17.')}
""")
H.append('<div class="two">' + html_table(corr_tab, "Table 7. Spearman correlation matrix of city-level variables (n = 17).", digits=2)
         + html_table(cont, "Table 8. Contingency table: SPI cities by province and premium tercile.", digits=0) + "</div>")
H.append(f"""
<p><b>Data quality.</b> Table 7 also shows that the premium correlates negatively with the share of a city's food records where MIN = MAX
(ρ = {spear.loc['premium', 'single_quote_share']:.2f}). MIN = MAX means every quote
PBS collected for that item in that city was identical: either a single quote or several equal ones. The published average then carries no
information about price spread within the city. This share ranges from {pct('h4.single_quote_min', 0)} in {g('h4.single_quote_min_city')} to
{pct('h4.single_quote_max', 0)} in {g('h4.single_quote_max_city')} (Fig. 7), a measurement-quality disparity that H4 carries forward.
<b>Representation.</b> The census districts that contain the 17 SPI cities hold {pct('representation.spi_share_national_urban', 0)} of the urban
population of the four provinces and Islamabad. Because each city is smaller than its district, this is an upper bound on what the SPI markets
represent. The panel includes no rural markets, AJK or Gilgit-Baltistan, and only two cities each for KP and Balochistan.</p>
{fig('fig07_single_quote_share', 'Fig. 7. Share of each city’s food price records (%) in which PBS’s minimum and maximum quote are equal, coloured by province. High shares mean the published city average carries no information on within-city price spread.', '68%')}
""")

H.append(f"""
<h2>4. Hypotheses and analytical plan</h2>
<p>Each hypothesis is mapped to master-dataset columns in Table 9. Targets marked † are derived from existing columns, and the rule for each is
fixed now. abnormal_next_week has been derived provisionally in the notebook, which is the source of the base rate and row count quoted below,
using thresholds fitted on training weeks only. The final label and the model-error columns will be built in Milestone 03.</p>
<ul>
<li><b>H1, persistent premiums (SDG 2.1).</b> (a) In log p<sub>c,i,t</sub> = α<sub>i,t</sub> + β<sub>c</sub> + ε with Σβ<sub>c</sub> = 0:
H0: β<sub>c</sub> = 0 for every city c; H1: β<sub>c</sub> ≠ 0 for at least one c. (b) H0: P(above median at t+1 | above at t) = 0.5;
H1: &gt; 0.5, tested on consecutive weeks and repeated for changed prices only. If some cities persistently pay more for the same basket, access to
food differs by place in a way the national figure hides.</li>
<li><b>H2, perishability and dispersion (SDG 2.c).</b> H0: μ<sub>CV, perishable</sub> = μ<sub>CV, storable</sub>; H1: μ<sub>CV, perishable</sub>
&gt; μ<sub>CV, storable</sub>. Greater dispersion in perishables points to market-functioning frictions.</li>
<li><b>H3, abnormal next-week movement (SDG 2.c, early warning).</b> H0: a classifier on lagged features achieves AUC = 0.5 on a time-ordered test
set; H1: AUC &gt; 0.5. "Abnormal" means a change larger than the item's own 90th percentile of non-zero changes in the training weeks. We use
non-zero changes because {pct('summary.dlog_share_zero_food', 0)} of weekly food price changes are zero, and the plain 90th percentile would be 0
for {g('h3.items_p90_zero')} of the 32 food items. Base rate: {pct('h3.base_rate_refined')}.</li>
<li><b>H4, representation and model fairness (SDG 10).</b> H0: model error rates and data quality (share of MIN = MAX records) are equal across
city groups (province, population tercile, deprivation tercile); H1: at least one group differs.</li>
<li><b>H5, burden versus price (SDG 2.1, SDG 10).</b> H0: Spearman ρ(city premium, deprivation) = 0; H1: ρ ≠ 0, two-sided, n = 17. EDA already
points to ρ &lt; 0 (§3.5). The test will be repeated under both deprivation definitions.</li>
</ul>
""")
H.append(html_table(hyp, "Table 9. Variable mapping: target, predictors and controls for each hypothesis, with column names from the master dataset or its views."))
H.append(f"""
<p><b>Analytical plan.</b> Within the course's agreed scope (classification, regression, clustering and fairness analysis), Milestone 03 will:
<ul>
<li>estimate H1 with a fixed-effects regression;</li>
<li>fit logistic regression and tree-ensemble classifiers for H3;</li>
<li>validate on a time-ordered split, because random folds would leak later weeks of the same city-item series;</li>
<li>open the frozen holdout once, at the end;</li>
<li>evaluate with AUC and precision-recall rather than accuracy, since a model that always predicts "normal" would be about
{100 - 100 * g('h3.base_rate_refined'):.0f}% accurate;</li>
<li>break all results down by city group for H4.</li>
</ul>
The notebook also contains exploratory clustering, PCA and nearest-neighbour analyses, which are not part of this report.</p>

<h2>5. Limitations</h2>
<ul>
<li><b>Coverage:</b> {g('panel.weeks')} weeks with gaps and only {g('panel.consecutive_pairs')} consecutive pairs, which limits volatility analysis and
the sample size for H3.</li>
<li><b>Sample:</b> 17 cities chosen by PBS, urban only. Results do not generalise to rural or small-town markets.</li>
<li><b>Deprivation:</b> 2019-20 district totals that include rural areas (Bannu's district is {bannu_urban:.1f}% urban). These are measures of
deprivation, not city income.</li>
<li><b>Measurement:</b> SPI records retail prices only. PSLM values were extracted from PDF tables; each is stored with its page number for checking.</li>
</ul>

<h2>References</h2>
<p class="note">Pakistan Bureau of Statistics. Weekly Sensitive Price Indicator, Annexure workbooks, weeks ended {g('panel.first_week')} to
{g('panel.last_week')}. https://www.pbs.gov.pk/<br/>
Pakistan Bureau of Statistics. Pakistan Social and Living Standards Measurement Survey 2019-20, District Level: Tables 2.14(a), 2.15, 7.1, 9.1.<br/>
Pakistan Bureau of Statistics. 7th Population and Housing Census 2023 (Digital Census): Table 1, district results.<br/>
Pakistan Bureau of Statistics. CPI (Urban) Group-wise Cumulative Indices, base 2015-16; Monthly Review on Price Indices, August and September 2026.<br/>
Wickham, H. (2014). Tidy Data. <i>Journal of Statistical Software</i>, 59(10).<br/>
Anscombe, F. J. (1973). Graphs in Statistical Analysis. <i>The American Statistician</i>, 27(1), 17-21.</p>

<h2 class="pb">Appendix</h2>
""")
half = (len(miss_tab) + 1) // 2
H.append('<p class="cap">Table A1. Missing values (count and % of ' + f"{g('panel.rows'):,}" + ' rows) for every column of the master dataset, with Unit 02 attribute type. '
         'Weekly-change columns are missing wherever the previous week is not 6-8 days earlier. Price columns are missing only for the structural non-quotes. price_z_item_week is undefined where every quoting city charges the same price (SD = 0). days_since_prev is missing in the first week.</p>'
         '<div class="two">' + html_table(miss_tab.iloc[:half], "", cls="small", digits=2) + html_table(miss_tab.iloc[half:], "", cls="small", digits=2) + "</div>")
H.append(html_table(city_tab, "Table A2. City attributes: Census 2023 district urban population and PSLM 2019-20 district deprivation indicators (higher = more deprived).", cls="small", digits=2))
H.append(html_table(item_tab, "Table A3. Per-item summary over the 27 training weeks. Mean price = average of the quoting cities’ average prices in each week, then averaged over weeks, in PKR per the PBS unit shown (e.g. eggs per dozen, wheat flour per 20 kg bag). Mean CV = cross-city coefficient of variation, averaged over weeks. Share unchanged = share of consecutive-week price changes that are exactly 0. Share MIN = MAX = share of city-week records where all quotes were equal.", cls="small"))
H.append("</body></html>")

# Number figures and tables in order of appearance.
doc = "\n".join(H)
fig_order = [int(x) for x in re.findall(r'<img src="figures/fig(\d+)_', doc)]
fig_map = {old: new for new, old in enumerate(fig_order, start=1)}
tab_order = [int(x) for x in re.findall(r'<p class="cap">Table (\d+)\.', doc)]
tab_map = {old: new for new, old in enumerate(tab_order, start=1)}


def _renum(match, mapping):
    return re.sub(r"\d+", lambda d: str(mapping.get(int(d.group()), d.group())), match.group())


doc = re.sub(r"Figs?\. \d+(?:(?:, | and )\d+)*", lambda mt: _renum(mt, fig_map), doc)
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
