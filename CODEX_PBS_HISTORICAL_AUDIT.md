# Codex Task — Phase 2 PBS Historical Retrieval & Panel Feasibility Audit

## Objective

The first audit correctly rejected PBS × AMIS because AMIS historical data was not reproducibly retrievable. However, it left the PBS-only fallback at YELLOW because only one 2026 weekly Annexure was verified.

This second audit has one purpose:

> **Determine whether we can construct a multi-year PBS city × commodity × week panel from official Pakistan Bureau of Statistics weekly SPI Annexures, and if so, prove it with a reproducible extraction pipeline.**

Do not investigate AMIS again.
Do not train ML models.
Do not write the semester proposal yet.

The decision at the end must be one of:

- **PBS-ONLY GO**
- **PBS-ONLY GO WITH RESTRICTIONS**
- **PBS-ONLY NO-GO**

---

# Important Evidence Already Known

The following official PBS historical sources have been independently found and should be treated as starting leads, not assumptions:

1. Weekly SPI archive/list page with releases including January 2023 and December 2022:
   `https://www.pbs.gov.pk/cpi-press-release-june-14/`

2. Official 2023 weekly Annexure:
   `https://www.pbs.gov.pk/sites/default/files/price_statistics/weekly_spi/SPI_Annex%26USCP_20072023.pdf`

   This source contains Appendix-A citywise prices and explicitly covers the week ended 20-07-2023.

3. Official 2022 weekly Annexure:
   `https://www.pbs.gov.pk/sites/default/files/price_statistics/weekly_spi/spi_annex_31032022.pdf`

4. Search-discoverable older official-path patterns exist under:
   `https://www.pbs.gov.pk/sites/default/files/price_statistics/weekly_spi/`

Do not assume every old link still returns successfully. Verify all URLs yourself and record status codes.

---

# 1. Preserve Existing Audit Work

Inspect the repository.

Do not delete or overwrite the first feasibility audit.

Add Phase 2 outputs cleanly, preferably under:

```text
data/raw/pbs_history/
data/interim/pbs_history/
data/processed/
reports/
scripts/
```

Reuse existing utilities where appropriate.

---

# 2. Historical Discovery — Use Multiple Official-Site Strategies

The previous audit relied too heavily on the current PBS price page.

Use multiple discovery methods.

## A. PBS archive/release pages

Inspect:

- current Price Statistics page;
- PBS blog/news archive pages;
- historical Weekly SPI release pages;
- the known January 2023 / December 2022 listing page.

Follow Annexure links.

## B. PBS file directories / historical URL patterns

Investigate official file paths such as:

```text
/sites/default/files/price_statistics/weekly_spi/
/wp-content/uploads/
```

Infer filename patterns only to generate candidates; verify each candidate with an actual HTTP response before treating it as data.

Potential historical naming patterns may vary, for example:

```text
spi_annex_DDMMYYYY.pdf
SPI_Annex...
SPI Annex DD.MM.YYYY.pdf
Annex DD.MM.YY.pdf
```

Do not assume one universal pattern.

## C. PBS sitemap / WordPress metadata

Inspect official PBS sitemaps, archive pages, WordPress endpoints, HTML links, or other publicly exposed metadata if useful.

Do not bypass access controls.

## D. Search engines may be used for discovery only

Search engines may help locate official PBS URLs.

The final dataset must come from `pbs.gov.pk`, not from cached snippets or third-party mirrors.

---

# 3. Required Historical Sample

Before attempting bulk extraction, verify at least:

- 4 distinct weeks from 2026;
- 4 distinct weeks from 2025;
- 4 distinct weeks from 2024;
- 4 distinct weeks from 2023;
- 4 distinct weeks from 2022.

Also attempt to verify at least 2 releases from each of:

- 2021;
- 2020;
- 2019.

Older years are optional after that.

A year counts as **verified** only if an official PBS source file can actually be retrieved and its relevant table inspected.

Create:

`data/processed/pbs_historical_inventory.csv`

with:

```text
week_end
year
source_url
artifact_type
http_status
download_success
city_item_table_present
parser_success
item_count
city_count
notes
```

---

# 4. Identify the Correct Historical Table

We care about **city-level prices for essential items**, not merely the national SPI index.

A source is useful only if it contains a table equivalent to:

> Appendix-A — Consumer Prices of Essential Items

Verify whether each sampled release includes:

- item description;
- unit;
- the 17 city columns / urban centres;
- price values;
- MIN / AVG / MAX if present;
- previous-week / corresponding-week national averages if present.

Document when table semantics change.

---

# 5. Key Base-Year / Basket Change Audit

This is critical.

Determine whether PBS changed:

- SPI base year;
- basket size;
- commodity definitions;
- city coverage;
- units;
- table semantics

during the usable history.

Specifically identify the transition into the current:

> **Base 2015-16, 51-item SPI, 17 cities**

If earlier years use a different basket/base/methodology, do not casually merge them.

Recommend a clean historical cutoff based on comparability.

The preferred dataset is not necessarily the longest dataset.

It is the longest **methodologically defensible** dataset.

---

# 6. Build a Robust Historical Parser

Create/update:

`scripts/build_pbs_weekly_panel.py`

The parser should:

1. accept a manifest of official weekly files;
2. parse XLSX when available;
3. parse native-text PDF tables when Excel is unavailable;
4. use OCR only as a last resort and only if absolutely necessary;
5. detect headers instead of hard-coding a single row number;
6. normalize wrapped city names such as:
   - `Rawal-pindi`
   - `Gujran-wala`
   - `Faisal-abad`
   - `Sar-godha`
   - `Baha-walpur`
   - `Hyder-abad`
   - `Pesha-war`
   - `Khuz-dar`
7. retain raw item names and units;
8. preserve N.A./missing markers correctly;
9. validate expected price fields;
10. fail loudly on unrecognized schemas rather than silently producing bad rows.

Normalized output:

```text
week_end
city
commodity_raw
commodity_canonical
unit_raw
unit_canonical
price
source_url
source_file
parser_version
```

If the weekly table provides MIN/AVG/MAX per city rather than a single price, preserve them as separate variables instead of flattening incorrectly.

---

# 7. Commodity Harmonization

Do not automatically use all 51 items.

Create:

`data/processed/pbs_commodity_history.csv`

For every item observed across verified years:

```text
commodity_raw
canonical_commodity
unit_raw
unit_canonical
first_seen
last_seen
weeks_observed
definition_stable
unit_stable
include_recommended
notes
```

Identify a **stable core food basket** suitable for longitudinal analysis.

Prefer items with:

- consistent definitions;
- consistent units;
- high weekly completeness;
- social relevance;
- meaningful city-level variation.

Candidate foods of interest may include:

- Tomatoes
- Onions
- Potatoes
- Wheat Flour
- Rice Basmati Broken
- Rice IRRI-6/9
- Sugar
- Eggs
- Chicken
- Pulses
- Milk

but include only what the data supports.

---

# 8. City Harmonization

Create:

`data/processed/pbs_city_history.csv`

with:

```text
city_raw
canonical_city
first_seen
last_seen
weeks_observed
coverage_percent
include_recommended
notes
```

Determine whether the 17-city universe is stable across the recommended time range.

---

# 9. Build a REAL Multi-Year Panel

Once the representative sample works, collect enough historical weekly files to build a substantive panel.

### Minimum target

Aim for at least **104 consecutive weekly releases** (≈2 years).

### Preferred target

If reliable, build **January 2022 through August 2026** or the longest clean interval supported by the current 51-item / 17-city methodology.

Do not force a five-year range if schema/methodology changes make it questionable.

Save:

`data/processed/pbs_weekly_panel.csv`

and preferably Parquet as well:

`data/processed/pbs_weekly_panel.parquet`

---

# 10. Continuity Audit

Create a complete expected weekly calendar.

Report:

- expected number of weekly releases;
- discovered official releases;
- downloaded releases;
- successfully parsed releases;
- missing weeks;
- duplicate weeks;
- weeks with schema failures.

Calculate:

```text
discovery_rate
download_success_rate
parser_success_rate
calendar_coverage_rate
```

A high row count is not enough; we need continuity.

---

# 11. Panel Quality Audit

For the recommended date range calculate:

- total rows;
- unique weeks;
- unique cities;
- unique commodities;
- per-item missingness;
- per-city missingness;
- non-positive prices;
- duplicate city-item-week keys;
- abrupt unit changes;
- item-definition changes;
- price outlier flags;
- weeks with unusual row counts.

Create:

`data/processed/pbs_panel_quality.csv`

---

# 12. Test the Actual Analytical Variables

Without training ML, determine whether the panel supports the intended analysis.

Create, for stable items:

```text
weekly_pct_change
lag_1_pct_change
lag_2_pct_change
rolling_4w_volatility
city_relative_price_gap
national_item_median
city_vs_national_gap
```

Do not compute percentage change across missing weeks without handling the gap explicitly.

Report the distributions and missingness.

---

# 13. Price-Shock Target Feasibility — Descriptive Only

We are likely to later define an abnormal retail price increase.

Do NOT choose the final target threshold yet.

Instead calculate candidate empirical thresholds for each commodity:

- 75th percentile weekly increase;
- 90th percentile;
- 95th percentile;
- robust z-score / median-absolute-deviation alternative.

Report how many positive events each definition would generate.

We need to know whether classification is viable and whether the target would be catastrophically imbalanced.

Do not train a classifier.

---

# 14. Cross-City Dispersion Feasibility

For every stable commodity-week calculate:

- city min;
- city median;
- city max;
- max/min ratio where valid;
- interquartile range;
- coefficient of variation;
- each city's deviation from national/cross-city median.

Answer empirically:

> Is there enough meaningful cross-city variation to support a market-dispersion research question?

---

# 15. Week-14 Holdout Design

Today's semester timeline requires a later new-data integration stage.

Recommend an exact freeze.

Preferred logic:

- historical training/EDA panel ends on a clearly specified pre-project cutoff;
- every new PBS weekly release after that cutoff is stored separately;
- Week 14 combines the new observations and evaluates model stability/retraining.

Create:

```text
data/raw/pbs_future_holdout/
```

and document a future collection workflow.

Do not add future weekly files into the initial training panel once the cutoff has been fixed.

---

# 16. Automated Future Collector

Create:

`scripts/collect_new_pbs_week.py`

It should:

- discover the newest official weekly SPI release;
- detect whether it is already in the manifest;
- download the Annexure;
- hash it;
- schema-validate it;
- store it in the future-holdout directory;
- update provenance;
- never modify historical rows silently.

Do not schedule it externally; just make the script runnable.

---

# 17. Required Report

Create:

`reports/PBS_HISTORICAL_FEASIBILITY_REPORT.md`

Use this exact structure:

```markdown
# PBS Historical Feasibility Report

## Executive Decision
PBS-ONLY GO / GO WITH RESTRICTIONS / NO-GO

## 1. Historical Discovery Results

## 2. Verified Date Range

## 3. Base-Year and Methodology Compatibility

## 4. Weekly Annexure Structure

## 5. Schema Drift

## 6. Commodity Stability

## 7. City Stability

## 8. Historical Panel Built
- rows
- weeks
- cities
- items
- date range

## 9. Continuity
- expected weeks
- discovered
- downloaded
- parsed
- missing weeks
- coverage %

## 10. Data Quality

## 11. Price-Change Variable Feasibility

## 12. Price-Shock Classification Feasibility

## 13. Cross-City Dispersion Feasibility

## 14. Recommended Scope
Exact:
- date range
- cities
- commodities
- variables

## 15. Week-14 Extension Plan
Exact freeze date and future-data workflow.

## 16. Curation Contribution
Explain precisely what our team is constructing that PBS does not provide as one model-ready dataset.

## 17. Risks

## 18. Final Scorecard

| Dimension | Green/Yellow/Red | Evidence |
|---|---|---|
| Historical discoverability | | |
| Download reliability | | |
| Parser reliability | | |
| Methodology compatibility | | |
| Commodity stability | | |
| City stability | | |
| Weekly continuity | | |
| Missingness | | |
| Dataset scale | | |
| Shock-target feasibility | | |
| Dispersion-analysis feasibility | | |
| Week-14 extensibility | | |
| Reproducibility | | |
| Overall | | |

## 19. Bottom Line
Maximum 10 bullets.
```

---

# 18. Decision Thresholds

## PBS-ONLY GO

Preferably:

- ≥ 2 years of weekly data;
- ≥ 90% weekly release coverage;
- ≥ 90% parser success;
- current methodology consistent over selected period;
- ≥ 10 stable food commodities, or a smaller basket with strong justification;
- all/most 17 cities stable;
- manageable missingness;
- enough price-shock events for classification;
- new weekly data will continue during the semester.

## GO WITH RESTRICTIONS

Examples:

- 1–2 years only;
- several schema branches;
- 5–9 highly stable food items;
- some cities/items must be dropped;
- 75–90% calendar coverage.

## NO-GO

Examples:

- less than one usable year;
- old Annexures cannot actually be retrieved;
- methodology cannot be harmonized;
- parser succeeds only sporadically;
- city/item data are too incomplete for the intended analysis.

---

# 19. Provenance

Continue using the existing source manifest or create a compatible Phase 2 manifest.

Every artifact must record:

```text
source
source_url
week_end
retrieved_at_utc
local_path
content_type
http_status
file_size_bytes
sha256
parser_status
notes
```

---

# 20. Final Questions That MUST Be Answered

1. What is the earliest official weekly city-item Annexure actually retrieved?
2. What is the latest?
3. What is the longest continuous clean range?
4. How many weekly files were expected, found, downloaded, and parsed?
5. What percentage of weeks are covered?
6. Did the SPI basket/base/methodology change during that range?
7. Which food commodities are stable enough to use?
8. Which cities are stable enough to use?
9. What is the exact final panel size?
10. How much missingness exists?
11. Are weekly price changes usable without excessive gaps?
12. Are there enough abnormal increase events for classification?
13. Is cross-city price dispersion substantial enough to analyze?
14. What exact initial modelling cutoff should be frozen?
15. What exact new-data period should be reserved for Week 14?
16. What curation work can we truthfully claim in Milestone 1?
17. Is this project now GREEN, YELLOW, or RED?
18. Should we commit the semester project to PBS-only?

---

# 21. Final Console Output

Print:

```text
=== PBS HISTORICAL AUDIT COMPLETE ===
Decision:
Earliest verified week:
Latest verified week:
Continuous clean range:
Weekly files expected:
Weekly files parsed:
Coverage:
Stable cities:
Stable food commodities:
Panel rows:
Recommended modelling cutoff:
Future holdout start:
Top risk:
Report: reports/PBS_HISTORICAL_FEASIBILITY_REPORT.md
```

---

# 22. Execution Order

Follow exactly:

1. Inspect existing Phase 1 work.
2. Discover historical PBS releases using multiple official-site strategies.
3. Verify representative releases across years.
4. Audit methodology/base-year compatibility.
5. Implement robust parser branches.
6. Build commodity/city harmonization tables.
7. Retrieve a multi-year weekly corpus.
8. Build the normalized panel.
9. Audit continuity and data quality.
10. Construct descriptive derived variables.
11. Assess shock-target feasibility without ML.
12. Assess cross-city dispersion feasibility.
13. Define a Week-14 holdout workflow.
14. Create the future collector script.
15. Write the report.
16. Re-run from a clean state.
17. Print the final decision.

Do not train any ML model.
