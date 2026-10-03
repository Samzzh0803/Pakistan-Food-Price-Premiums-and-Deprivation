# Codex Task — Feasibility Audit for Pakistan Food Price Supply-Chain Project

## Role

Act as a **senior data engineer + data scientist conducting a pre-project feasibility audit**.

Do **not** build the final ML project yet.

Your job is to determine, with evidence, whether we can reliably construct a longitudinal dataset linking:

1. **Pakistan Bureau of Statistics (PBS) weekly Sensitive Price Indicator / item-price data**
2. **Punjab Agriculture Marketing Information Service (AMIS) wholesale / auction prices**

for a semester project on **food-price shock prediction and wholesale-to-retail price transmission**.

The output of this task must let us make a hard decision:

- **GO:** PBS × AMIS is technically feasible and should be the project.
- **GO WITH RESTRICTIONS:** feasible only for a narrower set of commodities/cities/years.
- **PBS-ONLY FALLBACK:** AMIS is too unreliable, but PBS alone supports a strong project.
- **NO-GO:** the data is not sufficiently accessible/comparable.

Do not assume the project idea is correct. Try to falsify it.

---

# 1. Project Context

The course is **Data Science for Social Good**.

The intended career alignment is:

> **Data Science × Supply Chain Analytics × Business Consulting**

The candidate research direction is:

> Can upstream wholesale-market price movements help identify abnormal retail food-price increases across Pakistani cities?

The social-good framing is food affordability / food-security resilience.

The likely primary modelling task later will be **classification**:

> Given only information available up to week `t`, does a city–commodity pair experience an abnormal retail price increase at `t+1`?

Secondary analyses may later include regression and clustering.

### Important: this task is only the DATA FEASIBILITY AUDIT.

Do not train Random Forests, XGBoost, neural networks, ARIMA, Prophet, etc.

Basic descriptive statistics are allowed only to assess the data.

---

# 2. Core Rules

1. **Use official sources for the core data.**
   - PBS: `pbs.gov.pk`
   - AMIS: `amis.pk`
2. Do not use Kaggle or third-party mirrors as substitutes.
3. Never fabricate rows, dates, prices, units, or mappings.
4. Preserve raw downloaded files exactly.
5. Record the source URL and retrieval timestamp for every downloaded artifact.
6. Prefer stable HTTP requests and downloadable Excel/CSV files over browser automation.
7. If AMIS requires form submission / ASP.NET postbacks, inspect and reproduce legitimate requests.
8. Use Playwright only if normal HTTP access is not practical.
9. Do not bypass CAPTCHAs, access controls, rate limits, or anti-bot protections.
10. Rate-limit requests politely and cache everything locally.
11. If something cannot be collected reliably, report that explicitly.
12. Do not silently coerce incompatible commodity definitions or units.
13. All assumptions must be documented.

---

# 3. Start by Inspecting the Repository

Before writing code:

- list the current repository structure;
- identify existing Python environment/configuration;
- preserve existing work;
- do not delete or rewrite unrelated files.

Create new work in clearly named directories if they do not already exist:

```text
data/
  raw/
    pbs/
    amis/
  interim/
  processed/

scripts/
reports/
```

If the repository already has an equivalent structure, use it rather than duplicating folders.

---

# 4. Source A — PBS Feasibility Audit

## Known official starting points

- PBS Price Statistics:
  `https://www.pbs.gov.pk/price-statistics/`
- Recent weekly SPI pages expose both **Annexure PDF** and **Annexure Excel** links.

Do not hard-code only one current file. Discover the actual file URLs from official PBS pages.

## What to test

Inspect at least **4 weekly releases from materially different periods**, where available:

- one recent 2026 release;
- one 2025 release;
- one 2024 release;
- one older release, preferably 2021–2023.

If some years are unavailable, document exactly what you found instead of substituting unsupported data.

For each sampled week:

1. find the official release page;
2. identify every relevant downloadable artifact;
3. download the Annexure Excel where available;
4. also download the Annexure PDF for structural verification;
5. record:
   - release date;
   - week-ended date;
   - URL;
   - file type;
   - file size;
   - workbook sheet names;
   - table names/sections if identifiable;
   - row/column dimensions;
   - header structure;
   - whether city-level item prices are present;
   - whether units are explicitly given;
   - whether min/max/average values are available;
   - number of cities;
   - number of commodities/items.

## PBS schema-drift investigation

Determine whether the historical files are structurally stable.

Specifically inspect changes in:

- workbook sheet names;
- merged cells / multi-row headers;
- table starting rows;
- city naming;
- commodity naming;
- units;
- item definitions;
- number of items;
- base-year changes;
- missing-value markers;
- footnotes;
- city coverage;
- whether price semantics change across years.

Create a normalized sample schema such as:

```text
week_end
city
commodity_raw
commodity_canonical
unit_raw
unit_canonical
price_min
price_max
price_avg
source
source_url
source_file
retrieved_at
```

Only include fields actually supported by the source. If `price_min`, `price_max`, or `price_avg` do not exist at the city-item level, do not invent them.

## PBS history discoverability

Test whether historical weekly releases can be enumerated programmatically.

Investigate:

- archive pages;
- pagination;
- predictable release-page links;
- downloadable file link discovery;
- gaps in the archive.

Report an estimated count of retrievable weekly releases and the earliest reliably accessible date you can verify.

Do not attempt to download five years of data during this audit unless it is necessary. The goal is to prove feasibility with a representative sample.

---

# 5. Source B — AMIS Feasibility Audit

## Official starting point

- `https://www.amis.pk/`

AMIS exposes wholesale/auction price functionality such as prices by city, prices by commodity, daily price change, market reports, and price trends.

## First inspect the site technically

Determine:

- whether data is server-rendered;
- whether GET query parameters are sufficient;
- whether forms use ASP.NET WebForms postbacks;
- whether hidden fields such as `__VIEWSTATE` / `__EVENTVALIDATION` are involved;
- whether historical dates can be selected;
- whether there is a discoverable API/XHR endpoint;
- whether prices can be queried reproducibly without manual clicking;
- whether pagination exists;
- whether requests are stable across multiple pages.

Document the mechanism.

## Test commodities

Prioritize highly comparable fresh food items:

1. Tomato
2. Potato
3. Onion

Then test additional candidates if available:

- wheat / wheat flour;
- rice;
- sugar;
- chicken;
- eggs;
- selected pulses.

Do not assume wholesale and PBS definitions are equivalent. Verify each one.

## Test cities/markets

Start with Punjab locations likely to overlap with PBS urban centres, such as:

- Lahore
- Faisalabad
- Rawalpindi
- Gujranwala
- Multan

Use only locations confirmed by the source.

Try to establish whether at least **5 overlapping cities/markets** can be collected reliably.

## Test dates

For each of the three priority commodities, collect several dates spanning:

- a recent week;
- a date several months earlier;
- a date in a prior year if the AMIS interface supports it.

Determine the actual historical depth available.

## Record AMIS fields

For each accessible record, identify:

- date;
- market/city;
- commodity;
- variety/grade if present;
- unit;
- minimum price;
- maximum price;
- modal/fair/average price if present;
- arrivals/volume if present;
- source URL/request details.

Build a normalized sample schema only from fields that actually exist.

---

# 6. Commodity Compatibility Audit

Create:

`data/processed/commodity_mapping.csv`

with columns like:

```text
pbs_commodity_raw
amis_commodity_raw
canonical_commodity
pbs_unit
amis_unit
conversion_possible
conversion_formula
semantic_match
confidence
notes
```

Classify `semantic_match` as:

- `DIRECT`
- `APPROXIMATE`
- `NOT_COMPARABLE`

Examples of issues to investigate:

- tomato vs tomato variety/grade;
- wheat grain vs wheat flour;
- 20kg flour bag vs wholesale wheat per 100kg;
- rice variety differences;
- fresh chicken vs live chicken;
- retail packaged sugar vs wholesale sugar.

Do not merge `APPROXIMATE` records in the final feasibility panel unless the approximation is explicitly justified.

Our preferred final project should focus on **DIRECT** matches.

---

# 7. Geographic Compatibility Audit

Create:

`data/processed/city_mapping.csv`

with:

```text
pbs_city
amis_market
canonical_city
exact_geographic_match
notes
```

Distinguish between:

- same named city;
- a specific wholesale market within the same city;
- nearby but non-equivalent markets.

Do not pretend a nearby mandi is automatically equivalent to a PBS urban retail market.

---

# 8. Temporal Alignment Test

PBS is weekly; AMIS may be daily.

For DIRECT commodity/city matches, test at least these weekly aggregation strategies:

### Strategy A — same-day / nearest-day
Use AMIS price on the PBS week-ended date, or nearest earlier available trading day.

### Strategy B — trailing 7-day median
Aggregate AMIS daily prices over the seven days ending on the PBS week-ended date.

### Strategy C — trailing 7-day mean
Same period, arithmetic mean.

For each strategy report:

- number of matched city–commodity–week rows;
- missingness;
- days between AMIS observation and PBS week end;
- sensitivity of the wholesale price estimate.

Do not choose the strategy solely because it yields the highest row count. Recommend the one that is conceptually defensible.

---

# 9. Build a SMALL End-to-End Feasibility Panel

Create:

`data/processed/feasibility_sample.csv`

This is not the final dataset.

Aim for a small real sample containing multiple:

- dates;
- cities;
- commodities.

Suggested schema:

```text
week_end
city
commodity
pbs_retail_price
pbs_retail_unit
amis_wholesale_price
amis_wholesale_unit
wholesale_to_retail_spread
wholesale_pct_change
retail_pct_change
source_pbs
source_amis
```

Only compute `wholesale_to_retail_spread` if the prices are on genuinely comparable units and commodity definitions.

If they are not comparable, set it missing and state why.

---

# 10. Data Quality Profile

Create:

`data/processed/data_quality_summary.csv`

For each source and important variable report:

- total rows;
- unique dates;
- unique cities;
- unique commodities;
- missing count;
- missing percentage;
- duplicate count;
- impossible/non-positive prices;
- outlier flags;
- unit variants;
- date range;
- matching rate between PBS and AMIS.

Also report:

- number of DIRECT commodity matches;
- number of overlapping cities;
- number of matched weekly observations;
- estimated final row count under realistic historical coverage.

Do not estimate final row count by blindly multiplying dimensions. Base it on observed coverage and explicitly state the assumptions.

---

# 11. Minimum Viable Dataset Thresholds

Evaluate the project against these thresholds.

## Green — strong GO

Preferably:

- ≥ 5 DIRECT commodity matches;
- ≥ 5 overlapping Punjab cities/markets;
- ≥ 2 years of usable overlapping history;
- PBS historical parse success ≥ 90% on sampled releases;
- AMIS can be collected programmatically and reproducibly;
- ≥ 70% temporal match rate for selected city–commodity pairs;
- units can be normalized without dubious assumptions;
- future data will continue to become available during the semester.

## Yellow — GO WITH RESTRICTIONS

Examples:

- only 3–4 direct commodities;
- only a few cities;
- 1–2 years of history;
- notable but manageable missingness;
- one source requires brittle scraping;
- unit/definition drift forces us to narrow scope.

A Yellow decision is acceptable if the remaining data still creates a meaningful semester dataset.

## Red — fallback / no-go

Examples:

- historical AMIS observations cannot be programmatically retrieved;
- commodity definitions are mostly incomparable;
- historical coverage is too shallow;
- source structure is so unstable that collection is unrealistic for a 3-person semester team;
- final matched panel would be too sparse;
- automation would require bypassing protections.

---

# 12. PBS-Only Fallback Audit

Even if AMIS fails, determine whether PBS alone can support:

> **city × commodity × week retail price shock / price-dispersion analysis**

For the PBS-only fallback, report whether we can construct a stable panel with:

```text
week_end
city
commodity
unit
retail_price
weekly_pct_change
lagged_pct_change
rolling_volatility
city_relative_price_gap
commodity_relative_price_gap
```

Also determine whether PBS alone provides sufficient historical depth and weekly updates for later new-data integration.

Give the PBS-only option its own GREEN / YELLOW / RED score.

---

# 13. Do Not Model Yet

During this audit you may calculate:

- missingness;
- distributions;
- price-change summaries;
- simple correlations;
- counts;
- overlap;
- descriptive plots if useful.

Do **not**:

- optimize a predictive model;
- choose a final algorithm;
- claim predictive performance;
- invent a target threshold without inspecting distributions.

The purpose is to decide whether the data supports the research question.

---

# 14. Required Code

Write reproducible scripts, not one-off notebook cells.

At minimum create:

```text
scripts/audit_pbs.py
scripts/audit_amis.py
scripts/build_feasibility_panel.py
```

If helpful, add modules under `src/`.

Requirements:

- Python 3.11+;
- type hints where reasonable;
- clear functions;
- retry/backoff for network calls;
- polite request rate;
- local caching;
- deterministic output;
- useful logging;
- no secrets;
- no hard-coded local absolute paths.

Add/update `requirements.txt` or `pyproject.toml` with only the dependencies actually used.

---

# 15. Provenance

For every downloaded raw file or collected response, maintain a provenance log:

`data/raw/source_manifest.csv`

with:

```text
source
source_url
retrieved_at_utc
local_path
content_type
file_size_bytes
sha256
notes
```

This is important because the project is explicitly about data curation.

---

# 16. Required Reports

## A. `reports/SOURCE_INVENTORY.md`

For PBS and AMIS:

- what exists;
- how it is accessed;
- historical depth verified;
- update frequency;
- exact sample URLs used;
- file/page structure;
- technical retrieval method.

## B. `reports/SCHEMA_DRIFT.md`

Document changes across sampled dates:

- headers;
- fields;
- sheets;
- units;
- names;
- city coverage;
- commodity coverage;
- formatting;
- parsing implications.

## C. `reports/DATA_FEASIBILITY_REPORT.md`

This is the main decision document.

Use this exact structure:

```markdown
# Data Feasibility Report

## Executive Decision
GO / GO WITH RESTRICTIONS / PBS-ONLY FALLBACK / NO-GO

## 1. What Data Actually Exists

## 2. PBS Findings

## 3. AMIS Findings

## 4. Commodity Match Findings

## 5. Geographic Match Findings

## 6. Temporal Alignment Findings

## 7. Sample Panel
- exact row count
- date range
- cities
- commodities
- variables

## 8. Data Quality
- missingness
- duplicates
- inconsistent units
- schema drift
- anomalous values

## 9. Estimated Full Dataset
State assumptions explicitly.

## 10. Risks
Rank each:
- LOW
- MEDIUM
- HIGH

## 11. Recommended Scope
Give exact:
- commodities
- cities
- date range
- sources

## 12. Recommended Primary Research Question

## 13. Recommended Initial Hypotheses
Only hypotheses supported by variables we can actually collect.

## 14. Week-14 Extension Feasibility
Explain exactly what new observations should become available and how they can be integrated later.

## 15. PBS-Only Fallback

## 16. Final Scorecard

| Dimension | Green/Yellow/Red | Evidence |
|---|---|---|
| PBS accessibility | | |
| PBS historical depth | | |
| PBS schema stability | | |
| AMIS accessibility | | |
| AMIS historical depth | | |
| Commodity comparability | | |
| City overlap | | |
| Temporal alignment | | |
| Dataset size | | |
| Missingness | | |
| Reproducibility | | |
| Week-14 extensibility | | |
| Overall | | |

## 17. Bottom Line
Maximum 10 bullets. Be decisive.
```

## D. `reports/feasibility_scorecard.json`

Machine-readable version of the scorecard.

---

# 17. Evidence Standards

Every major claim in the report must be backed by one of:

- a downloaded source artifact;
- an observed HTML/form/API response;
- a computed statistic;
- an official source URL.

Distinguish clearly between:

- **verified**
- **inferred**
- **unknown**

Never write “likely” when a quick empirical check can answer the question.

---

# 18. Important Questions You MUST Answer

At the end of the audit, I need explicit answers to all of these:

1. Can we automatically retrieve historical PBS weekly item-price data?
2. What is the earliest weekly PBS date you actually verified?
3. Does PBS expose the city × commodity prices we need?
4. How stable is its schema across years?
5. Can AMIS historical prices be programmatically queried?
6. What is the earliest AMIS date you actually verified?
7. Which commodities are DIRECT matches between the two sources?
8. Which Punjab cities/markets overlap?
9. Are the units truly comparable?
10. What aggregation should convert AMIS daily data to PBS weekly data?
11. What percentage of candidate records can actually be matched?
12. How many real rows would a 2-year / 3-year / 5-year panel approximately contain?
13. Is wholesale-to-retail price transmission a defensible analysis with these exact sources?
14. If not, what is the strongest PBS-only research design supported by the data?
15. Will genuinely new observations become available between now and mid-November 2026?
16. What exact data should we freeze for initial modelling, and what should be held for the later new-data/retraining milestone?
17. What are the top three execution risks?
18. Should we GO, narrow the project, fall back to PBS-only, or abandon it?

---

# 19. Final Console Output

After all scripts complete, print a short final summary:

```text
=== DATA FEASIBILITY AUDIT COMPLETE ===
Decision:
PBS status:
AMIS status:
Direct commodity matches:
Overlapping cities:
Verified overlapping date range:
Sample matched rows:
Recommended scope:
Top risk:
Main report: reports/DATA_FEASIBILITY_REPORT.md
```

---

# 20. Execution Order

Follow this order exactly:

1. Inspect repository.
2. Discover and sample PBS.
3. Audit PBS schema drift.
4. Discover and sample AMIS.
5. Audit AMIS retrieval mechanism/history.
6. Build commodity mapping.
7. Build city mapping.
8. Test daily→weekly alignment.
9. Build small matched feasibility panel.
10. Profile quality and coverage.
11. Evaluate PBS-only fallback.
12. Write reports.
13. Run scripts from a clean state to verify reproducibility.
14. Print final decision.

Do not proceed into final modelling after step 14.

The goal is **not to prove the proposed idea works**.

The goal is to produce enough real evidence that we can confidently decide whether it is the best project before committing our semester to it.
