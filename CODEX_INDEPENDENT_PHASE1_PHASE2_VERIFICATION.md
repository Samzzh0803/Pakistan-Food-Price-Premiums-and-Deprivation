# Codex Task — Independent Verification of Phase 1 + Phase 2 Feasibility Audits

## Purpose

You are acting as an **independent reviewer**, not as the original auditor.

Two earlier feasibility audits have already been run:

- **Phase 1:** PBS × AMIS feasibility audit
- **Phase 2:** PBS historical-only feasibility audit

Your job is to **verify whether those audits reached the correct conclusions**, identify any implementation or reasoning errors, independently reproduce the important checks, and recommend whether this semester project should proceed.

Do not assume either previous audit is correct.

The final goal is a hard project decision:

- **GO: PBS × AMIS**
- **GO: PBS-only**
- **GO WITH RESTRICTIONS**
- **NO-GO: choose a different project**

Do not train ML models yet.

---

# 1. Read Everything First

Before doing any work:

1. inspect the repository;
2. locate and read all Phase 1 and Phase 2 reports, scripts, manifests, mappings, CSVs, JSONs, logs, and raw evidence;
3. understand exactly what each previous audit attempted;
4. identify what was actually verified versus what was only inferred.

At minimum, review the equivalent of:

## Phase 1
- `reports/DATA_FEASIBILITY_REPORT.md`
- `reports/SOURCE_INVENTORY.md`
- `reports/SCHEMA_DRIFT.md`
- `reports/feasibility_scorecard.json`
- Phase 1 scripts
- Phase 1 source manifest / raw evidence
- commodity mapping
- city mapping
- AMIS responses/errors

## Phase 2
- `reports/PBS_HISTORICAL_FEASIBILITY_REPORT.md`
- `pbs_historical_inventory.csv`
- `pbs_continuity.json`
- `pbs_commodity_history.csv`
- `pbs_city_history.csv`
- `pbs_panel_quality.csv`
- `pbs_weekly_panel.csv` / `.parquet`
- `pbs_shock_thresholds.csv`
- `pbs_dispersion_summary.csv`
- Phase 2 scripts
- Phase 2 provenance/raw files

Do not modify or delete the previous audit outputs.

---

# 2. Core Review Question

The most important question is:

> **Did the audits fail because the public data is genuinely insufficient, or because the discovery/parsing implementation was incomplete?**

Treat this as an adversarial review.

You must actively search for evidence that contradicts the previous audit conclusions.

---

# 3. Phase 1 Review — PBS × AMIS

The previous Phase 1 conclusion was:

> **PBS-ONLY FALLBACK**

because AMIS historical data could not be reproducibly retrieved.

Independently verify that conclusion.

## PBS side

Check whether the Phase 1 PBS observations were accurate:

- current official PBS weekly SPI Annexure is retrievable;
- city-item level prices exist;
- units exist;
- MIN / AVG / MAX semantics exist;
- number of cities/items is correctly understood;
- parser did not accidentally extract only part of the table.

## AMIS side

Re-evaluate AMIS once, carefully, but do not spend unlimited time.

Use official `amis.pk` only.

Determine whether:

- HTTPS failure is real and reproducible;
- plain HTTP exposes useful price data;
- historical dates can be queried;
- data is server-rendered;
- ASP.NET WebForms postback can be reproduced;
- there is a hidden API / XHR / query endpoint;
- official page parameters can retrieve historical prices;
- price-by-city / price-by-commodity pages are machine-queryable;
- a browser session reveals a reproducible request that `requests` did not.

Allowed:
- normal HTTP requests;
- browser devtools-equivalent inspection;
- Playwright if necessary;
- ASP.NET form POST reproduction.

Not allowed:
- CAPTCHA bypass;
- anti-bot bypass;
- credential/access-control bypass.

### AMIS stop rule

Do not spend more than a reasonable bounded effort.

If historical AMIS cannot be reproducibly queried after inspecting:
- page forms,
- network calls,
- hidden fields,
- GET/POST parameters,
- browser behaviour,

then confirm it as unusable for the semester.

### Required Phase 1 verdict

State one of:

- `PHASE 1 CORRECT — AMIS NOT VIABLE`
- `PHASE 1 PARTIALLY WRONG — AMIS IS VIABLE WITH ...`
- `PHASE 1 WRONG — AMIS HISTORICAL DATA IS REPRODUCIBLY ACCESSIBLE`

If AMIS is viable, prove it with at least:
- 3 commodities;
- 3 cities;
- 3 historical dates;
- real raw responses;
- normalized rows.

---

# 4. Phase 2 Review — PBS Historical Retrieval

The previous Phase 2 conclusion was:

> **PBS-ONLY NO-GO**

because only one weekly release was parsed.

This conclusion must be reviewed much more aggressively.

## Important suspicion

The previous audit reported:
- only 1 parsed week;
- only 7 city groups;
- 357 rows;
- no validated multi-year history.

This may reflect:
- incomplete historical discovery;
- stale direct file URLs;
- parsing only part of an Annexure;
- failing to follow individual weekly PBS release pages;
- treating PDF parsing limitations as data unavailability.

Your job is to distinguish those possibilities.

---

# 5. Independently Discover Historical PBS Releases

Use multiple official-site strategies.

## A. Individual weekly release pages

Search/inspect official PBS release pages for 2025 and 2026.

Do not rely only on one current statistics page.

For each release page:
- confirm date;
- discover Annexure Excel/PDF links;
- record actual downloadable URLs.

## B. Historical archive pages

Search official PBS pages for:
- weekly SPI;
- Sensitive Price Indicator;
- price statistics;
- old press releases;
- archive listings.

## C. Official file paths

Inspect links under:
- `/wp-content/uploads/`
- `/sites/default/files/price_statistics/weekly_spi/`

Use candidate URL inference only for discovery, never as evidence unless HTTP succeeds.

## D. Search-engine discovery

Search engines may be used to locate official `pbs.gov.pk` release pages/files.

Do not use third-party datasets as the source.

---

# 6. Mandatory 2025 Sanity Test

Before making any historical conclusion, perform this simple test.

Retrieve and parse at least **10 distinct 2025 weekly SPI Annexures** from official PBS release pages.

Prefer a spread across the year, for example:
- Jan / Feb
- Mar / Apr
- May / Jun
- Jul / Aug
- Sep / Oct
- Nov / Dec

For each:

- release page URL;
- Annexure URL;
- artifact type;
- HTTP status;
- sheet names;
- row/column dimensions;
- number of cities parsed;
- number of items parsed;
- parser success.

If 10 weeks cannot be found, explain exactly why.

This sanity test is mandatory because a conclusion of "no historical data" is not credible until individual historical release pages have been checked.

---

# 7. Mandatory 17-City Parser Validation

PBS officially describes SPI coverage as 17 urban centres.

Verify whether the underlying Annexure actually contains all 17 cities.

If the parser returns only 7 cities:

1. inspect every sheet / appendix;
2. inspect horizontally split tables;
3. inspect print-page segmentation;
4. inspect merged header ranges;
5. inspect whether Appendix-A continues onto another sheet/page;
6. compare raw workbook visually to parsed output.

Do not accept a 7-city result until you have proven the source itself only contains 7 cities.

Create a table:

```text
week_end
source_city_count
parsed_city_count
missing_cities
reason
```

If the parser was wrong, fix it and document the bug.

---

# 8. Build a Controlled PBS Test Panel

Do not immediately attempt five years.

First prove the pipeline using:

- **10 weeks from 2025**
- **4 weeks from 2026**

If those work, build:

- all retrievable 2025 weeks;
- then all retrievable 2024 weeks;
- then extend backwards if feasible.

For each year report:

```text
expected_weeks
release_pages_found
annexures_found
downloaded
parsed
unique_weeks
coverage_percent
```

---

# 9. XLSX vs PDF

Prefer XLSX where available.

If older years are PDF-only:

- inspect whether PDF contains embedded text tables;
- test `pdfplumber`, `camelot`, `tabula`-style extraction or equivalent;
- use OCR only as a last resort;
- do not mark the data unavailable simply because the XLSX format is missing.

But do not force unreliable PDF extraction into production.

Classify each year:

- `XLSX CLEAN`
- `PDF CLEAN`
- `PDF POSSIBLE BUT BRITTLE`
- `UNUSABLE`

---

# 10. Methodology / Basket Consistency

Independently verify:

- SPI base year;
- number of items;
- city coverage;
- item definitions;
- unit definitions;
- structural methodology.

Identify the longest period that can be defensibly harmonized.

It is acceptable to recommend:
- 2024–2026 only,
- 2025–2026 only,
- or another narrower range,

if that range is methodologically cleaner.

Do not maximize history at the expense of validity.

---

# 11. Check Previous Parser Outputs for Bugs

Audit the existing scripts.

Specifically inspect:

- city-header reconstruction;
- multi-row headers;
- merged cells;
- Appendix-A / Appendix-B handling;
- whether only one portion of a horizontally split table was parsed;
- whether a workbook has multiple relevant sheets;
- item row detection;
- MIN/AVG/MAX column selection;
- unit extraction;
- date parsing;
- duplicate filtering;
- accidental file overwrites;
- stale cache reuse;
- overly narrow URL selectors;
- archive pagination bugs;
- hard-coded current-week assumptions.

Document every material bug found.

---

# 12. Rebuild a Verified PBS Sample Panel

Create a fresh independent output:

`data/verification/pbs_verified_sample.csv`

Minimum goal:

- 10 weekly releases;
- all available cities;
- all available items;
- real values only.

Schema:

```text
week_end
city
commodity_raw
unit_raw
price_min
price_avg
price_max
source_release_page
source_annexure_url
source_file
```

Do not overwrite the old Phase 2 panel.

---

# 13. Compare Old vs New Results

Create:

`reports/AUDIT_COMPARISON.md`

Include:

| Question | Phase 1/2 Result | Independent Verification | Correct? | Explanation |
|---|---|---|---|---|

At minimum compare:

- PBS accessibility
- PBS historical discoverability
- city count
- commodity count
- current Annexure structure
- historical Annexure availability
- parser reliability
- AMIS accessibility
- AMIS historical depth
- PBS-only viability

---

# 14. Project Feasibility Thresholds

## PBS-only GREEN

Preferably:

- ≥ 52 weeks of verified history;
- ideally ≥ 104 weeks;
- ≥ 90% parser success;
- ≥ 90% calendar coverage in selected period;
- ≥ 12 usable cities;
- ≥ 8 stable food commodities;
- manageable missingness;
- reproducible weekly updates;
- enough variation for price-shock classification and cross-city dispersion.

## PBS-only YELLOW

Examples:

- 26–51 usable weeks;
- only 8–11 cities;
- several parser branches required;
- 5–7 stable food items;
- 70–90% calendar coverage.

## PBS-only RED

Examples:

- < 26 usable weeks;
- historical releases genuinely inaccessible;
- parser cannot reliably reconstruct city-item prices;
- methodology changes prevent harmonization.

---

# 15. Descriptive Analytical Feasibility

Do not train ML.

For the independently verified panel, calculate only:

- weekly % price changes;
- item-specific volatility;
- cross-city price dispersion;
- candidate shock-event frequencies;
- missingness;
- city/item coverage.

Determine whether the data has enough temporal and cross-sectional variation for:

1. classification of abnormal next-week price increases;
2. regression on price changes / relative price gaps;
3. clustering cities or commodities by volatility/dispersion patterns.

No final modelling.

---

# 16. Week-14 Extensibility

Verify whether new PBS weekly data continues to be published during Fall 2026.

Recommend an exact data-freeze strategy.

For example:

- initial historical panel cutoff;
- post-cutoff weekly observations stored separately;
- Week 14 evaluation/retraining period.

Do not claim this is feasible unless current weekly publication behaviour supports it.

---

# 17. Required Final Report

Create:

`reports/INDEPENDENT_FEASIBILITY_VERIFICATION.md`

Use:

```markdown
# Independent Feasibility Verification

## Executive Decision
GO PBS × AMIS / GO PBS-ONLY / GO WITH RESTRICTIONS / NO-GO

## 1. What the Previous Audits Concluded

## 2. Phase 1 Verification
### PBS
### AMIS
### Phase 1 verdict

## 3. Phase 2 Verification
### Historical PBS discovery
### 2025 sanity test
### 17-city parser test
### Phase 2 verdict

## 4. Bugs / Weaknesses Found in Previous Audits

## 5. Verified PBS Dataset
- weeks
- date range
- rows
- cities
- items
- coverage
- parser success

## 6. Methodology Compatibility

## 7. Data Quality

## 8. Analytical Feasibility
### Price shocks
### Cross-city dispersion
### Clustering

## 9. Week-14 Extension Feasibility

## 10. Career / Course Fit
Evaluate:
- Data Science
- Supply Chain / market analysis
- Business Consulting
- Social Good / SDG
- Dataset curation requirement

## 11. Recommended Exact Project Scope

## 12. Final Project Decision

## 13. Confidence Level
HIGH / MEDIUM / LOW

## 14. Remaining Unknowns
```

---

# 18. Required Final Decision

Answer these explicitly:

1. Was Phase 1 correct to reject PBS × AMIS?
2. Was Phase 2 correct to reject PBS-only?
3. Did the previous PBS parser miss cities or sheets?
4. Are historical PBS weekly Annexures actually available?
5. How many 2025 weeks were independently retrieved?
6. How many cities are actually present in those Annexures?
7. What is the longest verified usable historical range?
8. Is PBS-only sufficient for a 3-person semester project?
9. Does it satisfy the course's strong data-curation requirement?
10. Is Week-14 new-data integration genuinely possible?
11. Should we commit to PBS-only or choose a completely different project?
12. What exact research question should be used if GO?

---

# 19. Evidence Standard

Every major claim must be one of:

- **VERIFIED** — backed by an official source artifact or computed result
- **INFERRED** — reasonable interpretation, clearly labelled
- **UNKNOWN** — not proven

No unsupported optimism or pessimism.

---

# 20. Final Console Summary

Print:

```text
=== INDEPENDENT FEASIBILITY VERIFICATION COMPLETE ===
Phase 1 verdict:
Phase 2 verdict:
AMIS status:
PBS historical status:
2025 weeks retrieved:
Verified city count:
Verified item count:
Longest verified PBS range:
Verified panel rows:
PBS-only feasibility:
Recommended project decision:
Confidence:
Main report: reports/INDEPENDENT_FEASIBILITY_VERIFICATION.md
```

Do not start ML modelling after this.
