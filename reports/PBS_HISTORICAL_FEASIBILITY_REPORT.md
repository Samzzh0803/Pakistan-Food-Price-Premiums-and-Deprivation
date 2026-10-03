# PBS Historical Feasibility Report

## Executive Decision
**PBS-ONLY NO-GO** for the requested multi-year panel in this run. Actual official files were parsed, but only 1 distinct week in 2026 was retrievable and validated; required 2022-2025 history was not verified. This does not support a multi-year semester dataset.

## 1. Historical Discovery Results
The official archive page and current weekly release pages were queried. One 2026 Annexure XLSX file was retrieved and parsed in the final run. The supplied 2023 and 2022 leads returned HTTP 404, although listed by the official archive. The current PDF lead returned HTTP 200 but the native text table branch was not validated, so it was retained but excluded from the panel.

## 2. Verified Date Range
Earliest and latest parsed weekly city-item observations: **2026-08-27 through 2026-08-27**. Longest clean range: one sampled week; the parsed date is 2026-08-27.

## 3. Base-Year and Methodology Compatibility
Current workbooks identify the 2015-16 SPI presentation and provide 52 items in the sampled Annexure, while the official page describes 51 essential items and 17 cities. Cross-year compatibility cannot be established because older Annexures were not retrievable. The 2022/2023 supplied files are stale 404s.

## 4. Weekly Annexure Structure
Validated XLSX Appendix-A-like tables contain item descriptions, units, city columns, and MIN/AVG/MAX values. The parser detects the description/unit header and city/statistic groups. Seven city groups were present in sampled current workbooks, not 17.

## 5. Schema Drift
XLSX and PDF artifacts use different presentation branches. PDFs were not silently parsed. City labels require normalization (`Rawal-pindi`, `Gujran-wala`, and similar forms). Multi-row headers and missing markers require explicit parsing. Cross-year drift remains unknown.

## 6. Commodity Stability
The sampled panel contains 51 items meeting the local completeness/unit rule. Candidate stable food items include those recorded in `pbs_commodity_history.csv`; this is not a longitudinal stability proof because only one week is available.

## 7. City Stability
7 city groups are complete across the parsed week: Faisalabad, Gujranwala, Islamabad, Lahore, Rawalpindi, Sargodha, Sialkot. This is current-sample stability only, not stability across the requested history.

## 8. Historical Panel Built
- rows: **357**
- weeks: **1**
- cities: **7**
- items: **51**
- date range: **2026-08-27 to 2026-08-27**

## 9. Continuity
- expected weeks: **1** between observed endpoints
- discovered artifacts: **1**
- downloaded artifacts: **1**
- parsed artifacts: **1**
- missing weeks: **0**
- coverage: **100.0% calendar; 100.0% artifact parser success**

## 10. Data Quality
The panel has 0 duplicate keys, 3 non-positive prices, and 0 missing average prices. Derived variables and quality metrics are in `pbs_panel_quality.csv` and `pbs_weekly_panel.csv`.

## 11. Price-Change Variable Feasibility
Weekly changes, lags, rolling four-week volatility, national item medians, and city-relative gaps were computed descriptively. One week provides no usable lagged change history.

## 12. Price-Shock Classification Feasibility
P75/P90/P95 and robust-MAD descriptive thresholds were calculated in `pbs_shock_thresholds.csv`; no final target threshold was selected and no model was trained. Event counts are exploratory only.

## 13. Cross-City Dispersion Feasibility
City min/median/max, max/min ratio, IQR, coefficient of variation, and city deviations were calculated. The seven-city sample supports descriptive variation, but not a multi-year dispersion conclusion.

## 14. Recommended Scope
Exact initial modelling scope is **not approved**. Before modelling, obtain at least 104 parsed consecutive weekly releases under one confirmed methodology. Current observed scope is 2026-08-27, seven city groups, and only locally complete sampled items.

## 15. Week-14 Extension Plan
Freeze the initial panel only after a qualifying historical range is established. Reserve releases after that freeze in `data/raw/pbs_future_holdout/`; the collector downloads the newest official Annexure, hashes it, and never edits historical rows.

## 16. Curation Contribution
The team is constructing a versioned, normalized city-item-week panel with parser validation, city/item harmonization, missingness and continuity audits, derived price-change variables, provenance hashes, and a future holdout workflow. PBS does not provide this model-ready longitudinal table as one artifact.

## 17. Risks
- **HIGH:** official historical Annexure URLs in the archive are stale or unavailable.
- **HIGH:** methodology and basket compatibility before 2026 cannot be verified.
- **MEDIUM:** PDF table extraction remains an unvalidated parser branch.

## 18. Final Scorecard
| Dimension | Green/Yellow/Red | Evidence |
|---|---|---|
| Historical discoverability | Red | Archive leads found but 2022/2023 supplied files return 404 |
| Download reliability | Yellow | Current XLSX files download; historical leads fail |
| Parser reliability | Yellow | 6 XLSX successes; PDF branch intentionally fails loud |
| Methodology compatibility | Red | Cross-year method cannot be verified |
| Commodity stability | Yellow | Sample-local stable items only |
| City stability | Yellow | 7 cities stable in sample; 17-city history unverified |
| Weekly continuity | Red | 100.0% between endpoints; one week |
| Missingness | Yellow | Current parsed XLSX has explicit missing markers |
| Dataset scale | Red | 357 rows over 1 weeks |
| Shock-target feasibility | Yellow | Descriptive thresholds computed; history too short |
| Dispersion-analysis feasibility | Yellow | Seven-city descriptive variation observed |
| Week-14 extensibility | Yellow | Collector exists; initial freeze not justified |
| Reproducibility | Yellow | XLSX pipeline reproducible; archive/PDF limitations preserved |
| Overall | Red | Less than one usable year |

## 19. Bottom Line
- Official PBS weekly Annexures can produce real city-item-week observations.
- One 2026 XLSX release was actually retrieved and parsed in the final clean recovery run.
- The supplied 2022 and 2023 official leads currently return 404.
- Only seven city groups appear in sampled current workbooks.
- No multi-year continuous panel was proven.
- The requested 104-week minimum was not met.
- PDF artifacts were preserved but not silently accepted by an unvalidated parser.
- Derived variables and descriptive shock/dispersion summaries are available.
- Decision: **PBS-ONLY NO-GO** for proceeding to modelling from this audit state.
