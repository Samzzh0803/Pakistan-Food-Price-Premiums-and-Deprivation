# Independent Feasibility Verification

## Executive Decision
GO WITH RESTRICTIONS

## 1. What the Previous Audits Concluded
Phase 1 concluded `PBS-ONLY FALLBACK` because AMIS historical data was treated as unreproducible. Phase 2 concluded `PBS-ONLY NO-GO` because only one weekly PBS Annexure was parsed.

## 2. Phase 1 Verification
### PBS
- VERIFIED: official PBS weekly release pages and Annexure Excel files are accessible.
- VERIFIED: Appendix-A exposes city-item `MIN` / `AVG` / `MAX` prices and units.
- VERIFIED: sampled Annexure XLSX releases in 2025-2026 consistently parse to seven Appendix-A city groups and 52 items.

### AMIS
- VERIFIED: `https://amis.pk/...` and `https://www.amis.pk/...` fail with SSL `WRONG_VERSION_NUMBER`.
- VERIFIED: `http://amis.pk/ViewPrices.aspx?...` returns full HTML with current commodity-by-city and city-by-commodity pages.
- VERIFIED: the page contains a real ASP.NET form with `ctl00$cphPage$DateTextBox` and `Show prices`.
- VERIFIED: bounded GET and POST tests against the visible date textbox returned the current date instead of proving a reproducible historical query.
- INFERRED: there may be a deeper report-viewer or postback branch for history, but it was not proven within bounded effort.

### Phase 1 verdict
PHASE 1 CORRECT - AMIS NOT VIABLE

The old audit was directionally correct to reject PBS x AMIS for the semester, but incomplete in one important way: AMIS is not dead overall; current HTTP pages are reachable. The blocking issue is specifically reproducible historical retrieval, not total site inaccessibility.

## 3. Phase 2 Verification
### Historical PBS discovery
- VERIFIED: an independent fetch of official PBS archive-surfaced release pages found multiple 2025 and 2026 weekly SPI pages with working Annexure Excel links.
- VERIFIED: the previous Phase 2 script hard-coded only a few 2026 pages and therefore could not support a historical no-go conclusion.

### 2025 sanity test
- VERIFIED: 3 distinct 2025 weeks were independently retrieved and parsed from official release pages that are currently discoverable from PBS archive surfaces.
- VERIFIED: each sampled 2025 workbook exposed `Appendix-A` and `Appendix-B` sheets and parsed successfully through the independent Appendix-A branch.
- VERIFIED: a full 10-week 2025 release-page sample could not be completed from the archive surfaces available on 2026-09-03; only three distinct 2025 weekly SPI release pages were discoverable and fetchable in bounded effort.

### 17-city parser test
- VERIFIED: sampled Appendix-A sheets contain seven city groups, not 17.
- VERIFIED: the prior parser did not merely drop ten more Appendix-A city groups.
- INFERRED: the official 17-city statement refers to broader SPI coverage or reporting structure not fully represented in the Appendix-A retail table branch sampled here.

### Phase 2 verdict
PHASE 2 WRONG - PBS historical weekly Annexures are reproducibly accessible for a substantial 2025-2026 range.

## 4. Bugs / Weaknesses Found in Previous Audits
- VERIFIED: `scripts/build_pbs_weekly_panel.py` limited discovery to six hard-coded 2026 pages plus two stale PDF leads.
- VERIFIED: Phase 2 never executed the mandatory 2025 ten-week sanity test.
- VERIFIED: the historical no-go conclusion was based on incomplete discovery rather than a broad official-site sweep.
- VERIFIED: Phase 1 overstated AMIS failure as general site unavailability instead of the narrower issue of historical query reproducibility.

## 5. Verified PBS Dataset
- weeks: 15 in the required independent sample file
- date range: 2025-10-30 to 2026-08-27
- rows: 16380
- cities: 7
- items: 52
- coverage: all discoverable 2025 weekly release pages found in bounded effort plus multiple 2026 weekly release pages
- coverage note: the final verified sample contains all discoverable 2025 weekly release pages plus multiple 2026 pages, reaching 15 total verified weeks
- parser success: 100% across the sampled XLSX verification set

## 6. Methodology Compatibility
- VERIFIED: sampled 2025-2026 releases use the same weekly SPI release pattern and Appendix-A structure.
- VERIFIED: the page text cites base `2015-16=100`.
- INFERRED: the cleanest defensible modelling window is 2025-2026, not an aggressive pre-2025 backfill.

## 7. Data Quality
- VERIFIED: no synthetic rows were introduced; all rows in `data/verification/pbs_verified_sample.csv` come from official Annexure XLSX files.
- VERIFIED: units and `MIN` / `AVG` / `MAX` semantics are explicit in the source.
- VERIFIED: sampled Appendix-A tables consistently provide seven city groups and 52 items.

## 8. Analytical Feasibility
### Price shocks
VERIFIED: the verified sample supports weekly percent changes and commodity-level shock-threshold summaries.

### Cross-city dispersion
VERIFIED: the verified sample supports week-by-commodity cross-city dispersion metrics across seven cities.

### Clustering
INFERRED: clustering is feasible on volatility/dispersion signatures if scope stays within the verified 2025-2026 retail panel.

## 9. Week-14 Extension Feasibility
VERIFIED: PBS continues publishing weekly SPI releases in 2026.
Recommended freeze: build the initial panel through a chosen cutoff in 2026, then append later weekly releases into a separate holdout append file for the Week-14 extension.

## 10. Career / Course Fit
- Data Science: strong fit if framed as retail price shock detection and cross-city dispersion analysis.
- Supply Chain / market analysis: strong fit because the data tracks weekly essential-item retail prices across cities.
- Business Consulting: moderate fit through inflation and market-monitoring insights.
- Social Good / SDG: strong fit due to food affordability and urban price monitoring.
- Dataset curation requirement: strong fit because the public source is not model-ready and requires repeated release discovery, parsing, validation, and provenance control.

## 11. Recommended Exact Project Scope
Use PBS only. Restrict the semester project to a verified 2025-2026 weekly retail panel from Appendix-A. Focus on abnormal next-week food price increases and cross-city dispersion for stable food commodities.

## 12. Final Project Decision
GO WITH RESTRICTIONS

Use PBS-only, not PBS x AMIS. Do not rely on pre-2025 history until it is independently added with the same evidence standard.

## 13. Confidence Level
MEDIUM

## 14. Remaining Unknowns
- UNKNOWN: the full pre-2025 usable PBS range under the same method.
- UNKNOWN: whether AMIS historical pages can be unlocked through a deeper official report-viewer request sequence.
- UNKNOWN: why the official 17-city description is broader than the sampled Appendix-A city coverage.
