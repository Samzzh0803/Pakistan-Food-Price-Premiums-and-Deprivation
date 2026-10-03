# Data Feasibility Report

## Executive Decision
**PBS-ONLY FALLBACK.** The PBS x AMIS project is a NO-GO on the evidence collected: AMIS historical observations were not reproducibly queryable and therefore there are zero verified matched rows. PBS alone is a YELLOW fallback pending historical weekly retrieval.

## 1. What Data Actually Exists
PBS provides an official weekly Annexure workbook with city-item prices, units, and MIN/AVG/MAX fields. AMIS provided only transport/static-page evidence, not a verified date-filtered price dataset.

## 2. PBS Findings
**Verified:** one weekly release, week ended 2026-08-27, was downloaded and opened. Appendix-A is 173 x 25 and exposes the required city-item price semantics. The current official page states 51 SPI items, 50 markets, and 17 cities. **Unknown:** earliest historical weekly date and reliable multi-year archive enumeration.

## 3. AMIS Findings
**Verified:** HTTPS failed TLS negotiation; one plain-HTTP host returned a static page, and another returned a proxy notice. **Unknown:** reproducible query mechanism, historical depth, fields, and daily records. No requests were escalated to bypass protections.

## 4. Commodity Match Findings
Tomato, potato, and onion were retained as candidate rows in `commodity_mapping.csv`, but all are **UNKNOWN**, not DIRECT: AMIS commodity names, grades, and units could not be verified. Verified DIRECT matches: 0.

## 5. Geographic Match Findings
Lahore, Faisalabad, Rawalpindi, Gujranwala, and Multan are candidate PBS locations, but AMIS markets could not be confirmed. Verified overlap: 0.

## 6. Temporal Alignment Findings
No strategy can be empirically evaluated without AMIS daily observations. Same-day/nearest-earlier, trailing 7-day median, and trailing 7-day mean all have 0 matched rows and 100% missingness in the alignment summary. Do not select a strategy until AMIS access is restored.

## 7. Sample Panel
- exact row count: **0 matched rows**
- date range: none; PBS evidence date is 2026-08-27 only
- cities: none verified
- commodities: none verified
- variables: schema created as requested; AMIS-derived and spread/change values are absent

## 8. Data Quality
The raw evidence is hashed in `data/raw/source_manifest.csv`. PBS has explicit units and report-level blanks/`N.A.` markers; schema requires multi-row-header parsing. AMIS quality cannot be profiled. Matching rate is 0% because no AMIS records were collected.

## 9. Estimated Full Dataset
Observed-based estimate is **0 matched rows** for 2, 3, or 5 years. This is not a dimension multiplication estimate; it reflects the absence of verified AMIS records. A future estimate must be recomputed after a reproducible AMIS historical query exists.

## 10. Risks
- **HIGH:** AMIS HTTPS/data access and historical reproducibility.
- **HIGH:** insufficient verified PBS historical weekly depth.
- **MEDIUM:** PBS multi-row headers, city spelling, missing markers, and possible unobserved cross-year drift.

## 11. Recommended Scope
Use PBS only until AMIS access is independently demonstrated: the 51-item SPI basket, all 17 PBS cities, and only the historical weekly range that passes parser tests. Do not freeze a multi-year range yet.

## 12. Recommended Primary Research Question
Can weekly retail price changes and cross-city dispersion in PBS SPI items identify abnormal food-price increases across Pakistani cities?

## 13. Recommended Initial Hypotheses
PBS-only hypotheses may use lagged item changes, rolling volatility, and city/commodity relative gaps. Wholesale-to-retail transmission is not supported by this audit.

## 14. Week-14 Extension Feasibility
If PBS continues publishing weekly releases, freeze an initial verified range for modelling and hold later weekly Annexure files out of training for the new-data/retraining milestone. Each new file should be downloaded, hashed, schema-validated, normalized, and appended by week end. AMIS cannot yet be assigned an extension workflow.

## 15. PBS-Only Fallback
**YELLOW:** the required panel shape is supported by the observed weekly workbook, but historical depth and archive enumeration remain unverified. Build no model until at least two years of weekly parser-success evidence is collected.

## 16. Final Scorecard
| Dimension | Green/Yellow/Red | Evidence |
|---|---|---|
| PBS accessibility | Green | Official page and weekly workbook retrieved |
| PBS historical depth | Red | Only 2026-08-27 verified |
| PBS schema stability | Yellow | One weekly and one monthly structure show multi-format parsing |
| AMIS accessibility | Red | HTTPS TLS failure; no reproducible query |
| AMIS historical depth | Red | Earliest date unknown |
| Commodity comparability | Red | 0 verified direct matches |
| City overlap | Red | 0 verified overlaps |
| Temporal alignment | Red | 0 matched rows |
| Dataset size | Red | 0 matched rows |
| Missingness | Red | AMIS/matched missingness 100% |
| Reproducibility | Yellow | PBS reproducible; AMIS unavailable |
| Week-14 extensibility | Yellow | PBS process defined, AMIS unknown |
| Overall | Red | PBS x AMIS is not viable on verified evidence |

## 17. Bottom Line
- PBS city-item weekly prices exist and are structurally parseable for 2026-08-27.
- Automatic historical PBS enumeration was not proven beyond that release.
- AMIS historical prices were not reproducibly retrieved.
- No DIRECT commodity or city matches are verified.
- No weekly alignment strategy can be selected empirically.
- The matched panel has exactly 0 real rows.
- Wholesale-to-retail transmission is not defensible with the collected evidence.
- PBS-only retail shock/dispersion analysis is the strongest supported fallback.
- Decision: **PBS-ONLY FALLBACK**, with a YELLOW feasibility score pending PBS history verification.
