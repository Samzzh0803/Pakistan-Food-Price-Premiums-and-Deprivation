"""Generate the audit reports from captured evidence and processed outputs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    inventory = """# Source Inventory

## PBS
- **Verified:** `https://www.pbs.gov.pk/price-statistics/` was retrieved on 2026-09-03.
- **Verified:** the page exposed the weekly Annexure Excel `https://www.pbs.gov.pk/wp-content/uploads/2020/07/Annex_27.08.2026.xlsx` and weekly SPI report Excel `https://www.pbs.gov.pk/wp-content/uploads/2020/07/3.-SPI-Report-27.08.2026.xlsx`.
- **Verified:** Annexure-A contains city columns, MIN/AVG/MAX fields, item descriptions, and units; sampled workbook has sheets `Appendix-A` and `Appendix-B`.
- **Verified:** PBS describes SPI as weekly, 51 essential items, 50 markets, and 17 cities on the official page.
- **Unknown:** an older weekly archive was not enumerated from the current page; no weekly PDF link was discovered in the sampled HTML.

## AMIS
- **Verified:** `https://www.amis.pk/` failed TLS negotiation with `WRONG_VERSION_NUMBER`.
- **Verified:** `http://www.amis.pk/` redirected to a proxy notice; `http://amis.pk/` returned a 359459-byte HTML page, cached at `data/raw/amis/response_4.bin`.
- **Unknown:** no reproducible HTTPS data query, historical response, API/XHR endpoint, form post, or date-filtered price record was established. The static HTTP page was preserved but not treated as data.
- **Retrieval method:** ordinary `requests` with a research user-agent, timeout, and local caching; no CAPTCHA, access-control, or anti-bot bypass was attempted.
"""
    (REPORTS / "SOURCE_INVENTORY.md").write_text(inventory, encoding="utf-8")
    drift = """# Schema Drift

## PBS sampled artifacts
| Artifact | Observed structure | Parsing implication |
|---|---|---|
| `sample_2.xlsx`, week ended 2026-08-27 | `Appendix-A`, 173 rows x 25 columns; title rows, city group headers, MIN/AVG/MAX subheaders, item/unit rows | Requires header-row detection and city-group reconstruction |
| `sample_3.xlsx`, week ended 2026-08-27 | `Page 1`, 47 rows x 8 columns; separate SPI report presentation | Not interchangeable with Annexure-A without parser branch |
| `sample_1.xlsx`, August 2026 monthly | `Items 1-51`, `Items 52-66`, `Sheet1`; 1-3 header rows and 25/21 columns | Monthly artifact is not a weekly substitute |

**Verified:** PBS uses merged/multi-row report headers, hyphenated city names such as `Rawal-pindi` and `Gujran-wala`, explicit units such as `20 Kg` and `1 Kg`, and `N.A.`/blank cells. The sampled weekly Annexure includes MIN/AVG/MAX at city-item level.

**Unknown:** cross-year drift in sheets, base year, item definitions, and city coverage. The official page did not expose older weekly files in this run, so no unsupported historical comparison is claimed.

## AMIS
No schema drift can be assessed: no historical AMIS data response was retrieved. The raw HTML transport responses and errors are retained under `data/raw/amis/`.
"""
    (REPORTS / "SCHEMA_DRIFT.md").write_text(drift, encoding="utf-8")
    scorecard = {
        "decision": "PBS-ONLY FALLBACK",
        "dimensions": {
            "PBS accessibility": "GREEN", "PBS historical depth": "RED", "PBS schema stability": "YELLOW",
            "AMIS accessibility": "RED", "AMIS historical depth": "RED", "Commodity comparability": "RED",
            "City overlap": "RED", "Temporal alignment": "RED", "Dataset size": "RED", "Missingness": "RED",
            "Reproducibility": "YELLOW", "Week-14 extensibility": "YELLOW", "Overall": "RED for PBS x AMIS"
        },
        "verified": {"pbs_earliest_week": "2026-08-27", "amis_earliest_date": None, "matched_rows": 0,
                     "direct_commodity_matches": 0, "overlapping_cities": 0, "candidate_alignment_rate_percent": 0.0},
        "fallback": {"decision": "YELLOW", "design": "PBS city x commodity x week retail price, weekly changes, lagged changes, rolling volatility, and relative price gaps", "reason": "One real weekly Annexure was parsed structurally, but historical depth remains unverified."}
    }
    (REPORTS / "feasibility_scorecard.json").write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    report = """# Data Feasibility Report

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
"""
    (REPORTS / "DATA_FEASIBILITY_REPORT.md").write_text(report, encoding="utf-8")
    print("=== DATA FEASIBILITY AUDIT COMPLETE ===")
    print("Decision: PBS-ONLY FALLBACK")
    print("PBS status: GREEN accessibility; RED verified historical depth")
    print("AMIS status: RED - no reproducible historical query")
    print("Direct commodity matches: 0 verified")
    print("Overlapping cities: 0 verified")
    print("Verified overlapping date range: none (PBS verified 2026-08-27 only)")
    print("Sample matched rows: 0")
    print("Recommended scope: PBS-only SPI city x commodity x week retail panel")
    print("Top risk: AMIS historical access and reproducibility")
    print("Main report: reports/DATA_FEASIBILITY_REPORT.md")


if __name__ == "__main__":
    main()