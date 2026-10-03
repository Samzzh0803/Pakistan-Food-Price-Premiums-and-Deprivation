# Schema Drift

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
