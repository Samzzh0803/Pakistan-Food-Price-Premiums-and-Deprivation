# Source Inventory

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
