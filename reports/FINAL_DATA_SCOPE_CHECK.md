# Final Data Scope Check

## 1. Appendix-B Result
- Appendix-A cities: 7 (Islamabad, Rawalpindi, Gujranwala, Sialkot, Lahore, Faisalabad, Sargodha)
- Appendix-B cities: 17 (Islamabad, Rawalpindi, Gujranwala, Sialkot, Lahore, Faisalabad, Sargodha, Multan, Bahawalpur, Karachi, Hyderabad, Sukkur, Larkana, Peshawar, Bannu, Quetta, Khuzdar)
- Combined unique cities: 17
- Exact city list: Islamabad, Rawalpindi, Gujranwala, Sialkot, Lahore, Faisalabad, Sargodha, Multan, Bahawalpur, Karachi, Hyderabad, Sukkur, Larkana, Peshawar, Bannu, Quetta, Khuzdar
- National / regional scope justified? NO for national essential-item scope; YES for regional scope.
- Explanation: Across three inspected workbooks, Appendix-B contains the previously missing cities. It is not a continuation of the Appendix-A essential-item table: it contains separate retail tables for fertilizers, cement, CNG, wage rates, and wheat. Its cement, wage, and wheat tables show all 17 named cities; Appendix-A alone has seven city-item MIN/AVG/MAX groups.

## 2. Item Count Result
- Raw parsed count: 52 unique labels under the prior parser.
- True unique item count: 51 per week.
- Cause of 51-vs-52 discrepancy: parser bug. Each Appendix-A city-panel copy includes a numeric column-number header row with serial `1`, description `2`, and unit `3`. The prior parser accepted description `2` as an item, producing a spurious 52nd unique label. The 51 real items are repeated three times across horizontal city panels.
- Item count proposal should state: 51 essential items.
- Stable food examples: Rice Basmati Broken, Rice IRRI-6/9, pulses (Masoor, Moong, Mash, Gram), sugar, salt, and tea.

## 3. Historical Backfill Result
- 2025 Thursdays tested: 52
- 2025 valid Annexures: 13
- 2024 Thursdays tested: 52
- 2024 valid Annexures: 0
- Earliest verified week: 2025-05-15
- Latest verified week: 2026-08-27
- Total verified unique weeks: 26
- Direct URL method improved history? YES

## 4. Verified Dataset Scope
- Unique weeks: 26
- Date range: 2025-05-15 to 2026-08-27
- Unique cities: 7 (Appendix-A retail essential-item panel)
- Unique items: 51
- Stable food commodities recommended: Rice Basmati Broken, Rice IRRI-6/9, pulses (Masoor, Moong, Mash, Gram), sugar, salt, and tea.
- Total city-item-week rows: 9282
- Missingness summary: price_min=0, price_avg=0, price_max=0; zero values remain source values and are not converted to missing.
- Duplicate-key summary: 0 duplicate `(week_end, city, commodity)` keys; expected cardinality is one row per key.

## 5. Proposal Language Decision
"Punjab and Islamabad urban markets"

## 6. Exact Numbers to Insert Into Proposal
Replace:
- [[N]] = 26 weekly observations
- [[start date]] = 2025-05-15
- [[end date]] = 2026-08-27
- [[R]] = 9,282 city-item-week rows
- [[C]] = 7 cities and 51 essential items

## 7. Final Project Framing
The current title is supported only with a geographic restriction. The verified Appendix-A retail price panel supports persistent price premiums and dispersion analysis for Punjab and Islamabad urban markets. It provides city-item-level MIN/AVG/MAX retail prices, a reproducible multi-week historical panel, and future weekly releases that can be reserved for Week 14. It does not support national essential-item analysis across all 17 cities, because Appendix-B's wider coverage is for different measures, not Appendix-A retail essential items. The source is retail price data only; no upstream supply-chain causation can be claimed.

## 8. Remaining Limitations
- Direct URL testing used only the observed `Annex_DD.MM.YYYY.xlsx` filename convention and stopped after 2024-2025 as required.
- Appendix-A's retail essential-item panel covers seven cities, not the 17 cities represented elsewhere in Appendix-B.
- The backfill range should be described as verified file coverage, not assumed complete PBS history.
- Some source price cells are missing or zero and should be handled explicitly in analysis.
- Next-week classification remains secondary and depends on the recovered temporal depth after a fixed modelling cutoff.

## 9. Bottom Line
- Appendix-B contains all 17 named cities, including Karachi, Hyderabad, Sukkur, Larkana, Quetta, Peshawar, Bannu, Khuzdar, Multan, and Bahawalpur.
- Appendix-B is not an essential-item continuation and cannot enlarge the Appendix-A retail panel to 17 cities.
- The correct Appendix-A item count is 51; 52 came from the parser accepting a numeric header row.
- The direct URL sweep tested every Thursday in 2024 and 2025 using the verified official filename convention.
- The verified retail panel contains 26 weeks, 7 cities, 51 items, and 9,282 rows.
- Use "Punjab and Islamabad urban markets" in the proposal.
- Primary cross-city price-premium and dispersion analysis is supported.
- Project scope status: LOCK WITH RESTRICTIONS.
