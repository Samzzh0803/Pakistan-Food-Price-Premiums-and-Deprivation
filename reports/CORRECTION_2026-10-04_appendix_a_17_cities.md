# Correction, 4 October 2026: PBS SPI Appendix-A covers 17 cities, not 7

## What was wrong

`scripts/final_data_scope_check.py` reads Appendix-A through `first_a_item_block()`, which stops after the first table. Appendix-A is three tables stacked in one sheet. All three share the same 51 item rows:

| Block | Header row (approx.) | Columns |
|---|---|---|
| 1 | 3 | Islamabad (01), Rawalpindi (02), Gujranwala (03), Sialkot (04), Lahore (05), Faisalabad (06), Sargodha (07) |
| 2 | 61 | Multan (08), Bahawalpur (09), Karachi (10), Hyderabad (11), Sukkur (12), Larkana (13), Peshawar (14) |
| 3 | 119 | Bannu (15), Quetta (16), Khuzdar (17), then PBS national summary columns (not cities) |

That one parsing error affected three artefacts:

1. `reports/FINAL_DATA_SCOPE_CHECK.md` says Appendix-A has seven cities. **That is false.** Its 51-item finding is correct.
2. `data/verification/pbs_verified_sample.csv` (16,380 rows) has three rows for each (week, city, item) key. All three blocks were read, but the first block's seven city names were stamped on every block, so the second "Islamabad" row is really Multan and the third is Bannu. Some rows are national summary columns. Milestone 01 explained 16,380 as 5,460 records times MIN/AVG/MAX; that reading was a coincidence and is wrong. The summaries derived from this file under `data/verification/` are contaminated.
3. `data/processed/pbs_weekly_panel.csv` holds one week only (357 rows, 7 cities), and its lag and volatility columns are empty.

None of these files has been deleted. They stay as the audit trail. **Do not reuse them.**

## What replaces them

`pfp/parse_annex.py` (parser version `appendix-a-v3.0`) reads every block of Appendix-A. The pipeline `scripts/m02_build.py` asserts the following on every week:

- 3 blocks, 17 cities, 51 items and 867 rows per week, with no duplicate (week, city, item) key
- MIN <= AVG <= MAX wherever prices are positive; zeros appear only as all-zero triples
- across the 17 cities, the minimum and maximum equal PBS's National Average MIN and MAX
- the geometric mean of the quoting cities' AVG is within 0.1% of PBS's National Average AVG

Results are in `data/processed/m02/acceptance_checks_by_week.csv`. The Milestone 02 report gives the totals.

## Consequences

- Coverage is PBS's 17 SPI cities: 8 in Punjab, 4 in Sindh, 2 in Khyber Pakhtunkhwa, 2 in Balochistan, plus Islamabad. The "Punjab and Islamabad only" limitation is retired.
- The reference price is the geometric mean across quoting cities, which is PBS's own national average.
- Appendix-B (fertiliser, cement, CNG, wages and wheat) is a different set of tables and is not part of the panel.
