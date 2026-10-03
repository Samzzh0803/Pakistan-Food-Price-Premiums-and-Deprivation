# Codex Task — Final Data Scope Check Before Milestone 01 Proposal

## Purpose

Do **only the final unresolved data checks** needed to lock the project scope before we rewrite the Milestone 01 proposal.

Do not train models.
Do not redesign the project.
Do not investigate AMIS again.
Do not write the proposal.

We already have Phase 1, Phase 2, and an independent feasibility verification. Preserve all of that work.

The project direction is currently:

> **Who Pays More, and When? Regional Food Price Premiums and Abnormal Price Movements in Pakistani Urban Markets**

The unresolved questions are:

1. Does `Appendix-B` contain the cities missing from `Appendix-A`?
2. Why does the parser currently report **52 items** while PBS describes the SPI as **51 essential items**?
3. How many additional weekly Annexures can be recovered by directly testing patterned week-ending-date URLs?
4. What exact verified figures should replace the placeholders in the proposal?

---

# 1. Inspect Existing Work First

Read the existing audit reports, scripts, verified sample panel, and raw downloaded PBS workbooks.

Do not overwrite previous outputs.

Use new outputs under:

```text
data/verification/final_scope/
reports/
```

---

# 2. Appendix-B City Coverage Check

This is the highest-priority check.

Open at least **3 verified Annexure XLSX workbooks** from different dates, including:

- one 2025 workbook;
- one recent 2026 workbook;
- one additional verified week.

For each workbook:

1. inspect **every sheet**;
2. inspect `Appendix-A`;
3. inspect `Appendix-B`;
4. visually inspect merged headers / split tables if necessary;
5. list every city represented in each sheet;
6. determine whether `Appendix-B` contains the cities missing from `Appendix-A`.

Do not infer from PBS's stated 17-city coverage. Read the workbook itself.

Create:

`data/verification/final_scope/appendix_city_audit.csv`

with:

```text
week_end
sheet_name
city_raw
city_canonical
city_order
source_file
notes
```

Then answer explicitly:

- How many unique cities are in Appendix-A?
- How many unique cities are in Appendix-B?
- How many unique cities are in A + B combined?
- What are the exact city names?
- Are Karachi, Hyderabad, Sukkur, Larkana, Quetta, Peshawar, Bannu, Khuzdar, Multan, Bahawalpur or other previously missing cities present?
- Is the full workbook consistent with PBS's stated 17-city coverage?

If Appendix-B has no city-item table, say exactly what it contains instead.

---

# 3. 51 vs 52 Item Audit

The current parser reports 52 items, while PBS describes SPI as 51 essential items.

Resolve this before any proposal is finalized.

For at least the same 3 workbooks:

1. enumerate every parsed item row;
2. inspect the raw workbook row labels;
3. identify:
   - duplicate items;
   - headings incorrectly treated as items;
   - subtotal/summary rows;
   - extra non-SPI items;
   - changed item definitions;
   - parser off-by-one errors;
   - Appendix-A / Appendix-B duplication;
4. compare the true item rows across sheets.

Create:

`data/verification/final_scope/item_count_audit.csv`

with:

```text
week_end
sheet_name
row_number
item_raw
unit_raw
is_real_item
is_duplicate
reason
canonical_item
notes
```

Answer explicitly:

- What is the true number of unique essential-item rows per week?
- Why did the prior parser produce 52?
- Is the extra row a parser bug, a non-SPI row, a duplicate, or a real additional item?
- What item count should the proposal state?
- Which food commodities are stable enough to mention as examples?

Do not force the result to equal 51. Report what the workbook actually supports and explain any discrepancy.

---

# 4. Direct Historical URL Backfill Test

Do not build a giant archive crawler.

Perform a **bounded direct URL-pattern test** to determine whether more historical weeks are recoverable than archive-page discovery found.

The known current pattern includes URLs like:

```text
https://www.pbs.gov.pk/wp-content/uploads/2020/07/Annex_DD.MM.YYYY.xlsx
```

The SPI week-ended dates are typically Thursdays.

## Test window

Generate candidate **Thursday week-ending dates** for:

- all of 2025;
- all of 2024.

For each date, test reasonable official PBS filename/path variants derived from already verified files.

Examples may include:

```text
Annex_DD.MM.YYYY.xlsx
Annex_DD.MM.YY.xlsx
SPI_Annex_DD.MM.YYYY.xlsx
```

Only add variants that are supported by observed official naming patterns in the repository or official pages.

For every candidate request, record:

```text
week_end
candidate_url
http_status
content_type
file_size
looks_like_xlsx
download_success
workbook_open_success
appendix_a_present
appendix_b_present
notes
```

Save to:

`data/verification/final_scope/backfill_url_test.csv`

### Important

- Do not treat an HTTP 200 HTML error page as an XLSX success.
- Validate ZIP/XLSX signature and workbook open.
- Rate-limit politely.
- Cache successful files.
- Stop after the 2024–2025 bounded sweep.
- Do not expand into older years in this task.

Then report:

- total candidate Thursdays tested;
- successful valid XLSX weeks in 2025;
- successful valid XLSX weeks in 2024;
- earliest newly verified week;
- latest verified week;
- whether direct URL construction materially improves coverage.

---

# 5. Exact Proposal Scope Figures

Using **only verified files**, compute the exact figures needed for the proposal.

Create:

`reports/FINAL_DATA_SCOPE_CHECK.md`

with this exact structure:

```markdown
# Final Data Scope Check

## 1. Appendix-B Result
- Appendix-A cities:
- Appendix-B cities:
- Combined unique cities:
- Exact city list:
- National / regional scope justified? YES / NO
- Explanation:

## 2. Item Count Result
- Raw parsed count:
- True unique item count:
- Cause of 51-vs-52 discrepancy:
- Item count proposal should state:
- Stable food examples:

## 3. Historical Backfill Result
- 2025 Thursdays tested:
- 2025 valid Annexures:
- 2024 Thursdays tested:
- 2024 valid Annexures:
- Earliest verified week:
- Latest verified week:
- Total verified unique weeks:
- Direct URL method improved history? YES / NO

## 4. Verified Dataset Scope
- Unique weeks:
- Date range:
- Unique cities:
- Unique items:
- Stable food commodities recommended:
- Total city-item-week rows:
- Missingness summary:
- Duplicate-key summary:

## 5. Proposal Language Decision
Choose one:
- "Pakistani urban markets"
- "selected Pakistani urban markets"
- "Punjab and Islamabad urban markets"
- another exact wording justified by the data

## 6. Exact Numbers to Insert Into Proposal
Replace:
- [[N]] =
- [[start date]] =
- [[end date]] =
- [[R]] =
- [[C]] =

## 7. Final Project Framing
State whether the current title and scope are supported.

## 8. Remaining Limitations
Maximum 5 bullets.

## 9. Bottom Line
Maximum 8 bullets.
```

---

# 6. Project Logic to Preserve

Do not redesign this unless the data check forces a wording change.

Current intended structure:

### Primary analysis
- cross-city price dispersion;
- persistent city price premiums;
- differences between perishable and storable commodities.

### Secondary analysis
- next-week abnormal price-movement classification, only if enough temporal depth is recovered.

The primary analysis must survive even if historical backfill remains shallow.

---

# 7. Proposal Claims That Need Evidence

Explicitly verify whether we can truthfully say:

1. PBS Annexures provide MIN / AVG / MAX price information at city-item level.
2. The dataset covers 17 cities, or a smaller number.
3. The usable data is geographically national, or only regional.
4. The item universe is 51 or 52.
5. We have a reproducible multi-week historical panel.
6. New weekly releases can be reserved for Week 14.
7. The source is retail price data only, so no upstream supply-chain causation can be claimed.

---

# 8. Final Console Output

Print only:

```text
=== FINAL DATA SCOPE CHECK COMPLETE ===
Appendix-A cities:
Appendix-B cities:
Combined cities:
Item count:
51-vs-52 cause:
2025 valid weeks:
2024 valid weeks:
Earliest verified week:
Latest verified week:
Total verified weeks:
Verified rows:
Recommended geographic wording:
Project scope status: LOCK / LOCK WITH RESTRICTIONS / DO NOT LOCK
Report: reports/FINAL_DATA_SCOPE_CHECK.md
```

Stop after this.
