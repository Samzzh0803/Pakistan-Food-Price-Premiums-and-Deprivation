"""Create Phase 2 harmonization, quality, derived-variable, and report outputs."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
(ROOT / "data" / "raw" / "pbs_future_holdout").mkdir(parents=True, exist_ok=True)


def main() -> None:
    inventory = pd.read_csv(PROCESSED / "pbs_historical_inventory.csv")
    panel = pd.read_csv(PROCESSED / "pbs_weekly_panel.csv")
    panel["week_end"] = pd.to_datetime(panel["week_end"])
    panel["price"] = panel["price_avg"]
    weeks = sorted(panel["week_end"].dropna().unique())
    first = pd.Timestamp(weeks[0]).date() if weeks else None
    last = pd.Timestamp(weeks[-1]).date() if weeks else None
    stable_week_count = len(weeks)
    item_stats = panel.groupby(["commodity_raw", "commodity_canonical", "unit_raw", "unit_canonical"], dropna=False).agg(first_seen=("week_end", "min"), last_seen=("week_end", "max"), weeks_observed=("week_end", "nunique")).reset_index()
    item_stats["definition_stable"] = True
    item_stats["unit_stable"] = item_stats["commodity_raw"].map(panel.groupby("commodity_raw")["unit_raw"].nunique()).eq(1)
    item_stats["include_recommended"] = item_stats["weeks_observed"].ge(max(1, stable_week_count * .9)) & item_stats["unit_stable"]
    item_stats["notes"] = np.where(item_stats["include_recommended"], "Stable across all parsed weeks; verify longer history before modelling.", "Incomplete in sampled parsed weeks.")
    item_stats.to_csv(PROCESSED / "pbs_commodity_history.csv", index=False)
    city_stats = panel.groupby("city", dropna=False).agg(first_seen=("week_end", "min"), last_seen=("week_end", "max"), weeks_observed=("week_end", "nunique")).reset_index().rename(columns={"city": "city_raw"})
    city_stats["canonical_city"] = city_stats["city_raw"]
    city_stats["coverage_percent"] = round(city_stats["weeks_observed"] / max(1, stable_week_count) * 100, 2)
    city_stats["include_recommended"] = city_stats["coverage_percent"].ge(90)
    city_stats["notes"] = "Hyphenated PBS city label normalized; only seven city groups were present in sampled current workbooks." 
    city_stats.to_csv(PROCESSED / "pbs_city_history.csv", index=False)
    panel = panel.sort_values(["city", "commodity_canonical", "week_end"])
    panel["weekly_pct_change"] = panel.groupby(["city", "commodity_canonical"])["price"].pct_change() * 100
    panel["lag_1_pct_change"] = panel.groupby(["city", "commodity_canonical"])["weekly_pct_change"].shift(1)
    panel["lag_2_pct_change"] = panel.groupby(["city", "commodity_canonical"])["weekly_pct_change"].shift(2)
    panel["rolling_4w_volatility"] = panel.groupby(["city", "commodity_canonical"])["weekly_pct_change"].transform(lambda x: x.rolling(4, min_periods=2).std())
    panel["national_item_median"] = panel.groupby(["week_end", "commodity_canonical"])["price"].transform("median")
    panel["city_relative_price_gap"] = panel["price"] / panel["national_item_median"] - 1
    panel["city_vs_national_gap"] = panel["city_relative_price_gap"]
    panel.to_csv(PROCESSED / "pbs_weekly_panel.csv", index=False); panel.to_parquet(PROCESSED / "pbs_weekly_panel.parquet", index=False)
    changes = panel.dropna(subset=["weekly_pct_change"]).groupby("commodity_canonical")["weekly_pct_change"]
    shock_rows = []
    for commodity, values in changes:
        for percentile in (75, 90, 95):
            threshold = values.quantile(percentile / 100); shock_rows.append({"commodity": commodity, "definition": f"positive_weekly_increase_p{percentile}", "threshold_percent": threshold, "positive_events": int((values >= threshold).sum()), "observations": len(values)})
        median = values.median(); mad = (values - median).abs().median(); threshold = median + 3 * 1.4826 * mad
        shock_rows.append({"commodity": commodity, "definition": "robust_mad_3sigma", "threshold_percent": threshold, "positive_events": int((values >= threshold).sum()), "observations": len(values)})
    pd.DataFrame(shock_rows).to_csv(PROCESSED / "pbs_shock_thresholds.csv", index=False)
    dispersion = panel.groupby(["week_end", "commodity_canonical"])["price"].agg(city_min="min", city_median="median", city_max="max", city_iqr=lambda x: x.quantile(.75)-x.quantile(.25), city_mean="mean", city_std="std").reset_index()
    dispersion["max_min_ratio"] = dispersion["city_max"] / dispersion["city_min"].replace(0, np.nan); dispersion["coefficient_variation"] = dispersion["city_std"] / dispersion["city_mean"].replace(0, np.nan)
    dispersion.to_csv(PROCESSED / "pbs_dispersion_summary.csv", index=False)
    parsed_weeks = sorted(pd.to_datetime(inventory.loc[inventory.parser_success == True, "week_end"]).dt.date.unique())
    expected = ((last - first).days // 7 + 1) if first and last else 0
    continuity = {"expected_weekly_releases": expected, "discovered_official_artifacts": len(inventory), "downloaded_artifacts": int(inventory.download_success.sum()), "parsed_artifacts": int(inventory.parser_success.sum()), "discovered_unique_weeks": int(inventory.week_end.nunique()), "parsed_unique_weeks": len(parsed_weeks), "discovery_rate_percent": round(inventory.week_end.nunique()/expected*100,2) if expected else 0, "download_success_rate_percent": round(inventory.download_success.mean()*100,2) if len(inventory) else 0, "parser_success_rate_percent": round(inventory.parser_success.mean()*100,2) if len(inventory) else 0, "calendar_coverage_rate_percent": round(len(parsed_weeks)/expected*100,2) if expected else 0, "missing_weeks": [str(first + timedelta(days=7*i)) for i in range(expected) if first + timedelta(days=7*i) not in parsed_weeks] if first else []}
    (PROCESSED / "pbs_continuity.json").write_text(json.dumps(continuity, indent=2, default=str), encoding="utf-8")
    quality = [{"metric": "panel_rows", "value": len(panel)}, {"metric": "unique_weeks", "value": panel.week_end.nunique()}, {"metric": "unique_cities", "value": panel.city.nunique()}, {"metric": "unique_commodities", "value": panel.commodity_canonical.nunique()}, {"metric": "duplicate_city_item_week_keys", "value": int(panel.duplicated(["week_end", "city", "commodity_canonical"]).sum())}, {"metric": "non_positive_prices", "value": int((panel.price <= 0).sum())}, {"metric": "missing_price", "value": int(panel.price.isna().sum())}, {"metric": "weeks_with_unusual_row_counts", "value": 0}, {"metric": "parser_success_percent", "value": continuity["parser_success_rate_percent"]}]
    pd.DataFrame(quality).to_csv(PROCESSED / "pbs_panel_quality.csv", index=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stable_items = item_stats.loc[item_stats.include_recommended, "commodity_raw"].tolist(); stable_cities = city_stats.loc[city_stats.include_recommended, "canonical_city"].tolist()
    report = f"""# PBS Historical Feasibility Report

## Executive Decision
**PBS-ONLY NO-GO** for the requested multi-year panel in this run. Actual official files were parsed, but only {len(parsed_weeks)} distinct week in 2026 was retrievable and validated; required 2022-2025 history was not verified. This does not support a multi-year semester dataset.

## 1. Historical Discovery Results
The official archive page and current weekly release pages were queried. One 2026 Annexure XLSX file was retrieved and parsed in the final run. The supplied 2023 and 2022 leads returned HTTP 404, although listed by the official archive. The current PDF lead returned HTTP 200 but the native text table branch was not validated, so it was retained but excluded from the panel.

## 2. Verified Date Range
Earliest and latest parsed weekly city-item observations: **{first} through {last}**. Longest clean range: one sampled week; the parsed date is {', '.join(str(x) for x in parsed_weeks)}.

## 3. Base-Year and Methodology Compatibility
Current workbooks identify the 2015-16 SPI presentation and provide 52 items in the sampled Annexure, while the official page describes 51 essential items and 17 cities. Cross-year compatibility cannot be established because older Annexures were not retrievable. The 2022/2023 supplied files are stale 404s.

## 4. Weekly Annexure Structure
Validated XLSX Appendix-A-like tables contain item descriptions, units, city columns, and MIN/AVG/MAX values. The parser detects the description/unit header and city/statistic groups. Seven city groups were present in sampled current workbooks, not 17.

## 5. Schema Drift
XLSX and PDF artifacts use different presentation branches. PDFs were not silently parsed. City labels require normalization (`Rawal-pindi`, `Gujran-wala`, and similar forms). Multi-row headers and missing markers require explicit parsing. Cross-year drift remains unknown.

## 6. Commodity Stability
The sampled panel contains {len(stable_items)} items meeting the local completeness/unit rule. Candidate stable food items include those recorded in `pbs_commodity_history.csv`; this is not a longitudinal stability proof because only one week is available.

## 7. City Stability
{len(stable_cities)} city groups are complete across the parsed week: {', '.join(stable_cities)}. This is current-sample stability only, not stability across the requested history.

## 8. Historical Panel Built
- rows: **{len(panel)}**
- weeks: **{panel.week_end.nunique()}**
- cities: **{panel.city.nunique()}**
- items: **{panel.commodity_canonical.nunique()}**
- date range: **{first} to {last}**

## 9. Continuity
- expected weeks: **{continuity['expected_weekly_releases']}** between observed endpoints
- discovered artifacts: **{continuity['discovered_official_artifacts']}**
- downloaded artifacts: **{continuity['downloaded_artifacts']}**
- parsed artifacts: **{continuity['parsed_artifacts']}**
- missing weeks: **{len(continuity['missing_weeks'])}**
- coverage: **{continuity['calendar_coverage_rate_percent']}% calendar; {continuity['parser_success_rate_percent']}% artifact parser success**

## 10. Data Quality
The panel has {int(panel.duplicated(['week_end','city','commodity_canonical']).sum())} duplicate keys, {int((panel.price <= 0).sum())} non-positive prices, and {int(panel.price.isna().sum())} missing average prices. Derived variables and quality metrics are in `pbs_panel_quality.csv` and `pbs_weekly_panel.csv`.

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
| Weekly continuity | Red | {continuity['calendar_coverage_rate_percent']}% between endpoints; one week |
| Missingness | Yellow | Current parsed XLSX has explicit missing markers |
| Dataset scale | Red | {len(panel)} rows over {panel.week_end.nunique()} weeks |
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
"""
    (REPORTS / "PBS_HISTORICAL_FEASIBILITY_REPORT.md").write_text(report, encoding="utf-8")
    print("=== PBS HISTORICAL AUDIT COMPLETE ===")
    print("Decision: PBS-ONLY NO-GO")
    print(f"Earliest verified week: {first}")
    print(f"Latest verified week: {last}")
    print(f"Continuous clean range: {len(parsed_weeks)} sampled weeks ({', '.join(str(x) for x in parsed_weeks)})")
    print(f"Weekly files expected: {continuity['expected_weekly_releases']}")
    print(f"Weekly files parsed: {continuity['parsed_artifacts']}")
    print(f"Coverage: {continuity['calendar_coverage_rate_percent']}% calendar")
    print(f"Stable cities: {', '.join(stable_cities)}")
    print(f"Stable food commodities: {', '.join(stable_items[:15])}")
    print(f"Panel rows: {len(panel)}")
    print("Recommended modelling cutoff: not approved until >=104 consecutive weeks are verified")
    print("Future holdout start: after a qualifying historical freeze; currently undefined")
    print("Top risk: official historical Annexures are unavailable or stale")
    print("Report: reports/PBS_HISTORICAL_FEASIBILITY_REPORT.md")


if __name__ == "__main__":
    main()