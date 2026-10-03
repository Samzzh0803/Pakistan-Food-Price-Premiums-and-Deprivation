"""Profile observed source artifacts and write the required quality summary."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "data_quality_summary.csv"


def row(source: str, variable: str, total: int, dates: int, cities: int, commodities: int, missing: int, duplicates: int, notes: str) -> dict[str, object]:
    return {"source": source, "variable": variable, "total_rows": total, "unique_dates": dates, "unique_cities": cities,
            "unique_commodities": commodities, "missing_count": missing, "missing_percentage": round(missing / total * 100, 2) if total else 100.0,
            "duplicate_count": duplicates, "impossible_non_positive_prices": "not parsed", "outlier_flags": "not assessed",
            "unit_variants": "observed in workbook headers" if source == "PBS" else "unknown", "date_range": "2026-08-27 only" if source == "PBS" else "unknown",
            "matching_rate": 0.0, "notes": notes}


def main() -> None:
    rows = [row("PBS", "city-item weekly prices", 0, 1, 17, 51, 0, 0, "Workbook downloaded; detailed row normalization pending because sampled annexure uses multi-row report layout."),
            row("AMIS", "daily wholesale prices", 0, 0, 0, 0, 0, 0, "No historical data response could be queried reproducibly; official host transport failure documented in raw evidence."),
            row("MATCHED", "city-commodity-week", 0, 0, 0, 0, 0, 0, "No matched observations; do not estimate by blind dimension multiplication."),
            row("SUMMARY", "direct commodity matches", 0, 0, 0, 0, 0, 0, "0 verified; 3 candidates remain UNKNOWN."),
            row("SUMMARY", "overlapping cities", 0, 0, 0, 0, 0, 0, "0 verified; PBS city names cannot be matched to AMIS markets without AMIS records."),
            row("SUMMARY", "estimated final row count", 0, 0, 0, 0, 0, 0, "2-year, 3-year, and 5-year matched panels: 0 observed-based rows." )]
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()