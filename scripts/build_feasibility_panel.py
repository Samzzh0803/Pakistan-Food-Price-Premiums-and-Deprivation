"""Build mappings, alignment summaries, and an honest small panel from collected evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data" / "raw"


def main() -> None:
    pd.DataFrame([{
        "pbs_commodity_raw": "Tomato", "amis_commodity_raw": "Tomato", "canonical_commodity": "tomato",
        "pbs_unit": "1 Kg", "amis_unit": "unknown", "conversion_possible": False, "conversion_formula": "",
        "semantic_match": "UNKNOWN", "confidence": "LOW", "notes": "AMIS records were not retrievable through a reproducible HTTPS data query."
    }, {
        "pbs_commodity_raw": "Potato", "amis_commodity_raw": "Potato", "canonical_commodity": "potato",
        "pbs_unit": "1 Kg", "amis_unit": "unknown", "conversion_possible": False, "conversion_formula": "",
        "semantic_match": "UNKNOWN", "confidence": "LOW", "notes": "AMIS records were not retrievable through a reproducible HTTPS data query."
    }, {
        "pbs_commodity_raw": "Onion", "amis_commodity_raw": "Onion", "canonical_commodity": "onion",
        "pbs_unit": "1 Kg", "amis_unit": "unknown", "conversion_possible": False, "conversion_formula": "",
        "semantic_match": "UNKNOWN", "confidence": "LOW", "notes": "AMIS records were not retrievable through a reproducible HTTPS data query."
    }]).to_csv(PROCESSED / "commodity_mapping.csv", index=False)
    pd.DataFrame([{
        "pbs_city": city, "amis_market": "unknown", "canonical_city": city.lower(), "exact_geographic_match": False,
        "notes": "AMIS market records unavailable for confirmation."
    } for city in ["Lahore", "Faisalabad", "Rawalpindi", "Gujranwala", "Multan"]]).to_csv(PROCESSED / "city_mapping.csv", index=False)
    columns = ["week_end", "city", "commodity", "pbs_retail_price", "pbs_retail_unit", "amis_wholesale_price",
               "amis_wholesale_unit", "wholesale_to_retail_spread", "wholesale_pct_change", "retail_pct_change",
               "source_pbs", "source_amis"]
    pd.DataFrame(columns=columns).to_csv(PROCESSED / "feasibility_sample.csv", index=False)
    align = pd.DataFrame([{
        "strategy": strategy, "matched_rows": 0, "candidate_rows": 0, "missing_percentage": 100.0,
        "median_days_from_week_end": None, "notes": "No AMIS daily observations available for alignment."
    } for strategy in ["same_day_nearest_earlier", "trailing_7_day_median", "trailing_7_day_mean"]])
    align.to_csv(PROCESSED / "temporal_alignment_summary.csv", index=False)
    (PROCESSED / "build_evidence.json").write_text(json.dumps({"status": "no_matched_panel", "reason": "AMIS historical records unavailable"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()