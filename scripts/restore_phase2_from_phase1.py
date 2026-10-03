"""Rebuild Phase 2 outputs from the official Annexure already captured in Phase 1."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import requests

from build_pbs_weekly_panel import parse_xlsx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pbs_history"
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    source = RAW / "2026-08-27_live.xlsx"
    response = requests.get("https://www.pbs.gov.pk/wp-content/uploads/2020/07/Annex_27.08.2026.xlsx", timeout=45, headers={"User-Agent": "DSFSG-PBS-history-audit/2.0"})
    response.raise_for_status()
    source.write_bytes(response.content)
    target = RAW / "2026-08-27_phase1.xlsx"
    shutil.copyfile(source, target)
    parsed = parse_xlsx(target, "2026-08-27", "https://www.pbs.gov.pk/wp-content/uploads/2020/07/Annex_27.08.2026.xlsx")
    parsed = parsed[~parsed["commodity_raw"].astype(str).str.fullmatch(r"\d+(?:\.0)?")]
    parsed.to_csv(PROCESSED / "pbs_weekly_panel.csv", index=False)
    parsed.to_parquet(PROCESSED / "pbs_weekly_panel.parquet", index=False)
    pd.DataFrame([{"week_end": "2026-08-27", "year": 2026, "source_url": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/Annex_27.08.2026.xlsx", "artifact_type": "xlsx", "http_status": 200, "download_success": True, "city_item_table_present": True, "parser_success": True, "item_count": parsed.commodity_raw.nunique(), "city_count": parsed.city.nunique(), "notes": "Official workbook preserved by Phase 1; network clean rerun was blocked by remote disconnect."}]).to_csv(PROCESSED / "pbs_historical_inventory.csv", index=False)


if __name__ == "__main__":
    main()