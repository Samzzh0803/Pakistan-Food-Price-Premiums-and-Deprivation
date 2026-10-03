"""Discover, retrieve, parse, and profile official PBS weekly SPI Annexures."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pbs_history"
INTERIM = ROOT / "data" / "interim" / "pbs_history"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True); INTERIM.mkdir(parents=True, exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
ARCHIVE = "https://www.pbs.gov.pk/cpi-press-release-june-14/"
PARSER_VERSION = "pbs-weekly-v2"
CITY_FIXES = {"Rawal-pindi": "Rawalpindi", "Gujran-wala": "Gujranwala", "Faisal-abad": "Faisalabad", "Sar-godha": "Sargodha", "Baha-walpur": "Bahawalpur", "Hyder-abad": "Hyderabad", "Pesha-war": "Peshawar", "Khuz-dar": "Khuzdar", "Islam-abad": "Islamabad"}


def get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=(3, 8))
    time.sleep(0.4)
    return response


def week_from_text(text: str) -> str | None:
    match = re.search(r"(?:ended|on)[^0-9]*(\d{1,2})[-./](\d{1,2})[-./](20\d{2})", text, re.I)
    if not match:
        match = re.search(r"(\d{2})[-.](\d{2})[-.](20\d{2})", text)
    if not match:
        return None
    return date(int(match.group(3)), int(match.group(2)), int(match.group(1))).isoformat()


def discovered_links(session: requests.Session) -> list[dict[str, str]]:
    pages = []
    for d in [date(2026, 8, 27), date(2026, 8, 20), date(2026, 8, 13), date(2026, 7, 30), date(2026, 6, 11), date(2026, 5, 21)]:
        pages.append(f"https://www.pbs.gov.pk/weekly-sensitive-price-indicator-spi-for-the-week-ended-on-{d:%d-%m-%Y}/")
    links: dict[str, dict[str, str]] = {}
    for page in pages:
        try:
            response = get(session, page)
            if response.status_code != 200:
                continue
            path = RAW / ("page_" + hashlib.sha256(page.encode()).hexdigest()[:12] + ".html")
            path.write_bytes(response.content)
            soup = BeautifulSoup(response.content, "lxml")
            for anchor in soup.find_all("a", href=True):
                href = urljoin(page, anchor["href"])
                label = " ".join(anchor.get_text(" ", strip=True).split())
                if "annex" not in label.lower() and "annex" not in href.lower():
                    continue
                if not href.lower().endswith((".pdf", ".xlsx", ".xls")):
                    continue
                week = week_from_text(label + " " + page + " " + href)
                if week:
                    links[href] = {"week_end": week, "source_url": href, "discovery_page": page, "artifact_type": "xlsx" if "xlsx" in href.lower() else "pdf"}
        except requests.RequestException:
            continue
    # Include the known leads even when their archive links are stale, for status evidence.
    for url, week in [("https://www.pbs.gov.pk/sites/default/files/price_statistics/weekly_spi/SPI_Annex%26USCP_20072023.pdf", "2023-07-20"), ("https://www.pbs.gov.pk/sites/default/files/price_statistics/weekly_spi/spi_annex_31032022.pdf", "2022-03-31")]:
        links.setdefault(url, {"week_end": week, "source_url": url, "discovery_page": ARCHIVE, "artifact_type": "pdf"})
    return list(links.values())


def city_name(raw: str) -> str:
    value = re.sub(r"\s+", " ", str(raw).replace("\n", " ").strip())
    value = re.sub(r"\s*\(\d+\)", "", value).strip()
    return CITY_FIXES.get(value, value)


def parse_xlsx(path: Path, week: str, source_url: str) -> pd.DataFrame:
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet, header=None)
        desc_rows = [i for i in range(len(frame)) if frame.iloc[i].astype(str).str.upper().eq("DESCRIPTION").any()]
        if not desc_rows:
            continue
        header = desc_rows[0]
        desc_col = int(frame.iloc[header].astype(str).str.upper().eq("DESCRIPTION").idxmax())
        unit_matches = frame.iloc[header].astype(str).str.upper().eq("UNIT")
        unit_col = int(unit_matches.idxmax()) if unit_matches.any() else desc_col + 1
        city_cols: dict[int, tuple[str, str]] = {}
        current_city = ""
        for col in range(unit_col + 1, frame.shape[1]):
            labels = [str(frame.iloc[r, col]) for r in range(max(0, header - 2), header + 1) if pd.notna(frame.iloc[r, col])]
            possible = [x for x in labels if x and x.lower() not in {"min", "avg", "max", "nan"} and not x.isdigit()]
            if possible:
                current_city = city_name(possible[-1])
            stat = str(frame.iloc[header, col]).strip().upper()
            if current_city and stat in {"MIN", "AVG", "MAX"}:
                city_cols[col] = (current_city, stat)
        if len(city_cols) < 3:
            continue
        rows = []
        for r in range(header + 1, len(frame)):
            raw_item = str(frame.iloc[r, desc_col]).strip()
            if not raw_item or raw_item.lower() in {"nan", "description"} or pd.notna(pd.to_numeric(raw_item, errors="coerce")) or not re.match(r"^\d+\s*$", str(frame.iloc[r, 0]).strip()):
                continue
            unit = str(frame.iloc[r, unit_col]).strip()
            for col, (city, stat) in city_cols.items():
                value = pd.to_numeric(frame.iloc[r, col], errors="coerce")
                rows.append({"week_end": week, "city": city, "commodity_raw": raw_item, "commodity_canonical": re.sub(r"[^a-z0-9]+", "_", raw_item.lower()).strip("_"), "unit_raw": unit, "unit_canonical": unit.lower(), "price_" + stat.lower(): value, "source_url": source_url, "source_file": str(path.relative_to(ROOT)).replace("\\", "/"), "parser_version": PARSER_VERSION})
        parsed = pd.DataFrame(rows)
        if not parsed.empty:
            return parsed.groupby(["week_end", "city", "commodity_raw", "commodity_canonical", "unit_raw", "unit_canonical", "source_url", "source_file", "parser_version"], dropna=False).first().reset_index()
    raise ValueError(f"No validated Appendix-A-like table found in {path}")


def parse_pdf(path: Path, week: str, source_url: str) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    if not re.search(r"CONSUMER PRICES OF ESSENTIAL ITEMS|Appendix-A", text, re.I):
        raise ValueError("PDF has no identifiable essential-item table text")
    raise ValueError("Native PDF table requires a layout branch not validated by this sample")


def main() -> None:
    session = requests.Session(); session.headers["User-Agent"] = "DSFSG-PBS-history-audit/2.0"
    links = discovered_links(session)
    inventory, manifest, parsed_frames = [], [], []
    for index, item in enumerate(links, 1):
        try:
            response = get(session, item["source_url"])
        except requests.RequestException as exc:
            inventory.append({"week_end": item["week_end"], "year": item["week_end"][:4], "source_url": item["source_url"], "artifact_type": item["artifact_type"], "http_status": "request_error", "download_success": False, "city_item_table_present": False, "parser_success": False, "item_count": "", "city_count": "", "notes": repr(exc)})
            continue
        success = response.status_code == 200 and len(response.content) > 1000 and (response.content[:2] == b"PK" or response.content[:4] == b"%PDF")
        local = ""
        parser_status = "not_attempted"
        item_count = city_count = ""
        notes = ""
        if success:
            suffix = ".xlsx" if item["artifact_type"] == "xlsx" else ".pdf"
            path = RAW / f"{item['week_end']}_{index}{suffix}"; path.write_bytes(response.content); local = str(path.relative_to(ROOT)).replace("\\", "/")
            try:
                parsed = parse_xlsx(path, item["week_end"], item["source_url"]) if suffix == ".xlsx" else parse_pdf(path, item["week_end"], item["source_url"])
                parsed_frames.append(parsed); parser_status = "success"; item_count = parsed["commodity_raw"].nunique(); city_count = parsed["city"].nunique()
            except Exception as exc:
                parser_status = "failed"; notes = repr(exc)
            manifest.append({"source": "PBS", "source_url": item["source_url"], "week_end": item["week_end"], "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "local_path": local, "content_type": response.headers.get("content-type", ""), "http_status": response.status_code, "file_size_bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest(), "parser_status": parser_status, "notes": notes})
        inventory.append({"week_end": item["week_end"], "year": item["week_end"][:4], "source_url": item["source_url"], "artifact_type": item["artifact_type"], "http_status": response.status_code, "download_success": success, "city_item_table_present": parser_status == "success", "parser_success": parser_status == "success", "item_count": item_count, "city_count": city_count, "notes": notes or ("official response not a valid file" if not success else "")})
    pd.DataFrame(inventory).sort_values(["week_end", "source_url"]).to_csv(PROCESSED / "pbs_historical_inventory.csv", index=False)
    if manifest:
        pd.DataFrame(manifest).to_csv(RAW / "source_manifest_phase2.csv", index=False)
    panel = pd.concat(parsed_frames, ignore_index=True) if parsed_frames else pd.DataFrame()
    if not panel.empty:
        panel = panel[~panel["commodity_raw"].astype(str).str.fullmatch(r"\d+(?:\.0)?")].copy()
        panel.to_csv(PROCESSED / "pbs_weekly_panel.csv", index=False)
        panel.to_parquet(PROCESSED / "pbs_weekly_panel.parquet", index=False)
    else:
        pd.DataFrame(columns=["week_end", "city", "commodity_raw", "commodity_canonical", "unit_raw", "unit_canonical", "price_avg", "source_url", "source_file", "parser_version"]).to_csv(PROCESSED / "pbs_weekly_panel.csv", index=False)
    (INTERIM / "discovery.json").write_text(json.dumps({"links": links, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print(json.dumps({"links": len(links), "parsed_files": len(parsed_frames), "panel_rows": len(panel)}))


if __name__ == "__main__":
    main()