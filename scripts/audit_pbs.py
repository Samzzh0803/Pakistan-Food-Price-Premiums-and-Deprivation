"""Discover and sample official PBS weekly SPI releases."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pbs"
RAW.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "data" / "raw" / "source_manifest.csv"
ENTRY = "https://www.pbs.gov.pk/price-statistics/"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_row(source: str, url: str, path: Path, data: bytes, content_type: str, notes: str) -> dict[str, str]:
    return {"source": source, "source_url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "local_path": str(path.relative_to(ROOT)).replace("\\", "/"), "content_type": content_type,
            "file_size_bytes": str(len(data)), "sha256": sha256(data), "notes": notes}


def main() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "DSFSG-feasibility-audit/1.0 (research; contact not provided)"
    response = session.get(ENTRY, timeout=60)
    response.raise_for_status()
    retrieved = datetime.now(timezone.utc).isoformat()
    page_path = RAW / "pbs_price_statistics.html"
    page_path.write_bytes(response.content)
    links = []
    for anchor in BeautifulSoup(response.content, "lxml").find_all("a", href=True):
        href = urljoin(ENTRY, anchor["href"])
        text = " ".join(anchor.get_text(" ", strip=True).split())
        if "SPI" in href.upper() or "SPI" in text.upper() or "ANNEX" in href.upper() or "ANNEX" in text.upper():
            links.append({"text": text, "url": href})
    # Prefer four materially separated years, then supplement from discovered links.
    dated: list[dict[str, str]] = []
    for link in links:
        match = re.search(r"(20\d{2})", link["url"] + " " + link["text"])
        if match:
            link["year"] = match.group(1)
            dated.append(link)
    selected: list[dict[str, str]] = []
    for year in ("2026", "2025", "2024", "2023", "2022", "2021"):
        candidates = [x for x in dated if x.get("year") == year and x["url"].lower().endswith((".xlsx", ".xls"))
                  and ("week ended" in x["text"].lower() or "annex_" in x["url"].lower())]
        if candidates:
            selected.append(candidates[0])
    if not selected:
        selected = [x for x in links if x["url"].lower().endswith((".xlsx", ".xls"))
                and ("week ended" in x["text"].lower() or "annex_" in x["url"].lower())][:4]
    rows = [manifest_row("PBS", ENTRY, page_path, response.content, response.headers.get("content-type", ""), "official discovery page")]
    samples = []
    for index, link in enumerate(selected):
        time.sleep(1)
        try:
            item = session.get(link["url"], timeout=60)
            item.raise_for_status()
            suffix = ".xlsx" if "xlsx" in item.headers.get("content-type", "") or link["url"].lower().endswith("xlsx") else ".xls"
            path = RAW / f"sample_{index + 1}{suffix}"
            path.write_bytes(item.content)
            rows.append(manifest_row("PBS", link["url"], path, item.content, item.headers.get("content-type", ""), link["text"]))
            excel = pd.ExcelFile(path)
            sheets = []
            for sheet in excel.sheet_names:
                frame = pd.read_excel(path, sheet_name=sheet, header=None)
                sheets.append({"sheet": sheet, "rows": int(frame.shape[0]), "columns": int(frame.shape[1]),
                               "preview": frame.iloc[:8, :8].fillna("").astype(str).values.tolist()})
            samples.append({"url": link["url"], "text": link["text"], "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "size_bytes": len(item.content), "sheets": sheets})
        except Exception as exc:
            samples.append({"url": link["url"], "text": link["text"], "error": repr(exc)})
    pd.DataFrame(rows).to_csv(MANIFEST, mode="a" if MANIFEST.exists() else "w", header=not MANIFEST.exists(), index=False)
    (RAW / "pbs_discovery.json").write_text(json.dumps({"entry_url": ENTRY, "retrieved_at": retrieved,
        "discovered_link_count": len(links), "selected_samples": samples, "all_relevant_links": links}, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "downloaded": sum("local_path" in x for x in samples), "links": len(links)}))


if __name__ == "__main__":
    main()