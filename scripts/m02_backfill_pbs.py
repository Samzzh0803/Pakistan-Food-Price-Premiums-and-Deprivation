"""Backfill PBS weekly SPI Annexure workbooks from the release pages.

For each week from PANEL_START to today, try the release page for the Thursday,
then Wednesday, then Friday, and download the Annexure .xlsx it links. Weeks after
TRAIN_CUTOFF go to the holdout folder and are only hashed, never parsed.
Every attempt (hit or miss) is written to the backfill manifest.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import sys
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfp.config import BACKFILL_DIR, BACKFILL_MANIFEST, HOLDOUT_DIR, PANEL_START, TRAIN_CUTOFF  # noqa: E402

PAGE = "https://www.pbs.gov.pk/weekly-sensitive-price-indicator-spi-for-the-week-ended-on-{:%d-%m-%Y}/"
UA = {"User-Agent": "Mozilla/5.0 (academic research; Habib University DSfSG course project)"}
DELAY = 1.0
FIELDS = ["thursday", "candidate_date", "page_url", "page_status", "annex_url", "local_path",
          "sha256", "bytes", "retrieved_at_utc", "split", "outcome"]


def is_xlsx(content: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            return any(n.startswith("xl/") for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def main() -> None:
    s = requests.Session()
    s.headers.update(UA)
    rows = []
    today = date.today()
    thu = PANEL_START
    while thu <= today:
        found = False
        for offset in (0, -1, 1):  # Thursday, Wednesday, Friday
            d = thu + timedelta(days=offset)
            url = PAGE.format(d)
            rec = {"thursday": thu.isoformat(), "candidate_date": d.isoformat(), "page_url": url}
            try:
                r = s.get(url, timeout=30)
                time.sleep(DELAY)
            except requests.RequestException as e:
                rows.append({**rec, "page_status": "error", "outcome": f"page_error:{type(e).__name__}"})
                continue
            rec["page_status"] = r.status_code
            if r.status_code != 200:
                rows.append({**rec, "outcome": "page_not_found"})
                continue
            links = re.findall(r'href="([^"]*Annex[^"]*\.xlsx)"', r.text, re.I)
            if not links:
                rows.append({**rec, "outcome": "no_annex_xlsx_link"})
                continue
            annex = links[0]
            split = "holdout" if d > TRAIN_CUTOFF else "train"
            dest_dir = HOLDOUT_DIR if split == "holdout" else BACKFILL_DIR
            dest = dest_dir / f"pbs_annex_{d.isoformat()}.xlsx"
            try:
                a = s.get(annex, timeout=60)
                time.sleep(DELAY)
            except requests.RequestException as e:
                rows.append({**rec, "annex_url": annex, "outcome": f"annex_error:{type(e).__name__}"})
                continue
            if a.status_code != 200 or not is_xlsx(a.content):
                rows.append({**rec, "annex_url": annex, "outcome": f"annex_invalid:{a.status_code}"})
                continue
            dest.write_bytes(a.content)
            rows.append({**rec, "annex_url": annex, "local_path": dest.relative_to(dest_dir.parents[2]).as_posix(),
                         "sha256": hashlib.sha256(a.content).hexdigest(), "bytes": len(a.content),
                         "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "split": split,
                         "outcome": "downloaded"})
            print(thu, d, split, "downloaded", flush=True)
            found = True
            break
        if not found:
            print(thu, "missing", flush=True)
        thu += timedelta(days=7)
    with BACKFILL_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
