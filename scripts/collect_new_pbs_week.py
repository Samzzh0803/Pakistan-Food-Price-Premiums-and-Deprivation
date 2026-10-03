"""Collect and validate the newest official PBS weekly Annexure into holdout storage."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "raw" / "pbs_future_holdout"
HOLDOUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    page = requests.get("https://www.pbs.gov.pk/price-statistics/", timeout=60, headers={"User-Agent": "DSFSG-PBS-history-audit/2.0"})
    page.raise_for_status()
    links = [a["href"] for a in BeautifulSoup(page.content, "lxml").find_all("a", href=True) if "Annex" in a.get_text(" ", strip=True) and a["href"].lower().endswith((".xlsx", ".pdf"))]
    if not links:
        raise RuntimeError("No newest Annexure link found")
    url = links[0]; response = requests.get(url, timeout=90, headers={"User-Agent": "DSFSG-PBS-history-audit/2.0"}); response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest(); suffix = ".xlsx" if url.lower().endswith("xlsx") else ".pdf"; path = HOLDOUT / (digest[:16] + suffix)
    if not path.exists(): path.write_bytes(response.content)
    print(f"collected={path.relative_to(ROOT).as_posix()} url={url} sha256={digest} retrieved_at_utc={datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()