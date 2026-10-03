"""Audit AMIS access without bypassing transport or access controls."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "amis"
RAW.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "data" / "raw" / "source_manifest.csv"
URLS = ["https://www.amis.pk/", "http://www.amis.pk/", "https://amis.pk/", "http://amis.pk/"]


def main() -> None:
    records = []
    for url in URLS:
        try:
            response = requests.get(url, timeout=45, allow_redirects=True, headers={"User-Agent": "DSFSG-feasibility-audit/1.0"})
            body = response.content
            path = RAW / ("response_" + str(len(records) + 1) + ".bin")
            path.write_bytes(body)
            records.append({"url": url, "final_url": response.url, "status": response.status_code,
                            "content_type": response.headers.get("content-type", ""), "bytes": len(body),
                            "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "sha256": hashlib.sha256(body).hexdigest(), "text_snippet": body[:500].decode("utf-8", "replace")})
        except Exception as exc:
            error = {"url": url, "error": repr(exc), "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
            (RAW / ("error_" + str(len(records) + 1) + ".json")).write_text(json.dumps(error, indent=2), encoding="utf-8")
            records.append(error)
    (RAW / "amis_access.json").write_text(json.dumps({"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "attempts": records}, indent=2), encoding="utf-8")
    rows = []
    for record in records:
        if "local_path" in record:
            rows.append({"source": "AMIS", "source_url": record["url"], "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                         "local_path": record["local_path"], "content_type": record.get("content_type", ""),
                         "file_size_bytes": record["bytes"], "sha256": record["sha256"], "notes": "official entry point transport probe"})
    if rows:
        pd.DataFrame(rows).to_csv(MANIFEST, mode="a" if MANIFEST.exists() else "w", header=not MANIFEST.exists(), index=False)
    print(json.dumps({"attempts": len(records), "successful_http_responses": sum("status" in x for x in records)}))


if __name__ == "__main__":
    main()