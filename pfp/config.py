"""Project-wide constants for the Milestone 02 pipeline."""
from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Training/holdout freeze. Weeks ending after this date are holdout: downloaded and
# hashed, never parsed into the master, plotted, or used to fit anything in M02.
TRAIN_CUTOFF = date(2026, 9, 24)
PANEL_START = date(2025, 5, 15)

PARSER_VERSION = "appendix-a-v3.0"

RAW_ANNEX_DIRS = [
    ROOT / "data" / "raw" / "verification",
    ROOT / "data" / "raw" / "verification" / "final_scope",
    ROOT / "data" / "raw" / "pbs_backfill",
]
HOLDOUT_DIR = ROOT / "data" / "raw" / "pbs_holdout"
BACKFILL_DIR = ROOT / "data" / "raw" / "pbs_backfill"
BACKFILL_MANIFEST = ROOT / "data" / "raw" / "pbs_backfill_manifest.csv"

EXTERNAL_RAW = ROOT / "data" / "external" / "raw"
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed" / "m02"
REPORTS = ROOT / "reports" / "m02"
FIGURES = REPORTS / "figures"
NUMBERS_JSON = REPORTS / "m02_numbers.json"

N_CITIES = 17
N_ITEMS = 51
ROWS_PER_WEEK = N_CITIES * N_ITEMS  # 867

# PBS SPI city codes -> province. Islamabad is the Islamabad Capital Territory.
CITY_PROVINCE = {
    "Islamabad": "ICT",
    "Rawalpindi": "Punjab",
    "Gujranwala": "Punjab",
    "Sialkot": "Punjab",
    "Lahore": "Punjab",
    "Faisalabad": "Punjab",
    "Sargodha": "Punjab",
    "Multan": "Punjab",
    "Bahawalpur": "Punjab",
    "Karachi": "Sindh",
    "Hyderabad": "Sindh",
    "Sukkur": "Sindh",
    "Larkana": "Sindh",
    "Peshawar": "KP",
    "Bannu": "KP",
    "Quetta": "Balochistan",
    "Khuzdar": "Balochistan",
}

# Okabe-Ito colour-blind-safe palette, keyed by province.
PROVINCE_COLORS = {
    "ICT": "#000000",
    "Punjab": "#0072B2",
    "Sindh": "#E69F00",
    "KP": "#009E73",
    "Balochistan": "#CC79A7",
}
