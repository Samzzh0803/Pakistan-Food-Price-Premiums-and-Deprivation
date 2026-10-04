"""External sources: Census 2023 population, PSLM 2019-20 deprivation, national food CPI.

Every value read here comes from a file saved under data/external/raw/, recorded in
data/external/external_manifest.csv with its URL, sha256 and sheet/page reference.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pdfplumber

from .config import EXTERNAL, EXTERNAL_RAW

CENSUS_DIR = EXTERNAL_RAW / "census2023"
CENSUS_FILES = ["punjab_districts", "sindh_districts", "kp_districts", "balochistan_districts", "islamabad"]


def parse_census_table1() -> pd.DataFrame:
    """District rows (total, rural, urban) from Census 2023 Table 1, one row per district."""
    out = []
    for name in CENSUS_FILES:
        path = CENSUS_DIR / f"table_1_{name}.xlsx"
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        for i, r in enumerate(rows):
            label = str(r[0]).strip().upper() if r[0] is not None else ""
            # Islamabad's file has no "DISTRICT" row; its top row is the whole territory.
            is_district = label.endswith(" DISTRICT") or (name == "islamabad" and label.startswith("ISLAMABAD") and i < 8)
            if not is_district or not isinstance(r[2], (int, float)):
                continue
            rec = {"census_unit": label.replace(" DISTRICT", "").strip(), "census_file": path.name,
                   "census_sheet": ws.title, "census_row": i + 1,
                   "pop_total": float(r[2]), "area_km2": r[1], "urban_proportion_pct": r[8]}
            for j in (1, 2):
                nxt = rows[i + j] if i + j < len(rows) else None
                if nxt and str(nxt[0]).strip().upper() in ("RURAL", "URBAN"):
                    rec[f"pop_{str(nxt[0]).strip().lower()}"] = float(nxt[2]) if isinstance(nxt[2], (int, float)) else 0.0
            out.append(rec)
            if name == "islamabad":
                break
    df = pd.DataFrame(out)
    df["pop_urban"] = df.get("pop_urban").fillna(0.0)
    return df


# ----------------------------------------------------------------------------- PSLM
PSLM_PDF = EXTERNAL_RAW / "PSLM_2019_20_District_Level.pdf"

# SPI city -> (PSLM 2019-20 district(s), Census 2023 unit(s)).
CROSSWALK = {
    "Islamabad": (["Islamabad"], ["ISLAMABAD"]),
    "Rawalpindi": (["Rawalpindi"], ["RAWALPINDI"]),
    "Gujranwala": (["Gujranwala"], ["GUJRANWALA"]),
    "Sialkot": (["Sialkot"], ["SIALKOT"]),
    "Lahore": (["Lahore"], ["LAHORE"]),
    "Faisalabad": (["Faisalabad"], ["FAISALABAD"]),
    "Sargodha": (["Sargodha"], ["SARGODHA"]),
    "Multan": (["Multan"], ["MULTAN"]),
    "Bahawalpur": (["Bahawalpur"], ["BAHAWALPUR"]),
    "Karachi": (["Karachi Central", "Karachi East", "Karachi Malir", "Karachi South", "Karachi West", "Korangi"],
                ["KARACHI CENTRAL", "KARACHI EAST", "MALIR", "KARACHI SOUTH", "KARACHI WEST", "KEAMARI", "KORANGI"]),
    "Hyderabad": (["Hyderabad"], ["HYDERABAD"]),
    "Sukkur": (["Sukkur"], ["SUKKUR"]),
    "Larkana": (["Larkana"], ["LARKANA"]),
    "Peshawar": (["Peshawar"], ["PESHAWAR"]),
    "Bannu": (["Bannu"], ["BANNU"]),
    "Quetta": (["Quetta"], ["QUETTA"]),
    "Khuzdar": (["Khuzdar"], ["KHUZDAR"]),
}
# Karachi weights: PSLM has six districts, Census 2023 seven. Keamari was carved out of
# Karachi West after PSLM 2019-20 fieldwork, so its population joins West's weight.
KARACHI_PSLM_TO_CENSUS = {"Karachi Central": ["KARACHI CENTRAL"], "Karachi East": ["KARACHI EAST"],
                          "Karachi Malir": ["MALIR"], "Karachi South": ["KARACHI SOUTH"],
                          "Karachi West": ["KARACHI WEST", "KEAMARI"], "Korangi": ["KORANGI"]}

# 0-based pdf page ranges of each table (pdf page = printed page + 2).
PSLM_PAGES = {
    "fies": range(578, 582),          # Table 9.1, printed pp. 577-580
    "literacy": range(132, 136),      # Table 2.14(a), printed pp. 131-134
    "out_of_school": range(144, 148),  # Table 2.15, printed pp. 143-146
    "water": range(474, 482),         # Table 7.1, printed pp. 474-481
}
PSLM_TABLE_NAMES = {"fies": "Table 9.1", "literacy": "Table 2.14(a)", "out_of_school": "Table 2.15", "water": "Table 7.1"}
NUM = r"-?\d*\.?\d+"
# Strata PSLM did not sample (verified on the printed rows).
PSLM_RURAL_ONLY = {"Khuzdar"}  # blank Urban row in Table 7.1; 0 0 0 urban in Table 2.15
PSLM_URBAN_ONLY = {"Lahore", "Karachi Central", "Karachi East", "Karachi South", "Korangi"}


def _district_lines(pdf, pages, district):
    """Yield (pdf_page_1based, numbers, next_line) for lines that start with the district name."""
    rx = re.compile(r"^\s*" + re.escape(district) + r"\s+(" + NUM + r"(?:\s+" + NUM + r")*)\s*$", re.I)
    for i in pages:
        lines = (pdf.pages[i].extract_text() or "").split("\n")
        for j, line in enumerate(lines):
            m = rx.match(line)
            if m:
                yield i + 1, [float(x) for x in m.group(1).split()], (lines[j + 1] if j + 1 < len(lines) else "")


def _one(pdf, key, name):
    hits = list(_district_lines(pdf, PSLM_PAGES[key], name))
    assert len(hits) == 1, f"PSLM {key}: expected one row for {name}, found {len(hits)}"
    return hits[0]


def extract_pslm(districts: list[str]) -> pd.DataFrame:
    """District totals (and urban values where published) for four PSLM 2019-20 indicators.

    Numbers on a district row, by table:
      9.1  : [moderate-or-severe, MoE, severe, MoE]
      2.14a: [U m,f,t | R m,f,t | T m,f,t | rank | 2014-15]; a district with one stratum
             unsampled prints only that stratum's triple and the total triple.
      2.15 : [U m,f,t | R m,f,t | T m,f,t | rank]; an unsampled stratum prints 0 0 0.
      7.1  : [tap, hand pump, motor pump, dug well, tanker/bearer, filtration, other, total,
             rank, 2014-15], followed by an 'Urban' row with the same shares.
    """
    out = []
    with pdfplumber.open(PSLM_PDF) as pdf:
        for d in districts:
            name = "Karachi central" if d == "Karachi Central" else d
            rec = {"pslm_district": d}
            pg, v, _ = _one(pdf, "fies", d)
            rec.update(fies_mod_sev_pct=v[0], fies_mod_sev_moe=v[1], fies_severe_pct=v[2], fies_page=pg)
            pg, v, _ = _one(pdf, "literacy", d)
            if d in PSLM_RURAL_ONLY or d in PSLM_URBAN_ONLY:
                total, urban = v[5], (np.nan if d in PSLM_RURAL_ONLY else v[2])
            else:
                total, urban = v[8], v[2]
            rec.update(literacy10_total_pct=total, literacy10_urban_pct=urban, literacy_page=pg)
            pg, v, _ = _one(pdf, "out_of_school", d)
            rec.update(out_of_school_total_pct=v[8],
                       out_of_school_urban_pct=np.nan if d in PSLM_RURAL_ONLY else v[2], out_of_school_page=pg)
            pg, v, nxt = _one(pdf, "water", name)
            um = re.match(r"^\s*Urban\s+(" + NUM + r"(?:\s+" + NUM + r")*)\s*$", nxt)
            rec.update(tap_water_total_pct=v[0], tap_water_urban_pct=float(um.group(1).split()[0]) if um else np.nan,
                       water_page=pg)
            out.append(rec)
    df = pd.DataFrame(out)
    df["pslm_source"] = PSLM_PDF.name
    df["method"] = "pdf_text_extraction"
    return df


# ----------------------------------------------------------------------------- CPI
CPI_DIR = EXTERNAL_RAW / "cpi"


def _docx_tables(path: Path) -> list[list[list[str]]]:
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    tables = []
    for t in re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S):
        rows = []
        for r in re.findall(r"<w:tr[ >].*?</w:tr>", t, re.S):
            rows.append(["".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", c)) for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S)])
        tables.append(rows)
    return tables


def parse_cpi_urban_food() -> pd.DataFrame:
    """Monthly CPI (Urban) 'Food and non-alcoholic Beverages' index, base 2015-16 = 100.

    FY2024-25 and FY2025-26 from the cumulative group-wise PDF; Jul-Sep 2026 from the
    Monthly Review docx files. Months reported by two sources must agree.
    """
    recs = []
    with pdfplumber.open(CPI_DIR / "CpI-Urban-Groupwise-Cumulative-Indices.pdf") as pdf:
        for i, p in enumerate(pdf.pages):
            lines = (p.extract_text() or "").split("\n")
            m = re.search(r"July-June \((\d{4}) - (\d{4})\).*Base Year=2015-16", lines[0])
            if not m or int(m.group(1)) < 2024:
                continue
            y0 = int(m.group(1))
            for line in lines:
                if "Food and non-alcoholic Beverages" in line:
                    vals = [float(x) for x in line.split("Beverages")[1].split()][:12]
                    for k, val in enumerate(vals):
                        recs.append({"month": pd.Timestamp(y0 if k < 6 else y0 + 1, (k + 6) % 12 + 1, 1),
                                     "cpi_urban_food": val, "cpi_source": "CpI-Urban-Groupwise-Cumulative-Indices.pdf",
                                     "cpi_ref": f"pdf page {i + 1}, row 'Food and non-alcoholic Beverages'"})
    for fname, cur, prev in [("Monthly-Review-August-2026.docx", (2026, 8), (2026, 7)),
                             ("Monthly-Review-September2026.docx", (2026, 9), (2026, 8))]:
        for rows in _docx_tables(CPI_DIR / fname):
            if rows and rows[0] and "Consumer Price Index (Urban) by Group" in rows[0][0]:
                food = [r for r in rows if len(r) > 4 and r[1].startswith("Food &amp; Non-alcoholic")][0]
                for (y, mo), val in [(cur, food[3]), (prev, food[4])]:
                    recs.append({"month": pd.Timestamp(y, mo, 1), "cpi_urban_food": float(val), "cpi_source": fname,
                                 "cpi_ref": "Table 2 CPI (Urban) by Group, row 'Food & Non-alcoholic Bev.'"})
    df = pd.DataFrame(recs)
    spread = df.groupby("month")["cpi_urban_food"].agg(lambda s: s.max() - s.min())
    assert (spread <= 0.011).all(), f"CPI sources disagree: {spread[spread > 0.011]}"
    return df.sort_values(["month", "cpi_source"]).drop_duplicates("month").reset_index(drop=True)
