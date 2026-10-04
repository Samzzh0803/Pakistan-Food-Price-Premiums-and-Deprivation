"""Parse all three stacked blocks of PBS SPI Appendix-A into a tidy city-item-week panel.

Appendix-A stacks three tables that share the same 51 item rows:
  block 1: cities 01-07, block 2: cities 08-14,
  block 3: cities 15-17 followed by PBS's national summary columns.
Each table has a city-label row ("Name (NN)", each spanning MIN/AVG/MAX), a
sub-header row whose second cell is "DESCRIPTION", a numeric column-index row,
and a "PRICES ON dd-mm-yyyy" row. The earlier parser stopped after block 1.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from .config import ROOT, CITY_PROVINCE, N_CITIES, N_ITEMS, PARSER_VERSION, ROWS_PER_WEEK

CITY_RE = re.compile(r"^\s*([A-Za-z][A-Za-z\-\s\.]*?)\s*\((\d{2})\)\s*$")
DATE_RE = re.compile(r"PRICES\s+ON\s+(\d{1,2})[-./](\d{1,2})[-./](\d{4})", re.I)
FNAME_DATE_RES = [
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), (1, 2, 3)),
    (re.compile(r"(\d{2})\.(\d{2})\.(\d{4})"), (3, 2, 1)),
]

# Reviewed wording changes in PBS item labels (same serial, unit and price level).
# Item 42 was relabelled in the 2026-09-24 release; price 2566.5/MMBTU unchanged from 2026-08-27.
KNOWN_LABEL_VARIANTS = {42: {"Gas Charges for Q1", "Gas Charges upto 3.3719 MMBTU"}}

# National columns in block 3, in sheet order, with tidy names.
NATIONAL_COLS = [
    "nat_min", "nat_avg", "nat_max",
    "nat_avg_prev_week", "nat_avg_same_week_last_year",
    "pct_chg_prev_week", "pct_chg_same_week_last_year",
    "yearly_avg_recent", "yearly_avg_prior", "yearly_avg_diff", "yearly_avg_pct_chg",
]


def relpath(path: Path) -> str:
    """Repo-relative POSIX path, so outputs do not depend on where the repo is cloned."""
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_city_label(raw: str) -> tuple[str, str] | None:
    """Return (city, code) for a 'Name (NN)' label, repairing hyphenated or wrapped names."""
    text = re.sub(r"\s+", " ", str(raw).replace("\n", " ")).strip()
    m = CITY_RE.match(text)
    if not m:
        return None
    name = re.sub(r"-\s*", "", m.group(1)).replace(" ", "")  # "Rawal- pindi" -> "Rawalpindi"
    name = name[:1].upper() + name[1:].lower()
    return name, m.group(2)


def date_from_filename(path: Path) -> date | None:
    for rx, (y, mo, d) in FNAME_DATE_RES:
        m = rx.search(Path(path).name)
        if m:
            return date(int(m.group(y)), int(m.group(mo)), int(m.group(d)))
    return None


def _num(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        raise ValueError(f"non-numeric price token: {v!r}")


def parse_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse one Annexure workbook. Returns (city panel, national reference, metadata)."""
    path = Path(path)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Appendix-A"]
    rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
    wb.close()

    week_dates = set()
    for r in rows:
        for c in r:
            if isinstance(c, str):
                m = DATE_RE.search(c)
                if m:
                    week_dates.add(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
    if len(week_dates) != 1:
        raise ValueError(f"{path.name}: expected one 'PRICES ON' date, found {week_dates}")
    week_end = week_dates.pop()
    fname_date = date_from_filename(path)

    header_idx = [i for i, r in enumerate(rows)
                  if len(r) > 1 and isinstance(r[1], str) and r[1].strip().upper() == "DESCRIPTION"]

    records, national = [], []
    for b, h in enumerate(header_idx, start=1):
        label_row = rows[h - 1]
        cities, nat_start = [], None
        for col, cell in enumerate(label_row):
            if not isinstance(cell, str):
                continue
            parsed = clean_city_label(cell)
            if parsed:
                cities.append((col, *parsed))
            elif cell.strip().lower().startswith("national average") and nat_start is None:
                nat_start = col
        # Item rows: serial number in col 0 and a description string; skip the 1,2,3 index row.
        end = header_idx[b] - 1 if b < len(header_idx) else len(rows)
        for r in rows[h + 1:end]:
            if not (isinstance(r[0], (int, float)) and isinstance(r[1], str)):
                continue
            serial = int(r[0])
            if serial == 1 and str(r[1]).strip() == "2":
                continue
            label, unit = str(r[1]).strip(), str(r[2]).strip() if r[2] is not None else ""
            for col, city, code in cities:
                records.append({
                    "week_end": week_end, "block": b, "city": city, "city_code": code,
                    "item_id": serial, "item_label": label, "unit": unit,
                    "price_min": _num(r[col]), "price_avg": _num(r[col + 1]), "price_max": _num(r[col + 2]),
                })
            if nat_start is not None:
                vals = [_num(r[nat_start + k]) for k in range(len(NATIONAL_COLS))]
                national.append({"week_end": week_end, "item_id": serial, "item_label": label,
                                 "unit": unit, **dict(zip(NATIONAL_COLS, vals))})

    panel = pd.DataFrame(records)
    nat = pd.DataFrame(national)
    meta = {
        "source_file": relpath(path), "sha256": sha256_file(path), "week_end": week_end,
        "filename_date": fname_date, "filename_date_matches": fname_date == week_end,
        "n_blocks": len(header_idx), "parser_version": PARSER_VERSION,
    }
    panel["source_file"] = meta["source_file"]
    panel["sha256"] = meta["sha256"]
    panel["parser_version"] = PARSER_VERSION
    return panel, nat, meta


def geomean_quoting(avg: pd.Series) -> float:
    pos = avg[avg > 0]
    return float(np.exp(np.log(pos).mean())) if len(pos) else np.nan


def validate_week(panel: pd.DataFrame, nat: pd.DataFrame, meta: dict) -> dict:
    """Run the acceptance checks for one week. Raises AssertionError on failure."""
    wk = meta["week_end"]
    assert meta["n_blocks"] == 3, f"{wk}: {meta['n_blocks']} blocks"
    assert len(panel) == ROWS_PER_WEEK, f"{wk}: {len(panel)} rows, expected {ROWS_PER_WEEK}"
    assert panel["city"].nunique() == N_CITIES, f"{wk}: {panel['city'].nunique()} cities"
    assert set(panel["city"]) == set(CITY_PROVINCE), f"{wk}: unexpected cities {set(panel['city']) ^ set(CITY_PROVINCE)}"
    assert panel["item_id"].nunique() == N_ITEMS, f"{wk}: {panel['item_id'].nunique()} items"
    assert not panel.duplicated(["week_end", "city", "item_id"]).any(), f"{wk}: duplicate keys"
    assert panel[["price_min", "price_avg", "price_max"]].notna().all().all(), f"{wk}: blank prices"
    pos = panel[(panel[["price_min", "price_avg", "price_max"]] > 0).all(axis=1)]
    order_bad = pos[(pos.price_min > pos.price_avg + 1e-9) | (pos.price_avg > pos.price_max + 1e-9)]
    assert order_bad.empty, f"{wk}: MIN<=AVG<=MAX violated in {len(order_bad)} rows"
    zero = (panel[["price_min", "price_avg", "price_max"]] == 0)
    partial_zero = zero.any(axis=1) & ~zero.all(axis=1)
    assert not partial_zero.any(), f"{wk}: {partial_zero.sum()} rows with partial zeros"

    # Cross-check against PBS's own national columns.
    q = panel[panel.price_avg > 0]
    agg = q.groupby("item_id").agg(mn=("price_min", "min"), mx=("price_max", "max"))
    agg["gm"] = q.groupby("item_id")["price_avg"].apply(geomean_quoting)
    agg["am"] = q.groupby("item_id")["price_avg"].mean()
    agg["md"] = q.groupby("item_id")["price_avg"].median()
    chk = agg.join(nat.set_index("item_id")[["nat_min", "nat_avg", "nat_max"]])
    assert len(chk) == N_ITEMS and chk[["nat_min", "nat_avg", "nat_max"]].notna().all().all(), f"{wk}: national columns missing"
    min_ok = np.isclose(chk.mn, chk.nat_min, atol=0.006)
    max_ok = np.isclose(chk.mx, chk.nat_max, atol=0.006)
    gm_err = (chk.gm / chk.nat_avg - 1).abs()
    gm_ok = gm_err <= 0.001
    exceptions = chk[~(min_ok & max_ok & gm_ok)]
    assert exceptions.empty, f"{wk}: national check failed for items {list(exceptions.index)}"
    return {
        "week_end": wk, "items_checked": int(len(chk)), "min_max_pass": int((min_ok & max_ok).sum()),
        "gm_pass": int(gm_ok.sum()), "gm_max_abs_rel_err": float(gm_err.max()),
        "am_median_abs_rel_err": float((chk.am / chk.nat_avg - 1).abs().median()),
        "md_median_abs_rel_err": float((chk.md / chk.nat_avg - 1).abs().median()),
        "structural_zero_rows": int(zero.all(axis=1).sum()),
    }


def load_panel(dirs: list[Path], cutoff: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse and validate every workbook in `dirs` with week_end <= cutoff.

    Returns (panel, national reference, file inventory, per-week validation).
    Files with identical sha256 (or identical parsed content for the same week)
    are deduplicated; the first path in sorted order is kept and the rest recorded.
    """
    # Directory order is priority order: the first copy of a duplicated file is the one kept.
    files = []
    for d in dirs:
        if Path(d).exists():
            files += [p.resolve() for p in sorted(Path(d).glob("*.xlsx")) if p.resolve() not in files]
    inventory, kept_panels, kept_nat, checks = [], {}, {}, []
    seen_hash: dict[str, str] = {}
    for f in files:
        fdate = date_from_filename(f)
        if fdate is not None and fdate > cutoff:
            inventory.append({"source_file": relpath(f), "status": "skipped_after_cutoff_by_filename"})
            continue
        h = sha256_file(f)
        if h in seen_hash:
            inventory.append({"source_file": relpath(f), "sha256": h, "status": "duplicate_hash",
                              "duplicate_of": seen_hash[h]})
            continue
        panel, nat, meta = parse_workbook(f)
        wk = meta["week_end"]
        if wk > cutoff:
            inventory.append({"source_file": relpath(f), "sha256": h, "week_end": wk, "status": "skipped_after_cutoff"})
            continue
        if wk in kept_panels:
            prev = kept_panels[wk]
            cols = ["city", "item_id", "price_min", "price_avg", "price_max"]
            same = prev[cols].sort_values(cols[:2]).reset_index(drop=True).equals(
                panel[cols].sort_values(cols[:2]).reset_index(drop=True))
            inventory.append({"source_file": relpath(f), "sha256": h, "week_end": wk,
                              "status": "duplicate_content" if same else "CONFLICT_same_week_different_content",
                              "duplicate_of": prev["source_file"].iat[0]})
            if not same:
                raise AssertionError(f"Two different workbooks claim week {wk}: {f} vs {prev['source_file'].iat[0]}")
            continue
        seen_hash[h] = relpath(f)
        checks.append(validate_week(panel, nat, meta))
        kept_panels[wk], kept_nat[wk] = panel, nat
        inventory.append({"source_file": relpath(f), "sha256": h, "week_end": wk, "status": "kept",
                          "filename_date": meta["filename_date"], "filename_date_matches": meta["filename_date_matches"]})

    panel = pd.concat([kept_panels[k] for k in sorted(kept_panels)], ignore_index=True)
    nat = pd.concat([kept_nat[k] for k in sorted(kept_nat)], ignore_index=True)
    inv = pd.DataFrame(inventory)
    chk = pd.DataFrame(checks).sort_values("week_end").reset_index(drop=True)

    # Panel-level checks.
    n_weeks = panel["week_end"].nunique()
    assert len(panel) == ROWS_PER_WEEK * n_weeks, "row count != 867 x weeks"
    assert not panel.duplicated(["week_end", "city", "item_id"]).any()
    units = panel.groupby("item_id")["unit"].nunique()
    assert (units == 1).all(), f"item units vary across weeks: {units[units > 1]}"
    labels = panel.groupby("item_id")["item_label"].unique()
    for item_id, labs in labels.items():
        unexpected = set(labs) - KNOWN_LABEL_VARIANTS.get(item_id, set()) - {labs[0]}
        assert not unexpected, f"item {item_id} label changed to {unexpected}; review and add to KNOWN_LABEL_VARIANTS"
    # Canonical label = the label in the earliest week.
    first = panel.sort_values("week_end").drop_duplicates("item_id").set_index("item_id")["item_label"]
    panel["item_label_raw"] = panel["item_label"]
    panel["item_label"] = panel["item_id"].map(first)
    code = panel.groupby("city")["city_code"].nunique()
    assert (code == 1).all(), "a city maps to more than one code"
    assert panel["week_end"].max() <= cutoff, "holdout week leaked into panel"
    panel["province"] = panel["city"].map(CITY_PROVINCE)
    panel["week_end"] = pd.to_datetime(panel["week_end"])
    nat["week_end"] = pd.to_datetime(nat["week_end"])
    return panel, nat, inv, chk
