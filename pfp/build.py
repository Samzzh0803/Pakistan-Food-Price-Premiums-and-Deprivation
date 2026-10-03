"""Build the Milestone 02 master dataset (one row = city x item x week) and derived views."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import (CITY_PROVINCE, EXTERNAL, EXTERNAL_RAW, PROCESSED, RAW_ANNEX_DIRS, ROOT, TRAIN_CUTOFF)
from .external import (CROSSWALK, KARACHI_PSLM_TO_CENSUS, PSLM_TABLE_NAMES, extract_pslm, parse_census_table1,
                       parse_cpi_urban_food)
from .parse_annex import load_panel

ITEM_MAP = ROOT / "data" / "reference" / "item_mapping.csv"

EXTERNAL_URLS = {
    "PSLM_2019_20_District_Level.pdf": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/PSLM_2019_20_District_Level.pdf",
    "pslm_page.html": "https://www.pbs.gov.pk/pslm-3/",
    "census2023_result_excel.html": "https://www.pbs.gov.pk/result-excel/",
    "census2023/table_1_punjab_districts.xlsx": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/table_1_punjab_districts.xlsx",
    "census2023/table_1_sindh_districts.xlsx": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/table_1_sindh_districts.xlsx",
    "census2023/table_1_kp_districts.xlsx": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/table_1_kp_districts.xlsx",
    "census2023/table_1_balochistan_districts.xlsx": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/table_1_balochistan_districts.xlsx",
    "census2023/table_1_islamabad.xlsx": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/table_1_islamabad.xlsx",
    "census2023/table_1_national.xlsx": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/table_1_national.xlsx",
    "cpi/CpI-Urban-Groupwise-Cumulative-Indices.pdf": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/CpI-Urban-Groupwise-Cumulative-Indices.pdf",
    "cpi/Monthly-Review-August-2026.docx": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/Monthly-Review-August-2026.docx",
    "cpi/Monthly-Review-September2026.docx": "https://www.pbs.gov.pk/wp-content/uploads/2020/07/Monthly-Review-September2026.docx",
    "cpi/monthly_inflation_august_2026.html": "https://www.pbs.gov.pk/monthly-inflation-report-for-august-2026/",
    "cpi/monthly_inflation_sep_2026.html": "https://www.pbs.gov.pk/monthly-inflation-report-for-sep-2026/",
    "cpi/price_statistics_live.html": "https://www.pbs.gov.pk/price-statistics/",
}
EXTERNAL_REFS = {
    "PSLM_2019_20_District_Level.pdf": "Tables 2.14(a) pp.131-134, 2.15 pp.143-146, 7.1 pp.474-481, 9.1 pp.577-580 (printed page numbers)",
    "census2023/": "Table 1, district rows and their RURAL/URBAN sub-rows, column POPULATION-2023 ALL SEXES",
    "cpi/CpI-Urban-Groupwise-Cumulative-Indices.pdf": "pdf pp.28 and 31, row 'Food and non-alcoholic Beverages'",
    "cpi/Monthly-Review": "Table 2 CPI (Urban) by Group, row 'Food & Non-alcoholic Bev.'",
}


def write_external_manifest() -> pd.DataFrame:
    rows = []
    for rel, url in EXTERNAL_URLS.items():
        p = EXTERNAL_RAW / rel
        if not p.exists():
            continue
        ref = next((v for k, v in EXTERNAL_REFS.items() if rel.startswith(k)), "")
        rows.append({"local_path": p.relative_to(ROOT).as_posix(), "url": url,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size,
                     "file_mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                     "reference": ref})
    df = pd.DataFrame(rows)
    df.to_csv(EXTERNAL / "external_manifest.csv", index=False)
    return df


# Deprivation indicators: all oriented so that higher = more deprived.
DEPRIVATION = {
    "fies_mod_sev_pct": "Moderate or severe food insecurity (FIES), % of population (PSLM Table 9.1)",
    "illiteracy10_pct": "Population 10+ not literate, % (100 - PSLM Table 2.14(a) total)",
    "out_of_school_pct": "Children 5-16 out of school, % (PSLM Table 2.15 total)",
    "no_tap_water_pct": "Households without tap water as main drinking source, % (100 - PSLM Table 7.1 tap)",
}


def minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


def build_city_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """17-row city table with population (Census 2023) and deprivation (PSLM 2019-20)."""
    census = parse_census_table1().set_index("census_unit")
    districts = [d for v in CROSSWALK.values() for d in v[0]]
    pslm = extract_pslm(districts).set_index("pslm_district")
    ind_cols = ["fies_mod_sev_pct", "fies_severe_pct", "literacy10_total_pct", "literacy10_urban_pct",
                "out_of_school_total_pct", "out_of_school_urban_pct", "tap_water_total_pct", "tap_water_urban_pct"]
    rows, cross = [], []
    for city, (pslm_units, census_units) in CROSSWALK.items():
        cen = census.loc[census_units]
        rec = {"city": city, "province": CITY_PROVINCE[city], "pop_total": cen.pop_total.sum(),
               "pop_urban": cen.pop_urban.sum()}
        if len(pslm_units) == 1:
            vals = pslm.loc[pslm_units[0], ind_cols]
            rule = "single district"
        else:  # Karachi: population-weighted mean of PSLM districts, weights from Census 2023 totals
            w = pd.Series({d: census.loc[KARACHI_PSLM_TO_CENSUS[d], "pop_total"].sum() for d in pslm_units})
            sub = pslm.loc[pslm_units, ind_cols]
            vals = sub.apply(lambda col: np.average(col.dropna(), weights=w[col.dropna().index]) if col.notna().any() else np.nan)
            rule = "population-weighted mean of 6 PSLM districts (Census 2023 weights; Keamari added to Karachi West)"
        rec.update(vals.to_dict())
        rec["pslm_rule"] = rule
        rows.append(rec)
        notes = {
            "Karachi": "Seven census districts form one SPI market; PSLM 2019-20 reports six (Keamari created later).",
            "Islamabad": "Islamabad and Rawalpindi are one contiguous market but two administrative units; kept separate, each with its own district values.",
            "Rawalpindi": "See Islamabad.",
            "Khuzdar": "PSLM 2019-20 has no urban sample for Khuzdar: urban indicators are missing, district totals used.",
            "Bannu": "Census 2023 urban share of Bannu district is 3.6%; district totals describe a mostly rural district.",
            "Lahore": "District is 100% urban in Census 2023.",
        }
        cross.append({"city": city, "city_code": None, "province": CITY_PROVINCE[city],
                      "pslm_districts": "; ".join(pslm_units), "census_units": "; ".join(census_units),
                      "pslm_rule": rule, "population_rule": "sum of census units (total and urban)",
                      "notes": notes.get(city, "")})
    city = pd.DataFrame(rows)
    city["illiteracy10_pct"] = 100 - city["literacy10_total_pct"]
    city["out_of_school_pct"] = city["out_of_school_total_pct"]
    city["no_tap_water_pct"] = 100 - city["tap_water_total_pct"]
    city["illiteracy10_urban_pct"] = 100 - city["literacy10_urban_pct"]
    city["no_tap_water_urban_pct"] = 100 - city["tap_water_urban_pct"]
    for c in DEPRIVATION:
        city[c + "_mm"] = minmax(city[c])
    city["deprivation_composite"] = city[[c + "_mm" for c in DEPRIVATION]].mean(axis=1)
    # Sensitivity variant without the water indicator (tap water partly reflects infrastructure type).
    city["deprivation_composite_nowater"] = city[[c + "_mm" for c in list(DEPRIVATION)[:3]]].mean(axis=1)
    city["log_pop_urban"] = np.log(city["pop_urban"])
    city["log_pop_total"] = np.log(city["pop_total"])
    return city, pd.DataFrame(cross)


def build_master(panel: pd.DataFrame, items: pd.DataFrame, city: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    n0 = len(df)
    pc = ["price_min", "price_avg", "price_max"]
    df["is_structural_missing"] = (df[pc] == 0).all(axis=1)
    df.loc[df["is_structural_missing"], pc] = np.nan

    # Item classification (m:1 on item_id).
    df = df.merge(items[["item_id", "item_short", "is_food", "item_category"]], on="item_id", how="left", validate="m:1")
    assert len(df) == n0 and df["item_category"].notna().all()

    # Within-week derived measures.
    g = df.groupby(["week_end", "item_id"])["price_avg"]
    df["national_gm"] = g.transform(lambda s: np.exp(np.log(s.dropna()).mean()))
    df["rel_price"] = np.log(df["price_avg"] / df["national_gm"])
    df["cross_city_median"] = g.transform("median")
    df["above_median"] = np.where(df["price_avg"].isna(), np.nan, (df["price_avg"] > df["cross_city_median"]).astype(float))
    df["within_city_range_pct"] = 100 * (df["price_max"] - df["price_min"]) / df["price_avg"]
    df["single_quote"] = np.where(df["price_avg"].isna(), np.nan, (df["price_min"] == df["price_max"]).astype(float))
    df["log_price"] = np.log(df["price_avg"])
    df["price_z_item_week"] = g.transform(lambda s: (s - s.mean()) / s.std(ddof=1))

    # Temporal derived measures, only across consecutive weeks (6-8 days apart).
    df = df.sort_values(["city", "item_id", "week_end"]).reset_index(drop=True)
    gi = df.groupby(["city", "item_id"])
    prev_week = gi["week_end"].shift(1)
    df["days_since_prev"] = (df["week_end"] - prev_week).dt.days
    df["is_consecutive"] = df["days_since_prev"].between(6, 8)
    dlog = df["log_price"] - gi["log_price"].shift(1)
    df["dlog_price"] = dlog.where(df["is_consecutive"])
    df["price_changed"] = np.where(df["dlog_price"].isna(), np.nan, (df["dlog_price"].abs() > 1e-12).astype(float))
    df["dlog_price_lag1"] = gi["dlog_price"].shift(1).where(df["is_consecutive"])
    df["above_median_prev"] = gi["above_median"].shift(1).where(df["is_consecutive"])

    # Date parts.
    df["year"] = df["week_end"].dt.year
    df["month"] = df["week_end"].dt.month
    df["iso_week"] = df["week_end"].dt.isocalendar().week.astype(int)

    # City attributes (m:1 on city) and CPI (m:1 on month).
    keep = ["city", "pop_total", "pop_urban", "log_pop_urban", "fies_mod_sev_pct", "illiteracy10_pct",
            "out_of_school_pct", "no_tap_water_pct", "deprivation_composite", "deprivation_composite_nowater"]
    df = df.merge(city[keep], on="city", how="left", validate="m:1")
    assert len(df) == n0
    df["month_start"] = df["week_end"].dt.to_period("M").dt.to_timestamp()
    df = df.merge(cpi[["month", "cpi_urban_food"]].rename(columns={"month": "month_start"}), on="month_start",
                  how="left", validate="m:1")
    assert len(df) == n0 and df["cpi_urban_food"].notna().all(), "CPI missing for a panel month"
    df["real_price_avg"] = df["price_avg"] * 100 / df["cpi_urban_food"]  # PKR at 2015-16 urban food prices
    df = df.drop(columns=["month_start"])

    order = ["week_end", "year", "month", "iso_week", "city", "city_code", "province", "item_id", "item_label",
             "item_short", "unit", "is_food", "item_category", "price_min", "price_avg", "price_max",
             "is_structural_missing", "log_price", "national_gm", "rel_price", "cross_city_median", "above_median",
             "above_median_prev", "within_city_range_pct", "single_quote", "price_z_item_week", "days_since_prev",
             "is_consecutive", "dlog_price", "price_changed", "dlog_price_lag1", "pop_total", "pop_urban",
             "log_pop_urban", "fies_mod_sev_pct", "illiteracy10_pct", "out_of_school_pct", "no_tap_water_pct",
             "deprivation_composite", "deprivation_composite_nowater", "cpi_urban_food", "real_price_avg",
             "block", "source_file", "sha256", "parser_version"]
    df = df[order].sort_values(["week_end", "city_code", "item_id"]).reset_index(drop=True)
    return df


def build_all(write: bool = True) -> dict:
    panel, nat, inv, checks = load_panel(RAW_ANNEX_DIRS, TRAIN_CUTOFF)
    items = pd.read_csv(ITEM_MAP)
    city, cross = build_city_table()
    codes = panel.drop_duplicates("city").set_index("city")["city_code"]
    city.insert(1, "city_code", city["city"].map(codes))
    cross["city_code"] = cross["city"].map(codes)
    cpi = parse_cpi_urban_food()
    master = build_master(panel, items, city, cpi)

    food = master[master.is_food == 1]
    city_item = food.groupby(["city", "item_id"])["rel_price"].mean().unstack("item_id")
    disp = (master.groupby(["week_end", "item_id", "item_short", "is_food", "item_category"])
            .agg(n_quoting=("price_avg", "count"), mean_price=("price_avg", "mean"), sd_price=("price_avg", "std"),
                 sd_rel_price=("rel_price", "std"), min_price=("price_avg", "min"), max_price=("price_avg", "max"))
            .reset_index())
    disp["cv"] = disp["sd_price"] / disp["mean_price"]
    disp["max_min_ratio"] = disp["max_price"] / disp["min_price"]

    out = {"panel": panel, "national": nat, "inventory": inv, "checks": checks, "items": items, "city": city,
           "crosswalk": cross, "cpi": cpi, "master": master, "city_item": city_item, "dispersion": disp}
    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        master.to_csv(PROCESSED / "master_city_item_week.csv", index=False)
        master.to_parquet(PROCESSED / "master_city_item_week.parquet", index=False)
        nat.to_csv(PROCESSED / "pbs_national_reference.csv", index=False)
        inv.to_csv(PROCESSED / "annex_file_inventory.csv", index=False)
        checks.to_csv(PROCESSED / "acceptance_checks_by_week.csv", index=False)
        city.to_csv(PROCESSED / "city_table.csv", index=False)
        city.to_parquet(PROCESSED / "city_table.parquet", index=False)
        cross.to_csv(EXTERNAL / "city_district_crosswalk.csv", index=False)
        cpi.to_csv(EXTERNAL / "cpi_urban_food_monthly.csv", index=False)
        city_item.to_csv(PROCESSED / "city_by_food_item_mean_rel_price.csv")
        city_item.reset_index().to_parquet(PROCESSED / "city_by_food_item_mean_rel_price.parquet", index=False)
        disp.to_csv(PROCESSED / "item_week_dispersion.csv", index=False)
        disp.to_parquet(PROCESSED / "item_week_dispersion.parquet", index=False)
        write_external_manifest()
    return out
