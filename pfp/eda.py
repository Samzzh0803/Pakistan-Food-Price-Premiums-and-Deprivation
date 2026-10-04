"""Milestone 02 EDA: course techniques (Units 02-07) and exploratory later-syllabus methods.

Every function takes the built objects, returns tables, and records headline numbers in
the shared dict `N`, which `save_numbers` writes to reports/m02/m02_numbers.json. The
manuscript reads its statistics only from that file.
"""
from __future__ import annotations

import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .config import (CITY_PROVINCE, FIGURES, NUMBERS_JSON, PANEL_START, PROVINCE_COLORS, TRAIN_CUTOFF)

N: dict = {}
SEED = 20261004
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
CAT_ORDER = ["perishable", "storable_staple", "branded_packaged", "prepared_food"]
CAT_LABEL = {"perishable": "Perishable", "storable_staple": "Storable staple", "branded_packaged": "Branded/packaged",
             "prepared_food": "Prepared food", "administered_utility": "Administered/utility", "other_nonfood": "Other non-food"}
CAT_COLORS = dict(zip(CAT_ORDER, ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]))
PROV_ORDER = ["ICT", "Punjab", "Sindh", "KP", "Balochistan"]


def _r(x, k=4):
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), k)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def put(path: str, value):
    """Store a number in N under a dotted path."""
    d = N
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = _r(value)


def save_numbers():
    NUMBERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_JSON.write_text(json.dumps(N, indent=1, default=str, sort_keys=True), encoding="utf-8")


def style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False})


def savefig(fig, name):
    """Save a figure. Figure numbers are assigned by the manuscript, so any 'Fig. N.' prefix is dropped."""
    import re
    for ax in fig.axes:
        ax.set_title(re.sub(r"^Fig\. \d+\. ", "", ax.get_title()))
    if fig._suptitle is not None:
        fig._suptitle.set_text(re.sub(r"^Fig\. \d+\. ", "", fig._suptitle.get_text()))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return FIGURES / f"{name}.png"


def food_rows(master: pd.DataFrame) -> pd.DataFrame:
    return master[(master.is_food == 1) & master.price_avg.notna()].copy()


def city_order(master):
    return master.drop_duplicates("city").sort_values("city_code")["city"].tolist()


# ============================================================================ inventory
def inventory(o: dict) -> pd.DataFrame:
    m = o["master"]
    weeks = np.sort(m.week_end.unique())
    gaps = np.diff(weeks).astype("timedelta64[D]").astype(int)
    put("panel.rows", len(m)); put("panel.cols", m.shape[1])
    put("panel.weeks", len(weeks)); put("panel.first_week", str(pd.Timestamp(weeks[0]).date()))
    put("panel.last_week", str(pd.Timestamp(weeks[-1]).date()))
    put("panel.cities", m.city.nunique()); put("panel.items", m.item_id.nunique())
    put("panel.food_items", int(o["items"].is_food.sum()))
    put("panel.consecutive_pairs", int(((gaps >= 6) & (gaps <= 8)).sum()))
    put("panel.pairs_7d", int((gaps == 7).sum())); put("panel.pairs_6d", int((gaps == 6).sum()))
    put("panel.gap_min_nonconsec", int(gaps[gaps > 8].min())); put("panel.gap_max", int(gaps.max()))
    wd = pd.Series(pd.to_datetime(weeks)).dt.day_name().value_counts()
    put("panel.thursdays", int(wd.get("Thursday", 0))); put("panel.wednesdays", int(wd.get("Wednesday", 0)))
    put("panel.structural_missing_rows", int(m.is_structural_missing.sum()))
    put("panel.structural_missing_pct", 100 * m.is_structural_missing.mean())
    put("panel.train_cutoff", str(TRAIN_CUTOFF))
    chk = o["checks"]
    put("validation.item_weeks_checked", int(chk.items_checked.sum()))
    put("validation.min_max_pass", int(chk.min_max_pass.sum())); put("validation.gm_pass", int(chk.gm_pass.sum()))
    put("validation.gm_max_abs_rel_err_pct", 100 * chk.gm_max_abs_rel_err.max())
    put("validation.am_median_abs_rel_err_pct", 100 * chk.am_median_abs_rel_err.median())
    put("validation.md_median_abs_rel_err_pct", 100 * chk.md_median_abs_rel_err.median())
    inv = o["inventory"]
    put("files.kept", int((inv.status == "kept").sum())); put("files.duplicates", int(inv.status.str.startswith("duplicate").sum()))
    put("files.raw_annex_files", len(inv))
    prov = m.drop_duplicates("city").province.value_counts()
    for p_, n_ in prov.items():
        put(f"panel.cities_by_province.{p_}", int(n_))
    return pd.DataFrame({"week_end": pd.to_datetime(weeks), "gap_days": np.r_[np.nan, gaps]})


def backfill_summary(manifest: pd.DataFrame) -> dict:
    thursdays = manifest.thursday.nunique()
    got = manifest[manifest.outcome == "downloaded"]
    put("backfill.thursdays_attempted", thursdays)
    put("backfill.pages_requested", len(manifest))
    put("backfill.weeks_found", got.thursday.nunique())
    put("backfill.weeks_missing", thursdays - got.thursday.nunique())
    put("backfill.train_found", int((got.split == "train").sum()))
    put("backfill.holdout_found", int((got.split == "holdout").sum()))
    put("backfill.holdout_weeks", ", ".join(got[got.split == "holdout"].candidate_date))
    return N["backfill"]


def external_summary(o: dict) -> pd.DataFrame:
    """Facts about the merged external sources, for the inventory table."""
    m, city, cpi = o["master"], o["city"], o["cpi"]
    months = m.week_end.dt.to_period("M").dt.to_timestamp().drop_duplicates().sort_values()
    c = cpi.set_index("month").loc[months, "cpi_urban_food"]
    put("external.cpi_months_in_panel", len(months))
    put("external.cpi_first_month", str(months.iloc[0].date())[:7]); put("external.cpi_last_month", str(months.iloc[-1].date())[:7])
    put("external.cpi_first", c.iloc[0]); put("external.cpi_last", c.iloc[-1])
    put("external.cpi_change_pct", 100 * (c.iloc[-1] / c.iloc[0] - 1))
    put("external.pslm_districts_used", sum(len(x.split("; ")) for x in o["crosswalk"].pslm_districts))
    put("external.census_units_used", sum(len(x.split("; ")) for x in o["crosswalk"].census_units))
    put("external.karachi_pslm_districts", 6); put("external.karachi_census_districts", 7)
    put("external.pop_urban_17_cities", city.pop_urban.sum())
    put("external.deprivation_indicators", 4)
    for col in ["fies_mod_sev_pct", "illiteracy10_pct", "out_of_school_pct", "no_tap_water_pct", "deprivation_composite"]:
        put(f"external.range.{col}.min", city[col].min()); put(f"external.range.{col}.max", city[col].max())
        put(f"external.range.{col}.min_city", city.loc[city[col].idxmin(), "city"])
        put(f"external.range.{col}.max_city", city.loc[city[col].idxmax(), "city"])
    return cpi


# ============================================================================ Unit 02
def codebook(m: pd.DataFrame) -> pd.DataFrame:
    types = {
        "week_end": ("interval", "differences in days; order; no true zero"),
        "year": ("interval", "order, differences"), "month": ("ordinal (cyclic)", "order within year; mode"),
        "iso_week": ("ordinal (cyclic)", "order within year"),
        "city": ("nominal", "equality, counts, mode"), "city_code": ("nominal", "equality only (stored as text)"),
        "province": ("nominal", "equality, counts"), "item_id": ("nominal", "equality only (number is a PBS serial)"),
        "item_label": ("nominal", "equality"), "item_short": ("nominal", "equality"), "unit": ("nominal", "equality"),
        "is_food": ("binary (nominal)", "counts, proportions"), "item_category": ("nominal", "counts, mode"),
        "price_min": ("ratio", "all arithmetic; ratios meaningful within an item"),
        "price_avg": ("ratio", "all arithmetic; ratios meaningful within an item"),
        "price_max": ("ratio", "all arithmetic; ratios meaningful within an item"),
        "is_structural_missing": ("binary", "counts"), "log_price": ("interval", "differences = log ratios"),
        "national_gm": ("ratio", "all arithmetic within an item-week"),
        "rel_price": ("interval", "differences, means; ratios of rel_price are meaningless"),
        "cross_city_median": ("ratio", "within an item-week"), "above_median": ("binary", "proportions"),
        "above_median_prev": ("binary", "proportions"), "within_city_range_pct": ("ratio", "all arithmetic"),
        "single_quote": ("binary", "proportions"), "price_z_item_week": ("interval", "differences, means"),
        "days_since_prev": ("ratio", "all arithmetic"), "is_consecutive": ("binary", "counts"),
        "dlog_price": ("interval", "differences, means"), "price_changed": ("binary", "proportions"),
        "dlog_price_lag1": ("interval", "differences, means"), "pop_total": ("ratio", "all arithmetic"),
        "pop_urban": ("ratio", "all arithmetic"), "log_pop_urban": ("interval", "differences"),
        "fies_mod_sev_pct": ("ratio", "all arithmetic"), "illiteracy10_pct": ("ratio", "all arithmetic"),
        "out_of_school_pct": ("ratio", "all arithmetic"), "no_tap_water_pct": ("ratio", "all arithmetic"),
        "deprivation_composite": ("interval (0-1 scaled)", "order, differences; zero is the sample minimum"),
        "deprivation_composite_nowater": ("interval (0-1 scaled)", "order, differences"),
        "cpi_urban_food": ("ratio (index)", "ratios between months"), "real_price_avg": ("ratio", "all arithmetic"),
        "block": ("nominal", "provenance only"), "source_file": ("nominal", "provenance only"),
        "sha256": ("nominal", "provenance only"), "parser_version": ("nominal", "provenance only"),
    }
    rows = []
    for c in m.columns:
        t, ops = types.get(c, ("?", "?"))
        rows.append({"column": c, "dtype": str(m[c].dtype), "attribute_type": t, "meaningful_operations": ops,
                     "n_missing": int(m[c].isna().sum()), "pct_missing": round(100 * m[c].isna().mean(), 2)})
    cb = pd.DataFrame(rows)
    put("codebook.columns", len(cb))
    return cb


# ============================================================================ Unit 03
def tidy_demo(o: dict, path) -> dict:
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["Appendix-A"]
    rows = list(ws.iter_rows(values_only=True))
    raw_r, raw_c = len(rows), max(len(r) for r in rows)
    one_week = o["master"][o["master"].week_end == o["master"].week_end.max()]
    put("tidy.raw_rows", raw_r); put("tidy.raw_cols", raw_c); put("tidy.raw_blocks", 3)
    put("tidy.raw_value_cells", 51 * 17 * 3)
    put("tidy.tidy_rows_per_week", len(one_week)); put("tidy.tidy_cols_core", 10)
    return {"raw_shape": (raw_r, raw_c), "tidy_shape_per_week": (len(one_week), 10)}


def cleaning_log(o: dict) -> pd.DataFrame:
    panel, inv = o["panel"], o["inventory"]
    relabel = int((panel["item_label_raw"] != panel["item_label"]).sum())
    rows = [
        ("City labels with hyphen or line break repaired", 0, "Repair rule in parser; none needed in the 27 training files"),
        ("Zero MIN/AVG/MAX triples set to missing", int(o["master"].is_structural_missing.sum()),
         "Structural non-quotes; PBS excludes them from its national mean"),
        ("Week dates read from 'PRICES ON' text", int(inv[inv.status == "kept"].shape[0]),
         "All match the date in the file name" if bool(inv[inv.status == "kept"].filename_date_matches.all()) else "MISMATCH"),
        ("Item label wording changes mapped to first label", relabel, "Item 42 renamed in 2026-09-24 release, unit and price unchanged"),
        ("Duplicate workbooks dropped (identical sha256)", int(inv.status.str.startswith("duplicate").sum()),
         "Earlier repo copy kept; re-downloads byte-identical"),
        ("Rows deleted", 0, "Nothing deleted; missing values are flagged, not dropped"),
    ]
    put("cleaning.relabelled_rows", relabel)
    return pd.DataFrame(rows, columns=["operation", "rows_or_files", "note"])


def missing_table(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sm = m.is_structural_missing
    lag_missing = m.dlog_price.isna() & ~sm
    kinds = pd.DataFrame([
        {"kind": "Structural non-quote (PBS printed 0/0/0)", "rows": int(sm.sum()),
         "options_considered": "remove; impute from other cities (internal); constant",
         "choice": "set to missing, flag, do not impute",
         "reason": "Imputing invents a market price PBS does not record; PBS itself excludes these cities from its national mean"},
        {"kind": "Weekly change across a non-consecutive gap", "rows": int(lag_missing.sum()),
         "options_considered": "interpolate weeks; impute from CPI (external); leave missing",
         "choice": "leave missing",
         "reason": "Interpolation would manufacture week-to-week changes nobody observed"},
        {"kind": "Weeks absent from the PBS site", "rows": None,
         "options_considered": "impute whole weeks; restrict to observed weeks",
         "choice": "treat as a coverage gap",
         "reason": "No source to impute from; stated as a limitation, not filled"},
    ])
    put("missing.structural_rows", int(sm.sum()))
    put("missing.lag_missing_rows", int(lag_missing.sum()))
    put("missing.lag_missing_pct", 100 * lag_missing.mean())
    by_col = m.isna().sum().rename("n_missing").to_frame()
    by_col["pct"] = 100 * by_col.n_missing / len(m)
    return kinds, by_col[by_col.n_missing > 0]


def integration_table(o: dict) -> pd.DataFrame:
    m = o["master"]
    rows = [
        ("stack", "3 Appendix-A blocks per workbook", "same 51 item rows", "rows: 7+7+3 cities x 51 = 867 per week", 867),
        ("stack", f"{m.week_end.nunique()} weekly workbooks", "week_end", f"867 x {m.week_end.nunique()}", len(m)),
        ("merge", "item_mapping.csv", "item_id", "many-to-one (51 items)", len(m)),
        ("merge", "city table (Census 2023 + PSLM 2019-20)", "city", "many-to-one (17 cities)", len(m)),
        ("merge", "urban food CPI", "calendar month of week_end", "many-to-one (months)", len(m)),
    ]
    put("integration.rows_after_all_joins", len(m))
    return pd.DataFrame(rows, columns=["operation", "source", "key", "cardinality", "rows_after"])


# ============================================================================ Unit 04
def aggregation(m: pd.DataFrame) -> pd.DataFrame:
    f = food_rows(m)
    city_prem = f.groupby(["item_id", "city", "province"])["rel_price"].mean().reset_index()
    sd_city = city_prem.groupby("item_id")["rel_price"].std().mean()
    prov_prem = city_prem.groupby(["item_id", "province"])["rel_price"].mean().reset_index()
    sd_prov = prov_prem.groupby("item_id")["rel_price"].std().mean()
    within = city_prem.groupby(["item_id", "province"])["rel_price"].agg(lambda s: s.max() - s.min())
    punjab_range = within.xs("Punjab", level="province").mean()
    # Week -> month: SD of item-level national mean log price changes, weekly vs monthly.
    wk = f.groupby(["week_end"])["rel_price"].std()
    month = f.assign(m_=f.week_end.dt.to_period("M")).groupby(["m_", "city", "item_id"])["rel_price"].mean()
    sd_week_level = f["rel_price"].std()
    sd_month_level = month.std()
    city_mean = f.groupby("city")["rel_price"].mean()
    put("aggregation.mean_sd_across_cities", sd_city)
    put("aggregation.mean_sd_across_provinces", sd_prov)
    put("aggregation.punjab_mean_within_range_logpts", punjab_range)
    put("aggregation.sd_rel_price_city_item_week", sd_week_level)
    put("aggregation.sd_rel_price_city_item_month", sd_month_level)
    put("aggregation.sd_city_overall_premium", city_mean.std())
    return pd.DataFrame({"level": ["city x item x week", "city x item x month", "city x item (mean over weeks)",
                                   "province x item (mean of cities)", "national (reference)"],
                         "sd_rel_price": [sd_week_level, sd_month_level, city_prem.rel_price.std(),
                                          prov_prem.rel_price.std(), 0.0]})


def splitting(m: pd.DataFrame) -> pd.DataFrame:
    s = m.drop_duplicates("week_end")[["week_end", "year", "month", "iso_week"]].sort_values("week_end")
    put("splitting.years", sorted(s.year.unique().tolist()).__str__())
    put("splitting.months_covered", int(s.assign(ym=s.year * 100 + s.month).ym.nunique()))
    return s


def city_premiums(m: pd.DataFrame) -> pd.DataFrame:
    """City x food item mean rel_price; then per-city summaries and t-intervals (n = items)."""
    f = food_rows(m)
    ci = f.groupby(["city", "item_id"])["rel_price"].mean().reset_index()
    out = []
    for c, g in ci.groupby("city"):
        x = g.rel_price.values
        n = len(x)
        mean, sd = x.mean(), x.std(ddof=1)
        se = sd / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        t, p = stats.ttest_1samp(x, 0.0)
        out.append({"city": c, "province": CITY_PROVINCE[c], "n_items": n, "mean": mean, "median": np.median(x),
                    "sd": sd, "se": se, "ci_lo": mean - tcrit * se, "ci_hi": mean + tcrit * se, "t": t, "p": p,
                    "skew": stats.skew(x)})
    df = pd.DataFrame(out)
    df["pct_mean"] = 100 * (np.exp(df["mean"]) - 1)
    df["pct_median"] = 100 * (np.exp(df["median"]) - 1)
    df["bonferroni_sig"] = df.p < 0.05 / len(df)
    df["sig_05"] = df.p < 0.05
    df = df.sort_values("mean", ascending=False).reset_index(drop=True)
    put("premium.n_sig_05", int(df.sig_05.sum())); put("premium.n_sig_bonferroni", int(df.bonferroni_sig.sum()))
    put("premium.bonferroni_alpha", 0.05 / len(df))
    for r in df.itertuples():
        put(f"premium.city.{r.city}.pct_mean", r.pct_mean); put(f"premium.city.{r.city}.pct_median", r.pct_median)
        put(f"premium.city.{r.city}.ci_lo_pct", 100 * (np.exp(r.ci_lo) - 1))
        put(f"premium.city.{r.city}.ci_hi_pct", 100 * (np.exp(r.ci_hi) - 1))
        put(f"premium.city.{r.city}.p", r.p); put(f"premium.city.{r.city}.n_items", r.n_items)
    put("premium.top_city", df.iloc[0].city); put("premium.bottom_city", df.iloc[-1].city)
    diff = (df.pct_mean - df.pct_median).abs()
    put("premium.max_mean_median_gap_city", df.loc[diff.idxmax(), "city"])
    put("premium.max_mean_median_gap_pts", diff.max())
    return df


def bootstrap_median(m: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    f = food_rows(m)
    ci = f.groupby(["city", "item_id"])["rel_price"].mean().unstack("item_id")
    rng = np.random.default_rng(SEED)
    out = []
    for c in ci.index:
        x = ci.loc[c].dropna().values
        boots = np.median(rng.choice(x, size=(reps, len(x)), replace=True), axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"city": c, "median": np.median(x), "boot_lo": lo, "boot_hi": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0)})
    df = pd.DataFrame(out)
    put("bootstrap.reps", reps); put("bootstrap.n_cities_excluding_zero", int(df.excludes_zero.sum()))
    return df


def abnormal_labels(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-item thresholds for 'abnormal next-week movement', fitted on training weeks only.

    Rule as contracted: |dlog| above the item's own 90th percentile of |dlog|.
    Refinement: the 90th percentile among non-zero changes, because most changes are exactly 0.
    """
    f = food_rows(m).sort_values(["city", "item_id", "week_end"])
    assert f.week_end.max() <= pd.Timestamp(TRAIN_CUTOFF)
    g = f.groupby(["city", "item_id"])
    f["dlog_next"] = g["dlog_price"].shift(-1)
    ch = f[f.dlog_price.notna()]
    thr = ch.groupby("item_id")["dlog_price"].agg(
        p90_all=lambda s: np.percentile(s.abs(), 90),
        p90_changed=lambda s: np.percentile(s.abs()[s.abs() > 1e-12], 90) if (s.abs() > 1e-12).sum() >= 5 else np.nan,
        share_zero=lambda s: (s.abs() <= 1e-12).mean(), n=lambda s: len(s))
    f = f.merge(thr[["p90_all", "p90_changed"]], left_on="item_id", right_index=True, how="left")
    f["abnormal_next_contract"] = np.where(f.dlog_next.isna(), np.nan, (f.dlog_next.abs() > f.p90_all).astype(float))
    f["abnormal_next_refined"] = np.where(f.dlog_next.isna() | f.p90_changed.isna(), np.nan,
                                          (f.dlog_next.abs() > f.p90_changed).astype(float))
    put("h3.items_p90_zero", int((thr.p90_all <= 1e-12).sum()))
    put("h3.items_too_few_changes", int(thr.p90_changed.isna().sum()))
    put("h3.base_rate_contract", f.abnormal_next_contract.mean())
    put("h3.base_rate_refined", f.abnormal_next_refined.mean())
    put("h3.labelled_rows_contract", int(f.abnormal_next_contract.notna().sum()))
    put("h3.labelled_rows_refined", int(f.abnormal_next_refined.notna().sum()))
    return f, thr


def feature_selection(m: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    f = food_rows(m)
    lp = np.log(f[["price_min", "price_avg", "price_max"]])
    corr = lp.corr()
    put("feature_sel.corr_logmin_logavg", corr.loc["price_min", "price_avg"])
    put("feature_sel.corr_logmax_logavg", corr.loc["price_max", "price_avg"])
    d = labelled[labelled.abnormal_next_refined.notna()].copy()
    d["abs_dlog"] = d.dlog_price.abs()
    d["abs_dlog_lag1"] = d.dlog_price_lag1.abs()
    d["abs_rel_price"] = d.rel_price.abs()
    d["is_perishable"] = (d.item_category == "perishable").astype(float)
    cands = ["abs_dlog", "abs_dlog_lag1", "price_changed", "rel_price", "abs_rel_price", "within_city_range_pct",
             "single_quote", "is_perishable", "deprivation_composite", "log_pop_urban"]
    rows = []
    for c in cands:
        a = d.loc[d.abnormal_next_refined == 1, c].dropna()
        b = d.loc[d.abnormal_next_refined == 0, c].dropna()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        rows.append({"feature": c, "mean_abnormal": a.mean(), "mean_normal": b.mean(), "n_abnormal": len(a),
                     "n_normal": len(b), "score": abs(a.mean() - b.mean()) / se if se > 0 else np.nan})
    fs = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    put("feature_sel.top_feature", fs.iloc[0].feature); put("feature_sel.top_score", fs.iloc[0].score)
    return fs


def discretisation(city: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["pop_urban", "deprivation_composite"]:
        ew = pd.cut(city[col], 3, labels=["low", "mid", "high"]).value_counts().reindex(["low", "mid", "high"])
        ef = pd.qcut(city[col], 3, labels=["low", "mid", "high"]).value_counts().reindex(["low", "mid", "high"])
        rows.append({"variable": col, "method": "equal-width", **ew.to_dict()})
        rows.append({"variable": col, "method": "equal-frequency (terciles)", **ef.to_dict()})
        put(f"discretisation.{col}.equal_width", "/".join(map(str, ew.values)))
        put(f"discretisation.{col}.equal_freq", "/".join(map(str, ef.values)))
    return pd.DataFrame(rows)


def add_terciles(city: pd.DataFrame, prem: pd.DataFrame) -> pd.DataFrame:
    c = city.merge(prem[["city", "mean", "median"]].rename(columns={"mean": "premium", "median": "premium_median"}), on="city")
    c["pop_tercile"] = pd.qcut(c.pop_urban, 3, labels=["small", "medium", "large"])
    c["deprivation_tercile"] = pd.qcut(c.deprivation_composite, 3, labels=["low", "mid", "high"])
    c["premium_tercile"] = pd.qcut(c.premium, 3, labels=["low", "mid", "high"])
    return c


def normalisation_demo(m: pd.DataFrame, city: pd.DataFrame) -> pd.DataFrame:
    wf = m[(m.item_id == 1) & (m.week_end == m.week_end.max())][["city", "price_avg", "price_z_item_week"]].head(5)
    put("normalisation.decimal_scaling_j_wheat", int(np.ceil(np.log10(m.loc[m.item_id == 1, "price_avg"].abs().max()))))
    return wf


def encoding_demo(m: pd.DataFrame) -> pd.DataFrame:
    """Before/after example of 0/1 dummy coding, one row per province, using the model-ready table's rule."""
    pick = ["Islamabad", "Lahore", "Karachi", "Peshawar", "Quetta"]
    wk = m[m.week_end == m.week_end.max()]
    rows = pd.concat([wk[(wk.item_id == 5) & wk.city.isin(pick)],
                      wk[wk.item_id.isin([13, 31]) & (wk.city == "Islamabad")]])
    rows = rows.sort_values(["item_id", "city_code"])[["city", "province", "item_short", "item_category"]]
    enc = encode_dummies(rows).filter(regex="^(prov_|cat_)").drop(columns=["cat_administered_utility", "cat_other_nonfood"])
    return pd.concat([rows.reset_index(drop=True), enc.reset_index(drop=True)], axis=1)


PROV_REF, CAT_REF = "Punjab", "storable_staple"


def encode_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """0/1 dummies for province and item category with fixed reference levels (Punjab, storable staple).

    Categories are fixed lists, so every subset encodes to the same columns.
    """
    out = df.copy()
    for p_ in PROV_ORDER:
        if p_ != PROV_REF:
            out[f"prov_{p_}"] = (out.province == p_).astype(int)
    for c in CAT_ORDER + ["administered_utility", "other_nonfood"]:
        if c != CAT_REF:
            out[f"cat_{c}"] = (out.item_category == c).astype(int)
    return out


def sum_to_zero_matrix(cities: list[str]) -> pd.DataFrame:
    """Deviation (sum-to-zero) coding: k-1 columns, the last city coded -1 on every column."""
    k = len(cities)
    mat = np.vstack([np.eye(k - 1), -np.ones(k - 1)])
    return pd.DataFrame(mat, index=cities, columns=[f"c_{c}" for c in cities[:-1]])


# ============================================================================ Unit 05
def central_tendency(m: pd.DataFrame) -> pd.DataFrame:
    wk = m.week_end.max()
    rows = []
    for item in [1, 7, 23, 24]:
        s = m[(m.item_id == item) & (m.week_end == wk)].price_avg.dropna()
        mode = s.round(0).mode()
        rows.append({"item": m.loc[m.item_id == item, "item_short"].iat[0], "unit": m.loc[m.item_id == item, "unit"].iat[0],
                     "arithmetic_mean": s.mean(), "geometric_mean": np.exp(np.log(s).mean()), "median": s.median(),
                     "mode_rounded": mode.iat[0] if len(mode) == 1 else np.nan, "n_modes": len(mode)})
    return pd.DataFrame(rows)


def variation_by_item(m: pd.DataFrame) -> pd.DataFrame:
    disp = m.groupby(["week_end", "item_id", "item_short", "is_food", "item_category"])["price_avg"].agg(
        mean="mean", sd="std", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), mn="min", mx="max").reset_index()
    disp["cv"] = disp["sd"] / disp["mean"]
    disp["iqr"] = disp.q3 - disp.q1
    disp["range"] = disp.mx - disp.mn
    item = disp.groupby(["item_id", "item_short", "is_food", "item_category"]).agg(
        mean_cv=("cv", "mean"), mean_range=("range", "mean"), mean_iqr=("iqr", "mean"), mean_sd=("sd", "mean"),
        mean_price=("mean", "mean")).reset_index()
    # Volatility and stickiness for each item (consecutive changes only).
    ch = m[m.dlog_price.notna()].groupby("item_id")["dlog_price"].agg(
        volatility=lambda s: s.std(), share_unchanged=lambda s: (s.abs() <= 1e-12).mean())
    item = item.merge(ch, on="item_id", how="left")
    sq = m.groupby("item_id")["single_quote"].mean().rename("share_single_quote")
    item = item.merge(sq, on="item_id")
    for cat, g in item[item.is_food == 1].groupby("item_category"):
        put(f"variation.mean_cv.{cat}", g.mean_cv.mean()); put(f"variation.n_items.{cat}", len(g))
        put(f"variation.share_unchanged.{cat}", g.share_unchanged.mean())
    return item


def summary_table(m: pd.DataFrame, city: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, SD, IQR and skewness of key numeric variables, each at its own unit of analysis."""
    f = food_rows(m)
    lab = f.groupby("item_id")["price_avg"]
    cols = {
        "price_z_item_week (food; z-score, unitless)": f.price_z_item_week,
        "rel_price (food; log points)": f.rel_price,
        "dlog_price (food; log points, consecutive weeks)": f.dlog_price.dropna(),
        "within_city_range_pct (food; %)": f.within_city_range_pct,
        "single_quote (food; share MIN = MAX)": f.single_quote,
        "fies_mod_sev_pct (17 cities; %)": city.fies_mod_sev_pct,
        "illiteracy10_pct (17 cities; %)": city.illiteracy10_pct,
        "out_of_school_pct (17 cities; %)": city.out_of_school_pct,
        "no_tap_water_pct (17 cities; %)": city.no_tap_water_pct,
        "deprivation_composite (17 cities; 0-1)": city.deprivation_composite,
        "pop_urban (17 cities; millions)": city.pop_urban / 1e6,
        "log_pop_urban (17 cities; log persons)": city.log_pop_urban,
    }
    rows = []
    for k, s_ in cols.items():
        s_ = s_.dropna()
        rows.append({"variable": k, "n": int(len(s_)), "mean": s_.mean(), "median": s_.median(), "sd": s_.std(),
                     "iqr": s_.quantile(.75) - s_.quantile(.25), "skew": stats.skew(s_), "min": s_.min(), "max": s_.max()})
    t = pd.DataFrame(rows)
    put("summary.dlog_share_zero_food", (f.dlog_price.dropna().abs() <= 1e-12).mean())
    put("summary.single_quote_share_food", f.single_quote.mean())
    put("summary.pop_urban_skew", stats.skew(city.pop_urban)); put("summary.log_pop_urban_skew", stats.skew(city.log_pop_urban))
    put("summary.pop_urban_mean_m", city.pop_urban.mean() / 1e6); put("summary.pop_urban_median_m", city.pop_urban.median() / 1e6)
    put("summary.range_pct_mean", f.within_city_range_pct.mean()); put("summary.range_pct_median", f.within_city_range_pct.median())
    put("summary.range_pct_skew", stats.skew(f.within_city_range_pct.dropna()))
    return t


def outliers(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [("rel_price", "rel_price, all 51 items"), ("dlog_price", "weekly log change, all items")]:
        s = m[col].dropna()
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        mu, sd = s.mean(), s.std()
        r = {"variable": label, "n": len(s), "Q1": q1, "Q3": q3, "IQR": iqr, "mean": mu, "sd": sd,
             "iqr_mild_1.5": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
             "iqr_regular_3": int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()),
             "sd_mild_2": int(((s - mu).abs() > 2 * sd).sum()), "sd_regular_3": int(((s - mu).abs() > 3 * sd).sum())}
        rows.append(r)
        key = "rel" if col == "rel_price" else "dlog"
        for k in ["n", "IQR", "iqr_mild_1.5", "iqr_regular_3", "sd_mild_2", "sd_regular_3"]:
            put(f"outliers.{key}.{k.replace('.', '_')}", r[k])
    return pd.DataFrame(rows)


def extreme_check(m: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    """Largest |rel_price| values, checked against PBS's own national MIN/MAX columns for that item-week."""
    ext = m.loc[m.rel_price.abs().sort_values(ascending=False).index[:8],
                ["week_end", "city", "item_id", "item_short", "price_avg", "price_min", "price_max", "rel_price", "source_file"]]
    ext = ext.merge(national[["week_end", "item_id", "nat_min", "nat_max", "nat_avg"]], on=["week_end", "item_id"])
    ext["equals_pbs_national_min_or_max"] = np.isclose(ext.price_min, ext.nat_min) | np.isclose(ext.price_max, ext.nat_max)
    put("outliers.top8_match_pbs_extremes", int(ext.equals_pbs_national_min_or_max.sum()))
    return ext


def correlations(cityx: pd.DataFrame, m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sq = food_rows(m).groupby("city")["single_quote"].mean().rename("single_quote_share")
    c = cityx.merge(sq, on="city")
    cols = ["premium", "deprivation_composite", "fies_mod_sev_pct", "illiteracy10_pct", "out_of_school_pct",
            "no_tap_water_pct", "log_pop_urban", "single_quote_share"]
    put("corr.pearson_premium_single_quote", c[["premium", "single_quote_share"]].corr().iloc[0, 1])
    return c[cols].cov(), c[cols].corr(method="pearson"), c[cols].corr(method="spearman")


def premium_vs_deprivation(cityx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dep in ["deprivation_composite", "fies_mod_sev_pct", "deprivation_composite_nowater"]:
        x, y = cityx[dep].values, cityx.premium.values
        rp, pp = stats.pearsonr(x, y); rs, ps = stats.spearmanr(x, y)
        loo = []
        for i in range(len(x)):
            mask = np.arange(len(x)) != i
            loo.append((cityx.city.iat[i], stats.pearsonr(x[mask], y[mask])[0], stats.spearmanr(x[mask], y[mask])[0]))
        loo = pd.DataFrame(loo, columns=["left_out", "pearson", "spearman"])
        rows.append({"deprivation_measure": dep, "n": len(x), "pearson_r": rp, "pearson_p": pp, "spearman_rho": rs,
                     "spearman_p": ps, "loo_pearson_min": loo.pearson.min(), "loo_pearson_max": loo.pearson.max(),
                     "loo_spearman_min": loo.spearman.min(), "loo_spearman_max": loo.spearman.max(),
                     "loo_most_influential": loo.loc[(loo.pearson - rp).abs().idxmax(), "left_out"]})
        for k in ["pearson_r", "pearson_p", "spearman_rho", "spearman_p", "loo_pearson_min", "loo_pearson_max",
                  "loo_spearman_min", "loo_spearman_max", "loo_most_influential"]:
            put(f"h5.{dep}.{k}", rows[-1][k])
    return pd.DataFrame(rows)


def burden_reranking(cityx: pd.DataFrame) -> pd.DataFrame:
    c = cityx[["city", "province", "premium", "deprivation_composite", "fies_mod_sev_pct"]].copy()
    c["rank_premium"] = c.premium.rank(ascending=False).astype(int)
    c["rank_deprivation"] = c.deprivation_composite.rank(ascending=False).astype(int)
    c["rank_fies"] = c.fies_mod_sev_pct.rank(ascending=False).astype(int)
    # Two burden definitions: (1) premium x deprivation composite (premium scaled to 0-1);
    # (2) premium x FIES food-insecurity share. Both only re-rank; they are not welfare measures.
    pm = (c.premium - c.premium.min()) / (c.premium.max() - c.premium.min())
    c["burden_composite"] = pm * c.deprivation_composite
    c["burden_fies"] = np.exp(c.premium) * c.fies_mod_sev_pct
    c["rank_burden_composite"] = c.burden_composite.rank(ascending=False).astype(int)
    c["rank_burden_fies"] = c.burden_fies.rank(ascending=False).astype(int)
    top5 = lambda col: set(c.nsmallest(5, col).city)
    put("h5.top5_premium", ", ".join(c.nsmallest(5, "rank_premium").city))
    put("h5.top5_deprivation", ", ".join(c.nsmallest(5, "rank_deprivation").city))
    put("h5.top5_fies", ", ".join(c.nsmallest(5, "rank_fies").city))
    put("h5.top5_overlap_premium_deprivation", len(top5("rank_premium") & top5("rank_deprivation")))
    put("h5.top5_overlap_premium_fies", len(top5("rank_premium") & top5("rank_fies")))
    put("h5.top3_burden_composite", ", ".join(c.nsmallest(3, "rank_burden_composite").city))
    put("h5.top3_burden_fies", ", ".join(c.nsmallest(3, "rank_burden_fies").city))
    return c.sort_values("rank_premium")


def contingency(cityx: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(cityx.province, cityx.premium_tercile).reindex(PROV_ORDER)
    return ct


# ============================================================================ Units 06-07
def normality(m: pd.DataFrame, prem: pd.DataFrame) -> dict:
    f = food_rows(m)
    s = f.rel_price
    put("normal.rel_price_skew", stats.skew(s)); put("normal.rel_price_excess_kurtosis", stats.kurtosis(s))
    put("normal.city_item_skew_median", prem["skew"].median())
    raw = f.groupby("item_id")["price_avg"].apply(lambda x: stats.skew(x)).median()
    logd = f.groupby("item_id")["log_price"].apply(lambda x: stats.skew(x)).median()
    put("normal.median_within_item_skew_raw", raw); put("normal.median_within_item_skew_log", logd)
    put("normal.pooled_raw_price_skew", stats.skew(f.price_avg))
    return {"skew": stats.skew(s), "kurt": stats.kurtosis(s)}


def h2_test(item: pd.DataFrame) -> dict:
    a = item[(item.is_food == 1) & (item.item_category == "perishable")].mean_cv
    b = item[(item.is_food == 1) & (item.item_category == "storable_staple")].mean_cv
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = (a.mean() - b.mean()) / se
    t, p2 = stats.ttest_ind(a, b, equal_var=False)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    u, pu = stats.mannwhitneyu(a, b, alternative="greater")
    res = {"n_perishable": len(a), "n_storable": len(b), "mean_cv_perishable": a.mean(), "mean_cv_storable": b.mean(),
           "se_diff": se, "t_welch": t, "p_one_tailed": p1, "mannwhitney_p_one_tailed": pu}
    for k, v in res.items():
        put(f"h2.{k}", v)
    return res


def persistence(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    f = food_rows(m)
    pairs = f[(f.above_median_prev == 1) & f.above_median.notna()]
    pairs_ch = pairs[pairs.price_changed == 1]
    by_city = pairs.groupby("city")["above_median"].agg(stay_rate="mean", n_pairs="count")
    by_city["stay_rate_changed"] = pairs_ch.groupby("city")["above_median"].mean()
    by_city["n_pairs_changed"] = pairs_ch.groupby("city")["above_median"].count()
    by_city = by_city.reset_index()
    t, p2 = stats.ttest_1samp(by_city.stay_rate, 0.5)
    tc, pc2 = stats.ttest_1samp(by_city.stay_rate_changed.dropna(), 0.5)
    res = {"overall_stay_rate": pairs.above_median.mean(), "n_pairs": len(pairs),
           "overall_stay_rate_changed": pairs_ch.above_median.mean(), "n_pairs_changed": len(pairs_ch),
           "n_cities": len(by_city), "city_rate_min": by_city.stay_rate.min(), "city_rate_max": by_city.stay_rate.max(),
           "t_vs_half": t, "p_one_tailed": p2 / 2 if t > 0 else 1 - p2 / 2,
           "t_vs_half_changed": tc, "p_one_tailed_changed": pc2 / 2 if tc > 0 else 1 - pc2 / 2,
           "n_cities_changed": int(by_city.stay_rate_changed.notna().sum()),
           "city_rate_changed_min": by_city.stay_rate_changed.min()}
    for k, v in res.items():
        put(f"h1b.{k}", v)
    return by_city, res


# ============================================================================ Part B (exploratory)
def city_item_matrix(m: pd.DataFrame) -> pd.DataFrame:
    f = food_rows(m)
    mat = f.groupby(["city", "item_id"])["rel_price"].mean().unstack("item_id")
    mat = mat.dropna(axis=1)  # drops the item with structural gaps (rice IRRI-6/9)
    put("partb.matrix_items", mat.shape[1]); put("partb.matrix_cities", mat.shape[0])
    return mat.loc[city_order(m)]


def hier_cluster(mat: pd.DataFrame, k: int = 3):
    z = StandardScaler().fit_transform(mat.values)
    L = linkage(z, method="ward")
    lab = fcluster(L, k, criterion="maxclust")
    cl = pd.DataFrame({"city": mat.index, "cluster": lab, "province": [CITY_PROVINCE[c] for c in mat.index]})
    for kk, g in cl.groupby("cluster"):
        put(f"partb.ward_k{k}.cluster_{kk}", ", ".join(g.city))
    return L, cl


def nearest_neighbours(mat: pd.DataFrame) -> pd.DataFrame:
    z = StandardScaler().fit_transform(mat.values)
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = pd.DataFrame({"city": mat.index, "nearest": mat.index[d.argmin(1)], "distance": d.min(1)})
    nn["mutual"] = [nn.set_index("city").loc[r.nearest, "nearest"] == r.city for r in nn.itertuples()]
    for r in nn.itertuples():
        put(f"partb.nn.{r.city}", r.nearest)
    return nn


def pca(mat: pd.DataFrame, prem: pd.DataFrame):
    z = StandardScaler().fit_transform(mat.values)
    p = PCA().fit(z)
    sc = p.transform(z)
    pm = prem.set_index("city").loc[mat.index, "mean"].values
    r = np.corrcoef(sc[:, 0], pm)[0, 1]
    put("partb.pca_var_pc1", p.explained_variance_ratio_[0]); put("partb.pca_var_pc2", p.explained_variance_ratio_[1])
    put("partb.pca_pc1_corr_mean_premium", abs(r))
    return p, pd.DataFrame({"city": mat.index, "pc1": sc[:, 0] * np.sign(r), "pc2": sc[:, 1]})


def item_kmeans(item: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    fi = item[item.is_food == 1].copy()
    X = StandardScaler().fit_transform(fi[["mean_cv", "volatility", "share_unchanged"]])
    km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
    fi["kmeans_cluster"] = km.labels_
    # Name clusters by their stickiness so labels are stable across runs.
    order = fi.groupby("kmeans_cluster").share_unchanged.mean().sort_values().index
    names = dict(zip(order, ["flexible", "intermediate", "sticky"][:k]))
    fi["market_type"] = fi.kmeans_cluster.map(names)
    ct = pd.crosstab(fi.item_category, fi.market_type)
    put("partb.kmeans_k", k)
    for t, g in fi.groupby("market_type"):
        put(f"partb.kmeans.{t}.n", len(g)); put(f"partb.kmeans.{t}.items", ", ".join(g.item_short))
    return fi, ct


def regression(m: pd.DataFrame, prem: pd.DataFrame, cityx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Descriptive OLS: log p = item-week effects + sum-to-zero city effects (food items)."""
    f = food_rows(m)
    cities = city_order(m)
    iw = pd.factorize(f.week_end.astype(str) + "_" + f.item_id.astype(str))[0]
    S = sum_to_zero_matrix(cities)
    Xc = S.loc[f.city].values
    Xiw = np.zeros((len(f), iw.max() + 1)); Xiw[np.arange(len(f)), iw] = 1
    X = np.hstack([Xiw, Xc]); y = f.log_price.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - np.linalg.matrix_rank(X)
    s2 = resid @ resid / dof
    # Covariance only for the city block (via the Frisch-Waugh partialling of item-week means).
    Xc_dm = Xc - pd.DataFrame(Xc).groupby(iw).transform("mean").values
    cov_c = s2 * np.linalg.inv(Xc_dm.T @ Xc_dm)
    bc = beta[-(len(cities) - 1):]
    b_last = -bc.sum()
    se_last = np.sqrt(np.ones(len(bc)) @ cov_c @ np.ones(len(bc)))
    coef = pd.DataFrame({"city": cities, "coef": np.r_[bc, b_last], "se": np.r_[np.sqrt(np.diag(cov_c)), se_last]})
    coef["ci_lo"] = coef.coef - 1.96 * coef.se; coef["ci_hi"] = coef.coef + 1.96 * coef.se
    coef = coef.merge(prem[["city", "mean"]], on="city")
    r = np.corrcoef(coef.coef, coef["mean"])[0, 1]
    put("partb.reg.n_obs", len(y)); put("partb.reg.corr_coef_vs_simple_mean", r)
    put("partb.reg.max_abs_diff_coef_vs_mean", (coef.coef - coef["mean"]).abs().max())
    # F test: all city effects zero (H1a), restricted model = item-week effects only.
    rss_u = resid @ resid
    rr = y - pd.Series(y).groupby(iw).transform("mean").values
    rss_r = rr @ rr
    q = len(cities) - 1
    F = ((rss_r - rss_u) / q) / (rss_u / dof)
    put("h1a.F", F); put("h1a.df1", q); put("h1a.df2", dof); put("h1a.p", stats.f.sf(F, q, dof))
    put("h1a.r2_within", 1 - rss_u / rss_r)
    # Simple line: premium on deprivation composite, n = 17.
    lr = stats.linregress(cityx.deprivation_composite, cityx.premium)
    tcrit = stats.t.ppf(0.975, len(cityx) - 2)
    simple = {"slope": lr.slope, "slope_ci_lo": lr.slope - tcrit * lr.stderr, "slope_ci_hi": lr.slope + tcrit * lr.stderr,
              "intercept": lr.intercept, "r2": lr.rvalue ** 2, "p": lr.pvalue, "n": len(cityx)}
    for k, v in simple.items():
        put(f"partb.simple_reg.{k}", v)
    return coef, simple


def representation(city: pd.DataFrame, census_all: pd.DataFrame) -> pd.DataFrame:
    """SPI cities' share of each province's urban population (Census 2023)."""
    from .external import CENSUS_FILES
    prov_of_file = {"punjab_districts": "Punjab", "sindh_districts": "Sindh", "kp_districts": "KP",
                    "balochistan_districts": "Balochistan", "islamabad": "ICT"}
    census_all = census_all.assign(province=census_all.census_file.str.replace("table_1_", "").str.replace(".xlsx", "").map(prov_of_file))
    prov_urban = census_all.groupby("province").pop_urban.sum()
    spi = city.groupby("province").agg(n_cities=("city", "count"), spi_pop_urban=("pop_urban", "sum"))
    rep = spi.join(prov_urban.rename("province_pop_urban")).reindex(PROV_ORDER)
    rep["share_of_province_urban"] = rep.spi_pop_urban / rep.province_pop_urban
    rep["share_of_national_urban"] = rep.province_pop_urban / rep.province_pop_urban.sum()
    rep["share_of_spi_cities"] = rep.n_cities / rep.n_cities.sum()
    put("representation.national_urban_pop_4prov_ict", rep.province_pop_urban.sum())
    put("representation.spi_share_national_urban", rep.spi_pop_urban.sum() / rep.province_pop_urban.sum())
    for p_, r in rep.iterrows():
        put(f"representation.{p_}.share_of_province_urban", r.share_of_province_urban)
        put(f"representation.{p_}.share_of_national_urban", r.share_of_national_urban)
        put(f"representation.{p_}.share_of_spi_cities", r.share_of_spi_cities)
    return rep.reset_index()


def measurement_quality(m: pd.DataFrame, cityx: pd.DataFrame) -> pd.DataFrame:
    f = food_rows(m)
    q = f.groupby(["city", "province"]).agg(single_quote_share=("single_quote", "mean"),
                                            mean_within_range_pct=("within_city_range_pct", "mean")).reset_index()
    q = q.merge(cityx[["city", "pop_tercile", "deprivation_tercile"]], on="city")
    q = q.sort_values("single_quote_share")
    put("h4.single_quote_min_city", q.iloc[0].city); put("h4.single_quote_min", q.iloc[0].single_quote_share)
    put("h4.single_quote_max_city", q.iloc[-1].city); put("h4.single_quote_max", q.iloc[-1].single_quote_share)
    for g in ["province", "pop_tercile", "deprivation_tercile"]:
        groups = [x.single_quote_share.values for _, x in q.groupby(g, observed=True)]
        h, p = stats.kruskal(*groups)
        put(f"h4.kruskal_{g}.H", h); put(f"h4.kruskal_{g}.p", p)
    by_prov = q.groupby("province").single_quote_share.mean()
    for p_, v in by_prov.items():
        put(f"h4.single_quote_by_province.{p_}", v)
    return q


def unchanged_by_category(m: pd.DataFrame) -> pd.DataFrame:
    f = food_rows(m)
    d = f[f.dlog_price.notna()]
    t = d.groupby("item_category").dlog_price.apply(lambda s: (s.abs() <= 1e-12).mean()).rename("share_unchanged").reset_index()
    for r in t.itertuples():
        put(f"eval.share_unchanged.{r.item_category}", r.share_unchanged)
    put("eval.share_unchanged_food_all", (d.dlog_price.abs() <= 1e-12).mean())
    return t


# ============================================================================ figures
def fig_coverage(o: dict, manifest: pd.DataFrame):
    style()
    m = o["master"]
    weeks = pd.to_datetime(sorted(m.week_end.unique()))
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    thu = pd.to_datetime(manifest.thursday.unique())
    ax.vlines(thu, 0, 1, color="#BBBBBB", lw=0.8, label="Thursday checked on the PBS site (grey only: no Annexure available)")
    ax.vlines(weeks, 0, 1, color="#0072B2", lw=2.2, label=f"Training week in panel (n = {len(weeks)})")
    hold = pd.to_datetime(manifest.loc[(manifest.outcome == "downloaded") & (manifest.split == "holdout"), "candidate_date"])
    ax.vlines(hold, 0, 1, color="#D55E00", lw=2.2, linestyles="dotted", label="Holdout week (downloaded, unopened)")
    ax.axvline(pd.Timestamp(TRAIN_CUTOFF), color="black", lw=1, ls="--")
    ax.text(pd.Timestamp(TRAIN_CUTOFF), 1.04, f"freeze {TRAIN_CUTOFF}", ha="right", fontsize=8)
    ax.set_yticks([]); ax.set_ylim(0, 1.15); ax.grid(False)
    ax.set_xlabel("Week ending (date)")
    ax.set_title(f"Fig. 1. Panel coverage: {len(weeks)} training weeks; {manifest.thursday.nunique()} Thursdays swept since May 2025")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=7.5)
    return savefig(fig, "fig01_coverage")


def fig_city_premiums(prem: pd.DataFrame):
    style()
    d = prem.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    pct = lambda x: 100 * (np.exp(x) - 1)
    for i, r in enumerate(d.itertuples()):
        col = PROVINCE_COLORS[r.province]
        ax.plot([pct(r.ci_lo), pct(r.ci_hi)], [i, i], color=col, lw=2)
        ax.plot(pct(r.mean), i, "o", color=col, ms=6, mfc=col if r.bonferroni_sig else "white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.city)
    ax.set_xlabel("Mean food price premium vs national geometric mean (%)")
    ax.set_title("Fig. 2. City food price premiums, 95% t-intervals over food items")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=p) for p, c in PROVINCE_COLORS.items()]
    handles.append(plt.Line2D([], [], color="grey", marker="o", mfc="white", lw=0, label="not significant after Bonferroni"))
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)
    return savefig(fig, "fig02_city_premiums")


def fig_dispersion(item: pd.DataFrame):
    style()
    fi = item[item.is_food == 1]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    data = [fi[fi.item_category == c].mean_cv.values for c in CAT_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], CAT_ORDER):
        patch.set_facecolor(CAT_COLORS[c]); patch.set_alpha(0.45)
    rng = np.random.default_rng(1)
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=14, color="black", zorder=3)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"{CAT_LABEL[c]}\n(n = {len(d)})" for c, d in zip(CAT_ORDER, data)])
    ax.set_ylabel("Mean cross-city coefficient of variation")
    ax.set_title("Fig. 3. Cross-city price dispersion by food item category (one dot = one item)")
    return savefig(fig, "fig03_dispersion_by_category")


LABEL_OFFSETS = {
    "deprivation_composite": {"Gujranwala": (-4, 4), "Faisalabad": (4, -7), "Hyderabad": (-4, -8), "Lahore": (3, 4),
                              "Sialkot": (-4, -6), "Bannu": (-4, -6), "Sukkur": (4, 2)},
    "fies_mod_sev_pct": {"Faisalabad": (-4, 3), "Gujranwala": (4, 2), "Sialkot": (4, -6)},
}


def fig_premium_deprivation(cityx: pd.DataFrame, h5: pd.DataFrame):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, dep, lab in [(axes[0], "deprivation_composite", "Deprivation composite (0 = least, 1 = most deprived)"),
                         (axes[1], "fies_mod_sev_pct", "Moderate or severe food insecurity, FIES (%)")]:
        for r in cityx.itertuples():
            ax.scatter(getattr(r, dep), 100 * (np.exp(r.premium) - 1), color=PROVINCE_COLORS[r.province], s=30, zorder=3)
            off = LABEL_OFFSETS.get(dep, {}).get(r.city, (3, 2))
            ax.annotate(r.city, (getattr(r, dep), 100 * (np.exp(r.premium) - 1)), fontsize=6.5, xytext=off,
                        textcoords="offset points", ha="right" if off[0] < 0 else "left")
        h = h5.set_index("deprivation_measure").loc[dep]
        ax.set_xlabel(lab); ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"Spearman rho = {h.spearman_rho:.2f} (p = {h.spearman_p:.2f}), n = 17", fontsize=8.5)
    axes[0].set_ylabel("Mean food price premium (%)")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=0, label=p) for p, c in PROVINCE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
    fig.suptitle("Fig. 4. City food price premium against district deprivation (PSLM 2019-20)", fontsize=10)
    fig.tight_layout()
    return savefig(fig, "fig04_premium_vs_deprivation")


def fig_dendrogram(L, mat: pd.DataFrame):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    dn = dendrogram(L, labels=mat.index.tolist(), ax=ax, color_threshold=0, above_threshold_color="#555555", leaf_rotation=60)
    cut = (L[-3, 2] + L[-2, 2]) / 2
    ax.axhline(cut, color="#D55E00", ls="--", lw=1)
    ax.text(ax.get_xlim()[0] + 2, cut + 0.2, "cut giving 3 clusters", color="#D55E00", fontsize=7.5)
    for t in ax.get_xticklabels():
        t.set_color(PROVINCE_COLORS[CITY_PROVINCE[t.get_text()]]); t.set_fontsize(8)
    ax.set_ylabel("Ward linkage distance (standardised log points)"); ax.grid(False)
    ax.set_title("Fig. 5. Hierarchical clustering of cities by food price profile (exploratory)")
    handles = [plt.Line2D([], [], color=c, marker="s", lw=0, label=p) for p, c in PROVINCE_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, title="label colour = province", title_fontsize=7.5)
    return savefig(fig, "fig05_dendrogram")


def fig_corr(corr: pd.DataFrame, title_suffix="Spearman"):
    style()
    labels = {"premium": "Premium", "deprivation_composite": "Deprivation comp.", "fies_mod_sev_pct": "FIES food insec.",
              "illiteracy10_pct": "Illiteracy 10+", "out_of_school_pct": "Out of school 5-16",
              "no_tap_water_pct": "No tap water", "log_pop_urban": "log urban pop.", "single_quote_share": "Single-quote share"}
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels([labels[c] for c in corr.columns], rotation=45, ha="right")
    ax.set_yticklabels([labels[c] for c in corr.index]); ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label=f"{title_suffix} correlation")
    ax.set_title(f"Fig. 6. City-level {title_suffix} correlations (n = 17)")
    return savefig(fig, "fig06_correlation")


def fig_single_quote(q: pd.DataFrame):
    style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    d = q.sort_values("single_quote_share")
    ax.barh(d.city, 100 * d.single_quote_share, color=[PROVINCE_COLORS[p] for p in d.province])
    ax.set_xlabel("Food records where MIN = MAX (%)")
    ax.set_title("Share of food price records where PBS's minimum and maximum quote are equal, by city")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=p) for p, c in PROVINCE_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)
    return savefig(fig, "fig07_single_quote_share")


def fig_relprice_hist(m: pd.DataFrame):
    style()
    s = m.rel_price.dropna()
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1; mu, sd = s.mean(), s.std()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.hist(s, bins=120, density=True, color="#56B4E9", alpha=0.8, label="rel_price, all 51 items")
    x = np.linspace(s.min(), s.max(), 400)
    ax.plot(x, stats.norm.pdf(x, mu, sd), color="black", lw=1, label="normal with same mean and SD")
    for v, ls, lab in [(q1 - 1.5 * iqr, "--", "Q1 - 1.5 IQR / Q3 + 1.5 IQR"), (q3 + 1.5 * iqr, "--", None),
                       (mu - 3 * sd, ":", "mean -/+ 3 SD"), (mu + 3 * sd, ":", None)]:
        ax.axvline(v, color="#D55E00", ls=ls, lw=1, label=lab)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xlabel("Relative price: log(city price / national geometric mean), log points")
    ax.set_ylabel("Density"); ax.legend(fontsize=7)
    ax.set_title("Fig. 8. Distribution of relative prices with outlier fences")
    return savefig(fig, "fig08_relprice_hist")


def fig_pca(scores: pd.DataFrame, p):
    style()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for r in scores.itertuples():
        ax.scatter(r.pc1, r.pc2, color=PROVINCE_COLORS[CITY_PROVINCE[r.city]], s=30)
        ax.annotate(r.city, (r.pc1, r.pc2), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({100 * p.explained_variance_ratio_[0]:.0f}% of variance; aligned with mean premium)")
    ax.set_ylabel(f"PC2 ({100 * p.explained_variance_ratio_[1]:.0f}% of variance)")
    ax.set_title("Fig. 9. PCA map of city food price profiles (exploratory)")
    return savefig(fig, "fig09_pca")


def fig_province_tercile(ct: pd.DataFrame):
    style()
    prop = ct.div(ct.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    left = np.zeros(len(prop))
    for lab, col in zip(["low", "mid", "high"], ["#56B4E9", "#F0E442", "#D55E00"]):
        ax.barh(prop.index, prop[lab], left=left, color=col, label=f"{lab} premium tercile")
        left += prop[lab].values
    for i, p_ in enumerate(prop.index):
        ax.text(1.01, i, f"n = {ct.loc[p_].sum()}", va="center", fontsize=7.5)
    ax.set_xlabel("Proportion of the province's SPI cities"); ax.set_xlim(0, 1.12)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.set_title("Fig. 10. Premium tercile by province (conditional proportions)")
    return savefig(fig, "fig10_province_tercile")


def fig_box_province_category(m: pd.DataFrame):
    style()
    f = food_rows(m)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    d1 = [f[f.province == p].rel_price.values for p in PROV_ORDER]
    axes[0].boxplot(d1, flierprops={"markersize": 1.5, "alpha": 0.3}); axes[0].set_xticks(range(1, 6)); axes[0].set_xticklabels(PROV_ORDER)
    axes[0].set_ylabel("Relative price (log points)"); axes[0].set_title("by province", fontsize=9)
    d2 = [f[f.item_category == c].rel_price.values for c in CAT_ORDER]
    axes[1].boxplot(d2, flierprops={"markersize": 1.5, "alpha": 0.3}); axes[1].set_xticks(range(1, 5))
    axes[1].set_xticklabels([CAT_LABEL[c] for c in CAT_ORDER], rotation=15); axes[1].set_title("by item category", fontsize=9)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.6)
    fig.suptitle("Fig. 11. Food relative prices (city x item x week) by province and by item category", fontsize=10)
    fig.tight_layout()
    return savefig(fig, "fig11_box_province_category")


def fig_split_design(o: dict):
    style()
    weeks = pd.to_datetime(sorted(o["master"].week_end.unique()))
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    n_tr = int(round(len(weeks) * 0.7))
    ax.scatter(weeks[:n_tr], [2] * n_tr, color="#0072B2", s=18, label="M03 model-fitting weeks (earliest ~70%)")
    ax.scatter(weeks[n_tr:], [2] * (len(weeks) - n_tr), color="#E69F00", s=18, label="M03 validation weeks (latest ~30%)")
    ax.scatter([pd.Timestamp("2026-10-01")], [2], color="#D55E00", marker="x", s=30, label="Week-14 holdout (frozen, unopened)")
    ax.axvline(pd.Timestamp(TRAIN_CUTOFF), color="black", ls="--", lw=1)
    ax.set_yticks([]); ax.set_ylim(1.5, 2.5); ax.grid(False)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    ax.set_title("Fig. 12. Planned time-ordered split; random folds would leak future weeks (design only)")
    return savefig(fig, "fig12_split_design")


def fig_aggregation(agg: pd.DataFrame):
    style()
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    ax.barh(agg.level[::-1], agg.sd_rel_price[::-1], color="#0072B2")
    ax.set_xlabel("SD of food relative price (log points)")
    ax.set_title("Fig. 13. Variability shrinks as prices are aggregated")
    return savefig(fig, "fig13_aggregation")


def fig_reranking(rr: pd.DataFrame):
    style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for r in rr.itertuples():
        col = PROVINCE_COLORS[r.province]
        ax.plot([0, 1], [r.rank_premium, r.rank_deprivation], color=col, lw=1.2, marker="o", ms=4)
        ax.text(-0.03, r.rank_premium, r.city, ha="right", va="center", fontsize=7)
        ax.text(1.03, r.rank_deprivation, r.city, ha="left", va="center", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Rank by food price premium\n(1 = pays most)",
                                                "Rank by deprivation composite\n(1 = most deprived)"])
    ax.invert_yaxis(); ax.set_yticks([]); ax.grid(False); ax.set_xlim(-0.5, 1.5)
    ax.spines["left"].set_visible(False)
    ax.set_title("Fig. 14. City ranks by food price premium and by deprivation (H5, descriptive)")
    return savefig(fig, "fig14_reranking")
