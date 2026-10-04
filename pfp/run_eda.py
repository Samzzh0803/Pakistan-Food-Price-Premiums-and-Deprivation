"""Run the full M02 EDA on built objects: tables to reports/m02/tables, figures, numbers JSON."""
from __future__ import annotations

import pandas as pd

from . import eda
from .config import BACKFILL_MANIFEST, REPORTS
from .external import parse_census_table1

TABLES = REPORTS / "tables"


def run_all(o: dict) -> dict:
    eda.N.clear()
    TABLES.mkdir(parents=True, exist_ok=True)
    m, city = o["master"], o["city"]
    manifest = pd.read_csv(BACKFILL_MANIFEST)
    r = {}
    r["weeks"] = eda.inventory(o)
    r["backfill"] = eda.backfill_summary(manifest)
    r["cpi"] = eda.external_summary(o)
    r["codebook"] = eda.codebook(m)
    r["tidy"] = eda.tidy_demo(o, o["inventory"].query("status == 'kept'").source_file.iloc[-1])
    r["cleaning"] = eda.cleaning_log(o)
    r["missing_kinds"], r["missing_cols"] = eda.missing_table(m)
    r["integration"] = eda.integration_table(o)
    r["aggregation"] = eda.aggregation(m)
    r["splitting"] = eda.splitting(m)
    r["premium"] = eda.city_premiums(m)
    r["bootstrap"] = eda.bootstrap_median(m)
    r["labelled"], r["thresholds"] = eda.abnormal_labels(m)
    r["feature_sel"] = eda.feature_selection(m, r["labelled"])
    r["discretisation"] = eda.discretisation(city)
    r["cityx"] = eda.add_terciles(city, r["premium"])
    r["normalisation"] = eda.normalisation_demo(m, city)
    r["encoding"] = eda.encoding_demo(m)
    r["central"] = eda.central_tendency(m)
    r["item"] = eda.variation_by_item(m)
    r["summary"] = eda.summary_table(m, city)
    r["outliers"] = eda.outliers(m)
    r["extremes"] = eda.extreme_check(m, o["national"])
    r["cov"], r["pearson"], r["spearman"] = eda.correlations(r["cityx"], m)
    r["h5"] = eda.premium_vs_deprivation(r["cityx"])
    r["rerank"] = eda.burden_reranking(r["cityx"])
    r["contingency"] = eda.contingency(r["cityx"])
    r["normal"] = eda.normality(m, r["premium"])
    r["h2"] = eda.h2_test(r["item"])
    r["persist_city"], r["persist"] = eda.persistence(m)
    # Part B (exploratory)
    r["mat"] = eda.city_item_matrix(m)
    r["linkage"], r["clusters"] = eda.hier_cluster(r["mat"])
    r["nn"] = eda.nearest_neighbours(r["mat"])
    r["pca"], r["pca_scores"] = eda.pca(r["mat"], r["premium"])
    r["kmeans"], r["kmeans_ct"] = eda.item_kmeans(r["item"])
    r["reg_coef"], r["reg_simple"] = eda.regression(m, r["premium"], r["cityx"])
    r["representation"] = eda.representation(city, parse_census_table1())
    r["quality"] = eda.measurement_quality(m, r["cityx"])
    r["unchanged"] = eda.unchanged_by_category(m)
    # Figures
    r["figs"] = [
        eda.fig_coverage(o, manifest), eda.fig_city_premiums(r["premium"]), eda.fig_dispersion(r["item"]),
        eda.fig_premium_deprivation(r["cityx"], r["h5"]), eda.fig_dendrogram(r["linkage"], r["mat"]),
        eda.fig_corr(r["spearman"]), eda.fig_single_quote(r["quality"]), eda.fig_relprice_hist(m),
        eda.fig_pca(r["pca_scores"], r["pca"]), eda.fig_province_tercile(r["contingency"]),
        eda.fig_box_province_category(m), eda.fig_split_design(o), eda.fig_aggregation(r["aggregation"]),
        eda.fig_reranking(r["rerank"]),
    ]
    eda.put("figures.count", len(r["figs"]))
    for k, v in r.items():
        if isinstance(v, pd.DataFrame) and k not in ("labelled", "weeks"):
            v.to_csv(TABLES / f"{k}.csv", index=k in ("cov", "pearson", "spearman", "contingency", "kmeans_ct", "mat", "missing_cols"))
    eda.save_numbers()
    return r
