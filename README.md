# Who Pays More, and Who Can Least Afford It?

**Food price premiums, deprivation and abnormal price movements across selected Pakistani urban markets**

Course project for CS/SDP 312/314 Data Science for Social Good, Habib University, Fall 2026.
Team: Yousuf Uyghur (Computer Science) and Sameer Hassan (Computer Science).

Pakistan's weekly Sensitive Price Indicator (SPI) is reported as one national figure. Beneath it, the Pakistan Bureau of Statistics (PBS) publishes prices for 51 items in 17 cities. The project asks three questions:

- Do persistent city-level food price differences exist?
- Are the cities that pay most the ones least able to absorb it?
- Is the published city detail enough to detect localised price stress?

Primary goal: SDG 2 (Targets 2.1 and 2.c). Secondary goal: SDG 10.

## Milestone 02 deliverables (branch `milestone-02`)

| Deliverable | Path |
|---|---|
| Manuscript draft, 8 pages with appendix (PDF and its HTML source) | `reports/m02/M02_report.pdf`, `reports/m02/M02_report.html` |
| Executed notebook | `notebooks/M02_data_prep_eda.ipynb` |
| Every statistic quoted in the manuscript | `reports/m02/m02_numbers.json` |
| Result tables and figures | `reports/m02/tables/`, `reports/m02/figures/` |
| Master dataset (one row = city x item x week) | `data/processed/m02/master_city_item_week.{csv,parquet}` |
| Derived views | `city_table`, `city_by_food_item_mean_rel_price`, `item_week_dispersion` in `data/processed/m02/` |
| Model-ready food table (provenance dropped, 0/1 dummies for province and item category) | `data/processed/m02/model_ready_food.parquet` |
| Parser acceptance checks, per week | `data/processed/m02/acceptance_checks_by_week.csv` |
| Correction note (the panel has 17 cities, not 7) | `reports/CORRECTION_2026-10-04_appendix_a_17_cities.md` |

## Reproduce

Python 3.12. Everything runs offline from files already in the repository.

```bash
python -m venv .venv && .venv/Scripts/activate        # or source .venv/bin/activate
pip install -r requirements.txt
python scripts/m02_eda.py          # parse 27 weeks, merge sources, write master, tables, figures, numbers JSON
python scripts/m02_make_notebook.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace M02_data_prep_eda.ipynb && cd ..
python scripts/m02_report.py       # HTML manuscript, printed to PDF with headless Microsoft Edge if present
```

The only step that uses the network is `python scripts/m02_backfill_pbs.py`, which sweeps the PBS release pages for new weeks. Do not run it unless you mean to refresh the archive.

Code layout: `pfp/config.py` holds constants, including the training cutoff `TRAIN_CUTOFF`. `pfp/parse_annex.py` is the Appendix-A parser and its acceptance assertions. `pfp/external.py` reads the census, PSLM and CPI files. `pfp/build.py` builds the master dataset. `pfp/eda.py` holds every analysis and figure. `pfp/run_eda.py` is the runner the script uses; the notebook calls the same `pfp/eda.py` functions directly, step by step.

## Data sources

All sources are official publications of the Pakistan Bureau of Statistics (pbs.gov.pk), used for non-commercial academic research. Every downloaded file is stored unchanged together with its URL and SHA-256 hash.

| Source | Role | Files | Manifest |
|---|---|---|---|
| Weekly SPI Annexure, Appendix-A (2025-05-15 to 2026-09-24) | City-item prices: the panel | `data/raw/verification/`, `data/raw/pbs_backfill/` | `data/raw/pbs_backfill_manifest.csv`, `data/raw/source_manifest.csv`, `data/processed/m02/annex_file_inventory.csv` |
| PSLM District Level Survey 2019-20 (Tables 2.14(a), 2.15, 7.1, 9.1) | District deprivation | `data/external/raw/PSLM_2019_20_District_Level.pdf` | `data/external/external_manifest.csv` |
| Census 2023, Table 1 | Population (total and urban) | `data/external/raw/census2023/` | same |
| CPI (Urban), food and non-alcoholic beverages | Deflator only | `data/external/raw/cpi/` | same |

The city-to-district alignment rules (Karachi as several districts; Islamabad and Rawalpindi as one market but two administrative units) are documented in `data/external/city_district_crosswalk.csv`.

**Holdout.** Weeks ending after 2026-09-24 are saved to `data/raw/pbs_holdout/` and hashed. They are never parsed, plotted, or used to fit anything in Milestone 02.

## Correction to Milestone 01

Milestone 01 reported a 7-city panel. The earlier parser read only the first of the three tables stacked in Appendix-A. The files built from that parse are kept as an audit trail: `data/processed/pbs_weekly_panel.csv`, `data/verification/pbs_verified_sample.csv` and the summaries derived from it, and `reports/FINAL_DATA_SCOPE_CHECK.md`. **Do not reuse them.** `reports/CORRECTION_2026-10-04_appendix_a_17_cities.md` explains what was wrong and what replaces it.
