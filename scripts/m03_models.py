"""Milestone 03: run the baseline models for H3 and write numbers, tables and figures.

Writes reports/m03/m03_numbers.json (every number quoted in the manuscript), reports/m03/tables/*.csv and
reports/m03/figures/*.png. Runs offline from data/processed/m02/master_city_item_week.parquet in about two minutes.
The holdout weeks (data/raw/pbs_holdout/) are never read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfp import model  # noqa: E402

if __name__ == "__main__":
    o = model.run_all(write=True)
    n = model.N
    print(f"labelled rows {n['data']['rows']:,}, positives {n['data']['positives']}, base rate {n['data']['base_rate']:.4f}")
    print(f"train {n['split']['train_rows']:,} rows / {n['split']['train_pos']} pos; "
          f"test {n['split']['test_rows']:,} rows / {n['split']['test_pos']} pos")
    print(o["main"][["model", "cv_roc_auc_mean", "cv_roc_auc_sd", "test_roc_auc", "test_auc_ci_lo", "test_auc_ci_hi",
                     "test_pr_auc"]].round(3).to_string(index=False))
    print(f"random K-fold AUC {n['leak']['random_kfold_auc_mean']:.3f} vs forward-chaining {n['leak']['forward_auc_mean']:.3f}; "
          f"shuffled-label test AUC mean {n['leak']['shuffled_mean']:.3f} over {n['leak']['shuffled_reps']} shuffles")
    print("wrote", model.NUMBERS_JSON)
