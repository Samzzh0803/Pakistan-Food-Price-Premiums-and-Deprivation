"""Milestone 03: baseline classifiers for H3 (abnormal next-week price movement).

One row = one city x food item x week t. The target y = 1 if the item's price in that city
moves "abnormally" between week t and the next consecutive week t+1. Abnormal means the
absolute weekly log change is larger than the item's own 90th percentile of NON-ZERO changes,
with that percentile fitted on training weeks only.

Every function is small and does one step, so the notebook can call them one at a time.
Headline numbers go into the dict `N`; `save_numbers` writes them to reports/m03/m03_numbers.json.

Leakage rules enforced here (each has an assert):
  1. Nothing after TRAIN_CUTOFF is used; the holdout folder is never opened.
  2. Label thresholds are fitted on the training rows of each split / fold only.
  3. Purge: a training row at week t whose t+1 week is an evaluation week is dropped.
  4. Scalers and the category base rates are fitted on training rows only (Pipelines / fit()).
  5. Hyperparameters and decision thresholds are chosen from cross-validation only.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

from .config import CITY_PROVINCE, PROCESSED, ROOT, TRAIN_CUTOFF  # noqa: E402

SEED = 42
REPORTS = ROOT / "reports" / "m03"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
NUMBERS_JSON = REPORTS / "m03_numbers.json"
N: dict = {}

# Features, all known at week t. Missing changes are set to 0 together with a has_ flag.
CONT_FEATURES = ["abs_dlog", "abs_dlog_lag1", "rel_price", "within_city_range_pct"]
CAT_FEATURES = ["cat_perishable", "cat_branded_packaged", "cat_prepared_food"]
FEATURES = ["abs_dlog", "has_dlog", "price_changed", "abs_dlog_lag1", "has_lag1", "rel_price",
            "within_city_range_pct", "single_quote", *CAT_FEATURES,
            "prov_Sindh", "prov_KP", "prov_Balochistan", "prov_ICT", "log_pop_urban", "deprivation_composite"]
FEATURE_WHY = {
    "abs_dlog": "size of this week's change: prices that just moved may keep moving or snap back",
    "has_dlog": "1 if this week's change is observed (previous week published); flag instead of guessing",
    "price_changed": "1 if the price changed at all this week; most weeks prices are flat",
    "abs_dlog_lag1": "size of last week's change: a longer momentum window",
    "has_lag1": "1 if last week's change is observed",
    "rel_price": "log price relative to the national geometric mean: a city far from the others may correct",
    "within_city_range_pct": "(max - min) / avg across the city's outlets: disagreement may precede a move",
    "single_quote": "1 if min = max (one effective quote): a data-quality flag (H4)",
    "cat_perishable": "perishable item (reference: storable staple); expected to be most volatile (H2)",
    "cat_branded_packaged": "branded / packaged item; printed prices change rarely",
    "cat_prepared_food": "prepared food item",
    "prov_Sindh": "province dummy (reference: Punjab)",
    "prov_KP": "province dummy (reference: Punjab)",
    "prov_Balochistan": "province dummy (reference: Punjab)",
    "prov_ICT": "Islamabad Capital Territory dummy (reference: Punjab)",
    "log_pop_urban": "log urban population of the city's district: market size",
    "deprivation_composite": "PSLM 2019-20 district deprivation (0-1): do poorer places see more shocks?",
}
CAT_LABEL = {"perishable": "Perishable", "storable_staple": "Storable staple", "branded_packaged": "Branded/packaged",
             "prepared_food": "Prepared food"}
MODEL_LABEL = {"majority": "Majority class", "category": "Category base-rate rule", "logistic": "Logistic regression",
               "tree": "Decision tree", "knn": "KNN"}
MODEL_COLORS = {"majority": "#999999", "category": "#E69F00", "logistic": "#0072B2", "tree": "#009E73", "knn": "#CC79A7"}
R_VALUES = [1, 5, 10, 20, 35]
# Reference cost ratio for the confusion matrices and error analysis. It is a value judgment (Section 7). We use the
# smallest r in R_VALUES at which every model's CV-chosen threshold raises any alerts; at r <= 10 most models' cheapest
# CV policy is to flag nothing, which leaves no errors of the second kind to analyse.
R_REF = 20
ALERT_N = [10, 25, 50]
TERCILE_WEIGHT = {"low": 1.0, "mid": 1.5, "high": 2.0}


# ------------------------------------------------------------------ bookkeeping
def _r(x, k=4):
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), k)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def put(path: str, value):
    """Store a number in N under a dotted path, e.g. put('data.rows', 7095)."""
    d = N
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = _r(value)


def save_numbers():
    NUMBERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_JSON.write_text(json.dumps(N, indent=1, default=str, sort_keys=True), encoding="utf-8")


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> pd.DataFrame:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=index, float_format="%.6g")
    return df


def style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False})


def savefig(fig, name) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


# ------------------------------------------------------------------ data and features
def load_master() -> pd.DataFrame:
    """Read the M02 master dataset. It holds training weeks only; the holdout folder is never opened."""
    m = pd.read_parquet(PROCESSED / "master_city_item_week.parquet")
    assert m.week_end.max() <= pd.Timestamp(TRAIN_CUTOFF), "master contains a week after the training cutoff"
    return m


def build_features(master: pd.DataFrame) -> pd.DataFrame:
    """Food rows with a known price, the week-t features, and the next-week change that the label is built from.

    Only rows whose next week is a consecutive release (6-8 days later) can carry a label, because the
    'next-week change' is undefined across a gap in PBS's published archive.
    """
    f = master[(master.is_food == 1) & master.price_avg.notna()].sort_values(["city", "item_id", "week_end"]).copy()
    g = f.groupby(["city", "item_id"])
    # dlog_price at row t+1 is the change from t to t+1 (it is already NaN when t and t+1 are not consecutive).
    f["dlog_next"] = g["dlog_price"].shift(-1)
    f["next_week"] = g["week_end"].shift(-1)
    f = f[f.dlog_next.notna()].copy()
    assert f.next_week.max() <= pd.Timestamp(TRAIN_CUTOFF), "a label would come from a holdout week"
    assert ((f.next_week - f.week_end).dt.days.between(6, 8)).all()

    # Week-t features. Missing changes become 0 together with a has_ flag (flag, don't impute).
    f["has_dlog"] = f.dlog_price.notna().astype(int)
    f["abs_dlog"] = f.dlog_price.abs().fillna(0.0)
    f["price_changed"] = f.price_changed.fillna(0.0)
    f["has_lag1"] = f.dlog_price_lag1.notna().astype(int)
    f["abs_dlog_lag1"] = f.dlog_price_lag1.abs().fillna(0.0)
    for c in ["perishable", "branded_packaged", "prepared_food"]:
        f[f"cat_{c}"] = (f.item_category == c).astype(int)
    for p in ["Sindh", "KP", "Balochistan", "ICT"]:
        f[f"prov_{p}"] = (f.province == p).astype(int)
    assert f[FEATURES].notna().all().all(), "a feature is missing after flagging"

    # City groups used later for the error breakdown (city-level terciles over the 17 cities).
    city = f.drop_duplicates("city").set_index("city")
    pop_t = pd.qcut(city.pop_urban, 3, labels=["small", "medium", "large"])
    dep_t = pd.qcut(city.deprivation_composite, 3, labels=["low", "mid", "high"])
    f["pop_tercile"] = f.city.map(pop_t).astype(str)
    f["dep_tercile"] = f.city.map(dep_t).astype(str)
    keep = ["week_end", "next_week", "city", "province", "item_id", "item_short", "item_category", "pop_tercile",
            "dep_tercile", "dlog_next"] + FEATURES
    return f[keep].sort_values(["week_end", "city", "item_id"]).reset_index(drop=True)


def labelled_weeks(rows: pd.DataFrame) -> list:
    return sorted(rows.week_end.unique())


def time_split(rows: pd.DataFrame, n_test: int = 4) -> tuple[list, list]:
    """Time-ordered split: the first labelled weeks train, the last n_test labelled weeks test."""
    w = labelled_weeks(rows)
    return w[:-n_test], w[-n_test:]


def forward_chain_folds(train_weeks: list, first: int = 6) -> list[tuple[list, list]]:
    """Expanding-window folds: train on weeks 1..k, validate on week k+1, for k = first .. len-1."""
    return [(train_weeks[:k], [train_weeks[k]]) for k in range(first, len(train_weeks))]


def make_labels(rows: pd.DataFrame, fit_rows: pd.DataFrame, rule: str = "refined") -> tuple[pd.DataFrame, pd.Series]:
    """Fit per-item thresholds on fit_rows only, then label every row in rows.

    refined:  90th percentile of the item's non-zero |changes|; items with < 5 non-zero changes get no label.
    contract: plain 90th percentile of all |changes| (the M02 contract rule; 0 for many items).
    """
    a = fit_rows.dlog_next.abs()
    if rule == "refined":
        nz = a[a > 1e-12]
        thr = nz.groupby(fit_rows.loc[nz.index, "item_id"]).agg(lambda s: np.percentile(s, 90) if len(s) >= 5 else np.nan)
        thr = thr.reindex(sorted(fit_rows.item_id.unique()))
    else:
        thr = a.groupby(fit_rows.item_id).agg(lambda s: np.percentile(s, 90))
    out = rows.copy()
    out["threshold"] = out.item_id.map(thr)
    out = out[out.threshold.notna()].copy()
    out["y"] = (out.dlog_next.abs() > out.threshold).astype(int)
    return out, thr


def split_data(rows: pd.DataFrame, train_weeks: list, eval_weeks: list, rule: str = "refined") -> dict:
    """Build one leakage-free split: purge, fit label thresholds on the purged training rows, label both parts."""
    train_weeks, eval_weeks = list(train_weeks), list(eval_weeks)
    assert max(train_weeks) < min(eval_weeks), "training weeks must come before evaluation weeks"
    tr = rows[rows.week_end.isin(train_weeks)]
    # Purge: a training row at week t is labelled from week t+1; if t+1 is an evaluation week, drop it.
    purged = tr[tr.next_week.isin(eval_weeks)]
    tr = tr[~tr.next_week.isin(eval_weeks)]
    ev = rows[rows.week_end.isin(eval_weeks)]
    tr_l, thr = make_labels(tr, tr, rule)          # thresholds come from training rows only
    ev_l, _ = make_labels(ev, tr, rule)            # ...and are applied unchanged to evaluation rows
    assert not tr_l.next_week.isin(eval_weeks).any() and not tr_l.week_end.isin(eval_weeks).any()
    assert set(ev_l.week_end) <= set(eval_weeks)
    return {"train": tr_l.reset_index(drop=True), "eval": ev_l.reset_index(drop=True), "thresholds": thr,
            "n_purged": len(purged), "purged_weeks": sorted(purged.week_end.unique()),
            "items_unlabelled": int(thr.isna().sum()), "train_weeks": train_weeks, "eval_weeks": eval_weeks}


# ------------------------------------------------------------------ models
class CategoryRateRule(BaseEstimator, ClassifierMixin):
    """Naive benchmark 2: score each row by its item category's abnormal rate in the training rows."""

    def _cat(self, X):
        X = pd.DataFrame(X, columns=FEATURES) if not isinstance(X, pd.DataFrame) else X
        cat = pd.Series("storable_staple", index=X.index)
        for c in ["perishable", "branded_packaged", "prepared_food"]:
            cat[X[f"cat_{c}"] == 1] = c
        return cat

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.rates_ = y.groupby(self._cat(X)).mean()
        self.overall_ = float(y.mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self._cat(X).map(self.rates_).fillna(self.overall_).to_numpy()
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_models() -> dict:
    """Each entry: an unfitted estimator, its hyperparameter grid, and the feature list it uses.

    The scaler sits inside the Pipeline, so it is refitted on the training rows of every fold.
    """
    return {
        "majority": {"est": DummyClassifier(strategy="most_frequent"), "grid": [{}], "features": FEATURES},
        "category": {"est": CategoryRateRule(), "grid": [{}], "features": FEATURES},
        "logistic": {"est": Pipeline([("scale", StandardScaler()),
                                      ("clf", LogisticRegression(class_weight="balanced", max_iter=5000,
                                                                 random_state=SEED))]),
                     "grid": [{"clf__C": c} for c in [0.01, 0.1, 1, 10]], "features": FEATURES},
        "tree": {"est": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                 "grid": [{"criterion": c, "max_depth": d, "min_samples_leaf": m}
                          for c, d, m in itertools.product(["gini", "entropy"], [2, 3, 4, 5, 6], [10, 30, 60])],
                 "features": FEATURES},
        "knn": {"est": Pipeline([("scale", MinMaxScaler()), ("clf", KNeighborsClassifier())]),
                "grid": [{"clf__n_neighbors": k, "clf__weights": w, "clf__metric": mt, "features": fs}
                         for fs, k, w, mt in itertools.product(["all", "continuous"], [5, 15, 25, 51, 101, 201],
                                                               ["uniform", "distance"], ["euclidean", "manhattan"])],
                "features": FEATURES},
    }


def _features(params: dict, default: list) -> list:
    fs = params.get("features")
    return CONT_FEATURES if fs == "continuous" else default


def fit_predict(spec: dict, params: dict, train: pd.DataFrame, ev: pd.DataFrame, y_col: str = "y"):
    """Fit a fresh copy of the model on train and return (fitted model, scores for ev)."""
    feats = _features(params, spec["features"])
    est = clone(spec["est"]).set_params(**{k: v for k, v in params.items() if k != "features"})
    est.fit(train[feats], train[y_col])
    return est, est.predict_proba(ev[feats])[:, 1]


# ------------------------------------------------------------------ metrics
def evaluate(y, score, threshold: float = 0.5) -> dict:
    """Unit 10 metrics. Positive class = abnormal move. A row is flagged when score >= threshold."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "flagged": int(tp + fp)}


def bootstrap_auc(y, scores: dict, reps: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Percentile bootstrap of test ROC-AUC: resample test rows with replacement, recompute AUC.

    The same resamples are used for every model, so paired differences are also available.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(reps, len(y)))
    keep = [i for i in idx if 0 < y[i].sum() < len(i)]  # both classes needed for an AUC
    out = {name: np.array([roc_auc_score(y[i], np.asarray(s)[i]) for i in keep]) for name, s in scores.items()}
    return pd.DataFrame(out)


# ------------------------------------------------------------------ cross-validation and tuning
def make_fold_data(rows: pd.DataFrame, folds: list, rule: str = "refined") -> list[dict]:
    """Labels, purge and thresholds are rebuilt inside every fold."""
    out = []
    for i, (tw, vw) in enumerate(folds, start=1):
        d = split_data(rows, tw, vw, rule)
        assert d["eval"].y.sum() > 0, f"fold {i} has no positives"
        d["fold"] = i
        out.append(d)
    return out


def cv_scores(spec: dict, params: dict, fold_data: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold metrics (threshold 0.5) and pooled out-of-fold scores for one setting."""
    per, pooled = [], []
    for d in fold_data:
        _, s = fit_predict(spec, params, d["train"], d["eval"])
        per.append({"fold": d["fold"], "val_week": str(d["eval_weeks"][0])[:10], **evaluate(d["eval"].y, s)})
        pooled.append(d["eval"].assign(score=s, fold=d["fold"]))
    return pd.DataFrame(per), pd.concat(pooled, ignore_index=True)


def tune(name: str, spec: dict, fold_data: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Grid search inside forward-chaining CV, scored by validation ROC-AUC.

    Selection: the best mean AUC, then the simplest setting whose mean is within one fold-SD of the best
    (smaller C, smaller depth with larger leaves, larger K), because the folds are small and noisy.
    """
    rows = []
    for params in spec["grid"]:
        per, _ = cv_scores(spec, params, fold_data)
        rows.append({**params, "mean_auc": per.roc_auc.mean(), "sd_auc": per.roc_auc.std(ddof=1),
                     "mean_pr_auc": per.pr_auc.mean(), **{f"auc_fold{r.fold}": r.roc_auc for r in per.itertuples()}})
    res = pd.DataFrame(rows)
    if len(res) == 1:
        return res, spec["grid"][0]
    best = res.loc[res.mean_auc.idxmax()]
    near = res[res.mean_auc >= best.mean_auc - best.sd_auc].copy()
    if name == "logistic":
        near = near.sort_values(["clf__C", "mean_auc"], ascending=[True, False])
    elif name == "tree":
        near = near.sort_values(["max_depth", "min_samples_leaf", "mean_auc"], ascending=[True, False, False])
    elif name == "knn":
        near = near.sort_values(["clf__n_neighbors", "mean_auc"], ascending=[False, False])
    chosen = near.iloc[0]
    params = {k: chosen[k] for k in spec["grid"][0]}
    params = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
              for k, v in params.items()}
    res["chosen"] = [all(r[k] == v for k, v in params.items()) for _, r in res.iterrows()]
    res["best_mean"] = res.index == best.name
    return res, params


def random_kfold_auc(spec: dict, params: dict, train: pd.DataFrame) -> pd.DataFrame:
    """Leakage demonstration: ordinary shuffled StratifiedKFold on the training rows (ignores time order)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = []
    for i, (a, b) in enumerate(skf.split(train, train.y), start=1):
        _, s = fit_predict(spec, params, train.iloc[a], train.iloc[b])
        out.append({"fold": i, "roc_auc": roc_auc_score(train.y.iloc[b], s),
                    "pr_auc": average_precision_score(train.y.iloc[b], s)})
    return pd.DataFrame(out)


def shuffled_label_test(spec: dict, params: dict, train: pd.DataFrame, test: pd.DataFrame, reps: int = 200) -> np.ndarray:
    """Sanity check: refit on randomly permuted training labels and score the real test labels.

    If nothing leaks, these AUCs scatter around 0.5. One shuffle is noisy with ~40 test positives, so we repeat it.
    The first entry uses seed 42; the share of shuffled AUCs >= the real AUC is a permutation p-value for H3.
    """
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(reps):
        _, s = fit_predict(spec, params, train.assign(y=rng.permutation(train.y.to_numpy())), test)
        out.append(roc_auc_score(test.y, s))
    return np.array(out)


# ------------------------------------------------------------------ asymmetric costs
def best_threshold(y, score, r: float, w=None) -> tuple[float, float]:
    """Threshold that minimises total cost = FP + r * (weighted) FN when rows with score >= t are flagged.

    Candidates are the distinct scores plus +inf ('flag nothing'). Ties go to the higher threshold (fewer alerts).
    """
    y = np.asarray(y).astype(float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    s = np.asarray(score, float)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_s, w_s = s[order], y[order], w[order]
    fn_total = (y_s * w_s).sum()
    cum_tp_w = np.cumsum(y_s * w_s)
    cum_fp = np.cumsum(1 - y_s)
    # Flagging everything with score >= s_sorted[k] means flagging the first (last index of that value) + 1 rows.
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    costs = cum_fp[last] + r * (fn_total - cum_tp_w[last])
    cands = np.r_[np.inf, s_sorted[last]]
    costs = np.r_[r * fn_total, costs]
    k = int(np.argmin(costs))  # first minimum = highest threshold among ties
    return float(cands[k]), float(costs[k])


def expected_cost(y, score, threshold: float, r: float, w=None) -> float:
    """Cost per 1,000 city-item-weeks: (FP + r * weighted FN) / n * 1000."""
    y = np.asarray(y).astype(int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    flag = np.asarray(score) >= threshold
    fp = ((flag == 1) & (y == 0)).sum()
    fn_w = (w * ((flag == 0) & (y == 1))).sum()
    return 1000 * (fp + r * fn_w) / len(y)


def cost_analysis(cv_pooled: dict, test: pd.DataFrame, test_scores: dict, weights: str | None = None) -> pd.DataFrame:
    """For each model and r: choose the threshold on pooled CV predictions, then apply it to the test set once."""
    rows = []
    wt = None if weights is None else test[weights].map(TERCILE_WEIGHT).to_numpy()
    for r in R_VALUES:
        for name, pool in cv_pooled.items():
            wc = None if weights is None else pool[weights].map(TERCILE_WEIGHT).to_numpy()
            t, _ = best_threshold(pool.y, pool.score, r, wc)
            m = evaluate(test.y, test_scores[name], t)
            rows.append({"r": r, "policy": name, "threshold": t, "test_cost_per_1000": expected_cost(
                test.y, test_scores[name], t, r, wt), "test_flagged_per_1000": 1000 * m["flagged"] / len(test),
                "test_recall": m["recall"], "test_precision": m["precision"], "tp": m["tp"], "fp": m["fp"],
                "fn": m["fn"]})
        # Two fixed policies: flag nothing (what the majority-class model does) and flag everything.
        pos, neg = int(test.y.sum()), int((1 - test.y).sum())
        for pol, flag_all in [("flag_nothing", False), ("flag_everything", True)]:
            score = np.full(len(test), 1.0 if flag_all else 0.0)
            rows.append({"r": r, "policy": pol, "threshold": 0.5,
                         "test_cost_per_1000": expected_cost(test.y, score, 0.5, r, wt),
                         "test_flagged_per_1000": 1000.0 if flag_all else 0.0,
                         "test_recall": 1.0 if flag_all else 0.0,
                         "test_precision": pos / len(test) if flag_all else np.nan,
                         "tp": pos if flag_all else 0, "fp": neg if flag_all else 0, "fn": 0 if flag_all else pos})
    return pd.DataFrame(rows)


def alert_budget(test: pd.DataFrame, test_scores: dict, ns=ALERT_N) -> pd.DataFrame:
    """If an analyst can check only the top N city-items each week, how many real abnormal moves are caught?

    Rows tied at the cut-off score are taken at random, so we report the expected count: the tied block contributes
    (places left / tied rows) x (positives in the tied block). This matters for the category rule, whose scores tie.
    """
    rows = []
    for name, s in test_scores.items():
        d = test.assign(score=np.asarray(s))
        for n in ns:
            caught = pos = 0.0
            for _, wk in d.groupby("week_end"):
                sc = wk.score.sort_values(ascending=False)
                cut = sc.iloc[n - 1]
                above = wk[wk.score > cut]
                tied = wk[wk.score == cut]
                left = n - len(above)
                caught += above.y.sum() + left / len(tied) * tied.y.sum()
                pos += wk.y.sum()
            weeks = d.week_end.nunique()
            rows.append({"model": name, "N_per_week": n, "caught": caught, "positives": int(pos),
                         "recall_at_N": caught / pos, "precision_at_N": caught / (n * weeks)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ error analysis and importance
def error_breakdown(test: pd.DataFrame, score, threshold: float) -> pd.DataFrame:
    """FN rate (missed share of real abnormal moves) and FP rate (false-alarm share of normal rows) by group."""
    d = test.assign(flag=(np.asarray(score) >= threshold).astype(int))
    d["week"] = d.week_end.dt.strftime("%Y-%m-%d")
    out = []
    for col, label in [("item_category", "item category"), ("province", "province"), ("pop_tercile", "city-size tercile"),
                       ("dep_tercile", "deprivation tercile"), ("week", "test week")]:
        for lvl, g in d.groupby(col):
            pos, neg = g.y.sum(), (1 - g.y).sum()
            out.append({"grouping": label, "group": lvl, "rows": len(g), "positives": int(pos),
                        "base_rate": g.y.mean(), "flagged": int(g.flag.sum()),
                        "fn_rate": ((g.flag == 0) & (g.y == 1)).sum() / pos if pos else np.nan,
                        "fp_rate": ((g.flag == 1) & (g.y == 0)).sum() / neg if neg else np.nan})
    return pd.DataFrame(out)


def edge_cases(test: pd.DataFrame, score, threshold: float, k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The k most confident false negatives (lowest score among missed positives) and false positives."""
    d = test.assign(score=np.asarray(score), flag=(np.asarray(score) >= threshold).astype(int))
    d["week"] = d.week_end.dt.strftime("%Y-%m-%d")
    cols = ["city", "item_short", "item_category", "week", "score", "dlog_next", "threshold", "abs_dlog", "has_dlog",
            "price_changed", "abs_dlog_lag1", "rel_price", "within_city_range_pct", "single_quote"]
    fn = d[(d.y == 1) & (d.flag == 0)].sort_values("score", kind="mergesort").head(k)[cols]
    fp = d[(d.y == 0) & (d.flag == 1)].sort_values("score", ascending=False, kind="mergesort").head(k)[cols]
    return fn.reset_index(drop=True), fp.reset_index(drop=True)


def importance(logit: Pipeline, tree: DecisionTreeClassifier, features: list) -> pd.DataFrame:
    """Standardised logistic coefficients (odds ratio per 1 SD) and tree impurity importance, with ranks."""
    coef = logit.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": features, "coef_per_sd": coef, "odds_ratio_per_sd": np.exp(coef),
                       "tree_importance": tree.feature_importances_})
    df["rank_logistic"] = df.coef_per_sd.abs().rank(ascending=False, method="min").astype(int)
    df["rank_tree"] = df.tree_importance.rank(ascending=False, method="min").astype(int)
    df.loc[df.tree_importance == 0, "rank_tree"] = np.nan
    return df.sort_values("rank_logistic").reset_index(drop=True)


def tree_rules(tree: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series, min_rows: int = 30) -> pd.DataFrame:
    """Every leaf as an IF-THEN rule with its actual (unweighted) abnormal rate on the training rows."""
    t = tree.tree_
    feats = list(X.columns)
    paths = {}

    def walk(node, conds):
        if t.children_left[node] == -1:
            paths[node] = conds
            return
        f, thr = feats[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.3g}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.3g}"])

    walk(0, [])
    leaf = tree.apply(X)
    g = pd.DataFrame({"leaf": leaf, "y": np.asarray(y)}).groupby("leaf").y.agg(["size", "sum", "mean"])
    out = pd.DataFrame({"leaf": g.index, "rule": ["IF " + " AND ".join(paths[i]) for i in g.index],
                        "train_rows": g["size"].values, "train_positives": g["sum"].values,
                        "abnormal_rate": g["mean"].values})
    return out[out.train_rows >= min_rows].sort_values("abnormal_rate", ascending=False).reset_index(drop=True)


def gini_example(tree: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Worked Gini calculation for the root split, using unweighted training counts (as on the Unit 09 slides)."""
    t = tree.tree_
    f, thr = X.columns[t.feature[0]], t.threshold[0]
    left = X[f] <= thr
    y = np.asarray(y)

    def gini(v):
        p = v.mean() if len(v) else 0.0
        return 1 - p ** 2 - (1 - p) ** 2

    n, nl, nr = len(y), left.sum(), (~left).sum()
    g0, gl, gr = gini(y), gini(y[left.to_numpy()]), gini(y[~left.to_numpy()])
    # The tree was fitted with class_weight="balanced": each positive counts n / (2 x positives), each negative
    # n / (2 x negatives), so both classes carry equal total weight. This is the Gini sklearn actually minimised.
    wp, wn = n / (2 * y.sum()), n / (2 * (n - y.sum()))

    def wstats(v):
        pw, nw = v.sum() * wp, (len(v) - v.sum()) * wn
        q = pw / (pw + nw)
        return pw + nw, 1 - q ** 2 - (1 - q) ** 2

    W0, wg0 = wstats(y)
    Wl, wgl = wstats(y[left.to_numpy()])
    Wr, wgr = wstats(y[~left.to_numpy()])
    weighted = {"w_pos": wp, "w_neg": wn, "wgini_parent": wg0, "wgini_left": wgl, "wgini_right": wgr,
                "wshare_left": Wl / W0, "wgini_weighted_children": Wl / W0 * wgl + Wr / W0 * wgr,
                "wgini_gain": wg0 - (Wl / W0 * wgl + Wr / W0 * wgr)}
    return {**weighted, "feature": f, "threshold": float(thr), "n": int(n), "pos": int(y.sum()), "n_left": int(nl),
            "pos_left": int(y[left.to_numpy()].sum()), "n_right": int(nr), "pos_right": int(y[~left.to_numpy()].sum()),
            "gini_parent": g0, "gini_left": gl, "gini_right": gr,
            "gini_weighted_children": nl / n * gl + nr / n * gr, "gini_gain": g0 - (nl / n * gl + nr / n * gr)}


# ------------------------------------------------------------------ figures
def fig_design(rows: pd.DataFrame, train_weeks: list, test_weeks: list, folds: list, purged_final: list):
    style()
    weeks = labelled_weeks(rows)
    lab = [pd.Timestamp(w).strftime("%d %b %y") for w in weeks]
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    lines = [(f"CV fold {i}", tw, vw) for i, (tw, vw) in enumerate(folds, start=1)] + [("Final train / test", train_weeks, test_weeks)]
    for yi, (name, tw, vw) in enumerate(reversed(lines)):
        for xi, w in enumerate(weeks):
            nxt = rows.loc[rows.week_end == w, "next_week"].iloc[0]
            if w in tw and nxt in vw:
                c, mk = "#bbbbbb", "x"   # purged
            elif w in tw:
                c, mk = "#0072B2", "s"
            elif w in vw:
                c, mk = "#D55E00", "s"
            else:
                continue
            ax.scatter(xi, yi, c=c, marker=mk, s=60)
    ax.set_yticks(range(len(lines)), [n for n, _, _ in reversed(lines)])
    ax.set_xticks(range(len(weeks)), lab, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Labelled week t (week ending)")
    ax.grid(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="s", ls="", c="#0072B2", label="train"),
                       Line2D([], [], marker="s", ls="", c="#D55E00", label="validate / test"),
                       Line2D([], [], marker="x", ls="", c="#bbbbbb", label="purged (t+1 is an evaluation week)")],
              loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    return savefig(fig, "fig15_split_design")


def fig_tuning(tun: dict):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    lg = tun["logistic"]
    ax = axes[0]
    ax.errorbar(range(len(lg)), lg.mean_auc, yerr=lg.sd_auc, marker="o", capsize=3, c=MODEL_COLORS["logistic"])
    ax.set_xticks(range(len(lg)), [str(c) for c in lg["clf__C"]])
    ax.set_xlabel("C (inverse regularisation)"); ax.set_ylabel("CV ROC-AUC (mean ± SD)"); ax.set_title("Logistic regression")
    tr = tun["tree"]
    ax = axes[1]
    for (crit, leaf), g in tr.groupby(["criterion", "min_samples_leaf"]):
        ax.plot(g.max_depth, g.mean_auc, marker="o" if crit == "gini" else "^", ms=3.5, lw=1,
                ls="-" if crit == "gini" else "--", label=f"{crit}, leaf {leaf}")
    ax.set_xlabel("max_depth"); ax.set_title("Decision tree")
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    kn = tun["knn"]
    ax = axes[2]
    for (fs, w, mt), g in kn.groupby(["features", "clf__weights", "clf__metric"]):
        ax.plot(g["clf__n_neighbors"], g.mean_auc, marker="o", ms=3, lw=1, ls="-" if fs == "all" else ":",
                label=f"{fs}, {w}, {mt[:3]}")
    ax.set_xscale("log"); ax.set_xlabel("K (neighbours, log scale)"); ax.set_title("KNN")
    ax.legend(fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    for ax in axes:
        ax.axhline(0.5, c="k", lw=0.6, ls=":")
    return savefig(fig, "fig16_tuning")


def fig_roc_pr(y, test_scores: dict, base_rate: float):
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for name, s in test_scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        a.plot(fpr, tpr, c=MODEL_COLORS[name], lw=1.4, label=f"{MODEL_LABEL[name]} ({roc_auc_score(y, s):.2f})")
        pr, rc, _ = precision_recall_curve(y, s)
        pr, rc = pr[:-1], rc[:-1]  # drop sklearn's artificial end point (recall 0, precision 1)
        b.plot(rc, pr, c=MODEL_COLORS[name], lw=1.4, drawstyle="steps-post",
               label=f"{MODEL_LABEL[name]} ({average_precision_score(y, s):.3f})")
    a.plot([0, 1], [0, 1], c="k", ls=":", lw=0.8, label="random (0.50)")
    a.set_xlabel("False positive rate (1 − specificity)"); a.set_ylabel("True positive rate (recall)")
    a.set_title("ROC curves, test set (AUC)"); a.legend(fontsize=7, loc="lower right")
    b.axhline(base_rate, c="k", ls=":", lw=0.8, label=f"random = base rate ({base_rate:.3f})")
    b.set_xlabel("Recall"); b.set_ylabel("Precision"); b.set_title("Precision-recall curves, test set (PR-AUC)")
    b.legend(fontsize=7, loc="upper right"); b.set_ylim(0, 0.2)
    return savefig(fig, "fig17_roc_pr")


def fig_confusion(cms: list[tuple[str, dict]]):
    """2 x 2 matrices in the Unit 10 layout: actual in rows, predicted in columns, positive class first."""
    style()
    fig, axes = plt.subplots(1, len(cms), figsize=(2.3 * len(cms), 2.5))
    for ax, (title, m) in zip(axes, cms):
        mat = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
        rowpct = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(rowpct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}\n{100 * rowpct[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if rowpct[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1], ["abnormal", "normal"], fontsize=7)
        ax.set_yticks([0, 1], ["abnormal", "normal"], fontsize=7)
        ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("actual", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(False)
    fig.tight_layout()
    return savefig(fig, "fig18_confusion")


def fig_costs(costs: pd.DataFrame):
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    styles = {**{k: (MODEL_COLORS[k], "-") for k in MODEL_COLORS},
              "flag_nothing": ("#666666", ":"), "flag_everything": ("#000000", "--")}
    labels = {**MODEL_LABEL, "flag_nothing": "Flag nothing (= majority class)", "flag_everything": "Flag everything"}
    for pol, g in costs[costs.policy != "majority"].groupby("policy", sort=False):
        c, ls = styles[pol]
        ax.plot(g.r, g.test_cost_per_1000, marker="o", ms=3.5, c=c, ls=ls, label=labels[pol])
    ax.set_xlabel("Cost ratio r = cost(missed abnormal move) / cost(false alarm)")
    ax.set_ylabel("Test cost per 1,000 city-item-weeks\n(false-alarm units)")
    ax.set_xticks(R_VALUES); ax.legend(fontsize=7)
    return savefig(fig, "fig19_costs")


def fig_importance(imp: pd.DataFrame):
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    d = imp.sort_values("coef_per_sd")
    a.barh(d.feature, d.coef_per_sd, color=np.where(d.coef_per_sd > 0, "#D55E00", "#0072B2"))
    a.axvline(0, c="k", lw=0.7)
    a.set_xlabel("Logistic coefficient per 1 SD (log-odds)"); a.set_title("Logistic regression (odds ratio = e^coef)")
    a.tick_params(axis="y", labelsize=7)
    d = imp[imp.tree_importance > 0].sort_values("tree_importance")
    b.barh(d.feature, d.tree_importance, color=MODEL_COLORS["tree"])
    b.set_xlabel("Impurity-based importance (sums to 1)"); b.set_title("Decision tree")
    b.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return savefig(fig, "fig20_importance")


def fig_tree(tree: DecisionTreeClassifier, features: list):
    style()
    fig, ax = plt.subplots(figsize=(13, 5.6))
    plot_tree(tree, feature_names=features, class_names=["normal", "abnormal"], max_depth=3, filled=True,
              impurity=True, proportion=False, rounded=True, fontsize=6, ax=ax)
    return savefig(fig, "fig21_tree")


# ------------------------------------------------------------------ the full pipeline
def run_all(write: bool = True) -> dict:
    """Run every step in order. The notebook does the same steps one by one."""
    N.clear()
    np.random.seed(SEED)
    master = load_master()
    rows = build_features(master)
    weeks = labelled_weeks(rows)
    put("data.labelled_weeks", len(weeks))
    put("data.weeks_in_panel", master.week_end.nunique())
    put("data.candidate_rows", len(rows))
    put("data.cutoff", str(TRAIN_CUTOFF))
    put("data.max_week_used", str(rows.next_week.max().date()))

    # Whole-period labels: for the data card only (the models never see these thresholds).
    allw, _ = make_labels(rows, rows)
    put("data.rows", len(allw)); put("data.positives", int(allw.y.sum())); put("data.base_rate", allw.y.mean())
    put("data.rows_per_week", len(allw) / len(weeks))
    put("data.items_food", rows.item_id.nunique()); put("data.items_labelled", allw.item_id.nunique())
    put("data.share_no_dlog", 1 - allw.has_dlog.mean()); put("data.share_no_lag1", 1 - allw.has_lag1.mean())
    put("data.share_price_changed", allw.price_changed.mean())

    train_weeks, test_weeks = time_split(rows)
    folds = forward_chain_folds(train_weeks)
    final = split_data(rows, train_weeks, test_weeks)
    train, test = final["train"], final["eval"]
    put("split.train_weeks", len(train_weeks)); put("split.test_weeks", len(test_weeks))
    put("split.train_first", str(train_weeks[0])[:10]); put("split.train_last", str(train_weeks[-1])[:10])
    put("split.test_list", ", ".join(str(w)[:10] for w in test_weeks))
    put("split.train_rows", len(train)); put("split.train_pos", int(train.y.sum())); put("split.train_rate", train.y.mean())
    put("split.test_rows", len(test)); put("split.test_pos", int(test.y.sum())); put("split.test_rate", test.y.mean())
    put("split.purged_rows", final["n_purged"]); put("split.purged_week", ", ".join(str(w)[:10] for w in final["purged_weeks"]))
    put("split.train_weeks_after_purge", train.week_end.nunique())
    put("split.items_unlabelled", final["items_unlabelled"])
    put("split.test_rows_per_week", len(test) / len(test_weeks)); put("split.train_items", train.item_id.nunique())
    put("split.week_share_train", len(train_weeks) / len(weeks))
    put("split.row_share_train", len(train) / (len(train) + len(test)))

    # Data card.
    card = pd.DataFrame([{"part": "all labelled weeks (data card only)", "weeks": len(weeks), "rows": len(allw),
                          "positives": int(allw.y.sum()), "base_rate": allw.y.mean(),
                          "share_has_dlog_0": 1 - allw.has_dlog.mean(), "share_has_lag1_0": 1 - allw.has_lag1.mean()}]
                        + [{"part": p, "weeks": d.week_end.nunique(), "rows": len(d), "positives": int(d.y.sum()),
                            "base_rate": d.y.mean(), "share_has_dlog_0": 1 - d.has_dlog.mean(),
                            "share_has_lag1_0": 1 - d.has_lag1.mean()} for p, d in [("train (after purge)", train), ("test", test)]])
    per_week = pd.concat([train.assign(part="train"), test.assign(part="test")]).groupby(["week_end", "part"]).y.agg(
        rows="size", positives="sum", base_rate="mean").reset_index()
    per_week["week_end"] = per_week.week_end.dt.strftime("%Y-%m-%d")

    # Forward-chaining CV folds (labels, purge and thresholds rebuilt in each).
    fold_data = make_fold_data(rows, folds)
    fold_tab = pd.DataFrame([{"fold": d["fold"], "train_weeks": f"{str(d['train_weeks'][0])[:10]} to {str(d['train_weeks'][-1])[:10]}",
                              "n_train_weeks": len(d["train_weeks"]), "val_week": str(d["eval_weeks"][0])[:10],
                              "purged_rows": d["n_purged"], "train_rows": len(d["train"]),
                              "train_pos": int(d["train"].y.sum()), "val_rows": len(d["eval"]),
                              "val_pos": int(d["eval"].y.sum()), "val_rate": d["eval"].y.mean(),
                              "items_unlabelled": d["items_unlabelled"]} for d in fold_data])
    put("cv.folds", len(fold_data)); put("cv.val_pos_min", fold_tab.val_pos.min()); put("cv.val_pos_max", fold_tab.val_pos.max())
    put("cv.purged_folds", int((fold_tab.purged_rows > 0).sum()))

    # Tuning.
    specs = make_models()
    tuning, chosen = {}, {}
    for name, spec in specs.items():
        tuning[name], chosen[name] = tune(name, spec, fold_data)
        if len(spec["grid"]) > 1:
            best = tuning[name].loc[tuning[name].mean_auc.idxmax()]
            sel = tuning[name][tuning[name].chosen].iloc[0]
            put(f"tune.{name}.best_mean_auc", best.mean_auc); put(f"tune.{name}.best_sd_auc", best.sd_auc)
            put(f"tune.{name}.chosen_mean_auc", sel.mean_auc); put(f"tune.{name}.chosen_sd_auc", sel.sd_auc)
            put(f"tune.{name}.grid_size", len(spec["grid"]))
            put(f"tune.{name}.best_params", ", ".join(f"{k.replace('clf__', '')}={best[k]}" for k in spec["grid"][0]))
            put(f"tune.{name}.chosen_params", ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in chosen[name].items()))
    kn = tuning["knn"]
    for fs in ["all", "continuous"]:
        put(f"tune.knn.best_auc_{fs}", kn[kn.features == fs].mean_auc.max())

    # CV metrics of the chosen settings, plus pooled out-of-fold predictions for threshold choice.
    cv_per, cv_pool = {}, {}
    for name, spec in specs.items():
        cv_per[name], cv_pool[name] = cv_scores(spec, chosen[name], fold_data)

    # Final fit on all (purged) training weeks; score the test set once.
    fitted, test_scores = {}, {}
    for name, spec in specs.items():
        fitted[name], test_scores[name] = fit_predict(spec, chosen[name], train, test)

    metrics_cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
    boots = bootstrap_auc(test.y, test_scores)
    put("test.bootstrap_reps_used", len(boots))
    main_rows = []
    for name in specs:
        per = cv_per[name]
        te = evaluate(test.y, test_scores[name], 0.5)
        lo, hi = np.percentile(boots[name], [2.5, 97.5])
        row = {"model": MODEL_LABEL[name], "key": name}
        for c in metrics_cols:
            row[f"cv_{c}_mean"] = per[c].mean(); row[f"cv_{c}_sd"] = per[c].std(ddof=1); row[f"test_{c}"] = te[c]
            put(f"cv.{name}.{c}_mean", per[c].mean()); put(f"cv.{name}.{c}_sd", per[c].std(ddof=1))
            put(f"test.{name}.{c}", te[c])
        row.update({"test_auc_ci_lo": lo, "test_auc_ci_hi": hi, "test_flagged_at_0.5": te["flagged"]})
        put(f"test.{name}.auc_ci_lo", lo); put(f"test.{name}.auc_ci_hi", hi)
        put(f"test.{name}.flagged_at_05", te["flagged"]); put(f"test.{name}.tp_at_05", te["tp"]); put(f"test.{name}.fp_at_05", te["fp"]); put(f"test.{name}.reject_h0", bool(lo > 0.5))
        main_rows.append(row)
    main = pd.DataFrame(main_rows)
    diff = boots["logistic"] - boots["category"]
    put("test.diff_logistic_minus_category", roc_auc_score(test.y, test_scores["logistic"]) - roc_auc_score(test.y, test_scores["category"]))
    put("test.diff_logistic_minus_category_ci_lo", np.percentile(diff, 2.5))
    put("test.diff_logistic_minus_category_ci_hi", np.percentile(diff, 97.5))
    diff_t = boots["tree"] - boots["category"]
    put("test.diff_tree_minus_category_ci_lo", np.percentile(diff_t, 2.5))
    put("test.diff_tree_minus_category_ci_hi", np.percentile(diff_t, 97.5))
    cv_auc = pd.DataFrame({n: cv_per[n].roc_auc.values for n in specs}, index=cv_per["logistic"].fold)
    put("cv.folds_logistic_beats_category", int((cv_auc.logistic > cv_auc.category).sum()))
    put("cv.folds_tree_beats_category", int((cv_auc.tree > cv_auc.category).sum()))
    per_fold = pd.concat([cv_per[n].assign(model=MODEL_LABEL[n]) for n in specs], ignore_index=True)
    ml = ["logistic", "tree", "knn"]
    put("h3.best_test_model", max(ml, key=lambda n: N["test"][n]["roc_auc"]))
    put("h3.n_models_reject", int(sum(N["test"][n]["reject_h0"] for n in ml)))

    # Leakage demonstration and sanity check.
    rk = random_kfold_auc(specs["logistic"], chosen["logistic"], train)
    put("leak.random_kfold_auc_mean", rk.roc_auc.mean()); put("leak.random_kfold_auc_sd", rk.roc_auc.std(ddof=1))
    put("leak.forward_auc_mean", cv_per["logistic"].roc_auc.mean())
    put("leak.gap", rk.roc_auc.mean() - cv_per["logistic"].roc_auc.mean())
    perm = shuffled_label_test(specs["logistic"], chosen["logistic"], train, test)
    put("leak.shuffled_label_test_auc", perm[0]); put("leak.shuffled_reps", len(perm))
    put("leak.shuffled_mean", perm.mean()); put("leak.shuffled_sd", perm.std(ddof=1))
    put("leak.shuffled_p05", np.percentile(perm, 5)); put("leak.shuffled_p95", np.percentile(perm, 95))
    put("leak.shuffled_share_ge_real", (perm >= N["test"]["logistic"]["roc_auc"]).mean())
    assert abs(perm.mean() - 0.5) < 0.03, "shuffled labels still predict on average: something leaks"
    # Transparency: what the purge costs. Same model, trained without dropping the purged week.
    nopurge_train, _ = make_labels(rows[rows.week_end.isin(train_weeks)], rows[rows.week_end.isin(train_weeks)])
    _, s_np = fit_predict(specs["logistic"], chosen["logistic"], nopurge_train, test)
    put("leak.no_purge_logistic_test_auc", roc_auc_score(test.y, s_np))

    # Ablation: category dummies only vs the full feature set (same C).
    abl_spec = {**specs["logistic"], "features": CAT_FEATURES}
    per_abl, _ = cv_scores(abl_spec, chosen["logistic"], fold_data)
    _, s_abl = fit_predict(abl_spec, chosen["logistic"], train, test)
    ablation = pd.DataFrame([
        {"features": "item-category dummies only (3)", "cv_auc_mean": per_abl.roc_auc.mean(), "cv_auc_sd": per_abl.roc_auc.std(ddof=1),
         "test_auc": roc_auc_score(test.y, s_abl), "test_pr_auc": average_precision_score(test.y, s_abl)},
        {"features": f"full feature set ({len(FEATURES)})", "cv_auc_mean": cv_per["logistic"].roc_auc.mean(),
         "cv_auc_sd": cv_per["logistic"].roc_auc.std(ddof=1), "test_auc": N["test"]["logistic"]["roc_auc"],
         "test_pr_auc": N["test"]["logistic"]["pr_auc"]}])
    put("ablation.cat_only_cv_auc", per_abl.roc_auc.mean()); put("ablation.cat_only_test_auc", roc_auc_score(test.y, s_abl))

    # Sensitivity: the M02 contract label (plain 90th percentile).
    con = split_data(rows, train_weeks, test_weeks, rule="contract")
    _, s_con = fit_predict(specs["logistic"], chosen["logistic"], con["train"], con["eval"])
    _, s_con_cat = fit_predict(specs["category"], {}, con["train"], con["eval"])
    put("contract.test_rows", len(con["eval"])); put("contract.test_rate", con["eval"].y.mean())
    put("contract.train_rate", con["train"].y.mean())
    put("contract.logistic_test_auc", roc_auc_score(con["eval"].y, s_con))
    put("contract.category_test_auc", roc_auc_score(con["eval"].y, s_con_cat))

    # Asymmetric costs: thresholds from pooled CV predictions only, then applied to test.
    ml4 = ["category", "logistic", "tree", "knn"]  # the majority model has a constant score; its policy is 'flag nothing'
    costs = cost_analysis({k: cv_pool[k] for k in ml4}, test, {k: test_scores[k] for k in ml4})
    costs_eq = cost_analysis({k: cv_pool[k] for k in ml4}, test, {k: test_scores[k] for k in ml4}, weights="dep_tercile")
    for r in R_VALUES:
        for pol in ["category", "logistic", "tree", "knn", "flag_nothing", "flag_everything"]:
            c = costs[(costs.r == r) & (costs.policy == pol)].iloc[0]
            put(f"cost.r{r}.{pol}.cost", c.test_cost_per_1000); put(f"cost.r{r}.{pol}.recall", c.test_recall)
            put(f"cost.r{r}.{pol}.flagged", c.test_flagged_per_1000)
            if pol in ml4:
                put(f"cost.r{r}.{pol}.threshold", c.threshold); put(f"cost.r{r}.{pol}.precision", c.test_precision)
        sub = costs[(costs.r == r) & costs.policy.isin(["category", "logistic", "tree", "knn", "flag_nothing", "flag_everything"])]
        put(f"cost.r{r}.cheapest", sub.loc[sub.test_cost_per_1000.idxmin(), "policy"])
    put("cost.inverse_base_rate", 1 / train.y.mean())
    put("cost.r_ref", R_REF)
    for r in [10, 20, 35]:
        sub = costs_eq[costs_eq.r == r]
        put(f"cost_equity.r{r}.cheapest", sub.loc[sub.test_cost_per_1000.idxmin(), "policy"])
        put(f"cost_equity.r{r}.logistic_threshold", sub[sub.policy == "logistic"].threshold.iloc[0])
        put(f"cost_equity.r{r}.logistic_recall", sub[sub.policy == "logistic"].test_recall.iloc[0])
        put(f"cost_equity.r{r}.logistic_cost", sub[sub.policy == "logistic"].test_cost_per_1000.iloc[0])

    # Metrics at the reference cost-based threshold (r = R_REF).
    ref_thr = {n: costs[(costs.r == R_REF) & (costs.policy == n)].threshold.iloc[0] for n in ml4}
    at_ref = []
    for name in ml4:
        te = evaluate(test.y, test_scores[name], ref_thr[name])
        at_ref.append({"model": MODEL_LABEL[name], "threshold": ref_thr[name], **{c: te[c] for c in metrics_cols if c not in ("roc_auc", "pr_auc")},
                       "tp": te["tp"], "fp": te["fp"], "fn": te["fn"], "tn": te["tn"]})
        for c in ["accuracy", "precision", "recall", "specificity", "f1", "tp", "fp", "fn", "tn"]:
            put(f"test_ref.{name}.{c}", te[c])
        put(f"test_ref.{name}.threshold", ref_thr[name]); put(f"test_ref.{name}.flagged_share", te["flagged"] / len(test))
    at_ref = pd.DataFrame(at_ref)

    # Alert budget.
    budget = alert_budget(test, {k: test_scores[k] for k in ["category", "logistic", "tree", "knn"]})
    for r_ in budget.itertuples():
        put(f"budget.{r_.model}.n{r_.N_per_week}.recall", r_.recall_at_N)
        put(f"budget.{r_.model}.n{r_.N_per_week}.precision", r_.precision_at_N)
        put(f"budget.{r_.model}.n{r_.N_per_week}.caught", r_.caught)
    put("budget.test_positives", int(test.y.sum()))
    put("budget.random_precision", test.y.mean())

    # Confusion matrices: logistic / tree / KNN at 0.5 and at the r = R_REF threshold, plus the majority class.
    cms = [("Majority class", evaluate(test.y, test_scores["majority"], 0.5))]
    cm_rows = []
    for name in ["logistic", "tree", "knn"]:
        for lab, t in [("threshold 0.5", 0.5), (f"cost threshold, r = {R_REF}", ref_thr[name])]:
            m_ = evaluate(test.y, test_scores[name], t)
            cms.append((f"{MODEL_LABEL[name]}\n{lab} ({t:.2f})", m_))
    for title, m_ in cms:
        cm_rows.append({"matrix": title.replace("\n", ", "), "TP": m_["tp"], "FN": m_["fn"], "FP": m_["fp"], "TN": m_["tn"],
                        "recall (TP row %)": m_["recall"], "specificity (TN row %)": m_["specificity"]})
    cm_tab = pd.DataFrame(cm_rows)

    # Error analysis for the logistic model at the reference threshold.
    errs = error_breakdown(test, test_scores["logistic"], ref_thr["logistic"])
    fn_cases, fp_cases = edge_cases(test, test_scores["logistic"], ref_thr["logistic"])
    d = test.assign(flag=(test_scores["logistic"] >= ref_thr["logistic"]).astype(int))
    fn_all, tp_all = d[(d.y == 1) & (d.flag == 0)], d[(d.y == 1) & (d.flag == 1)]
    put("errors.n_fn", len(fn_all)); put("errors.n_tp", len(tp_all))
    for c in ["price_changed", "has_dlog", "single_quote", "cat_perishable"]:
        put(f"errors.fn_share_{c}", fn_all[c].mean()); put(f"errors.tp_share_{c}", tp_all[c].mean())
    put("errors.fn_share_flat_or_unobserved", (fn_all.price_changed == 0).mean())
    put("errors.positives_flat_or_unobserved_rate",
        d[d.price_changed == 0].y.mean())
    put("errors.positives_changed_rate", d[d.price_changed == 1].y.mean())
    for e in errs.itertuples():
        put(f"errors.by.{e.grouping.replace(' ', '_').replace('-', '_')}.{e.group}.fn_rate", e.fn_rate)
        put(f"errors.by.{e.grouping.replace(' ', '_').replace('-', '_')}.{e.group}.fp_rate", e.fp_rate)
        put(f"errors.by.{e.grouping.replace(' ', '_').replace('-', '_')}.{e.group}.positives", e.positives)
        put(f"errors.by.{e.grouping.replace(' ', '_').replace('-', '_')}.{e.group}.base_rate", e.base_rate)

    # Importance and rules.
    imp = importance(fitted["logistic"], fitted["tree"], FEATURES)
    for r_ in imp.itertuples():
        put(f"imp.{r_.feature}.coef", r_.coef_per_sd); put(f"imp.{r_.feature}.odds_ratio", r_.odds_ratio_per_sd)
        put(f"imp.{r_.feature}.tree", r_.tree_importance); put(f"imp.{r_.feature}.rank_logistic", r_.rank_logistic)
    rules = tree_rules(fitted["tree"], train[FEATURES], train.y)
    for i, r_ in enumerate(rules.head(3).itertuples(), start=1):
        put(f"rules.top{i}.rate", r_.abnormal_rate); put(f"rules.top{i}.rows", r_.train_rows)
    put("rules.tree_leaves", fitted["tree"].get_n_leaves()); put("rules.tree_depth", fitted["tree"].get_depth())
    gex = gini_example(fitted["tree"], train[FEATURES], train.y)
    for k, v in gex.items():
        put(f"gini_example.{k}", v)
    rules_text = export_text(fitted["tree"], feature_names=FEATURES, show_weights=False)
    cat_rates = fitted["category"].rates_
    for k, v in cat_rates.items():
        put(f"category_rule.rate.{k}", v)
    corr = train[["abs_dlog", "price_changed"]].corr().iloc[0, 1]
    put("imp.corr_abs_dlog_price_changed", corr)

    out = {"rows": rows, "train": train, "test": test, "final": final, "folds": folds, "fold_data": fold_data,
           "fold_tab": fold_tab, "card": card, "per_week": per_week, "specs": specs, "tuning": tuning, "chosen": chosen,
           "cv_per": cv_per, "cv_pool": cv_pool, "per_fold": per_fold, "fitted": fitted, "test_scores": test_scores,
           "main": main, "boots": boots, "random_kfold": rk, "ablation": ablation, "costs": costs, "costs_eq": costs_eq,
           "ref_thr": ref_thr, "at_ref": at_ref, "budget": budget, "cm_tab": cm_tab, "cms": cms, "errors": errs,
           "fn_cases": fn_cases, "fp_cases": fp_cases, "importance": imp, "rules": rules, "rules_text": rules_text,
           "gini_example": gex, "train_weeks": train_weeks, "test_weeks": test_weeks, "shuffled_aucs": perm}
    if write:
        write_outputs(out)
    return out


def write_outputs(o: dict):
    """Tables, figures and the numbers JSON."""
    save_table(o["card"], "data_card"); save_table(o["per_week"], "rows_per_week"); save_table(o["fold_tab"], "folds")
    for name, t in o["tuning"].items():
        if len(t) > 1:
            save_table(t, f"tuning_{name}")
    save_table(o["per_fold"], "cv_per_fold"); save_table(o["main"], "metrics_main"); save_table(o["random_kfold"], "random_kfold")
    save_table(o["ablation"], "ablation"); save_table(o["costs"], "costs"); save_table(o["costs_eq"], "costs_equity")
    save_table(o["at_ref"], "metrics_at_cost_threshold"); save_table(o["budget"], "alert_budget")
    save_table(o["cm_tab"], "confusion_matrices"); save_table(o["errors"], "error_breakdown")
    save_table(o["fn_cases"], "edge_false_negatives"); save_table(o["fp_cases"], "edge_false_positives")
    save_table(o["importance"], "importance"); save_table(o["rules"], "tree_rules")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "tree_rules.txt").write_text(o["rules_text"], encoding="utf-8")
    save_table(pd.DataFrame({"feature": FEATURES, "why_it_might_matter": [FEATURE_WHY[f] for f in FEATURES]}), "features")
    fig_design(o["rows"], o["train_weeks"], o["test_weeks"], o["folds"], o["final"]["purged_weeks"])
    fig_tuning(o["tuning"])
    fig_roc_pr(o["test"].y, o["test_scores"], o["test"].y.mean())
    fig_confusion([c for c in o["cms"] if c[0] == "Majority class" or "Logistic" in c[0] or "tree" in c[0]])
    fig_costs(o["costs"])
    fig_importance(o["importance"])
    fig_tree(o["fitted"]["tree"], FEATURES)
    save_numbers()
