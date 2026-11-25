"""Encoders, candidate models, fold makers, and the CV score.

Only three model families are used, deliberately: the benchmark is about the
decision around the model, not about squeezing a leaderboard.  A0, A1 and A2 all
draw from the same candidate list, so any difference between them comes from what
they do to the data and how they choose, never from a better learner.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit


# ---------------------------------------------------------------- encoding

class Design:
    def __init__(self, xtr: np.ndarray, y: np.ndarray, xev: np.ndarray, names: List[str]):
        self.xtr = xtr
        self.y = y
        self.xev = xev
        self.names = names


def encode(train: pd.DataFrame, ev: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str],
           target: str, date_col: Optional[str] = None) -> Design:
    """One-hot everything categorical, keep numerics as they are.

    NaN is left in place: the gradient-boosted trees handle it natively and the
    linear model sits behind a median imputer, so missingness is never silently
    turned into a number by the encoder itself.
    """
    tr = train.copy()
    evv = ev.copy()
    if date_col is not None and date_col in tr.columns:
        for df in (tr, evv):
            t = pd.to_datetime(df[date_col], errors="coerce")
            df[date_col + "__t"] = (t - pd.Timestamp("2024-01-01")).dt.days.astype(float)

    cols = [c for c in numeric]
    if date_col is not None:
        cols = cols + [date_col + "__t"]
    cat = [c for c in categorical]

    trn = tr[cols].astype(float) if cols else pd.DataFrame(index=tr.index)
    evn = evv[cols].astype(float) if cols else pd.DataFrame(index=evv.index)
    if cat:
        trc = pd.get_dummies(tr[cat].astype(str), dummy_na=False)
        evc = pd.get_dummies(evv[cat].astype(str), dummy_na=False)
        trc, evc = trc.align(evc, join="outer", axis=1, fill_value=0)
        x_tr = pd.concat([trn, trc], axis=1) if len(cols) else trc
        x_ev = pd.concat([evn, evc], axis=1) if len(cols) else evc
    else:
        x_tr, x_ev = trn, evn
    x_tr = x_tr.reindex(columns=x_tr.columns, fill_value=0.0)
    x_ev = x_ev.reindex(columns=x_tr.columns, fill_value=0.0)
    return Design(x_tr.values.astype(float), np.asarray(train[target], dtype=float),
                  x_ev.values.astype(float), list(x_tr.columns))


# ---------------------------------------------------------------- candidates

def candidates(kind: str, wide: bool = False) -> List[Tuple[str, str, dict]]:
    """(family, variant, params).  `wide` is only used by the winner-curse task,
    where the point is that many candidates on a small sample is a trap."""
    out = []
    if wide:
        for c in (0.03, 0.3, 3.0):
            out.append(("lr", "C%s" % c, {"C": c}))
        for leaf in (1, 2, 5, 15):
            out.append(("rf", "leaf%d" % leaf, {"min_samples_leaf": leaf}))
        for leaf in (3, 7, 15, 31, 63):
            out.append(("hgb", "leaf%d" % leaf, {"max_leaf_nodes": leaf}))
        return out
    out.append(("lr", "C1.0", {"C": 1.0}))
    out.append(("lr", "C0.1", {"C": 0.1}))
    out.append(("rf", "leaf5", {"min_samples_leaf": 5}))
    out.append(("hgb", "leaf15", {"max_leaf_nodes": 15}))
    return out


def build(kind: str, family: str, params: dict, seed: int, balanced: bool = False):
    if kind == "binary":
        if family == "lr":
            return LogisticRegression(C=params.get("C", 1.0), max_iter=3000, solver="lbfgs",
                                      class_weight="balanced" if balanced else None)
        if family == "rf":
            return RandomForestClassifier(
                n_estimators=70, min_samples_leaf=params.get("min_samples_leaf", 5),
                max_features=0.6 if not balanced else "sqrt", n_jobs=1, random_state=seed,
                class_weight="balanced_subsample" if balanced else None)
        if family == "hgb":
            return HistGradientBoostingClassifier(
                max_iter=110, learning_rate=0.10, max_leaf_nodes=params.get("max_leaf_nodes", 15),
                min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
                random_state=seed, class_weight="balanced" if balanced else None)
    else:
        if family == "lr":
            return Ridge(alpha=1.0 / max(params.get("C", 1.0), 1e-6))
        if family == "rf":
            return RandomForestRegressor(
                n_estimators=70, min_samples_leaf=params.get("min_samples_leaf", 5),
                max_features=0.6, n_jobs=1, random_state=seed)
        if family == "hgb":
            return HistGradientBoostingRegressor(
                max_iter=110, learning_rate=0.10, max_leaf_nodes=params.get("max_leaf_nodes", 15),
                min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
                random_state=seed)
    raise ValueError("unknown model %s/%s" % (kind, family))


class MedianImputer:
    """Median for numerics, 0 for one-hot columns (they are already sparse)."""

    def __init__(self):
        self.med = None

    def fit(self, x: np.ndarray):
        with np.errstate(all="ignore"):
            self.med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype=float, copy=True)
        idx = np.where(~np.isfinite(out))
        out[idx] = np.take(self.med, idx[1])
        return out


def fit_predict(kind: str, family: str, params: dict, xtr: np.ndarray, y: np.ndarray,
                xva: np.ndarray, seed: int, balanced: bool = False) -> np.ndarray:
    model = build(kind, family, params, seed, balanced)
    xa, xb = xtr, xva
    if family != "hgb":
        # The boosted trees split on NaN natively; the forest and the linear model
        # need a number, so they get the training median.
        imp = MedianImputer().fit(xtr)
        xa, xb = imp.transform(xtr), imp.transform(xva)
    if kind == "binary":
        model.fit(xa, y)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(xb)
            return p[:, 1] if p.shape[1] > 1 else np.zeros(len(xb))
        return model.decision_function(xb)
    model.fit(xa, y)
    return model.predict(xb)


# ---------------------------------------------------------------- folds

def folds(n: int, mode: str, seed: int, k: int = 4,
          groups: Optional[np.ndarray] = None,
          order: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """mode: random | group | time | nested_inner (same as random but with its own seed)."""
    if mode == "group" and groups is not None:
        gkf = GroupKFold(n_splits=k)
        return [(tr, va) for tr, va in gkf.split(np.zeros(n), groups=groups)]
    if mode == "time":
        idx = np.arange(n) if order is None else np.asarray(order)
        tss = TimeSeriesSplit(n_splits=k)
        return [(idx[tr], idx[va]) for tr, va in tss.split(idx)]
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in kf.split(np.arange(n))]


# ---------------------------------------------------------------- CV score

def cv_value(kind: str, metric: str, y_true: np.ndarray, pred: np.ndarray) -> float:
    """The number a fold reports.  Higher is better for every metric."""
    if metric == "auc":
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, pred))
    if metric == "ap":
        if len(np.unique(y_true)) < 2:
            return float(np.mean(y_true))
        return float(average_precision_score(y_true, pred))
    if metric == "rmse":
        return -float(np.sqrt(mean_squared_error(y_true, pred)))
    raise ValueError("no CV value for metric %r" % metric)
