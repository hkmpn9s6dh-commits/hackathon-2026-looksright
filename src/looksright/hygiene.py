"""The seven generic health checks behind A1.

This is the whole of A1.  Every rule below is applied uniformly to every task;
none of them looks at the task's name, and none of them knows which trap the
task is carrying.  Each rule is something a careful engineer does to any table
before modelling it, and each one reports a `code` when it fires.

What the suite deliberately cannot do
-------------------------------------
It reads columns, not decisions.  It has no idea what the business measures, so
it cannot notice that a model is being selected on the wrong metric, that the
question is causal rather than predictive, or that the best of twelve candidates
is an over-estimate.  Those three failures are structural, not hygienic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TIME_HINTS = ("date", "datetime", "timestamp", "time", "ds", "month_end", "日期", "时间")
ID_HINTS = ("id", "uuid", "guid", "key", "编号", "序号")
SENTINEL_CODES = (-9999.0, -999.0, -99.0, -9.0, 9999.0, 999.0, 0.0)


@dataclass
class Diagnosis:
    """What the checks found.  Empty lists mean a clean table."""

    dup_groups: Optional[pd.Series] = None  # per-training-row group id; None when no duplicates
    n_dup_rows: int = 0
    sentinel_cols: Dict[str, float] = field(default_factory=dict)  # col -> count replaced
    sentinel_rules: Dict[str, tuple] = field(default_factory=dict)  # col -> (lo, hi) that failed
    id_cols: List[str] = field(default_factory=list)
    time_col: Optional[str] = None
    unbalanced: bool = False
    minority_share: float = 1.0
    leak_cols: List[str] = field(default_factory=list)
    train_only_cols: List[str] = field(default_factory=list)
    constant_cols: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.n_dup_rows or self.sentinel_cols or self.id_cols or self.time_col
                    or self.unbalanced or self.leak_cols or self.train_only_cols
                    or self.constant_cols)

    def summary(self) -> str:
        bits = []
        if self.n_dup_rows:
            bits.append("%d duplicate rows" % self.n_dup_rows)
        if self.sentinel_cols:
            bits.append("sentinels in " + ",".join(sorted(self.sentinel_cols)))
        if self.id_cols:
            bits.append("ids " + ",".join(self.id_cols))
        if self.time_col:
            bits.append("time " + self.time_col)
        if self.unbalanced:
            bits.append("minority %.1f%%" % (100 * self.minority_share))
        if self.leak_cols:
            bits.append("near-perfect " + ",".join(self.leak_cols))
        if self.train_only_cols:
            bits.append("train-only " + ",".join(self.train_only_cols))
        if self.constant_cols:
            bits.append("constant " + ",".join(self.constant_cols))
        return "; ".join(bits) if bits else "nothing found"


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns
            if c != "row_id" and pd.api.types.is_numeric_dtype(df[c])]


def _looks_unique(s: pd.Series, n: int) -> bool:
    if n < 20:
        return False
    try:
        return s.nunique(dropna=False) >= n
    except TypeError:
        return False


def _name_is_id(col: str) -> bool:
    low = col.lower()
    return any(low == h or low.endswith("_" + h) for h in ID_HINTS) or low in ID_HINTS


def _robust_out_of_band(s: pd.Series) -> float:
    """Count values that sit absurdly far from a robust centre."""
    v = pd.to_numeric(s, errors="coerce").dropna().values
    if len(v) < 30:
        return 0.0
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    iqr = max(q3 - q1, 1e-9)
    return float(np.sum((v < med - 40 * iqr) | (v > med + 40 * iqr)))


def scan(train: pd.DataFrame, ev_feats: pd.DataFrame, columns_meta: list, target: str) -> Diagnosis:
    d = Diagnosis()
    n = len(train)
    feats = [c for c in train.columns if c not in ("row_id", target)]

    # 1. duplicate rows -----------------------------------------------------
    key = train[feats].astype(str).agg("|".join, axis=1)
    counts = key.value_counts()
    dups = counts[counts > 1]
    if len(dups):
        d.n_dup_rows = int(dups.sum() - len(dups))  # extra copies beyond the first
        d.dup_groups = key.map({k: i for i, k in enumerate(dups.index)}).fillna(-1).astype(int)
        d.dup_groups[d.dup_groups < 0] = -1
        d.flags.append("duplicate_rows")
        d.notes.append("%d extra copies of %d distinct rows" % (d.n_dup_rows, len(dups)))

    # 2. sentinel values ----------------------------------------------------
    ranges = {c["name"]: c.get("plausible_range") for c in columns_meta
              if c.get("plausible_range")}
    for c in _numeric_cols(train):
        if c in (target, "row_id"):
            continue
        s = pd.to_numeric(train[c], errors="coerce")
        bad = 0
        rng = ranges.get(c)
        if rng:
            lo, hi = rng
            bad = int(((s < lo) | (s > hi)).sum())
            if bad and bad < 0.001 * n:
                bad = 0  # noise, not a coding convention
        if bad == 0:
            bad = int(_robust_out_of_band(s))
            if bad and bad < 0.001 * n:
                bad = 0
        if bad:
            d.sentinel_cols[c] = float(bad)
            d.sentinel_rules[c] = (rng[0], rng[1]) if rng else (float("-inf"), float("inf"))
    if d.sentinel_cols:
        d.flags.append("sentinel_values")
        d.notes.append("out-of-range codes in " + ",".join(sorted(d.sentinel_cols)))

    # 3. identifier-like columns -------------------------------------------
    for c in feats:
        if _looks_unique(train[c], n) and (not pd.api.types.is_numeric_dtype(train[c])
                                           or _name_is_id(c)):
            d.id_cols.append(c)
    if d.id_cols:
        d.flags.append("pseudo_id")

    # 4. columns that will not exist at scoring time ------------------------
    ev_cols = set(c for c in ev_feats.columns if c != "row_id")
    d.train_only_cols = [c for c in feats if c not in ev_cols]
    d.constant_cols = [c for c in feats
                       if c not in d.train_only_cols and train[c].nunique(dropna=False) <= 1]
    if d.train_only_cols:
        d.flags.append("post_outcome_feature")

    # 5. time ordering ------------------------------------------------------
    for c in feats:
        if c in d.train_only_cols:
            continue
        low = c.lower()
        if any(h in low for h in TIME_HINTS) and not pd.api.types.is_numeric_dtype(train[c]):
            d.time_col = c
            break
    if d.time_col:
        d.flags.append("temporal_split")

    # 6. a single feature that nearly separates the outcome -----------------
    if target in train.columns:
        y = train[target].values
        is_binary = (len(np.unique(y[~pd.isna(y)])) == 2)
        from sklearn.metrics import roc_auc_score
        for c in _numeric_cols(train):
            if c in ("row_id", target) or c in d.id_cols:
                continue
            v = pd.to_numeric(train[c], errors="coerce")
            mask = v.notna().values
            if mask.sum() < 50 or len(np.unique(v[mask])) < 3:
                continue
            try:
                if is_binary:
                    auc = roc_auc_score(y[mask], v[mask])
                    strength = max(auc, 1.0 - auc)
                else:
                    strength = abs(float(np.corrcoef(v[mask], y[mask])[0, 1]))
            except Exception:
                continue
            if strength >= 0.90:
                d.leak_cols.append(c)
        if d.leak_cols:
            d.flags.append("post_outcome_feature")
            d.notes.append("single-feature separation >= 0.90 on " + ",".join(d.leak_cols))

    # 7. class imbalance ----------------------------------------------------
    if target in train.columns:
        y = train[target].dropna().values
        u = np.unique(y)
        if len(u) == 2:
            share = min((y == u[0]).mean(), (y == u[1]).mean())
            d.minority_share = float(share)
            if share < 0.15:
                d.unbalanced = True
    return d
