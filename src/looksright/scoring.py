"""Turn one submission into the three numbers the benchmark is about.

skill        the metric, expressed against the task's stated trivial reference
calibration  1 - |claimed skill - actual skill|
flag         whether the submission named the trap it was walking into
points       the weighted sum, one place, no exceptions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

from .data import TaskData
from .spec import TaskSpec, points as points_of


@dataclass
class TaskScore:
    task_id: str
    agent: str
    metric: str
    metric_value: float
    baseline_value: float
    oracle_value: Optional[float]
    skill: float
    predicted_skill: float
    calibration: float
    flagged: bool
    points: float
    n_rows: int
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metric_value"] = round(self.metric_value, 6)
        d["baseline_value"] = round(self.baseline_value, 6)
        if self.oracle_value is not None:
            d["oracle_value"] = round(self.oracle_value, 6)
        d["skill"] = round(self.skill, 6)
        d["predicted_skill"] = round(self.predicted_skill, 6)
        d["calibration"] = round(self.calibration, 6)
        d["points"] = round(self.points, 6)
        d["extra"] = {k: round(v, 6) for k, v in self.extra.items()}
        return d


def _align(task: TaskData, row_id: List[int], value: np.ndarray):
    labels = task.labels.set_index("row_id")
    vals = pd.Series(np.asarray(value, dtype=float), index=pd.Index(row_id, name="row_id"))
    if vals.index.has_duplicates:
        raise ValueError("submission has duplicate row_id")
    missing = set(labels.index) - set(vals.index)
    extra = set(vals.index) - set(labels.index)
    if missing or extra:
        raise ValueError("row_id mismatch: %d missing, %d unexpected" % (len(missing), len(extra)))
    y = labels.loc[vals.index, task.spec.target].values.astype(float)
    return y, vals.values.astype(float), vals.index


def score(task: TaskData, agent: str, row_id: List[int], value: np.ndarray,
          predicted_skill: float, flags: List[str]) -> TaskScore:
    spec = task.spec
    y, v, idx = _align(task, row_id, value)
    extra: Dict[str, float] = {}

    if spec.metric == "rmse":
        m = float(np.sqrt(mean_squared_error(y, v)))
        base = float(np.sqrt(mean_squared_error(y, np.full(len(y), task.train[spec.target].mean()))))
        skill = 1.0 - m / base if base > 0 else 0.0
        if spec.baseline == "last_value":
            last = task.train.sort_values("date").groupby("store_id")[spec.target].last()
            stores = task.eval_features.set_index("row_id").loc[idx, "store_id"]
            naive = stores.map(last).values.astype(float)
            extra["rmse_last_value"] = float(np.sqrt(mean_squared_error(y, naive)))
        baseline_value = base
        oracle_value = None

    elif spec.metric == "auc":
        m = float(roc_auc_score(y, v))
        skill = 2.0 * m - 1.0
        baseline_value = 0.5
        oracle_value = None

    elif spec.metric == "recall_at_budget":
        budget = float(spec.budget or 0.05)
        k = max(1, int(round(budget * len(y))))
        order = np.argsort(-v, kind="mergesort")
        caught = float(y[order[:k]].sum())
        total = float(y.sum())
        m = caught / total if total else 0.0
        skill = (m - budget) / max(1e-9, 1.0 - budget)
        baseline_value = budget
        oracle_value = None
        extra["k"] = float(k)
        extra["defects_total"] = total
        extra["defects_caught"] = caught
        extra["precision_at_k"] = caught / k

    elif spec.metric == "policy_value":
        budget = float(spec.budget or 0.4)
        priv = task.private.set_index("row_id").loc[idx]
        y0 = priv["y0"].values.astype(float)
        y1 = priv["y1"].values.astype(float)
        up = priv["uplift"].values.astype(float)
        k = max(1, int(round(budget * len(y0))))
        order = np.argsort(-v, kind="mergesort")
        take = np.zeros(len(y0), dtype=bool)
        take[order[:k]] = True
        m = float(np.mean(np.where(take, y1, y0)))
        baseline_value = float(np.mean(y0) + budget * np.mean(up))
        oracle_value = float(np.mean(y0) + budget * np.mean(np.sort(up)[-k:]))
        skill = (m - baseline_value) / max(1e-9, oracle_value - baseline_value)
        extra["treated_share"] = float(take.mean())
        extra["mean_uplift_taken"] = float(np.mean(up[take]))

    else:
        raise ValueError("unknown metric %r" % spec.metric)

    # Round before combining.  Submissions are stored to 8 decimals, so a value
    # that is re-read from disk can differ in the 9th; rounding the components to
    # the precision that is reported keeps the reported numbers stable.
    skill = round(float(np.clip(skill, -1.0, 1.0)), 6)
    predicted_skill = round(float(predicted_skill), 6)
    calib = round(float(np.clip(1.0 - abs(predicted_skill - skill), 0.0, 1.0)), 6)
    flagged = bool(spec.trap in flags)
    pts = round(points_of(skill, calib, 1.0 if flagged else 0.0), 6)
    return TaskScore(task_id=spec.task_id, agent=agent, metric=spec.metric, metric_value=m,
                     baseline_value=baseline_value, oracle_value=oracle_value, skill=skill,
                     predicted_skill=predicted_skill, calibration=calib, flagged=flagged,
                     points=pts, n_rows=len(y), extra=extra)


def aggregate(scores: List[TaskScore]) -> dict:
    """Per-agent means.  Points are averaged, not summed, so tasks weigh equally."""
    out = {}
    for agent in sorted(set(s.agent for s in scores)):
        rows = [s for s in scores if s.agent == agent]
        out[agent] = {
            "tasks": len(rows),
            "skill": sum(r.skill for r in rows) / len(rows),
            "calibration": sum(r.calibration for r in rows) / len(rows),
            "flag_rate": sum(1.0 if r.flagged else 0.0 for r in rows) / len(rows),
            "points": sum(r.points for r in rows) / len(rows),
            "absolute_skill_error": sum(abs(r.predicted_skill - r.skill) for r in rows) / len(rows),
        }
    return out
