"""Load a task: the training file, the scoring file, the labels the grader keeps."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .spec import TASK_BY_ID, TaskSpec


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


@dataclass
class TaskData:
    spec: TaskSpec
    train: pd.DataFrame
    eval_features: pd.DataFrame
    labels: pd.DataFrame
    meta: Dict[str, Any]
    private: Optional[pd.DataFrame] = None

    @property
    def target(self) -> str:
        return self.spec.target

    def features(self) -> list:
        """Training feature columns: everything but the join key and the target."""
        return [c for c in self.train.columns if c not in ("row_id", self.target)]

    def eval_features_columns(self) -> list:
        return [c for c in self.eval_features.columns if c != "row_id"]


def load_meta(root: Optional[str] = None) -> Dict[str, Any]:
    root = root or repo_root()
    with io.open(os.path.join(root, "data", "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_task(task_id: str, root: Optional[str] = None) -> TaskData:
    root = root or repo_root()
    spec = TASK_BY_ID[task_id]
    d = os.path.join(root, "data", task_id)
    train = pd.read_csv(os.path.join(d, "train.csv"))
    ev = pd.read_csv(os.path.join(d, "eval_features.csv"))
    labels = pd.read_csv(os.path.join(root, "data", "_labels", task_id + ".csv"))
    private = None
    p = os.path.join(root, "data", "_private", task_id + "_potential.csv")
    if os.path.exists(p):
        private = pd.read_csv(p)
    return TaskData(spec=spec, train=train, eval_features=ev, labels=labels,
                    meta=load_meta(root), private=private)
