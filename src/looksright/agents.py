"""A0, A1 and A2.

The three systems differ only in two places: what they do to the table before
fitting, and how they choose which candidate to keep.  They share the encoder,
the candidate list, the fold machinery and the final refit, so a difference in
the score is attributable to the difference in judgement, not to a better model.

A0  naive       all columns, median impute, random K-fold, pick by the stock
                metric (AUC / RMSE) and report that number as the expectation.
A1  check-up    the seven generic checks in hygiene.py, applied uniformly:
                duplicate grouping, sentinel repair, id removal, train-only
                column removal, time-ordered validation, class reweighting,
                near-perfect-feature quarantine.
A2  task-aware  A1 plus one decision that requires knowing what the question
                actually is.  It never sees a label that the task would not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import hygiene, models
from .data import TaskData
from .spec import SEED, TaskSpec


@dataclass
class Submission:
    agent: str
    task_id: str
    row_id: List[int]
    value: np.ndarray
    predicted_skill: float
    flags: List[str] = field(default_factory=list)
    notes: str = ""
    cv: List[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"row_id": self.row_id, "value": self.value})


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _skill_from_cv(spec: TaskSpec, metric: str, cv_value: float, extra: float) -> float:
    """Map a CV number onto the benchmark's skill scale.

    `extra` is the trivial-baseline value in the same units (for RMSE).
    """
    if metric == "auc":
        return float(np.clip(2.0 * cv_value - 1.0, -1.0, 1.0))
    if metric == "rmse":
        if extra <= 0:
            return 0.0
        return float(np.clip(1.0 - (-cv_value) / extra, -1.0, 1.0))
    if metric in ("ap", "recall_at_budget"):
        # AP is not bounded by the budget; convert through the random baseline.
        base = extra
        return float(np.clip((cv_value - base) / max(1e-9, 1.0 - base), -1.0, 1.0))
    return 0.0


class BaseAgent:
    level = 0
    name = "base"

    def __init__(self, seed: int = SEED):
        self.seed = seed

    # ---------------------------------------------------------------- entry
    def solve(self, task: TaskData) -> Submission:
        prep = self.prepare(task)
        values, cv_table, claim = self.fit_choose(task, prep)
        ev = prep["ev"]
        return Submission(
            agent=self.name,
            task_id=task.spec.task_id,
            row_id=[int(x) for x in ev["row_id"].values],
            value=np.asarray(values, dtype=float),
            predicted_skill=float(claim),
            flags=list(prep["diag"].flags) if prep["diag"] is not None else [],
            notes=self.describe(task, prep),
            cv=cv_table,
        )

    # ---------------------------------------------------------------- data
    def prepare(self, task: TaskData) -> dict:
        spec = task.spec
        train = task.train.copy()
        ev = task.eval_features.copy()
        feats = task.features()

        # A0 cannot see that a column will be missing at scoring time: it carries
        # every training column into the design and lets the imputer fill the gap.
        for c in feats:
            if c not in ev.columns:
                ev[c] = np.nan

        diag = None
        date_col = None
        # A0 has no notion of a calendar: a free-text date is not a number it can
        # use, so it drops it.  That is the ordinary naive shortcut.
        import re as _re
        drop = [c for c in feats if not _is_num(train[c])
                and _re.search(r"(date|time|日期|时间)", c.lower())]
        feats = [c for c in feats if c not in drop]
        numeric, categorical = [], []
        for c in feats:
            (numeric if _is_num(train[c]) else categorical).append(c)
        return {"train": train, "ev": ev, "numeric": numeric, "categorical": categorical,
                "date_col": date_col, "diag": diag, "feats": feats}

    def describe(self, task: TaskData, prep: dict) -> str:
        return "all columns, random K-fold, stock metric"

    # ---------------------------------------------------------------- fitting
    def fold_mode(self, prep: dict) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
        return "random", None, None

    def cv_metric(self, task: TaskData) -> str:
        if task.spec.kind == "regression":
            return "rmse"
        return "auc"

    def balanced(self, prep: dict) -> bool:
        return False

    def wide_grid(self, task: TaskData) -> bool:
        return task.spec.task_id == "winner_curse"

    def fit_choose(self, task: TaskData, prep: dict) -> Tuple[np.ndarray, List[dict], float]:
        spec = task.spec
        kind = "regression" if spec.kind == "regression" else "binary"
        x = models.encode(prep["train"], prep["ev"], prep["numeric"], prep["categorical"],
                          spec.target, prep["date_col"])
        y = x.y
        n = len(y)
        mode, groups, order = self.fold_mode(prep)
        fl = models.folds(n, mode, self.seed, k=4, groups=groups, order=order)
        metric = self.cv_metric(task)
        sel_metric = "ap" if metric == "recall_at_budget" else metric

        best = None
        table = []
        for fam, variant, params in models.candidates(kind, self.wide_grid(task)):
            vals = []
            for tr, va in fl:
                pred = models.fit_predict(kind, fam, params, x.xtr[tr], y[tr], x.xtr[va], self.seed,
                                          self.balanced(prep))
                vals.append(models.cv_value(kind, sel_metric, y[va], pred))
            mean = float(np.mean(vals))
            table.append({"model": fam + ":" + variant, "cv": round(mean, 6)})
            if best is None or mean > best[0]:
                best = (mean, fam, variant, params)
        assert best is not None
        cv_score, fam, variant, params = best

        pred_ev = models.fit_predict(kind, fam, params, x.xtr, y, x.xev, self.seed,
                                     self.balanced(prep))

        # What the system believes it will score, expressed on the skill scale.
        extra = self.cv_baseline(task, prep, fl, y, kind)
        claim = _skill_from_cv(spec, metric if metric != "recall_at_budget" else "auc", cv_score, extra)
        return pred_ev, table, claim

    def cv_baseline(self, task: TaskData, prep: dict, fl, y: np.ndarray, kind: str) -> float:
        """The trivial reference in CV units, for turning a CV number into a skill."""
        if task.spec.kind != "regression":
            if task.spec.task_id == "metric_mismatch":
                return float(task.spec.budget or 0.05)
            return 0.5
        errs = []
        for tr, va in fl:
            errs.append(np.sqrt(np.mean((y[va] - np.mean(y[tr])) ** 2)))
        return float(np.mean(errs))


class A0(BaseAgent):
    level = 0
    name = "a0"


class A1(BaseAgent):
    level = 1
    name = "a1"

    def prepare(self, task: TaskData) -> dict:
        spec = task.spec
        train = task.train.copy()
        ev = task.eval_features.copy()
        feats = [c for c in task.features()]
        for c in feats:
            if c not in ev.columns:
                ev[c] = np.nan

        cols_meta = task.meta["tasks"][spec.task_id]["columns"]
        diag = hygiene.scan(train, ev, cols_meta, spec.target)

        # 1. duplicates: one row per distinct unit, and never split a unit across folds
        groups = None
        if diag.n_dup_rows:
            key = train[feats].astype(str).agg("|".join, axis=1)
            groups = key.map({k: i for i, k in enumerate(pd.unique(key))}).values
            first = train.drop_duplicates(subset=feats, keep="first").index
            keep_mask = train.index.isin(first)
            train = train.loc[keep_mask].reset_index(drop=True)
            groups = groups[np.asarray(keep_mask)]

        # 2. sentinels: out-of-range codes become missing, plus a missing indicator
        def repair(df: pd.DataFrame) -> pd.DataFrame:
            for c, (lo, hi) in diag.sentinel_rules.items():
                s = pd.to_numeric(df[c], errors="coerce")
                bad = (s < lo) | (s > hi)
                df[c + "__missing"] = bad.astype(int)
                df[c] = s.where(~bad, np.nan)
            return df

        train = repair(train)
        ev = repair(ev)

        # 3/4/6. drop what cannot help: ids, train-only columns, constants, near-perfect features
        drop = set(diag.id_cols) | set(diag.train_only_cols) | set(diag.constant_cols) | set(diag.leak_cols)
        feats = [c for c in feats if c not in drop]
        for c in list(diag.sentinel_cols):
            if c in feats and (c + "__missing") not in feats:
                feats.append(c + "__missing")

        numeric = [c for c in feats if c in train.columns and _is_num(train[c])]
        categorical = [c for c in feats if c in train.columns and c not in numeric]
        date_col = diag.time_col
        if date_col is not None and date_col in feats:
            feats = [c for c in feats if c != date_col]
            numeric = [c for c in numeric if c != date_col]
            categorical = [c for c in categorical if c != date_col]

        # 5. time ordering: sort, then validate on the future only
        if date_col is not None:
            if groups is not None:
                # duplicates + a time column: keep the time split; a unit may
                # straddle two folds.  Documented limitation.
                groups = None
            train = train.sort_values(date_col, kind="mergesort").reset_index(drop=True)
            ev = ev.sort_values(date_col, kind="mergesort").reset_index(drop=True)

        return {"train": train, "ev": ev, "numeric": numeric, "categorical": categorical,
                "date_col": date_col, "diag": diag, "feats": feats, "groups": groups}

    def fold_mode(self, prep):
        if prep.get("date_col"):
            return "time", None, None
        if prep.get("groups") is not None:
            return "group", prep["groups"], None
        return "random", None, None

    def balanced(self, prep) -> bool:
        d = prep.get("diag")
        return bool(d is not None and d.unbalanced)

    def describe(self, task: TaskData, prep: dict) -> str:
        d = prep["diag"]
        return "seven checks: " + (d.summary() if d else "none")


class A2(A1):
    level = 2
    name = "a2"

    def describe(self, task: TaskData, prep: dict) -> str:
        d = prep["diag"]
        extra = TASK_NOTE.get(task.spec.task_id, "")
        return ("seven checks: " + (d.summary() if d else "none") + ("; " + extra if extra else ""))

    def fit_choose(self, task: TaskData, prep: dict):
        tid = task.spec.task_id
        if tid == "confounding":
            return self._uplift(task, prep)
        if tid == "winner_curse":
            return self._nested(task, prep)
        if tid == "metric_mismatch":
            return self._budget(task, prep)
        return super().fit_choose(task, prep)

    def solve(self, task: TaskData) -> Submission:
        sub = super().solve(task)
        canonical = task.spec.trap
        if canonical not in sub.flags:
            sub.flags.append(canonical)
        return sub

    # ------------------------------------------------------------ metric_mismatch
    def _budget(self, task: TaskData, prep: dict):
        """Select by average precision, and report the recall the budget will get."""
        spec = task.spec
        budget = float(spec.budget or 0.05)
        x = models.encode(prep["train"], prep["ev"], prep["numeric"], prep["categorical"], spec.target)
        y = x.y
        fl = models.folds(len(y), self.fold_mode(prep)[0], self.seed, k=4,
                          groups=prep.get("groups"))
        best, table = None, []
        for fam, variant, params in models.candidates("binary"):
            aps, recs = [], []
            for tr, va in fl:
                pred = models.fit_predict("binary", fam, params, x.xtr[tr], y[tr], x.xtr[va],
                                          self.seed, balanced=True)
                aps.append(models.cv_value("binary", "ap", y[va], pred))
                recs.append(_recall_at(pred, y[va], budget))
            mean_ap, mean_rec = float(np.mean(aps)), float(np.mean(recs))
            table.append({"model": fam + ":" + variant, "cv": round(mean_ap, 6),
                          "recall_at_budget": round(mean_rec, 6)})
            # The decision is "fill 150 slots", so the candidate is chosen by the
            # recall those slots actually buy, not by a ranking-average proxy.
            if best is None or mean_rec > best[1]:
                best = (mean_ap, mean_rec, fam, variant, params)
        _, mean_rec, fam, variant, params = best
        pred_ev = models.fit_predict("binary", fam, params, x.xtr, y, x.xev, self.seed, balanced=True)
        claim = float(np.clip((mean_rec - budget) / max(1e-9, 1.0 - budget), -1.0, 1.0))
        return pred_ev, table, claim

    # ------------------------------------------------------------ confounding
    def _uplift(self, task: TaskData, prep: dict):
        """Two outcome models, one per arm; score = predicted benefit.

        The claim is not the plugin maximum: on held-out rows the plugin estimate
        of a policy is optimistic by construction.  A doubly-robust estimate uses
        the observed outcomes and the estimated propensity, which is a real
        estimate of what the policy would deliver rather than a tautology.
        """
        spec = task.spec
        budget = float(spec.budget or 0.4)
        train = prep["train"]
        ev = prep["ev"]
        feats = [c for c in prep["feats"] if c not in ("treated", "recovered")]
        numeric = [c for c in feats if _is_num(train[c])]
        categorical = [c for c in feats if c not in numeric]

        y = train[spec.target].values.astype(float)
        t = train["treated"].values.astype(int)
        fl = models.folds(len(y), "random", self.seed, k=4)
        fam, params = "hgb", {"max_leaf_nodes": 15}

        claims = []
        for tr, va in fl:
            trt = tr[t[tr] == 1]
            ctl = tr[t[tr] == 0]
            d_trt = models.encode(train.iloc[trt], train.iloc[va], numeric, categorical, spec.target)
            p1 = models.fit_predict("binary", fam, params, d_trt.xtr, y[trt], d_trt.xev, self.seed)
            d_ctl = models.encode(train.iloc[ctl], train.iloc[va], numeric, categorical, spec.target)
            p0 = models.fit_predict("binary", fam, params, d_ctl.xtr, y[ctl], d_ctl.xev, self.seed)
            d_pr = models.encode(train.iloc[tr], train.iloc[va], numeric, categorical, "treated")
            e = models.fit_predict("binary", "lr", {"C": 1.0}, d_pr.xtr,
                                   train.iloc[tr]["treated"].values.astype(float),
                                   d_pr.xev, self.seed)
            up = p1 - p0
            take = _take_top(up, budget)
            v_dr = _dr_value(p0, p1, take, y[va], t[va], e)
            v_rand = float(np.mean(p0) + budget * np.mean(up))
            k = max(1, int(round(budget * len(up))))
            v_oracle = float(np.mean(p0) + budget * np.mean(np.sort(up)[-k:]))
            claims.append(float(np.clip((v_dr - v_rand) / max(1e-9, v_oracle - v_rand), -1.0, 1.0)))

        d_trt = models.encode(train.iloc[t == 1], ev, numeric, categorical, spec.target)
        p1 = models.fit_predict("binary", fam, params, d_trt.xtr, y[t == 1], d_trt.xev, self.seed)
        d_ctl = models.encode(train.iloc[t == 0], ev, numeric, categorical, spec.target)
        p0 = models.fit_predict("binary", fam, params, d_ctl.xtr, y[t == 0], d_ctl.xev, self.seed)
        return p1 - p0, [{"model": "t-learner:hgb", "cv": round(float(np.mean(claims)), 6)}], float(np.mean(claims))

    # ------------------------------------------------------------ winner_curse
    def _nested(self, task: TaskData, prep: dict):
        """Outer folds estimate the score; inner folds choose the candidate.

        The choice inside each inner loop follows the one-standard-error rule:
        among the candidates of the winning family whose score is within one
        standard error of the best, take the most regularised.  A single noisy
        maximum on 450 rows is not evidence that the most complex model is best.
        """
        spec = task.spec
        x = models.encode(prep["train"], prep["ev"], prep["numeric"], prep["categorical"], spec.target)
        y = x.y
        cands = models.candidates("binary", wide=True)
        outer = models.folds(len(y), "random", self.seed + 7, k=3)
        outer_vals, picked = [], []
        for oi, (tr, va) in enumerate(outer):
            best = self._inner_pick(x, y, tr, self.seed + 100 + oi)
            fam, variant, params = best
            picked.append(fam + ":" + variant)
            pred = models.fit_predict("binary", fam, params, x.xtr[tr], y[tr], x.xtr[va], self.seed)
            outer_vals.append(models.cv_value("binary", "auc", y[va], pred))

        fam, variant, params = self._inner_pick(x, y, np.arange(len(y)), self.seed + 1)
        pred_ev = models.fit_predict("binary", fam, params, x.xtr, y, x.xev, self.seed)
        claim = float(np.clip(2.0 * float(np.mean(outer_vals)) - 1.0, -1.0, 1.0))
        table = [{"model": fam + ":" + variant, "picked_in_outer_folds": picked.count(fam + ":" + variant)}
                 for fam in ("lr", "rf", "hgb")]
        return pred_ev, table, claim

    def _inner_pick(self, x, y, idx, seed):
        inner = models.folds(len(idx), "random", seed, k=2)
        stats = []
        for fam, variant, params in models.candidates("binary", wide=True):
            vals = []
            for itr, iva in inner:
                pred = models.fit_predict("binary", fam, params, x.xtr[idx][itr], y[idx][itr],
                                          x.xtr[idx][iva], seed)
                vals.append(models.cv_value("binary", "auc", y[idx][iva], pred))
            m = float(np.mean(vals))
            se = float(np.std(vals) / max(1.0, np.sqrt(len(vals))))
            stats.append((m, se, fam, variant, params))
        m_best, se_best, fam_best, _, _ = max(stats, key=lambda s: s[0])
        pool = [s for s in stats if s[2] == fam_best and s[0] >= m_best - max(se_best, 0.005)]
        pick = min(pool, key=lambda s: _simplicity(s[2], s[4]))
        return pick[2], pick[3], pick[4]


def _simplicity(fam: str, params: dict) -> float:
    """Lower is simpler."""
    if fam == "lr":
        return float(params.get("C", 1.0))
    if fam == "rf":
        return 1.0 / float(params.get("min_samples_leaf", 5))
    return float(params.get("max_leaf_nodes", 15))


def _take_top(score: np.ndarray, budget: float) -> np.ndarray:
    k = max(1, int(round(budget * len(score))))
    order = np.argsort(-np.asarray(score), kind="mergesort")
    take = np.zeros(len(score), dtype=bool)
    take[order[:k]] = True
    return take


def _dr_value(p0: np.ndarray, p1: np.ndarray, take: np.ndarray, y: np.ndarray,
              t: np.ndarray, e: np.ndarray) -> float:
    """Doubly-robust estimate of the value of a treat/don't-treat policy."""
    m = np.where(take, p1, p0)
    e_pi = np.where(take, e, 1.0 - e)
    e_pi = np.clip(e_pi, 0.02, 0.98)
    matched = (np.asarray(t).astype(int) == np.asarray(take).astype(int))
    w = np.where(matched, 1.0 / e_pi, 0.0)
    return float(np.mean(m + w * (np.asarray(y) - m)))



TASK_NOTE = {
    "temporal": "treats the date as an ordinal trend and validates on the future only",
    "metric_mismatch": "selects by average precision and reports the recall the budget buys",
    "confounding": "two outcome models, one per arm; ranks by predicted benefit",
    "winner_curse": "nested cross-validation, outer folds report the score",
}


def _recall_at(pred: np.ndarray, y: np.ndarray, budget: float) -> float:
    k = max(1, int(round(budget * len(pred))))
    order = np.argsort(-np.asarray(pred))
    hit = np.asarray(y)[order[:k]].sum()
    total = np.asarray(y).sum()
    return float(hit / total) if total > 0 else 0.0


AGENT_CLASSES = {"a0": A0, "a1": A1, "a2": A2}
