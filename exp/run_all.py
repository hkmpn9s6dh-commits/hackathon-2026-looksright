"""Run the whole benchmark and write results.json plus every per-run artefact.

    py -3 exp/run_all.py [--out DIR]

Deterministic: same seed, same data, same code -> byte-identical results.json.
The thread environment is pinned to one thread at the top of this file because
OpenMP on a 1k-row table spends more time in barriers than in arithmetic, and
because a single thread keeps the floating-point reductions reproducible.
"""

from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import io
import json
import platform
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import sklearn  # noqa: E402

from looksright import scoring  # noqa: E402
from looksright.agents import AGENT_CLASSES  # noqa: E402
from looksright.data import load_task  # noqa: E402
from looksright.spec import AGENTS, BENCHMARK, SEED, TASKS, VERSION  # noqa: E402


def run(out_dir: str, quiet: bool = False):
    res_dir = os.path.join(out_dir, "results")
    pred_dir = os.path.join(res_dir, "predictions")
    for a in AGENTS:
        d = os.path.join(pred_dir, a)
        if not os.path.isdir(d):
            os.makedirs(d)

    per_task = {}
    all_scores = []
    for spec in TASKS:
        task = load_task(spec.task_id, ROOT)
        per_task[spec.task_id] = {}
        for a in AGENTS:
            t0 = time.time()
            sub = AGENT_CLASSES[a]().solve(task)
            # Score exactly what is written to disk, so that re-scoring the stored
            # prediction file reproduces results.json to the last digit.
            values = np.round(np.asarray(sub.value, dtype=float), 8)
            frame = pd.DataFrame({"row_id": sub.row_id, "value": values})
            frame.to_csv(os.path.join(pred_dir, a, spec.task_id + ".csv"),
                         index=False, float_format="%.8f", lineterminator="\n")
            claim = {
                "agent": a,
                "task_id": spec.task_id,
                "predicted_skill": round(float(sub.predicted_skill), 8),
                "flags": sorted(set(sub.flags)),
                "notes": sub.notes,
                "cv": sub.cv,
            }
            with io.open(os.path.join(pred_dir, a, spec.task_id + ".json"), "w",
                         encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True))
                f.write("\n")
            sc = scoring.score(task, a, sub.row_id, values, sub.predicted_skill, sub.flags)
            per_task[spec.task_id][a] = sc.to_dict()
            all_scores.append(sc)
            if not quiet:
                print("  %-16s %-2s  %-22s skill %+.3f  claim %+.3f  flag %d  pts %5.1f  (%4.1fs)" % (
                    spec.task_id, a, spec.metric, sc.skill, sc.predicted_skill,
                    1 if sc.flagged else 0, sc.points, time.time() - t0))

    aggregate = scoring.aggregate(all_scores)
    totals = {a: aggregate[a]["points"] for a in AGENTS}
    per_task_ok = all(per_task[t]["a2"]["points"] >= per_task[t]["a0"]["points"] - 1e-9 for t in per_task)
    strict = sum(1 for t in per_task if per_task[t]["a2"]["points"] > per_task[t]["a0"]["points"] + 0.5)

    results = {
        "benchmark": BENCHMARK,
        "version": VERSION,
        "seed": SEED,
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "tasks": [{
            "task_id": t.task_id, "en": t.en, "zh": t.zh, "metric": t.metric,
            "kind": t.kind, "trap": t.trap, "budget": t.budget, "baseline": t.baseline,
            "n_train": t.n_train, "n_eval": t.n_eval,
        } for t in TASKS],
        "per_task": per_task,
        "aggregate": aggregate,
        "total_points": totals,
        "ordering": {
            "a0_lt_a1": bool(totals["a0"] < totals["a1"]),
            "a1_lt_a2": bool(totals["a1"] < totals["a2"]),
            "a2_ge_a0_every_task": bool(per_task_ok),
            "tasks_with_strict_gain_a0_to_a2": int(strict),
        },
    }

    path = os.path.join(out_dir, "results.json")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    return results, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    results, path = run(args.out, args.quiet)

    # Every figure is drawn from results.json, so it belongs to the same command.
    sys.path.insert(0, HERE)
    from make_figures import make_all
    figs = make_all(results, os.path.join(args.out, "figures"))
    if not args.quiet:
        print("")
        print("figures:", ", ".join(os.path.basename(f) for f in figs))
    if not args.quiet:
        print("\n== totals (mean points over 8 tasks) ==")
        for a in AGENTS:
            print("  %-2s  %6.2f   skill %+.3f  calib %.3f  flags %.0f%%" % (
                a, results["aggregate"][a]["points"], results["aggregate"][a]["skill"],
                results["aggregate"][a]["calibration"], 100 * results["aggregate"][a]["flag_rate"]))
        print("  ordering:", results["ordering"])
        print("\nwrote", path, "in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
