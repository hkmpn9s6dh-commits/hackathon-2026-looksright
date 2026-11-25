"""Score one agent on one task, from the artefacts already on disk.

    py -3 evaluate.py --agent a1 --task wine_dup
    py -3 evaluate.py --all

If the submission for that pair has not been produced yet, it is produced now and
written to results/predictions/; otherwise the stored prediction is re-scored, so
the number you see here is exactly the number in results.json.
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

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from looksright import scoring  # noqa: E402
from looksright.agents import AGENT_CLASSES  # noqa: E402
from looksright.data import load_task  # noqa: E402
from looksright.spec import AGENTS, TASKS, TASK_BY_ID  # noqa: E402


def load_or_run(agent: str, task_id: str, root: str):
    pred = os.path.join(root, "results", "predictions", agent, task_id + ".csv")
    claim = os.path.join(root, "results", "predictions", agent, task_id + ".json")
    task = load_task(task_id, root)
    if os.path.exists(pred) and os.path.exists(claim):
        frame = pd.read_csv(pred)
        with io.open(claim, encoding="utf-8") as f:
            c = json.load(f)
        return task, frame["row_id"].astype(int).tolist(), frame["value"].values, \
            float(c["predicted_skill"]), list(c["flags"])
    sub = AGENT_CLASSES[agent]().solve(task)
    d = os.path.dirname(pred)
    if not os.path.isdir(d):
        os.makedirs(d)
    pd.DataFrame({"row_id": sub.row_id, "value": np.round(sub.value, 8)}).to_csv(
        pred, index=False, float_format="%.8f", lineterminator="\n")
    with io.open(claim, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"agent": agent, "task_id": task_id,
                            "predicted_skill": round(float(sub.predicted_skill), 8),
                            "flags": sorted(set(sub.flags)), "notes": sub.notes,
                            "cv": sub.cv}, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return task, sub.row_id, sub.value, float(sub.predicted_skill), list(sub.flags)


def report(sc) -> str:
    lines = [
        "task      %s  (%s, metric %s)" % (sc.task_id, TASK_BY_ID[sc.task_id].zh, sc.metric),
        "agent     %s" % sc.agent,
        "metric    %.4f    baseline %.4f%s" % (
            sc.metric_value, sc.baseline_value,
            "    oracle %.4f" % sc.oracle_value if sc.oracle_value is not None else ""),
        "skill     %+.4f" % sc.skill,
        "claim     %+.4f    calibration %.4f" % (sc.predicted_skill, sc.calibration),
        "trap      %-22s named: %s" % (TASK_BY_ID[sc.task_id].trap, "yes" if sc.flagged else "no"),
        "points    %.1f" % sc.points,
    ]
    if sc.extra:
        lines.append("extra     " + "  ".join("%s=%g" % (k, v) for k, v in sorted(sc.extra.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=AGENTS)
    ap.add_argument("--task", choices=[t.task_id for t in TASKS])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", default=HERE)
    args = ap.parse_args()

    if args.all:
        pairs = [(a, t.task_id) for t in TASKS for a in AGENTS]
    else:
        if not args.agent or not args.task:
            ap.error("give --agent and --task, or --all")
        pairs = [(args.agent, args.task)]

    bad = 0
    for agent, task_id in pairs:
        task, row_id, value, claim, flags = load_or_run(agent, task_id, args.root)
        sc = scoring.score(task, agent, row_id, value, claim, flags)
        print(report(sc))
        print("")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
