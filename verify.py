"""The gate.  Exit 0 only if every claim this repository makes is checkable.

    py -3 verify.py            # everything, about 7 minutes (it runs the whole
                               # benchmark twice to prove byte-identical output)
    py -3 verify.py --quick    # everything except the second full run

Checks
  01  data files are the ones meta.json hashes
  02  results.json exists, has 8 tasks x 3 agents, and every field is present
  03  re-scoring the stored predictions reproduces results.json exactly
  04  A0 < A1 < A2 on total points
  05  A2 is not worse than A0 on any task, and is strictly better on >= 5
  06  every task's skill in results.json matches the metric it claims
  07  two independent full runs produce byte-identical results.json
"""

from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from looksright import scoring  # noqa: E402
from looksright.data import load_task  # noqa: E402
from looksright.spec import AGENTS, TASKS  # noqa: E402

FAILS = []
CHECKS = [0]


def check(cond, label, detail=""):
    CHECKS[0] += 1
    if cond:
        print("  PASS  %s" % label)
    else:
        print("  FAIL  %s%s" % (label, ("  -- " + detail) if detail else ""))
        FAILS.append(label)
    return bool(cond)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_data():
    print("\n01 data integrity")
    with io.open(os.path.join(HERE, "data", "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    for t in meta["tasks"]:
        p = os.path.join(HERE, "data", t, "train.csv")
        check(sha256(p) == meta["tasks"][t]["files"]["train_sha256"],
              "train.csv sha256 %s" % t)
        p = os.path.join(HERE, "data", t, "eval_features.csv")
        check(sha256(p) == meta["tasks"][t]["files"]["eval_features_sha256"],
              "eval_features.csv sha256 %s" % t)


def check_results(root: str):
    print("\n02 results.json shape")
    path = os.path.join(root, "results.json")
    if not check(os.path.exists(path), "results.json exists"):
        return None
    with io.open(path, encoding="utf-8") as f:
        res = json.load(f)
    ok = True
    for spec in TASKS:
        node = res["per_task"].get(spec.task_id)
        ok = ok and check(node is not None, "task present: %s" % spec.task_id)
        if node is None:
            continue
        for a in AGENTS:
            ok = ok and check(a in node and "points" in node[a], "%s / %s scored" % (spec.task_id, a))
    return res


def check_rescore(root: str, res: dict):
    print("\n03 re-score the stored predictions")
    for spec in TASKS:
        task = load_task(spec.task_id, root)
        for a in AGENTS:
            p = os.path.join(root, "results", "predictions", a, spec.task_id + ".csv")
            c = os.path.join(root, "results", "predictions", a, spec.task_id + ".json")
            if not check(os.path.exists(p) and os.path.exists(c), "artefact present %s/%s" % (a, spec.task_id)):
                continue
            frame = pd.read_csv(p)
            with io.open(c, encoding="utf-8") as f:
                claim = json.load(f)
            sc = scoring.score(task, a, frame["row_id"].astype(int).tolist(), frame["value"].values,
                               float(claim["predicted_skill"]), list(claim["flags"]))
            got = json.loads(json.dumps(sc.to_dict(), sort_keys=True))
            want = json.loads(json.dumps(res["per_task"][spec.task_id][a], sort_keys=True))
            check(got == want, "matches results.json %s/%s" % (a, spec.task_id),
                  "" if got == want else "recomputed %s vs stored %s" % (got.get("points"), want.get("points")))


def check_ordering(res: dict):
    print("\n04 ordering")
    tot = res["total_points"]
    check(tot["a0"] < tot["a1"], "A0 < A1 on total points", "%.3f vs %.3f" % (tot["a0"], tot["a1"]))
    check(tot["a1"] < tot["a2"], "A1 < A2 on total points", "%.3f vs %.3f" % (tot["a1"], tot["a2"]))

    print("\n05 per-task direction")
    worse = [t for t in res["per_task"]
             if res["per_task"][t]["a2"]["points"] < res["per_task"][t]["a0"]["points"] - 1e-9]
    check(not worse, "A2 >= A0 on every task", "worse on " + ",".join(worse) if worse else "")
    strict = [t for t in res["per_task"]
              if res["per_task"][t]["a2"]["points"] > res["per_task"][t]["a0"]["points"] + 0.5]
    check(len(strict) >= 5, "A2 strictly above A0 on >= 5 tasks", "%d tasks" % len(strict))

    print("\n06 metric bounds")
    for spec in TASKS:
        for a in AGENTS:
            node = res["per_task"][spec.task_id][a]
            ok = -1.0 <= node["skill"] <= 1.0 and 0.0 <= node["calibration"] <= 1.0
            check(ok, "ranges %s/%s" % (a, spec.task_id),
                  "skill %.3f calib %.3f" % (node["skill"], node["calibration"]))


def check_determinism():
    print("\n07 determinism: two full runs")
    tmp = tempfile.mkdtemp(prefix="looksright-")
    try:
        hashes = []
        for i in (1, 2):
            out = os.path.join(tmp, "run%d" % i)
            os.makedirs(out)
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "exp", "run_all.py"), "--out", out, "--quiet"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=HERE)
            if proc.returncode != 0:
                check(False, "run %d completed" % i, proc.stderr.decode("utf-8", "replace")[-400:])
                return
            hashes.append(sha256(os.path.join(out, "results.json")))
        check(hashes[0] == hashes[1], "results.json byte-identical across two runs",
              "%s vs %s" % (hashes[0][:12], hashes[1][:12]))
        with io.open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
            committed = hashlib.sha256(f.read().encode("utf-8")).hexdigest()
        check(committed == hashes[0], "committed results.json equals a fresh run",
              "%s vs %s" % (committed[:12], hashes[0][:12]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the two full runs")
    ap.add_argument("--root", default=HERE)
    args = ap.parse_args()

    print("LooksRight verification")
    check_data()
    res = check_results(args.root)
    if res is not None:
        check_rescore(args.root, res)
        check_ordering(res)
    else:
        check(False, "results.json present")

    if not args.quick:
        check_determinism()
    else:
        print("\n07 determinism: skipped (--quick)")

    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("  - " + f)
        return 1
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
