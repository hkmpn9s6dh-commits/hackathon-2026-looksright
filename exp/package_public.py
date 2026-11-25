"""Build the public resource pack: everything except the grader-side files.

    py -3 exp/package_public.py            -> dist/looksright-public-v1.0.zip

Leaves out data/_labels/, data/_private/ and results/predictions/, because those
are the only things a participant must not see before submitting.  The zip still
verifies for structure (data hashes are in meta.json), but verify.py's scoring
checks need the private labels and will not run from the unpacked copy -- that is
the intended behaviour, and the reason a maintainer keeps the full copy.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SKIP_DIRS = {"data/_labels", "data/_private", "results/predictions", "__pycache__", ".git", "dist"}
SKIP_FILES = {".DS_Store"}


def main():
    with io.open(os.path.join(ROOT, "results.json"), encoding="utf-8") as f:
        ver = json.load(f)["version"]
    dist = os.path.join(ROOT, "dist")
    if not os.path.isdir(dist):
        os.makedirs(dist)
    out = os.path.join(dist, "looksright-public-v%s.zip" % ver)
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            rel = os.path.relpath(dirpath, ROOT).replace("\\", "/")
            rel = "" if rel == "." else rel
            dirnames[:] = [d for d in dirnames if (rel + "/" + d).lstrip("/") not in SKIP_DIRS
                           and d not in ("__pycache__", ".git")]
            if any(rel == s for s in SKIP_DIRS):
                continue
            for name in filenames:
                if name in SKIP_FILES or name.endswith(".pyc"):
                    continue
                arc = (rel + "/" + name).lstrip("/")
                z.write(os.path.join(dirpath, name), arc)
                count += 1
    print("wrote %s (%d files, %.1f MB)" % (out, count, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
