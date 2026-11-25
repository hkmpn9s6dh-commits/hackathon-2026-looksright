"""Draw the four figures from results.json.  No numbers are invented here."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AGENTS = ["a0", "a1", "a2"]
LABEL = {"a0": "A0 naive", "a1": "A1 check-up", "a2": "A2 task-aware"}
COLOR = {"a0": "#b3bcc9", "a1": "#4a7ebb", "a2": "#12355b"}
INK = "#1c2430"
GRID = "#dfe3e8"


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa4b2")
    ax.tick_params(colors="#5a6572", labelsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def fig_skill(res: dict, out: str):
    tasks = [t["task_id"] for t in res["tasks"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    y = np.arange(len(tasks))
    h = 0.26
    for i, a in enumerate(AGENTS):
        vals = [res["per_task"][t][a]["skill"] for t in tasks]
        ax.barh(y + (i - 1) * h, vals, height=h, color=COLOR[a], label=LABEL[a],
                edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#5a6572", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("skill, against the task's stated trivial reference", color=INK, fontsize=9)
    ax.set_title("Skill on the 8 tasks: where generic check-up stops working",
                 color=INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_calibration(res: dict, out: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=200)
    ax.plot([-0.8, 1.0], [-0.8, 1.0], color="#9aa4b2", linewidth=0.9, linestyle=(0, (4, 3)))
    for a in AGENTS:
        xs = [res["per_task"][t][a]["skill"] for t in res["per_task"]]
        ys = [res["per_task"][t][a]["predicted_skill"] for t in res["per_task"]]
        ax.scatter(xs, ys, s=38, color=COLOR[a], edgecolor="white", linewidth=0.6,
                   label=LABEL[a], zorder=3)
        for t in res["per_task"]:
            node = res["per_task"][t][a]
            err = node["predicted_skill"] - node["skill"]
            if abs(err) >= 0.55:
                up = err > 0
                ax.annotate("%s / %s" % (t, a.upper()),
                            (node["skill"], node["predicted_skill"]),
                            textcoords="offset points",
                            xytext=(7, 5) if up else (7, -11), fontsize=6.8,
                            color="#5a6572")
    ax.set_xlim(-0.85, 1.0)
    ax.set_ylim(-0.85, 1.05)
    ax.set_xlabel("what actually happened (skill)", color=INK, fontsize=9)
    ax.set_ylabel("what the system said would happen (skill)", color=INK, fontsize=9)
    ax.set_title("Self-estimate against outcome, 24 runs", color=INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_points(res: dict, out: str):
    tasks = [t["task_id"] for t in res["tasks"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    y = np.arange(len(tasks))
    w = 0.8 / 3
    for i, a in enumerate(AGENTS):
        vals = [res["per_task"][t][a]["points"] for t in tasks]
        ax.barh(y + (i - 1) * w, vals, height=w * 0.92, color=COLOR[a], label=LABEL[a],
                edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("points (65% skill + 25% self-knowledge + 10% naming the trap)", color=INK, fontsize=8.5)
    ax.set_title("Points per task", color=INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_gap(res: dict, out: str):
    """What A1 adds over A0, and what A2 adds over A1, split by axis."""
    tasks = [t["task_id"] for t in res["tasks"]]
    d1 = [res["per_task"][t]["a1"]["calibration"] - res["per_task"][t]["a0"]["calibration"] for t in tasks]
    d2 = [res["per_task"][t]["a2"]["calibration"] - res["per_task"][t]["a1"]["calibration"] for t in tasks]
    s1 = [res["per_task"][t]["a1"]["skill"] - res["per_task"][t]["a0"]["skill"] for t in tasks]
    s2 = [res["per_task"][t]["a2"]["skill"] - res["per_task"][t]["a1"]["skill"] for t in tasks]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    y = np.arange(len(tasks))
    h = 0.2
    series = [
        (s1, "skill gained, A0 -> A1", "#a8c4e0"),
        (d1, "self-knowledge gained, A0 -> A1", "#4a7ebb"),
        (s2, "skill gained, A1 -> A2", "#c9b79a"),
        (d2, "self-knowledge gained, A1 -> A2", "#8a6a3b"),
    ]
    for i, (vals, label, color) in enumerate(series):
        ax.barh(y + (i - 1.5) * h, vals, height=h * 0.9, color=color, label=label,
                edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#5a6572", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("change in the two reported components", color=INK, fontsize=9)
    ax.set_title("What each step of care actually buys", color=INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def make_all(res: dict, outdir: str):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    fig_skill(res, os.path.join(outdir, "fig1_skill.png"))
    fig_calibration(res, os.path.join(outdir, "fig2_calibration.png"))
    fig_points(res, os.path.join(outdir, "fig3_points.png"))
    fig_gap(res, os.path.join(outdir, "fig4_what_each_step_buys.png"))
    return [os.path.join(outdir, n) for n in
            ("fig1_skill.png", "fig2_calibration.png", "fig3_points.png", "fig4_what_each_step_buys.png")]
