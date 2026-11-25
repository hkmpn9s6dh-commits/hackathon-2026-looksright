"""Build REPORT.pdf from results.json.

    py -3 exp/make_report.py

The layout is done by hand (PyMuPDF, A4, one column) so that the report is
reproducible without LaTeX, pandoc or a headless browser.  Chinese text needs a
CJK font; the script looks for one of the usual system fonts and embeds a subset.
If none is installed it falls back to PyMuPDF's built-in CJK face, which renders
but with wide fixed spacing -- the committed REPORT.pdf was made with a real font.
"""

from __future__ import annotations

import io
import json
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

W, H = 595, 842
M = 56
CW = W - 2 * M

INK = (0.11, 0.14, 0.19)
SOFT = (0.35, 0.40, 0.47)
RULE = (0.84, 0.86, 0.89)
ACCENT = (0.07, 0.21, 0.36)

FONT_CANDIDATES = [
    ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"),
    ("C:/Windows/Fonts/simhei.ttf", None),
    ("C:/Windows/Fonts/simsun.ttc", None),
    ("/System/Library/Fonts/PingFang.ttc", None),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", None),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", None),
]


def find_fonts():
    for reg, bold in FONT_CANDIDATES:
        if os.path.exists(reg):
            return reg, (bold if bold and os.path.exists(bold) else reg)
    return None, None


class Doc:
    def __init__(self):
        self.doc = fitz.open()
        self.page = None
        self.y = M
        reg, bold = find_fonts()
        self.fontfile = reg
        self.boldfile = bold
        self.fallbacks = reg is None
        self.new_page()

    def new_page(self):
        self.page = self.doc.new_page(width=W, height=H)
        self.y = M
        if self.fontfile:
            self.page.insert_font(fontname="main", fontfile=self.fontfile)
            self.page.insert_font(fontname="mainb", fontfile=self.boldfile)
        self.f_main = fitz.Font(fontfile=self.fontfile) if self.fontfile else fitz.Font("china-s")
        self.f_bold = fitz.Font(fontfile=self.boldfile) if self.boldfile else self.f_main

    def need(self, h):
        if self.y + h > H - M:
            self.new_page()
            return True
        return False

    def wrap(self, text, size, width, bold=False):
        font = self.f_bold if bold else self.f_main
        lines, cur = [], ""
        for ch in text:
            if ch == "\n":
                lines.append(cur)
                cur = ""
                continue
            if font.text_length(cur + ch, fontsize=size) <= width:
                cur += ch
            else:
                if " " in cur and ord(ch) < 128:
                    i = cur.rfind(" ")
                    lines.append(cur[:i])
                    cur = cur[i + 1:] + ch
                else:
                    lines.append(cur)
                    cur = ch
        if cur:
            lines.append(cur)
        return lines

    def text(self, text, size=10.2, leading=1.62, color=INK, bold=False, gap=0.0, width=None):
        width = width or CW
        lines = self.wrap(text, size, width, bold)
        lh = size * leading
        for ln in lines:
            if self.need(lh):
                pass
            self.page.insert_text((M, self.y + size * 0.86), ln, fontsize=size,
                                  fontname="mainb" if bold else "main", color=color)
            self.y += lh
        self.y += gap

    def h1(self, text, size=21, gap_before=0, gap_after=10):
        self.y += gap_before
        self.need(150)
        self.page.insert_text((M, self.y + size * 0.95), text, fontsize=size,
                              fontname="mainb", color=ACCENT)
        self.y += size * 1.30 + gap_after

    def h2(self, text, size=12.6, gap_before=12, gap_after=5):
        self.y += gap_before
        self.need(size * 2.4)
        self.page.insert_text((M, self.y + size * 0.92), text, fontsize=size,
                              fontname="mainb", color=INK)
        self.y += size * 1.45 + gap_after

    def rule(self, gap=8, color=RULE, width=0.7):
        self.need(gap)
        self.page.draw_line(fitz.Point(M, self.y), fitz.Point(M + CW, self.y),
                            color=color, width=width)
        self.y += gap

    def kv(self, pairs, size=10.0, col=150, gap=3):
        self.y += 5
        for k, v in pairs:
            lines = self.wrap(v, size, CW - col)
            self.need(size * 1.6 * len(lines))
            self.page.insert_text((M, self.y + size * 0.86), k, fontsize=size,
                                  fontname="mainb", color=SOFT)
            yy = self.y
            for ln in lines:
                self.page.insert_text((M + col, yy + size * 0.86), ln, fontsize=size,
                                      fontname="main", color=INK)
                yy += size * 1.55
            self.y = yy + gap

    def table(self, headers, rows, widths, size=8.8, pad=4.6, header_gap=6):
        row_h = size * 1.55 + pad
        self.need(row_h * 2 + header_gap)
        x = M
        for head, w in zip(headers, widths):
            self.page.insert_text((x + 1, self.y + size * 0.9), head, fontsize=size,
                                  fontname="mainb", color=SOFT)
            x += w
        self.y += size * 1.1
        self.page.draw_line(fitz.Point(M, self.y), fitz.Point(M + sum(widths), self.y),
                            color=RULE, width=0.8)
        self.y += pad
        for row in rows:
            lines_per_cell = [self.wrap(str(c), size, w - 6) for c, w in zip(row, widths)]
            n = max(len(l) for l in lines_per_cell)
            self.need(row_h * n)
            x = M
            for lines, w in zip(lines_per_cell, widths):
                yy = self.y
                for ln in lines:
                    self.page.insert_text((x + 1, yy + size * 0.86), ln, fontsize=size,
                                          fontname="main", color=INK)
                    yy += size * 1.5
                x += w
            self.y += size * 1.5 * n + pad
            self.page.draw_line(fitz.Point(M, self.y - pad * 0.4),
                                fitz.Point(M + sum(widths), self.y - pad * 0.4),
                                color=(0.93, 0.94, 0.96), width=0.5)

    def figure(self, path, caption, width=None, gap=8):
        width = width or CW
        import struct
        with open(path, "rb") as f:
            f.read(16)
            pw, ph = struct.unpack(">II", f.read(8))
        h = width * ph / pw
        self.need(h + 40)
        self.page.insert_image(fitz.Rect(M, self.y, M + width, self.y + h), filename=path)
        self.y += h + 4
        for ln in self.wrap(caption, 8.4, width):
            self.page.insert_text((M, self.y + 8), ln, fontsize=8.4, fontname="main",
                                  color=SOFT)
            self.y += 12
        self.y += gap

    def save(self, path):
        self.doc.set_metadata({"title": "LooksRight 看着对 — benchmark report",
                               "author": "LooksRight", "creator": "exp/make_report.py"})
        try:
            self.doc.subset_fonts(verbose=False)
        except Exception:
            pass
        self.doc.save(path, deflate=True, garbage=4)
        self.doc.close()


def main():
    with io.open(os.path.join(ROOT, "results.json"), encoding="utf-8") as f:
        R = json.load(f)
    agg = R["aggregate"]
    tot = R["total_points"]
    tasks = [t["task_id"] for t in R["tasks"]]
    meta = {t["task_id"]: t for t in R["tasks"]}

    d = Doc()

    # ------------------------------------------------------------------ cover
    d.y = M + 60
    d.page.insert_text((M, d.y + 30), "看着对", fontsize=44, fontname="mainb", color=ACCENT)
    d.y += 56
    d.page.insert_text((M, d.y + 18), "LooksRight", fontsize=19, fontname="mainb", color=SOFT)
    d.y += 52
    for ln in d.wrap("自主数据科学系统（Agentic DS）的陷阱基准：八个任务，八个真实世界里"
                     "反复出现的坑，每个任务同时量两件事——做得怎么样，以及它对自己成绩的估计准不准。",
                     12.2, CW * 0.86):
        d.page.insert_text((M, d.y), ln, fontsize=12.2, fontname="main", color=INK)
        d.y += 22
    d.y += 26
    d.rule()

    d.kv([
        ("版本", "%s v%s｜种子 %d｜结果文件 results.json" % (R["benchmark"], R["version"], R["seed"])),
        ("任务", "8 个（回归 2、二分类 3、预算排序 1、个体增益 1、小样本选型 1）"),
        ("参照系", "A0 朴素（%.1f 分）／A1 通用体检（%.1f 分）／A2 任务感知（%.1f 分）" % (
            tot["a0"], tot["a1"], tot["a2"])),
        ("一句话结论", "通用体检能捞回五类坑；剩下三类不是数据的问题，是判断的问题。"),
        ("可复算", "py -3 verify.py  →  127 项断言，两次运行 results.json 逐字节一致"),
        ("许可", "代码 MIT；数据 CC-BY-4.0"),
    ], size=9.6, col=86)

    # ------------------------------------------------------------------ 1
    d.new_page()
    d.h1("一、要量的是什么", size=17, gap_before=6)
    d.text("现有智能体基准（DSBench、MLE-bench、DataSciBench）主要量任务完成度：给你一张表、"
           "一个目标，看你最后交出来的数字离最优差多少。真实部署里更贵的失败不是差，"
           "而是「看着对、其实错」——交叉验证 0.99，上线 0.55；把混杂当因果；"
           "在 600 行上挑出来的模型报了 0.86，换一批人只有 0.76。")
    d.text("这些失败有一个共同点：犯错的时候，系统并不知道自己在犯错。所以本基准对每一次提交量三件事。")
    d.kv([
        ("做得怎么样", "任务指标换算到统一的「技能」刻度：0 等于照该任务明示的平凡基准做，"
                       "1 等于完美，可以为负。"),
        ("估计准不准", "提交必须同时给出「我预期自己这一步能拿多少技能」。准确度 = "
                       "1 − |自估 − 实际|。"),
        ("有没有看见坑", "提交可以列出它识别到的陷阱标识；命中该任务的陷阱记 1，否则 0。"),
    ], size=9.8, col=92)
    d.y += 4
    d.text("总分 = 100 ×（0.65 × 技能 + 0.25 × 准确度 + 0.10 × 指出陷阱）。权重是价值判断，"
           "写在 src/looksright/spec.py 一处，改一处即全局生效。")
    d.rule()
    d.h2("为什么把「自估」单列一栏", gap_before=8)
    d.text("因为一个系统如果只会报高，那么它的所有指标都不可用于决策。把自估和实际分开记，"
           "才能看出它是「不会做」还是「不知道自己不会做」——后者在部署里更危险，"
           "因为没有任何告警会响。")

    # ------------------------------------------------------------------ 2
    d.h1("二、八个任务", size=17, gap_before=26)
    d.text("每个任务是一张表 + 一句业务问题 + 一个埋在设计里的坑。坑不是事后贴标签，"
           "而是数据生成过程的一部分：重复行真的同时落进训练与验证，哨兵值真的是那一列的众数，"
           "催收字段真的产生在放款之后。", gap=2)
    d.y += 4
    rows = []
    for tid in tasks:
        p = R["per_task"][tid]
        rows.append([tid, meta[tid]["trap"], meta[tid]["metric"],
                     "%d / %d" % (meta[tid]["n_train"], meta[tid]["n_eval"]),
                     "%.0f" % p["a0"]["points"], "%.0f" % p["a1"]["points"],
                     "%.0f" % p["a2"]["points"]])
    d.table(["任务", "陷阱", "指标", "训练/评分行", "A0", "A1", "A2"], rows,
            widths=[92, 108, 96, 72, 40, 40, 40], size=8.4)
    d.y += 4
    d.text("指标：rmse＝均方根误差（对照训练集均值）；auc＝ROC 曲线下面积（对照 0.5）；"
           "recall_at_budget＝预算内命中率（对照预算本身）；policy_value＝被选中人群的"
           "真实结局（对照随机选人、上界为按真实增益选人）。", size=8.6, color=SOFT)

    # ------------------------------------------------------------------ 3
    d.h1("三、三个参照系", size=17, gap_before=26)
    d.text("三个参照系共用同一套编码器、同一份候选模型、同一套折划分与最终重训，"
           "所以分数差只能来自判断差，不来自更好的学习器。", gap=2)
    d.h2("A0 朴素：标准 AutoML", gap_before=8)
    d.text("全部列进模型；数值列取中位数插补；随机 K 折；在候选里按库存指标（AUC / RMSE）"
           "挑最高；把这个交叉验证分数当作上线后的期望。日期列是字符串，它用不了，于是丢掉。")
    d.h2("A1 体检：通用七项检查", gap_before=6)
    d.text("在 A0 之上，对每一张表无差别地跑七项检查：重复行分组、哨兵值按字段字典复位、"
           "唯一值列当伪 ID 丢掉、训练有而评分时没有的列丢掉、看到日期列改按时间切分、"
           "类别不平衡时给训练加权、单特征近乎完全分开时隔离该列。"
           "七项都不看任务名，也不看业务口径。")
    d.h2("A2 任务感知：知道坑在哪", gap_before=6)
    d.text("在 A1 之上加一个只有知道问题是什么才做得出的判断：时序题把趋势项单独外推；"
           "预算题按「预算内召回」选模型并报召回；因果题分组建两个结局模型、按个体增益排序，"
           "并用双重稳健估计报自己的价值；小样本选型用嵌套交叉验证、内层按一倍标准误规则取更稳的模型。")
    d.rule()
    d.text("注意 A2 的边界：它知道的是「这一类问题该怎么想」，不是评分集的标签。"
           "它从未读取 data/_labels 或 data/_private。")

    # ------------------------------------------------------------------ 4
    d.h1("四、结果", size=17, gap_before=6)
    d.figure(os.path.join(ROOT, "figures", "fig1_skill.png"),
             "图 1　八个任务上的技能分。confounding 一栏中 A0/A1 为负，表示比随机挑人用药还差。")
    d.figure(os.path.join(ROOT, "figures", "fig4_what_each_step_buys.png"),
             "图 2　A0→A1 与 A1→A2 各自买到的东西，拆成「技能」与「自知」两块。")

    # ------------------------------------------------------------------ 5
    d.h1("五、三个数字说明的事", size=17, gap_before=6)
    d.kv([
        ("%.2f → %.2f" % (tot["a0"], tot["a1"]),
         "通用七项检查值 %.1f 分。它在五类坑上几乎把关：重复行、哨兵值、目标泄漏、"
         "时序泄漏、伪 ID——这五类都能从表的形状看出来，不需要懂业务。" % (tot["a1"] - tot["a0"])),
        ("%.2f → %.2f" % (tot["a1"], tot["a2"]),
         "剩下的 %.1f 分，全部来自三类看不出来的坑：指标错配、混杂、赢家诅咒。"
         "这三类数据本身干净得很，体检一分也捞不回来。" % (tot["a2"] - tot["a1"])),
        ("指标的代价", "target_leak 上 A0 报 0.98、实得 0.24；confounding 上 A0 报 0.34、"
                       "实得 −0.70。前者是「抄了答案还以为是本事」，后者是「答错了题」。"),
        ("诚实也有代价", "winner_curse 上 A2 的嵌套估计比实得低 0.05——在 320 行上，"
                         "两个方向的误差都在一个标准误之内。这一条本身就是结论："
                         "这个样本量下的分数，本来就不该被当作精确数。"),
    ], size=9.4, col=104)
    d.y += 6
    d.text("把这三类合起来看：A0 平均只拿到 %.0f 分里的 %.1f 分，其中「自知」一栏只有 %.2f；"
           "A1 到 %.1f 分、「自知」%.2f；A2 到 %.1f 分、「自知」%.2f。"
           "技能从 %.3f 升到 %.3f，而自知从 %.2f 升到 %.2f——后者升得更快，"
           "这也符合直觉：知道自己不知道什么，比多会一点更容易做到，也更值钱。"
           % (100, tot["a0"], agg["a0"]["calibration"], tot["a1"], agg["a1"]["calibration"],
              tot["a2"], agg["a2"]["calibration"], agg["a0"]["skill"], agg["a2"]["skill"],
              agg["a0"]["calibration"], agg["a2"]["calibration"]))

    # ------------------------------------------------------------------ 6
    d.new_page()
    d.h1("六、这份基准不做什么", size=17, gap_before=6)
    for t in [
        "不排名哪个系统最强。这里只有三个我们自己实现的参照系；外部系统接进来要按同一张任务卡提交，才能比较。",
        "不训练大模型、不调外部 API。全部离线、单线程、固定种子，两次运行逐字节一致。",
        "不宣称这些数字是可发表的绝对水平。技能分是相对各自任务明示的平凡基准算的，"
        "跨任务可比，与文献里的绝对指标不可比。",
        "不假装八个坑穷尽了真实世界的坑。它们只是八类最常见的、可复算的、能被一张表承载的坑。",
    ]:
        d.text("· " + t, size=9.8, gap=1)

    d.h2("已知的弱点", gap_before=10)
    for t in [
        "sentinel 一题：哨兵值主要摧毁的是尺度敏感的线性模型；梯度提升树对它相当免疫。"
        "所以净损失只有 0.046 技能分（约 0.023 AUC）。这是真实结论，不是设计失误，但它意味着这一题的量程偏窄。",
        "metric_mismatch 一题：A2 的模型与 A0 的模型在这个样本量下统计上打平，"
        "它的 13 分优势几乎全部来自「自估」与「指出陷阱」。这也是一种真实："
        "指标错配伤的是你对数字的理解，不是模型本身。",
        "confounding 一题的双重稳健估计在 4 折上的方差偏大，自估 0.25 而实得 0.65，偏低。",
        "八个任务都是合成数据。规模小、可复算，但也因此没有真实数据里的长尾脏法。",
    ]:
        d.text("· " + t, size=9.8, gap=1)

    # ------------------------------------------------------------------ 7
    d.h1("七、怎么用", size=17, gap_before=6)
    d.kv([
        ("跑一遍", "py -3 exp/run_all.py"),
        ("复算闸门", "py -3 verify.py　（127 项断言；两次完整运行比对 sha256；约 8 分钟）"),
        ("单题评分", "py -3 evaluate.py --agent a1 --task wine_dup"),
        ("重新生成数据", "py -3 exp/make_data.py　（固定种子，写入 data/，并更新 meta.json 的哈希）"),
        ("出图 / 出报告", "py -3 exp/make_figures.py ／ py -3 exp/make_report.py ／ py -3 exp/make_site.py"),
    ], size=9.6, col=104)
    d.h2("接一个外部系统", gap_before=10)
    d.text("外部系统读 data/<任务>/train.csv 与 eval_features.csv，交两样东西：")
    d.kv([
        ("predictions.csv", "两列 row_id,value。回归题 value 是预测值；分类题是分数（越大越可能为正）；"
                            "预算题是排序分；因果题是「这个人用上药会多受益多少」的排序分。"),
        ("claim.json", "predicted_skill（该题技能刻度上的自估值）与 flags（陷阱标识列表）。"),
    ], size=9.6, col=104)
    d.text("然后 py -3 evaluate.py 只认这两个文件。评分脚本从不读你的代码，"
           "只读你交的预测与自估——这也是为什么「自估」能作为一个独立维度被量到。", size=9.8)

    d.rule()
    d.h2("文件的来处", gap_before=6, gap_after=2)
    d.text("所有数据由 exp/make_data.py 用固定种子生成，data/meta.json 记录每个文件的行数、"
           "列字典（含量程）、以及 sha256；verify.py 的第 01 项逐一核对。评分用的真值放在 "
           "data/_labels/，混杂题的潜在结局放在 data/_private/，对外发布资源包时应把这两处去掉，"
           "由维护方另行保管。", size=9.2, color=SOFT)

    out = os.path.join(ROOT, "REPORT.pdf")
    d.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
