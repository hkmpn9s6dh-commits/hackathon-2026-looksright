"""Write site/index.html: one self-contained page, no external requests.

The template uses @@token@@ placeholders and plain string replacement, so that
CSS percentages and Chinese punctuation never meet a format string.
"""

from __future__ import annotations

import base64
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>看着对 LooksRight — 自主数据科学系统的陷阱基准</title>
<style>
:root{--ink:#1a1d22;--soft:#5b6572;--line:#e2ddd4;--paper:#faf8f4;--accent:#12355b;--warm:#8a6a3b}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font:16px/1.75 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:980px;margin:0 auto;padding:0 28px}
header{padding:88px 0 44px;border-bottom:1px solid var(--line)}
h1{font:700 54px/1.1 "Songti SC","SimSun",Georgia,serif;margin:0 0 10px;letter-spacing:.04em;color:var(--accent)}
.sub{font:400 15px/1.4 Georgia,serif;color:var(--soft);letter-spacing:.22em;text-transform:uppercase;margin:0 0 28px}
.lead{font-size:19px;line-height:1.85;max-width:740px;margin:0}
h2{font:700 27px/1.3 "Songti SC","SimSun",Georgia,serif;margin:64px 0 6px;color:var(--accent)}
h2 .n{color:var(--warm);font-size:15px;letter-spacing:.12em;display:block;margin-bottom:10px;font-family:Georgia,serif}
p{margin:0 0 16px;max-width:780px}
.small{font-size:13.5px;color:var(--soft)}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:34px 0 8px}
.card{background:#fff;border:1px solid var(--line);padding:22px 20px 18px}
.card .t{font:700 12px/1.3 Georgia,serif;letter-spacing:.16em;color:var(--warm);text-transform:uppercase}
.card .v{font:700 40px/1.1 Georgia,serif;margin:12px 0 2px;color:var(--accent)}
.card .d{font-size:13px;color:var(--soft);line-height:1.6}
table{width:100%;border-collapse:collapse;margin:22px 0 8px;font-size:14.5px;background:#fff;
 border:1px solid var(--line)}
th,td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line)}
th{font:700 11.5px/1.3 Georgia,serif;letter-spacing:.1em;text-transform:uppercase;color:var(--soft);
 background:#f3efe8}
td.num{font-family:Georgia,serif;text-align:right;font-variant-numeric:tabular-nums;width:64px}
tr:last-child td{border-bottom:none}
figure{margin:30px 0 8px}
figure img{width:100%;border:1px solid var(--line);background:#fff}
figcaption{font-size:13.5px;color:var(--soft);margin-top:10px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:26px 0}
.box{background:#fff;border:1px solid var(--line);padding:20px 22px}
.box h4{margin:0 0 8px;font-size:15px;letter-spacing:.02em}
.box p{font-size:13.8px;margin:0;color:var(--soft);line-height:1.7}
pre{background:#12171e;color:#e6e9ee;padding:16px 18px;overflow-x:auto;font-size:13px;line-height:1.75}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.94em}
ul{margin:0 0 16px;padding-left:20px}
li{margin:0 0 8px;max-width:760px}
footer{margin:80px 0 60px;padding-top:26px;border-top:1px solid var(--line);font-size:13.5px;color:var(--soft)}
.kicker{font:700 12px/1.3 Georgia,serif;letter-spacing:.18em;color:var(--warm);text-transform:uppercase}
@media(max-width:760px){.cards{grid-template-columns:1fr}.cols{grid-template-columns:1fr}h1{font-size:40px}}
</style></head><body>
<div class="wrap">

<header>
  <h1>看着对</h1>
  <p class="sub">LooksRight &nbsp;·&nbsp; v@@ver@@</p>
  <p class="lead">给自主数据科学系统造一把尺子，专门量它会不会把「看着对、其实错」的结果交出去：
  八个任务，每个埋一个真实世界里反复出现的坑；每个任务同时量两件事——<b>实际做得怎么样</b>，
  以及<b>它对自己成绩的估计准不准</b>。</p>
  <div class="cards">
    @@c0@@
    @@c1@@
    @@c2@@
  </div>
  <p class="small">总分 = 100 ×（0.65 × 技能 + 0.25 × 自知 + 0.10 × 指出陷阱）。三个参照系是我们自己实现的，
  不是排行榜；这张页面不发布「哪个系统最强」的结论。</p>
</header>

<h2><span class="n">01 / 结果</span>通用体检能捞回五类坑，剩下三类不行</h2>
<p>A0 是标准 AutoML：全列进模型、随机 K 折、按库存指标挑最高分、把交叉验证当期望。A1 在它之上加七项
无差别的数据体检（重复行、哨兵值、伪 ID、评分时缺失的列、时序、类别不平衡、单特征近乎完全分开）。
A2 再加一个只有知道问题是什么才做得出的判断。</p>

<figure><img alt="skill per task" src="data:image/png;base64,@@fig1@@">
<figcaption>八个任务上的技能分（0 = 照该任务明示的平凡基准做，1 = 完美；可以为负）。
confounding 一栏 A0/A1 为负：比随机挑人用药还差。</figcaption></figure>

<table>
<thead><tr><th>任务</th><th>陷阱</th><th>指标</th><th class="num">A0</th><th class="num">A1</th><th class="num">A2</th></tr></thead>
<tbody>
@@rows@@
</tbody></table>

<p class="small">A0 → A1 多拿 @@d01@@ 分，A1 → A2 再多拿 @@d12@@ 分。前一截全部来自五类「能从表的形状看出来」的坑；
后一截全部来自三类「表本身干净得很」的坑：指标错配、混杂、赢家诅咒。</p>

<h2><span class="n">02 / 自知</span>它报的数字，和它实际拿到的数字</h2>
<p>横轴是实际，纵轴是自估。落在虚线上是老实，落在虚线上方是自信地错。24 次运行里，A0 最远的一次
报 0.98、实得 0.24（target_leak：把放款后才产生的催收字段当成了特征）；另一次报 0.34、
实得 −0.70（confounding：答错了题）。</p>
<figure><img alt="self estimate vs outcome" src="data:image/png;base64,@@fig2@@">
<figcaption>自估与实际，24 次运行。自知一栏：A0 @@c01@@，A1 @@c11@@，A2 @@c21@@。</figcaption></figure>

<figure><img alt="what each step buys" src="data:image/png;base64,@@fig4@@">
<figcaption>每一步照护买到的东西，拆成「技能」与「自知」两块。自知升得比技能快——这是直觉，
也是这份基准想说的事。</figcaption></figure>

<h2><span class="n">03 / 八个坑</span>都是可复算的</h2>
<div class="cols">
  <div class="box"><h4>wine_dup　重复行</h4><p>同一酒样被复检，重复行同时落进训练与验证。交叉验证被抬高，训练权重被那一小批酒带偏。</p></div>
  <div class="box"><h4>sentinel　哨兵值</h4><p>缺测写成 999 / −999。模型读到「胆固醇 999」，而那 25% 其实是随机漏测的人。</p></div>
  <div class="box"><h4>target_leak　目标泄漏</h4><p>催收次数、追回金额、首次还款天数——都是放款之后才有的字段。交叉验证 0.95，上线时这三列是空的。</p></div>
  <div class="box"><h4>temporal　时序</h4><p>随机切分拿明天的数据预测今天；而且树不会外推趋势，真实未来一段只有一半的分数。</p></div>
  <div class="box"><h4>metric_mismatch　指标错配</h4><p>按 AUC 选模型，决策却是「每天只能看 150 件」。AUC 0.84 看着能用，前 5% 只捞到三成缺陷。</p></div>
  <div class="box"><h4>confounding　混杂</h4><p>医生把药开给重症，观察到的「用药组恢复好」是病情决定的。要答的是「谁会因为用药而受益」。</p></div>
  <div class="box"><h4>winner_curse　赢家诅咒</h4><p>320 行上比 12 个候选，挑出来的那个的分数是挑出来的最高分，不是它的水平。</p></div>
  <div class="box"><h4>pseudo_id　伪 ID</h4><p>导出表按结果排序后顺序编号，行号本身就是答案。新导出的 id 全在训练区间之外，树只能给同一个常数。</p></div>
</div>

<h2><span class="n">04 / 怎么用</span>三条命令，一份可复算的结果</h2>
<pre><code>py -3 exp/run_all.py                          # 8 个任务 × 3 个参照系，写 results.json 与全部预测
py -3 verify.py                               # 127 项断言；两次完整运行 results.json 逐字节一致
py -3 evaluate.py --agent a1 --task wine_dup  # 单题评分</code></pre>
<p>接一个外部系统：读 <code>data/&lt;任务&gt;/train.csv</code> 与 <code>eval_features.csv</code>，
交 <code>predictions.csv</code>（两列 <code>row_id,value</code>）与 <code>claim.json</code>
（<code>predicted_skill</code> 与 <code>flags</code>）。评分脚本只认这两样，从不读你的代码。</p>

<h2><span class="n">05 / 边界</span>不做什么</h2>
<ul>
<li>不排名哪个系统最强。外部系统要按同一张任务卡提交才能比较。</li>
<li>不训练大模型、不调外部 API。全部离线、单线程、固定种子。</li>
<li>不宣称这些数字是可发表的绝对水平。技能分是相对各自任务明示的平凡基准算的。</li>
<li>不假装八个坑穷尽了真实世界的坑。</li>
</ul>
<p class="small">已知弱点写在报告第六节：sentinel 一题对树模型几乎无伤（净损失 0.046 技能分）；
metric_mismatch 一题 A2 的模型与 A0 统计上打平，优势全在自估；confounding 的双重稳健估计方差偏大；
八个任务都是合成数据。</p>

<footer>
  <div class="kicker">LooksRight v@@ver@@</div>
  <p class="small">代码 MIT，数据 CC-BY-4.0。数据由 <code>exp/make_data.py</code> 用固定种子生成，
  <code>data/meta.json</code> 记录每个文件的行数、列字典与 sha256；评分真值在
  <code>data/_labels/</code>，混杂题的潜在结局在 <code>data/_private/</code>。
  本页所有数字来自 <code>results.json</code>，由 <code>exp/make_site.py</code> 渲染。</p>
</footer>

</div></body></html>
"""


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def card(value: float, label: str, desc: str) -> str:
    return ('<div class="card"><div class="t">%s</div><div class="v">%.1f</div>'
            '<div class="d">%s</div></div>' % (label, value, desc))


def build(res: dict) -> str:
    tot = res["total_points"]
    agg = res["aggregate"]
    rows = []
    for t in res["tasks"]:
        p = res["per_task"][t["task_id"]]
        rows.append(
            "<tr><td><code>%s</code><div class='small'>%s</div></td>"
            "<td><code>%s</code></td><td>%s</td>"
            "<td class='num'>%.0f</td><td class='num'>%.0f</td><td class='num'>%.0f</td></tr>"
            % (t["task_id"], t["zh"], t["trap"], t["metric"],
               p["a0"]["points"], p["a1"]["points"], p["a2"]["points"]))
    tokens = {
        "ver": res["version"],
        "c0": card(tot["a0"], "A0 naive", "标准 AutoML：全部列、随机 K 折、按库存指标挑最高分。"),
        "c1": card(tot["a1"], "A1 check-up", "加七项通用数据检查，不看业务口径。"),
        "c2": card(tot["a2"], "A2 task-aware", "再加一个只有知道问题是什么才做得出的判断。"),
        "fig1": b64(os.path.join(ROOT, "figures", "fig1_skill.png")),
        "fig2": b64(os.path.join(ROOT, "figures", "fig2_calibration.png")),
        "fig4": b64(os.path.join(ROOT, "figures", "fig4_what_each_step_buys.png")),
        "rows": "\n".join(rows),
        "d01": "%.1f" % (tot["a1"] - tot["a0"]),
        "d12": "%.1f" % (tot["a2"] - tot["a1"]),
        "c01": "%.2f" % agg["a0"]["calibration"],
        "c11": "%.2f" % agg["a1"]["calibration"],
        "c21": "%.2f" % agg["a2"]["calibration"],
    }
    out = TEMPLATE
    for k, v in tokens.items():
        out = out.replace("@@" + k + "@@", str(v))
    return out


def main():
    with io.open(os.path.join(ROOT, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    html = build(res)
    out = os.path.join(ROOT, "site", "index.html")
    if not os.path.isdir(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print("wrote", out, "%.0f KB" % (len(html.encode("utf-8")) / 1024))


if __name__ == "__main__":
    main()
