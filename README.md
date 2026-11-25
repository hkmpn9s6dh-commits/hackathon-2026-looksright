# 看着对 LooksRight

自主数据科学系统（Agentic DS）的**陷阱基准**。

八个任务，每个埋一个真实世界里反复出现的坑；每个任务同时量两件事——
**实际做得怎么样**，以及**它对自己成绩的估计准不准**。

三个参照系：A0 标准 AutoML（33.4 分）、A1 通用七项体检（50.2 分）、A2 任务感知（61.4 分）。
差额说明一件事：通用体检能捞回五类坑（重复行、哨兵值、目标泄漏、时序、伪 ID），
剩下三类（指标错配、混杂、赢家诅咒）不是数据的问题，是判断的问题。

## 跑起来

```bash
pip install numpy pandas scikit-learn matplotlib PyMuPDF

py -3 exp/run_all.py                          # 8 任务 × 3 参照系 → results.json（约 3 分钟）
py -3 verify.py                               # 127 项断言 + 两次运行逐字节比对（约 8 分钟）
py -3 evaluate.py --agent a1 --task wine_dup  # 单题评分
```

`verify.py` 退出码为 0 才算复现成功。它是唯一的验收口径：数据哈希、results.json 结构、
用存档预测重算的每一个数字、总分方向、逐题方向、取值域、两次完整运行的 sha256 一致。

## 看结果

| 文件 | 是什么 |
|---|---|
| `REPORT.pdf` | 五页的报告：问题、八个任务、三个参照系、结果、局限 |
| `site/index.html` | 单文件结果页（图片内嵌，无外部请求），可直接托管到 Pages |
| `results.json` | 唯一真源；所有页面、图表、报告的数字都从这里来 |
| `results/predictions/<agent>/<task>.csv` | 24 份逐行预测，可单独复算 |

## 目录

```
data/            8 套 CSV + meta.json（列字典、量程、行数、sha256）
  _labels/       评分真值（评分侧，不对外）
  _private/      混杂题的潜在结局（评分侧，不对外）
docs/            01 任务定义包｜02 数据集资源包｜03 解决方案基准包｜04 提交说明
src/looksright/  spec（任务与计分）｜hygiene（七项检查）｜models｜agents（A0/A1/A2）｜scoring
exp/             make_data｜run_all｜make_figures｜make_report｜make_site｜package_public
evaluate.py      单题或全量评分
verify.py        复算闸门
```

## 接到自己的系统上

读 `data/<任务>/train.csv` 与 `eval_features.csv`，交两份文件：

```
predictions.csv    row_id,value
claim.json         {"predicted_skill": 0.0, "flags": []}
```

`value` 的语义按题而定（回归＝预测值，分类＝越大越像正类，预算题＝抽检优先级，
因果题＝「用上药会多受益多少」的排序分）。`predicted_skill` 必须落在该题的技能刻度上：
**0 = 照该题明示的平凡基准做，1 = 完美**。然后 `py -3 evaluate.py`。

细节见 `docs/01-TASK.md`；为什么这么设计、错在哪、下一步做什么，见 `docs/03-SOLUTION.md`。

## 边界

不排名哪个系统最强；不训练大模型、不调外部 API；不宣称这些数字是可发表的绝对水平
（技能分相对各题明示的平凡基准，跨任务可比，与文献里的绝对指标不可比）；八个坑不代表全部。

## 许可

代码 MIT，数据 CC-BY-4.0。引用：`LooksRight v1.0（2026），自主数据科学系统陷阱基准，
清华数据库组 × 和鲸社区共建资源包`。
