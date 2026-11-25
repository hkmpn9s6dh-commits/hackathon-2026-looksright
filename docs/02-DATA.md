# 数据集资源包（B 类资源）

## 一、目录与文件

```
data/
  meta.json                     机器可读的字段字典：类型、单位、量程、行数、sha256
  wine_dup/train.csv            训练表（含目标列）
  wine_dup/eval_features.csv    评分表（只含预测时点拿得到的列）
  ...
  _labels/<task>.csv            评分真值（只给评分脚本；对外发布时应剔除）
  _private/confounding_potential.csv   混杂题的潜在结局 y0 / y1 / uplift（只给评分脚本）
```

八套表，规模从 320 行到 2600 行不等，全部是 CSV（UTF-8，逗号分隔，`\n` 换行，浮点保留 4 位）。
目标是**离线、可复算、几秒钟跑完**，不是规模。

| 任务 | 训练行 | 评分行 | 目标 | 阳性率 / 取值 |
|---|---:|---:|---|---|
| `wine_dup` | 1140 | 420 | `quality` | 3–9，均值 6.03 |
| `sentinel` | 1300 | 700 | `readmitted` | 19.3% |
| `target_leak` | 1400 | 700 | `default_90d` | 9.6% |
| `temporal` | 1920 | 270 | `units` | 100–700 件/日 |
| `metric_mismatch` | 2400 | 2000 | `defect` | 4.3% |
| `confounding` | 2600 | 1400 | `recovered` | 55% |
| `winner_curse` | 320 | 2000 | `churn` | 30% |
| `pseudo_id` | 1000 | 600 | `converted` | 19.4% |

## 二、存储形式与读取

普通 CSV，任何工具都能读。`row_id` 是行键：训练表从 1 开始，评分表从 100001 开始，
两表不重叠；`predictions.csv` 用它对齐。**`row_id` 是导出行号，不是业务主键**，任何参照系都该丢掉它；
`pseudo_id` 那题的 `record_id` 是另一列，那才是坑。

`meta.json` 的结构：

```json
{"benchmark": "LooksRight", "version": "1.0", "seed": 20261125,
 "tasks": {"sentinel": {
   "target": "readmitted", "metric": "auc", "trap": "sentinel_values",
   "n_train": 1300, "n_eval": 700,
   "columns": [{"name": "chol", "dtype": "float", "unit": "mg/dL",
                "plausible_range": [100.0, 350.0], "range_source": "data dictionary (设备/业务量程)"}, ...],
   "files": {"train_sha256": "...", "eval_features_sha256": "..."}}}}
```

`plausible_range` 是整个哨兵值那一题的关键：字典说胆固醇的合理量程是 100–350，
出现 999 就不是测量值。**量程是业务方给的，不是从数据里统计出来的**——这一点很重要，
否则 25% 的 999 会把统计量程拉到读不出来。

## 三、血缘：每个文件怎么来的

全部由 `exp/make_data.py` 用固定种子（`SEED = 20261125`）生成，一个任务一个生成函数，
一个 `numpy.random.default_rng(SEED + 1000*(i+1))`。重跑一遍得到逐字节相同的文件
（`verify.py` 的第 01 项核对 sha256）。

每题的生成过程都写在那段代码里，包括陷阱是怎么嵌进去的：

| 任务 | 陷阱是怎么写进去的 |
|---|---|
| `wine_dup` | 先造 820 支酒，再从高分端抽出 260 支重复录入（200 支多一次、60 支多两次），最后打乱行序 |
| `sentinel` | 先按真实信号造干净数据，再 MCAR 地把 25% 的胆固醇写成 999、30% 的糖化血红蛋白写成 −99、8% 血糖 −999、3% BMI −999、2% 收缩压 0 |
| `target_leak` | 先造申请时点的字段与违约标签，再**按违约标签**生成催收次数、追回金额、首还天数；评分表只保留申请时点的列 |
| `temporal` | 日销 = 基线 + 线性趋势（0.24–0.40/天）+ 年周期 + 周内效应 + 促销 + 价格 + 气温 + 噪声；评分集是最后 90 天 |
| `metric_mismatch` | 两条缺陷路径叠加：一条稀但几乎确定的批次簇（B3/B5 + 刀具磨损），一条铺在全体上的弱线性路径 |
| `confounding` | 用药概率随病情上升（logistic）；y0 随病情下降；uplift 只在重症端显著。潜在结局只给评分脚本 |
| `winner_curse` | 十个弱效应叠加，真实 AUC 约 0.60；样本只有 320 行，足够让 12 个候选的 CV 最大值偏乐观 |
| `pseudo_id` | 训练表按 `converted + N(0, 0.72)` 排序后顺序编号；评分表的编号从 1001 接着走，与标签无关 |

## 四、约束与已知问题

- **`_labels/` 与 `_private/` 是评分侧的。** 对外发布资源包时应把这两处剔除，由维护方另行保管；
  `evaluate.py --labels` 指向由维护方提供的目录即可。这是本仓库唯一需要「守住」的东西。
- **`temporal` 的最强平凡基准是 naive forecast**（把上一年最后一天平推，RMSE 31.9），
  比本基准写进技能刻度的「训练集均值」（RMSE 156.9）强得多。这是刻意保留的不一致：
  技能刻度要跨任务可比，所以统一用均值；naive 的数值作为 `extra.rmse_last_value` 一并输出，
  供时间序列读者核对。
- **全部是合成数据。** 规模小、可复算，也因此没有真实数据里的长尾脏法（同一个字段三种编码、
  时区、双字节空格……）。第二版应至少混入两套真实公开数据。
- **无个人信息。** 没有任何真实个体；字段名取自公开数据集的常见命名（UCI wine quality、
  diabetes readmission、lending club 一类的形状），但取值为生成。
- **不做缺失值之外的隐私处理**，因为本来就没有隐私。

## 五、许可

- 数据：CC-BY-4.0。可商用、可改编，署名即可。
- 代码：MIT。
- 引用：`LooksRight v1.0（2026），自主数据科学系统陷阱基准，清华数据库组 × 和鲸社区共建资源包`。
