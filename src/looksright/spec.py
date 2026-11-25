"""LooksRight — the task specification.

One place defines what each of the eight tasks is, what its trap is, how a
submission is turned into a number, and how three numbers (skill, calibration,
trap flag) become points.  Everything else in the repository reads this module;
no other file may hard-code a task list or a metric rule.

Terminology
-----------
skill        how well the submission actually did, mapped onto a common scale
             relative to a *stated trivial reference*, so that eight tasks with
             four different metrics can be averaged honestly.  Range [-1, 1].
calibration  1 - |predicted_skill - skill|.  How honest the system's estimate of
             its own result was.  Range [0, 1].
flag         whether the submission named the trap it was walking into.
points       100 * (0.65 * skill + 0.25 * calibration + 0.10 * flag), clipped to
             [0, 100].  The weights are a value judgement, declared here once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BENCHMARK = "LooksRight"
VERSION = "1.0"
SEED = 20261125

# Points weights.  Chosen so that being right outweighs being confident about
# it, and so that noticing the trap is worth something but cannot rescue a bad
# answer: a submission that flags the trap but scores zero skill still gets 10.
W_SKILL = 0.65
W_CALIB = 0.25
W_FLAG = 0.10

# Canonical trap identifiers.  A submission names a trap by putting one of these
# strings in claim.json["flags"].  They are stable API: downstream tools group by
# them.
FLAGS: Dict[str, str] = {
    "duplicate_rows": "the same unit appears more than once (train/test or fold/fold)",
    "sentinel_values": "missing values are coded as numbers that look like measurements",
    "post_outcome_feature": "a feature is only known after the outcome",
    "temporal_split": "rows are ordered in time and a random split leaks the future",
    "metric_mismatch": "the metric being optimised is not the metric the decision uses",
    "confounding": "treatment and outcome share a cause; the observed contrast is not the effect",
    "selection_bias": "the best of many noisy candidates is reported at its own selection score",
    "pseudo_id": "an identifier column carries the label through the row order or the export batch",
}


@dataclass(frozen=True)
class TaskSpec:
    """One task.  Immutable: the benchmark contract."""

    task_id: str
    en: str
    zh: str
    target: str
    kind: str  # "regression" | "binary" | "ranking" | "uplift"
    metric: str  # "rmse" | "auc" | "recall_at_budget" | "policy_value"
    trap: str  # one of FLAGS
    trap_zh: str
    budget: Optional[float] = None  # fraction of rows acted on, for recall_at_budget / policy_value
    baseline: str = "mean"  # the stated trivial reference: mean | random | last_value | random_policy
    n_train: int = 1000
    n_eval: int = 400
    prompt: str = ""
    decision: str = ""  # what the row-level score means, in business words
    baseline_zh: str = ""  # the trivial reference the skill is measured against, in prose
    gold_zh: str = ""  # what a task-aware system does
    naive_zh: str = ""  # what the naive system does, and why it looks fine
    hygiene_zh: str = ""  # what generic health checks buy, and what they miss
    columns: List[Tuple[str, str, str]] = field(default_factory=list)  # (name, dtype, note)


TASKS: List[TaskSpec] = [
    TaskSpec(
        task_id="wine_dup",
        en="Wine re-test batch",
        zh="酒样复检批次",
        target="quality",
        kind="regression",
        metric="rmse",
        trap="duplicate_rows",
        trap_zh="同一个酒样被复检多次，重复行同时落进训练与验证",
        n_train=1140,
        n_eval=420,
        baseline="mean",
        prompt=(
            "A cooperative winery sends its tasting panel results. Some bottles were "
            "re-tasted and the panel's sheet repeats them verbatim. Predict the panel "
            "score (0-10) for 420 bottles that have not been tasted yet."
        ),
        decision="a predicted score per bottle, so the panel can pre-sort the flight",
        baseline_zh="用训练集均值预测（RMSE 的零技能基准）",
        gold_zh="按酒样分组：同一酒样的重复行只进同一折，训练前按酒样去重，再做回归",
        naive_zh="随机 K 折、全表训练：重复行把验证折的分数抬上去，同时让被复检的那一小批酒在损失里被过量加权",
        hygiene_zh="分组去重是通用体检里就有的一条，能同时修好「报得太高」和「权重被带偏」",
        columns=[
            ("row_id", "int", "行号，不是酒样编号"),
            ("fixed_acidity", "float", "g/L"),
            ("volatile_acidity", "float", "g/L，醋酸"),
            ("citric_acid", "float", "g/L"),
            ("residual_sugar", "float", "g/L"),
            ("chlorides", "float", "g/L"),
            ("free_so2", "float", "mg/L"),
            ("total_so2", "float", "mg/L"),
            ("density", "float", "g/mL"),
            ("ph", "float", ""),
            ("sulphates", "float", "g/L"),
            ("alcohol", "float", "% vol"),
            ("quality", "float", "评审团均分，0-10；目标列"),
        ],
    ),
    TaskSpec(
        task_id="sentinel",
        en="Clinic readmission sheet",
        zh="体检回访表",
        target="readmitted",
        kind="binary",
        metric="auc",
        trap="sentinel_values",
        trap_zh="缺测被写成 999 / -999，模型当成「胆固醇高到爆」",
        n_train=1300,
        n_eval=700,
        baseline="random",
        prompt=(
            "A clinic exports two years of annual check-ups with a 90-day readmission "
            "flag. The lab system writes 999 when a test was not ordered. Rank 480 "
            "patients by readmission risk."
        ),
        decision="a risk score per patient; the top 10% get a nurse call",
        baseline_zh="随机排序（AUC 0.5 的零技能基准）",
        gold_zh="按字段字典把 999 / -999 还原成缺测，插补并保留「未测」指示列，再建风险模型",
        naive_zh="把 999 当真实测量值：模型学到「胆固醇 999 = 高危」，而那 25% 其实是随机漏测的人",
        hygiene_zh="哨兵值检测是通用体检的一条：超出字段字典范围的值一律按缺测处理",
        columns=[
            ("row_id", "int", ""),
            ("age", "int", "岁"),
            ("sex", "str", "F/M"),
            ("bmi", "float", "kg/m2"),
            ("sbp", "int", "mmHg，收缩压"),
            ("chol", "float", "mg/dL；999 表示未测"),
            ("glucose", "float", "mg/dL；-999 表示未测"),
            ("hba1c", "float", "%；-99 表示未测"),
            ("prior_admissions", "int", "近两年住院次数"),
            ("smoker", "str", "yes/no"),
            ("readmitted", "int", "90 天内再入院，0/1；目标列"),
        ],
    ),
    TaskSpec(
        task_id="target_leak",
        en="Loan arrears",
        zh="小额贷逾期",
        target="default_90d",
        kind="binary",
        metric="auc",
        trap="post_outcome_feature",
        trap_zh="三个字段是放款之后才产生的，等于把答案抄进来",
        n_train=1400,
        n_eval=700,
        baseline="random",
        prompt=(
            "A lender wants a 90-day default score computed at application time. The "
            "training export was pulled from the warehouse after the loans closed, so it "
            "still carries the collections fields. The scoring file only carries what "
            "exists on the application form."
        ),
        decision="a score at the moment of application; later fields will not exist",
        baseline_zh="随机排序（AUC 0.5）",
        gold_zh="只留申请时点就有的字段，按时间切分验证",
        naive_zh="全字段进模型：催收次数几乎单独定胜负，交叉验证 0.95，上线时这三列是空的",
        hygiene_zh="「单变量近乎分得开」的通用检查会把这三列挑出来隔离，剩下的字段还够用",
        columns=[
            ("row_id", "int", ""),
            ("income", "float", "月收入"),
            ("loan_amount", "float", "元"),
            ("dti", "float", "负债收入比"),
            ("credit_score", "int", "300-850"),
            ("employment_years", "float", ""),
            ("region", "str", "六个大区"),
            ("collections_30d", "int", "放款后 30 天内催收次数；申请时点不存在"),
            ("recovered_amount", "float", "放款后追回金额；申请时点不存在"),
            ("days_to_first_payment", "float", "放款到首次还款天数；申请时点不存在"),
            ("default_90d", "int", "0/1；目标列"),
        ],
    ),
    TaskSpec(
        task_id="temporal",
        en="Store demand",
        zh="门店日销",
        target="units",
        kind="regression",
        metric="rmse",
        trap="temporal_split",
        trap_zh="随机切分会拿明天的数据预测今天，而且树不会外推趋势",
        n_train=1920,
        n_eval=270,
        baseline="last_value",
        prompt=(
            "Three stores, two years of daily sales. Forecast the next 90 days per store. "
            "Rows are one store-day each."
        ),
        decision="a units forecast per store-day for the next quarter (purchasing plan)",
        baseline_zh="用训练集均值预测",
        gold_zh="按时间切分验证；把趋势从目标里剥出来（对残差建模，趋势项单独外推），按门店分层",
        naive_zh="随机 K 折：相邻日期互为验证，交叉验证 0.93；真实未来一段只有 0.5 左右，因为树只能平推，接不住 18 个月的趋势",
        hygiene_zh="看到日期列就换时间切分——这一条能修好「报得太高」，但不会让模型学会外推",
        columns=[
            ("row_id", "int", ""),
            ("date", "str", "YYYY-MM-DD"),
            ("store_id", "str", "S1/S2/S3"),
            ("dow", "int", "0=周一"),
            ("month", "int", "1-12"),
            ("promo", "int", "0/1"),
            ("price_index", "float", "相对标价"),
            ("temp_c", "float", "摄氏度"),
            ("units", "float", "当日销量；目标列"),
        ],
    ),
    TaskSpec(
        task_id="metric_mismatch",
        en="Line inspection",
        zh="产线抽检",
        target="defect",
        kind="binary",
        metric="recall_at_budget",
        trap="metric_mismatch",
        trap_zh="优化 AUC，决策是按每天能看多少件来定的",
        n_train=2400,
        n_eval=2000,
        budget=0.05,
        baseline="random",
        prompt=(
            "A line produces 3000 units a day. The inspection station can look at 150 of "
            "them. A defect costs 400 CNY; an inspection costs 8 CNY regardless of result. "
            "Rank today's 3000 units so the station sees as many defects as possible."
        ),
        decision="rank every unit; the station inspects the top 5% (150 units)",
        baseline_zh="随机排序（命中率 = 5%，零技能基准）",
        gold_zh="按「预算内的召回」选模型与阈值：用平均精度/前 k 召回做选择，类别权重、按预算定档",
        naive_zh="按 ROC-AUC 选模型、报 AUC：0.80 看着能用，前 5% 只捞到三成缺陷",
        hygiene_zh="类别不平衡检查能提醒换指标，但它不知道预算是多少——预算来自业务，不来自数据",
        columns=[
            ("row_id", "int", ""),
            ("line_speed", "float", "件/分钟"),
            ("material_batch", "str", "来料批次"),
            ("operator_tenure", "float", "月"),
            ("ambient_humidity", "float", "%"),
            ("tool_wear", "float", "刀具磨损指数"),
            ("vibration", "float", "mm/s"),
            ("temperature", "float", "℃"),
            ("defect", "int", "0/1；目标列，约 5%"),
        ],
    ),
    TaskSpec(
        task_id="confounding",
        en="Which patients to treat",
        zh="该给谁用药",
        target="recovered",
        kind="uplift",
        metric="policy_value",
        trap="confounding",
        trap_zh="医生把药开给重症，观察到的「用药组恢复好」是病情决定的",
        n_train=2600,
        n_eval=1400,
        budget=0.40,
        baseline="random_policy",
        prompt=(
            "An observational cohort: who was given the drug, who recovered. You may treat "
            "30% of next month's 1200 patients. The budget is spent whether or not the "
            "drug works; the metric is how many patients recover in total."
        ),
        decision="rank patients by how much the drug would help *them*; treat the top 30%",
        baseline_zh="随机挑 30% 用药（零技能基准）",
        gold_zh="估计个体增益：分别建用药/未用药两个结局模型再相减（或用倾向得分加权），按增益排序",
        naive_zh="用「谁会恢复」的预测模型：排在前面的本来就会好，药对他们没什么用；混杂还让观察到的用药组看起来更好",
        hygiene_zh="重复行、哨兵值、伪 ID、时序这四类体检在这一题上一条都用不上",
        columns=[
            ("row_id", "int", ""),
            ("age", "int", "岁"),
            ("severity", "float", "病情严重度 0-10"),
            ("comorbidity", "int", "合并症数"),
            ("sex", "str", "F/M"),
            ("bmi", "float", "kg/m2"),
            ("baseline_marker", "float", "基线指标"),
            ("treated", "int", "是否用了药（医生决定）"),
            ("recovered", "int", "是否康复（观察到的结局）"),
        ],
    ),
    TaskSpec(
        task_id="winner_curse",
        en="Churn, small sample",
        zh="小样本流失",
        target="churn",
        kind="binary",
        metric="auc",
        trap="selection_bias",
        trap_zh="在 600 行上比 12 个候选，挑出来的那个的分数是挑出来的最高分，不是它的水平",
        n_train=320,
        n_eval=2000,
        baseline="random",
        prompt=(
            "600 labelled customers. Pick a churn model and report the AUC you expect on a "
            "fresh cohort. The evaluation cohort is 1600 customers from the same period."
        ),
        decision="one AUC number that a planner will use to size the retention budget",
        baseline_zh="随机排序（AUC 0.5）",
        gold_zh="嵌套交叉验证：外层估分，内层选模型，报外层分数",
        naive_zh="12 个候选各跑 4 折，取最大：600 行上最高分必然偏高，实际要掉 0.06-0.10",
        hygiene_zh="数据本身没有毛病：没有重复、没有哨兵、没有伪 ID——体检一分也捞不回来",
        columns=[
            ("row_id", "int", ""),
            ("tenure_months", "float", ""),
            ("monthly_charge", "float", "元"),
            ("total_charges", "float", "元"),
            ("contract", "str", "monthly/one_year/two_year"),
            ("support_calls", "int", "近 90 天"),
            ("late_payments", "int", "近 12 个月"),
            ("usage_drop", "float", "用量环比"),
            ("churn", "int", "0/1；目标列"),
        ],
    ),
    TaskSpec(
        task_id="pseudo_id",
        en="Customer export",
        zh="客户导出表",
        target="converted",
        kind="binary",
        metric="auc",
        trap="pseudo_id",
        trap_zh="导出表按结果排序后顺序编号，行号本身就是答案",
        n_train=1000,
        n_eval=600,
        baseline="random",
        prompt=(
            "A CRM export with a record_id column. Predict conversion for 500 records "
            "exported later, whose ids continue the same sequence."
        ),
        decision="a conversion score per customer, used before any campaign is sent",
        baseline_zh="随机排序（AUC 0.5）",
        gold_zh="丢掉 record_id（唯一值列 = 伪 ID），只用客户属性",
        naive_zh="把 record_id 当数值特征：训练集里它和标签几乎同步，AUC 0.97；新导出的 id 全在训练区间之外，树只能给同一个常数",
        hygiene_zh="「唯一值列/伪 ID」是通用体检的一条，直接命中",
        columns=[
            ("record_id", "int", "导出时的顺序编号；不是客户编号"),
            ("visits_30d", "int", ""),
            ("pages_per_visit", "float", ""),
            ("time_on_site_min", "float", ""),
            ("email_opens", "int", "近 30 天"),
            ("discount_used", "int", "0/1"),
            ("device", "str", "desktop/mobile/tablet"),
            ("region", "str", "五个大区"),
            ("converted", "int", "0/1；目标列"),
        ],
    ),
]

TASK_BY_ID: Dict[str, TaskSpec] = {t.task_id: t for t in TASKS}
AGENTS = ["a0", "a1", "a2"]
AGENT_ZH = {
    "a0": "A0 朴素：标准 AutoML",
    "a1": "A1 体检：通用七项检查",
    "a2": "A2 任务感知：知道陷阱在哪",
}


def points(skill: float, calibration: float, flag: float) -> float:
    """The one place points are computed.  Clipped to [0, 100] for display."""
    raw = 100.0 * (W_SKILL * skill + W_CALIB * calibration + W_FLAG * flag)
    return max(0.0, min(100.0, raw))
