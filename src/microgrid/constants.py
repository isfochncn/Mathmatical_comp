"""全局固定参数（四问共用）。

本模块是"题面事实"与"已确认口径"的唯一落脚点。
任何数值都不允许在其它模块里重复硬编码。

来源标注
--------
[题面]  2026 高教社杯 C 题正文 / 附录1 / 附录2
[口径]  Pr/题意解析与建模衔接备忘录.md 第 1 节"已冻结的高影响口径"
[工程]  Pr/四问代码实现路线与使用指南.md 第 4 节代码约定
"""

from __future__ import annotations

from enum import IntEnum

# --------------------------------------------------------------------------
# 时间网格  [题面][口径]
# --------------------------------------------------------------------------
#: 调度颗粒度：10 分钟
MINUTES_PER_INTERVAL = 10

#: 每段时长（小时）
DELTA_T_HOURS = 1.0 / 6.0

#: 一天中的交易段数（区间 t = 0..143）
N_INTERVAL = 144

#: 一天中的状态边界数（边界 s = 0..144），比区间数恰好多 1
N_BOUNDARY = 145

#: Source labels are interval starts; the tail denotes next-day 00:00.
INTERVAL_LABEL_IS_END_TIME = False

# --------------------------------------------------------------------------
# 储能设备物理参数  [题面 附录1]
# --------------------------------------------------------------------------
#: 额定容量（kWh）
E_RATED = 12_000.0

#: 允许运行的电量上/下限（kWh）——注意：额定容量不等于运行上限
E_MIN = 1_200.0
E_MAX = 10_800.0

#: 最大充放电功率（kW，母线侧）
P_MAX_KW = 5_000.0

#: 充/放电效率
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9

#: 往返效率（供论文说明用，不是独立参数）
ETA_ROUND_TRIP = ETA_CHARGE * ETA_DISCHARGE  # 0.81

# --------------------------------------------------------------------------
# 已确认口径：有效充放电量上限  [口径]
# --------------------------------------------------------------------------
# 备忘录原文："两向5000×0.9=4500kW；每10分钟实际有效电量上限750kWh"、
# "不得将有效750kWh再乘0.9作为实际充电上限"。
#
# q_ch is energy stored in the battery; q_dis is energy delivered to the bus.
# Each effective direction is limited to 4500 kW = 750 kWh per interval.
P_MAX_EFFECTIVE_KW = P_MAX_KW * ETA_CHARGE
Q_MAX = P_MAX_EFFECTIVE_KW * DELTA_T_HOURS
Q_DIS_MAX = Q_MAX

#: 母线侧充电输入换算系数：存入 q_ch 需要母线付出 q_ch / ETA_CHARGE
CHARGE_BUS_FACTOR = 1.0 / ETA_CHARGE
#: 电池侧放电消耗换算系数：送达 q_dis 需要电池消耗 q_dis / ETA_DISCHARGE
DISCHARGE_BATTERY_FACTOR = 1.0 / ETA_DISCHARGE

# --------------------------------------------------------------------------
# 全天恒等式 / 损耗系数
# --------------------------------------------------------------------------
# 严格推导（η_c = η_d = η = 0.9）：
#
#   母线守恒   v_t = D_t - G_t + r_t + q_ch_t/η - q_dis_t
#   状态转移   δE_t = E_{t+1} - E_t = q_ch_t - q_dis_t/η
#              =>  q_dis_t = η (q_ch_t - δE_t)
#   代回母线守恒：
#              v_t = (D_t - G_t + r_t) + q_ch_t (1/η - η) + η δE_t
#
# 全天求和（Σ δE = E_144 - E_0）：
#   Σ(h+u) = Σ(D - G + r) + K·Σq_ch + η·(E_144 - E_0),     K = 1/η - η
#
# 在 η = 0.9 下：K = 1/0.9 - 0.9 = 19/90，η = 0.9，
# 正是备忘录第 3 节给出的形式
#   Σ(h+u) = Σ(D-G+r) + (19/90)·Σq_ch + 0.9·(E_144 - E_0)
#
LOSS_COEFF = 1.0 / ETA_CHARGE - ETA_DISCHARGE  # 19/90

#: 状态项系数 β = η_d（不是 η_c·η_d）。δE 以"实际存入"为口径，故只带放电效率。
STATE_TERM_COEFF = ETA_DISCHARGE  # 0.9

#: 损耗循环（同一段内同时充放，δE = 0）的系数：
#: 此时 q_dis = η·q_ch，母线净付出 = q_ch/η - q_dis = K·q_ch。
LOSS_CYCLE_COEFF = LOSS_COEFF  # 19/90

#: 充电环节的单次损耗比例（母线付出中真正损耗掉的部分）= 1/η - 1 = 1/9
CHARGE_LOSS_FRACTION = 1.0 / ETA_CHARGE - 1.0  # 1/9

# --------------------------------------------------------------------------
# 初始状态与日边界  [题面][口径]
# --------------------------------------------------------------------------
#: 初始储电量（kWh）：滚动模型设于2025-01-01 00:10；问题一仍为典型日0:00。
E_INIT_2025_01_01 = 6_000.0
# Rolling runs initialize at the first observed interval, Jan 1 00:10.
ROLLING_START_ABS_MINUTE = 10

#: 只有问题一要求日首日末相等  [题面 问题1]
P1_REQUIRE_DAILY_CYCLE = True

# --------------------------------------------------------------------------
# 费率与市场制度  [题面][口径]
# --------------------------------------------------------------------------
FEE_NORMAL = 1.0          # 普通计划购电
FEE_ADJUST_UP = 1.5       # 调整购电量高于计划的部分
FEE_EMERGENCY = 5.0       # 紧急购电
FEE_PENALTY_DOWN = 0.5    # 计划量高于调整量（减少）部分的违约


class FeeClass(IntEnum):
    """购电事件的费率类别。事件类别由"当时真实执行动作"确定，
    不用最终相对 0 点的净差额重新分类。"""

    NORMAL = 1        # 普通计划购电，1.0 倍事件当时价
    ADJUST_UP = 2     # 实际调整增购，1.5 倍
    EMERGENCY = 3     # 紧急购电，5.0 倍


FEE_RATE_BY_CLASS: dict[FeeClass, float] = {
    FeeClass.NORMAL: FEE_NORMAL,
    FeeClass.ADJUST_UP: FEE_ADJUST_UP,
    FeeClass.EMERGENCY: FEE_EMERGENCY,
}

# --------------------------------------------------------------------------
# 需要输出结果文件的日期范围  [题面 附件5]
# --------------------------------------------------------------------------
#: 问题2/3/4 输出 2025-02-01 至 2025-12-31，共 334 天；1 月为预热期
OUTPUT_START = (2025, 2, 1)
OUTPUT_END = (2025, 12, 31)
N_OUTPUT_DAYS = 334

#: 论文表3 指定的四个日期
SPECIAL_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")

# --------------------------------------------------------------------------
# 问题三 / 4-3 的策略调整节点  [题面][口径]
# --------------------------------------------------------------------------
#: 预报发布时刻（小时）：0、6、12、18。对应区间起点 tau_k = 36k
FORECAST_PUBLISH_HOURS = (0, 6, 12, 18)
ADJUST_NODE_INTERVALS = tuple(h * 6 for h in FORECAST_PUBLISH_HOURS)  # (0, 36, 72, 108)

#: 4-3 允许调整（不含 0 点日计划）的节点
ADJUST_NODE_INTERVALS_EXCL_START = ADJUST_NODE_INTERVALS[1:]  # (36, 72, 108)

# --------------------------------------------------------------------------
# 数据文件相对路径（相对项目根目录）  [工程]
# --------------------------------------------------------------------------
DATA_DIR = "data"
ATTACHMENT_1 = "data/附件1.xlsx"
ATTACHMENT_2 = "data/附件2.xlsx"
ATTACHMENT_3 = "data/附件3.xlsx"
ATTACHMENT_4 = "data/附件4.xlsx"
TEMPLATE_DIR = "data/附件5"
TEMPLATE_FILES = {
    "result1": "data/附件5/result1.xlsx",
    "result2": "data/附件5/result2.xlsx",
    "result3": "data/附件5/result3.xlsx",
    "result4-2": "data/附件5/result4-2.xlsx",
    "result4-3": "data/附件5/result4-3.xlsx",
}

#: Excel 锁文件前缀，必须排除
LOCK_PREFIX = "~$"

# --------------------------------------------------------------------------
# 数值容差  [工程] 规范第 13 节：不同物理量用不同容差
# --------------------------------------------------------------------------
TOL_ENERGY_KWH = 1e-6      # 电量（kWh）
TOL_POWER_KW = 1e-6        # 功率（kW）
TOL_COST_YUAN = 1e-4       # 费用（元）
TOL_RELATIVE = 1e-9        # 相对残差
