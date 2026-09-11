"""数据契约：四问共用的输入、状态、轨迹、账单结构。

设计原则（Pr/四问代码实现路线与使用指南.md 第 4—5 节）：
  * 数学符号短，工程变量带语义与单位后缀（_kwh / _kw / _yuan_per_kwh）；
  * 一律 float64，索引用整数段号，绝不用浮点小时作键；
  * 交易 144 段、SOC 145 个边界，shape 写在 docstring 里，不靠隐式广播；
  * 优化器不直接读写 Excel，只消费本模块规范化后的对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np

from .constants import FeeClass, N_BOUNDARY, N_INTERVAL

# ==========================================================================
# 1. 输入层
# ==========================================================================


@dataclass(frozen=True)
class DayInput:
    """单日策略可用的输入（已是"信息视图"过滤后的结果）。

    调用方必须保证：这里只包含**决策时刻已经可见**的量。
    仿真层负责逐事件释放信息，策略模块拿不到未来实测表。

    Attributes
    ----------
    day : 自然日
    demand_kwh : (144,) 区间负载电量，D_t = L_t / 6
    pv_kwh : (144,) 区间光伏电量，G_t = P_t / 6
    price_yuan_per_kwh : (144,) 用于**规划**的电价（问题1/2/3 为重复曲线，
        问题4 为预测价；不得传未来真实价）
    """

    day: date
    demand_kwh: np.ndarray
    pv_kwh: np.ndarray
    price_yuan_per_kwh: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("demand_kwh", "pv_kwh", "price_yuan_per_kwh"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (N_INTERVAL,):
                raise ValueError(f"{name} 必须是 ({N_INTERVAL},)，得到 {arr.shape}")
            object.__setattr__(self, name, arr)


@dataclass(frozen=True)
class ObservedState:
    """决策时刻的真实可观测状态。"""

    day: date
    interval: int           # 当前段号 0..143
    soc_kwh: float          # 当前真实储电量（E_boundary[interval]）
    soc_history_kwh: np.ndarray = field(default_factory=lambda: np.zeros(0))
    recent_demand_kw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    recent_pv_kw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    recent_price: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ==========================================================================
# 2. 策略层
# ==========================================================================


@dataclass(frozen=True)
class ActivePlan:
    """当前有效的普通购电策略。

    问题二/4-2：全天只在 0 点生成一次，日内不修改。
    问题三/4-3：0/6/12/18 节点可修改**尚未执行**的区间。

    Attributes
    ----------
    created_at_interval : 本版本生成时刻的段号（0/36/72/108）
    grid_kwh : (144,) 普通购电计划。索引 t < created_at_interval 的部分是
        已执行历史，不得改写；>= 的部分才是可调整的未来计划。
    """

    created_at_interval: int
    grid_kwh: np.ndarray
    version: int = 0

    def __post_init__(self) -> None:
        arr = np.asarray(self.grid_kwh, dtype=np.float64)
        if arr.shape != (N_INTERVAL,):
            raise ValueError(f"grid_kwh 必须是 ({N_INTERVAL},)，得到 {arr.shape}")
        object.__setattr__(self, "grid_kwh", arr)


@dataclass(frozen=True)
class PlanUpdate:
    """一次策略调整的记录，用于违约计算与费用审计。

    只保留复核费用所需的最小信息（时刻、减少量、当时价、违约费），
    不要求永久保存每次完整弃用计划。
    """

    at_interval: int
    price_at_interval_yuan_per_kwh: float
    reduced_kwh: np.ndarray          # (144,) delta^-_{a,t}，非负，未调整段为 0
    penalty_yuan: float

    @property
    def total_reduced_kwh(self) -> float:
        return float(np.sum(self.reduced_kwh))


# ==========================================================================
# 3. 执行层
# ==========================================================================


@dataclass(frozen=True)
class ExecutedEvent:
    """一笔**已经实际执行**的购电事件。

    只有实际执行的事件才进入最终购电量与费用。
    被放弃的未执行计划量不产生事件。
    """

    day: date
    interval: int
    quantity_kwh: float
    price_yuan_per_kwh: float     # 事件**执行当时**的价格
    fee_class: FeeClass

    @property
    def rate(self) -> float:
        from .constants import FEE_RATE_BY_CLASS

        return FEE_RATE_BY_CLASS[self.fee_class]

    @property
    def cost_yuan(self) -> float:
        return self.rate * self.price_yuan_per_kwh * self.quantity_kwh


@dataclass
class Trajectory:
    """单日实际执行轨迹。区间量 144 个，状态边界 145 个。"""

    day: date
    grid_actual_kwh: np.ndarray          # (144,) 普通实际购电 h_t
    emergency_actual_kwh: np.ndarray     # (144,) 紧急实际购电 u_t
    charge_stored_kwh: np.ndarray        # (144,) q^ch_t，实际存入电池
    discharge_delivered_kwh: np.ndarray  # (144,) q^dis_t，实际送达微网
    curtail_kwh: np.ndarray              # (144,) r_t，仅弃光
    soc_kwh: np.ndarray                  # (145,) E_s，s = 0..144
    price_actual: np.ndarray             # (144,) 实际执行价（问题4 为附件4）
    surplus_disposed_kwh: np.ndarray | None = None
    """(144,) 已披露的安全阀：因"计划购电过量且储能已满"而无法消纳的富余。

    规范第 6 节禁止用拒收、售电或损耗烧电掩盖这种富余，并要求如实报告。
    故这里把它显式记账：它不是"丢弃外购电"这一被禁止的操作，
    而是"该计划在当日物理条件下不可行"的**缺口量**，必须进入论文的风险披露。
    为 None 表示当日不存在该情形（此时逐段守恒严格成立）。
    """
    events: list[ExecutedEvent] = field(default_factory=list)
    updates: list[PlanUpdate] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = {
            "grid_actual_kwh": N_INTERVAL,
            "emergency_actual_kwh": N_INTERVAL,
            "charge_stored_kwh": N_INTERVAL,
            "discharge_delivered_kwh": N_INTERVAL,
            "curtail_kwh": N_INTERVAL,
            "price_actual": N_INTERVAL,
            "soc_kwh": N_BOUNDARY,
        }
        for name, n in expected.items():
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(f"{name} 必须是 ({n},)，得到 {arr.shape}")
            setattr(self, name, arr)
        if self.surplus_disposed_kwh is not None:
            s = np.asarray(self.surplus_disposed_kwh, dtype=np.float64)
            if s.shape != (N_INTERVAL,):
                raise ValueError(
                    f"surplus_disposed_kwh 必须是 ({N_INTERVAL},)，得到 {s.shape}"
                )
            self.surplus_disposed_kwh = s

    @property
    def soc_start_kwh(self) -> float:
        return float(self.soc_kwh[0])

    @property
    def soc_end_kwh(self) -> float:
        return float(self.soc_kwh[-1])


# ==========================================================================
# 4. 结算层
# ==========================================================================


@dataclass(frozen=True)
class Bill:
    """最终账单。数量只需实际轨迹，费用还需保留的违约费。

    规范要求：相同最终购电量不保证费用相同，因此量、费分别复算。
    """

    day: date
    normal_kwh: float
    adjust_up_kwh: float
    emergency_kwh: float
    total_purchased_kwh: float
    purchase_cost_yuan: float
    penalty_yuan: float
    total_cost_yuan: float
    cost_by_class_yuan: dict[FeeClass, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float]:
        return {
            "normal_kwh": self.normal_kwh,
            "adjust_up_kwh": self.adjust_up_kwh,
            "emergency_kwh": self.emergency_kwh,
            "total_kwh": self.total_purchased_kwh,
            "purchase_cost_yuan": self.purchase_cost_yuan,
            "penalty_yuan": self.penalty_yuan,
            "total_cost_yuan": self.total_cost_yuan,
        }


# ==========================================================================
# 5. 求解结果包装
# ==========================================================================


@dataclass
class SolveReport:
    """优化器返回的完整状态——不只返回一个购电数组。"""

    problem: str
    status: str                  # optimal / infeasible / unbounded / feasible / time_limit ...
    feasible: bool
    objective_yuan: float | None
    termination: str
    solver: str
    wall_seconds: float
    max_residual: float = float("nan")
    gap: float | None = None
    message: str = ""

    def require_ok(self) -> None:
        """只有经过复算的可行解才能继续执行。"""
        if not self.feasible or self.objective_yuan is None:
            raise RuntimeError(
                f"[{self.problem}] 未获得可用解：status={self.status} "
                f"feasible={self.feasible} termination={self.termination} "
                f"message={self.message}"
            )


DayMode = Literal["p1", "p2", "p3", "p4-2", "p4-3"]

__all__ = [
    "DayInput",
    "ObservedState",
    "ActivePlan",
    "PlanUpdate",
    "ExecutedEvent",
    "Trajectory",
    "Bill",
    "SolveReport",
    "DayMode",
]
