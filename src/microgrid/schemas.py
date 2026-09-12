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
    """(144,) 已付费但无法消纳的弃购电；与弃光独立记账。
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


# ==========================================================================
# 6. 绝对时间线模型（2026-09-11 定稿口径）
# ==========================================================================
#
# 旧结构（按"自然日 + 144 段"组织）无法表达三件新要求：
#   * result 行覆盖 [当日 00:10, 次日 00:10)，与自然日错开一段；
#   * 优化窗口固定在 [a, T)（当前十分钟起点到次日 24:00），长度随 a 变化；
#   * 问题三要保存原计划剩余量 O 与调整增购剩余量 A。
# 因此新增下面这组以**绝对分钟**为主键的结构；旧结构保留供问题一使用。


@dataclass
class SessionPlan:
    """一个优化窗口 [a, T) 上的可行计划（绝对时间主键）。

    ``values`` 的每一项都是 ``(n, )``，n = 窗口内的十分钟区间数，
    与 ``abs_minutes`` 一一对应。``soc_kwh`` 有 n+1 个边界。
    """

    a_abs: int
    values: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.values["grid_kwh"].size)

    def abs_minutes(self) -> np.ndarray:
        return self.values["abs_minute"]

    def slice_from(self, abs_from: int) -> "SessionPlan":
        """取 [abs_from, T) 的部分（用于滚动窗口推进）。"""
        minutes = self.values["abs_minute"]
        lo = int(np.searchsorted(minutes, abs_from))
        out = {k: (v[lo:] if v.size == minutes.size else v[lo : lo + 1]) for k, v in self.values.items()}
        return SessionPlan(a_abs=abs_from, values=out)


@dataclass
class CommittedBalances:
    """问题三的 O/A 状态：每个未执行区间还剩多少原计划量、多少调整增购量。

    ``y = O + A`` 是当前有效量。递推（规范第 6 节）：
        O' = min(O, y')，A' = [y' − O]_+，δ⁻ = [O + A − y']_+
    记账约定：调减先扣 A 再扣 O，恢复量归 A。
    """

    abs_minutes: np.ndarray          # (n,) 未执行区间的绝对起点
    o_kwh: np.ndarray                # (n,) 原计划剩余量
    a_kwh: np.ndarray                # (n,) 调整增购剩余量

    def __post_init__(self) -> None:
        self.abs_minutes = np.asarray(self.abs_minutes, dtype=np.int64)
        self.o_kwh = np.asarray(self.o_kwh, dtype=np.float64)
        self.a_kwh = np.asarray(self.a_kwh, dtype=np.float64)
        if not np.isfinite(self.o_kwh).all() or not np.isfinite(self.a_kwh).all() or np.any(self.o_kwh < -1e-6) or np.any(self.a_kwh < -1e-6):
            raise ValueError("Invalid commitment quantities")
        self.o_kwh = np.maximum(self.o_kwh, 0)
        self.a_kwh = np.maximum(self.a_kwh, 0)
        if np.any(np.diff(self.abs_minutes) <= 0):
            raise ValueError("Commitment timestamps must be unique and ordered")
        if not (self.abs_minutes.shape == self.o_kwh.shape == self.a_kwh.shape):
            raise ValueError("CommittedBalances 三个数组形状必须一致")

    @property
    def effective_kwh(self) -> np.ndarray:
        return self.o_kwh + self.a_kwh

    @classmethod
    def from_initial_plan(cls, abs_minutes: np.ndarray, plan_kwh: np.ndarray) -> "CommittedBalances":
        return cls(
            abs_minutes=np.asarray(abs_minutes, dtype=np.int64),
            o_kwh=np.asarray(plan_kwh, dtype=np.float64).copy(),
            a_kwh=np.zeros_like(np.asarray(plan_kwh, dtype=np.float64)),
        )

    def revise(self, new_y_kwh: np.ndarray) -> tuple["CommittedBalances", np.ndarray]:
        """按规范递推一次调整，返回 (新状态, δ⁻ 减少量)。"""
        y_new = np.asarray(new_y_kwh, dtype=np.float64)
        if y_new.shape != self.o_kwh.shape:
            raise ValueError("新计划形状与 O/A 状态不一致")
        if not np.isfinite(y_new).all() or np.any(y_new < -1e-6):
            raise ValueError("Invalid revised quantities")
        y_new = np.maximum(y_new, 0)
        o_new = np.minimum(self.o_kwh, y_new)
        a_new = np.maximum(y_new - self.o_kwh, 0.0)
        delta_minus = np.maximum(self.o_kwh + self.a_kwh - y_new, 0.0)
        return (
            CommittedBalances(self.abs_minutes.copy(), o_new, a_new),
            delta_minus,
        )

    def take(self, abs_minute: int) -> tuple[float, float]:
        """Execute and remove exactly one existing commitment."""
        idx = int(np.searchsorted(self.abs_minutes, abs_minute))
        if idx >= self.abs_minutes.size or self.abs_minutes[idx] != abs_minute:
            raise ValueError(f"Missing commitment for interval {abs_minute}")
        o, a = float(self.o_kwh[idx]), float(self.a_kwh[idx])
        self.abs_minutes = np.delete(self.abs_minutes, idx)
        self.o_kwh = np.delete(self.o_kwh, idx)
        self.a_kwh = np.delete(self.a_kwh, idx)
        return o, a


@dataclass
class PenaltyEvent:
    """一次调减产生的违约事件（按真实发生时间记录）。"""

    at_abs: int
    price_at_yuan_per_kwh: float
    reduced_kwh: np.ndarray          # 与当时未执行区间对齐的 δ⁻
    reduced_abs: np.ndarray

    @property
    def total_reduced_kwh(self) -> float:
        return float(np.sum(self.reduced_kwh))

    @property
    def penalty_yuan(self) -> float:
        return 0.5 * self.price_at_yuan_per_kwh * self.total_reduced_kwh


@dataclass
class DispatchEvent:
    """一个十分钟区间**已经实际执行**的购电动作，带费率标签。

    费率标签由当时真实执行动作确定，不用最终净差额重新分类。
    """

    at_abs: int
    o_exec_kwh: float                # 执行的原计划部分，1.0 倍
    a_exec_kwh: float                # 执行的调整增购部分，1.5 倍
    emergency_kwh: float             # 紧急购电，5.0 倍
    price_actual_yuan_per_kwh: float

    @property
    def total_kwh(self) -> float:
        return self.o_exec_kwh + self.a_exec_kwh + self.emergency_kwh

    @property
    def cost_yuan(self) -> float:
        return self.price_actual_yuan_per_kwh * (
            self.o_exec_kwh + 1.5 * self.a_exec_kwh + 5.0 * self.emergency_kwh
        )


@dataclass
class AbsoluteStep:
    """一个已实际执行区间的完整物理与交易记录。"""

    abs_minute: int
    grid_kwh: float                  # 非紧急普通实际购电 = O + A
    emergency_kwh: float
    charge_kwh: float                # q_ch（电池实际存入量）
    discharge_kwh: float             # q_dis（实际送达量）
    curtail_kwh: float
    surplus_kwh: float               # 已付费弃购电，既不退费也不改变O/A标签
    soc_end_kwh: float
    price_actual_yuan_per_kwh: float


@dataclass
class AbsoluteRun:
    """一段连续绝对时间上的实际执行结果。"""

    abs_minutes: np.ndarray
    steps: list[AbsoluteStep]
    soc_boundary_kwh: np.ndarray     # (n+1,)
    events: list[DispatchEvent]
    penalties: list[PenaltyEvent]
    initial_plans: dict[int, float] = field(default_factory=dict)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "abs_minute": self.abs_minutes,
            "grid_kwh": np.array([s.grid_kwh for s in self.steps]),
            "emergency_kwh": np.array([s.emergency_kwh for s in self.steps]),
            "charge_kwh": np.array([s.charge_kwh for s in self.steps]),
            "discharge_kwh": np.array([s.discharge_kwh for s in self.steps]),
            "curtail_kwh": np.array([s.curtail_kwh for s in self.steps]),
            "surplus_kwh": np.array([s.surplus_kwh for s in self.steps]),
            "price_actual": np.array([s.price_actual_yuan_per_kwh for s in self.steps]),
            "soc_kwh": np.asarray(self.soc_boundary_kwh, dtype=np.float64),
        }


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
    "SessionPlan",
    "CommittedBalances",
    "PenaltyEvent",
    "DispatchEvent",
    "AbsoluteStep",
    "AbsoluteRun",
]
